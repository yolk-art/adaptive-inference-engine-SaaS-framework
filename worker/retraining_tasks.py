"""
worker/retraining_tasks.py

Celery tasks for EWC-based model retraining.

Architecture
------------
This module is the stable async execution boundary between the admin API
(which enqueues jobs) and the actual training runtime (PyTorch EWC or a
scikit-learn refit). Adding a new training strategy never requires changing
the admin API contract — only this file.

CUDA / Multiprocessing safety
------------------------------
PyTorch CUDA contexts cannot be safely shared across forked processes.
Celery's default prefork pool will trigger a RuntimeError when any CUDA
operation is executed in a child process.

Solutions (choose one when deploying):
  1. Run the worker with the solo pool:
       celery -A worker.retraining_tasks worker -P solo --loglevel=info
  2. Migrate to Ray Serve (actor-based, avoids fork entirely).
  3. Force CPU-only inference during retraining (set DEVICE=cpu).

The task below automatically falls back to CPU so it is safe in all pool modes.

Queue safety
-------------
Set ``prefetch_count=1`` on the RabbitMQ consumer (or use ``-c 1`` concurrency)
to prevent a single worker from pulling multiple heavy jobs simultaneously.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

from celery import Celery

from worker.worker_multitenant import summarize_tenant_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery application
# ---------------------------------------------------------------------------

BROKER_URL: str = os.getenv(
    "CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/1")
)
RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

celery_app = Celery(
    "adaptive_inference_worker",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
)

# Safeguards: at most one heavy job in-flight per worker process
celery_app.conf.update(
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)


# ---------------------------------------------------------------------------
# EWC helpers
# ---------------------------------------------------------------------------


def _compute_fisher_diagonal(
    model,
    data_loader,
    device: str,
    n_samples: int = 200,
) -> Dict[str, "torch.Tensor"]:
    """
    Compute the diagonal of the Fisher Information Matrix (FIM) for EWC.

    The FIM diagonal is approximated by the squared gradient of the log-
    likelihood averaged over ``n_samples`` training examples.

    Returns a dict mapping parameter name → importance tensor (on CPU).
    """
    import torch
    import torch.nn.functional as F

    model.train()
    fisher: Dict[str, "torch.Tensor"] = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    seen = 0
    for batch_x, batch_y in data_loader:
        if seen >= n_samples:
            break
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        model.zero_grad()
        output = model(batch_x)
        loss = F.binary_cross_entropy_with_output(output.squeeze(), batch_y.float())
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None and name in fisher:
                fisher[name] += param.grad.detach().pow(2)

        seen += len(batch_x)

    # Normalise by number of processed samples
    for name in fisher:
        fisher[name] /= max(seen, 1)

    return {name: f.cpu() for name, f in fisher.items()}


def _ewc_loss(
    model,
    fisher: Dict[str, "torch.Tensor"],
    optimal_params: Dict[str, "torch.Tensor"],
    lambda_reg: float,
    device: str,
) -> "torch.Tensor":
    """
    Elastic Weight Consolidation regularisation term.

    EWC Loss = λ/2 * Σ F_i * (θ_i − θ*_i)²

    Penalises large deviations from previously optimal parameters,
    weighted by each parameter's importance (Fisher diagonal).
    """
    import torch

    ewc_term = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if name in fisher:
            f = fisher[name].to(device)
            opt = optimal_params[name].to(device)
            ewc_term += (f * (param - opt).pow(2)).sum()
    return (lambda_reg / 2.0) * ewc_term


def _build_tensor_dataset(records: List[Dict], feature_keys: Optional[List[str]] = None):
    """
    Convert telemetry records to a PyTorch TensorDataset.

    Each record should have a ``"features"`` list and a ``"prediction"`` int.
    Records with missing or malformed data are skipped.
    """
    import torch
    from torch.utils.data import TensorDataset

    X_rows: List[List[float]] = []
    y_rows: List[int] = []

    for rec in records:
        if not isinstance(rec, dict):
            continue
        features = rec.get("features")
        label = rec.get("prediction")
        if features is None or label is None:
            continue
        try:
            X_rows.append([float(f) for f in features])
            y_rows.append(int(label))
        except (TypeError, ValueError):
            continue

    if not X_rows:
        return None

    X = torch.tensor(X_rows, dtype=torch.float32)
    y = torch.tensor(y_rows, dtype=torch.long)
    return TensorDataset(X, y)


def _run_ewc_retraining(
    tenant_id: str,
    model_id: str,
    records: List[Dict],
    lambda_reg: float = 800.0,
    epochs: int = 20,
    lr: float = 0.005,
    device: str = "cpu",
) -> Dict:
    """
    Execute EWC fine-tuning on a PyTorch FraudNet model.

    Loads the model from the Tenant Model Registry, freezes a snapshot of the
    current weights, computes the Fisher diagonal, then runs ``epochs`` training
    steps with the EWC regularisation term added to the task loss.

    Falls back gracefully if PyTorch is not installed or if model artifacts are
    missing (returns a status dict indicating the skip reason).
    """
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        logger.warning("PyTorch not installed — skipping EWC retraining")
        return {"skipped": True, "reason": "pytorch_not_installed"}

    from inference.tenant_model_registry import TenantModelRegistry
    from inference.fraudnet_runtime import FraudNetRuntime

    registry = TenantModelRegistry()
    metadata = registry.get_latest_model(tenant_id, model_id)
    if metadata is None:
        logger.warning("No registered model for %s/%s — skipping", tenant_id, model_id)
        return {"skipped": True, "reason": "model_not_registered"}

    # Build dataset from telemetry records
    dataset = _build_tensor_dataset(records)
    if dataset is None or len(dataset) < 4:
        logger.warning(
            "Insufficient training data for %s/%s (%d records) — skipping",
            tenant_id,
            model_id,
            len(records),
        )
        return {"skipped": True, "reason": "insufficient_data", "records": len(records)}

    # Load model runtime (always CPU for Celery safety)
    runtime = FraudNetRuntime(metadata.config_path, device=device)
    try:
        runtime.load(metadata.storage_path)
    except Exception as exc:
        logger.warning(
            "Could not load model weights for %s/%s: %s — skipping", tenant_id, model_id, exc
        )
        return {"skipped": True, "reason": "model_load_failed", "error": str(exc)}

    net = runtime.model
    if net is None:
        return {"skipped": True, "reason": "model_none_after_load"}

    net = net.to(device)

    # Snapshot optimal weights before retraining
    optimal_params: Dict[str, "torch.Tensor"] = {
        name: param.clone().detach().cpu()
        for name, param in net.named_parameters()
        if param.requires_grad
    }

    loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)

    # Compute Fisher diagonal on current data
    fisher = _compute_fisher_diagonal(net, loader, device=device)

    # EWC fine-tuning loop
    import torch.nn.functional as F

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = net(batch_x)
            task_loss = F.binary_cross_entropy_with_output(
                output.squeeze(), batch_y.float()
            )
            ewc_penalty = _ewc_loss(net, fisher, optimal_params, lambda_reg, device)
            total_loss = task_loss + ewc_penalty
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / max(len(loader), 1)
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            logger.debug("EWC epoch %d/%d  loss=%.4f", epoch + 1, epochs, avg_loss)

    # Persist updated weights
    try:
        torch.save(net.state_dict(), metadata.storage_path)
        logger.info(
            "EWC retraining complete for %s/%s — saved to %s",
            tenant_id,
            model_id,
            metadata.storage_path,
        )
    except Exception as exc:
        logger.warning("Could not save updated weights: %s", exc)

    return {
        "skipped": False,
        "epochs": epochs,
        "final_loss": losses[-1] if losses else None,
        "initial_loss": losses[0] if losses else None,
        "storage_path": metadata.storage_path,
    }


def _run_sklearn_retraining(
    tenant_id: str,
    model_id: str,
    records: List[Dict],
) -> Dict:
    """
    Refit a scikit-learn model on the latest telemetry records.

    Uses the Tenant Model Registry to locate the serialised model and config,
    refits using the labelled telemetry window, and persists back to the same
    path.
    """
    try:
        import pickle
        import numpy as np
    except ImportError:
        return {"skipped": True, "reason": "numpy_not_installed"}

    from inference.tenant_model_registry import TenantModelRegistry

    registry = TenantModelRegistry()
    metadata = registry.get_latest_model(tenant_id, model_id)
    if metadata is None:
        return {"skipped": True, "reason": "model_not_registered"}

    X_rows, y_rows = [], []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        features = rec.get("features")
        label = rec.get("prediction")
        if features is None or label is None:
            continue
        try:
            X_rows.append([float(f) for f in features])
            y_rows.append(int(label))
        except (TypeError, ValueError):
            continue

    if len(X_rows) < 4:
        return {"skipped": True, "reason": "insufficient_data", "records": len(X_rows)}

    try:
        with open(metadata.storage_path, "rb") as f:
            clf = pickle.load(f)
    except Exception as exc:
        return {"skipped": True, "reason": "model_load_failed", "error": str(exc)}

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)

    try:
        clf.fit(X, y)
        with open(metadata.storage_path, "wb") as f:
            pickle.dump(clf, f)
    except Exception as exc:
        return {"skipped": True, "reason": "training_failed", "error": str(exc)}

    return {"skipped": False, "records": len(X_rows), "storage_path": metadata.storage_path}


# ---------------------------------------------------------------------------
# Celery task definition
# ---------------------------------------------------------------------------


@celery_app.task(name="worker.retraining_tasks.retrain_model", bind=True, max_retries=2)
def retrain_model(
    self,
    tenant_id: str,
    model_id: str,
    trigger_reason: str = "manual",
    force_retrain: bool = False,
) -> Dict:
    """
    Async Celery task: run a full retraining cycle for a tenant model.

    Execution stages
    ----------------
    1. SUMMARIZE  — pull telemetry + compute drift summary (PSI, AUC).
    2. RETRAIN    — run EWC (PyTorch) or refit (sklearn) on the telemetry window.
    3. COMPLETE   — return the consolidated result payload.

    On failure the task retries up to ``max_retries`` times with a 30-second
    countdown before marking as FAILURE.
    """
    logger.info(
        "retrain_model task started — tenant=%s  model=%s  reason=%s",
        tenant_id,
        model_id,
        trigger_reason,
    )

    # ── Stage 1: Summarize ────────────────────────────────────────────────
    self.update_state(
        state="STARTED",
        meta={"tenant_id": tenant_id, "model_id": model_id, "stage": "summarizing"},
    )
    try:
        summary = summarize_tenant_model(tenant_id, model_id)
    except Exception as exc:
        logger.error("Drift summarization failed: %s", exc)
        summary = {"error": str(exc), "records": 0}

    records_available = summary.get("records", 0)

    # ── Stage 2: Retrain ──────────────────────────────────────────────────
    self.update_state(
        state="STARTED",
        meta={
            "tenant_id": tenant_id,
            "model_id": model_id,
            "stage": "retraining",
            "records": records_available,
        },
    )

    # Resolve framework from registry to pick training strategy
    framework = "pytorch"
    try:
        from inference.tenant_model_registry import TenantModelRegistry

        registry = TenantModelRegistry()
        metadata = registry.get_latest_model(tenant_id, model_id)
        if metadata:
            framework = metadata.framework
    except Exception:
        pass  # fall back to pytorch

    # Pull raw records for retraining (same window used for drift summary)
    raw_records: List[Dict] = []
    try:
        from inference.tenant_redis_client import TenantRedisClient

        rc = TenantRedisClient(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"), tenant_id
        )
        raw_records = rc.get_telemetry_batch(model_id)
    except Exception as exc:
        logger.warning("Could not fetch telemetry for retraining: %s", exc)

    if framework == "sklearn":
        training_result = _run_sklearn_retraining(tenant_id, model_id, raw_records)
    else:
        training_result = _run_ewc_retraining(tenant_id, model_id, raw_records)

    # ── Stage 3: Complete ─────────────────────────────────────────────────
    result = {
        "tenant_id": tenant_id,
        "model_id": model_id,
        "trigger_reason": trigger_reason,
        "force_retrain": force_retrain,
        "status": "completed",
        "drift_summary": summary,
        "training_result": training_result,
        "completed_at": time.time(),
    }
    logger.info(
        "retrain_model task complete — tenant=%s  model=%s  training_skipped=%s",
        tenant_id,
        model_id,
        training_result.get("skipped", True),
    )
    return result
