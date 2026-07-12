"""
worker/metrics.py

Production-grade metric helpers for the multi-tenant drift detection worker.

Provides:
  - population_stability_index(): proper histogram-bin PSI (not mean-difference)
  - adversarial_auc(): binary classifier AUC to detect distribution shift
  - Prometheus instrumentation (optional — gracefully absent when prometheus_client
    is not installed so unit tests can run without the dependency)
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus setup (optional dependency)
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram

    _DRIFT_PSI_GAUGE = Gauge(
        "mlops_drift_psi",
        "Current PSI drift score for a tenant model",
        ["tenant_id", "model_id"],
    )
    _DRIFT_AUC_GAUGE = Gauge(
        "mlops_drift_adversarial_auc",
        "Adversarial validation AUC for a tenant model",
        ["tenant_id", "model_id"],
    )
    _DRIFT_EVENT_COUNTER = Counter(
        "mlops_drift_events_total",
        "Number of drift threshold crossings detected",
        ["tenant_id", "model_id", "metric"],
    )
    _RETRAIN_TRIGGER_COUNTER = Counter(
        "mlops_retraining_triggers_total",
        "Number of retraining jobs triggered by the worker",
        ["tenant_id", "model_id", "reason"],
    )
    _TELEMETRY_RECORDS_GAUGE = Gauge(
        "mlops_telemetry_queue_depth",
        "Number of telemetry records in the tenant queue",
        ["tenant_id", "model_id"],
    )
    _INFERENCE_LATENCY_HISTOGRAM = Histogram(
        "mlops_worker_check_latency_seconds",
        "Time taken per drift check cycle",
        ["tenant_id", "model_id"],
    )

    PROMETHEUS_AVAILABLE = True
    logger.info("Prometheus client available — metrics will be exported")

except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus_client not installed; Prometheus metrics will be skipped. "
        "Install with: pip install prometheus_client"
    )


# ---------------------------------------------------------------------------
# PSI — Population Stability Index
# ---------------------------------------------------------------------------

_DEFAULT_BINS = 10
_EPSILON = 1e-8  # avoids log(0) and division by zero


def population_stability_index(
    expected: Iterable[float],
    actual: Iterable[float],
    bins: int = _DEFAULT_BINS,
) -> float:
    """
    Compute Population Stability Index using the standard histogram-bin formula.

    PSI = Σ (actual_pct - expected_pct) * ln(actual_pct / expected_pct)

    A PSI < 0.10 indicates no significant drift.
    A PSI between 0.10 – 0.25 is a moderate shift worth monitoring.
    A PSI >= 0.25 indicates significant population shift — retraining is advised.

    Args:
        expected: Reference distribution (baseline telemetry).
        actual:   Current distribution (recent telemetry).
        bins:     Number of histogram bins (default: 10).

    Returns:
        PSI score (float ≥ 0). Returns 0.0 if either distribution is empty.
    """
    expected_values = list(expected)
    actual_values = list(actual)

    if not expected_values or not actual_values:
        return 0.0

    # Build shared bin edges from the combined range
    combined = expected_values + actual_values
    min_val, max_val = min(combined), max(combined)

    if min_val == max_val:
        # All values identical — no drift measurable
        return 0.0

    bin_width = (max_val - min_val) / bins
    edges: List[float] = [min_val + i * bin_width for i in range(bins + 1)]
    edges[-1] = max_val + _EPSILON  # ensure the max value falls in the last bin

    def _bin_counts(values: List[float]) -> List[int]:
        counts = [0] * bins
        for v in values:
            for b in range(bins):
                if edges[b] <= v < edges[b + 1]:
                    counts[b] += 1
                    break
        return counts

    expected_counts = _bin_counts(expected_values)
    actual_counts = _bin_counts(actual_values)

    n_expected = len(expected_values)
    n_actual = len(actual_values)

    psi = 0.0
    for exp_c, act_c in zip(expected_counts, actual_counts):
        exp_pct = (exp_c / n_expected) if exp_c > 0 else _EPSILON
        act_pct = (act_c / n_actual) if act_c > 0 else _EPSILON
        psi += (act_pct - exp_pct) * math.log(act_pct / exp_pct)

    return max(0.0, psi)


# ---------------------------------------------------------------------------
# Adversarial AUC
# ---------------------------------------------------------------------------


def adversarial_auc(
    baseline_features: List[List[float]],
    current_features: List[List[float]],
) -> float:
    """
    Estimate distribution shift via adversarial validation AUC.

    Labels baseline samples as 0 and current samples as 1, then trains a
    lightweight Random Forest binary classifier and returns its cross-validated
    AUC-ROC score.

    An AUC close to 0.50 means the two distributions are indistinguishable.
    An AUC >= 0.72 (configurable) indicates meaningful drift.

    Requires: scikit-learn

    Returns:
        AUC-ROC float in [0.5, 1.0], or 0.5 if sklearn is unavailable or
        insufficient data.
    """
    if not baseline_features or not current_features:
        return 0.5

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        import numpy as np
    except ImportError:
        logger.warning(
            "scikit-learn not installed; adversarial AUC check will be skipped. "
            "Install with: pip install scikit-learn"
        )
        return 0.5

    try:
        X = np.array(baseline_features + current_features, dtype=float)
        y = np.array(
            [0] * len(baseline_features) + [1] * len(current_features), dtype=int
        )

        if len(X) < 10:
            # Not enough samples for reliable cross-validation
            return 0.5

        clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1)
        scores = cross_val_score(clf, X, y, cv=min(3, len(X) // 2), scoring="roc_auc")
        return float(scores.mean())
    except Exception as exc:
        logger.warning("Adversarial AUC computation failed: %s", exc)
        return 0.5


# ---------------------------------------------------------------------------
# Prometheus emission helpers (no-ops when Prometheus is unavailable)
# ---------------------------------------------------------------------------


def record_psi(tenant_id: str, model_id: str, psi: float) -> None:
    """Emit PSI gauge to Prometheus."""
    if PROMETHEUS_AVAILABLE:
        _DRIFT_PSI_GAUGE.labels(tenant_id=tenant_id, model_id=model_id).set(psi)


def record_adversarial_auc(tenant_id: str, model_id: str, auc: float) -> None:
    """Emit adversarial AUC gauge to Prometheus."""
    if PROMETHEUS_AVAILABLE:
        _DRIFT_AUC_GAUGE.labels(tenant_id=tenant_id, model_id=model_id).set(auc)


def record_drift_event(tenant_id: str, model_id: str, metric: str) -> None:
    """Increment the drift-events counter."""
    if PROMETHEUS_AVAILABLE:
        _DRIFT_EVENT_COUNTER.labels(
            tenant_id=tenant_id, model_id=model_id, metric=metric
        ).inc()


def record_retrain_trigger(tenant_id: str, model_id: str, reason: str) -> None:
    """Increment the retraining-trigger counter."""
    if PROMETHEUS_AVAILABLE:
        _RETRAIN_TRIGGER_COUNTER.labels(
            tenant_id=tenant_id, model_id=model_id, reason=reason
        ).inc()


def record_queue_depth(tenant_id: str, model_id: str, depth: int) -> None:
    """Set the telemetry queue depth gauge."""
    if PROMETHEUS_AVAILABLE:
        _TELEMETRY_RECORDS_GAUGE.labels(tenant_id=tenant_id, model_id=model_id).set(depth)


def drift_check_timer(tenant_id: str, model_id: str):
    """
    Context manager / decorator for timing a drift check cycle.

    Usage:
        with drift_check_timer("client-a", "fraudnet-v1"):
            ...
    """
    if PROMETHEUS_AVAILABLE:
        return _INFERENCE_LATENCY_HISTOGRAM.labels(
            tenant_id=tenant_id, model_id=model_id
        ).time()

    # Fallback no-op context manager
    import contextlib
    return contextlib.nullcontext()
