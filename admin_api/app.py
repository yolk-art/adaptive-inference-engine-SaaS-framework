"""
FastAPI control plane for tenant and model administration.

State is persisted through TenantModelRegistry. Set DATABASE_URL to use
PostgreSQL; omit it for the in-memory development backend.
"""

import time
from datetime import datetime, timezone
from typing import Dict

from fastapi import FastAPI, Header, HTTPException, Query, status

from admin_api.auth import create_access_token, verify_bearer_token
from admin_api.models import (
    HealthResponse,
    ModelRegistrationRequest,
    ModelRegistrationResponse,
    RetrainingRequest,
    TenantRegistrationRequest,
)
from admin_api.rate_limiter import RateLimiter
from admin_api.retraining_orchestrator import (
    enqueue_retraining_job,
    get_retraining_job_status,
)
from inference.tenant_model_registry import ModelMetadata, TenantModelRegistry


app = FastAPI(title="Adaptive Inference Admin API", version="0.1.0")
started_at = time.time()
rate_limiter = RateLimiter()
model_registry = TenantModelRegistry()
retraining_jobs: Dict[str, Dict] = {}


def require_tenant(authorization: str, x_tenant_id: str) -> str:
    token_tenant_id = verify_bearer_token(authorization)
    if token_tenant_id != x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token tenant does not match X-Tenant-ID",
        )
    if not rate_limiter.is_allowed(x_tenant_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Tenant rate limit exceeded",
        )
    return token_tenant_id


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        version=app.version,
        uptime_seconds=time.time() - started_at,
        active_tenants=model_registry.count_tenants(),
        active_models=model_registry.count_models(),
    )


@app.post("/auth/token")
def issue_token(tenant_id: str = Query(...)):
    return {"access_token": create_access_token(tenant_id), "token_type": "bearer"}


@app.post("/register-tenant", status_code=status.HTTP_201_CREATED)
def register_tenant(request: TenantRegistrationRequest):
    model_registry.register_tenant(
        tenant_id=request.tenant_id,
        tenant_name=request.tenant_name,
        contact_email=request.contact_email,
        tier=request.tier,
    )
    return {
        "status": "success",
        "tenant_id": request.tenant_id,
        "message": f"Tenant {request.tenant_id} registered",
    }


@app.post("/models/register", response_model=ModelRegistrationResponse)
def register_model(
    request: ModelRegistrationRequest,
    x_tenant_id: str = Header(...),
    authorization: str = Header(...),
):
    require_tenant(authorization, x_tenant_id)
    config_path = request.config_path or _default_config_for_framework(request.framework)
    metadata = model_registry.register_model(
        tenant_id=x_tenant_id,
        model_id=request.model_id,
        model_version=request.model_version,
        storage_path=request.storage_path,
        config_path=config_path,
        schema_definition=request.schema_definition,
        drift_thresholds=request.drift_thresholds,
        framework=request.framework,
    )
    return ModelRegistrationResponse(
        status="success",
        model_id=metadata.model_id,
        model_version=metadata.model_version,
        tenant_id=metadata.tenant_id,
        registration_id=f"{metadata.tenant_id}:{metadata.model_id}:{metadata.model_version}",
        created_at=datetime.now(timezone.utc),
        message=f"Model {metadata.model_id}:{metadata.model_version} registered",
    )


@app.get("/models")
def list_models(x_tenant_id: str = Header(...), authorization: str = Header(...)):
    require_tenant(authorization, x_tenant_id)
    return {
        "tenant_id": x_tenant_id,
        "models": [
            _metadata_to_dict(metadata)
            for metadata in model_registry.list_tenant_models(x_tenant_id).values()
        ],
    }


@app.post("/models/{model_id}/retrain", status_code=status.HTTP_202_ACCEPTED)
def request_retraining(
    model_id: str,
    request: RetrainingRequest,
    x_tenant_id: str = Header(...),
    authorization: str = Header(...),
):
    require_tenant(authorization, x_tenant_id)
    if request.model_id != model_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path model_id must match request model_id",
        )
    job = enqueue_retraining_job(
        tenant_id=x_tenant_id,
        model_id=model_id,
        trigger_reason=request.trigger_reason,
        force_retrain=request.force_retrain,
    )
    retraining_jobs[job["job_id"]] = job
    return job


@app.get("/retraining/{job_id}")
def retraining_status(
    job_id: str,
    x_tenant_id: str = Header(...),
    authorization: str = Header(...),
):
    require_tenant(authorization, x_tenant_id)
    status_payload = get_retraining_job_status(job_id)
    known_job = retraining_jobs.get(job_id)
    if known_job and known_job["tenant_id"] != x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access another tenant's retraining job",
        )
    return status_payload


@app.get("/status")
def status_report():
    return {
        "tenants": model_registry.count_tenants(),
        "models": model_registry.count_models(),
        "retraining_jobs": len(retraining_jobs),
    }


def _metadata_to_dict(metadata: ModelMetadata) -> Dict:
    return {
        "tenant_id": metadata.tenant_id,
        "model_id": metadata.model_id,
        "model_version": metadata.model_version,
        "storage_path": metadata.storage_path,
        "config_path": metadata.config_path,
        "framework": metadata.framework,
        "drift_thresholds": metadata.drift_thresholds,
    }


def _default_config_for_framework(framework: str) -> str:
    if framework == "sklearn":
        return "inference/config_churn.json"
    return "inference/config_fraudnet.json"
