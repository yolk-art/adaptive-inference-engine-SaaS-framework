"""
Async retraining orchestration using Celery.

Redis is used as both broker and result backend by default:
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/2
"""

import os
from typing import Dict


TASK_NAME = "worker.retraining_tasks.retrain_model"


def enqueue_retraining_job(
    tenant_id: str,
    model_id: str,
    trigger_reason: str = "manual",
    force_retrain: bool = False,
) -> Dict:
    celery_app = _get_celery_app()
    async_result = celery_app.send_task(
        TASK_NAME,
        kwargs={
            "tenant_id": tenant_id,
            "model_id": model_id,
            "trigger_reason": trigger_reason,
            "force_retrain": force_retrain,
        },
    )
    return {
        "job_id": async_result.id,
        "tenant_id": tenant_id,
        "model_id": model_id,
        "trigger_reason": trigger_reason,
        "force_retrain": force_retrain,
        "status": "queued",
    }


def get_retraining_job_status(job_id: str) -> Dict:
    celery_app = _get_celery_app()
    result = celery_app.AsyncResult(job_id)
    response = {
        "job_id": job_id,
        "status": result.status.lower(),
    }
    if result.successful():
        response["result"] = result.result
    elif result.failed():
        response["error"] = str(result.result)
    return response


def _get_celery_app():
    try:
        from celery import Celery
    except ImportError as exc:
        raise ImportError("celery is required for async retraining") from exc

    broker_url = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/1"))
    result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
    return Celery("adaptive_inference_admin", broker=broker_url, backend=result_backend)
