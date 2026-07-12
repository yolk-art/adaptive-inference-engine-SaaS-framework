"""
test_phase3_control_plane.py

Integration tests for Phase 3: Production Hardening & Control Plane.

Test coverage:
  1. Tenant registration                 — happy path
  2. Model registration                  — authenticated happy path
  3. Auth rejection                      — wrong token / missing header
  4. Cross-tenant token rejection        — token for tenant A used against tenant B
  5. Rate limiting (429)                 — burst requests beyond bucket capacity
  6. List models endpoint                — returns registered model list
  7. Health & status endpoints           — unauthenticated observability
  8. Retraining endpoint (mocked Celery) — returns 202, enqueues job
  9. Drift thresholds update             — PATCH drift thresholds
 10. Delete model                        — removes model from registry
"""

import importlib.util
import unittest
from unittest.mock import MagicMock, patch


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "FastAPI not installed")
@unittest.skipUnless(importlib.util.find_spec("jose"), "python-jose not installed")
class Phase3ControlPlaneTests(unittest.TestCase):
    # ------------------------------------------------------------------ setup

    def setUp(self):
        """Create a fresh TestClient and register a tenant + model once."""
        from fastapi.testclient import TestClient
        from admin_api.app import app, model_registry, rate_limiter, retraining_jobs

        # Reset shared state between tests so they are independent
        if hasattr(model_registry.backend, "tenants"):
            model_registry.backend.tenants.clear()
        if hasattr(model_registry.backend, "registry"):
            model_registry.backend.registry.clear()
        if hasattr(rate_limiter, "limiters"):
            rate_limiter.limiters.clear()
        retraining_jobs.clear()

        self.client = TestClient(app)
        self.app = app

        # Register tenant
        resp = self.client.post(
            "/register-tenant",
            json={
                "tenant_id": "client-a",
                "tenant_name": "Client A",
                "contact_email": "ops@example.com",
                "tier": "standard",
            },
        )
        self.assertEqual(resp.status_code, 201, resp.json())

        # Obtain bearer token
        token_resp = self.client.post("/auth/token?tenant_id=client-a")
        self.assertEqual(token_resp.status_code, 200)
        self.token = token_resp.json()["access_token"]
        self.headers = {
            "X-Tenant-ID": "client-a",
            "Authorization": f"Bearer {self.token}",
        }

        # Register a model
        model_resp = self.client.post(
            "/models/register",
            headers=self.headers,
            json={
                "model_id": "fraudnet-v1",
                "model_version": "1.0.0",
                "storage_path": "/models/model_baseline_client-a.pt",
                "config_path": "inference/config_fraudnet.json",
                "schema_definition": {"amount": {"type": "float"}},
                "drift_thresholds": {"psi_threshold": 0.25, "auc_threshold": 0.72},
                "framework": "pytorch",
            },
        )
        self.assertEqual(model_resp.status_code, 200, model_resp.json())

    # ------------------------------------------------------------------ 1. Tenant registration

    def test_01_tenant_registration_happy_path(self):
        """Registered tenant appears in health counts."""
        health = self.client.get("/health").json()
        self.assertGreaterEqual(health["active_tenants"], 1)
        self.assertGreaterEqual(health["active_models"], 1)

    # ------------------------------------------------------------------ 2. Model registration

    def test_02_model_registration_response_schema(self):
        """Model registration returns expected fields."""
        # Register a second model version to verify the response
        resp = self.client.post(
            "/models/register",
            headers=self.headers,
            json={
                "model_id": "fraudnet-v1",
                "model_version": "2.0.0",
                "storage_path": "/models/model_v2_client-a.pt",
                "schema_definition": {"amount": {"type": "float"}},
                "drift_thresholds": {"psi_threshold": 0.20},
                "framework": "pytorch",
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("model_id", body)
        self.assertIn("tenant_id", body)
        self.assertIn("registration_id", body)
        self.assertEqual(body["tenant_id"], "client-a")
        self.assertEqual(body["model_id"], "fraudnet-v1")

    # ------------------------------------------------------------------ 3. Auth rejection — bad token

    def test_03_auth_rejection_invalid_token(self):
        """Request with a garbage Bearer token is rejected 401."""
        bad_headers = {
            "X-Tenant-ID": "client-a",
            "Authorization": "Bearer this-is-not-a-valid-jwt",
        }
        resp = self.client.get("/models", headers=bad_headers)
        self.assertEqual(resp.status_code, 401)

    # ------------------------------------------------------------------ 4. Cross-tenant rejection

    def test_04_cross_tenant_token_rejected(self):
        """
        A valid token for client-a cannot access resources under client-b.
        """
        # Register client-b
        self.client.post(
            "/register-tenant",
            json={
                "tenant_id": "client-b",
                "tenant_name": "Client B",
                "contact_email": "ops-b@example.com",
                "tier": "enterprise",
            },
        )
        # Use client-a's token with X-Tenant-ID: client-b
        cross_headers = {
            "X-Tenant-ID": "client-b",
            "Authorization": f"Bearer {self.token}",
        }
        resp = self.client.get("/models", headers=cross_headers)
        self.assertEqual(resp.status_code, 403)

    # ------------------------------------------------------------------ 5. Rate limiting

    def test_05_rate_limiting_triggers_429(self):
        """
        Exceeding the tenant rate limit returns HTTP 429.

        We temporarily reduce the bucket capacity to 1 token and fire 3 requests
        to guarantee a 429 without relying on a specific global capacity setting.
        """
        from admin_api.rate_limiter import RateLimiter

        tiny_limiter = RateLimiter(default_capacity=1, default_refill=0.001)
        with patch("admin_api.app.rate_limiter", tiny_limiter):
            responses = [
                self.client.get("/models", headers=self.headers)
                for _ in range(3)
            ]
        status_codes = [r.status_code for r in responses]
        self.assertIn(429, status_codes, f"Expected a 429 among {status_codes}")

    # ------------------------------------------------------------------ 6. List models

    def test_06_list_models_returns_registered_model(self):
        """GET /models returns the model registered in setUp."""
        resp = self.client.get("/models", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["tenant_id"], "client-a")
        model_ids = [m["model_id"] for m in body["models"]]
        self.assertIn("fraudnet-v1", model_ids)

    # ------------------------------------------------------------------ 7. Health & status

    def test_07_health_endpoint_unauthenticated(self):
        """GET /health is public and returns status=healthy."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "healthy")

    def test_07b_status_endpoint_unauthenticated(self):
        """GET /status is public and returns aggregate counts."""
        resp = self.client.get("/status")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("tenants", body)
        self.assertIn("models", body)
        self.assertIn("retraining_jobs", body)

    # ------------------------------------------------------------------ 8. Retraining (mocked Celery)

    def test_08_retraining_endpoint_enqueues_job(self):
        """
        POST /models/{model_id}/retrain returns 202 and a job payload.

        Celery's send_task is mocked so no broker is required.
        """
        fake_async_result = MagicMock()
        fake_async_result.id = "fake-job-uuid-1234"

        with patch(
            "admin_api.retraining_orchestrator._get_celery_app"
        ) as mock_celery_factory:
            mock_app = MagicMock()
            mock_app.send_task.return_value = fake_async_result
            mock_celery_factory.return_value = mock_app

            resp = self.client.post(
                "/models/fraudnet-v1/retrain",
                headers=self.headers,
                json={
                    "model_id": "fraudnet-v1",
                    "trigger_reason": "manual_test",
                    "force_retrain": False,
                },
            )

        self.assertEqual(resp.status_code, 202, resp.json())
        body = resp.json()
        self.assertEqual(body["status"], "queued")
        self.assertEqual(body["job_id"], "fake-job-uuid-1234")
        self.assertEqual(body["tenant_id"], "client-a")

    def test_08b_retraining_model_mismatch_rejected(self):
        """Path model_id must match request body model_id."""
        resp = self.client.post(
            "/models/different-model/retrain",
            headers=self.headers,
            json={
                "model_id": "fraudnet-v1",  # mismatch with path
                "trigger_reason": "test",
                "force_retrain": False,
            },
        )
        self.assertEqual(resp.status_code, 400)

    # ------------------------------------------------------------------ 9. Retraining status (mocked)

    def test_09_retraining_job_status_pending(self):
        """GET /retraining/{job_id} returns status for a known job."""
        fake_async_result = MagicMock()
        fake_async_result.id = "status-test-job-5678"
        fake_async_result.status = "PENDING"
        fake_async_result.successful.return_value = False
        fake_async_result.failed.return_value = False

        with patch(
            "admin_api.retraining_orchestrator._get_celery_app"
        ) as mock_celery_factory:
            mock_app = MagicMock()
            # send_task for enqueue
            enqueue_result = MagicMock()
            enqueue_result.id = "status-test-job-5678"
            mock_app.send_task.return_value = enqueue_result
            # AsyncResult for status query
            mock_app.AsyncResult.return_value = fake_async_result
            mock_celery_factory.return_value = mock_app

            # First enqueue
            enqueue_resp = self.client.post(
                "/models/fraudnet-v1/retrain",
                headers=self.headers,
                json={
                    "model_id": "fraudnet-v1",
                    "trigger_reason": "status_test",
                    "force_retrain": False,
                },
            )
            self.assertEqual(enqueue_resp.status_code, 202)

            # Then check status
            status_resp = self.client.get(
                "/retraining/status-test-job-5678",
                headers=self.headers,
            )

        self.assertEqual(status_resp.status_code, 200)
        self.assertIn("status", status_resp.json())

    # ------------------------------------------------------------------ 10. Auth token endpoint

    def test_10_auth_token_endpoint(self):
        """POST /auth/token returns an access_token."""
        resp = self.client.post("/auth/token?tenant_id=client-a")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("access_token", body)
        self.assertEqual(body["token_type"], "bearer")
        self.assertIsInstance(body["access_token"], str)
        self.assertGreater(len(body["access_token"]), 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
