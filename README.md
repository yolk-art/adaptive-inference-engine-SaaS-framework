# Adaptive Inference Engine - SaaS Multi-Tenant Framework

A production-ready, multi-tenant Machine Learning Operations (MLOps) platform that transforms monolithic ML pipelines into enterprise-grade SaaS services.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    CLIENT APPLICATIONS                           │
│          (Client-A, Client-B, Client-C, ...)                    │
└──────────────────────────────┬──────────────────────────────────┘
                   │ HTTP (X-Tenant-ID header)
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                  ENVOY PROXY (Layer 7)                           │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Lua Filter: Extract X-Tenant-ID, Route to Cluster         │  │
│  │ Dynamic Cluster Selection: tenant-cluster-{id}            │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────┬────────────────────────────┬──────────────────────────┘
    ┌──────┴──────────────────┬────────┴──────────────┬────────────┐
    ▼                        ▼                       ▼            ▼
┌──────────┐   ┌──────────┐  ┌───────┐  ┌────────────────┐
│ Client-A │   │ Client-B │  │Client-C│  │ Admin API      │
│Inference │   │Inference │  │Inference  │ (Phase 3)      │
└──────────┘   └──────────┘  └───────┘  └────────────────┘
    │               │             │
    └───────────────┼─────────────┘
                    ▼
        ┌──────────────────────────┐
        │    SHARED REDIS          │
        │ {client-a}:telemetry     │
        │ {client-b}:telemetry     │
        │ {client-c}:telemetry     │
        │ (Hash-tagged keys)       │
        └──────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │  Multi-Tenant Worker     │
        │  - Drift Detection       │
        │  - EWC Retraining        │
        │  - Tenant Isolation      │
        └──────────────────────────┘
```

## Phases

### Phase 1: Generality Foundation ✅
- **ModelRuntime** abstract base class for framework-agnostic inference
- Dynamic schema validation (Pydantic)
- Externalized configuration (JSON/YAML)
- **Acid Test**: Swap PyTorch FraudNet ↔ Scikit-Learn Churn Predictor without code changes

### Phase 2: Multi-Tenancy & Data Isolation ✅
- **Redis Hash-Tagging**: `{tenant_id}:key` prevents CROSSSLOT errors in clustered Redis
- **Tenant Model Registry**: Maps tenant_id → model_id:version → storage_path
- **Envoy Dynamic Routing**: Lua filters extract X-Tenant-ID, route to tenant-specific clusters
- **Logical Isolation**: RabbitMQ virtual hosts per tenant, Redis namespace prefixes
- **Telemetry Isolation**: Client A data never leaks into Client B drift detection

### Phase 3: Production Hardening & Control Plane ✅
- **FastAPI Admin Control Plane**:
  - `/register-tenant` — Self-service tenant registration
  - `/models/register` — Upload models and configure drift thresholds
  - `/models/{id}/retrain` — Manually initiate EWC retraining
  - `/health`, `/status` — Observability endpoints
- **Async Retraining**: Celery task with full EWC training loop (Fisher diagonal + regularisation loss) and sklearn refit branch
- **Rate Limiting**: Token bucket algorithm per tenant
- **Auth & Authorization**: Bearer token validation, X-Tenant-ID header enforcement
- **Prometheus Metrics**: Per-tenant drift detection, retraining latency, request throughput

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.9+
- Redis
- PostgreSQL (for production registry)

### Local Development

```bash
# Clone the repo
git clone https://github.com/yolk-art/adaptive-inference-engine-SaaS-framework.git
cd adaptive-inference-engine-SaaS-framework

# Install dependencies
pip install -r requirements.txt

# Run all Phase tests
python test_phase1_generality.py
python test_phase2_multitenant.py
python test_phase3_control_plane.py

# Start with docker-compose
docker-compose -f docker-compose.multitenant.yml up -d

# Test inference endpoint
curl -X POST http://localhost:8080/predict \
  -H "X-Tenant-ID: client-a" \
  -H "X-Model-ID: fraudnet-v1" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 100.0,
    "distance": 50.0,
    "velocity": 10.0,
    "age": 365.0,
    "risk_score": 0.5
  }'

# Test admin API (get token)
curl -X POST http://localhost:8003/auth/token?tenant_id=client-a

# Register a model
curl -X POST http://localhost:8003/models/register \
  -H "X-Tenant-ID: client-a" \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "fraudnet-v1",
    "model_version": "1.0.0",
    "storage_path": "s3://bucket/models/fraudnet.pt",
    "schema_definition": {"amount": {"type": "float"}},
    "drift_thresholds": {"psi_threshold": 0.25}
  }'
```

## Project Structure

```
adaptive-inference-engine-SaaS-framework/
├── README.md
├── requirements.txt
├── docker-compose.multitenant.yml
│
├── inference/                          # Phase 1-2: Inference Services
│   ├── model_runtime.py               # Abstract base class (Phase 1)
│   ├── fraudnet_runtime.py            # PyTorch implementation (Phase 1)
│   ├── churn_runtime.py               # Scikit-Learn implementation (Phase 1)
│   ├── config_fraudnet.json           # FraudNet config (Phase 1)
│   ├── config_churn.json              # Churn config (Phase 1)
│   ├── tenant_redis_client.py         # Redis hash-tagging (Phase 2)
│   ├── tenant_model_registry.py       # Tenant model metadata (Phase 2)
│   ├── app_multitenant.py             # FastAPI multi-tenant service (Phase 2)
│   └── Dockerfile
│
├── worker/                             # Phase 2-3: Background Workers
│   ├── worker_multitenant.py          # Multi-tenant drift detection (Phase 2-3)
│   ├── retraining_tasks.py            # Celery EWC retraining tasks (Phase 3)
│   ├── metrics.py                     # PSI, Adversarial AUC, Prometheus (Phase 3)
│   ├── Dockerfile.worker
│   └── requirements.txt
│
├── admin_api/                          # Phase 3: Control Plane
│   ├── app.py                         # FastAPI admin endpoints (Phase 3)
│   ├── models.py                      # Pydantic schemas (Phase 3)
│   ├── auth.py                        # Token validation (Phase 3)
│   ├── retraining_orchestrator.py     # Retraining job boundary (Phase 3)
│   ├── rate_limiter.py                # Token bucket middleware (Phase 3)
│   ├── Dockerfile
│   └── requirements.txt
│
├── envoy-config/                       # Phase 2-3: Proxy Configuration
│   └── envoy_multitenant.yaml         # Lua filter + dynamic routing
│
├── database/                           # Phase 2-3: Persistence
│   └── tenant_model_registry.sql      # PostgreSQL schema
│
├── test_phase1_generality.py          # Phase 1: Acid test
├── test_phase2_multitenant.py         # Phase 2: Isolation test
├── test_phase3_control_plane.py       # Phase 3: Integration test (12 cases)
│
└── docs/
    ├── ARCHITECTURE.md
    ├── DEPLOYMENT.md
    ├── MULTI_TENANCY.md
    └── TROUBLESHOOTING.md
```

## Key Features

### Generality (Phase 1)
✅ Framework-agnostic model serving (PyTorch, Scikit-Learn, ONNX, etc.)
✅ Dynamic schema validation from config files
✅ Zero code changes to swap model types

### Multi-Tenancy (Phase 2)
✅ Redis hash-tagged keys prevent clustering errors
✅ Tenant-isolated telemetry and metrics
✅ Envoy Lua filters for dynamic routing
✅ Per-tenant model registries and drift thresholds

### Production Hardening (Phase 3)
✅ Self-service tenant registration
✅ Async EWC retraining via Celery (solo pool safe; Ray Serve-compatible)
✅ Token bucket rate limiting per tenant
✅ Bearer token authentication
✅ Prometheus metrics: PSI, adversarial AUC, drift events, retraining triggers
✅ PostgreSQL backend (set `DATABASE_URL`); in-memory fallback for dev/test

## Testing

```bash
# Run Phase 1 test
python test_phase1_generality.py
# Output: ✓✓✓ ALL TESTS PASSED ✓✓✓

# Run Phase 2 test
python test_phase2_multitenant.py
# Output: ✓✓✓ ALL PHASE 2 TESTS PASSED ✓✓✓

# Run Phase 3 test (12 test cases)
python test_phase3_control_plane.py
# Output: ✓✓✓ ALL PHASE 3 TESTS PASSED ✓✓✓
```

## Deployment

### Local (Docker Compose)
```bash
docker-compose -f docker-compose.multitenant.yml up
```

### Kubernetes (Kustomize)
```bash
kubectl apply -k kubernetes/overlays/dev
kubectl apply -k kubernetes/overlays/prod
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for full deployment guide including
environment variables, secrets management, TLS/mTLS, and CI/CD pipeline blueprint.

## Production Checklist

- [ ] Set `SECRET_KEY` to a strong random value via secrets manager
- [ ] Set `DATABASE_URL` to activate PostgreSQL registry backend
- [ ] Set up Redis Sentinel or Redis Cluster for HA
- [ ] Configure TLS/mTLS for inter-service communication
- [ ] Set up Vault for secrets management
- [ ] Configure RBAC and Kubernetes NetworkPolicy (default-deny)
- [ ] Set up log aggregation (ELK, Loki, etc.)
- [ ] Configure Prometheus + Grafana alert thresholds
- [ ] Load test with 2+ concurrent tenants
- [ ] Set up CI/CD pipeline for model deployment
- [ ] Document SLAs and runbooks

## License

Dual-licensed under AGPLv3 and Apache 2.0
