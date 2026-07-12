# Architecture

## Overview

The Adaptive Inference Engine SaaS Platform is a three-service, event-driven
MLOps system designed for multi-tenant, production-grade model serving. It
transforms a single-tenant FraudNet PyTorch pipeline into an enterprise SaaS
platform capable of serving hundreds of independent clients from shared
infrastructure.

---

## Service Topology

```
┌──────────────────────────────────────────────────────────────────┐
│                    CLIENT APPLICATIONS                           │
│          (Client-A, Client-B, Client-C, ...)                    │
└──────────────────────┬───────────────────────────────────────────┘
                       │ HTTP (X-Tenant-ID header)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                  ENVOY PROXY (Layer 7)                           │
│  Lua Filter → extract X-Tenant-ID → inject x-target-cluster     │
│  cluster_header routing → tenant-cluster-{id}                   │
└────────┬────────────────────┬──────────────────┬────────────────┘
         ▼                    ▼                  ▼
  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐
  │  Inference  │    │  Inference  │    │     Admin API        │
  │  (Client-A) │    │  (Client-B) │    │  (Control Plane)     │
  └──────┬──────┘    └──────┬──────┘    └─────────┬───────────┘
         │                  │                      │
         └──────────────────┼──────────────────────┘
                            ▼
              ┌─────────────────────────┐
              │       SHARED REDIS       │
              │  {client-a}:telemetry    │
              │  {client-b}:telemetry    │
              │  {client-c}:telemetry    │
              │  (Hash-tagged keyspace)  │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │    DRIFT WORKER          │
              │  PSI + Adversarial AUC   │
              │  EWC Retraining (Celery) │
              │  Prometheus Metrics      │
              └─────────────────────────┘
```

---

## Component Descriptions

### 1. Inference Service (`inference/`)

- **Entry point**: `app_multitenant.py` (FastAPI)
- **Purpose**: Serves real-time predictions for any registered tenant model.
- **Key behaviours**:
  - Reads `X-Tenant-ID` and `X-Model-ID` headers on every request.
  - Lazily loads `ModelRuntime` instances per `(tenant_id, model_id)` pair into
    an in-process dict cache.
  - Writes inference telemetry (features, prediction, probability, timestamp)
    to Redis using `TenantRedisClient` with hash-tagged keys to guarantee
    CROSSSLOT safety in clustered Redis.
  - Falls back to the `TenantModelRegistry` to look up model storage paths and
    config when a model is not yet loaded.
- **Model runtimes**:
  - `FraudNetRuntime` — PyTorch binary classifier (5 features)
  - `ChurnRuntime` — Scikit-Learn Random Forest (10 features)
  - Both implement the `ModelRuntime` abstract base class (Phase 1).
- **Schema validation**: Pydantic schemas are built dynamically from
  `config_*.json` files via `ModelRuntime.get_feature_schema()`.

### 2. Admin API (`admin_api/`)

- **Entry point**: `app.py` (FastAPI, port 8003)
- **Purpose**: Control plane for self-service tenant and model management.
- **Endpoints**:

  | Method | Path | Auth | Description |
  |--------|------|------|-------------|
  | POST | `/auth/token` | None | Issue JWT for a tenant |
  | POST | `/register-tenant` | None | Create tenant record |
  | POST | `/models/register` | Bearer | Register model metadata |
  | GET | `/models` | Bearer | List tenant's models |
  | POST | `/models/{id}/retrain` | Bearer | Enqueue EWC retraining |
  | GET | `/retraining/{job_id}` | Bearer | Poll retraining status |
  | GET | `/health` | None | Service health |
  | GET | `/status` | None | Aggregate counts |

- **Auth**: HMAC-HS256 JWT tokens. The `X-Tenant-ID` header is validated
  against the `tenant_id` claim in the token to prevent cross-tenant access.
- **Rate Limiting**: Token bucket per tenant (default: 60 req/min, configurable).

### 3. Drift Detection Worker (`worker/`)

- **Entry point**: `worker_multitenant.py` (run as Celery daemon or standalone)
- **Purpose**: Continuously monitors prediction distribution drift and triggers
  automatic retraining.
- **Drift metrics**:
  - **PSI** (Population Stability Index): histogram-bin formula across the
    last `TELEMETRY_WINDOW_SIZE` predictions vs. a uniform 0.5 baseline.
    Threshold: PSI ≥ 0.25 triggers retraining.
  - **Adversarial AUC**: trains a lightweight Random Forest to distinguish
    the first half of the window (baseline) from the second half (current).
    Threshold: AUC ≥ 0.72 triggers retraining.
- **Retraining**: Enqueues a `retrain_model` Celery task when drift is detected.
  - PyTorch models: EWC fine-tuning (Fisher diagonal + regularisation loss).
  - Sklearn models: direct `clf.fit()` on the telemetry window.
- **Prometheus**: Emits per-tenant PSI, AUC, drift event counts, and retraining
  trigger counts via `prometheus_client`.

---

## Data Flow: Inference Request

```
Client → Envoy → Inference Service
  1. Envoy Lua filter reads X-Tenant-ID header.
  2. Envoy injects x-target-cluster = "tenant-cluster-{tenant_id}".
  3. Envoy routes to the correct upstream cluster.
  4. FastAPI endpoint validates X-Tenant-ID.
  5. Model is lazily loaded from disk / registry if not cached.
  6. Features are schema-validated against the dynamic Pydantic model.
  7. Prediction is returned as JSON.
  8. Telemetry is asynchronously written to Redis under {tenant_id}:key.
```

## Data Flow: EWC Retraining

```
Worker detects drift
  ↓
Calls admin_api.retraining_orchestrator.enqueue_retraining_job()
  ↓
Celery task "worker.retraining_tasks.retrain_model" is queued (Redis broker)
  ↓
Celery worker picks up task
  ↓
Pulls last N telemetry records from Redis
  ↓
Loads model weights from storage_path in Tenant Model Registry
  ↓
Computes Fisher Information Matrix diagonal on telemetry data
  ↓
Runs N epochs with task_loss + EWC_loss
  ↓
Saves updated weights back to storage_path
  ↓
Prometheus counters incremented; result returned to Celery result backend
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Redis hash-tags `{tenant_id}:key` | Prevents CROSSSLOT errors in clustered Redis when using multi-key pipelines |
| Abstract `ModelRuntime` base class | Framework-agnostic serving: swap PyTorch ↔ sklearn without code changes |
| JWT with `tenant_id` claim | Stateless auth that prevents cross-tenant token reuse |
| Token bucket rate limiter | Per-tenant fairness; prevents noisy-neighbor DoS |
| PSI + adversarial AUC | Complementary signals: PSI catches marginal shifts; AUC catches complex multi-feature drift |
| EWC regularisation | Prevents catastrophic forgetting of baseline performance during retraining |
| Celery `prefetch_count=1` | Prevents a single worker from pulling multiple heavy GPU jobs simultaneously |
| PostgreSQL backend (optional) | In-memory fallback for dev/test; production uses SQLAlchemy + PostgreSQL |

---

## Next Production Steps

1. Replace in-memory `TenantModelRegistry` with PostgreSQL (set `DATABASE_URL`).
2. Wire Envoy xDS API to receive dynamic cluster updates from the Admin API.
3. Add TLS/mTLS for all inter-service communication.
4. Integrate Vault for `SECRET_KEY` and database credentials.
5. Set up RBAC + Kubernetes NetworkPolicy (`default-deny` east-west traffic).
6. Add log aggregation (ELK stack or Grafana Loki).
7. Configure Grafana dashboards for the Prometheus metrics.
8. Set up CI/CD pipeline for automated model promotion.
