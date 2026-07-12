# Deployment Guide

## Prerequisites

| Tool | Minimum Version | Purpose |
|------|----------------|---------|
| Docker | 20.10+ | Container runtime |
| Docker Compose | 2.0+ | Local orchestration |
| Python | 3.9+ | Dev / test |
| kubectl | 1.26+ | Kubernetes deployment |
| kustomize | 5.0+ | Manifest management |

---

## Environment Variables

### Inference Service

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `DATABASE_URL` | *(unset — uses in-memory)* | PostgreSQL DSN for persistent registry |
| `MODEL_ROLE` | `baseline` | `baseline` or `candidate` |
| `DEVICE` | `cpu` | PyTorch device (`cpu` or `cuda`) |

### Admin API

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `dev-secret-key-change-in-production` | HMAC secret for JWT signing |
| `DATABASE_URL` | *(unset)* | PostgreSQL DSN |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis for rate-limiter state |
| `CELERY_BROKER_URL` | `redis://localhost:6379/1` | Celery task broker |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/2` | Celery result store |

### Worker

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Telemetry Redis |
| `CELERY_BROKER_URL` | `redis://localhost:6379/1` | Broker for retraining jobs |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/2` | Result backend |
| `WORKER_CHECK_INTERVAL_SECONDS` | `10` | Drift check frequency (seconds) |
| `TELEMETRY_WINDOW_SIZE` | `500` | Records per drift check |
| `DEFAULT_PSI_THRESHOLD` | `0.25` | Global PSI drift threshold |
| `DEFAULT_AUC_THRESHOLD` | `0.72` | Global adversarial AUC threshold |
| `WORKER_DAEMON` | `0` | Set to `1` for continuous polling mode |
| `RETRAINING_SIMULATION_SECONDS` | `0.1` | *(test only)* sleep duration |
| `TENANT_ID` | `client-a` | Tenant to monitor in standalone mode |
| `MODEL_ID` | `fraudnet-v1` | Model to monitor in standalone mode |

---

## Local Development

### Option A — Plain Python

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r admin_api/requirements.txt
pip install -r worker/requirements.txt

# 3. Start Redis (Docker for convenience)
docker run -d -p 6379:6379 redis:7-alpine

# 4. Run the services
uvicorn inference.app_multitenant:app --port 8080 --reload &
uvicorn admin_api.app:app --port 8003 --reload &

# 5. Start the Celery worker (solo pool avoids CUDA fork issues)
celery -A worker.retraining_tasks worker -P solo --loglevel=info &

# 6. Run all tests
python test_phase1_generality.py
python test_phase2_multitenant.py
python test_phase3_control_plane.py
```

### Option B — Docker Compose

```bash
docker compose -f docker-compose.multitenant.yml up --build
```

Services started:
- `inference` → http://localhost:8080
- `admin-api` → http://localhost:8003
- `worker`    → Celery daemon
- `redis`     → localhost:6379
- `postgres`  → localhost:5432 *(optional; set DATABASE_URL to activate)*

---

## Kubernetes Deployment (Kustomize)

### Base Manifests

```
kubernetes/
├── base/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secret.yaml               ← replace with your secret manager
│   ├── redis.yaml
│   ├── postgres.yaml
│   ├── inference-deployment.yaml
│   ├── admin-api-deployment.yaml
│   ├── worker-deployment.yaml
│   └── kustomization.yaml
└── overlays/
    ├── dev/
    └── prod/
```

### Deploy to dev

```bash
# Build and push images first
docker build -t yourregistry/inference:latest inference/
docker build -t yourregistry/admin-api:latest admin_api/
docker build -t yourregistry/worker:latest worker/
docker push yourregistry/inference:latest
docker push yourregistry/admin-api:latest
docker push yourregistry/worker:latest

# Apply dev overlay
kubectl apply -k kubernetes/overlays/dev
```

### Deploy to production

```bash
kubectl apply -k kubernetes/overlays/prod
```

> **Important**: Before production deployment, update `kubernetes/base/secret.yaml`
> to reference your cloud secret manager (AWS Secrets Manager, GCP Secret Manager,
> or HashiCorp Vault). Never commit real secrets to Git.

---

## Secrets Management

All sensitive configuration should be injected as environment variables from
a secrets backend — not stored in the image or plain ConfigMaps.

**Recommended approach (Vault + Kubernetes Auth):**

```yaml
# In your deployment spec
env:
  - name: SECRET_KEY
    valueFrom:
      secretKeyRef:
        name: mlops-secrets
        key: jwt-secret-key
  - name: DATABASE_URL
    valueFrom:
      secretKeyRef:
        name: mlops-secrets
        key: postgres-dsn
```

**Vault Agent Injector** can auto-populate secrets as files or environment
variables without any code changes in the application.

---

## TLS / mTLS

### Service-to-Service (mTLS)

Use a service mesh (Istio or Linkerd) to enforce mTLS between the inference,
admin, and worker services transparently at the network layer:

```bash
# Istio example — enable strict mTLS for the mlops namespace
kubectl apply -f - <<EOF
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: mlops
spec:
  mtls:
    mode: STRICT
EOF
```

### Ingress (TLS termination)

Terminate TLS at the Envoy/Ingress layer. If using the bundled Envoy config,
add a TLS `transport_socket` block to the listener:

```yaml
transport_socket:
  name: envoy.transport_sockets.tls
  typed_config:
    "@type": type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.DownstreamTlsContext
    common_tls_context:
      tls_certificates:
        - certificate_chain: { filename: "/certs/server.crt" }
          private_key: { filename: "/certs/server.key" }
```

---

## CI/CD Pipeline Blueprint

```
┌──────────────┐   push    ┌───────────────┐
│  Git commit  │ ────────► │   CI (GitHub  │
│              │           │   Actions /   │
│              │           │   Cloud Build)│
└──────────────┘           └──────┬────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  1. Lint & Type-check       │
                    │     (ruff, mypy)            │
                    │  2. Unit tests              │
                    │     (pytest, all phases)    │
                    │  3. Build Docker images     │
                    │  4. Push to registry        │
                    │  5. Deploy to dev cluster   │
                    │  6. Integration smoke test  │
                    │  7. Manual gate → prod      │
                    └─────────────────────────────┘
```

**Example GitHub Actions step:**

```yaml
- name: Run all tests
  run: |
    pip install -r requirements.txt
    python test_phase1_generality.py
    python test_phase2_multitenant.py
    python test_phase3_control_plane.py
```

---

## Resource Recommendations (Kubernetes)

| Service | CPU Request | CPU Limit | Memory Request | Memory Limit |
|---------|------------|-----------|----------------|--------------|
| inference | 250m | 1000m | 512Mi | 1Gi |
| admin-api | 100m | 500m | 256Mi | 512Mi |
| worker (CPU) | 500m | 2000m | 1Gi | 2Gi |
| worker (GPU) | 500m | 2000m | 2Gi | 4Gi |
| redis | 100m | 500m | 256Mi | 1Gi |
| postgres | 250m | 1000m | 512Mi | 2Gi |

> Worker memory limits are intentionally tight. If an EWC job exceeds the limit,
> the container restarts rather than causing an OOM cascade on the node.

---

## Production Checklist

- [ ] `SECRET_KEY` set to a strong random value (≥32 bytes) via secrets manager
- [ ] `DATABASE_URL` wired to PostgreSQL (removes in-memory registry)
- [ ] Redis Sentinel / Redis Cluster configured for HA
- [ ] TLS termination at Envoy ingress
- [ ] mTLS between services via service mesh
- [ ] `prefetch_count=1` on Celery worker (already configured in code)
- [ ] Kubernetes resource limits set per table above
- [ ] Prometheus + Grafana deployed and scraping `/metrics` endpoints
- [ ] Alert rules configured for PSI > 0.25, AUC > 0.72, error rate > 1%
- [ ] Log aggregation (ELK / Loki) connected to all pods
- [ ] CI/CD pipeline automated through to dev; manual gate for prod
- [ ] Load test with ≥ 2 concurrent tenants before go-live
- [ ] Runbooks documented for drift events, OOM restarts, and DB failover
