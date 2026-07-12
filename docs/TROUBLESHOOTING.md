# Troubleshooting

## Quick Reference

| Symptom | Likely Cause | Jump To |
|---------|-------------|---------|
| `CROSSSLOT` Redis error | Key namespacing not using hash-tags | [§ Redis](#redis-crossslot-error) |
| `RuntimeError: Cannot re-initialize CUDA in forked subprocess` | Celery prefork + PyTorch | [§ CUDA](#cuda-fork-conflict-in-celery) |
| `401 Unauthorized` on every request | Missing or expired JWT | [§ Auth](#jwt-authentication-errors) |
| `403 Forbidden` | Token tenant ≠ X-Tenant-ID | [§ Auth](#jwt-authentication-errors) |
| `404 Model not found` | Model not registered or wrong header | [§ Model Loading](#model-not-found-or-fails-to-load) |
| Worker never triggers retraining | PSI below threshold, Celery not running | [§ Worker](#worker-not-triggering-retraining) |
| Tests skip entirely | Missing optional dependency | [§ Tests](#tests-are-skipped) |
| `python` not found (Windows) | PATH / venv issue | [§ Windows](#windows-specific) |

---

## Redis: CROSSSLOT Error

### Symptom

```
redis.exceptions.ResponseError: CROSSSLOT Keys in request don't hash to the same slot
```

### Cause

You are running Redis Cluster and performing a multi-key operation (pipeline,
`MGET`, Lua script) on keys that belong to different cluster slots.

### Fix

Ensure **all** Redis keys for a tenant use the hash-tag format:

```python
# WRONG — keys may land on different slots
redis.set(f"{tenant_id}:model:metrics", value)
redis.set(f"{tenant_id}:telemetry", value)

# CORRECT — hash-tag forces same slot
redis.set(f"{{{tenant_id}}}:model:metrics", value)
redis.set(f"{{{tenant_id}}}:telemetry", value)
```

The `TenantRedisClient` in `inference/tenant_redis_client.py` handles this
automatically. If you are writing custom Redis code, always use:

```python
key = f"{{{tenant_id}}}:{your_suffix}"
```

---

## CUDA Fork Conflict in Celery

### Symptom

```
RuntimeError: Cannot re-initialize CUDA in forked subprocess.
To use CUDA with multiprocessing, you must use the 'spawn' start method.
```

### Cause

Celery's default execution pool (`prefork`) uses `os.fork()` to create worker
processes. PyTorch CUDA contexts are initialised in the parent and cannot be
safely duplicated across a fork boundary.

### Fixes (choose one)

**Option 1 — Solo pool (simplest, no concurrency):**

```bash
celery -A worker.retraining_tasks worker -P solo --loglevel=info
```

This runs all tasks in the main process thread, eliminating forking entirely.
Suitable for CPU-only retraining or low-throughput deployments.

**Option 2 — Force CPU inside the task (already implemented):**

The `_run_ewc_retraining` function defaults to `device="cpu"`, which avoids
CUDA initialisation entirely within the Celery task. Check that `DEVICE=cpu`
is set in the worker's environment.

**Option 3 — Migrate to Ray Serve (recommended for GPU):**

Ray uses gRPC-based actor pools with spawned (not forked) processes:

```bash
pip install "ray[serve]"
# Then replace celery_app.task with a Ray remote actor
```

Ray's actor model isolates CUDA contexts per worker by design.

---

## JWT Authentication Errors

### Symptom: `401 Unauthorized`

```json
{"detail": "Invalid token"}
```

**Possible causes and fixes:**

| Cause | Fix |
|-------|-----|
| Token has expired (default TTL = 60 min) | Request a new token via `POST /auth/token?tenant_id=<id>` |
| `SECRET_KEY` changed after token was issued | Re-issue all tokens; update `SECRET_KEY` consistently across services |
| `python-jose` not installed | `pip install python-jose[cryptography]` |
| Token passed without `Bearer ` prefix | Use `Authorization: Bearer <token>` |

### Symptom: `403 Forbidden`

```json
{"detail": "Token tenant does not match X-Tenant-ID"}
```

The JWT's `tenant_id` claim does not match the `X-Tenant-ID` header. You must
use a token issued **for the same tenant** you are operating on:

```bash
# Get token for client-a
TOKEN=$(curl -s -X POST "http://localhost:8003/auth/token?tenant_id=client-a" | jq -r .access_token)

# Use token with matching header
curl -H "X-Tenant-ID: client-a" -H "Authorization: Bearer $TOKEN" \
     http://localhost:8003/models
```

---

## Model Not Found or Fails to Load

### Symptom: `404 Not Found`

```json
{"detail": "Model fraudnet-v1 not found for tenant client-a"}
```

**Checklist:**
1. Was the model registered? Call `GET /models` (authenticated) to list.
2. Is `X-Model-ID` header set correctly on the `/predict` request?
3. Is `X-Tenant-ID` header set correctly?

### Symptom: Model loads but returns wrong results

The `MODEL_ROLE` environment variable may point the inference service to a
stale candidate model. Check:

```bash
# Inside the container
echo $MODEL_ROLE   # should be "baseline" unless a candidate is promoted
ls /app/models/    # verify model files exist
```

### Symptom: `FileNotFoundError` loading model weights

The `storage_path` registered in the Tenant Model Registry must be accessible
from inside the inference container. For local dev:

```bash
# Mount your model directory into the container
docker run -v /local/models:/app/models inference:latest
```

For production, use a shared volume (AWS EFS, GCP Filestore, Azure Files) or
download from S3/GCS on startup.

---

## Worker Not Triggering Retraining

### Symptom: Drift detected in logs but no Celery job appears

1. **Is Celery running?**

   ```bash
   celery -A worker.retraining_tasks inspect active
   ```

2. **Is the broker reachable?**

   ```bash
   redis-cli -u $CELERY_BROKER_URL ping
   ```

3. **Are thresholds too high?** Check the model's registered `drift_thresholds`:

   ```bash
   curl -H "X-Tenant-ID: client-a" -H "Authorization: Bearer $TOKEN" \
        http://localhost:8003/models
   ```

4. **Not enough telemetry?** The worker skips the check if fewer than 2 records
   exist. Generate some traffic first:

   ```bash
   curl -X POST http://localhost:8080/predict \
        -H "X-Tenant-ID: client-a" -H "X-Model-ID: fraudnet-v1" \
        -H "Content-Type: application/json" \
        -d '{"amount": 100, "distance": 50, "velocity": 10, "age": 365, "risk_score": 0.5}'
   ```

### Symptom: PSI always 0.0

- All prediction probabilities are identical (e.g., untrained model always
  outputs 0.5). Generate varied traffic or check the model weights.
- The telemetry window (`TELEMETRY_WINDOW_SIZE`) may be too small. Increase it.

---

## Tests Are Skipped

### Symptom: All Phase 3 tests skipped

```
SKIP: FastAPI not installed
SKIP: python-jose not installed
```

Install all test dependencies:

```bash
pip install -r requirements.txt
pip install fastapi[all] python-jose[cryptography] httpx
```

### Symptom: Phase 2 tests fail with `redis.exceptions.ConnectionError`

The Phase 2 tests require a live Redis instance. Start one:

```bash
docker run -d -p 6379:6379 redis:7-alpine
```

---

## Windows-Specific

### Symptom: `python` not found

Use the full path to your Python executable or activate a virtual environment:

```powershell
# Create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Run tests
python test_phase1_generality.py
```

### Symptom: `uvicorn` not found after pip install

Ensure the Scripts directory of your venv is on PATH:

```powershell
$env:PATH += ";$PWD\.venv\Scripts"
```

### Symptom: Line ending issues in shell scripts

If Docker build fails on Windows due to `\r\n` line endings in shell scripts,
convert them:

```powershell
# Using git
git config --global core.autocrlf input
git checkout -- .
```

Or use the `.gitattributes` file to enforce LF for shell scripts:

```
*.sh text eol=lf
Dockerfile* text eol=lf
```

---

## Prometheus Metrics Not Appearing

### Symptom: No metrics at `/metrics`

`prometheus_client` must be installed and the endpoint must be exposed. Add
to your FastAPI app:

```python
from prometheus_client import make_asgi_app
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

Or run a standalone metrics server from the worker:

```python
from prometheus_client import start_http_server
start_http_server(port=9090)
```

### Symptom: `prometheus_client not installed` warning in worker logs

```bash
pip install prometheus_client
```

All Prometheus calls are no-ops when the library is absent — this is safe for
dev/test environments but should be fixed before production.
