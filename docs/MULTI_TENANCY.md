# Multi-Tenancy

## Overview

The platform enforces strict tenant isolation across every infrastructure layer:
compute, cache, queuing, authentication, and rate limiting. The design follows
a **logical isolation** model (shared infrastructure with per-tenant namespacing)
that can be upgraded to **physical isolation** (dedicated pods per tenant) for
high-value enterprise accounts.

---

## Layer 1: Request Identity — `X-Tenant-ID` Header

Every HTTP request to the inference and admin services **must** carry an
`X-Tenant-ID` header. This header is the primary tenant identifier propagated
through the entire request lifecycle.

```
GET /predict HTTP/1.1
X-Tenant-ID: client-a
X-Model-ID: fraudnet-v1
Authorization: Bearer <JWT>
```

The JWT `Authorization` header contains a `tenant_id` claim that must match
`X-Tenant-ID`. If they differ, the request is rejected with HTTP 403.

---

## Layer 2: Envoy Proxy — Dynamic Routing

Envoy sits at the edge and routes requests to tenant-specific upstream clusters
**before** they reach any application code.

**How it works:**

1. A Lua filter reads `X-Tenant-ID` from the request headers.
2. The filter injects a new `x-target-cluster: tenant-cluster-{tenant_id}` header.
3. Envoy's `cluster_header` routing directive selects the correct upstream.
4. The injected header is stripped before forwarding to the service.

```lua
function envoy_on_request(request_handle)
  local tenant_id = request_handle:headers():get("X-Tenant-ID")
  if not tenant_id then
    request_handle:respond({[":status"] = "400"}, "Missing X-Tenant-ID header")
    return
  end
  request_handle:headers():replace("x-target-cluster", "tenant-cluster-" .. tenant_id)
end
```

This means routing decisions are made entirely in the data plane — no
control-plane restart required to add a new tenant cluster.

---

## Layer 3: Redis — Hash-Tag Key Namespacing

All Redis keys are prefixed with a **hash tag** to guarantee per-tenant
isolation in both standalone and clustered Redis:

```
{client-a}:fraudnet-v1:telemetry_queue
{client-b}:fraudnet-v1:telemetry_queue
{client-a}:fraudnet-v1:metrics:psi_probability
```

### Why hash tags?

In a Redis Cluster, keys are distributed across nodes by hashing the entire
key. Multi-key operations (e.g., `LRANGE`, `MGET`, atomic pipelines) fail with
`CROSSSLOT` errors if the keys hash to different cluster slots.

Hash tags (`{...}`) instruct Redis to hash **only the bracketed portion**,
guaranteeing all keys for a single tenant land on the same slot:

```
hash_slot("{client-a}") = slot X
{client-a}:model-1:telemetry  → slot X  ✓
{client-a}:model-2:metrics    → slot X  ✓
{client-b}:model-1:telemetry  → slot Y  ✓ (isolated)
```

### TenantRedisClient

`inference/tenant_redis_client.py` wraps a standard Redis client and
automatically applies the hash-tag prefix to every key operation:

```python
client = TenantRedisClient(redis_url="redis://localhost:6379/0", tenant_id="client-a")
client.push_telemetry("fraudnet-v1", {"probability": 0.92, ...})
# writes to key: {client-a}:fraudnet-v1:telemetry_queue
```

---

## Layer 4: RabbitMQ — Virtual Host Isolation

When using RabbitMQ as the Celery broker (instead of Redis), each tenant gets
a dedicated **Virtual Host** (vhost). Vhosts are completely isolated
namespaces within RabbitMQ — separate exchanges, queues, bindings, and user
permissions:

```python
connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host=rmq_host,
        virtual_host=f"/{tenant_id}",   # isolated per tenant
        credentials=credentials,
        heartbeat=60,
    )
)
```

Benefits:
- A misconfigured consumer in tenant A's vhost cannot accidentally read
  tenant B's retraining jobs.
- Per-vhost message rate limits can be configured in the RabbitMQ admin UI.
- Vhosts can be provisioned automatically by the Admin API on tenant registration.

---

## Layer 5: Tenant Model Registry

The `TenantModelRegistry` maps `(tenant_id, model_id, model_version)` triples
to model storage paths, config files, and drift thresholds.

```
tenant_id   model_id      model_version   storage_path
---------   -----------   -------------   ----------------------------
client-a    fraudnet-v1   1.0.0           s3://bucket/client-a/v1.pt
client-b    churn-rf      2.1.0           s3://bucket/client-b/churn.pkl
```

**Backends:**
- **In-memory** (default, dev/test): process-local dict; lost on restart.
- **PostgreSQL** (production): set `DATABASE_URL` to activate.
  The schema includes a foreign key from `tenant_models` → `tenants` with
  `ON DELETE CASCADE`, ensuring model records are removed when a tenant is
  deleted.

---

## Layer 6: JWT Authentication — Per-Tenant Tokens

Tokens are issued by `POST /auth/token?tenant_id=<id>` and signed with
HMAC-HS256. Each token encodes the `tenant_id` as a claim:

```json
{
  "tenant_id": "client-a",
  "exp": 1720000000
}
```

On every protected endpoint:
1. The `Authorization: Bearer <token>` header is decoded and verified.
2. The `tenant_id` claim is extracted.
3. It is compared to the `X-Tenant-ID` header — they **must** match.
4. Mismatch → HTTP 403 Forbidden.

This prevents a compromised token for `client-a` from being used to access
`client-b`'s models or retraining jobs.

> **Production note**: Replace the default `SECRET_KEY` with a strong random
> value (≥ 32 bytes) from your secrets manager. Consider rotating tokens
> by integrating with your corporate identity provider (OAuth2 / OIDC).

---

## Layer 7: Rate Limiting — Token Bucket Per Tenant

The Admin API applies a per-tenant token bucket rate limiter to all
authenticated endpoints:

```
Default: 60 tokens capacity, 1 token/second refill rate
```

When a tenant exhausts their bucket, subsequent requests receive HTTP 429:

```json
{
  "detail": "Tenant rate limit exceeded"
}
```

Custom limits can be set via `RateLimiter.set_limits(tenant_id, capacity, rate)`.

---

## Hybrid Tier Model (Recommended)

| Tier | Isolation Model | Cost | Use Case |
|------|----------------|------|----------|
| **Standard** | Logical (shared Redis, shared pods) | Low | SMBs, trial customers |
| **Enterprise** | Physical (dedicated namespace, pod-per-tenant) | High | Finance, healthcare, defence |

The Admin API's `tier` field on tenant registration drives provisioning logic.
A future Kubernetes Operator can watch for `tier: enterprise` tenants and
automatically provision dedicated namespaces and Helm releases.

---

## Data Isolation Guarantees

| What | Guarantee | How |
|------|-----------|-----|
| Telemetry | No cross-tenant reads/writes | Redis hash-tag keyspace isolation |
| Model weights | Separate storage paths | Tenant Model Registry path per `(tenant, model, version)` |
| Drift detection | Per-tenant threshold config | `drift_thresholds` JSONB in registry |
| API access | No cross-tenant token reuse | JWT `tenant_id` claim == `X-Tenant-ID` enforced |
| Request routing | Traffic goes to correct cluster | Envoy Lua filter + `cluster_header` routing |
| Queue jobs | Isolated vhosts or key namespaces | RabbitMQ vhosts / Redis key prefix |
