# Scaling Strategies — Deep Dive Notes

## The Scaling Challenge for LLM Apps

LLM apps have unique scaling characteristics:

| Property | Traditional API | LLM API |
|----------|----------------|---------|
| Response time | 50-200ms | 1-10s |
| Cost per request | ~$0 | $0.001-$0.05 |
| CPU during request | High | Low (waiting for LLM) |
| Memory | Low | Medium-High (contexts) |
| Bottleneck | Your server | LLM provider rate limits |

**Key insight:** Your server is usually NOT the bottleneck — the LLM API is. Scaling your servers alone won't help if you're hitting OpenAI rate limits.

---

## Horizontal vs Vertical Scaling

### Vertical (Scale Up)
```
Small server → Big server
1 CPU, 1 GB → 8 CPU, 16 GB
```
- Quick fix, limited ceiling
- For: Local models, vector stores in memory

### Horizontal (Scale Out)
```
1 server → 5 servers (behind load balancer)
```
- Production standard, unlimited ceiling
- For: API serving, any stateless service

### For LLM Apps
- **Horizontal** for the API layer (stateless, easy to scale)
- **Vertical** for local vector stores (need more RAM for larger indices)
- **Neither** fixes LLM rate limits (need multiple API keys or provider)

---

## Worker Configuration

### How Workers Work
```
[Gunicorn (master)]
  ├── [Uvicorn Worker 1] → handles requests
  ├── [Uvicorn Worker 2] → handles requests
  ├── [Uvicorn Worker 3] → handles requests
  └── [Uvicorn Worker 4] → handles requests
```

### Formula
```python
# Classic formula (CPU-bound):
workers = 2 * cpu_cores + 1

# For LLM apps (IO-bound, waiting for API):
workers = 4 * cpu_cores + 1  # Can go higher since mostly waiting
```

### Configuration
```python
# gunicorn.conf.py
workers = 9                          # For 4-core machine
worker_class = "uvicorn.workers.UvicornWorker"  # Async support
timeout = 120                        # LLM calls can be slow
max_requests = 1000                  # Restart after N requests (prevent memory leaks)
max_requests_jitter = 50             # Don't restart all at once
```

---

## Load Balancing

### Algorithms

| Algorithm | How It Works | Best For |
|-----------|-------------|----------|
| Round Robin | Each server in turn | Equal-capability servers |
| Least Connections | Send to least busy | Varying request duration |
| IP Hash | Same user → same server | Session affinity |
| Weighted | Some servers get more | Mixed server sizes |

### For LLM Apps: Least Connections
LLM requests have highly variable duration (0.5s - 30s). Round Robin can overload one server with all the slow requests.

### Nginx Configuration
```nginx
upstream api {
    least_conn;
    server api-1:8000;
    server api-2:8000;
    server api-3:8000;
}
```

### Important for Streaming
```nginx
# Required for SSE/streaming to work through load balancer:
proxy_buffering off;
proxy_cache off;
proxy_read_timeout 120s;
```

---

## Queue-Based Architecture

For long-running tasks (agents, complex RAG):

```
[API Server] → [Queue (Redis)] → [Workers]
     ↑                                |
     └────── [Result Store] ←─────────┘
```

### When to Use Queues
- Agent runs that take > 30s
- Batch processing jobs
- Tasks with unpredictable duration
- When you need guaranteed completion

### How It Works
1. API receives request → puts task in queue → returns task_id immediately
2. Worker picks up task → processes → stores result
3. Client polls for result (or receives webhook)

### Implementation: Celery + Redis
```python
# API side:
task = process_query.delay(question)  # Returns immediately
return {"task_id": task.id}

# Worker side:
@app.task
def process_query(question):
    answer = agent.invoke(question)  # Takes as long as it needs
    return {"answer": answer}

# Client polls:
GET /result/{task_id} → {"status": "completed", "result": {...}}
```

---

## Connection Pooling

### Problem Without Pooling
```
Request 1: Open connection → query → close connection  (50ms overhead)
Request 2: Open connection → query → close connection  (50ms overhead)
Request 3: Open connection → query → close connection  (50ms overhead)
```

### With Pooling
```
Request 1: Get connection from pool → query → return to pool  (1ms overhead)
Request 2: Get connection from pool → query → return to pool  (1ms overhead)
```

### Pool Configuration
```python
# Redis
pool = redis.ConnectionPool(max_connections=20)

# PostgreSQL
engine = create_engine(url,
    pool_size=10,         # Maintained connections
    max_overflow=20,      # Extra under load
    pool_timeout=30,      # Wait time for connection
)
```

### Sizing Guidelines
- Pool size ≈ number of workers × 2
- Max overflow ≈ pool size × 2
- For 4 workers: pool_size=10, max_overflow=20

---

## Auto-Scaling

### Metrics to Scale On

| Metric | Scale Up | Scale Down | Notes |
|--------|----------|------------|-------|
| CPU > 70% | Add instance | CPU < 30% | Standard, but LLM apps are IO-bound |
| Memory > 80% | Add instance | Memory < 40% | For vector store in memory |
| Request count | > 100/s per instance | < 20/s | Better for LLM apps |
| Queue depth | > 50 pending | < 5 pending | For worker scaling |
| Response time P95 | > 5s | < 1s | User experience-based |

### Scaling Configuration
```
Min instances: 2         (always available, no cold start)
Max instances: 10        (cost cap)
Scale up cooldown: 2 min (prevent thrashing)
Scale down cooldown: 5 min (don't kill instances too fast)
```

### Kubernetes HPA
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70
```

---

## Stateless vs Stateful

### Make Your API Stateless
```
STATEFUL (bad for scaling):
- Conversation history in server memory
- Vector store loaded in RAM per instance
- Session data in local files

STATELESS (good for scaling):
- History in Redis/PostgreSQL
- Vector store in Pinecone/managed service
- Sessions in external store
```

**Rule:** Any instance should be able to handle any request. No request should depend on hitting the same instance.

---

## LLM-Specific Scaling Patterns

### Pattern 1: Request Classification
```python
# Route simple queries to fast path, complex to queue
if is_simple(question):
    return fast_chain.invoke(question)  # Sync, < 3s
else:
    task = agent_task.delay(question)   # Async, may take 30s+
    return {"task_id": task.id}
```

### Pattern 2: Provider Load Balancing
```python
# Spread across multiple API keys / providers
providers = [
    ChatOpenAI(api_key=key_1),
    ChatOpenAI(api_key=key_2),
    ChatAnthropic(api_key=anthropic_key),
]
# Round-robin or least-loaded selection
```

### Pattern 3: Tiered Caching
```
L1: In-memory (per instance, fastest)
L2: Redis (shared across instances, fast)
L3: Database (permanent, slowest)
```

---

## Best Practices

1. **Stateless API** — external state stores for everything
2. **Queue for long tasks** — don't hold HTTP connections for minutes
3. **Least-connections LB** — handles variable response times
4. **Pool connections** — to Redis, Postgres, any external service
5. **Scale on requests, not CPU** — LLM apps are IO-bound
6. **Min 2 instances** — for availability + zero-downtime deploys
7. **Set max instances** — cost cap prevents surprise bills
8. **Classify requests** — route simple/complex differently

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| State in memory | Can't scale horizontally | External state store |
| Scale on CPU | LLM apps are IO-bound, CPU stays low | Scale on request count |
| No max instances | Cost explosion during traffic spike | Set max and alert |
| Buffering proxy | Streaming breaks | `proxy_buffering off` |
| No connection pool | Connection overhead per request | Pool all external connections |
| Same path for all requests | Simple queries wait behind complex ones | Request classification |
| No queue for agents | HTTP timeout on long runs | Async task queue |
| No cooldown period | Scaling thrashing (up-down-up-down) | 2-5 min cooldown |
