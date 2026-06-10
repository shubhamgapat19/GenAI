"""
Phase 7: Deployment — Scaling Strategies
==========================================
Scale LangChain apps to handle production traffic.

Topics covered:
- Horizontal vs vertical scaling
- Load balancing
- Worker processes (Gunicorn/Uvicorn)
- Queue-based architecture
- Database connection pooling
- Auto-scaling configuration
"""

# ============================================================
# 1. Scaling Dimensions
# ============================================================

print("=== Scaling Dimensions ===")
print("""
| Dimension | What It Scales | When |
|-----------|---------------|------|
| Vertical | CPU/RAM of single server | Quick fix, limited |
| Horizontal | Number of server instances | Production standard |
| Functional | Separate services for different functions | Complex apps |
| Data | Partition data across stores | Large datasets |

For LLM apps, the bottleneck is usually:
1. LLM API rate limits (not your server)
2. Concurrent connections
3. Memory (for conversation history/vector stores)
""")

# ============================================================
# 2. Worker Configuration (Uvicorn + Gunicorn)
# ============================================================

print("=== Worker Configuration ===")

GUNICORN_CONFIG = """
# ===== gunicorn.conf.py =====
import multiprocessing

# Workers = 2 * CPU cores + 1 (classic formula)
# For LLM apps (IO-bound), can go higher
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
bind = "0.0.0.0:8000"
keepalive = 120
timeout = 120
graceful_timeout = 30
max_requests = 1000          # Restart worker after N requests (prevent memory leaks)
max_requests_jitter = 50     # Random jitter to prevent all workers restarting at once
accesslog = "-"              # Log to stdout
errorlog = "-"
"""

print(GUNICORN_CONFIG)
print("""
# Run with Gunicorn:
gunicorn main:app -c gunicorn.conf.py

# Docker CMD:
CMD ["gunicorn", "main:app", "-c", "gunicorn.conf.py"]

# Workers for different scenarios:
#   1 CPU (dev):      2-3 workers
#   2 CPU (small):    5 workers
#   4 CPU (medium):   9 workers
#   8 CPU (large):    17 workers
""")

# ============================================================
# 3. Load Balancing
# ============================================================

NGINX_CONFIG = """
# ===== nginx.conf (reverse proxy + load balancer) =====
upstream langchain_api {
    # Round-robin by default
    server api-1:8000;
    server api-2:8000;
    server api-3:8000;
    
    # Health check
    # server api-1:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.example.com;
    
    # Timeouts for LLM (they can be slow)
    proxy_connect_timeout 10s;
    proxy_read_timeout 120s;    # LLM responses can take a while
    proxy_send_timeout 120s;
    
    # For SSE streaming
    proxy_buffering off;
    proxy_cache off;
    
    location / {
        proxy_pass http://langchain_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
    }
    
    location /health {
        proxy_pass http://langchain_api/health;
        access_log off;  # Don't log health checks
    }
}
"""

print("=== Nginx Load Balancer ===")
print(NGINX_CONFIG)

# ============================================================
# 4. Queue-Based Architecture (for heavy tasks)
# ============================================================

print("=== Queue Architecture ===")
print("""
For long-running tasks (agents, multi-step RAG):

[Client] → [API Server] → [Task Queue (Redis/RabbitMQ)] → [Workers]
    ↑                                                          |
    └──────────── [Result Store (Redis/DB)] ←──────────────────┘

Benefits:
- API responds immediately (async task)
- Workers can be scaled independently
- Retries built into queue
- No timeout issues for long agent runs
""")

# Celery worker example
CELERY_WORKER = '''
# ===== tasks.py (Celery worker) =====
from celery import Celery
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = Celery("langchain_tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/1")

# Configure task settings
app.conf.task_time_limit = 300          # Hard limit: 5 minutes
app.conf.task_soft_time_limit = 240     # Soft limit: 4 minutes (raises exception)
app.conf.task_acks_late = True          # Acknowledge after completion
app.conf.worker_prefetch_multiplier = 1 # One task at a time per worker

@app.task(bind=True, max_retries=3, default_retry_delay=30)
def process_query(self, question: str, session_id: str = None) -> dict:
    """Process a query asynchronously."""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = ChatPromptTemplate.from_messages([
            ("system", "Be helpful and concise."),
            ("human", "{question}"),
        ]) | llm | StrOutputParser()
        
        answer = chain.invoke({"question": question})
        
        return {"answer": answer, "session_id": session_id, "status": "completed"}
    
    except Exception as e:
        # Retry on transient errors
        self.retry(exc=e)
'''

# FastAPI + Celery integration
FASTAPI_CELERY = '''
# ===== main.py (API with async tasks) =====
from fastapi import FastAPI
from tasks import process_query

app = FastAPI()

@app.post("/chat/async")
async def async_chat(question: str):
    """Submit a task and return task_id immediately."""
    task = process_query.delay(question)
    return {"task_id": task.id, "status": "processing"}

@app.get("/chat/result/{task_id}")
async def get_result(task_id: str):
    """Poll for task result."""
    task = process_query.AsyncResult(task_id)
    if task.ready():
        return {"status": "completed", "result": task.result}
    return {"status": "processing"}
'''

print(CELERY_WORKER)
print(FASTAPI_CELERY)

# ============================================================
# 5. Connection Pooling
# ============================================================

print("=== Connection Pooling ===")
print("""
# Redis connection pool
import redis

pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    max_connections=20,
    decode_responses=True,
)
redis_client = redis.Redis(connection_pool=pool)

# PostgreSQL connection pool (with SQLAlchemy)
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    "postgresql://user:pass@host/db",
    poolclass=QueuePool,
    pool_size=10,           # Maintained connections
    max_overflow=20,        # Extra connections under load
    pool_timeout=30,        # Wait time for connection
    pool_recycle=3600,      # Recycle connections after 1 hour
)

# Why pooling matters:
# - Without: Each request opens/closes DB connection (slow)
# - With: Reuse existing connections (fast)
# - 100 concurrent requests → only needs ~20 connections
""")

# ============================================================
# 6. Auto-Scaling Configuration
# ============================================================

print("=== Auto-Scaling Strategies ===")
print("""
# Metric-based scaling:

| Metric | Scale Up When | Scale Down When | Good For |
|--------|-------------|----------------|----------|
| CPU | >70% | <30% | Compute-heavy (local models) |
| Memory | >80% | <40% | RAG with large indices |
| Request count | >100 req/s | <20 req/s | General API traffic |
| Queue depth | >50 pending | <5 pending | Async task workers |
| Response time | P95 > 5s | P95 < 1s | User experience |

# AWS Auto-Scaling Policy (example):
# - Min instances: 2 (always available)
# - Max instances: 10 (cost cap)
# - Scale up: CPU > 70% for 2 minutes
# - Scale down: CPU < 30% for 5 minutes
# - Cooldown: 3 minutes between scaling events

# Kubernetes HPA:
# apiVersion: autoscaling/v2
# kind: HorizontalPodAutoscaler
# spec:
#   minReplicas: 2
#   maxReplicas: 10
#   metrics:
#   - type: Resource
#     resource:
#       name: cpu
#       target:
#         type: Utilization
#         averageUtilization: 70
""")

# ============================================================
# 7. Scaling LLM-Specific Challenges
# ============================================================

print("=== LLM-Specific Scaling Challenges ===")
print("""
Challenge 1: LLM API Rate Limits
  Problem: OpenAI has rate limits (tokens/min, requests/min)
  Solution: Multiple API keys, request queuing, caching

Challenge 2: Long Response Times
  Problem: LLM calls take 1-10s (vs 50ms for normal APIs)
  Solution: Async processing, streaming, queue architecture

Challenge 3: Variable Cost Per Request
  Problem: Simple query = $0.001, complex agent = $0.05
  Solution: Request classification, budget per user, tiered pricing

Challenge 4: Stateful Conversations
  Problem: Chat history must be accessible across instances
  Solution: External state store (Redis/DB), not in-memory

Challenge 5: Large Context Windows
  Problem: RAG contexts can be 10K+ tokens
  Solution: Context compression, smart retrieval, caching
""")

# ============================================================
# 8. Architecture for Scale
# ============================================================

print("=== Production Architecture ===")
print("""
                    ┌─────────────────────────────────────────┐
                    │               CDN / WAF                   │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │          Load Balancer (ALB/Nginx)        │
                    └───┬────────────────┬───────────────┬────┘
                        │                │               │
                   ┌────┴───┐      ┌────┴───┐     ┌────┴───┐
                   │ API-1  │      │ API-2  │     │ API-3  │
                   └────┬───┘      └────┬───┘     └────┬───┘
                        │                │               │
              ┌─────────┴────────────────┴───────────────┴─────────┐
              │                                                      │
    ┌─────────┴──────┐    ┌──────────┐    ┌──────────┐    ┌────────┴───┐
    │  Redis Cache   │    │ Postgres │    │Task Queue│    │Vector Store│
    │  (sessions,    │    │ (users,  │    │ (Celery) │    │ (Pinecone/ │
    │   cache)       │    │  logs)   │    │          │    │  Chroma)   │
    └────────────────┘    └──────────┘    └────┬─────┘    └────────────┘
                                               │
                                    ┌──────────┴──────────┐
                                    │     Workers (N)      │
                                    │  (Agent execution)   │
                                    └─────────────────────┘
""")
