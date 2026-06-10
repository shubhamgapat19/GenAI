# Docker Containerization — Deep Dive Notes

## Why Docker for LLM Apps?

Docker solves "it works on my machine":
- **Consistent environment** — same Python version, same packages everywhere
- **Reproducible builds** — same Dockerfile → same image every time
- **Easy deployment** — push image, pull anywhere, run
- **Isolation** — no dependency conflicts with host system
- **Scaling** — run multiple containers behind a load balancer

---

## Dockerfile Anatomy

```dockerfile
FROM python:3.11-slim          # Base image (OS + Python)
WORKDIR /app                   # Working directory inside container
COPY requirements.txt .        # Copy deps list first (cache layer)
RUN pip install -r requirements.txt  # Install deps (cached if unchanged)
COPY . .                       # Copy application code
EXPOSE 8000                    # Document the port
CMD ["uvicorn", "main:app"]    # Default command to run
```

### Layer Caching (Key Performance Trick)

Docker caches each layer. If a layer hasn't changed, it's reused:

```
Layer 1: FROM python:3.11-slim         ← cached (rarely changes)
Layer 2: COPY requirements.txt         ← cached (deps rarely change)
Layer 3: RUN pip install               ← cached (if requirements unchanged)
Layer 4: COPY . .                      ← rebuilt (code changes often)
```

**Rule:** Copy things that change LEAST first, things that change MOST last.

---

## Multi-Stage Builds

Smaller, more secure images:

```dockerfile
# Stage 1: Install dependencies (has build tools)
FROM python:3.11-slim AS builder
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# Stage 2: Production (clean, no build tools)
FROM python:3.11-slim
COPY --from=builder /install /usr/local
COPY app/ ./app/
CMD ["uvicorn", "main:app"]
```

| Build Type | Image Size | Security |
|-----------|-----------|---------|
| Single stage | ~1.2 GB | Has build tools (attack surface) |
| Multi-stage | ~400 MB | Clean, minimal |

---

## Docker Compose (Full Stack)

Run your API + Redis + Postgres together:

```yaml
version: '3.8'
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      redis:
        condition: service_healthy

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
```

### Key Compose Features
- `depends_on` + `condition`: Start order with health checks
- `volumes`: Persist data between restarts
- `environment`: Pass secrets from .env file
- `deploy.resources.limits`: Cap memory/CPU per service
- `restart: unless-stopped`: Auto-restart on crash

---

## Security Best Practices

### 1. Non-Root User
```dockerfile
RUN useradd --create-home appuser
USER appuser
```
If the container is compromised, attacker has limited permissions.

### 2. Never Bake Secrets
```dockerfile
# BAD — secret in image!
ENV OPENAI_API_KEY=sk-secret

# GOOD — pass at runtime
# docker run -e OPENAI_API_KEY=sk-secret my-image
```

### 3. Pin Base Image Version
```dockerfile
# BAD — could change anytime
FROM python:latest

# GOOD — reproducible
FROM python:3.11-slim
```

### 4. Minimal Base Image
```
python:3.11        → 900 MB (full Debian)
python:3.11-slim   → 120 MB (minimal Debian)
python:3.11-alpine → 50 MB  (Alpine Linux, may have compat issues)
```

---

## Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

| Setting | Value | Meaning |
|---------|-------|---------|
| interval | 30s | Check every 30 seconds |
| timeout | 10s | Fail if no response in 10s |
| retries | 3 | Mark unhealthy after 3 failures |
| start_period | 10s | Grace period on startup |

---

## .dockerignore

Exclude files from the build context:

```
.git
.env
__pycache__
.venv
node_modules
*.md
```

Benefits:
- Faster builds (less data to copy)
- Smaller images (no junk)
- Security (no .env in image)

---

## Common Docker Commands

```bash
# Build
docker build -t langchain-api .
docker build -t langchain-api:v1.2 .

# Run
docker run -d -p 8000:8000 --env-file .env langchain-api
docker run -it langchain-api bash  # Debug shell

# Compose
docker-compose up -d           # Start
docker-compose up -d --build   # Rebuild + start
docker-compose logs -f api     # Follow logs
docker-compose down            # Stop

# Debug
docker exec -it <id> bash      # Shell into running container
docker logs <id> --tail 50     # Last 50 lines
docker stats                   # Resource usage
```

---

## Best Practices

1. **Layer caching** — requirements.txt FIRST, code LAST
2. **Multi-stage builds** — smaller, more secure images
3. **Non-root user** — never run as root in production
4. **No secrets in image** — env vars or secrets manager
5. **Pin versions** — base image AND package versions
6. **Health checks** — required for orchestrators
7. **.dockerignore** — exclude .env, .git, __pycache__
8. **Resource limits** — prevent runaway containers

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| `FROM python:latest` | Non-reproducible | Pin version: `python:3.11-slim` |
| No .dockerignore | Slow builds, large images | Exclude .git, venv, .env |
| Secrets in Dockerfile | Security breach | Use runtime env vars |
| Running as root | Container compromise = host compromise | `USER appuser` |
| No health check | Orchestrator can't detect failures | Add HEALTHCHECK |
| Copy code before deps | Cache invalidated every change | Copy requirements first |
| No resource limits | Container eats all RAM | Set memory/CPU limits |
| Single stage | Large image with build tools | Multi-stage build |
