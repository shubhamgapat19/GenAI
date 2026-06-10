"""
Phase 7: Deployment — Docker Containerization
================================================
Package LangChain applications into Docker containers.

Topics covered:
- Dockerfile best practices
- Multi-stage builds
- Docker Compose for development
- Environment variable management
- Container health checks
- Volume mounts for persistence
"""

# ============================================================
# 1. Basic Dockerfile
# ============================================================

BASIC_DOCKERFILE = """
# ===== Dockerfile =====
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system deps (for some LangChain packages)
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

print("=== Basic Dockerfile ===")
print(BASIC_DOCKERFILE)

# ============================================================
# 2. Multi-Stage Build (smaller image)
# ============================================================

MULTISTAGE_DOCKERFILE = """
# ===== Multi-Stage Dockerfile =====

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Production image
FROM python:3.11-slim AS production

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy only application code (no build tools)
COPY app/ ./app/
COPY main.py .

# Non-root user (security)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
"""

print("=== Multi-Stage Dockerfile ===")
print(MULTISTAGE_DOCKERFILE)

# ============================================================
# 3. Docker Compose (full stack)
# ============================================================

DOCKER_COMPOSE = """
# ===== docker-compose.yml =====
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@postgres:5432/langchain
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '1.0'

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: langchain
    ports:
      - "5432:5432"
    volumes:
      - pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 10s
      timeout: 5s
      retries: 3

volumes:
  redis_data:
  pg_data:
"""

print("=== Docker Compose ===")
print(DOCKER_COMPOSE)

# ============================================================
# 4. Requirements.txt for Docker
# ============================================================

REQUIREMENTS = """
# ===== requirements.txt =====
# Core
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-community>=0.3.0

# API Framework
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
python-dotenv>=1.0.0

# Production
redis>=5.0.0
psycopg2-binary>=2.9.0
httpx>=0.24.0

# Monitoring
langsmith>=0.1.0
"""

print("=== Requirements ===")
print(REQUIREMENTS)

# ============================================================
# 5. .dockerignore
# ============================================================

DOCKERIGNORE = """
# ===== .dockerignore =====
.git
.gitignore
.env
*.pyc
__pycache__
.pytest_cache
.venv
venv
node_modules
*.md
*.txt
!requirements.txt
.mypy_cache
.coverage
htmlcov
"""

print("=== .dockerignore ===")
print(DOCKERIGNORE)

# ============================================================
# 6. Environment Management
# ============================================================

ENV_EXAMPLE = """
# ===== .env.example =====
# LLM Provider
OPENAI_API_KEY=sk-your-key-here

# Observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls-your-key-here
LANGCHAIN_PROJECT=production

# Infrastructure
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost:5432/langchain

# App Config
APP_ENV=production
APP_PORT=8000
LOG_LEVEL=info
RATE_LIMIT_PER_MINUTE=30
MAX_TOKENS_PER_REQUEST=4000
"""

print("=== Environment Variables ===")
print(ENV_EXAMPLE)

# ============================================================
# 7. Docker Commands Cheatsheet
# ============================================================

print("=== Docker Commands ===")
print("""
# Build
docker build -t langchain-api .
docker build -t langchain-api:v1.0 --target production .

# Run
docker run -d -p 8000:8000 --env-file .env langchain-api
docker run -d -p 8000:8000 -e OPENAI_API_KEY=sk-xxx langchain-api

# Docker Compose
docker-compose up -d                    # Start all services
docker-compose up -d --build            # Rebuild and start
docker-compose logs -f api              # Follow API logs
docker-compose down                     # Stop all
docker-compose down -v                  # Stop and remove volumes

# Debugging
docker exec -it <container_id> bash     # Shell into container
docker logs <container_id> --tail 100   # Last 100 log lines
docker stats                            # Resource usage

# Cleanup
docker system prune -a                  # Remove unused images/containers
docker volume prune                     # Remove unused volumes
""")

# ============================================================
# 8. Production Dockerfile Tips
# ============================================================

print("=== Production Tips ===")
print("""
1. Layer caching: COPY requirements.txt FIRST, then code
   → Dependencies only rebuild when requirements change

2. Non-root user: Always run as non-root in production
   → RUN useradd appuser && USER appuser

3. .dockerignore: Exclude .env, .git, venv, __pycache__
   → Smaller build context, faster builds

4. Health checks: Add HEALTHCHECK in Dockerfile
   → Load balancers need to know if container is healthy

5. Multi-stage: Build deps in stage 1, copy to clean stage 2
   → Smaller final image (no build tools)

6. Pin versions: python:3.11-slim, not python:latest
   → Reproducible builds

7. No secrets in image: Use env vars or secrets management
   → NEVER bake API keys into the image

8. Resource limits: Set memory/CPU limits in compose
   → Prevent runaway containers
""")
