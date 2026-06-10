# Cloud Deployment — Deep Dive Notes

## Deployment Platform Decision

Choose based on your needs:

| If You Need | Choose | Why |
|-------------|--------|-----|
| Quick MVP / demo | Railway, Render | Git push deploy, minimal config |
| Auto-scaling containers | AWS ECS, GCP Cloud Run | Production-grade, pay-per-use |
| Serverless (low traffic) | AWS Lambda | $0 when idle, auto-scales |
| Full control + GPU | AWS EC2, GCP GCE | Custom setup, ML models |
| Microsoft ecosystem | Azure Container Apps | Integrates with Azure services |

---

## Option 1: AWS ECS Fargate (Recommended for Production)

```
[Internet] → [ALB] → [ECS Service] → [Fargate Tasks (containers)]
                          ↓
                    [Auto-scaling: 2-10 tasks]
```

### Pros
- No servers to manage (serverless containers)
- Auto-scales based on CPU/memory/requests
- Integrates with ALB, CloudWatch, Secrets Manager
- Pay only for running containers

### Cons
- More setup than Railway/Render
- AWS-specific knowledge needed
- Can be expensive at high scale without optimization

### Deployment Steps
1. Push Docker image to ECR (Elastic Container Registry)
2. Create ECS Task Definition (what to run)
3. Create ECS Service (how many, where)
4. Configure ALB (load balancer)
5. Set up auto-scaling policies

---

## Option 2: Google Cloud Run (Simplest Production)

```bash
# Build and deploy in two commands:
gcloud builds submit --tag gcr.io/PROJECT/langchain-api
gcloud run deploy langchain-api --image gcr.io/PROJECT/langchain-api
```

### Pros
- Simplest cloud deployment
- Scales to zero (pay nothing when idle)
- Auto-scales up to thousands of instances
- Built-in HTTPS
- No Kubernetes knowledge needed

### Cons
- Cold starts (1-5s after scale-to-zero)
- 60-minute max request timeout
- Limited to HTTP (no WebSocket native)

### Best For
- APIs with variable traffic
- Cost-sensitive deployments
- Teams without DevOps expertise

---

## Option 3: AWS Lambda (Serverless)

```python
# Use Mangum to adapt FastAPI to Lambda
from mangum import Mangum
from main import app

handler = Mangum(app, lifespan="off")
```

### Pros
- $0 when no traffic
- Auto-scales to thousands of concurrent requests
- No infrastructure to manage

### Cons
- Cold starts (5-15s for Python)
- 15-minute max execution time
- No streaming/SSE support (API Gateway limitation)
- Memory limited (10 GB max)

### Best For
- Low/bursty traffic
- Event-driven (webhook processing)
- Cost optimization for infrequent use

### NOT Good For
- Chat streaming (SSE not supported)
- Long-running agents (15min limit)
- High-traffic consistent load (containers cheaper)

---

## Option 4: Quick Deploy (MVP/Startup)

### Railway
```bash
# 1. Connect GitHub repo on railway.app
# 2. Set env vars in dashboard
# 3. Every push auto-deploys
# Done! URL: your-app.railway.app
```

### Render
```bash
# Create render.yaml
services:
  - type: web
    name: langchain-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Fly.io
```bash
fly launch
fly secrets set OPENAI_API_KEY=sk-xxx
fly deploy
# Done! URL: your-app.fly.dev
```

---

## Secrets Management

**Never hardcode API keys.** Use the platform's secrets manager:

| Platform | Secrets Solution |
|----------|-----------------|
| AWS | Secrets Manager, Parameter Store |
| GCP | Secret Manager |
| Azure | Key Vault |
| Railway | Environment Variables (encrypted) |
| Docker | Docker Secrets, .env file |

### AWS Secrets Manager Example
```json
// Task definition references secret
{
  "secrets": [
    {
      "name": "OPENAI_API_KEY",
      "valueFrom": "arn:aws:secretsmanager:us-east-1:123:secret:openai-key"
    }
  ]
}
```

---

## Infrastructure as Code (Terraform)

Define infrastructure in code → reproducible, version-controlled:

```hcl
resource "aws_ecs_service" "api" {
  name            = "langchain-api"
  cluster         = aws_ecs_cluster.main.id
  desired_count   = 2
  launch_type     = "FARGATE"
}

resource "aws_appautoscaling_policy" "cpu" {
  target_tracking_scaling_policy_configuration {
    target_value = 70.0  # Scale at 70% CPU
  }
}
```

### Benefits of IaC
- **Reproducible:** Same infra in dev/staging/prod
- **Version controlled:** Git history of all changes
- **Reviewable:** PR review for infrastructure changes
- **Automatable:** CI/CD can apply infra changes

---

## Cost Comparison (rough estimates)

| Platform | Low Traffic (1K req/day) | Medium (10K/day) | High (100K/day) |
|----------|------------------------|-------------------|-----------------|
| Lambda | $1-5/mo | $10-30/mo | $100-300/mo |
| Cloud Run | $5-15/mo | $30-80/mo | $200-500/mo |
| ECS Fargate | $30-60/mo | $60-150/mo | $200-600/mo |
| Railway | $5-20/mo | $20-50/mo | $50-200/mo |
| EC2 (t3.medium) | $30/mo flat | $30/mo flat | Need bigger/more |

**Note:** These are infrastructure costs only. LLM API costs are separate and usually dominate.

---

## Best Practices

1. **Start simple** — Railway/Render for MVP, migrate to ECS/Cloud Run later
2. **Use secrets manager** — never put API keys in code or Dockerfiles
3. **Enable auto-scaling** — handle traffic spikes without over-provisioning
4. **Set min instances > 0** — avoid cold starts for production
5. **Use health checks** — let the platform know when your app is broken
6. **Infrastructure as Code** — Terraform for anything beyond hobby projects
7. **Multi-region** — for global apps (reduces latency)
8. **Monitor costs** — set billing alerts on all platforms

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Hardcoded secrets | Security breach | Use secrets manager |
| No auto-scaling | Can't handle traffic spikes | Configure scaling policies |
| Lambda for streaming | SSE doesn't work | Use ECS/Cloud Run for chat |
| Scale-to-zero in prod | Cold starts frustrate users | Set min instances = 1+ |
| No health checks | Platform can't detect failures | Add /health endpoint |
| Over-provisioning | Paying for idle resources | Use auto-scaling |
| Single region | High latency for distant users | Multi-region or CDN |
| No IaC | Can't reproduce/recover infra | Terraform from day 1 |
