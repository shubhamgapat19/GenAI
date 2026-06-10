"""
Phase 7: Deployment — Cloud Deployment
=========================================
Deploy LangChain apps to cloud providers.

Topics covered:
- AWS deployment (ECS, Lambda, EC2)
- Google Cloud deployment (Cloud Run, GCE)
- Azure deployment (Container Apps, App Service)
- Infrastructure as Code (Terraform basics)
- Managed LangChain services (LangServe)
"""

# ============================================================
# 1. Deployment Options Overview
# ============================================================

print("=== Cloud Deployment Options ===")
print("""
| Platform | Service | Best For | Cold Start | Scale |
|----------|---------|----------|------------|-------|
| AWS | ECS Fargate | Containers, auto-scale | No | Horizontal |
| AWS | Lambda | Low traffic, event-driven | Yes (5-15s) | Auto |
| AWS | EC2 | Full control, GPU | No | Manual/ASG |
| GCP | Cloud Run | Containers, pay-per-use | Yes (1-5s) | Auto |
| GCP | GCE | Full control, GPU | No | Manual/MIG |
| Azure | Container Apps | Containers, serverless | Yes (1-3s) | Auto |
| Azure | App Service | PaaS, easy deploy | No | Manual/Auto |
| Railway | - | Quick deploy, hobbyist | No | Limited |
| Render | - | Simple containers | Yes | Auto |
""")

# ============================================================
# 2. AWS ECS Fargate Deployment
# ============================================================

ECS_TASK_DEFINITION = """
{
  "family": "langchain-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "ACCOUNT.dkr.ecr.REGION.amazonaws.com/langchain-api:latest",
      "portMappings": [
        {"containerPort": 8000, "protocol": "tcp"}
      ],
      "environment": [
        {"name": "APP_ENV", "value": "production"}
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT:secret:openai-key"
        }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/langchain-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
"""

print("=== AWS ECS Task Definition ===")
print(ECS_TASK_DEFINITION)

# ============================================================
# 3. AWS Lambda (Serverless)
# ============================================================

LAMBDA_HANDLER = '''
# ===== lambda_handler.py =====
import json
from mangum import Mangum
from main import app  # Your FastAPI app

# Mangum adapts FastAPI to Lambda
handler = Mangum(app, lifespan="off")

# Alternative: direct Lambda handler (no FastAPI)
def simple_handler(event, context):
    """Direct Lambda handler for simple use cases."""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    body = json.loads(event.get("body", "{}"))
    question = body.get("question", "")
    
    if not question:
        return {"statusCode": 400, "body": json.dumps({"error": "No question"})}
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = ChatPromptTemplate.from_messages([
        ("system", "Be concise."),
        ("human", "{question}"),
    ]) | llm | StrOutputParser()
    
    answer = chain.invoke({"question": question})
    
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"answer": answer}),
    }
'''

print("=== AWS Lambda Handler ===")
print(LAMBDA_HANDLER)

# ============================================================
# 4. Google Cloud Run
# ============================================================

CLOUD_RUN_DEPLOY = """
# ===== Deploy to Cloud Run =====

# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/langchain-api

# Deploy
gcloud run deploy langchain-api \\
  --image gcr.io/PROJECT_ID/langchain-api \\
  --platform managed \\
  --region us-central1 \\
  --port 8000 \\
  --memory 1Gi \\
  --cpu 1 \\
  --min-instances 1 \\
  --max-instances 10 \\
  --timeout 60 \\
  --set-secrets OPENAI_API_KEY=openai-key:latest \\
  --allow-unauthenticated

# Or with a service YAML:
# gcloud run services replace service.yaml
"""

CLOUD_RUN_YAML = """
# ===== service.yaml (Cloud Run) =====
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: langchain-api
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "10"
    spec:
      containerConcurrency: 80
      timeoutSeconds: 60
      containers:
        - image: gcr.io/PROJECT_ID/langchain-api
          ports:
            - containerPort: 8000
          resources:
            limits:
              memory: 1Gi
              cpu: "1"
          env:
            - name: APP_ENV
              value: production
          startupProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
"""

print("=== Google Cloud Run ===")
print(CLOUD_RUN_DEPLOY)
print(CLOUD_RUN_YAML)

# ============================================================
# 5. Azure Container Apps
# ============================================================

AZURE_DEPLOY = """
# ===== Deploy to Azure Container Apps =====

# Create resource group
az group create --name langchain-rg --location eastus

# Create container app environment
az containerapp env create \\
  --name langchain-env \\
  --resource-group langchain-rg \\
  --location eastus

# Deploy container app
az containerapp create \\
  --name langchain-api \\
  --resource-group langchain-rg \\
  --environment langchain-env \\
  --image your-registry.azurecr.io/langchain-api:latest \\
  --target-port 8000 \\
  --ingress external \\
  --min-replicas 1 \\
  --max-replicas 10 \\
  --cpu 1.0 \\
  --memory 2.0Gi \\
  --secrets openai-key=YOUR_KEY \\
  --env-vars OPENAI_API_KEY=secretref:openai-key
"""

print("=== Azure Container Apps ===")
print(AZURE_DEPLOY)

# ============================================================
# 6. Terraform (Infrastructure as Code)
# ============================================================

TERRAFORM_ECS = """
# ===== main.tf (AWS ECS with Terraform) =====
terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" { region = "us-east-1" }

# ECR Repository
resource "aws_ecr_repository" "api" {
  name = "langchain-api"
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "langchain-cluster"
}

# ECS Service
resource "aws_ecs_service" "api" {
  name            = "langchain-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [aws_security_group.api.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }
}

# Auto-scaling
resource "aws_appautoscaling_target" "api" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.api.resource_id
  scalable_dimension = aws_appautoscaling_target.api.scalable_dimension
  service_namespace  = aws_appautoscaling_target.api.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70.0
  }
}
"""

print("=== Terraform (IaC) ===")
print(TERRAFORM_ECS)

# ============================================================
# 7. Quick Deploy Options (for MVPs)
# ============================================================

print("=== Quick Deploy Options ===")
print("""
# Railway (simplest, git push to deploy)
# 1. Connect GitHub repo
# 2. Set environment variables
# 3. Done! Auto-deploys on push.

# Render
# 1. Create Web Service
# 2. Connect repo
# 3. Set: Build Command = pip install -r requirements.txt
#    Start Command = uvicorn main:app --host 0.0.0.0 --port $PORT

# Fly.io
fly launch                          # Create app
fly secrets set OPENAI_API_KEY=sk-xxx  # Set secrets
fly deploy                          # Deploy

# Heroku (with Procfile)
# Procfile: web: uvicorn main:app --host 0.0.0.0 --port $PORT
heroku create langchain-api
heroku config:set OPENAI_API_KEY=sk-xxx
git push heroku main
""")

# ============================================================
# 8. Choosing a Deployment Platform
# ============================================================

print("=== Decision Matrix ===")
print("""
| Factor | ECS/Cloud Run | Lambda/Serverless | EC2/GCE | Railway/Render |
|--------|--------------|-------------------|---------|----------------|
| Setup effort | Medium | Medium | High | Low |
| Cost (low traffic) | $20-50/mo | $0-5/mo | $20+/mo | $5-20/mo |
| Cost (high traffic) | $100-500/mo | Variable | Fixed | $50-200/mo |
| Auto-scaling | Yes | Yes | Manual/ASG | Limited |
| Cold starts | No | Yes (5-15s) | No | Sometimes |
| Streaming SSE | Yes | Limited | Yes | Yes |
| Long-running agents | Yes | 15min limit | Yes | Yes |
| GPU support | Limited | No | Yes | No |
| Best for | Production | Low/bursty traffic | ML/GPU | MVPs/startups |
""")
