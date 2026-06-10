"""
Phase 7: Deployment — CI/CD for LLM Applications
===================================================
Automate testing, building, and deploying LLM apps.

Topics covered:
- GitHub Actions workflows
- Pre-deployment evaluation
- Docker image building and pushing
- Environment promotion (dev → staging → prod)
- Rollback strategies
- Prompt versioning
"""

# ============================================================
# 1. CI/CD Pipeline Overview
# ============================================================

print("=== CI/CD Pipeline for LLM Apps ===")
print("""
┌──────┐    ┌──────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐
│ Code │ → │ Test │ → │  Eval   │ → │  Build  │ → │  Deploy  │
│ Push │    │      │    │(LLM-as- │    │(Docker) │    │(Cloud)   │
│      │    │      │    │ judge)  │    │         │    │          │
└──────┘    └──────┘    └─────────┘    └─────────┘    └──────────┘
   PR          Unit       Quality        Container      Staging →
             tests +      gate           image          Production
             lint         (>80%)
""")

# ============================================================
# 2. GitHub Actions Workflow
# ============================================================

GITHUB_ACTIONS = """
# ===== .github/workflows/deploy.yml =====
name: Deploy LangChain App

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # Job 1: Run tests
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt
      
      - name: Run unit tests
        run: pytest tests/unit/ -v
      
      - name: Run linting
        run: ruff check .

  # Job 2: LLM Evaluation (only on PR)
  evaluate:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    needs: test
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run LLM evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          LANGCHAIN_API_KEY: ${{ secrets.LANGCHAIN_API_KEY }}
        run: |
          python scripts/run_evaluation.py --dataset eval/smoke_test.json --threshold 0.8
      
      - name: Post evaluation results
        if: always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('eval_results.json'));
            const body = `## Evaluation Results
            - Score: ${results.avg_score}
            - Pass rate: ${results.pass_rate}%
            - ${results.pass_rate >= 80 ? '✅ PASSED' : '❌ FAILED'}`;
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });

  # Job 3: Build Docker image
  build:
    runs-on: ubuntu-latest
    needs: [test]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      
      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # Job 4: Deploy to staging
  deploy-staging:
    runs-on: ubuntu-latest
    needs: build
    environment: staging
    steps:
      - name: Deploy to staging
        run: |
          # AWS ECS example
          aws ecs update-service \\
            --cluster langchain-staging \\
            --service api \\
            --force-new-deployment
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_REGION: us-east-1
      
      - name: Wait for deployment
        run: |
          aws ecs wait services-stable \\
            --cluster langchain-staging \\
            --services api
      
      - name: Smoke test staging
        run: |
          curl -f https://staging-api.example.com/health
          curl -X POST https://staging-api.example.com/chat \\
            -H "Content-Type: application/json" \\
            -d '{"question": "ping"}' \\
            | jq '.answer'

  # Job 5: Deploy to production (manual approval)
  deploy-production:
    runs-on: ubuntu-latest
    needs: deploy-staging
    environment: production  # Requires manual approval in GitHub
    steps:
      - name: Deploy to production
        run: |
          aws ecs update-service \\
            --cluster langchain-production \\
            --service api \\
            --force-new-deployment
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_REGION: us-east-1
"""

print("=== GitHub Actions Workflow ===")
print(GITHUB_ACTIONS)

# ============================================================
# 3. Evaluation Script (used in CI)
# ============================================================

EVAL_SCRIPT = '''
# ===== scripts/run_evaluation.py =====
"""Run LLM evaluation as part of CI/CD."""
import json
import argparse
import sys
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def run_eval(dataset_path: str, threshold: float):
    """Run evaluation and exit with error if below threshold."""
    
    # Load test cases
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    # Initialize chain under test
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = ChatPromptTemplate.from_messages([
        ("system", "Answer based on context. Say I don't know if not in context.\\nContext: {context}"),
        ("human", "{question}"),
    ]) | llm | StrOutputParser()
    
    # Run evaluation
    scores = []
    for case in dataset:
        response = chain.invoke({"context": case["context"], "question": case["question"]})
        # Simple keyword check (replace with LLM-as-judge in production)
        score = 1.0 if any(kw.lower() in response.lower() for kw in case["expected_keywords"]) else 0.0
        scores.append(score)
    
    avg_score = sum(scores) / len(scores)
    pass_rate = sum(1 for s in scores if s >= 0.7) / len(scores) * 100
    
    results = {"avg_score": round(avg_score, 3), "pass_rate": round(pass_rate, 1), "total": len(scores)}
    
    # Write results for GitHub Action to read
    with open("eval_results.json", "w") as f:
        json.dump(results, f)
    
    print(f"Evaluation: avg={avg_score:.3f}, pass_rate={pass_rate:.1f}%")
    
    if avg_score < threshold:
        print(f"FAILED: Score {avg_score:.3f} below threshold {threshold}")
        sys.exit(1)
    else:
        print(f"PASSED: Score {avg_score:.3f} above threshold {threshold}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()
    run_eval(args.dataset, args.threshold)
'''

print("=== Evaluation Script ===")
print(EVAL_SCRIPT)

# ============================================================
# 4. Prompt Versioning
# ============================================================

print("=== Prompt Versioning ===")
print("""
# prompts/chat_v1.yaml
version: 1
name: chat_assistant
model: gpt-4o-mini
temperature: 0.7
messages:
  - role: system
    content: "You are a helpful assistant."
  - role: human
    content: "{question}"

# prompts/chat_v2.yaml (new version)
version: 2
name: chat_assistant
model: gpt-4o-mini
temperature: 0.5
messages:
  - role: system
    content: "You are a helpful assistant. Be concise. Cite sources when possible."
  - role: human
    content: "{question}"
""")

PROMPT_LOADER = '''
# ===== prompt_manager.py =====
import yaml
from pathlib import Path

class PromptManager:
    """Load and manage versioned prompts."""
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.dir = Path(prompts_dir)
    
    def load(self, name: str, version: int = None) -> dict:
        """Load a prompt by name and optional version."""
        if version:
            path = self.dir / f"{name}_v{version}.yaml"
        else:
            # Find latest version
            versions = sorted(self.dir.glob(f"{name}_v*.yaml"))
            path = versions[-1] if versions else self.dir / f"{name}.yaml"
        
        with open(path) as f:
            return yaml.safe_load(f)
    
    def get_chain(self, name: str, version: int = None):
        """Build a chain from prompt config."""
        config = self.load(name, version)
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        llm = ChatOpenAI(model=config["model"], temperature=config["temperature"])
        prompt = ChatPromptTemplate.from_messages(
            [(m["role"], m["content"]) for m in config["messages"]]
        )
        return prompt | llm | StrOutputParser()
'''

print(PROMPT_LOADER)

# ============================================================
# 5. Rollback Strategies
# ============================================================

print("=== Rollback Strategies ===")
print("""
Strategy 1: Blue-Green Deployment
  [Blue (current)] ←── Load Balancer
  [Green (new)]
  
  Deploy to Green → test → switch LB to Green
  If problems → switch LB back to Blue (instant rollback)

Strategy 2: Canary Deployment
  [Current v1] ←── 90% traffic
  [New v2]     ←── 10% traffic
  
  Monitor v2 metrics. If good → gradually shift to 100%
  If bad → route 100% back to v1

Strategy 3: Image Tag Rollback (Docker)
  # Deploy specific version
  docker pull registry/app:abc123    # Known good commit
  
  # AWS ECS rollback
  aws ecs update-service --task-definition app:PREVIOUS_VERSION

Strategy 4: Feature Flags
  # Use feature flags for prompt changes
  if feature_flags.is_enabled("new_prompt_v2", user_id):
      chain = prompt_v2 | llm | parser
  else:
      chain = prompt_v1 | llm | parser
""")

# ============================================================
# 6. Environment Promotion
# ============================================================

print("=== Environment Promotion ===")
print("""
┌─────────┐     ┌───────────┐     ┌──────────────┐
│   Dev   │ → │  Staging  │ → │  Production  │
│         │     │           │     │              │
│ - Local │     │ - Cloud   │     │ - Cloud      │
│ - Fake  │     │ - Real    │     │ - Real       │
│   data  │     │   data    │     │   data       │
│ - No    │     │ - Full    │     │ - Full       │
│   eval  │     │   eval    │     │   monitoring │
└─────────┘     └───────────┘     └──────────────┘
  PR merge       Auto-deploy       Manual approval
                 + smoke test      + full eval

Config differences:
| Setting | Dev | Staging | Production |
|---------|-----|---------|------------|
| Model | gpt-4o-mini | gpt-4o-mini | gpt-4o-mini |
| Rate limit | None | 100/min | 30/min/user |
| Tracing | Verbose | Full | Sampled (10%) |
| Eval threshold | None | 75% | 85% |
| Instances | 1 | 2 | 2-10 (auto) |
""")

# ============================================================
# 7. Monitoring After Deploy
# ============================================================

print("=== Post-Deploy Monitoring ===")
print("""
After every deploy, watch these for 30 minutes:

1. Error rate → should stay < 1%
2. P95 latency → should stay < 5s
3. LLM costs → should stay within 20% of baseline
4. User feedback → no spike in negative ratings

Automated rollback triggers:
- Error rate > 5% for 5 minutes → auto-rollback
- P95 latency > 15s for 5 minutes → auto-rollback
- No successful health check for 1 minute → auto-rollback

# AWS CloudWatch alarm for auto-rollback:
# Metric: TargetResponseTime > 15
# Period: 5 minutes
# Action: Roll back to previous task definition
""")
