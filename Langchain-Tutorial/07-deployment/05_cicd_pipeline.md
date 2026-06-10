# CI/CD for LLM Applications — Deep Dive Notes

## Why CI/CD for LLM Apps Is Different

Traditional CI/CD: Run tests → build → deploy. Deterministic, fast.

LLM CI/CD adds:
- **Evaluation gates** — non-deterministic outputs need LLM-as-judge
- **Cost awareness** — running eval suite costs money (LLM calls)
- **Prompt versioning** — prompts are code, need version control
- **Canary deploys** — LLM behavior changes are hard to predict
- **Monitoring after deploy** — need to watch quality metrics

---

## Pipeline Stages

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐
│  Lint   │ → │  Unit    │ → │  LLM     │ → │  Build  │ → │  Deploy  │
│  Type   │    │  Tests   │    │  Eval    │    │  Docker │    │  Cloud   │
│  Check  │    │  (fast)  │    │  (slow)  │    │  Image  │    │          │
└─────────┘    └──────────┘    └──────────┘    └─────────┘    └──────────┘
    ~30s           ~60s           ~5min           ~2min        ~5min
                                $0.50-2            0              0
```

---

## Stage 1: Linting & Type Checking

```yaml
- name: Lint
  run: ruff check .

- name: Type check
  run: mypy src/ --ignore-missing-imports
```

Fast, free, catches obvious errors.

---

## Stage 2: Unit Tests

Test everything EXCEPT LLM calls:

```python
# tests/unit/test_validation.py
def test_input_validation():
    with pytest.raises(ValueError):
        ChatInput(question="")  # Too short

def test_pii_redaction():
    redactor = PIIRedactor()
    assert "[REDACTED]" in redactor.redact("email: test@example.com")

def test_rate_limiter():
    limiter = RateLimiter(max=2, window=60)
    assert limiter.is_allowed("user1") == True
    assert limiter.is_allowed("user1") == True
    assert limiter.is_allowed("user1") == False  # Exceeded
```

### What to Unit Test
- Input validation
- Output sanitization
- Rate limiting logic
- Cache logic
- PII detection
- Prompt template rendering (without LLM)
- Tool argument parsing

### What NOT to Unit Test
- Actual LLM responses (use eval suite)
- External API integrations (use integration tests)

---

## Stage 3: LLM Evaluation

The unique stage for LLM apps:

```python
# scripts/run_evaluation.py
def run_eval(dataset_path, threshold=0.8):
    dataset = load_json(dataset_path)
    scores = []
    
    for case in dataset:
        response = chain.invoke(case["input"])
        score = llm_judge(case["input"], response, case["expected"])
        scores.append(score)
    
    avg = sum(scores) / len(scores)
    
    if avg < threshold:
        sys.exit(1)  # FAIL the CI build
```

### Evaluation Tiers

| Tier | Cases | Cost | When to Run |
|------|-------|------|-------------|
| Smoke test | 5-10 | $0.05 | Every PR |
| Standard | 30-50 | $0.50 | Every merge to main |
| Full suite | 100-200 | $2-5 | Nightly / before prod deploy |

### Posting Results to PR
```yaml
- name: Post eval results
  uses: actions/github-script@v7
  with:
    script: |
      const results = require('./eval_results.json');
      github.rest.issues.createComment({
        body: `## Eval: ${results.score >= 0.8 ? '✅' : '❌'} ${results.score}`
      });
```

---

## Stage 4: Build Docker Image

```yaml
- name: Build and push
  uses: docker/build-push-action@v5
  with:
    push: true
    tags: registry/app:${{ github.sha }}
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

### Image Tagging Strategy
```
registry/app:latest         → always points to newest
registry/app:abc123def     → commit SHA (immutable)
registry/app:v1.2.3        → semantic version (releases)
registry/app:staging       → environment tag
```

**Rule:** Deploy by commit SHA, not `latest`. Always know exactly what's running.

---

## Stage 5: Deploy

### Environment Promotion
```
PR merged → Deploy to staging (auto)
Staging healthy for 30 min → Deploy to production (manual approval)
```

### GitHub Environments
```yaml
deploy-production:
  environment: production  # Requires manual approval in GitHub settings
  steps:
    - name: Deploy
      run: aws ecs update-service ...
```

---

## Prompt Versioning

Prompts change more often than code. Version them:

### Approach 1: YAML Files in Git
```yaml
# prompts/chat_v3.yaml
version: 3
created: 2024-06-15
author: dev_team
changelog: "Added citation requirement"
messages:
  - role: system
    content: "Be concise. Cite sources."
```

### Approach 2: Feature Flags
```python
if flags.get("prompt_version") == "v3":
    chain = prompt_v3 | llm | parser
else:
    chain = prompt_v2 | llm | parser  # Default/safe
```

### Approach 3: LangSmith Hub
```python
from langsmith import hub
prompt = hub.pull("my-org/chat-assistant:v3")
```

### When to Version Prompts
- Any system prompt change
- Adding/removing tools from an agent
- Changing output format requirements
- Adjusting temperature or model

---

## Rollback Strategies

### Strategy 1: Blue-Green (Safest)
```
Blue (current) ←── ALL traffic
Green (new)    ←── NO traffic

Test green → switch traffic → if broken → switch back instantly
```

### Strategy 2: Canary (Gradual)
```
Current ←── 95% traffic
New     ←── 5% traffic

Monitor 5% → if good → 25% → 50% → 100%
If bad at any stage → roll back to 0%
```

### Strategy 3: Instant Rollback
```bash
# Deploy previous known-good image
aws ecs update-service --task-definition app:42  # Previous version
```

### Automated Rollback Triggers
- Error rate > 5% for 5 minutes
- P95 latency > 15s for 5 minutes
- Health check fails for 1 minute
- Eval score drops > 10% (if running live eval)

---

## Post-Deploy Monitoring

After every deploy, watch for 30 minutes:

```
[ ] Error rate staying < 1%
[ ] P95 latency staying < 5s
[ ] LLM cost per request normal
[ ] No spike in negative user feedback
[ ] Health checks all passing
[ ] No new error types in logs
```

### Automated Monitoring
```yaml
# CloudWatch alarm
MetricName: 5XXError
Threshold: 5
Period: 300  # 5 minutes
Action: SNS → PagerDuty → On-call engineer
```

---

## Complete Workflow Example

```
1. Developer changes prompt (prompts/chat_v4.yaml)
2. Opens PR → triggers CI:
   a. Lint ✅
   b. Unit tests ✅  
   c. Smoke eval (10 cases) → 92% ✅ (threshold: 80%)
   d. Posts results to PR comment
3. Code review + approval
4. Merge to main → triggers CD:
   a. Full eval (50 cases) → 88% ✅
   b. Build Docker image → push to registry
   c. Deploy to staging → health check passes
5. After 30 min in staging (no issues):
   a. Manual approval in GitHub
   b. Deploy to production (canary: 10% → 50% → 100%)
   c. Monitor for 30 min
   d. If issues → auto-rollback to previous version
```

---

## Best Practices

1. **Eval gate on every PR** — never merge without quality check
2. **Smoke tests are cheap** — run 5-10 cases on every PR (~$0.05)
3. **Full eval before prod** — 50+ cases before production deploy
4. **Deploy by commit SHA** — always know exactly what's running
5. **Canary deploys** — gradual rollout catches issues early
6. **Auto-rollback** — don't wait for humans to notice failures
7. **Version prompts** — treat prompts like code (Git, PRs, reviews)
8. **Post-deploy monitoring** — watch for 30 min after every deploy

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| No eval in CI | Deploying regressions | Add eval gate (smoke test minimum) |
| Deploy `latest` tag | Don't know what's running | Deploy by commit SHA |
| No staging environment | Testing in production | Add staging with auto-deploy |
| Manual deploys | Inconsistent, error-prone | Automate with GitHub Actions |
| No rollback plan | Stuck with broken deploy | Pre-configure rollback triggers |
| Full eval on every PR | Slow + expensive | Smoke test on PR, full eval on merge |
| No prompt versioning | Can't roll back prompt changes | Version prompts in Git |
| No post-deploy monitoring | Issues discovered by users | Watch metrics for 30 min |
