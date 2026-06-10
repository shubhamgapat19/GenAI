# Observability & Tracing — Deep Dive Notes

## Why Observability Matters

In production, you can't `print()` debug. You need:
- **Tracing:** What happened inside the chain/agent? Step-by-step visibility.
- **Monitoring:** Is it working? How fast? How expensive?
- **Debugging:** Why did it give a bad answer? What went wrong?
- **Optimization:** Where are the bottlenecks? What's costing the most?

---

## LangSmith: The Observability Platform

LangSmith is LangChain's built-in tracing & evaluation platform.

### Setup (Zero Code Change!)
```bash
# Set environment variables
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_api_key
export LANGCHAIN_PROJECT=my_project
```

Once set, **every LangChain call is automatically traced** — no code changes needed.

---

## What Gets Traced

Every trace captures:

| Field | Example |
|-------|---------|
| **Input** | The prompt/messages sent to LLM |
| **Output** | The response received |
| **Latency** | 1.2 seconds |
| **Token count** | 150 input + 80 output = 230 total |
| **Cost** | $0.000345 |
| **Model** | gpt-4o-mini |
| **Status** | success / error |
| **Parent/child** | Which chain/node produced this call |

### Nested Traces (parent → child)
```
[RAG Pipeline] (root)
  ├── [Retriever] → 3 documents found
  ├── [Prompt Template] → formatted prompt
  └── [LLM Call] → generated answer
       ├── Input: 450 tokens
       ├── Output: 120 tokens
       └── Latency: 0.8s
```

---

## Adding Metadata & Tags

```python
response = chain.invoke(
    {"question": "..."},
    config={
        "metadata": {
            "user_id": "user_123",
            "session_id": "sess_abc",
            "environment": "production",
            "app_version": "2.1.0",
        },
        "tags": ["production", "chat", "premium_user"],
    }
)
```

### Use Cases for Metadata
- **user_id:** Filter traces by user (debugging user-specific issues)
- **session_id:** Group all traces in a conversation
- **environment:** Separate dev/staging/production
- **app_version:** Track behavior changes across releases

### Use Cases for Tags
- Filter by feature (chat, search, summarize)
- A/B test variants ("prompt_v1", "prompt_v2")
- Priority levels ("free_tier", "premium")

---

## Named Runs

Give chains descriptive names for clarity:
```python
chain = (prompt | llm | parser).with_config(run_name="CustomerSupportChain")
```

In LangSmith dashboard, you'll see "CustomerSupportChain" instead of "RunnableSequence".

---

## @traceable Decorator

For custom Python functions (not just LangChain chains):

```python
from langsmith import traceable

@traceable(name="my_rag_pipeline", tags=["rag"])
def my_pipeline(question: str) -> str:
    # All LLM calls inside are automatically child spans
    context = retriever.invoke(question)
    answer = chain.invoke({"context": context, "question": question})
    return answer
```

Everything inside `@traceable` becomes a traced span with children.

---

## Custom Callbacks

For logging to your own systems (not just LangSmith):

```python
class ProductionLogger(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        # Log to Datadog, CloudWatch, etc.
        
    def on_llm_end(self, response, **kwargs):
        # Track tokens, latency
        
    def on_llm_error(self, error, **kwargs):
        # Alert on-call team
```

### Common Callback Integrations
- **Datadog:** APM traces, custom metrics
- **CloudWatch:** AWS logging and monitoring
- **Prometheus:** Metrics for Grafana dashboards
- **Sentry:** Error tracking and alerting

---

## Cost Tracking

```python
from langchain_community.callbacks import get_openai_callback

with get_openai_callback() as cb:
    response = chain.invoke(...)

print(f"Cost: ${cb.total_cost:.6f}")
print(f"Tokens: {cb.total_tokens}")
```

### Cost Budget Alerts
- Set daily budget per project ($10/day)
- Alert at 80% usage
- Hard stop at 100%
- Track per-user costs for abuse detection

---

## Feedback Collection

After every response, collect user feedback:

```python
from langsmith import Client
client = Client()

# Thumbs up/down
client.create_feedback(run_id=run_id, key="user_rating", score=1.0)

# Detailed feedback
client.create_feedback(
    run_id=run_id,
    key="correctness",
    score=0.0,
    comment="Answer was wrong about the release date",
)
```

### Feedback Loop
```
User gives thumbs down → flag trace → human reviews → 
  add to eval dataset → improve prompt → deploy → measure improvement
```

---

## Production Monitoring Checklist

| Metric | Alert Threshold | Action |
|--------|----------------|--------|
| Error rate | >5% | Page on-call |
| P95 latency | >10s | Investigate bottleneck |
| Cost/day | >$50 | Check for abuse/loops |
| Feedback score | <70% avg | Review recent traces |
| Token usage spike | >3x normal | Check for prompt injection |

---

## Best Practices

1. **Enable tracing from day 1** — free and zero-effort
2. **Tag by environment** — never mix dev/prod traces
3. **Add user_id to all calls** — essential for debugging user issues
4. **Name your chains** — "RunnableSequence" is not debuggable
5. **Track costs daily** — set alerts before you get a surprise bill
6. **Collect feedback** — builds your evaluation dataset for free
7. **Use callbacks for custom logging** — LangSmith + your observability stack
8. **Review low-scoring traces** — find patterns in failures
