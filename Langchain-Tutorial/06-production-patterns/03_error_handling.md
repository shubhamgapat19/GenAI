# Error Handling & Resilience — Deep Dive Notes

## Why LLM Apps Fail

LLM applications have unique failure modes:

| Failure | Frequency | Impact |
|---------|-----------|--------|
| Rate limit (429) | Common at scale | Requests dropped |
| Timeout | Common with large prompts | Hung requests |
| API outage | Rare but impactful | Total failure |
| Bad output (hallucination) | Frequent | Wrong answers |
| Content filter | Occasional | Blocked responses |
| Token limit exceeded | Occasional | Truncated/failed |

Traditional retry logic isn't enough — you need **layered resilience**.

---

## Layer 1: Built-in Retries

```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    max_retries=3,           # Auto-retry on 429, 500, 503
    request_timeout=30,       # Timeout per request
)
```

The LangChain client auto-retries on transient errors (429, 500, 503) with exponential backoff.

---

## Layer 2: Custom Retry with .with_retry()

```python
chain = prompt | llm | parser

robust_chain = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)
```

### Backoff Strategy
```
Attempt 1: immediate
Attempt 2: wait ~2s (1-3s with jitter)
Attempt 3: wait ~4s (3-5s with jitter)
→ fail after 3 attempts
```

**Jitter is critical** — without it, all retries hit at the same time (thundering herd).

---

## Layer 3: Fallback Chains

```python
primary = prompt | gpt4o_mini | parser
fallback = prompt | gpt4o | parser

resilient = primary.with_fallbacks([fallback])
```

### Fallback Strategies

| Strategy | When to Use |
|----------|-------------|
| Same provider, bigger model | Primary is small/fast, fallback is large/capable |
| Different provider | Provider outage (OpenAI → Anthropic) |
| Cached response | When freshness isn't critical |
| Static response | When any answer is better than no answer |

### Multi-level Fallback
```python
chain = primary.with_fallbacks([
    fallback_1,    # Try GPT-4o
    fallback_2,    # Try Anthropic
    static_response,  # Return canned response
])
```

---

## Layer 4: Circuit Breaker

Prevents cascading failures when a service is down:

```
CLOSED (normal) → failures pile up → OPEN (fast-fail)
    ↑                                      |
    └── HALF_OPEN (test one request) ←─────┘ (after cooldown)
```

### States

| State | Behavior | Transition |
|-------|----------|------------|
| **CLOSED** | Normal operation, count failures | → OPEN after N failures |
| **OPEN** | Reject immediately (fast fail) | → HALF_OPEN after cooldown |
| **HALF_OPEN** | Allow ONE test request | → CLOSED (success) or OPEN (fail) |

### Why Circuit Breaker Matters
- Without it: 100 users × 30s timeout = 100 hung requests
- With it: 100 users × instant rejection = immediate feedback + no server overload

---

## Layer 5: Input Validation

**Never trust user input going to an LLM:**

```python
class ChatInput(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    
    @field_validator("question")
    def no_injection(cls, v):
        # Block prompt injection patterns
        ...
    
    @field_validator("question")
    def reasonable_length(cls, v):
        if len(v.split()) > 1000:
            raise ValueError("Too long")
        return v
```

### What to Validate
- **Length:** Prevent token bombs (10K word inputs)
- **Injection patterns:** Block "ignore previous instructions"
- **Encoding:** Ensure valid UTF-8
- **Content type:** Reject binary/code if expecting natural language

---

## Layer 6: Timeout Management

| Component | Recommended Timeout |
|-----------|-------------------|
| Chat response | 10-15s |
| RAG pipeline | 20-30s |
| Agent loop (per iteration) | 15s |
| Agent total | 60-120s |
| Embedding call | 10s |
| Vector search | 5s |

```python
import asyncio

async def with_timeout(coroutine, timeout, default="Service unavailable"):
    try:
        return await asyncio.wait_for(coroutine, timeout=timeout)
    except asyncio.TimeoutError:
        return default
```

---

## Layer 7: Graceful Degradation

When things fail, degrade quality instead of crashing:

```
Level 1: Full RAG (high quality)     ← Primary
Level 2: Direct LLM (medium quality) ← No retrieval
Level 3: Cached response (low quality) ← No LLM call
Level 4: Static message (no quality)  ← Always works
```

```python
def graceful_answer(question, context=None):
    # Try each level, fall through on failure
    try:
        return rag_chain.invoke(...)      # Level 1
    except:
        pass
    try:
        return direct_chain.invoke(...)    # Level 2
    except:
        pass
    try:
        return cache.get(question)         # Level 3
    except:
        return "I'm temporarily unavailable."  # Level 4
```

---

## Error Response Design

Don't expose internal errors to users:

| Internal Error | User-Facing Message |
|---------------|-------------------|
| `openai.RateLimitError` | "High demand right now. Please try again in a moment." |
| `asyncio.TimeoutError` | "Taking longer than expected. Please retry." |
| `ValidationError` | "Could you rephrase your question?" |
| `ContentFilterError` | "I can't help with that topic." |
| `500 InternalError` | "Something went wrong. We're looking into it." |

**Never expose:** Stack traces, API keys, internal model names, error codes.

---

## Monitoring Errors

```python
# Track error types and frequencies
error_metrics = {
    "rate_limit": Counter(),    # Alert if > 10/min
    "timeout": Counter(),       # Alert if > 5/min
    "bad_output": Counter(),    # Alert if > 20% of responses
    "total_errors": Counter(),  # Alert if > 5% error rate
}
```

### Alert Rules
- Error rate > 5% → investigate
- Error rate > 20% → page on-call
- Same error > 10x in 1 min → likely outage, activate circuit breaker

---

## Best Practices

1. **Layer your resilience** — retries, fallbacks, circuit breaker, graceful degradation
2. **Never expose internal errors** — user-friendly messages always
3. **Add jitter to retries** — prevent thundering herd
4. **Set timeouts everywhere** — no request should hang forever
5. **Validate inputs** — before spending money on LLM calls
6. **Monitor error patterns** — spikes indicate systemic issues
7. **Test failure modes** — inject failures in staging
8. **Fallback to different providers** — don't depend on one API

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| No retry logic | Single failure = user error | Add .with_retry() |
| Retry without backoff | Hammers failing service | Exponential + jitter |
| No fallback | One provider down = app down | with_fallbacks() |
| Exposing errors | Security risk + bad UX | Friendly error messages |
| No timeout | Requests hang forever | Set timeout on every call |
| Retrying 400 errors | Wastes money (won't succeed) | Only retry transient (429, 5xx) |
| No circuit breaker | Cascading failures | Fast-fail when service is down |
