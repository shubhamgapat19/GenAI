# Retry, Fallback & Error Handling — Deep Dive Notes

## Why Error Handling Matters for LLM Apps

LLM applications are **inherently unreliable** compared to traditional software:
- API rate limits hit randomly
- Models return malformed output
- Timeouts on complex prompts
- Service outages (OpenAI, Azure, etc.)
- Non-deterministic responses break parsers

You MUST design for failure. A production LLM app without error handling will crash constantly.

---

## The Three Lines of Defense

```
Request
  ↓
[Retry]      ← Same call, try again (transient errors)
  ↓ (still failing)
[Fallback]   ← Different model/chain (if primary is down)
  ↓ (still failing)
[Graceful Degradation] ← Return cached/default response
```

---

## 1. with_retry — Transient Error Recovery

Retries the **same operation** multiple times. Good for:
- Rate limit errors (429)
- Temporary network issues
- Server errors (500, 503)

```python
chain = (prompt | llm | parser).with_retry(
    stop_after_attempt=3,           # Max 3 tries total
    wait_exponential_jitter=True,    # Exponential backoff + random jitter
)
```

### How Exponential Backoff Works
```
Attempt 1: immediate
Attempt 2: wait ~1 second (+ random jitter)
Attempt 3: wait ~2 seconds (+ random jitter)
Attempt 4: wait ~4 seconds (+ random jitter)
```

Jitter prevents "thundering herd" — if many clients retry at exactly the same intervals, they all hit the server simultaneously again.

### Selective Retry (retry only specific errors)
```python
from openai import RateLimitError, APITimeoutError

chain.with_retry(
    retry_if_exception_type=(RateLimitError, APITimeoutError),
    stop_after_attempt=3,
)
# Won't retry on AuthenticationError, ValidationError, etc.
```

### When to Use Retry
| Error Type | Retry? | Why |
|-----------|--------|-----|
| Rate limit (429) | ✅ Yes | Temporary, resolves with backoff |
| Timeout | ✅ Yes | Might work on next attempt |
| Server error (500) | ✅ Yes | Usually transient |
| Auth error (401) | ❌ No | Your key is wrong, retrying won't help |
| Invalid request (400) | ❌ No | Your input is wrong |
| Context length exceeded | ❌ No | Need to reduce input |

---

## 2. with_fallbacks — Alternative Model/Chain

When the primary chain fails, try a backup:

### Model-level fallback
```python
# Try GPT-4o → fall back to GPT-4o-mini
robust_llm = ChatOpenAI(model="gpt-4o").with_fallbacks([
    ChatOpenAI(model="gpt-4o-mini"),
])
```

### Chain-level fallback
```python
# Complex extraction → simple extraction → raw text
robust_chain = complex_extraction.with_fallbacks([
    simple_extraction,
    raw_text_chain,
])
```

### Multi-provider fallback
```python
# OpenAI → Anthropic → Google (provider redundancy)
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

robust_llm = ChatOpenAI(model="gpt-4o").with_fallbacks([
    ChatAnthropic(model="claude-sonnet-4-20250514"),
    ChatGoogleGenerativeAI(model="gemini-1.5-pro"),
])
```

### Fallback with exception filtering
```python
from openai import RateLimitError, APIConnectionError

# Only fallback on availability issues, not logic errors
chain.with_fallbacks(
    [fallback_chain],
    exceptions_to_handle=(RateLimitError, APIConnectionError, TimeoutError),
)
```

---

## 3. Graceful Degradation — When Everything Fails

When retries and fallbacks are exhausted:

```python
def graceful_chain(input_dict):
    """Last resort: return a helpful default."""
    try:
        return main_chain.invoke(input_dict)
    except Exception as e:
        return {
            "response": "I'm experiencing issues. Please try again shortly.",
            "error": str(e),
            "cached": get_cached_response(input_dict),  # Return last good response
        }
```

---

## Timeout Strategies

### Model-level timeout
```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    timeout=30,        # HTTP request timeout (seconds)
    max_retries=2,     # Built-in retry for HTTP errors
)
```

### Per-invocation timeout
```python
import asyncio

async def invoke_with_timeout(chain, input_dict, timeout=10):
    try:
        return await asyncio.wait_for(
            chain.ainvoke(input_dict),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return {"error": "Request timed out", "timeout": timeout}
```

### Timeout guidelines
| Operation | Suggested Timeout |
|-----------|------------------|
| Simple completion | 10-15s |
| Complex reasoning | 30-60s |
| Long document processing | 60-120s |
| Batch processing (per item) | 30s |

---

## Output Parsing Error Recovery

LLMs often return malformed output. Here's how to handle it:

### Strategy 1: Auto-fix with OutputFixingParser
```python
from langchain.output_parsers import OutputFixingParser

# Wraps your parser — sends errors back to LLM for correction
fixing_parser = OutputFixingParser.from_llm(
    parser=pydantic_parser,
    llm=llm,
)
# Cost: 1 extra LLM call when parsing fails
```

### Strategy 2: Regex extraction fallback
```python
import re
import json

def robust_json_parse(text: str) -> dict:
    # Try 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try 2: Extract from code blocks
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try 3: Find any JSON-like structure
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Give up gracefully
    return {"raw": text, "parse_failed": True}
```

### Strategy 3: .with_structured_output() (best)
```python
# API-level enforcement — JSON schema is guaranteed
structured_llm = llm.with_structured_output(MyModel)
# Almost never fails for well-defined schemas
```

---

## Production Error Handling Pattern

```python
import logging
from langchain_core.runnables import RunnableLambda

logger = logging.getLogger(__name__)

def build_production_chain():
    """Full production chain with all safety layers."""
    
    # Layer 1: Primary model with timeout
    primary = ChatOpenAI(model="gpt-4o", timeout=30, max_retries=2)
    
    # Layer 2: Fallback model
    fallback = ChatOpenAI(model="gpt-4o-mini", timeout=15, max_retries=2)
    
    # Layer 3: Model with fallback
    robust_model = primary.with_fallbacks([fallback])
    
    # Layer 4: Full chain with retry
    chain = (
        prompt
        | robust_model
        | StrOutputParser()
    ).with_retry(stop_after_attempt=2)
    
    # Layer 5: Graceful degradation wrapper
    def safe_invoke(input_dict):
        try:
            return {"response": chain.invoke(input_dict), "status": "success"}
        except Exception as e:
            logger.error(f"All retries/fallbacks failed: {e}")
            return {"response": "Service temporarily unavailable.", "status": "error"}
    
    return RunnableLambda(safe_invoke)
```

---

## Rate Limiting Strategies

### Client-side rate limiting
```python
import time
from threading import Semaphore

# Max 10 requests per minute
rate_limiter = Semaphore(10)

def rate_limited_invoke(chain, input_dict):
    rate_limiter.acquire()
    try:
        return chain.invoke(input_dict)
    finally:
        # Release after delay
        time.sleep(6)  # 10 requests per 60s = 1 per 6s
        rate_limiter.release()
```

### Batch with concurrency control (recommended)
```python
# LangChain's built-in concurrency control
results = chain.batch(
    inputs,
    config={"max_concurrency": 5}  # Max 5 parallel API calls
)
```

---

## Error Monitoring Checklist

In production, track these metrics:
- [ ] Retry rate (how often do retries fire?)
- [ ] Fallback rate (how often does primary fail?)
- [ ] Parse error rate (how often does output parsing fail?)
- [ ] Timeout rate
- [ ] Average latency per chain
- [ ] Cost per successful response
- [ ] Error distribution by type

---

## Best Practices

1. **Always have a fallback model** — single-provider dependency is a risk
2. **Retry transient errors, don't retry logic errors** — filter exceptions
3. **Set timeouts everywhere** — hanging requests waste resources
4. **Log every error** — you can't fix what you don't measure
5. **Use exponential backoff with jitter** — be a good API citizen
6. **Test failure modes** — intentionally break things during development
7. **Cache successful responses** — reduce reliance on API availability
8. **Alert on fallback activation** — know when your primary is degraded
