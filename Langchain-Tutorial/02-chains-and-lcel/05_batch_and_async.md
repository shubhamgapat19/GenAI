# Batch Processing & Async — Deep Dive Notes

## Why Batch & Async?

| Problem | Solution |
|---------|----------|
| Processing 1000 items one by one takes forever | `batch()` runs them in parallel |
| API call blocks your web server | `ainvoke()` is non-blocking |
| Hitting rate limits with too many parallel calls | `max_concurrency` throttles |
| Need multiple independent LLM calls | `asyncio.gather()` runs them concurrently |

---

## Batch Processing

### .batch() — Parallel execution of multiple inputs

```python
chain = prompt | llm | StrOutputParser()

inputs = [
    {"topic": "Python"},
    {"topic": "JavaScript"},
    {"topic": "Rust"},
]

# All three run in parallel (not sequentially!)
results = chain.batch(inputs)
# results = ["Python is...", "JavaScript is...", "Rust is..."]
```

### Concurrency Control

```python
# Without limit: ALL inputs sent simultaneously
results = chain.batch(inputs)  # 100 inputs = 100 API calls at once!

# With limit: controlled parallelism
results = chain.batch(inputs, config={"max_concurrency": 5})
# Processes 5 at a time, waits for a slot before starting next
```

### Choosing max_concurrency

| Scenario | Suggested Value |
|----------|----------------|
| OpenAI (free tier) | 3-5 |
| OpenAI (paid tier) | 10-20 |
| Azure OpenAI | 5-10 (depends on deployment) |
| Development/testing | 2-3 |
| Production (high volume) | 10-50 (based on rate limit) |

---

## Async Patterns

### Why Async?

**Sync (blocking):**
```python
# Thread is blocked during API call — can't do anything else
result = chain.invoke(input)  # Waits 2-5 seconds here
```

**Async (non-blocking):**
```python
# Thread is FREE during API call — can handle other requests
result = await chain.ainvoke(input)  # Does other work while waiting
```

### When to Use Async
- ✅ Web servers (FastAPI, Starlette)
- ✅ Running multiple chains concurrently
- ✅ Real-time applications
- ❌ Simple scripts (overkill)
- ❌ Jupyter notebooks (mostly)

---

## Async Methods

### ainvoke — Single async call
```python
result = await chain.ainvoke({"topic": "AI"})
```

### abatch — Async batch
```python
results = await chain.abatch(inputs, config={"max_concurrency": 5})
```

### astream — Async streaming
```python
async for chunk in chain.astream({"topic": "AI"}):
    print(chunk, end="")
```

---

## asyncio.gather — True Concurrency

Run multiple independent chains at the same time:

```python
import asyncio

async def analyze_document(doc):
    # Three independent analyses run concurrently
    summary_task = summary_chain.ainvoke({"text": doc})
    sentiment_task = sentiment_chain.ainvoke({"text": doc})
    entities_task = entities_chain.ainvoke({"text": doc})
    
    summary, sentiment, entities = await asyncio.gather(
        summary_task, sentiment_task, entities_task
    )
    
    return {
        "summary": summary,
        "sentiment": sentiment,
        "entities": entities,
    }
```

### gather with error handling
```python
results = await asyncio.gather(
    chain.ainvoke(input1),
    chain.ainvoke(input2),
    chain.ainvoke(input3),
    return_exceptions=True,  # Don't crash on individual failures
)

for i, result in enumerate(results):
    if isinstance(result, Exception):
        print(f"Input {i} failed: {result}")
    else:
        print(f"Input {i} succeeded: {result}")
```

---

## Real-World Batch Patterns

### Pattern 1: Dataset Processing
```python
import pandas as pd

df = pd.read_csv("reviews.csv")

# Convert DataFrame rows to chain inputs
inputs = [{"text": row["review"]} for _, row in df.iterrows()]

# Process in batches
results = chain.batch(inputs, config={"max_concurrency": 10})

# Add results back to DataFrame
df["sentiment"] = results
df.to_csv("reviews_analyzed.csv", index=False)
```

### Pattern 2: Chunked Batch (for very large datasets)
```python
def chunked_batch(chain, inputs, chunk_size=50, max_concurrency=5):
    """Process large datasets in manageable chunks."""
    all_results = []
    
    for i in range(0, len(inputs), chunk_size):
        chunk = inputs[i:i + chunk_size]
        results = chain.batch(chunk, config={"max_concurrency": max_concurrency})
        all_results.extend(results)
        print(f"Processed {min(i + chunk_size, len(inputs))}/{len(inputs)}")
    
    return all_results

# Process 10,000 items in chunks of 50
all_results = chunked_batch(chain, large_input_list)
```

### Pattern 3: Batch with Progress & Error Tracking
```python
async def robust_batch(chain, inputs, max_concurrency=5):
    """Production batch: progress, errors, results."""
    results = []
    errors = []
    
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_one(i, inp):
        async with semaphore:
            try:
                result = await chain.ainvoke(inp)
                return i, result, None
            except Exception as e:
                return i, None, str(e)
    
    tasks = [process_one(i, inp) for i, inp in enumerate(inputs)]
    
    for completed in asyncio.as_completed(tasks):
        i, result, error = await completed
        if error:
            errors.append({"index": i, "error": error})
        else:
            results.append({"index": i, "result": result})
        
        # Progress
        done = len(results) + len(errors)
        print(f"\r  Progress: {done}/{len(inputs)}", end="")
    
    print()
    return results, errors
```

---

## Performance Comparison

For 20 inputs, each taking ~1.5s:

| Method | Time | How |
|--------|------|-----|
| Sequential loop | ~30s | One at a time |
| `batch(max_concurrency=5)` | ~6s | 5 parallel |
| `batch(max_concurrency=10)` | ~3s | 10 parallel |
| `batch(max_concurrency=20)` | ~1.5s | All parallel |

**Speedup = total_items / max_concurrency** (approximately)

---

## Cost Awareness

Batch processing is great for speed but can burn through budget:

```python
# Quick cost estimate before running batch
def estimate_batch_cost(inputs, avg_input_tokens=100, avg_output_tokens=200, model="gpt-4o-mini"):
    costs = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # per 1M tokens
        "gpt-4o": {"input": 2.50, "output": 10.00},
    }
    
    model_cost = costs.get(model, costs["gpt-4o-mini"])
    total_input = len(inputs) * avg_input_tokens
    total_output = len(inputs) * avg_output_tokens
    
    cost = (total_input / 1_000_000 * model_cost["input"] + 
            total_output / 1_000_000 * model_cost["output"])
    
    print(f"Estimated cost for {len(inputs)} items: ${cost:.4f}")
    return cost

# Check before processing
estimate_batch_cost(inputs, model="gpt-4o-mini")
# "Estimated cost for 1000 items: $0.1125"
```

---

## Best Practices

1. **Always use `max_concurrency`** — unbounded parallelism hits rate limits
2. **Start conservative** — begin with max_concurrency=3, increase if stable
3. **Use async in web apps** — sync calls block your entire server
4. **Chunk large datasets** — don't try to batch 100K items at once
5. **Track errors per batch** — one failure shouldn't lose all results
6. **Estimate costs first** — batch 10K items with GPT-4o = serious money
7. **Add progress tracking** — long batches need visibility
8. **Cache results** — if you batch daily, cache to avoid re-processing
9. **Use `return_exceptions=True` with gather** — handle failures individually
10. **Log batch metadata** — items processed, time taken, error rate

---

## Quick Reference

```python
# Sync
chain.invoke(input)                          # Single
chain.batch(inputs)                          # Multiple (parallel)
chain.batch(inputs, config={"max_concurrency": 5})  # Throttled

# Async
await chain.ainvoke(input)                   # Single
await chain.abatch(inputs)                   # Multiple (parallel)
await asyncio.gather(*[chain.ainvoke(x) for x in inputs])  # Manual parallel

# Streaming
for chunk in chain.stream(input): ...        # Sync stream
async for chunk in chain.astream(input): ... # Async stream
```
