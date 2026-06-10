# RunnableParallel — Deep Dive Notes

## What is RunnableParallel?

`RunnableParallel` executes **multiple Runnables simultaneously** with the same input, then combines their outputs into a dictionary. It's how you run independent operations concurrently in LCEL.

---

## Core Concept

```
                    ┌─── Chain A ───→ result_a ─┐
Input ──→ Fork ────┼─── Chain B ───→ result_b ─┼──→ {"a": result_a, "b": result_b, "c": result_c}
                    └─── Chain C ───→ result_c ─┘
```

All branches receive the **same input**. All branches run **in parallel**. Output is a **dict** with named results.

---

## Three Ways to Create

### 1. Explicit RunnableParallel
```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    summary=summary_chain,
    sentiment=sentiment_chain,
    keywords=keywords_chain,
)
```

### 2. Dictionary Shorthand (most common)
```python
# This IS a RunnableParallel — Python dict literal
parallel = {
    "summary": summary_chain,
    "sentiment": sentiment_chain,
}
```

### 3. Using .pipe() with dict
```python
chain = some_chain.pipe({
    "summary": summary_chain,
    "sentiment": sentiment_chain,
})
```

---

## Input/Output Contract

| Input | Output |
|-------|--------|
| Whatever the individual chains expect (all get the same input) | `dict` with keys = names you gave each branch |

```python
parallel = RunnableParallel(
    a=chain_a,  # chain_a.invoke(input) → stored as "a"
    b=chain_b,  # chain_b.invoke(input) → stored as "b"
)

result = parallel.invoke({"topic": "AI"})
# result = {"a": <output of chain_a>, "b": <output of chain_b>}
```

---

## Common Patterns

### Pattern 1: Parallel Analysis
```python
# Analyze text from multiple angles simultaneously
analysis = RunnableParallel(
    summary=summary_chain,
    sentiment=sentiment_chain,
    entities=entity_extraction_chain,
    language=language_detection_chain,
)
# One input, four parallel analyses
```

### Pattern 2: Passthrough + Processing
```python
# Keep original input while also processing it
setup = {
    "question": RunnablePassthrough(),        # Original passes through
    "context": retriever_chain,                # Retrieval happens in parallel
    "language": language_detect_chain,         # Detection in parallel
}
# Then use all three in the next step
```

### Pattern 3: Parallel → Combine
```python
# Generate multiple perspectives, then synthesize
chain = (
    RunnableParallel(
        optimist=optimist_chain,
        pessimist=pessimist_chain,
        realist=realist_chain,
    )
    | combine_perspectives_chain
)
```

### Pattern 4: Nested Parallel
```python
# Groups of parallel operations
chain = RunnableParallel(
    technical=RunnableParallel(
        complexity=complexity_chain,
        stack=tech_stack_chain,
    ),
    business=RunnableParallel(
        cost=cost_chain,
        timeline=timeline_chain,
    ),
)
# result = {"technical": {"complexity": ..., "stack": ...}, "business": {...}}
```

---

## Performance Benefits

| Approach | 3 chains × 2s each | Total Time |
|----------|--------------------:|----------:|
| Sequential | 2s + 2s + 2s | **6s** |
| RunnableParallel | max(2s, 2s, 2s) | **2s** |

Parallel execution is 3x faster here. The total time equals the **slowest** branch.

---

## RunnablePassthrough in Parallel Context

`RunnablePassthrough()` passes its input unchanged — essential in parallel dicts:

```python
# Without passthrough: you lose the original input
bad = {
    "analysis": analysis_chain,  # Only analysis available downstream
}

# With passthrough: original input + analysis available
good = {
    "original": RunnablePassthrough(),  # Keeps {"topic": "AI"} 
    "analysis": analysis_chain,          # Adds analysis result
}
```

### RunnablePassthrough.assign()
Adds new keys while keeping all existing ones:

```python
# Adds "analysis" key to the existing input dict
chain = RunnablePassthrough.assign(
    analysis=analysis_chain
)

# Input:  {"topic": "AI", "user": "Bob"}
# Output: {"topic": "AI", "user": "Bob", "analysis": "..."}
```

---

## Error Handling in Parallel

If one branch fails, the entire `RunnableParallel` fails by default. Fix with fallbacks:

```python
# Add fallback to individual branches
safe_parallel = RunnableParallel(
    summary=summary_chain.with_fallbacks([simple_summary_chain]),
    sentiment=sentiment_chain.with_fallbacks([default_sentiment]),
)
```

---

## Best Practices

1. **Only parallelize independent operations** — if B needs A's output, they can't be parallel
2. **Use dict shorthand** — cleaner than explicit `RunnableParallel()`
3. **Name branches semantically** — `"summary"` not `"chain_1"`
4. **Add fallbacks per branch** — one failure shouldn't kill everything
5. **Watch API rate limits** — 5 parallel branches = 5 simultaneous API calls
6. **Use for the RAG pattern** — `{"context": retriever, "question": passthrough}`

---

## When NOT to Use

- When chains are **dependent** (output of one feeds into another) → use sequential
- When you only have **one chain** → just invoke directly
- When **order matters** → use sequential chain
- When you're **rate limited** → batch with `max_concurrency` instead
