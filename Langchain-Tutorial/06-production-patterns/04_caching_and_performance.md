# Caching & Performance Optimization — Deep Dive Notes

## Why Performance Matters

LLM calls are **slow** (1-5s) and **expensive** ($0.001-0.01 per call). At scale:
- 10K daily users × 5 calls = 50K LLM calls/day
- Without caching: $50-500/day, each user waits 2-5s
- With caching: $10-100/day (80% cache hit), most users get instant response

---

## Caching Strategies

| Type | How It Works | Hit Rate | Complexity |
|------|-------------|----------|------------|
| **Exact match** | Same input → same output | 20-40% | Low |
| **Semantic** | Similar input → same output | 50-70% | Medium |
| **Query-level** | Cache by normalized query | 30-50% | Low |
| **Fragment** | Cache sub-results (embeddings, retrieval) | 60-80% | Medium |

---

## Strategy 1: Exact Match Cache

```python
cache = {}

def cached_llm(input_data: dict) -> str:
    key = hash(json.dumps(input_data, sort_keys=True))
    if key in cache:
        return cache[key]  # Instant!
    result = chain.invoke(input_data)
    cache[key] = result
    return result
```

### When Exact Match Works
- FAQ-style questions (same question asked by many users)
- Repeated API calls (same parameters)
- Static content generation (product descriptions)

### When It Fails
- Unique/personalized questions
- Questions with slight wording differences
- Time-sensitive queries ("What's the date today?")

---

## Strategy 2: Semantic Cache

Uses embeddings to find similar previous queries:

```python
# Query: "What is machine learning?"
# Cache has: "Explain what machine learning is" (similarity: 0.95)
# → Cache HIT! Return cached response.
```

### Implementation
```python
class SemanticCache:
    def __init__(self, threshold=0.92):
        self.threshold = threshold
        self.entries = []  # [{embedding, query, response}]
    
    def get(self, query):
        query_emb = embed(query)
        for entry in self.entries:
            if cosine_sim(query_emb, entry["embedding"]) >= self.threshold:
                return entry["response"]
        return None  # Cache miss
```

### Threshold Tuning
- **0.98+:** Very strict, almost exact match (few hits)
- **0.92-0.95:** Good balance (catches paraphrases)
- **0.85-0.90:** Aggressive (may return wrong cached answer)

### Tradeoffs
- **Cost:** Each cache check = 1 embedding call (~$0.00001)
- **Risk:** Too low threshold → wrong answers from cache
- **Memory:** Store embeddings + responses (grows linearly)

---

## Strategy 3: Fragment Caching

Cache intermediate results, not just final answers:

```
[User Query]
    ↓
[Embedding] ← CACHE THIS (same query = same embedding)
    ↓
[Vector Search] ← CACHE THIS (same embedding = same docs)
    ↓
[LLM Generation] ← Can't cache (different context combinations)
```

### What to Cache at Each Level

| Component | Cache Strategy | TTL |
|-----------|---------------|-----|
| Embeddings | Exact match on input text | Long (months) |
| Document chunks | Store permanently | Until docs change |
| Retrieval results | By query embedding | Hours-days |
| LLM responses | Exact/semantic match | Minutes-hours |

---

## Prompt Optimization (Reduce Tokens)

Every token costs money and adds latency. Optimize:

### System Prompt
```
BAD (82 tokens):
"You are an incredibly helpful, knowledgeable, and friendly AI assistant 
who is always eager to help users with their questions..."

GOOD (18 tokens):
"You are a helpful assistant. Be concise. Say 'I don't know' if unsure."
```

### Token Counting
```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o-mini")
tokens = len(enc.encode(text))
```

### Optimization Techniques

| Technique | Savings | Risk |
|-----------|---------|------|
| Shorter system prompt | 30-80% of system tokens | May reduce quality |
| Trim retrieved docs | 50-70% of context tokens | May miss info |
| Compress conversation history | 60-80% of history tokens | May lose context |
| Remove examples from prompt | 50-90% of example tokens | May reduce accuracy |
| Use shorter model outputs | 50% of output tokens | May reduce detail |

---

## Batch Processing

Process multiple requests together:

```python
# Sequential: 5 calls × 2s = 10s total
results = [chain.invoke(q) for q in questions]

# Batch: 5 calls concurrent = 2-3s total
results = chain.batch(questions, config={"max_concurrency": 5})
```

### When to Batch
- Processing a dataset (eval, migration)
- Multiple independent sub-questions
- Background jobs (summarize all docs)

### When NOT to Batch
- Real-time chat (user waiting for ONE response)
- When results depend on each other (sequential reasoning)

---

## Async for Throughput

```python
import asyncio

async def handle_requests(requests):
    # Process many users concurrently
    tasks = [chain.ainvoke(req) for req in requests]
    return await asyncio.gather(*tasks)
```

### Sync vs Async

| Pattern | Best For |
|---------|----------|
| Sync (`invoke`) | Simple scripts, CLI tools |
| Async (`ainvoke`) | Web servers handling many concurrent requests |
| Batch (`batch`) | Processing datasets, bulk operations |
| Stream (`stream`) | Real-time chat, showing progress |

---

## Streaming for Perceived Performance

```
Without streaming: [wait 3 seconds...] "Here is the full answer displayed at once."
With streaming:     "H" "ere" " is" " the" " answer" " displayed" " as" " it's" " generated."
```

Time to first token (TTFT) → what users *feel* as response time.
- Without streaming: TTFT = total generation time (2-5s)
- With streaming: TTFT = ~200-500ms

```python
for chunk in chain.stream({"question": "..."}):
    print(chunk, end="", flush=True)
```

---

## Performance Optimization Playbook

### Quick Wins (implement first)
1. **Exact match cache** — handles repeated queries instantly
2. **Shorter system prompt** — save tokens every single call
3. **Streaming** — user perceives instant response
4. **Smaller model** — GPT-4o-mini instead of GPT-4o (10x cheaper)

### Medium Effort
5. **Semantic cache** — catches paraphrased queries
6. **Fragment caching** — cache embeddings + retrieval results
7. **Batch processing** — for background jobs
8. **Async serving** — handle more concurrent users

### High Effort
9. **Custom model** — fine-tune smaller model for your task
10. **Edge caching** — CDN-level caching for common queries
11. **Request deduplication** — merge identical concurrent requests
12. **Prompt compression** — LLMLingua-style token reduction

---

## Measuring Performance

| Metric | Target | How to Measure |
|--------|--------|---------------|
| TTFT (time to first token) | <500ms | Timestamp first stream chunk |
| Total latency | <3s (chat), <10s (RAG) | End-to-end timer |
| Cache hit rate | >50% | hits / (hits + misses) |
| Cost per query | <$0.01 | Track with get_openai_callback |
| Throughput | 100+ req/s | Load test with concurrent users |

---

## Best Practices

1. **Cache at multiple levels** — embeddings, retrieval, responses
2. **Start with exact match** — simple, high impact
3. **Stream everything in production** — users hate waiting
4. **Measure before optimizing** — find the actual bottleneck
5. **Set cache TTL** — stale data is worse than no cache
6. **Monitor cache hit rate** — if <20%, cache isn't helping
7. **Use the smallest model that works** — test quality vs. cost
8. **Batch background work** — don't use real-time patterns for batch jobs

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| No caching | Paying for same answer repeatedly | Add exact match cache |
| Cache without TTL | Stale answers | Set expiration (1h for dynamic, 24h for static) |
| Semantic cache threshold too low | Returns wrong cached answers | Start at 0.95, lower carefully |
| Verbose prompts | Wasting tokens every call | Trim to essential |
| Not streaming | Users think app is broken | Always stream in chat UX |
| Over-optimizing early | Wasted engineering time | Measure first, optimize bottlenecks |
| Caching personalized responses | Wrong answer for different user | Only cache non-personalized |
