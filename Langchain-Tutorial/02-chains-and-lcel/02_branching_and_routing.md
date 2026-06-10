# Branching & Routing — Deep Dive Notes

## What is Routing?

Routing means sending different inputs to **different chains** based on some condition. It's the "if/else" of LangChain — you classify the input, then direct it to the appropriate handler.

---

## Why Routing Matters

| Without Routing | With Routing |
|----------------|--------------|
| One-size-fits-all prompt | Specialized prompts per category |
| Same model for everything | Cheap model for easy, expensive for hard |
| Generic responses | Domain-expert responses |
| Wasted tokens on simple questions | Efficient token usage |

---

## Routing Approaches

### Approach 1: RunnableBranch (Rule-based)

Hardcoded conditions — fast, predictable, no LLM call:

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "code" in x["input"].lower(), code_chain),
    (lambda x: "math" in x["input"].lower(), math_chain),
    general_chain,  # default — REQUIRED
)
```

**How it works:**
1. Evaluates conditions top to bottom
2. First `True` condition wins
3. If none match → default chain runs

**Pros:** Fast (no API call), deterministic, easy to debug
**Cons:** Brittle (keyword matching), doesn't handle nuance

### Approach 2: LLM-based Router (Semantic)

Use an LLM to classify, then route based on classification:

```python
# Step 1: Classify
classifier = (
    ChatPromptTemplate.from_messages([
        ("system", "Classify as: code, math, or general. One word only."),
        ("human", "{input}"),
    ])
    | llm | StrOutputParser()
)

# Step 2: Route based on classification
def route(info):
    category = info["category"].strip().lower()
    chains = {"code": code_chain, "math": math_chain}
    selected = chains.get(category, general_chain)
    return selected.invoke({"input": info["input"]})

router = {"category": classifier, "input": lambda x: x["input"]} | RunnableLambda(route)
```

**Pros:** Handles nuance, understands context, adapts to new inputs
**Cons:** Extra API call (latency + cost), non-deterministic

### Approach 3: Embedding-based Router (Semantic, no LLM call)

Route based on semantic similarity (fast, no API call after setup):

```python
from langchain_core.embeddings import OpenAIEmbeddings
import numpy as np

# Pre-compute embeddings for each category
category_descriptions = {
    "code": "programming, coding, functions, classes, bugs",
    "math": "calculations, equations, algebra, statistics",
    "general": "general knowledge, facts, opinions",
}

embeddings = OpenAIEmbeddings()
category_embeddings = {
    cat: embeddings.embed_query(desc)
    for cat, desc in category_descriptions.items()
}

def embed_route(input_dict):
    query_embedding = embeddings.embed_query(input_dict["input"])
    # Find most similar category
    similarities = {
        cat: np.dot(query_embedding, emb)
        for cat, emb in category_embeddings.items()
    }
    best_category = max(similarities, key=similarities.get)
    chains = {"code": code_chain, "math": math_chain, "general": general_chain}
    return chains[best_category].invoke(input_dict)
```

**Pros:** Fast (one embedding call), semantic understanding, no LLM needed
**Cons:** Setup complexity, needs good category descriptions

---

## RunnableBranch — Deep Dive

### Syntax
```python
RunnableBranch(
    (condition_1, runnable_1),  # if condition_1(input) → runnable_1
    (condition_2, runnable_2),  # elif condition_2(input) → runnable_2
    (condition_3, runnable_3),  # elif condition_3(input) → runnable_3
    default_runnable,            # else → default_runnable
)
```

### Condition Functions
Conditions receive the **full input** and return `True`/`False`:

```python
# Simple keyword check
lambda x: "python" in x["input"].lower()

# Length-based routing
lambda x: len(x["input"]) > 500  # Long inputs → summarizer

# Multi-condition
lambda x: x.get("priority") == "high" and x.get("type") == "bug"

# Based on metadata
lambda x: x.get("user_tier") == "premium"
```

### Complex Example
```python
branch = RunnableBranch(
    # Priority routing
    (lambda x: x.get("urgent", False), urgent_chain),
    
    # Content-type routing
    (lambda x: x.get("type") == "code_review", code_review_chain),
    (lambda x: x.get("type") == "summarize", summary_chain),
    (lambda x: x.get("type") == "translate", translation_chain),
    
    # Length-based routing
    (lambda x: len(x.get("input", "")) > 1000, long_input_chain),
    
    # Default
    general_chain,
)
```

---

## Fallback Patterns

### Model Fallback (most common)
```python
# GPT-4o fails → try GPT-4o-mini → try Claude
robust_llm = primary_llm.with_fallbacks([
    secondary_llm,
    tertiary_llm,
])
```

### Chain Fallback
```python
# Complex JSON extraction fails → simpler text extraction
robust_chain = json_extraction_chain.with_fallbacks([
    text_extraction_chain,
    default_response_chain,
])
```

### Fallback with exception filtering
```python
# Only fallback on specific errors
from openai import RateLimitError, APITimeoutError

robust_chain = primary_chain.with_fallbacks(
    [fallback_chain],
    exceptions_to_handle=(RateLimitError, APITimeoutError),
    # Won't fallback on AuthenticationError — that should fail loudly
)
```

---

## Production Routing Architecture

```
User Input
    ↓
[Classifier] ──→ category
    ↓
[Router]
    ├── "simple" ──→ GPT-4o-mini (cheap, fast)
    ├── "complex" ──→ GPT-4o (expensive, smart)
    ├── "code" ──→ Specialized code chain
    ├── "unsafe" ──→ Rejection response (no LLM call)
    └── default ──→ General chain
```

### Cost Optimization Router
```python
# Route easy questions to cheap model, hard to expensive
difficulty_classifier = (
    ChatPromptTemplate.from_messages([
        ("system", "Rate question difficulty: easy or hard. One word."),
        ("human", "{input}"),
    ])
    | cheap_llm | StrOutputParser()
)

branch = RunnableBranch(
    (lambda x: "hard" in x["difficulty"], expensive_chain),
    cheap_chain,  # Easy questions → cheap model
)

cost_optimized = {"difficulty": difficulty_classifier, "input": lambda x: x["input"]} | branch
```

---

## Best Practices

1. **Put most specific conditions first** — RunnableBranch matches first-wins
2. **Always have a default** — never leave inputs unhandled
3. **Log routing decisions** — you need to debug why things went to wrong chain
4. **LLM router for user-facing** — keyword matching fails for natural language
5. **Keyword router for system inputs** — when format is predictable
6. **Test edge cases** — empty input, very long input, mixed signals
7. **Monitor routing distribution** — if 95% goes to default, your routes aren't working

---

## When to Use What

| Scenario | Approach |
|----------|----------|
| Fixed input format (API, forms) | `RunnableBranch` with conditions |
| Free-form user text | LLM-based classifier → route |
| High volume, low latency | Embedding-based router |
| Cost optimization | Difficulty classifier → cheap/expensive model |
| Safety filtering | Keyword + LLM classifier → reject/allow |
