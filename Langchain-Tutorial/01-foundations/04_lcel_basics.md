# LCEL (LangChain Expression Language) — Deep Dive Notes

## What is LCEL?

**LCEL** is LangChain's declarative syntax for composing chains using the **pipe operator** (`|`). It's the backbone of everything in LangChain — every chain, agent, and RAG pipeline is built with LCEL.

Think of it like Unix pipes:
```bash
# Unix: data flows left to right
cat file.txt | grep "error" | wc -l

# LCEL: data flows left to right
prompt | llm | parser
```

---

## Why LCEL Exists

Before LCEL (legacy LangChain):
```python
# Old way — verbose, no streaming, no async
chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
result = chain.run(topic="AI")
```

With LCEL:
```python
# New way — concise, streaming/async/batch built-in
chain = prompt | llm | parser
result = chain.invoke({"topic": "AI"})
```

**LCEL gives you for free:**
- ✅ Streaming (token-by-token)
- ✅ Async support
- ✅ Batch processing (parallel)
- ✅ Retries and fallbacks
- ✅ Tracing (LangSmith integration)
- ✅ Type checking between components

---

## The Runnable Protocol

Every component in LCEL implements the **Runnable** interface:

```python
class Runnable:
    def invoke(self, input)       # Single input → single output
    def batch(self, inputs)       # Multiple inputs → multiple outputs (parallel)
    def stream(self, input)       # Single input → streaming output
    async def ainvoke(self, input)  # Async single
    async def abatch(self, inputs)  # Async batch
    async def astream(self, input)  # Async stream
```

**Any Runnable can be piped with `|`**. This includes:
- Prompt templates
- Chat models
- Output parsers
- Custom functions (via RunnableLambda)
- Other chains

---

## Basic Chain Composition

### The simplest chain
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_messages([("human", "Explain {topic} simply")])
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# All these work automatically:
chain.invoke({"topic": "LCEL"})                      # → str
chain.batch([{"topic": "AI"}, {"topic": "ML"}])      # → [str, str]
for chunk in chain.stream({"topic": "LCEL"}):        # → streaming
    print(chunk, end="")
```

### Data flow visualization
```
{"topic": "LCEL"}
       ↓
[ChatPromptTemplate] → formats messages
       ↓
[ChatOpenAI] → calls API, returns AIMessage
       ↓
[StrOutputParser] → extracts .content string
       ↓
"LCEL is..."
```

---

## Core Runnable Types

### 1. RunnablePassthrough — Pass data through unchanged

```python
from langchain_core.runnables import RunnablePassthrough

# Passes input directly to output
passthrough = RunnablePassthrough()
passthrough.invoke("hello")  # → "hello"

# Most useful in parallel constructions (see RunnableParallel below)
```

**Use case:** When you need the original input alongside transformed data.

### 2. RunnableLambda — Custom Python functions in chains

```python
from langchain_core.runnables import RunnableLambda

# Wrap any function
def uppercase(text: str) -> str:
    return text.upper()

chain = prompt | llm | StrOutputParser() | RunnableLambda(uppercase)

# Or use the decorator
@RunnableLambda
def add_metadata(text: str) -> dict:
    return {"text": text, "word_count": len(text.split())}
```

**Use case:** Any custom transformation — data cleaning, formatting, API calls, logging.

### 3. RunnableParallel — Run multiple branches simultaneously

```python
from langchain_core.runnables import RunnableParallel

# Two chains run in parallel, results combined into a dict
parallel = RunnableParallel(
    joke=joke_chain,
    poem=poem_chain,
    fact=fact_chain,
)

result = parallel.invoke({"topic": "Python"})
# result = {"joke": "...", "poem": "...", "fact": "..."}
```

**Dictionary shorthand** (same thing):
```python
# This IS a RunnableParallel under the hood
chain = {"joke": joke_chain, "poem": poem_chain} | combine_chain
```

### 4. RunnableBranch — Conditional routing

```python
from langchain_core.runnables import RunnableBranch

# Route to different chains based on input
branch = RunnableBranch(
    (lambda x: "code" in x["topic"], code_chain),      # If topic has "code"
    (lambda x: "math" in x["topic"], math_chain),      # If topic has "math"
    general_chain,                                       # Default fallback
)

branch.invoke({"topic": "code review"})  # → goes to code_chain
branch.invoke({"topic": "math problem"}) # → goes to math_chain
branch.invoke({"topic": "cooking"})      # → goes to general_chain
```

---

## Advanced Patterns

### Pattern 1: Sequential chains (output of one → input of next)

```python
# Chain 1: Generate an outline
outline_chain = (
    ChatPromptTemplate.from_messages([
        ("human", "Create a 3-point outline for an article about {topic}")
    ])
    | llm
    | StrOutputParser()
)

# Chain 2: Write the article from the outline
article_chain = (
    ChatPromptTemplate.from_messages([
        ("human", "Write a short article based on this outline:\n\n{outline}")
    ])
    | llm
    | StrOutputParser()
)

# Connect them: output of chain1 feeds into chain2
full_chain = (
    {"outline": outline_chain}  # Run chain1, name output "outline"
    | article_chain              # Chain2 reads {outline} variable
)

result = full_chain.invoke({"topic": "LangChain"})
```

### Pattern 2: Extracting data alongside passing it through

```python
from langchain_core.runnables import RunnablePassthrough

# Get both the original question AND a classification
chain = (
    {
        "question": RunnablePassthrough(),              # Original input passes through
        "category": classification_chain,               # Runs classification in parallel
    }
    | route_to_specialist_chain                         # Uses both question + category
)
```

### Pattern 3: Adding context to the chain

```python
# Common RAG pattern: retrieve docs, then pass both question + docs to LLM
chain = (
    {
        "context": retriever,                    # Fetch relevant documents
        "question": RunnablePassthrough(),       # Keep original question
    }
    | rag_prompt                                 # Template uses {context} and {question}
    | llm
    | StrOutputParser()
)
```

### Pattern 4: Map over a list

```python
# Apply a chain to each item in a list
chain = prompt | llm | StrOutputParser()

# .map() applies the chain to each element
list_chain = chain.map()
results = list_chain.invoke([
    {"topic": "AI"},
    {"topic": "ML"},
    {"topic": "DL"},
])
# Returns: ["AI is...", "ML is...", "DL is..."]
```

---

## Configurable Chains

Make chains that can be configured at runtime:

```python
from langchain_core.runnables import ConfigurableField

# Make the model configurable
configurable_llm = ChatOpenAI(model="gpt-4o-mini").configurable_fields(
    model_name=ConfigurableField(
        id="model",
        name="Model Name",
    )
)

chain = prompt | configurable_llm | StrOutputParser()

# Use default (gpt-4o-mini)
chain.invoke({"topic": "AI"})

# Override at runtime
chain.invoke(
    {"topic": "AI"},
    config={"configurable": {"model": "gpt-4o"}}  # Use GPT-4o for this call
)
```

---

## Error Handling in LCEL

### Fallbacks — Try backup on failure
```python
# If primary model fails, use backup
chain_with_fallback = (
    primary_chain.with_fallbacks([fallback_chain])
)

# Real example: GPT-4o fails → fall back to GPT-4o-mini
main_llm = ChatOpenAI(model="gpt-4o")
backup_llm = ChatOpenAI(model="gpt-4o-mini")

robust_chain = (
    prompt
    | main_llm.with_fallbacks([backup_llm])
    | StrOutputParser()
)
```

### Retry — Auto-retry on transient failures
```python
chain_with_retry = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)
```

### Exception handling with RunnableLambda
```python
def safe_parse(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse", "raw": text}

chain = prompt | llm | StrOutputParser() | RunnableLambda(safe_parse)
```

---

## Debugging & Inspection

### See the chain structure
```python
chain.get_graph().print_ascii()

# Output:
#      +--------+
#      | Prompt |
#      +--------+
#           |
#      +-------+
#      |  LLM  |
#      +-------+
#           |
#      +--------+
#      | Parser |
#      +--------+
```

### Check input/output schemas
```python
chain.input_schema.model_json_schema()
# {'properties': {'topic': {'type': 'string'}}, 'required': ['topic']}

chain.output_schema.model_json_schema()
# {'type': 'string'}
```

### Add logging with callbacks
```python
from langchain_core.callbacks import StdOutCallbackHandler

result = chain.invoke(
    {"topic": "AI"},
    config={"callbacks": [StdOutCallbackHandler()]}
)
# Prints each step's input/output to console
```

---

## Performance: Batch & Async

### Batch (parallel execution)
```python
# Process 100 inputs — LangChain handles parallelism
inputs = [{"topic": f"topic_{i}"} for i in range(100)]

results = chain.batch(
    inputs,
    config={"max_concurrency": 10}  # Max 10 parallel calls
)
```

### Async (non-blocking)
```python
import asyncio

async def process_many():
    tasks = [
        chain.ainvoke({"topic": "AI"}),
        chain.ainvoke({"topic": "ML"}),
        chain.ainvoke({"topic": "DL"}),
    ]
    results = await asyncio.gather(*tasks)
    return results
```

### Streaming events (fine-grained)
```python
async for event in chain.astream_events({"topic": "AI"}, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
    elif event["event"] == "on_chain_end":
        print("\n\nChain finished!")
```

---

## LCEL vs. Writing Vanilla Python

| Aspect | LCEL | Plain Python |
|--------|------|-------------|
| Streaming | Free (built-in) | Manual generator logic |
| Async | Free (built-in) | Manual async wrappers |
| Batching | Free (parallel) | Manual ThreadPoolExecutor |
| Tracing | Automatic (LangSmith) | Manual instrumentation |
| Retries/Fallbacks | One-liner | Try/except + retry logic |
| Type safety | Schema validation | No guarantees |
| Composability | Pipe operator | Nested function calls |

---

## Common Patterns Cheat Sheet

```python
# Simple chain
chain = prompt | llm | parser

# Chain with parallel inputs
chain = {"a": chain_a, "b": chain_b} | combine_chain

# Chain with passthrough
chain = {"context": retriever, "q": RunnablePassthrough()} | prompt | llm

# Chain with custom function
chain = prompt | llm | StrOutputParser() | RunnableLambda(my_func)

# Chain with fallback
chain = (prompt | llm | parser).with_fallbacks([backup_chain])

# Chain with retry
chain = (prompt | llm | parser).with_retry(stop_after_attempt=3)

# Conditional routing
chain = RunnableBranch(
    (condition_1, chain_1),
    (condition_2, chain_2),
    default_chain,
)

# Configurable chain
chain = prompt | llm.configurable_fields(...) | parser
```

---

## Mental Model

Think of LCEL as a **data pipeline**:

```
Input Dict → [Transform] → [Transform] → [Transform] → Output
              (Prompt)       (LLM)         (Parser)

Each transform:
- Receives the output of the previous step
- Returns something for the next step
- Implements invoke/batch/stream/ainvoke/abatch/astream
```

Key insight: **The pipe operator doesn't execute anything.** It builds a blueprint. Execution happens when you call `.invoke()`, `.stream()`, etc.

---

## Best Practices

1. **Keep chains short** — 3-5 steps max. Split complex logic into sub-chains
2. **Name your chains** — `chain = (...).with_config(run_name="summarizer")`
3. **Use RunnableParallel for independent operations** — faster than sequential
4. **Always add fallbacks in production** — LLMs fail, APIs timeout
5. **Test each step independently** — invoke sub-chains separately before composing
6. **Use `.batch()` for multiple inputs** — much faster than looping `.invoke()`
7. **Prefer LCEL over manual orchestration** — you get streaming/tracing free


