# Chat Models — Deep Dive Notes

## What is a Chat Model?

A **Chat Model** is LangChain's abstraction over LLM APIs that work with **messages** (not raw text). Unlike older completion models that take a string and return a string, chat models understand conversation structure through message roles.

LangChain provides a **unified interface** — you can swap OpenAI for Anthropic, Google, or open-source models by changing one line.

---

## Architecture

```
Your Code
    ↓
ChatOpenAI (LangChain wrapper)
    ↓
OpenAI API (HTTP request)
    ↓
GPT-4o-mini (actual model)
    ↓
Response (AIMessage object)
```

LangChain's chat model is NOT the LLM itself — it's a **client wrapper** that handles:
- API authentication
- Message formatting
- Retry logic
- Token counting
- Streaming
- Caching

---

## Message Types

Every conversation is a **list of messages**. Each message has a `role`:

| Message Type | Role | Purpose | Example |
|-------------|------|---------|---------|
| `SystemMessage` | system | Sets behavior/persona | "You are a helpful tutor" |
| `HumanMessage` | user | User's input | "What is RAG?" |
| `AIMessage` | assistant | Model's response | "RAG stands for..." |
| `ToolMessage` | tool | Tool execution result | `{"result": 42}` |

### Why Roles Matter
- The LLM treats `system` messages as instructions it must follow
- `human` messages are what it responds to
- `AI` messages in history help it maintain consistency
- This is how **few-shot prompting** works — you show it example AI responses

### Code Example
```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="You are a Python expert."),
    HumanMessage(content="What is a decorator?"),
    AIMessage(content="A decorator is a function that wraps another function..."),
    HumanMessage(content="Give me an example"),  # LLM responds to this
]
```

---

## Model Initialization

### OpenAI
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",       # Model name
    temperature=0.7,            # Creativity (0-2)
    max_tokens=1000,            # Max response length
    timeout=30,                 # Request timeout (seconds)
    max_retries=2,              # Auto-retry on failure
    api_key="sk-...",           # Or set OPENAI_API_KEY env var
)
```

### Azure OpenAI
```python
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version="2024-02-01",
    azure_endpoint="https://your-resource.openai.azure.com/",
    # api_key from AZURE_OPENAI_API_KEY env var
)
```

### Anthropic (Claude)
```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0.7,
)
```

### Google (Gemini)
```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
)
```

---

## Key Parameters Explained

### Temperature (0.0 to 2.0)
Controls **randomness** of the output.

| Value | Behavior | Use Case |
|-------|----------|----------|
| 0.0 | Deterministic, picks most probable token | Factual Q&A, code generation, math |
| 0.3-0.5 | Slightly varied, mostly focused | Summaries, analysis |
| 0.7 | Balanced creativity | General chatbot, writing |
| 1.0+ | Highly creative, unpredictable | Brainstorming, poetry, fiction |

**Rule of thumb:** Use `0` for anything that needs to be correct. Use `0.7` for anything that needs to be interesting.

### Max Tokens
- Controls the **maximum length** of the response
- Does NOT guarantee the response will be that long
- If not set, model uses its default max
- 1 token ≈ 4 characters in English

### Top-P (Nucleus Sampling)
- Alternative to temperature
- `top_p=0.1` means only consider tokens in the top 10% probability
- Usually don't change both temperature AND top_p

### Frequency Penalty & Presence Penalty
- `frequency_penalty` (−2 to 2): Penalizes tokens based on how often they appeared
- `presence_penalty` (−2 to 2): Penalizes tokens that have appeared at all
- Useful to reduce repetition

---

## Invocation Methods

### `.invoke()` — Single call, wait for full response
```python
response = llm.invoke("Hello!")
# Returns: AIMessage(content="Hi there!", ...)
```

### `.stream()` — Token-by-token streaming
```python
for chunk in llm.stream("Tell me a story"):
    print(chunk.content, end="")
# Each chunk is an AIMessageChunk with a few tokens
```

### `.batch()` — Multiple inputs in parallel
```python
responses = llm.batch([
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?",
])
# Returns: [AIMessage, AIMessage, AIMessage]
```

### `.ainvoke()` — Async version
```python
import asyncio

async def main():
    response = await llm.ainvoke("Hello!")
    print(response.content)

asyncio.run(main())
```

### `.astream()` — Async streaming
```python
async for chunk in llm.astream("Tell me a story"):
    print(chunk.content, end="")
```

---

## Response Object (AIMessage)

When you call `.invoke()`, you get back an `AIMessage`:

```python
response = llm.invoke("What is AI?")

response.content          # "AI is..." (the text)
response.response_metadata  # Model info, finish reason
response.usage_metadata     # Token counts
response.id               # Unique response ID
```

### Token Usage
```python
response.usage_metadata
# {'input_tokens': 12, 'output_tokens': 45, 'total_tokens': 57}
```

This is critical for **cost tracking**:
- GPT-4o-mini: ~$0.15 / 1M input tokens, ~$0.60 / 1M output tokens
- GPT-4o: ~$2.50 / 1M input tokens, ~$10 / 1M output tokens

---

## Binding Parameters (Model Configuration)

You can pre-configure model parameters:

```python
# Create a model that always returns JSON
json_llm = llm.bind(response_format={"type": "json_object"})

# Create a model with specific stop sequences
limited_llm = llm.bind(stop=["\n\n"])  # Stop at double newline

# Create a model bound to specific tools
tool_llm = llm.bind_tools([my_tool])
```

### Why Bind?
- Keeps chain definitions clean
- Reusable configurations
- Separates model setup from chain logic

---

## Error Handling

### Common Errors
| Error | Cause | Fix |
|-------|-------|-----|
| `AuthenticationError` | Invalid API key | Check `.env` file |
| `RateLimitError` | Too many requests | Add retry/backoff |
| `InvalidRequestError` | Token limit exceeded | Reduce input size |
| `Timeout` | Slow response | Increase timeout |

### Built-in Retry
```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    max_retries=3,        # Retries on transient errors
    timeout=60,           # Seconds before timeout
)
```

### Manual Fallback (Preview — covered in Phase 2)
```python
llm_with_fallback = llm.with_fallbacks([backup_llm])
```

---

## Best Practices

1. **Always use environment variables for API keys** — never hardcode
2. **Set temperature=0 for deterministic tasks** — testing, code gen, extraction
3. **Track token usage** — costs add up fast in production
4. **Use the cheapest model that works** — start with gpt-4o-mini, upgrade only if needed
5. **Enable streaming for UX** — users hate waiting 5 seconds for a blank response
6. **Use `.bind()` over repeating params** — cleaner code

---

## Mental Model

Think of a Chat Model as a **function**:

```
f(messages: List[Message]) → AIMessage
```

- Input: ordered list of messages (system + history + current question)
- Output: one AI response message
- The model has NO memory — you must pass the full conversation every time
- This is why "memory" in LangChain is about managing that message list (Phase 4)

---

## Cost Cheat Sheet (May 2026 approximate)

| Model | Input $/1M tokens | Output $/1M tokens | Best For |
|-------|-------------------|--------------------:|----------|
| gpt-4o-mini | $0.15 | $0.60 | Learning, simple tasks |
| gpt-4o | $2.50 | $10.00 | Complex reasoning |
| claude-sonnet | $3.00 | $15.00 | Long documents, coding |
| gemini-1.5-flash | $0.075 | $0.30 | High volume, cheap |


