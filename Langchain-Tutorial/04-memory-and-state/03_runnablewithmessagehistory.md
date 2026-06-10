# RunnableWithMessageHistory — Deep Dive Notes

## What Is It?

`RunnableWithMessageHistory` is LangChain's **official, modern way** to add memory to any chain. It wraps an existing chain and automatically:
1. Loads history before each call
2. Passes it to the chain
3. Saves the new messages after each call

```
Before: You manually load/save/pass history every time
After:  Just call chain.invoke() — history is handled automatically
```

---

## The 4-Step Pattern

### Step 1: Create the Base Chain
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),   # ← Required placeholder
    ("human", "{input}"),
])
chain = prompt | llm | StrOutputParser()
```

### Step 2: Define History Getter
```python
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
```

### Step 3: Wrap the Chain
```python
from langchain_core.runnables.history import RunnableWithMessageHistory

chain_with_history = RunnableWithMessageHistory(
    chain,                              # Your base chain
    get_session_history,                 # How to get/create history
    input_messages_key="input",          # Which key has user message
    history_messages_key="history",      # Which placeholder gets history
)
```

### Step 4: Use It
```python
config = {"configurable": {"session_id": "user_123"}}
response = chain_with_history.invoke({"input": "Hello!"}, config=config)
```

---

## Key Parameters

| Parameter | Purpose | Required? |
|-----------|---------|-----------|
| `runnable` | The base chain to wrap | Yes |
| `get_session_history` | Function returning history for a session | Yes |
| `input_messages_key` | Key in input dict that has the user message | Yes |
| `history_messages_key` | Prompt placeholder name for history | Yes (if using MessagesPlaceholder) |
| `output_messages_key` | Key in output to save as AI message | No (auto-detected) |

---

## Session Configuration

The `session_id` is passed via the `config` parameter:

```python
# Each session gets isolated history
config = {"configurable": {"session_id": "session_abc123"}}
```

### Multi-key Configuration
You can use multiple identifiers:

```python
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(id="user_id", annotation=str),
        ConfigurableFieldSpec(id="conversation_id", annotation=str),
    ],
)

# Usage:
config = {"configurable": {"user_id": "user_1", "conversation_id": "conv_5"}}
```

---

## Common Patterns

### Pattern 1: Basic Chatbot
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
chain = prompt | llm | StrOutputParser()
```

### Pattern 2: RAG + Memory
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer from context:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
# Context comes from retriever, history is auto-managed
```

### Pattern 3: Structured Output + Memory
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract entities from the conversation."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
chain = prompt | llm.with_structured_output(EntitySchema)
```

---

## Streaming Support

Works seamlessly with `.stream()` and `.astream()`:

```python
for chunk in chain_with_history.stream(
    {"input": "Explain RAG in detail."},
    config={"configurable": {"session_id": "s1"}},
):
    print(chunk, end="", flush=True)
```

History is saved AFTER the full response is generated.

---

## Trimming Before Sending

Combine with `trim_messages` for cost control:

```python
from langchain_core.messages import trim_messages

def get_trimmed_history(session_id):
    history = get_full_history(session_id)
    # Only return last 10 messages to the chain
    trimmed = history.messages[-10:]
    # Return a ChatMessageHistory with trimmed messages
    trimmed_history = ChatMessageHistory()
    for msg in trimmed:
        trimmed_history.add_message(msg)
    return trimmed_history
```

Or trim within the chain itself using `RunnablePassthrough.assign`.

---

## When to Use vs. Manual History

| Scenario | Approach |
|----------|----------|
| Standard chatbot | `RunnableWithMessageHistory` ✓ |
| Custom state beyond messages | Manual or LangGraph |
| Simple prototype | Manual (fewer abstractions) |
| Multi-user production app | `RunnableWithMessageHistory` ✓ |
| Complex agent workflows | LangGraph (more control) |
| Need to modify history (summarize, filter) | Manual or custom get_session_history |

---

## Best Practices

1. **Always pass `config` with session_id** — forgetting it causes shared state bugs
2. **Use persistent backends in production** — SQLite, Redis, or Postgres
3. **Implement TTL for sessions** — don't store history forever
4. **Test session isolation** — verify User A can't see User B's history
5. **Monitor history size** — log message counts per session
6. **Use consistent session IDs** — typically `user_id + conversation_id`
7. **Handle errors gracefully** — what if history backend is down?

---

## Debugging Tips

```python
# Check what's in history
history = get_session_history("session_123")
for msg in history.messages:
    print(f"{msg.__class__.__name__}: {msg.content[:50]}")

# Check if history is being used
# Add a temporary print in get_session_history:
def get_session_history(session_id):
    print(f"Loading history for: {session_id}")
    ...
```

---

## Migration from Legacy Memory

Old LangChain (v0.1-0.2) used `ConversationBufferMemory`, `ConversationSummaryMemory`, etc.
These are **deprecated** in v0.3+.

```python
# OLD (deprecated):
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()

# NEW (current):
from langchain_core.runnables.history import RunnableWithMessageHistory
# + ChatMessageHistory or any persistent backend
```
