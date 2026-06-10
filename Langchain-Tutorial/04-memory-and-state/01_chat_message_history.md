# Chat Message History — Deep Dive Notes

## What Is Message History?

Message history is the **raw storage** of conversation messages. Without it, every LLM call is stateless — the AI has no memory of previous turns.

```
Turn 1: User: "My name is Shubham" → AI: "Nice to meet you!"
Turn 2: User: "What's my name?"    → AI: "I don't know" ← NO MEMORY!

With history:
Turn 2: User: "What's my name?"    → AI: "Your name is Shubham!" ← REMEMBERS!
```

---

## Message Types

| Type | Purpose | When Used |
|------|---------|-----------|
| `SystemMessage` | Instructions to the AI | Once at the start |
| `HumanMessage` | User's input | Every turn |
| `AIMessage` | AI's response | Every turn |
| `ToolMessage` | Tool execution result | When using tools/agents |
| `FunctionMessage` | Legacy function call result | Deprecated |

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

messages = [
    SystemMessage(content="You are a Python expert."),
    HumanMessage(content="What is a decorator?"),
    AIMessage(content="A decorator wraps a function to extend its behavior..."),
    HumanMessage(content="Show me an example."),  # Follows up with context
]
```

---

## ChatMessageHistory Class

The simplest in-memory history store:

```python
from langchain_community.chat_message_histories import ChatMessageHistory

history = ChatMessageHistory()
history.add_user_message("Hello!")
history.add_ai_message("Hi there!")

# Access all messages
history.messages  # [HumanMessage(...), AIMessage(...)]

# Clear everything
history.clear()
```

**Limitation:** In-memory only. Lost when program restarts.

---

## Session Management Pattern

In production, you manage histories per user/session:

```python
store: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# User A and User B have completely separate histories
history_a = get_session_history("user_a_session_1")
history_b = get_session_history("user_b_session_1")
```

---

## Using History in Chains

The key: use `MessagesPlaceholder` to inject history into prompts.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder("history"),  # ← Injects conversation history here
    ("human", "{input}"),
])
```

---

## File-based Persistence

For simple persistence across restarts:

```python
from langchain_community.chat_message_histories import FileChatMessageHistory

history = FileChatMessageHistory("session_123.json")
history.add_user_message("Remember this!")
# Immediately written to disk
# Survives program restart
```

---

## History Storage Backends

| Backend | Persistence | Speed | Use Case |
|---------|------------|-------|----------|
| `ChatMessageHistory` | None (RAM) | Fastest | Prototyping, testing |
| `FileChatMessageHistory` | Disk file | Fast | Single-user apps |
| `SQLChatMessageHistory` | SQLite/Postgres | Good | Multi-user, moderate traffic |
| `RedisChatMessageHistory` | Redis | Very fast | Production, high traffic |
| `MongoDBChatMessageHistory` | MongoDB | Good | Document-based apps |

---

## Best Practices

1. **Always scope history by session** — never share history between different users
2. **Use typed messages** — don't store raw strings, use `HumanMessage`/`AIMessage`
3. **Add metadata when relevant** — timestamps, user IDs, sources
4. **Clear history when conversation resets** — `history.clear()`
5. **Consider TTL (time-to-live)** — auto-expire old sessions in production
6. **Test with multi-turn conversations** — verify the AI actually uses history correctly

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Forgetting MessagesPlaceholder | History exists but isn't used | Add `MessagesPlaceholder("history")` to prompt |
| Shared history across users | User A sees User B's messages | Use session_id to isolate |
| In-memory in production | Lost on restart/deploy | Use persistent backend |
| No session cleanup | Memory leak / storage bloat | Implement TTL or manual cleanup |
| Storing sensitive data | Security/privacy risk | Encrypt or don't store PII |
