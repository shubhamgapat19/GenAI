# Persistent Memory (Database-backed) — Deep Dive Notes

## Why Persist Memory?

In-memory history (`ChatMessageHistory`) is lost when:
- Program restarts
- Server redeploys
- Process crashes
- User returns hours/days later

**Production apps need persistent memory** — stored in a database.

---

## Backend Options

| Backend | Best For | Setup Complexity | Speed |
|---------|----------|-----------------|-------|
| **SQLite** | Single server, low traffic | Minimal (file) | Good |
| **PostgreSQL** | Multi-server, existing Postgres | Medium | Good |
| **Redis** | High traffic, TTL needed | Medium | Fastest |
| **MongoDB** | Document-oriented apps | Medium | Good |
| **DynamoDB** | AWS-native, serverless | Low (managed) | Good |
| **Custom** | Special requirements | High | Varies |

---

## SQLite Backend

Simplest persistent option. Zero infrastructure — just a file.

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat.db",
    )
```

**Pros:** No server needed, just a file. Great for development and single-server deployments.
**Cons:** Single-writer (no concurrent access). Not suitable for distributed systems.

### Table Schema (auto-created)
```sql
CREATE TABLE message_store (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    message JSON
);
```

---

## Redis Backend

Production standard for high-traffic apps.

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379",
        ttl=3600,  # Expire after 1 hour
    )
```

### Key Redis Features for Chat
- **TTL (Time-to-Live):** Auto-delete old sessions → no cleanup code needed
- **Sub-millisecond:** Doesn't add latency to LLM calls
- **Distributed:** Works across multiple app servers
- **Pub/Sub:** Can notify when history updates (for real-time UI)

### When to Choose Redis
- High-traffic app (>100 concurrent users)
- Need TTL (auto-expiring sessions)
- Running multiple app instances
- Low-latency requirement

---

## PostgreSQL Backend

When you already have Postgres in your stack.

```python
from langchain_community.chat_message_histories import PostgresChatMessageHistory

def get_history(session_id: str):
    return PostgresChatMessageHistory(
        session_id=session_id,
        connection_string="postgresql://user:pass@localhost:5432/chatdb",
    )
```

### Advantages of Postgres for Chat
- **SQL queries on history:** "Show me all sessions where user asked about pricing"
- **JOIN with user tables:** Link conversations to user profiles
- **ACID transactions:** Guaranteed consistency
- **Already in stack:** No new infrastructure

### Advanced: Custom Queries
```sql
-- Find all sessions for a user
SELECT DISTINCT session_id FROM message_store 
WHERE message->>'additional_kwargs' LIKE '%user_id_123%';

-- Count messages per session
SELECT session_id, COUNT(*) as msg_count 
FROM message_store GROUP BY session_id;

-- Delete old sessions
DELETE FROM message_store 
WHERE created_at < NOW() - INTERVAL '30 days';
```

---

## Custom Backend Implementation

Implement `BaseChatMessageHistory` for any storage:

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

class MyCustomHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
    
    @property
    def messages(self) -> list[BaseMessage]:
        """Load messages from your storage."""
        raw_data = my_db.load(self.session_id)  # Your read logic
        return messages_from_dict(raw_data)
    
    def add_message(self, message: BaseMessage) -> None:
        """Save a single message."""
        data = messages_to_dict([message])[0]
        my_db.append(self.session_id, data)  # Your write logic
    
    def clear(self) -> None:
        """Delete all messages for this session."""
        my_db.delete(self.session_id)  # Your delete logic
```

### Required Methods
| Method | Purpose |
|--------|---------|
| `messages` (property) | Return all stored messages |
| `add_message(msg)` | Append one message |
| `clear()` | Delete all messages |

---

## Production Patterns

### Pattern 1: User-scoped Sessions
```python
def get_history(session_id: str):
    # session_id format: "user_123_conv_456"
    user_id, conv_id = session_id.split("_conv_")
    return PostgresChatMessageHistory(
        session_id=session_id,
        connection_string=DB_URL,
    )
```

### Pattern 2: TTL + Archival
```python
class ProductionHistory:
    def __init__(self, session_id):
        # Active in Redis (fast, TTL)
        self.active = RedisChatMessageHistory(session_id, ttl=3600)
        # Archive in Postgres (permanent, queryable)
        self.archive = PostgresChatMessageHistory(session_id)
    
    def add_message(self, msg):
        self.active.add_message(msg)
        self.archive.add_message(msg)  # Dual-write
```

### Pattern 3: Limit History Size
```python
class BoundedHistory(BaseChatMessageHistory):
    MAX_MESSAGES = 100
    
    def add_message(self, message):
        super().add_message(message)
        if len(self.messages) > self.MAX_MESSAGES:
            # Keep only last N messages
            self._trim()
```

### Pattern 4: Encryption for PII
```python
class EncryptedHistory(BaseChatMessageHistory):
    def add_message(self, message):
        encrypted = encrypt(message.content)
        # Store encrypted version
    
    @property
    def messages(self):
        raw = load_from_db()
        return [decrypt(msg) for msg in raw]
```

---

## Choosing a Backend

```
Single server, development? → SQLite
Already using Postgres?     → PostgreSQL
High traffic, need TTL?     → Redis
AWS serverless?             → DynamoDB
Need full-text search?      → MongoDB or Elasticsearch
Custom requirements?        → Build your own (BaseChatMessageHistory)
```

---

## Best Practices

1. **Use Redis for production web apps** — fastest, TTL built-in
2. **Use Postgres if you need analytics** — SQL queries on conversations
3. **Always implement TTL or cleanup** — don't store history forever
4. **Encrypt sensitive conversations** — PII, medical, financial data
5. **Monitor storage growth** — alert when DB size exceeds threshold
6. **Backup conversation data** — it's often valuable for fine-tuning
7. **Consider GDPR** — users may request deletion of their history
8. **Test failover** — what happens when the DB is down?

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| No TTL on Redis | Memory keeps growing | Set `ttl=3600` or similar |
| SQLite in multi-server | Write conflicts | Switch to Postgres/Redis |
| Storing passwords in history | Security breach | Sanitize before storing |
| No connection pooling | DB connection exhaustion | Use SQLAlchemy pooling |
| Unbounded history | Slow loads, high cost | Implement max message limit |
| No index on session_id | Slow lookups | Add database index |
