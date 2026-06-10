"""
Phase 4: Memory & State — Persistent Memory (Database-backed)
===============================================================
Store conversation history in databases for production.

Topics covered:
- SQLite-backed history
- Redis-backed history
- PostgreSQL-backed history
- Custom memory backends
- Production patterns
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ============================================================
# 1. SQLite-backed History (simple persistence)
# ============================================================

from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_sqlite_history(session_id: str):
    """Get history stored in SQLite database."""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat_history.db",  # File-based DB
    )

# Create chain with SQLite persistence
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Remember details about the user."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

sqlite_chain = RunnableWithMessageHistory(
    chain,
    get_sqlite_history,
    input_messages_key="input",
    history_messages_key="history",
)

print("=== SQLite-backed History ===")
config = {"configurable": {"session_id": "user_123"}}

# These messages persist across program restarts!
response = sqlite_chain.invoke(
    {"input": "Hi, I'm Shubham. I'm building a logistics SaaS."},
    config=config,
)
print(f"User: Hi, I'm Shubham. I'm building a logistics SaaS.")
print(f"AI: {response}\n")

response = sqlite_chain.invoke(
    {"input": "What's my name and what am I building?"},
    config=config,
)
print(f"User: What's my name and what am I building?")
print(f"AI: {response}\n")

# Check stored messages
history = get_sqlite_history("user_123")
print(f"Messages in DB: {len(history.messages)}")
print()

# ============================================================
# 2. Redis-backed History (fast, production-grade)
# ============================================================

# Redis is ideal for production: fast, TTL support, distributed

# from langchain_community.chat_message_histories import RedisChatMessageHistory
#
# def get_redis_history(session_id: str):
#     return RedisChatMessageHistory(
#         session_id=session_id,
#         url="redis://localhost:6379",
#         ttl=3600,  # Auto-expire after 1 hour
#     )
#
# redis_chain = RunnableWithMessageHistory(
#     chain,
#     get_redis_history,
#     input_messages_key="input",
#     history_messages_key="history",
# )

print("=== Redis-backed History (pattern) ===")
print("RedisChatMessageHistory features:")
print("  - Sub-millisecond reads/writes")
print("  - TTL: auto-expire old conversations")
print("  - Distributed: works across multiple servers")
print("  - Production-ready for high-traffic apps")
print()

# ============================================================
# 3. PostgreSQL-backed History
# ============================================================

# PostgreSQL: when you already use Postgres and want one DB for everything

# from langchain_community.chat_message_histories import PostgresChatMessageHistory
#
# def get_postgres_history(session_id: str):
#     return PostgresChatMessageHistory(
#         session_id=session_id,
#         connection_string="postgresql://user:pass@localhost:5432/chatdb",
#     )

print("=== PostgreSQL-backed History (pattern) ===")
print("PostgresChatMessageHistory features:")
print("  - ACID transactions")
print("  - SQL queries on history (analytics)")
print("  - Joins with user tables")
print("  - Already in your stack (no new infra)")
print()

# ============================================================
# 4. Custom Memory Backend
# ============================================================

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
import json
from pathlib import Path


class JSONFileChatHistory(BaseChatMessageHistory):
    """Custom history backend that stores messages in JSON files."""
    
    def __init__(self, session_id: str, storage_dir: str = "./chat_sessions"):
        self.session_id = session_id
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.file_path = self.storage_dir / f"{session_id}.json"
    
    @property
    def messages(self) -> list[BaseMessage]:
        """Load messages from JSON file."""
        if not self.file_path.exists():
            return []
        with open(self.file_path, "r") as f:
            data = json.load(f)
        return messages_from_dict(data)
    
    def add_message(self, message: BaseMessage) -> None:
        """Append a message to the JSON file."""
        current = self.messages
        current.append(message)
        with open(self.file_path, "w") as f:
            json.dump(messages_to_dict(current), f, indent=2)
    
    def clear(self) -> None:
        """Delete all messages."""
        if self.file_path.exists():
            self.file_path.unlink()


def get_json_history(session_id: str) -> JSONFileChatHistory:
    return JSONFileChatHistory(session_id)


# Use custom backend
json_chain = RunnableWithMessageHistory(
    chain,
    get_json_history,
    input_messages_key="input",
    history_messages_key="history",
)

print("=== Custom JSON Backend ===")
config_json = {"configurable": {"session_id": "custom_session_01"}}

response = json_chain.invoke(
    {"input": "Store this: my favorite language is Python and I work on AI projects."},
    config=config_json,
)
print(f"AI: {response}\n")

response = json_chain.invoke(
    {"input": "What do you remember about me?"},
    config=config_json,
)
print(f"AI: {response}\n")

# ============================================================
# 5. Production Patterns
# ============================================================

print("=== Production Memory Patterns ===")

# Pattern 1: Session with user metadata
from dataclasses import dataclass

@dataclass
class UserSession:
    user_id: str
    session_id: str
    metadata: dict

def get_production_history(session_id: str):
    """Production-grade history with additional features."""
    history = SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///production_chat.db",
    )
    return history

# Pattern 2: Trim + persist (don't let DB grow unbounded)
class TrimmedPersistentHistory:
    """Keeps full history in DB but trims what's sent to LLM."""
    
    def __init__(self, session_id: str, max_messages: int = 20):
        self.db_history = SQLChatMessageHistory(
            session_id=session_id,
            connection="sqlite:///production_chat.db",
        )
        self.max_messages = max_messages
    
    def get_trimmed_messages(self):
        """Get last N messages for the LLM."""
        all_messages = self.db_history.messages
        return all_messages[-self.max_messages:]

print("Pattern 1: SQLite/Postgres for persistence")
print("Pattern 2: Redis for high-traffic (with TTL)")
print("Pattern 3: Trim before sending to LLM (cost control)")
print("Pattern 4: Custom backends for special requirements")
print("Pattern 5: User-scoped sessions (multi-tenant)")
