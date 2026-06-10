"""
Phase 4: Memory & State — RunnableWithMessageHistory
======================================================
LangChain's built-in way to add memory to any chain.

Topics covered:
- RunnableWithMessageHistory (the modern pattern)
- Session management
- Different history backends
- Configurable memory in production
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ============================================================
# 1. RunnableWithMessageHistory — The Modern Pattern
# ============================================================

# This wraps ANY chain to automatically manage message history
# No need to manually pass/update history — it's handled for you

# Step 1: Create the base chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding assistant. Be concise."),
    MessagesPlaceholder("history"),   # <-- History is injected here
    ("human", "{input}"),
])

base_chain = prompt | llm | StrOutputParser()

# Step 2: Create a session store
store: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Retrieve or create history for a session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Step 3: Wrap the chain with message history
chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",       # Which input key has the user message
    history_messages_key="history",   # Which placeholder to inject history into
)

# Step 4: Use it! (pass session_id in config)
print("=== RunnableWithMessageHistory ===")

# Session A
config_a = {"configurable": {"session_id": "session_a"}}

response1 = chain_with_history.invoke(
    {"input": "Hi! I'm working on a FastAPI project."},
    config=config_a,
)
print(f"[Session A] User: Hi! I'm working on a FastAPI project.")
print(f"[Session A] AI: {response1}\n")

response2 = chain_with_history.invoke(
    {"input": "How do I add authentication?"},
    config=config_a,
)
print(f"[Session A] User: How do I add authentication?")
print(f"[Session A] AI: {response2}\n")

# It remembers! Ask about previous context
response3 = chain_with_history.invoke(
    {"input": "What framework am I using again?"},
    config=config_a,
)
print(f"[Session A] User: What framework am I using again?")
print(f"[Session A] AI: {response3}\n")

# ============================================================
# 2. Multiple Sessions (multi-user)
# ============================================================

# Session B — completely independent
config_b = {"configurable": {"session_id": "session_b"}}

response_b = chain_with_history.invoke(
    {"input": "I'm building a React Native app. What state management do you recommend?"},
    config=config_b,
)
print(f"[Session B] User: I'm building a React Native app...")
print(f"[Session B] AI: {response_b}\n")

# Verify isolation
print("=== Session Isolation ===")
print(f"Session A messages: {len(store['session_a'].messages)}")
print(f"Session B messages: {len(store['session_b'].messages)}")
print()

# ============================================================
# 3. With Additional Context (RAG + Memory)
# ============================================================

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a documentation assistant. Use the provided context AND conversation history to answer.

Context from documents:
{context}"""),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

rag_chain = rag_prompt | llm | StrOutputParser()

# Wrap with history
rag_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

print("=== RAG + Memory Combined ===")
config_rag = {"configurable": {"session_id": "rag_session"}}

# First query with context
response = rag_with_history.invoke(
    {
        "input": "What is LangGraph?",
        "context": "LangGraph is LangChain's framework for building stateful, multi-agent applications using a graph-based approach with nodes and edges.",
    },
    config=config_rag,
)
print(f"Q: What is LangGraph?")
print(f"A: {response}\n")

# Follow-up (uses history to understand "it")
response = rag_with_history.invoke(
    {
        "input": "How is it different from regular LangChain chains?",
        "context": "LangChain chains are linear pipelines. LangGraph adds cycles, branching, and persistent state, enabling complex workflows like agent loops.",
    },
    config=config_rag,
)
print(f"Q: How is it different from regular LangChain chains?")
print(f"A: {response}\n")

# ============================================================
# 4. Streaming with History
# ============================================================

print("=== Streaming with History ===")
config_stream = {"configurable": {"session_id": "stream_session"}}

# First message to establish context
chain_with_history.invoke(
    {"input": "I'm interested in LangChain agents."},
    config=config_stream,
)

# Stream the follow-up
print("Streaming response to 'How do they work?':")
for chunk in chain_with_history.stream(
    {"input": "How do they work? Explain briefly."},
    config=config_stream,
):
    print(chunk, end="", flush=True)
print("\n")

# ============================================================
# 5. Trimming History in RunnableWithMessageHistory
# ============================================================

from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnablePassthrough

# Create a chain that trims before using history
def trim_history(messages):
    """Keep only last 10 messages."""
    return messages[-10:]

prompt_with_trim = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

# Chain that trims history before passing to prompt
trimmed_chain = (
    RunnablePassthrough.assign(
        history=lambda x: trim_history(x["history"])
    )
    | prompt_with_trim
    | llm
    | StrOutputParser()
)

trimmed_with_history = RunnableWithMessageHistory(
    trimmed_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

print("=== Trimmed History Chain ===")
config_trim = {"configurable": {"session_id": "trimmed_session"}}
response = trimmed_with_history.invoke(
    {"input": "Hello! This chain auto-trims history to last 10 messages."},
    config=config_trim,
)
print(f"Response: {response}")
