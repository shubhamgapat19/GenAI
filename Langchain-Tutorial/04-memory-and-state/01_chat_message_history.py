"""
Phase 4: Memory & State — Chat Message History
=================================================
Store and manage conversation history.

Topics covered:
- Message types (Human, AI, System)
- ChatMessageHistory (in-memory)
- Persisting history to files/databases
- Trimming and managing long histories
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ============================================================
# 1. Message Types in LangChain
# ============================================================

# LangChain uses typed messages for conversation
messages = [
    SystemMessage(content="You are a helpful Python tutor."),
    HumanMessage(content="What is a list comprehension?"),
    AIMessage(content="A list comprehension is a concise way to create lists: [x**2 for x in range(10)]"),
    HumanMessage(content="Can you show a filtered example?"),
]

response = llm.invoke(messages)
print("=== Message Types ===")
print(f"Response: {response.content}")
print()

# ============================================================
# 2. ChatMessageHistory (in-memory)
# ============================================================

from langchain_community.chat_message_histories import ChatMessageHistory

# Create a history store
history = ChatMessageHistory()

# Add messages
history.add_user_message("Hi, I'm learning LangChain!")
history.add_ai_message("Welcome! LangChain is a great framework. What would you like to know?")
history.add_user_message("How does memory work?")
history.add_ai_message("Memory in LangChain stores conversation history so the AI can reference previous messages.")

print("=== ChatMessageHistory ===")
print(f"Total messages: {len(history.messages)}")
for msg in history.messages:
    role = msg.__class__.__name__.replace("Message", "")
    print(f"  [{role}] {msg.content[:60]}...")
print()

# Clear history
# history.clear()

# ============================================================
# 3. Using History in a Chain
# ============================================================

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the conversation history to provide context-aware answers."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

# Simulate a conversation
conversation_history = []

def chat(user_input: str) -> str:
    """Send a message and maintain history."""
    response = chain.invoke({
        "history": conversation_history,
        "input": user_input,
    })
    # Update history
    conversation_history.append(HumanMessage(content=user_input))
    conversation_history.append(AIMessage(content=response))
    return response

print("=== Conversation with History ===")
print(f"User: My name is Shubham")
print(f"AI: {chat('My name is Shubham')}")
print()
print(f"User: What's my name?")
print(f"AI: {chat('What is my name?')}")  # Should remember!
print()
print(f"User: I'm learning LangChain for production apps")
print(f"AI: {chat('I am learning LangChain for production apps')}")
print()
print(f"User: Based on what I told you, suggest a project")
print(f"AI: {chat('Based on what I told you, suggest a project for me')}")
print()

# ============================================================
# 4. File-based History (persists across restarts)
# ============================================================

from langchain_community.chat_message_histories import FileChatMessageHistory

# Each session gets its own file
file_history = FileChatMessageHistory("chat_history_session1.json")

# Add messages (persisted to disk immediately)
file_history.add_user_message("Tell me about embeddings")
file_history.add_ai_message("Embeddings are numerical representations of text in high-dimensional space.")

print("=== File-based History ===")
print(f"Messages saved to: chat_history_session1.json")
print(f"Stored messages: {len(file_history.messages)}")
print()

# ============================================================
# 5. Session-based History Management
# ============================================================

# In production, you have multiple users with separate histories
session_store: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Get or create history for a session."""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# User A
user_a_history = get_session_history("user_a")
user_a_history.add_user_message("I like Python")
user_a_history.add_ai_message("Python is great for AI development!")

# User B (completely separate)
user_b_history = get_session_history("user_b")
user_b_history.add_user_message("I prefer JavaScript")
user_b_history.add_ai_message("JavaScript is excellent for full-stack development!")

print("=== Session Management ===")
print(f"User A history: {len(session_store['user_a'].messages)} messages")
print(f"User B history: {len(session_store['user_b'].messages)} messages")
print(f"User A's preference: {session_store['user_a'].messages[0].content}")
print(f"User B's preference: {session_store['user_b'].messages[0].content}")
