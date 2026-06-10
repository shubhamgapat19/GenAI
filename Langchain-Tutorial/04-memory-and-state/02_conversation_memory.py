"""
Phase 4: Memory & State — Conversation Memory Strategies
==========================================================
Different strategies for managing conversation length.

Topics covered:
- Buffer Memory (full history)
- Window Memory (last N messages)
- Token Buffer (by token count)
- Summary Memory (LLM-generated summaries)
- Combined approaches
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ============================================================
# 1. The Problem: Context Window Limits
# ============================================================

# A 10-message conversation ≈ fine
# A 100-message conversation ≈ 50K tokens ≈ expensive
# A 1000-message conversation ≈ exceeds context window

# Solution: Memory strategies that keep conversations manageable

print("=== The Memory Problem ===")
print("Full history:   Accurate but expensive, eventually exceeds context window")
print("Window (last N): Cheap but forgets old info")
print("Summary:        Compact but loses detail")
print("Combined:       Best of both worlds")
print()

# ============================================================
# 2. Buffer Memory (Keep Everything)
# ============================================================

# Simplest approach: just keep all messages
buffer_history = [
    HumanMessage(content="Hi, I'm Shubham. I'm building a logistics app."),
    AIMessage(content="Hello Shubham! A logistics app sounds interesting. What technology stack are you using?"),
    HumanMessage(content="NestJS backend, Next.js frontend, PostgreSQL database."),
    AIMessage(content="Great stack! NestJS is excellent for microservices. How can I help you today?"),
    HumanMessage(content="I need help with real-time GPS tracking."),
    AIMessage(content="For real-time GPS tracking, you'll want WebSockets with NestJS. Use the @nestjs/websockets package with Socket.IO for reliable real-time communication."),
    HumanMessage(content="What about the database schema for storing locations?"),
    AIMessage(content="Use PostGIS extension for PostgreSQL. Store: vehicle_id, latitude, longitude, timestamp, speed, heading. Create a spatial index for geo-queries."),
]

print("=== Buffer Memory (Full History) ===")
print(f"Messages: {len(buffer_history)}")
print(f"Pros: Complete context, no information loss")
print(f"Cons: Grows unbounded, expensive with long conversations")
print()

# ============================================================
# 3. Window Memory (Last N Messages)
# ============================================================

def window_memory(messages: list, window_size: int = 6) -> list:
    """Keep only the last N messages."""
    return messages[-window_size:]

windowed = window_memory(buffer_history, window_size=4)

print("=== Window Memory (last 4 messages) ===")
for msg in windowed:
    role = "Human" if isinstance(msg, HumanMessage) else "AI"
    print(f"  [{role}] {msg.content[:60]}...")
print(f"\nLost: First {len(buffer_history) - 4} messages (name, stack info)")
print()

# ============================================================
# 4. Using trim_messages (LangChain's built-in)
# ============================================================

# trim_messages provides flexible trimming by token count
trimmed = trim_messages(
    buffer_history,
    max_tokens=200,                    # Max tokens to keep
    strategy="last",                   # Keep last messages ("first" = keep first)
    token_counter=llm,                 # Use LLM's tokenizer to count
    include_system=True,               # Always keep system message if present
    allow_partial=False,               # Don't cut messages in half
)

print("=== trim_messages ===")
print(f"Original: {len(buffer_history)} messages")
print(f"Trimmed: {len(trimmed)} messages")
for msg in trimmed:
    role = "Human" if isinstance(msg, HumanMessage) else "AI"
    print(f"  [{role}] {msg.content[:60]}...")
print()

# ============================================================
# 5. Summary Memory (LLM summarizes old messages)
# ============================================================

from langchain_core.messages import SystemMessage

async def summarize_history(messages: list, llm) -> str:
    """Use LLM to create a summary of conversation history."""
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following conversation in 2-3 sentences. Capture key facts: user's name, preferences, and what they're working on."),
        MessagesPlaceholder("messages"),
    ])
    
    chain = summary_prompt | llm | StrOutputParser()
    summary = await chain.ainvoke({"messages": messages})
    return summary

# Synchronous version for demonstration
def summarize_history_sync(messages: list) -> str:
    """Create a summary of conversation history."""
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Distill the conversation into key facts (2-3 sentences). Include: user identity, their project, technology choices, and current needs."),
        MessagesPlaceholder("messages"),
    ])
    chain = summary_prompt | llm | StrOutputParser()
    return chain.invoke({"messages": messages})

summary = summarize_history_sync(buffer_history)
print("=== Summary Memory ===")
print(f"Summary of {len(buffer_history)} messages:")
print(f"  {summary}")
print()

# ============================================================
# 6. Combined: Summary + Recent Messages
# ============================================================

def combined_memory(messages: list, recent_count: int = 4) -> list:
    """Keep summary of old messages + recent messages in full."""
    if len(messages) <= recent_count:
        return messages
    
    # Summarize old messages
    old_messages = messages[:-recent_count]
    summary = summarize_history_sync(old_messages)
    
    # Combine: system summary + recent full messages
    combined = [
        SystemMessage(content=f"Previous conversation summary: {summary}"),
    ] + messages[-recent_count:]
    
    return combined

combined = combined_memory(buffer_history, recent_count=4)

print("=== Combined Memory (Summary + Last 4) ===")
for msg in combined:
    role = msg.__class__.__name__.replace("Message", "")
    print(f"  [{role}] {msg.content[:80]}...")
print()

# ============================================================
# 7. Chain with Memory Strategy
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful technical assistant.{summary}"),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

class ConversationWithMemory:
    """Conversation that uses combined memory strategy."""
    
    def __init__(self, window_size: int = 6):
        self.full_history: list = []
        self.window_size = window_size
        self.summary = ""
    
    def chat(self, user_input: str) -> str:
        """Process a message with memory management."""
        # Build the effective history
        if len(self.full_history) > self.window_size:
            # Summarize old messages
            old = self.full_history[:-self.window_size]
            self.summary = summarize_history_sync(old)
            effective_history = self.full_history[-self.window_size:]
        else:
            effective_history = self.full_history
        
        # Generate response
        summary_text = f"\n\nPrevious context: {self.summary}" if self.summary else ""
        response = chain.invoke({
            "summary": summary_text,
            "history": effective_history,
            "input": user_input,
        })
        
        # Update full history
        self.full_history.append(HumanMessage(content=user_input))
        self.full_history.append(AIMessage(content=response))
        
        return response

# Demo
print("=== Conversation with Memory Management ===")
conv = ConversationWithMemory(window_size=4)
print(f"[Turn 1] User: I'm Shubham, building a GPS logistics platform")
print(f"[Turn 1] AI: {conv.chat('I am Shubham, building a GPS logistics platform')}\n")
print(f"[Turn 2] User: I use NestJS and PostgreSQL")
print(f"[Turn 2] AI: {conv.chat('I use NestJS and PostgreSQL')}\n")
print(f"[Turn 3] User: How should I structure my tracking service?")
print(f"[Turn 3] AI: {conv.chat('How should I structure my tracking service?')}\n")
