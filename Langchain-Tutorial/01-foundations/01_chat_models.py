"""
Phase 1: Foundations — Chat Models
===================================
Learn how to interact with LLMs using LangChain's Chat Model interface.

Topics covered:
- Setting up OpenAI / Azure OpenAI chat model
- Sending messages (System, Human, AI)
- Temperature and model parameters
- Streaming responses
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load API keys from .env file
load_dotenv()

# ============================================================
# 1. Basic Chat Model Setup
# ============================================================

# Initialize the model
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Cost-effective model for learning
    temperature=0.7,       # 0 = deterministic, 1 = creative
)

# Simple invocation with a string (auto-converted to HumanMessage)
response = llm.invoke("What is LangChain in one sentence?")
print("Simple response:")
print(response.content)
print(f"Tokens used: {response.usage_metadata}")
print()

# ============================================================
# 2. Message Types
# ============================================================

# LLMs understand different message roles
messages = [
    SystemMessage(content="You are a helpful AI tutor teaching LangChain. Be concise."),
    HumanMessage(content="What are the main components of LangChain?"),
]

response = llm.invoke(messages)
print("With system message:")
print(response.content)
print()

# ============================================================
# 3. Multi-turn Conversation (manually)
# ============================================================

conversation = [
    SystemMessage(content="You are a Python expert. Answer in 2-3 sentences max."),
    HumanMessage(content="What is LCEL?"),
    AIMessage(content="LCEL (LangChain Expression Language) is a declarative way to compose chains in LangChain using the pipe operator (|). It provides built-in streaming, async, and batch support."),
    HumanMessage(content="Give me a simple example"),
]

response = llm.invoke(conversation)
print("Multi-turn conversation:")
print(response.content)
print()

# ============================================================
# 4. Streaming Responses
# ============================================================

print("Streaming response:")
for chunk in llm.stream("Explain RAG in 3 bullet points"):
    print(chunk.content, end="", flush=True)
print("\n")

# ============================================================
# 5. Model Parameters
# ============================================================

# Creative model (high temperature)
creative_llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.0)

# Precise model (low temperature)
precise_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

print("Creative (temp=1.0):")
print(creative_llm.invoke("Write a one-line tagline for an AI company").content)
print()

print("Precise (temp=0.0):")
print(precise_llm.invoke("What is 2+2? Answer with just the number.").content)
