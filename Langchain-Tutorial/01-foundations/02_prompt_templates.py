"""
Phase 1: Foundations — Prompt Templates
========================================
Learn how to create reusable, parameterized prompts.

Topics covered:
- ChatPromptTemplate
- Variables and formatting
- MessagesPlaceholder for dynamic message lists
- Composing prompts
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ============================================================
# 1. Basic Prompt Template
# ============================================================

# Create a reusable prompt with variables
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {topic}. Explain concepts simply."),
    ("human", "{question}"),
])

# Format with variables
formatted = prompt.invoke({
    "topic": "machine learning",
    "question": "What is gradient descent?"
})

print("Formatted messages:")
for msg in formatted.messages:
    print(f"  [{msg.type}]: {msg.content[:80]}...")
print()

# Send to LLM
response = llm.invoke(formatted)
print("Response:")
print(response.content)
print()

# ============================================================
# 2. Template with Multiple Variables
# ============================================================

code_review_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior {language} developer doing code review."),
    ("human", """Review this code and provide feedback:
    
```{language}
{code}
```

Focus on: {focus_areas}"""),
])

response = llm.invoke(code_review_prompt.invoke({
    "language": "Python",
    "code": "def add(a,b): return a+b",
    "focus_areas": "type hints, docstrings, error handling"
}))

print("Code Review:")
print(response.content)
print()

# ============================================================
# 3. MessagesPlaceholder (for conversation history)
# ============================================================

# Useful when you want to inject a dynamic list of messages
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),  # Dynamic messages go here
    ("human", "{input}"),
])

# Simulate conversation history
history = [
    HumanMessage(content="My name is Shubham"),
]

formatted = chat_prompt.invoke({
    "history": history,
    "input": "What's my name?"
})

response = llm.invoke(formatted)
print("With history:")
print(response.content)
print()

# ============================================================
# 4. Prompt Composition (combining prompts)
# ============================================================

# You can build prompts from parts
system_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Your style is {style}."),
])

# Partial prompts — fix some variables early
partial_prompt = system_prompt.partial(style="concise and uses bullet points")

print("Partial prompt (style fixed):")
formatted = partial_prompt.invoke({"role": "data scientist"})
for msg in formatted.messages:
    print(f"  [{msg.type}]: {msg.content}")
