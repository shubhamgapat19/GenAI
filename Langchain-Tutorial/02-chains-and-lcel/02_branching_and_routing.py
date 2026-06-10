"""
Phase 2: Chains & Advanced LCEL — Branching & Routing
======================================================
Route inputs to different chains based on conditions.

Topics covered:
- RunnableBranch for conditional logic
- Custom router functions
- Dynamic chain selection
- Fallback patterns
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ============================================================
# 1. RunnableBranch — If/else logic in chains
# ============================================================

# Different chains for different types of questions
code_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a coding expert. Provide code examples."),
        ("human", "{input}"),
    ])
    | llm | StrOutputParser()
)

math_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a math tutor. Show step-by-step solutions."),
        ("human", "{input}"),
    ])
    | llm | StrOutputParser()
)

general_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise."),
        ("human", "{input}"),
    ])
    | llm | StrOutputParser()
)

# Route based on keywords in the input
branch = RunnableBranch(
    # (condition, chain) pairs — first match wins
    (lambda x: any(kw in x["input"].lower() for kw in ["code", "python", "function", "class"]), code_chain),
    (lambda x: any(kw in x["input"].lower() for kw in ["math", "calculate", "equation", "solve"]), math_chain),
    general_chain,  # Default (no condition)
)

print("=== RunnableBranch ===")
print("Code question:")
print(branch.invoke({"input": "Write a Python function for fibonacci"}))
print()
print("Math question:")
print(branch.invoke({"input": "Solve: what is the integral of x^2?"}))
print()
print("General question:")
print(branch.invoke({"input": "What's the weather like on Mars?"}))
print()

# ============================================================
# 2. LLM-based Router (classify then route)
# ============================================================

# Step 1: Classify the input using an LLM
classifier_chain = (
    ChatPromptTemplate.from_messages([
        ("system", """Classify the user's question into one of these categories:
- code: programming, coding, technical implementation
- math: calculations, equations, statistics
- general: everything else

Respond with ONLY the category name, nothing else."""),
        ("human", "{input}"),
    ])
    | llm | StrOutputParser()
)

# Step 2: Route based on classification
def route_by_category(info: dict) -> str:
    """Route to appropriate chain based on classification."""
    category = info["category"].strip().lower()
    question = info["input"]
    
    if "code" in category:
        return code_chain.invoke({"input": question})
    elif "math" in category:
        return math_chain.invoke({"input": question})
    else:
        return general_chain.invoke({"input": question})


smart_router = (
    {
        "category": classifier_chain,
        "input": lambda x: x["input"],
    }
    | RunnableLambda(route_by_category)
)

print("=== LLM-based Router ===")
result = smart_router.invoke({"input": "How do I implement a binary search tree?"})
print(f"Routed response: {result[:200]}...")
print()

# ============================================================
# 3. Multi-step Routing (Route → Process → Format)
# ============================================================

# Different output formats based on request
def format_router(input_dict: dict) -> str:
    """Route to different formatting based on user preference."""
    text = input_dict["text"]
    fmt = input_dict.get("format", "paragraph")
    
    format_prompts = {
        "bullets": "Reformat this as bullet points:\n\n{text}",
        "summary": "Summarize this in one sentence:\n\n{text}",
        "eli5": "Explain this like I'm 5 years old:\n\n{text}",
        "paragraph": "Clean up this text:\n\n{text}",
    }
    
    template = format_prompts.get(fmt, format_prompts["paragraph"])
    chain = (
        ChatPromptTemplate.from_messages([("human", template)])
        | llm | StrOutputParser()
    )
    return chain.invoke({"text": text})


format_chain = RunnableLambda(format_router)

print("=== Format Router ===")
sample_text = "LangChain is a framework for building LLM apps. It has chains, agents, and RAG."

print("Bullets:")
print(format_chain.invoke({"text": sample_text, "format": "bullets"}))
print()
print("ELI5:")
print(format_chain.invoke({"text": sample_text, "format": "eli5"}))
print()

# ============================================================
# 4. Fallback Chains — Graceful degradation
# ============================================================

# Primary chain (might fail with complex model)
primary_chain = (
    ChatPromptTemplate.from_messages([
        ("human", "Answer in exactly 3 words: {input}")
    ])
    | ChatOpenAI(model="gpt-4o", temperature=0)  # Expensive model
    | StrOutputParser()
)

# Fallback chain (cheaper, always works)
fallback_chain = (
    ChatPromptTemplate.from_messages([
        ("human", "Answer briefly: {input}")
    ])
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Cheap model
    | StrOutputParser()
)

# If primary fails (rate limit, timeout, etc.), use fallback
robust_chain = primary_chain.with_fallbacks([fallback_chain])

print("=== Fallback Chain ===")
result = robust_chain.invoke({"input": "What is Python?"})
print(f"Result: {result}")
