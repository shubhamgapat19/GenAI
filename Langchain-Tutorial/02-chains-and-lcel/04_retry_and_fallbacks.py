"""
Phase 2: Chains & Advanced LCEL — Retry, Fallback & Error Handling
===================================================================
Build resilient chains that handle failures gracefully.

Topics covered:
- with_retry for transient failures
- with_fallbacks for model/chain fallbacks
- Exception handling in chains
- Rate limiting awareness
- Timeout handling
"""

import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableConfig

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================
# 1. with_retry — Auto-retry on transient failures
# ============================================================

chain = (
    ChatPromptTemplate.from_messages([("human", "Explain {topic} in one sentence.")])
    | llm
    | StrOutputParser()
)

# Add retry logic: retries up to 3 times with exponential backoff
resilient_chain = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,  # Random jitter prevents thundering herd
)

print("=== with_retry ===")
result = resilient_chain.invoke({"topic": "retry patterns"})
print(f"Result: {result}")
print()

# ============================================================
# 2. with_fallbacks — Use backup when primary fails
# ============================================================

# Primary: expensive, powerful model
primary_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Fallback: cheap, reliable model
fallback_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Chain with automatic fallback
robust_chain = (
    ChatPromptTemplate.from_messages([("human", "{input}")])
    | primary_llm.with_fallbacks([fallback_llm])
    | StrOutputParser()
)

print("=== with_fallbacks ===")
result = robust_chain.invoke({"input": "What is 2+2?"})
print(f"Result: {result}")
print()

# ============================================================
# 3. Chain-level Fallbacks (entire chain, not just model)
# ============================================================

# Complex chain that might fail
complex_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You must respond in valid JSON only. No other text."),
        ("human", "Analyze: {input}"),
    ])
    | llm
    | StrOutputParser()
    | RunnableLambda(lambda x: __import__("json").loads(x))  # Might fail!
)

# Simple fallback chain
simple_fallback = (
    ChatPromptTemplate.from_messages([
        ("human", "Briefly describe: {input}")
    ])
    | llm
    | StrOutputParser()
    | RunnableLambda(lambda x: {"result": x, "fallback": True})
)

# If JSON parsing fails, fall back to simple text response
safe_chain = complex_chain.with_fallbacks([simple_fallback])

print("=== Chain-level Fallback ===")
result = safe_chain.invoke({"input": "Python programming language"})
print(f"Result: {result}")
print()

# ============================================================
# 4. Custom Error Handling with RunnableLambda
# ============================================================

def safe_json_parse(text: str) -> dict:
    """Parse JSON with graceful error handling."""
    import json
    
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON from markdown code blocks
    import re
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Last resort: return raw text in a dict
    return {"raw_response": text, "parse_error": True}


chain_with_safe_parse = (
    ChatPromptTemplate.from_messages([
        ("human", "Return a JSON object with keys 'name' and 'age' for: {input}")
    ])
    | llm
    | StrOutputParser()
    | RunnableLambda(safe_json_parse)
)

print("=== Safe JSON Parse ===")
result = chain_with_safe_parse.invoke({"input": "A 25 year old named Alice"})
print(f"Result: {result}")
print()

# ============================================================
# 5. Timeout Handling
# ============================================================

# Set timeout at the model level
fast_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    timeout=10,       # 10 second timeout
    max_retries=2,    # Retry twice on timeout
)

fast_chain = (
    ChatPromptTemplate.from_messages([("human", "{input}")])
    | fast_llm
    | StrOutputParser()
)

print("=== Timeout Handling ===")
result = fast_chain.invoke({"input": "Hello!"})
print(f"Result: {result}")
print()

# ============================================================
# 6. Rate Limit Awareness — Batch with concurrency control
# ============================================================

chain = (
    ChatPromptTemplate.from_messages([("human", "One word that describes {topic}.")])
    | llm
    | StrOutputParser()
)

topics = [{"topic": f"topic_{i}"} for i in range(10)]

print("=== Batch with Concurrency Limit ===")
start = time.time()
results = chain.batch(
    topics,
    config={"max_concurrency": 3}  # Max 3 parallel API calls
)
elapsed = time.time() - start
print(f"Processed {len(results)} items in {elapsed:.1f}s (max 3 concurrent)")
for i, r in enumerate(results[:5]):
    print(f"  {topics[i]['topic']}: {r}")
print()

# ============================================================
# 7. Combining Retry + Fallback + Timeout (Production Pattern)
# ============================================================

def build_production_chain():
    """Build a chain with full production resilience."""
    
    # Primary model with timeout
    primary = ChatOpenAI(model="gpt-4o", timeout=15, max_retries=2)
    
    # Fallback model (faster, cheaper)
    fallback = ChatOpenAI(model="gpt-4o-mini", timeout=10, max_retries=2)
    
    # Robust model with fallback
    robust_model = primary.with_fallbacks([fallback])
    
    # Chain with retry at chain level
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
        ])
        | robust_model
        | StrOutputParser()
    ).with_retry(
        stop_after_attempt=2,
        wait_exponential_jitter=True,
    )
    
    return chain


production_chain = build_production_chain()
print("=== Production Chain (Retry + Fallback + Timeout) ===")
result = production_chain.invoke({"input": "What's the capital of France?"})
print(f"Result: {result}")
