"""
Phase 1: Foundations — LCEL Basics (LangChain Expression Language)
==================================================================
Learn the pipe operator and how to compose chains.

Topics covered:
- Pipe operator (|) for chaining
- RunnablePassthrough
- RunnableLambda
- Chain invocation (invoke, batch, stream)
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ============================================================
# 1. The Pipe Operator — Building Chains
# ============================================================

# A chain is: prompt | llm | output_parser
# Data flows left to right through each component

simple_chain = (
    ChatPromptTemplate.from_messages([("human", "Tell me a joke about {topic}")])
    | llm
    | StrOutputParser()
)

# invoke() runs the chain with input
result = simple_chain.invoke({"topic": "programming"})
print("Simple chain:")
print(result)
print()

# ============================================================
# 2. Multi-step Chain
# ============================================================

# Step 1: Generate a joke
# Step 2: Explain why it's funny

joke_chain = (
    ChatPromptTemplate.from_messages([("human", "Tell me a short joke about {topic}")])
    | llm
    | StrOutputParser()
)

explain_chain = (
    ChatPromptTemplate.from_messages([
        ("human", "Explain why this joke is funny in one sentence:\n\n{joke}")
    ])
    | llm
    | StrOutputParser()
)

# Combine them: joke output feeds into explain input
full_chain = (
    {"joke": joke_chain, "topic": RunnablePassthrough()}
    | RunnableLambda(lambda x: {"joke": x["joke"]})
    | explain_chain
)

# Simpler approach — chain sequentially
print("Multi-step (sequential):")
joke = joke_chain.invoke({"topic": "Python"})
print(f"  Joke: {joke}")
explanation = explain_chain.invoke({"joke": joke})
print(f"  Why it's funny: {explanation}")
print()

# ============================================================
# 3. RunnablePassthrough — Pass input through unchanged
# ============================================================

# Useful when you need the original input alongside transformed data
chain_with_passthrough = (
    {
        "original_topic": RunnablePassthrough(),  # passes input as-is
        "uppercase_topic": RunnableLambda(lambda x: x.upper()),
    }
    | RunnableLambda(lambda x: f"Topic: {x['original_topic']} (formatted: {x['uppercase_topic']})")
)

result = chain_with_passthrough.invoke("langchain")
print("Passthrough example:")
print(result)
print()

# ============================================================
# 4. RunnableLambda — Custom transformation steps
# ============================================================

# Add custom Python logic anywhere in the chain
def word_count(text: str) -> str:
    """Add word count metadata."""
    count = len(text.split())
    return f"{text}\n\n[Word count: {count}]"


chain_with_lambda = (
    ChatPromptTemplate.from_messages([
        ("human", "Summarize {topic} in exactly 3 sentences.")
    ])
    | llm
    | StrOutputParser()
    | RunnableLambda(word_count)  # Post-processing step
)

result = chain_with_lambda.invoke({"topic": "LangChain"})
print("Chain with Lambda (word count added):")
print(result)
print()

# ============================================================
# 5. Batch Processing — Run multiple inputs at once
# ============================================================

topics = [
    {"topic": "Python"},
    {"topic": "JavaScript"},
    {"topic": "Rust"},
]

batch_chain = (
    ChatPromptTemplate.from_messages([
        ("human", "Describe {topic} in exactly 5 words.")
    ])
    | llm
    | StrOutputParser()
)

print("Batch processing (3 topics at once):")
results = batch_chain.batch(topics)
for topic, result in zip(topics, results):
    print(f"  {topic['topic']}: {result}")
print()

# ============================================================
# 6. Streaming — Token by token output
# ============================================================

stream_chain = (
    ChatPromptTemplate.from_messages([
        ("human", "Write a haiku about {topic}")
    ])
    | llm
    | StrOutputParser()
)

print("Streaming output:")
for chunk in stream_chain.stream({"topic": "coding"}):
    print(chunk, end="", flush=True)
print("\n")
