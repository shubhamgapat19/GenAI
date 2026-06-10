"""
Phase 2: Chains & Advanced LCEL — RunnableParallel
====================================================
Run multiple chains simultaneously and combine results.

Topics covered:
- RunnableParallel for concurrent execution
- Dictionary shorthand
- Combining parallel results
- Performance benefits
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ============================================================
# 1. Basic RunnableParallel — Run multiple chains at once
# ============================================================

# Three independent chains that process the same input
summary_chain = (
    ChatPromptTemplate.from_messages([("human", "Summarize {topic} in 2 sentences.")])
    | llm | StrOutputParser()
)

pros_chain = (
    ChatPromptTemplate.from_messages([("human", "List 3 advantages of {topic}.")])
    | llm | StrOutputParser()
)

cons_chain = (
    ChatPromptTemplate.from_messages([("human", "List 3 disadvantages of {topic}.")])
    | llm | StrOutputParser()
)

# Run all three in parallel
parallel_chain = RunnableParallel(
    summary=summary_chain,
    pros=pros_chain,
    cons=cons_chain,
)

print("=== RunnableParallel ===")
result = parallel_chain.invoke({"topic": "microservices architecture"})
print(f"Summary: {result['summary'][:100]}...")
print(f"Pros: {result['pros'][:100]}...")
print(f"Cons: {result['cons'][:100]}...")
print()

# ============================================================
# 2. Dictionary Shorthand (equivalent to RunnableParallel)
# ============================================================

# This dict IS a RunnableParallel under the hood
parallel_with_dict = {
    "summary": summary_chain,
    "pros": pros_chain,
    "cons": cons_chain,
}

# Use it as input to another chain
combine_prompt = ChatPromptTemplate.from_messages([
    ("human", """Based on this analysis of {{topic}}:

Summary: {summary}
Pros: {pros}
Cons: {cons}

Give a final verdict in one sentence."""),
])

# Full pipeline: parallel analysis → combine → verdict
full_chain = (
    parallel_with_dict
    | combine_prompt
    | llm
    | StrOutputParser()
)

print("=== Dict Shorthand → Combined ===")
verdict = full_chain.invoke({"topic": "microservices architecture"})
print(f"Verdict: {verdict}")
print()

# ============================================================
# 3. Passing Original Input Alongside Parallel Results
# ============================================================

# Common pattern: keep original input available for later steps
chain_with_context = (
    {
        "topic": RunnablePassthrough(),   # Original input passes through
        "analysis": summary_chain,         # Parallel processing
    }
    | ChatPromptTemplate.from_messages([
        ("human", "Topic: {topic}\nAnalysis: {analysis}\n\nSuggest 3 related topics to explore.")
    ])
    | llm
    | StrOutputParser()
)

print("=== Passthrough + Parallel ===")
result = chain_with_context.invoke({"topic": "LangChain"})
print(result)
print()

# ============================================================
# 4. Nested Parallel (parallel within parallel)
# ============================================================

technical_analysis = RunnableParallel(
    complexity=ChatPromptTemplate.from_messages([
        ("human", "Rate the learning complexity of {topic} from 1-10. Just the number.")
    ]) | llm | StrOutputParser(),
    prerequisites=ChatPromptTemplate.from_messages([
        ("human", "List 3 prerequisites for learning {topic}.")
    ]) | llm | StrOutputParser(),
)

career_analysis = RunnableParallel(
    jobs=ChatPromptTemplate.from_messages([
        ("human", "List 3 job roles that use {topic}.")
    ]) | llm | StrOutputParser(),
    salary=ChatPromptTemplate.from_messages([
        ("human", "What's the average salary range for {topic} experts? One line.")
    ]) | llm | StrOutputParser(),
)

# Top-level parallel runs two parallel groups
full_analysis = RunnableParallel(
    technical=technical_analysis,
    career=career_analysis,
)

print("=== Nested Parallel ===")
result = full_analysis.invoke({"topic": "LangChain"})
print(f"Technical: {result['technical']}")
print(f"Career: {result['career']}")
