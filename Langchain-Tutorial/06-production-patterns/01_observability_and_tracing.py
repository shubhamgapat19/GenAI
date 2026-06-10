"""
Phase 6: Production Patterns — LangSmith Observability & Tracing
==================================================================
Monitor, debug, and trace LLM applications in production.

Topics covered:
- LangSmith setup and integration
- Tracing chains and agents
- Custom run metadata
- Feedback collection
- Production monitoring dashboards
"""

from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================
# 1. LangSmith Setup
# ============================================================

# LangSmith is LangChain's observability platform
# Set these env vars to enable automatic tracing:
#
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your_api_key
# LANGCHAIN_PROJECT=my_project_name

print("=== LangSmith Setup ===")
print("Environment variables needed:")
print("  LANGCHAIN_TRACING_V2=true")
print("  LANGCHAIN_API_KEY=<your key>")
print("  LANGCHAIN_PROJECT=<project name>")
print()
print("Once set, ALL LangChain calls are automatically traced!")
print()

# ============================================================
# 2. Automatic Tracing (zero-code-change)
# ============================================================

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# This chain is automatically traced when LANGCHAIN_TRACING_V2=true
chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ])
    | llm
    | StrOutputParser()
)

# Every invoke is logged: input, output, latency, tokens, cost
response = chain.invoke({"question": "What is LangSmith?"})
print(f"=== Auto-traced Response ===")
print(f"Response: {response[:100]}...")
print("(This call was automatically logged to LangSmith)")
print()

# ============================================================
# 3. Custom Metadata and Tags
# ============================================================

# Add metadata to help filter and search traces
response = chain.invoke(
    {"question": "Explain RAG in one sentence."},
    config={
        "metadata": {
            "user_id": "user_123",
            "session_id": "sess_abc",
            "environment": "production",
            "feature": "chat",
        },
        "tags": ["production", "chat", "v2.1"],
    }
)

print("=== Custom Metadata ===")
print("Added to trace: user_id, session_id, environment, feature")
print("Tags: production, chat, v2.1")
print("Use these to filter traces in LangSmith dashboard")
print()

# ============================================================
# 4. Named Runs (easier debugging)
# ============================================================

# Give your chains descriptive names
named_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Summarize the text in 2 sentences."),
        ("human", "{text}"),
    ])
    | llm
    | StrOutputParser()
).with_config(run_name="TextSummarizer")

response = named_chain.invoke({"text": "LangChain is a framework for building LLM apps. It provides tools for prompts, chains, agents, and retrieval. The framework supports multiple providers."})
print(f"=== Named Run ===")
print(f"Run name: 'TextSummarizer' (visible in LangSmith)")
print(f"Response: {response}")
print()

# ============================================================
# 5. Manual Tracing with @traceable
# ============================================================

from langsmith import traceable

@traceable(name="custom_rag_pipeline", tags=["rag", "v2"])
def my_rag_pipeline(question: str) -> dict:
    """A custom function that's traced in LangSmith."""
    # Simulate retrieval
    context = "LangChain was created by Harrison Chase in 2022."
    
    # Generate answer
    answer = chain.invoke({"question": f"Context: {context}\n\nQuestion: {question}"})
    
    return {
        "question": question,
        "context": context,
        "answer": answer,
    }

result = my_rag_pipeline("Who created LangChain?")
print(f"=== @traceable Function ===")
print(f"Answer: {result['answer'][:80]}...")
print("Traced as 'custom_rag_pipeline' with child spans for each LLM call")
print()

# ============================================================
# 6. Feedback Collection
# ============================================================

from langsmith import Client

# client = Client()  # Uses LANGCHAIN_API_KEY

# After getting a response, collect user feedback
# run_id comes from the traced run

# client.create_feedback(
#     run_id="run_id_from_trace",
#     key="user_rating",
#     score=1.0,        # 0.0 to 1.0
#     comment="Accurate and helpful response",
# )

# client.create_feedback(
#     run_id="run_id_from_trace",
#     key="correctness",
#     score=0.0,        # Bad response
#     comment="Hallucinated information",
# )

print("=== Feedback Collection ===")
print("Feedback types:")
print("  - user_rating: thumbs up/down (0 or 1)")
print("  - correctness: was the answer factually correct?")
print("  - helpfulness: did it solve the user's problem?")
print("  - toxicity: was the response inappropriate?")
print()

# ============================================================
# 7. Callbacks for Custom Logging
# ============================================================

from langchain_core.callbacks import BaseCallbackHandler
from datetime import datetime


class ProductionLogger(BaseCallbackHandler):
    """Custom callback for production logging."""
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"  [LOG] LLM call started at {datetime.now().isoformat()}")
    
    def on_llm_end(self, response, **kwargs):
        tokens = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        print(f"  [LOG] LLM call ended. Tokens: {tokens}")
    
    def on_llm_error(self, error, **kwargs):
        print(f"  [ERROR] LLM call failed: {error}")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"  [LOG] Tool '{serialized.get('name', '?')}' started")
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        pass  # Can be noisy, log selectively
    
    def on_chain_end(self, outputs, **kwargs):
        pass


logger = ProductionLogger()

print("=== Custom Callbacks ===")
response = chain.invoke(
    {"question": "What is 2+2?"},
    config={"callbacks": [logger]},
)
print(f"  Response: {response}")
print()

# ============================================================
# 8. Cost Tracking
# ============================================================

from langchain_community.callbacks import get_openai_callback

print("=== Cost Tracking ===")
with get_openai_callback() as cb:
    response1 = chain.invoke({"question": "What is Python?"})
    response2 = chain.invoke({"question": "What is JavaScript?"})

print(f"Total tokens: {cb.total_tokens}")
print(f"  Prompt tokens: {cb.prompt_tokens}")
print(f"  Completion tokens: {cb.completion_tokens}")
print(f"Total cost: ${cb.total_cost:.6f}")
print(f"Calls made: {cb.successful_requests}")
