"""
Phase 2: Chains & Advanced LCEL — Streaming Deep Dive
======================================================
Master streaming for real-time user experiences.

Topics covered:
- Token-by-token streaming
- Streaming with chains
- astream_events for fine-grained control
- Streaming structured output
"""

import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ============================================================
# 1. Basic Streaming — Token by token
# ============================================================

chain = (
    ChatPromptTemplate.from_messages([("human", "Write a short poem about {topic}")])
    | llm
    | StrOutputParser()
)

print("=== Basic Streaming ===")
for chunk in chain.stream({"topic": "coding at night"}):
    print(chunk, end="", flush=True)
print("\n")

# ============================================================
# 2. Streaming with Metadata — Know when steps complete
# ============================================================

def process_stream_with_info():
    """Stream and track chunks."""
    total_tokens = 0
    full_response = ""
    
    for chunk in chain.stream({"topic": "Python decorators"}):
        full_response += chunk
        total_tokens += 1  # Approximate (chunks ≠ tokens exactly)
        print(chunk, end="", flush=True)
    
    print(f"\n\n[Received {total_tokens} chunks, {len(full_response)} chars]")

print("=== Stream with Tracking ===")
process_stream_with_info()
print()

# ============================================================
# 3. Async Streaming — For web servers (FastAPI, etc.)
# ============================================================

async def async_stream_example():
    """Async streaming — use this in FastAPI/web apps."""
    print("=== Async Streaming ===")
    
    async for chunk in chain.astream({"topic": "async programming"}):
        print(chunk, end="", flush=True)
    print("\n")

asyncio.run(async_stream_example())

# ============================================================
# 4. astream_events — Fine-grained event streaming
# ============================================================

async def stream_events_example():
    """Track every event in the chain execution."""
    print("=== Stream Events ===")
    
    multi_step_chain = (
        ChatPromptTemplate.from_messages([
            ("human", "List 3 facts about {topic}")
        ]).with_config(run_name="prompt_step")
        | llm.with_config(run_name="llm_step")
        | StrOutputParser().with_config(run_name="parser_step")
    ).with_config(run_name="facts_chain")
    
    events_seen = set()
    
    async for event in multi_step_chain.astream_events(
        {"topic": "Python"},
        version="v2"
    ):
        kind = event["event"]
        name = event.get("name", "")
        
        # Track unique events
        if kind not in events_seen:
            events_seen.add(kind)
        
        # Print streaming tokens from the LLM
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            print(chunk.content, end="", flush=True)
        
        # Print when chain starts/ends
        elif kind == "on_chain_start" and name == "facts_chain":
            print("[Chain started]")
        elif kind == "on_chain_end" and name == "facts_chain":
            print("\n[Chain ended]")
    
    print(f"\nEvent types seen: {events_seen}")

asyncio.run(stream_events_example())
print()

# ============================================================
# 5. Streaming JSON (partial objects)
# ============================================================

async def stream_json_example():
    """Stream JSON output — get partial results as they build up."""
    print("=== Streaming JSON ===")
    
    json_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "Always respond in valid JSON."),
            ("human", "Create a profile for a fictional character named {name}. Include: name, age, occupation, skills (list of 3), bio (2 sentences)."),
        ])
        | llm
        | JsonOutputParser()
    )
    
    # JsonOutputParser supports streaming — you get partial dicts!
    async for partial in json_chain.astream({"name": "Alex"}):
        print(f"  Partial: {partial}")
    
    print()

asyncio.run(stream_json_example())

# ============================================================
# 6. Custom Streaming Transformer
# ============================================================

def streaming_uppercase(chunks):
    """Transform stream chunks on the fly."""
    for chunk in chunks:
        yield chunk.upper()


chain_with_transform = (
    ChatPromptTemplate.from_messages([("human", "Say hello in 5 languages, one per line")])
    | llm
    | StrOutputParser()
    | streaming_uppercase  # Generator function works as streaming transform!
)

print("=== Custom Stream Transform (uppercase) ===")
for chunk in chain_with_transform.stream({}):
    print(chunk, end="", flush=True)
print("\n")

# ============================================================
# 7. Practical: SSE (Server-Sent Events) Pattern
# ============================================================

def simulate_sse_endpoint(topic: str):
    """
    Simulates what you'd do in a FastAPI SSE endpoint.
    In production: yield these as SSE events.
    """
    print("=== Simulated SSE Stream ===")
    print(f"data: {{'event': 'start', 'topic': '{topic}'}}")
    
    full_text = ""
    for chunk in chain.stream({"topic": topic}):
        full_text += chunk
        # In FastAPI, you'd yield: f"data: {json.dumps({'token': chunk})}\n\n"
        print(f"data: {{'token': '{chunk}'}}")
    
    print(f"data: {{'event': 'end', 'total_chars': {len(full_text)}}}")
    print()

simulate_sse_endpoint("streaming in LangChain")
