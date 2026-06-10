"""
Phase 2: Chains & Advanced LCEL — Batch Processing & Async
============================================================
Process multiple inputs efficiently.

Topics covered:
- batch() for parallel processing
- Concurrency control
- Async patterns (ainvoke, abatch, astream)
- Real-world batch use cases
"""

import asyncio
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ============================================================
# 1. Basic Batch — Process multiple inputs at once
# ============================================================

chain = (
    ChatPromptTemplate.from_messages([
        ("human", "In exactly 5 words, describe {topic}.")
    ])
    | llm
    | StrOutputParser()
)

topics = [
    {"topic": "Python"},
    {"topic": "JavaScript"},
    {"topic": "Rust"},
    {"topic": "Go"},
    {"topic": "TypeScript"},
]

print("=== Basic Batch ===")
start = time.time()
results = chain.batch(topics)
elapsed = time.time() - start

for topic, result in zip(topics, results):
    print(f"  {topic['topic']}: {result}")
print(f"  Time: {elapsed:.2f}s for {len(topics)} items")
print()

# ============================================================
# 2. Batch with Concurrency Control
# ============================================================

# Without limit: all run at once (may hit rate limits)
# With limit: controlled parallelism

print("=== Concurrency Comparison ===")

# High concurrency
start = time.time()
results = chain.batch(topics, config={"max_concurrency": 5})
fast_time = time.time() - start
print(f"  max_concurrency=5: {fast_time:.2f}s")

# Low concurrency (safer for rate limits)
start = time.time()
results = chain.batch(topics, config={"max_concurrency": 2})
slow_time = time.time() - start
print(f"  max_concurrency=2: {slow_time:.2f}s")
print()

# ============================================================
# 3. Async — Non-blocking execution
# ============================================================

async def async_examples():
    """Demonstrate async patterns."""
    
    print("=== Async invoke ===")
    # Single async call
    result = await chain.ainvoke({"topic": "async programming"})
    print(f"  ainvoke: {result}")
    
    # Multiple concurrent calls with asyncio.gather
    print("\n=== asyncio.gather (true parallel) ===")
    start = time.time()
    tasks = [
        chain.ainvoke({"topic": "Python"}),
        chain.ainvoke({"topic": "AI"}),
        chain.ainvoke({"topic": "Cloud"}),
    ]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start
    
    for r in results:
        print(f"  {r}")
    print(f"  Time: {elapsed:.2f}s (all ran concurrently)")
    
    # Async batch
    print("\n=== Async Batch ===")
    start = time.time()
    results = await chain.abatch(topics, config={"max_concurrency": 3})
    elapsed = time.time() - start
    print(f"  Processed {len(results)} items in {elapsed:.2f}s")

asyncio.run(async_examples())
print()

# ============================================================
# 4. Real-world: Batch Processing a Dataset
# ============================================================

# Simulate processing a list of customer reviews
reviews = [
    {"topic": "Great product, fast shipping!"},
    {"topic": "Terrible quality, broke after one day"},
    {"topic": "Average experience, nothing special"},
    {"topic": "Best purchase I ever made!"},
    {"topic": "Would not recommend to anyone"},
]

sentiment_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Classify the sentiment as: positive, negative, or neutral. Reply with one word only."),
        ("human", "{topic}"),
    ])
    | llm
    | StrOutputParser()
)

print("=== Real-world: Batch Sentiment Analysis ===")
start = time.time()
sentiments = sentiment_chain.batch(reviews, config={"max_concurrency": 5})
elapsed = time.time() - start

for review, sentiment in zip(reviews, sentiments):
    print(f"  [{sentiment.strip():8s}] {review['topic']}")
print(f"  Processed {len(reviews)} reviews in {elapsed:.2f}s")
print()

# ============================================================
# 5. Batch with Error Handling
# ============================================================

def safe_batch(chain, inputs, max_concurrency=3):
    """Batch with per-item error handling."""
    results = []
    
    # Process in chunks to handle errors gracefully
    for i, inp in enumerate(inputs):
        try:
            result = chain.invoke(inp)
            results.append({"input": inp, "output": result, "error": None})
        except Exception as e:
            results.append({"input": inp, "output": None, "error": str(e)})
    
    return results

# Better approach: use batch with return_exceptions
async def safe_batch_async(chain, inputs, max_concurrency=3):
    """Async batch that doesn't stop on individual failures."""
    tasks = [chain.ainvoke(inp) for inp in inputs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed = []
    for inp, result in zip(inputs, results):
        if isinstance(result, Exception):
            processed.append({"input": inp, "output": None, "error": str(result)})
        else:
            processed.append({"input": inp, "output": result, "error": None})
    
    return processed

print("=== Safe Batch (with error handling) ===")
safe_results = safe_batch(sentiment_chain, reviews[:3])
for r in safe_results:
    status = "✓" if r["error"] is None else "✗"
    print(f"  {status} {r['output'] or r['error']}")
print()

# ============================================================
# 6. Progress Tracking for Large Batches
# ============================================================

def batch_with_progress(chain, inputs, max_concurrency=3):
    """Process batch with progress reporting."""
    total = len(inputs)
    results = []
    
    print(f"  Processing {total} items...")
    
    for i, inp in enumerate(inputs, 1):
        result = chain.invoke(inp)
        results.append(result)
        
        # Progress bar
        pct = (i / total) * 100
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"\r  [{bar}] {i}/{total} ({pct:.0f}%)", end="", flush=True)
    
    print()  # Newline after progress bar
    return results

print("=== Batch with Progress ===")
results = batch_with_progress(sentiment_chain, reviews)
print(f"  Done! Got {len(results)} results")
