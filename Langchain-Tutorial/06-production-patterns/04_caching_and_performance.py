"""
Phase 6: Production Patterns — Caching & Performance Optimization
===================================================================
Make LLM applications faster and cheaper with caching.

Topics covered:
- Response caching (exact match)
- Semantic caching (similar queries)
- Prompt optimization (reduce tokens)
- Batch processing
- Async for throughput
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
import hashlib
import json

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise."),
        ("human", "{question}"),
    ])
    | llm
    | StrOutputParser()
)

# ============================================================
# 1. Exact Match Cache (simple, effective)
# ============================================================

class ExactMatchCache:
    """Cache LLM responses by exact input match."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: dict[str, str] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _key(self, input_data: dict) -> str:
        """Generate cache key from input."""
        serialized = json.dumps(input_data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def get(self, input_data: dict) -> str | None:
        key = self._key(input_data)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, input_data: dict, response: str):
        if len(self.cache) >= self.max_size:
            # Evict oldest (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        key = self._key(input_data)
        self.cache[key] = response
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0


cache = ExactMatchCache()


def cached_invoke(input_data: dict) -> tuple[str, bool]:
    """Invoke chain with caching. Returns (response, was_cached)."""
    cached = cache.get(input_data)
    if cached:
        return cached, True
    
    response = chain.invoke(input_data)
    cache.set(input_data, response)
    return response, False


print("=== Exact Match Cache ===")
start = time.time()
result1, cached1 = cached_invoke({"question": "What is Python?"})
time1 = time.time() - start
print(f"First call: {time1:.2f}s (cached: {cached1})")

start = time.time()
result2, cached2 = cached_invoke({"question": "What is Python?"})
time2 = time.time() - start
print(f"Second call: {time2:.4f}s (cached: {cached2})")
print(f"Speedup: {time1/max(time2, 0.001):.0f}x faster")
print(f"Hit rate: {cache.hit_rate:.0%}")
print()

# ============================================================
# 2. Semantic Cache (similar queries match)
# ============================================================

from langchain_openai import OpenAIEmbeddings
import numpy as np


class SemanticCache:
    """Cache that matches semantically similar queries."""
    
    def __init__(self, threshold: float = 0.92):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.cache: list[dict] = []  # [{embedding, input, response}]
        self.threshold = threshold
        self.hits = 0
        self.misses = 0
    
    def _cosine_sim(self, a, b) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def get(self, query: str) -> str | None:
        if not self.cache:
            self.misses += 1
            return None
        
        query_emb = self.embeddings.embed_query(query)
        
        best_score = 0
        best_response = None
        
        for entry in self.cache:
            score = self._cosine_sim(query_emb, entry["embedding"])
            if score > best_score:
                best_score = score
                best_response = entry["response"]
        
        if best_score >= self.threshold:
            self.hits += 1
            return best_response
        
        self.misses += 1
        return None
    
    def set(self, query: str, response: str):
        embedding = self.embeddings.embed_query(query)
        self.cache.append({
            "embedding": embedding,
            "input": query,
            "response": response,
        })


sem_cache = SemanticCache(threshold=0.90)

print("=== Semantic Cache ===")

# First query
question1 = "What is machine learning?"
response1 = chain.invoke({"question": question1})
sem_cache.set(question1, response1)
print(f"Stored: '{question1}'")

# Similar query (should hit cache!)
question2 = "Explain what machine learning is"
cached_response = sem_cache.get(question2)
if cached_response:
    print(f"Cache HIT for: '{question2}'")
    print(f"Response: {cached_response[:60]}...")
else:
    print(f"Cache MISS for: '{question2}'")

# Different query (should miss)
question3 = "What is the weather today?"
cached_response = sem_cache.get(question3)
print(f"Cache {'HIT' if cached_response else 'MISS'} for: '{question3}'")
print()

# ============================================================
# 3. Prompt Optimization (reduce tokens)
# ============================================================

print("=== Prompt Optimization ===")

# BAD: Verbose prompt (wastes tokens every call)
verbose_prompt = """You are an incredibly helpful, knowledgeable, and friendly AI assistant 
who is always eager to help users with their questions. You should provide comprehensive, 
detailed, and thoughtful answers that address all aspects of the user's query. Please make 
sure to be thorough in your responses while also being clear and easy to understand. If you 
don't know something, please say so honestly rather than making something up."""

# GOOD: Concise prompt (same behavior, fewer tokens)
concise_prompt = "You are a helpful assistant. Be thorough but concise. Say 'I don't know' if unsure."

import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o-mini")
verbose_tokens = len(enc.encode(verbose_prompt))
concise_tokens = len(enc.encode(concise_prompt))

print(f"Verbose prompt: {verbose_tokens} tokens")
print(f"Concise prompt: {concise_tokens} tokens")
print(f"Savings: {verbose_tokens - concise_tokens} tokens/call")
print(f"At 10K calls/day: {(verbose_tokens - concise_tokens) * 10000:,} tokens saved/day")
print()

# Token counting helper
def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

print(f"Token counts:")
print(f"  'Hello world': {count_tokens('Hello world')} tokens")
print(f"  'What is RAG?': {count_tokens('What is RAG?')} tokens")
print(f"  Long paragraph: {count_tokens('LangChain is a framework ' * 50)} tokens")
print()

# ============================================================
# 4. Batch Processing (efficient for bulk)
# ============================================================

questions = [
    {"question": "What is Python?"},
    {"question": "What is JavaScript?"},
    {"question": "What is Rust?"},
    {"question": "What is Go?"},
    {"question": "What is TypeScript?"},
]

print("=== Batch Processing ===")

# Sequential (slow)
start = time.time()
sequential_results = [chain.invoke(q) for q in questions[:3]]
seq_time = time.time() - start
print(f"Sequential (3 calls): {seq_time:.2f}s")

# Batch (concurrent via LangChain)
start = time.time()
batch_results = chain.batch(questions[:3], config={"max_concurrency": 3})
batch_time = time.time() - start
print(f"Batch (3 calls, concurrent): {batch_time:.2f}s")
print(f"Speedup: {seq_time/max(batch_time, 0.01):.1f}x")
print()

# ============================================================
# 5. Async for High Throughput
# ============================================================

import asyncio


async def async_batch(questions: list[dict]) -> list[str]:
    """Process multiple questions concurrently with async."""
    tasks = [chain.ainvoke(q) for q in questions]
    results = await asyncio.gather(*tasks)
    return results


print("=== Async Processing ===")
start = time.time()
async_results = asyncio.run(async_batch(questions[:3]))
async_time = time.time() - start
print(f"Async (3 calls): {async_time:.2f}s")
print(f"vs Sequential: {seq_time/max(async_time, 0.01):.1f}x faster")
print()

# ============================================================
# 6. Streaming for Perceived Performance
# ============================================================

print("=== Streaming (Time to First Token) ===")

start = time.time()
first_token_time = None

for chunk in chain.stream({"question": "Explain RAG in 3 sentences."}):
    if first_token_time is None:
        first_token_time = time.time() - start
    print(chunk, end="", flush=True)

total_time = time.time() - start
print(f"\n\nTime to first token: {first_token_time:.2f}s")
print(f"Total time: {total_time:.2f}s")
print("Streaming shows output immediately → better UX even if total time is same")
print()

# ============================================================
# 7. Performance Summary
# ============================================================

print("=== Performance Optimization Summary ===")
print("| Technique | Benefit | Tradeoff |")
print("|-----------|---------|----------|")
print("| Exact cache | Instant for repeated queries | Memory, stale data |")
print("| Semantic cache | Catches similar queries | Embedding cost, complexity |")
print("| Prompt optimization | Fewer tokens/call | Time to optimize |")
print("| Batch processing | Parallel execution | Memory for results |")
print("| Async | Higher throughput | Code complexity |")
print("| Streaming | Faster perceived response | Can't post-process |")
print("| Smaller model | Cheaper, faster | May lose quality |")
