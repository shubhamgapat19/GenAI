"""
Phase 3: RAG — Text Splitters
================================
Split large documents into smaller chunks for embedding.

Topics covered:
- Why splitting matters
- RecursiveCharacterTextSplitter
- Chunk size and overlap
- Different splitting strategies
"""

from langchain_core.documents import Document

# ============================================================
# 1. Why Split? The Problem
# ============================================================

# LLMs have context window limits (e.g., 128K tokens for GPT-4o)
# But for RAG, we need SMALL chunks for precise retrieval
# A full PDF in one chunk = poor retrieval quality

long_text = """
LangChain is a framework for developing applications powered by large language models (LLMs).
It provides tools for prompt management, chains, agents, and retrieval augmented generation.
The framework supports multiple LLM providers including OpenAI, Anthropic, and Google.

RAG (Retrieval Augmented Generation) is a technique that enhances LLM responses by providing
relevant context from external knowledge bases. The process involves: loading documents,
splitting them into chunks, creating embeddings, storing in a vector database, and retrieving
relevant chunks at query time.

Vector databases store document embeddings as high-dimensional vectors. When a user asks a
question, the question is also embedded, and the most similar document vectors are retrieved.
Popular vector databases include Pinecone, Weaviate, Chroma, and FAISS.

Text splitting is crucial because embedding entire documents leads to poor retrieval quality.
Smaller chunks allow for more precise matching between the user's question and relevant content.
The ideal chunk size depends on the use case, typically between 500-1000 characters.
""" * 5  # Make it long

print(f"Original text length: {len(long_text)} characters")
print()

# ============================================================
# 2. RecursiveCharacterTextSplitter (recommended default)
# ============================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,         # Max characters per chunk
    chunk_overlap=50,       # Overlap between consecutive chunks
    length_function=len,    # How to measure length
    separators=["\n\n", "\n", ". ", " ", ""],  # Split priority
)

chunks = splitter.split_text(long_text)

print("=== RecursiveCharacterTextSplitter ===")
print(f"Split into {len(chunks)} chunks")
print(f"Chunk sizes: {[len(c) for c in chunks[:5]]}")
print(f"\nFirst chunk:\n{chunks[0][:200]}...")
print(f"\nSecond chunk:\n{chunks[1][:200]}...")
print()

# ============================================================
# 3. Splitting Documents (preserves metadata)
# ============================================================

documents = [
    Document(page_content=long_text, metadata={"source": "langchain_guide.pdf", "page": 1}),
]

split_docs = splitter.split_documents(documents)

print("=== Split Documents (with metadata) ===")
print(f"Original: {len(documents)} doc(s)")
print(f"After split: {len(split_docs)} chunks")
print(f"First chunk metadata: {split_docs[0].metadata}")
print(f"Last chunk metadata: {split_docs[-1].metadata}")
print()

# ============================================================
# 4. Chunk Size and Overlap — Understanding the Tradeoff
# ============================================================

# Small chunks (200 chars): precise retrieval, but may lose context
small_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
small_chunks = small_splitter.split_text(long_text)

# Medium chunks (500 chars): good balance
medium_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
medium_chunks = medium_splitter.split_text(long_text)

# Large chunks (1000 chars): more context, but less precise retrieval
large_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
large_chunks = large_splitter.split_text(long_text)

print("=== Chunk Size Comparison ===")
print(f"Small (200): {len(small_chunks)} chunks, avg {sum(len(c) for c in small_chunks)//len(small_chunks)} chars")
print(f"Medium (500): {len(medium_chunks)} chunks, avg {sum(len(c) for c in medium_chunks)//len(medium_chunks)} chars")
print(f"Large (1000): {len(large_chunks)} chunks, avg {sum(len(c) for c in large_chunks)//len(large_chunks)} chars")
print()

# ============================================================
# 5. Token-based Splitting (more accurate for LLM context)
# ============================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split by token count instead of character count
token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4o-mini",
    chunk_size=100,          # 100 tokens per chunk
    chunk_overlap=10,        # 10 token overlap
)

token_chunks = token_splitter.split_text(long_text)

print("=== Token-based Splitting ===")
print(f"Split into {len(token_chunks)} chunks (by tokens)")
print(f"First chunk: {token_chunks[0][:150]}...")
print()

# ============================================================
# 6. Markdown Splitter (structure-aware)
# ============================================================

from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_text = """# Introduction
LangChain is a framework for LLM apps.

## Key Features
It provides chains, agents, and RAG capabilities.

### Chains
Chains connect multiple components together.

### Agents
Agents can use tools to accomplish tasks.

## Getting Started
Install with pip install langchain.
"""

headers_to_split_on = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
]

md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_chunks = md_splitter.split_text(markdown_text)

print("=== Markdown Splitter ===")
for chunk in md_chunks:
    print(f"  Content: {chunk.page_content[:60]}...")
    print(f"  Headers: {chunk.metadata}")
    print()

# ============================================================
# 7. Code Splitter (language-aware)
# ============================================================

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

python_code = """
import os

def hello_world():
    '''Greet the world.'''
    print("Hello, World!")

class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        return x + y
    
    def subtract(self, x, y):
        return x - y

def main():
    calc = Calculator()
    print(calc.add(2, 3))

if __name__ == "__main__":
    main()
"""

code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=20,
)

code_chunks = code_splitter.split_text(python_code)

print("=== Code Splitter (Python) ===")
for i, chunk in enumerate(code_chunks):
    print(f"Chunk {i+1}:\n{chunk}\n---")
