"""
Phase 3: RAG — Embeddings & Vector Stores
==========================================
Convert text to vectors and store for similarity search.

Topics covered:
- What are embeddings
- OpenAI embeddings
- Vector stores (Chroma, FAISS)
- Similarity search
- Metadata filtering
"""

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

# ============================================================
# 1. What Are Embeddings?
# ============================================================

from langchain_openai import OpenAIEmbeddings

# Embeddings convert text → numbers (high-dimensional vectors)
# Similar text → similar vectors (close in vector space)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Embed a single text
single_embedding = embeddings.embed_query("What is LangChain?")

print("=== Embeddings ===")
print(f"Embedding dimensions: {len(single_embedding)}")
print(f"First 5 values: {single_embedding[:5]}")
print()

# Embed multiple texts
texts = ["Python programming", "JavaScript coding", "Cooking pasta"]
multi_embeddings = embeddings.embed_documents(texts)

print(f"Embedded {len(multi_embeddings)} texts")
print(f"Each has {len(multi_embeddings[0])} dimensions")
print()

# ============================================================
# 2. Similarity — How embeddings relate
# ============================================================

import numpy as np

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Embed query and documents
query_emb = embeddings.embed_query("How to write Python code?")
doc1_emb = embeddings.embed_query("Python programming tutorial")
doc2_emb = embeddings.embed_query("Recipe for chocolate cake")

sim_relevant = cosine_similarity(query_emb, doc1_emb)
sim_irrelevant = cosine_similarity(query_emb, doc2_emb)

print("=== Similarity Scores ===")
print(f"Query vs 'Python programming': {sim_relevant:.4f} (high = relevant)")
print(f"Query vs 'Chocolate cake':     {sim_irrelevant:.4f} (low = irrelevant)")
print()

# ============================================================
# 3. Chroma Vector Store (local, persistent)
# ============================================================

from langchain_chroma import Chroma

# Sample documents to store
documents = [
    Document(page_content="LangChain is a framework for building LLM applications.", metadata={"source": "docs", "topic": "langchain"}),
    Document(page_content="RAG combines retrieval with generation for better answers.", metadata={"source": "docs", "topic": "rag"}),
    Document(page_content="Vector databases store embeddings for similarity search.", metadata={"source": "docs", "topic": "vectordb"}),
    Document(page_content="Python is a popular programming language for AI.", metadata={"source": "blog", "topic": "python"}),
    Document(page_content="Prompt engineering is the art of writing effective prompts.", metadata={"source": "blog", "topic": "prompts"}),
    Document(page_content="Agents use tools to perform actions autonomously.", metadata={"source": "docs", "topic": "agents"}),
]

# Create vector store from documents
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="langchain_learning",
    persist_directory="./chroma_db",  # Persists to disk
)

print("=== Chroma Vector Store ===")
print(f"Stored {len(documents)} documents")
print()

# ============================================================
# 4. Similarity Search
# ============================================================

# Find documents most similar to a query
results = vectorstore.similarity_search(
    query="How does RAG work?",
    k=3,  # Return top 3 results
)

print("=== Similarity Search: 'How does RAG work?' ===")
for i, doc in enumerate(results, 1):
    print(f"  {i}. [{doc.metadata['topic']}] {doc.page_content}")
print()

# ============================================================
# 5. Similarity Search with Scores
# ============================================================

results_with_scores = vectorstore.similarity_search_with_score(
    query="What is LangChain?",
    k=3,
)

print("=== Search with Scores ===")
for doc, score in results_with_scores:
    print(f"  Score: {score:.4f} | {doc.page_content[:60]}...")
print()

# ============================================================
# 6. Metadata Filtering
# ============================================================

# Only search within documents from "docs" source
filtered_results = vectorstore.similarity_search(
    query="What tools are available?",
    k=3,
    filter={"source": "docs"},  # Only search docs, not blog
)

print("=== Filtered Search (source='docs' only) ===")
for doc in filtered_results:
    print(f"  [{doc.metadata['source']}] {doc.page_content[:60]}...")
print()

# ============================================================
# 7. FAISS Vector Store (in-memory, fast)
# ============================================================

from langchain_community.vectorstores import FAISS

# Create FAISS index from documents
faiss_store = FAISS.from_documents(
    documents=documents,
    embedding=embeddings,
)

# Search
faiss_results = faiss_store.similarity_search("prompt engineering tips", k=2)

print("=== FAISS Search ===")
for doc in faiss_results:
    print(f"  {doc.page_content[:60]}...")
print()

# Save/Load FAISS index
faiss_store.save_local("./faiss_index")
# loaded_store = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)

# ============================================================
# 8. MMR Search (Maximum Marginal Relevance)
# ============================================================

# MMR balances relevance with diversity (avoids duplicate-ish results)
mmr_results = vectorstore.max_marginal_relevance_search(
    query="LangChain features",
    k=3,
    fetch_k=6,      # Fetch 6, then pick 3 most diverse
    lambda_mult=0.5, # 0=max diversity, 1=max relevance
)

print("=== MMR Search (diverse results) ===")
for doc in mmr_results:
    print(f"  {doc.page_content[:60]}...")
print()

# ============================================================
# 9. Using Vector Store as a Retriever
# ============================================================

# Convert to retriever (used in RAG chains)
retriever = vectorstore.as_retriever(
    search_type="similarity",  # or "mmr"
    search_kwargs={"k": 3},
)

# Retriever has .invoke() method — fits into LCEL chains
retrieved_docs = retriever.invoke("How do agents work?")

print("=== Retriever ===")
for doc in retrieved_docs:
    print(f"  {doc.page_content[:60]}...")
