# Embeddings & Vector Stores — Deep Dive Notes

## What Are Embeddings?

Embeddings convert text into **numerical vectors** (arrays of floats) that capture semantic meaning. Similar text → similar vectors → close in vector space.

```
"Python programming"  → [0.12, -0.34, 0.56, ..., 0.89]  (1536 dimensions)
"Coding in Python"    → [0.11, -0.33, 0.55, ..., 0.88]  (very similar!)
"Baking chocolate cake" → [-0.45, 0.67, -0.12, ..., 0.23] (very different)
```

---

## How Similarity Search Works

```
Query: "How to write Python code?"
                ↓
         [Embed query]
                ↓
    query_vector = [0.12, -0.34, ...]
                ↓
    [Compare with all stored vectors]
                ↓
    Find closest vectors (cosine similarity)
                ↓
    Return corresponding documents
```

### Cosine Similarity
- Range: -1 to 1 (higher = more similar)
- 0.9+ = very similar (almost same meaning)
- 0.7-0.9 = related
- 0.5-0.7 = somewhat related
- <0.5 = not related

---

## Embedding Models

| Model | Dimensions | Cost | Quality |
|-------|-----------|------|---------|
| `text-embedding-3-small` (OpenAI) | 1536 | $0.02/1M tokens | Good |
| `text-embedding-3-large` (OpenAI) | 3072 | $0.13/1M tokens | Better |
| `text-embedding-ada-002` (OpenAI) | 1536 | $0.10/1M tokens | Legacy |
| HuggingFace (free, local) | Varies | Free | Varies |
| Cohere embed-v3 | 1024 | $0.10/1M tokens | Good |

**Recommendation:** Start with `text-embedding-3-small` — best cost/quality ratio.

---

## Embedding in LangChain

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Single text (for queries)
query_vector = embeddings.embed_query("What is RAG?")

# Multiple texts (for documents — more efficient)
doc_vectors = embeddings.embed_documents(["doc 1 text", "doc 2 text"])
```

### embed_query vs embed_documents
- `embed_query`: single text, optimized for search queries
- `embed_documents`: batch of texts, optimized for storage
- Some models treat these differently (asymmetric embeddings)

---

## Vector Stores

Vector stores = databases optimized for storing and searching embedding vectors.

### Comparison

| Store | Type | Best For | Persistence |
|-------|------|----------|-------------|
| **Chroma** | Local | Development, small-medium | Disk |
| **FAISS** | Local | Speed, large datasets | File-based |
| **Pinecone** | Cloud | Production, managed | Cloud |
| **Weaviate** | Self-host/Cloud | Hybrid search | Configurable |
| **pgvector** | PostgreSQL extension | If you already use Postgres | Database |
| **Qdrant** | Self-host/Cloud | Performance, filtering | Configurable |

---

## Chroma (Recommended for Learning)

```python
from langchain_chroma import Chroma

# Create from documents
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="my_collection",
    persist_directory="./chroma_db",  # Save to disk
)

# Load existing
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="my_collection",
)

# Add more documents later
vectorstore.add_documents(new_docs)

# Delete
vectorstore.delete(ids=["doc_id_1", "doc_id_2"])
```

---

## FAISS (Fast, In-Memory)

```python
from langchain_community.vectorstores import FAISS

# Create
vectorstore = FAISS.from_documents(docs, embeddings)

# Save to disk
vectorstore.save_local("./faiss_index")

# Load from disk
vectorstore = FAISS.load_local(
    "./faiss_index", 
    embeddings,
    allow_dangerous_deserialization=True,
)

# Merge two indices
vectorstore.merge_from(other_vectorstore)
```

---

## Search Types

### 1. Similarity Search
Find the k most similar documents:
```python
results = vectorstore.similarity_search("query", k=4)
```

### 2. Similarity Search with Scores
Same as above but includes distance/similarity scores:
```python
results = vectorstore.similarity_search_with_score("query", k=4)
# Returns: [(Document, score), ...]
# Lower score = more similar (for distance metrics)
```

### 3. MMR (Maximum Marginal Relevance)
Balances relevance with diversity — avoids returning near-duplicate results:
```python
results = vectorstore.max_marginal_relevance_search(
    "query",
    k=4,           # Return 4 results
    fetch_k=20,    # Consider top 20 first
    lambda_mult=0.5,  # 0=max diversity, 1=max relevance
)
```

### 4. Metadata Filtering
Search only within a subset of documents:
```python
results = vectorstore.similarity_search(
    "query",
    k=4,
    filter={"source": "docs", "topic": "rag"},
)
```

---

## Retriever Interface

Convert any vector store to a Retriever for use in LCEL chains:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",     # "similarity", "mmr", "similarity_score_threshold"
    search_kwargs={
        "k": 4,                   # Number of results
        "score_threshold": 0.7,   # Min similarity (for threshold type)
        "filter": {"source": "docs"},  # Metadata filter
    },
)

# Now usable in LCEL chains
docs = retriever.invoke("What is RAG?")
```

---

## Embedding Costs and Optimization

### Cost Estimation
```
text-embedding-3-small: $0.02 per 1M tokens
1 page ≈ 500 tokens
1000 pages = 500K tokens = $0.01

It's CHEAP — embedding costs are negligible vs LLM generation costs.
```

### Optimization Tips
1. **Don't re-embed** — persist your vector store, don't rebuild each time
2. **Batch embed** — `embed_documents()` is more efficient than looping `embed_query()`
3. **Use smaller models** — `text-embedding-3-small` is 6.5x cheaper than `-large`
4. **Deduplicate before embedding** — don't embed the same text twice

---

## Best Practices

1. **Pick one embedding model and stick with it** — you can't mix models in one vector store
2. **Persist your vector store** — rebuilding is wasteful
3. **Use metadata generously** — enables filtering without re-embedding
4. **Start with Chroma for dev, Pinecone/pgvector for prod**
5. **Test MMR vs similarity** — MMR often gives better results for diverse queries
6. **Set a score threshold** — avoid returning irrelevant results when nothing matches
7. **Monitor index size** — thousands of docs = fine, millions = consider managed service

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Using different embedding models for indexing vs query | Results are garbage | Same model for both |
| Not persisting vector store | Re-embeds on every restart | Use `persist_directory` |
| No metadata | Can't filter or cite | Add metadata at loading time |
| k too high | Returns irrelevant results | Start with k=3-5 |
| k too low | Misses relevant info | Increase k or use multi-query |
| No score threshold | Returns bad results confidently | Set `score_threshold` |
