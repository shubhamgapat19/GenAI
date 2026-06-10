# Advanced RAG Techniques — Deep Dive Notes

## Why Basic RAG Isn't Enough

Basic RAG (embed → search → generate) works for simple cases but fails when:
- Query doesn't match document vocabulary (vocabulary mismatch)
- Multiple documents needed to answer one question
- Retrieval returns irrelevant but semantically similar text
- User asks complex multi-part questions
- Documents are long and only a sentence is relevant

Advanced techniques fix these specific failure modes.

---

## Technique 1: Hybrid Search

Combines **keyword search** (BM25) with **semantic search** (embeddings).

```
Keyword (BM25):    Exact word matching. "FAISS" → finds docs with "FAISS"
Semantic:          Meaning matching. "fast vector search" → finds FAISS docs
Hybrid:            Both! Best recall.
```

### When to Use
- Technical docs with specific terms (API names, error codes)
- When users search with exact phrases AND natural language
- When semantic search alone misses obvious keyword matches

### Implementation
```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

hybrid = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.4, 0.6],  # Tune these based on evaluation
)
```

**Weight tuning:**
- More keyword weight (0.6/0.4) → for technical/code docs
- More semantic weight (0.3/0.7) → for natural language content
- Equal (0.5/0.5) → good starting point

---

## Technique 2: Re-ranking

Two-stage retrieval: fast recall → precise re-ranking.

```
Stage 1: Retrieve top-20 (fast, imprecise)
Stage 2: Re-rank with cross-encoder → return top-5 (slow, precise)
```

### Why Re-ranking Works
- Bi-encoders (regular embeddings): encode query and doc separately. Fast but imprecise.
- Cross-encoders: encode query AND doc together. Slow but very precise.

### Implementation
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

reranker = CohereRerank(model="rerank-english-v3.0", top_n=5)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever,  # Returns top-20
)
# Result: top-5 most relevant after re-ranking
```

### When to Use
- You have many similar documents (legal, medical)
- Basic retrieval returns "close but not quite" results
- You can afford the extra latency (~200ms for Cohere)

---

## Technique 3: Multi-Query Retriever

Problem: One query may not surface all relevant docs.

Solution: Generate multiple query variations, retrieve for each, deduplicate.

```
Original: "How do I monitor my LLM app?"
Generated:
  1. "LLM application observability tools"
  2. "Monitoring and tracing for language model applications"  
  3. "How to debug LangChain applications"
  
Retrieve for all 3 → union → deduplicate → return
```

### When to Use
- Vague or short queries
- Technical topics with many synonyms
- When recall matters more than precision

---

## Technique 4: Parent-Child Retrieval

Problem: Small chunks = good retrieval precision but not enough context for the LLM.

Solution: Embed small chunks, but return their larger parent document.

```
Parent (stored, returned):  Full paragraph about vector stores (500 chars)
Children (embedded):        "Chroma is local" | "FAISS is fast" | "Pinecone is managed"

Query: "Which vector store is fastest?"
Match: "FAISS is fast" (child)
Return: Full paragraph about vector stores (parent) → more context for LLM
```

### When to Use
- Documents where context around a fact matters
- Technical docs where a sentence alone is insufficient
- When you're getting correct retrievals but poor LLM answers

---

## Technique 5: Contextual Compression

Problem: Retrieved documents contain both relevant AND irrelevant text.

Solution: Use LLM to extract only the relevant portion.

```
Retrieved: "LangChain was created in 2022. It uses Python. The weather is nice today."
Query: "When was LangChain created?"
Compressed: "LangChain was created in 2022."
```

### When to Use
- Large chunks that contain mixed content
- When you want to reduce tokens sent to the generation LLM
- When retrieval is good but generation has too much noise

### Trade-off
- Extra LLM call per retrieved document (cost + latency)
- Worth it for complex queries or expensive generation models

---

## Technique 6: Self-Query

Problem: Users mix content queries with metadata filters in natural language.

```
User: "Find Python tutorials from 2024"
      ├── Content query: "Python tutorials"
      └── Metadata filter: year = 2024
```

### How It Works
LLM extracts structured filters from natural language:
```python
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Programming tutorials",
    metadata_field_info=[
        AttributeInfo(name="language", type="string", description="Programming language"),
        AttributeInfo(name="year", type="integer", description="Publication year"),
        AttributeInfo(name="difficulty", type="string", description="easy, medium, hard"),
    ],
)
```

### When to Use
- Metadata-rich documents (dates, categories, tags)
- Users naturally filter in their questions
- E-commerce, documentation with versions, multi-category knowledge bases

---

## Technique 7: Query Transformation

Transform the user's query before retrieval:

### HyDE (Hypothetical Document Embedding)
Generate a hypothetical answer, then use THAT as the search query:
```
Query: "How does RAG work?"
Hypothetical answer: "RAG works by first retrieving relevant documents from a vector store, then passing them as context to an LLM which generates an answer grounded in the retrieved information."
Search with: hypothetical answer embedding (more similar to actual docs than the question)
```

### Step-back Prompting
Convert specific question to a broader one:
```
Query: "What's the chunk_size parameter in RecursiveCharacterTextSplitter?"
Step-back: "How does text splitting work in LangChain?"
```

---

## Evaluation Framework

### Key Metrics

| Metric | What It Measures | How |
|--------|-----------------|-----|
| **Context Precision** | Are retrieved docs relevant? | % of retrieved docs that are relevant |
| **Context Recall** | Did we get all relevant docs? | % of relevant docs that were retrieved |
| **Faithfulness** | Is the answer grounded in context? | Does answer match the provided docs |
| **Answer Relevancy** | Does the answer address the question? | Semantic similarity of answer to question |

### RAGAS Framework
```python
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, answer_relevancy

results = evaluate(
    dataset=test_dataset,  # questions + ground truth + contexts + answers
    metrics=[faithfulness, context_precision, answer_relevancy],
)
```

### Manual Evaluation Checklist
- [ ] Does the system say "I don't know" for out-of-scope questions?
- [ ] Are citations accurate (point to real sources)?
- [ ] Does it avoid hallucination (not making up facts)?
- [ ] Are follow-up questions handled correctly?
- [ ] Does retrieval find the right chunks?

---

## Choosing the Right Techniques

| Problem | Solution |
|---------|----------|
| Keyword queries returning no results | Add hybrid search (BM25) |
| Too many similar/duplicate results | Use MMR or re-ranking |
| Single query misses relevant docs | Multi-query retriever |
| Chunks too small for good answers | Parent-child retrieval |
| Too much noise in retrieved text | Contextual compression |
| Users filter by metadata naturally | Self-query retriever |
| Questions don't match doc vocabulary | HyDE or query transformation |

---

## Best Practices

1. **Start simple, add complexity only when needed** — basic RAG first, measure, then improve
2. **Evaluate before and after each technique** — prove it helps
3. **Combine techniques** — hybrid search + re-ranking + compression can stack
4. **Monitor retrieval quality separately from generation** — diagnose which stage fails
5. **Build a test dataset early** — 50+ question-answer pairs for evaluation
6. **Log everything** — retrieved docs, scores, final answers (use LangSmith)
7. **Iterate on chunking first** — often the cheapest improvement
