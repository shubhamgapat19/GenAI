# Retrieval Chains — Deep Dive Notes

## The Complete RAG Pipeline

```
User Question
      ↓
[Embed Question] → query vector
      ↓
[Vector Store Search] → top-k relevant documents
      ↓
[Format Context] → combine docs into text
      ↓
[Prompt Template] → "Given this context: {context}, answer: {question}"
      ↓
[LLM] → generates answer grounded in context
      ↓
Answer (with citations)
```

---

## The Basic RAG Chain (LCEL)

```python
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,   # Retrieve + format
        "question": RunnablePassthrough(),     # Original question
    }
    | rag_prompt    # Combine into prompt
    | llm           # Generate answer
    | StrOutputParser()
)

answer = rag_chain.invoke("What is RAG?")
```

This is the **most important pattern** in LangChain. Everything else builds on this.

---

## RAG Prompt Design

### Basic (works well)
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer based ONLY on the provided context.
If the answer isn't in the context, say "I don't know."

Context:
{context}"""),
    ("human", "{question}"),
])
```

### With Instructions
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a technical documentation assistant.
Rules:
- Answer ONLY from the provided context
- If unsure, say "I don't have enough information"
- Be concise (2-3 sentences max)
- Include relevant technical details

Context:
{context}"""),
    ("human", "{question}"),
])
```

### With Citations
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer based on context. Cite sources as [Source: X].

Context:
{context}"""),
    ("human", "{question}"),
])
```

---

## Retriever Types

### 1. Basic Vector Store Retriever
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```
Simple similarity search. Good baseline.

### 2. Multi-Query Retriever
Generates multiple query variations to improve recall:
```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
)
# "What is RAG?" becomes:
# 1. "What is Retrieval Augmented Generation?"
# 2. "How does RAG work in LLM applications?"
# 3. "Explain the RAG technique"
# Retrieves for all 3, deduplicates results
```

### 3. Contextual Compression Retriever
Extracts only the relevant parts from retrieved documents:
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)
# Retrieved doc: "LangChain was created in 2022. It supports Python and JS. The weather is nice."
# After compression: "LangChain was created in 2022. It supports Python and JS."
```

### 4. Self-Query Retriever
Extracts metadata filters from natural language:
```python
# User: "Show me documents about RAG from 2024"
# Self-query extracts: query="RAG", filter={"year": 2024}
```

### 5. Parent Document Retriever
Embed small chunks, return large parent documents:
```python
# Embed: "Harrison Chase created LangChain" (small, precise)
# Return: Full paragraph about LangChain history (more context)
```

---

## Conversational RAG

Problem: Follow-up questions like "What features does **it** have?" don't work with basic RAG because "it" has no context.

Solution: Reformulate the question using chat history before retrieval.

```python
# Step 1: Reformulate
"What features does it have?" + history about LangSmith
    → "What features does LangSmith have?"

# Step 2: Retrieve with standalone question
# Step 3: Generate answer
```

```python
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the follow-up as a standalone question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Chain: contextualize → retrieve → generate
```

---

## Returning Sources with Answers

### Pattern 1: Return docs alongside answer
```python
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(
    answer=rag_chain,
    sources=retriever,
)
# result = {"answer": "...", "sources": [Document, Document, ...]}
```

### Pattern 2: Include sources in the answer text
Format context with source info so LLM can cite:
```python
def format_docs_with_sources(docs):
    return "\n".join(
        f"[{doc.metadata['source']}]: {doc.page_content}"
        for doc in docs
    )
```

---

## RAG Failure Modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| **Retrieval failure** | Right answer exists but not retrieved | Multi-query, hybrid search, better chunking |
| **Context poisoning** | Irrelevant docs confuse the LLM | Score threshold, contextual compression |
| **Hallucination despite context** | LLM ignores context, makes up answer | Stronger system prompt, lower temperature |
| **Lost in the middle** | LLM ignores docs in the middle of context | Reorder docs (most relevant first/last) |
| **Insufficient context** | Chunk is too small to answer | Larger chunks, parent document retriever |
| **Wrong granularity** | Returns whole pages when sentence needed | Smaller chunks, contextual compression |

---

## Best Practices

1. **Always instruct "answer only from context"** — prevents hallucination
2. **Return sources** — users need to verify, and it builds trust
3. **Use multi-query for vague questions** — dramatically improves recall
4. **Set k=3-5** — more isn't always better (noise increases)
5. **Test with edge cases** — questions NOT in your knowledge base
6. **Add "I don't know" behavior** — never make up answers
7. **Evaluate systematically** — don't just try a few queries manually
8. **Consider conversation context** — users ask follow-up questions

---

## RAG vs Fine-tuning

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| Knowledge update | Add docs (minutes) | Retrain (hours/days) |
| Cost | Retrieval + generation | Training + generation |
| Accuracy | High (cites sources) | Moderate (may hallucinate) |
| Transparency | Shows sources | Black box |
| Best for | Factual Q&A, documents | Style, tone, formatting |
| Freshness | Always up to date | Stale until retrained |

**Use RAG for knowledge. Use fine-tuning for behavior.**
