# Text Splitters — Deep Dive Notes

## Why Split Documents?

1. **Embedding quality** — shorter text = more focused embedding = better retrieval
2. **Context window limits** — can't pass entire PDFs to LLM
3. **Precision** — small chunks match specific questions better
4. **Cost** — less irrelevant text = fewer tokens = lower cost

---

## The Splitting Tradeoff

```
Small chunks (200 chars)          Large chunks (2000 chars)
├── ✅ Precise retrieval          ├── ✅ More context per chunk
├── ✅ Less noise                 ├── ✅ Fewer chunks to search
├── ❌ May lose context           ├── ❌ Less precise matching
└── ❌ More chunks to store       └── ❌ More irrelevant text retrieved
```

**Sweet spot:** 500-1000 characters for most use cases.

---

## Splitter Types

| Splitter | How It Splits | Best For |
|----------|--------------|----------|
| `RecursiveCharacterTextSplitter` | By separators hierarchy: `\n\n` → `\n` → `. ` → ` ` | General text (default choice) |
| `TokenTextSplitter` | By token count | Precise token budgets |
| `MarkdownHeaderTextSplitter` | By markdown headings | Documentation, READMEs |
| `HTMLHeaderTextSplitter` | By HTML headers | Web pages |
| `RecursiveCharacterTextSplitter.from_language()` | By language syntax | Code files |
| `SemanticChunker` | By embedding similarity | When meaning boundaries matter |

---

## RecursiveCharacterTextSplitter (The Default)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Max chars per chunk
    chunk_overlap=50,     # Shared chars between consecutive chunks
    separators=["\n\n", "\n", ". ", " ", ""],  # Try these in order
)
```

### How It Works
1. Try to split by `\n\n` (paragraphs)
2. If a chunk is still too big, split by `\n` (lines)
3. Still too big? Split by `. ` (sentences)
4. Still too big? Split by ` ` (words)
5. Last resort: split by character

This preserves the most meaningful boundaries possible.

---

## Chunk Overlap — Why It Matters

```
Chunk 1: "...LangChain provides tools for building LLM apps. It was created"
Chunk 2: "It was created by Harrison Chase in 2022. The framework supports..."
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          This is the overlap (shared text)
```

**Without overlap:** A sentence split across two chunks may never be found by search.
**With overlap:** The shared text ensures boundary content is searchable from either chunk.

**Rule of thumb:** overlap = 10-20% of chunk_size

---

## Token-Based Splitting

Characters ≠ tokens. For accurate context window management:

```python
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4o-mini",
    chunk_size=256,        # 256 tokens (not characters!)
    chunk_overlap=25,
)
```

**When to use:** When you need precise control over how many tokens each chunk uses (e.g., fitting exactly N chunks into context window).

---

## Markdown Splitter

Preserves document structure as metadata:

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
)

# Result: each chunk knows which section it belongs to
# chunk.metadata = {"h1": "Introduction", "h2": "Installation"}
```

**When to use:** Documentation, technical guides, structured content.

---

## Code Splitter

Splits code respecting language syntax (functions, classes):

```python
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Supported: PYTHON, JS, TS, JAVA, GO, RUST, CPP, etc.
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50,
)
```

Python separators: class → function → decorator → newlines
JS separators: function → class → if/for → newlines

---

## Semantic Chunking (experimental)

Split based on **meaning changes**, not character counts:

```python
from langchain_experimental.text_splitter import SemanticChunker

splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
)
# Groups sentences with similar embeddings together
# Splits where meaning shifts significantly
```

**Pros:** Chunks are semantically coherent
**Cons:** Requires embedding calls during splitting (slow, costs money)

---

## Choosing Chunk Size — Guidelines

| Content Type | Recommended Size | Overlap |
|-------------|-----------------|---------|
| Short factual (FAQ) | 200-300 chars | 20 |
| General documents | 500-800 chars | 50-100 |
| Technical docs | 800-1200 chars | 100-150 |
| Code | 500-1000 chars | 50 |
| Legal/dense text | 1000-1500 chars | 150-200 |

---

## Best Practices

1. **Start with `RecursiveCharacterTextSplitter`** — it works for 90% of cases
2. **Experiment with chunk sizes** — test 300, 500, 800, 1000 on your data
3. **Always use overlap** — prevents losing boundary information
4. **Use token-based for LLM context math** — when you need exact token counts
5. **Preserve metadata through splitting** — `split_documents()` keeps metadata
6. **Use structure-aware splitters for structured content** — Markdown, HTML, Code
7. **Evaluate retrieval quality** — the "best" chunk size is the one that retrieves correctly

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| No overlap | Sentences at boundaries are lost | Add 10-20% overlap |
| Chunks too large | Poor retrieval precision | Reduce to 500-800 chars |
| Chunks too small | Lost context, many irrelevant results | Increase to 500+ chars |
| Splitting code by character | Breaks function logic | Use `from_language()` |
| Ignoring metadata | Can't filter or cite sources | Use `split_documents()` |
| One size for all docs | Different content needs different sizes | Customize per doc type |
