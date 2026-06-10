# Document Loaders — Deep Dive Notes

## What Are Document Loaders?

Document Loaders are LangChain's way to **ingest data from any source** into a standard format (`Document` objects). They're the first step in any RAG pipeline.

```
PDF / Web / CSV / DB / API
         ↓
   [Document Loader]
         ↓
   List[Document]
     ├── page_content: str
     └── metadata: dict
```

---

## The Document Object

Every loader returns `List[Document]`:

```python
from langchain_core.documents import Document

doc = Document(
    page_content="The actual text content...",
    metadata={
        "source": "file.pdf",
        "page": 1,
        "author": "John",
        # Any key-value pairs
    }
)
```

**Metadata matters** — it enables filtering, citation, and debugging later.

---

## Loader Categories

| Category | Loaders | Use Case |
|----------|---------|----------|
| **Files** | PyPDFLoader, TextLoader, CSVLoader, JSONLoader | Local documents |
| **Web** | WebBaseLoader, AsyncHtmlLoader, SitemapLoader | Websites, blogs |
| **Database** | SQLDatabaseLoader | SQL tables |
| **Cloud** | S3FileLoader, GCSFileLoader, AzureBlobLoader | Cloud storage |
| **Apps** | NotionLoader, SlackLoader, GitLoader | SaaS tools |
| **Structured** | DataFrameLoader, AirtableLoader | Structured data |

---

## Key Loaders in Detail

### PyPDFLoader
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("report.pdf")
docs = loader.load()  # Each page = 1 Document
# docs[0].metadata = {"source": "report.pdf", "page": 0}
```
- One Document per page
- Preserves page numbers in metadata
- Alternative: `PyMuPDFLoader` (faster), `UnstructuredPDFLoader` (better formatting)

### WebBaseLoader
```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com/article")
docs = loader.load()
# Uses BeautifulSoup to extract text from HTML
```
- Strips HTML tags, returns clean text
- Supports multiple URLs in one call
- Metadata includes title, source URL

### CSVLoader
```python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader("data.csv", source_column="url")
docs = loader.load()
# Each row becomes one Document
# Row fields are in page_content as "column: value" format
```

### DirectoryLoader
```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    "./docs/",
    glob="**/*.pdf",           # File pattern
    loader_cls=PyPDFLoader,     # Which loader for matched files
    show_progress=True,
    use_multithreading=True,    # Parallel loading
)
```

---

## Lazy Loading (for large datasets)

```python
# load() puts everything in memory at once
docs = loader.load()  # ⚠️ May use too much RAM for large datasets

# lazy_load() yields documents one at a time
for doc in loader.lazy_load():
    process(doc)  # Process without loading all into memory
```

---

## Custom Metadata

Add your own metadata during loading:

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("notes.txt")
docs = loader.load()

# Enrich metadata
for doc in docs:
    doc.metadata["department"] = "engineering"
    doc.metadata["date_loaded"] = "2026-05-24"
    doc.metadata["confidential"] = False
```

---

## Best Practices

1. **Always check metadata** — different loaders provide different metadata fields
2. **Use lazy_load for large datasets** — avoid memory issues
3. **Add custom metadata early** — harder to add after vectorization
4. **Use DirectoryLoader for bulk** — with multithreading for speed
5. **Test with a small sample first** — verify loading works before processing thousands of files
6. **Handle encoding issues** — `TextLoader("file.txt", encoding="utf-8")`
7. **Consider the source format** — PDF tables may need specialized loaders (Unstructured, Camelot)

---

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Empty page_content | PDF is image-based (scanned) | Use OCR loader (Unstructured + Tesseract) |
| Garbled text | Wrong encoding | Set `encoding="utf-8"` or `"latin-1"` |
| Missing metadata | Loader doesn't extract it | Add manually after loading |
| Memory error | Loading too many files at once | Use `lazy_load()` |
| Slow loading | Many files sequentially | Use `use_multithreading=True` |
