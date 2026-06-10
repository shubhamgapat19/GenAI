"""
Phase 3: RAG — Document Loaders
=================================
Load documents from various sources into LangChain.

Topics covered:
- PDF loading
- Web page loading
- CSV/Text file loading
- Directory loading (bulk)
"""

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. PDF Loader
# ============================================================

from langchain_community.document_loaders import PyPDFLoader

# Load a PDF file (each page becomes a Document)
pdf_loader = PyPDFLoader("sample_data/sample.pdf")
pdf_docs = pdf_loader.load()

print("=== PDF Loader ===")
print(f"Number of pages: {len(pdf_docs)}")
print(f"First page content (first 200 chars): {pdf_docs[0].page_content[:200]}")
print(f"Metadata: {pdf_docs[0].metadata}")
print()

# ============================================================
# 2. Web Page Loader
# ============================================================

from langchain_community.document_loaders import WebBaseLoader

# Load content from a web page
web_loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")
web_docs = web_loader.load()

print("=== Web Loader ===")
print(f"Number of documents: {len(web_docs)}")
print(f"Content (first 300 chars): {web_docs[0].page_content[:300]}")
print(f"Metadata: {web_docs[0].metadata}")
print()

# ============================================================
# 3. CSV Loader
# ============================================================

from langchain_community.document_loaders.csv_loader import CSVLoader

# Each row becomes a Document
csv_loader = CSVLoader("sample_data/data.csv")
csv_docs = csv_loader.load()

print("=== CSV Loader ===")
print(f"Number of rows: {len(csv_docs)}")
if csv_docs:
    print(f"First row: {csv_docs[0].page_content[:200]}")
    print(f"Metadata: {csv_docs[0].metadata}")
print()

# ============================================================
# 4. Text File Loader
# ============================================================

from langchain_community.document_loaders import TextLoader

text_loader = TextLoader("sample_data/notes.txt", encoding="utf-8")
text_docs = text_loader.load()

print("=== Text Loader ===")
print(f"Content (first 200 chars): {text_docs[0].page_content[:200]}")
print()

# ============================================================
# 5. Directory Loader (bulk load all files in a folder)
# ============================================================

from langchain_community.document_loaders import DirectoryLoader

# Load all .txt files from a directory
dir_loader = DirectoryLoader(
    "sample_data/",
    glob="**/*.txt",       # Pattern to match
    show_progress=True,     # Progress bar
)
dir_docs = dir_loader.load()

print("=== Directory Loader ===")
print(f"Loaded {len(dir_docs)} documents from directory")
for doc in dir_docs[:3]:
    print(f"  - {doc.metadata['source']}: {doc.page_content[:50]}...")
print()

# ============================================================
# 6. Multiple URLs at once
# ============================================================

urls = [
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/",
]

multi_loader = WebBaseLoader(urls)
multi_docs = multi_loader.load()

print("=== Multiple URLs ===")
print(f"Loaded {len(multi_docs)} documents from {len(urls)} URLs")
for doc in multi_docs:
    print(f"  - {doc.metadata.get('title', 'No title')[:50]}")
print()

# ============================================================
# 7. Document Object Structure
# ============================================================

# Every loader returns List[Document]
# Document has two fields:
#   - page_content: str (the actual text)
#   - metadata: dict (source info, page number, etc.)

from langchain_core.documents import Document

# You can also create documents manually
custom_doc = Document(
    page_content="This is my custom document content.",
    metadata={"source": "manual", "author": "Shubham", "topic": "LangChain"}
)

print("=== Document Structure ===")
print(f"Content: {custom_doc.page_content}")
print(f"Metadata: {custom_doc.metadata}")
