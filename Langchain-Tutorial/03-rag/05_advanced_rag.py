"""
Phase 3: RAG — Advanced RAG Techniques
========================================
Improve retrieval quality with advanced strategies.

Topics covered:
- Hybrid search (keyword + semantic)
- Re-ranking
- Parent-child document retrieval
- Self-query retriever
- Evaluation with RAGAS
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ============================================================
# 1. Hybrid Search (BM25 keyword + semantic)
# ============================================================

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

documents = [
    Document(page_content="LangChain provides document loaders for PDF, CSV, and web pages.", metadata={"topic": "loaders"}),
    Document(page_content="The RecursiveCharacterTextSplitter splits by paragraphs, then sentences, then words.", metadata={"topic": "splitters"}),
    Document(page_content="OpenAI's text-embedding-3-small model creates 1536-dimensional vectors.", metadata={"topic": "embeddings"}),
    Document(page_content="Chroma is an open-source vector database that runs locally.", metadata={"topic": "vectordb"}),
    Document(page_content="FAISS (Facebook AI Similarity Search) is optimized for fast similarity search.", metadata={"topic": "vectordb"}),
    Document(page_content="RAG evaluation metrics include faithfulness, relevancy, and context recall.", metadata={"topic": "evaluation"}),
    Document(page_content="The multi-query retriever generates multiple variations of a question to improve recall.", metadata={"topic": "retrieval"}),
    Document(page_content="Contextual compression removes irrelevant parts of retrieved documents.", metadata={"topic": "retrieval"}),
]

# BM25: keyword-based retriever (good for exact matches)
bm25_retriever = BM25Retriever.from_documents(documents, k=3)

# Semantic: embedding-based retriever (good for meaning)
vectorstore = Chroma.from_documents(documents, embeddings, collection_name="hybrid_demo")
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Ensemble: combines both (hybrid search)
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.4, 0.6],  # 40% keyword, 60% semantic
)

print("=== Hybrid Search ===")
query = "FAISS vector database"

bm25_results = bm25_retriever.invoke(query)
semantic_results = semantic_retriever.invoke(query)
hybrid_results = hybrid_retriever.invoke(query)

print(f"BM25 (keyword): {[d.page_content[:40] for d in bm25_results]}")
print(f"Semantic: {[d.page_content[:40] for d in semantic_results]}")
print(f"Hybrid: {[d.page_content[:40] for d in hybrid_results]}")
print()

# ============================================================
# 2. Contextual Compression
# ============================================================

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Compressor: uses LLM to extract only relevant parts
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=semantic_retriever,
)

print("=== Contextual Compression ===")
query = "What dimensions do OpenAI embeddings have?"
compressed_docs = compression_retriever.invoke(query)
for doc in compressed_docs:
    print(f"  Compressed: {doc.page_content}")
print()

# ============================================================
# 3. Self-Query Retriever (natural language metadata filtering)
# ============================================================

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo

# Define metadata fields the retriever can filter on
metadata_field_info = [
    AttributeInfo(
        name="topic",
        description="The topic category: loaders, splitters, embeddings, vectordb, evaluation, retrieval",
        type="string",
    ),
]

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Technical documentation about LangChain RAG components",
    metadata_field_info=metadata_field_info,
)

print("=== Self-Query Retriever ===")
# The LLM automatically extracts the filter from natural language
results = self_query_retriever.invoke("Tell me about vector databases")
for doc in results:
    print(f"  [{doc.metadata.get('topic')}] {doc.page_content[:60]}...")
print()

# ============================================================
# 4. Parent-Child Document Retrieval
# ============================================================

from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

# Concept: Embed SMALL chunks for precision, but return LARGE parent for context

# The full documents (parents)
full_documents = [
    Document(page_content="""LangChain Document Loaders: LangChain provides over 100 document loaders. 
These include PDF loaders (PyPDF, PDFMiner), web loaders (WebBaseLoader, AsyncHtmlLoader), 
database loaders (SQLDatabase), and file loaders (CSV, JSON, Text). Each loader returns 
a list of Document objects with page_content and metadata fields.""", metadata={"source": "ch1"}),
    Document(page_content="""Vector Stores in LangChain: Vector stores are databases optimized for 
storing and searching embeddings. Popular options include Chroma (local, open-source), 
FAISS (Facebook's fast similarity search), Pinecone (managed cloud service), and Weaviate 
(open-source with hybrid search). All implement the same VectorStore interface.""", metadata={"source": "ch2"}),
]

# Small splitter for child chunks (what gets embedded)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)

# Storage for parent documents
store = InMemoryStore()

parent_retriever = ParentDocumentRetriever(
    vectorstore=Chroma(embedding_function=embeddings, collection_name="parents"),
    docstore=store,
    child_splitter=child_splitter,
)

# Add documents (splits into children, stores parents)
parent_retriever.add_documents(full_documents)

print("=== Parent-Child Retrieval ===")
# Search matches small chunk, but returns full parent document
results = parent_retriever.invoke("What is Pinecone?")
for doc in results:
    print(f"  Full parent returned ({len(doc.page_content)} chars): {doc.page_content[:100]}...")
print()

# ============================================================
# 5. RAG Evaluation (basic metrics)
# ============================================================

def evaluate_retrieval(retriever, test_cases):
    """Basic retrieval evaluation: checks if relevant doc is in top-k."""
    hits = 0
    total = len(test_cases)
    
    for query, expected_topic in test_cases:
        results = retriever.invoke(query)
        retrieved_topics = [doc.metadata.get("topic") for doc in results]
        
        if expected_topic in retrieved_topics:
            hits += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"  {status} Query: '{query[:40]}' → Expected: {expected_topic}, Got: {retrieved_topics}")
    
    accuracy = hits / total
    print(f"\n  Retrieval Accuracy: {hits}/{total} = {accuracy:.0%}")
    return accuracy

# Test cases: (query, expected_topic_in_results)
test_cases = [
    ("How to load PDF files?", "loaders"),
    ("What is FAISS?", "vectordb"),
    ("How to measure RAG quality?", "evaluation"),
    ("How does text splitting work?", "splitters"),
    ("What is multi-query retrieval?", "retrieval"),
]

print("=== Retrieval Evaluation ===")
evaluate_retrieval(semantic_retriever, test_cases)
