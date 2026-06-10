"""
Phase 3: RAG — Retrieval Chains
=================================
Build the complete RAG pipeline: retrieve → augment → generate.

Topics covered:
- Basic RAG chain
- Retriever types
- Multi-query retriever
- Contextual compression
- Source citations
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_chroma import Chroma

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ============================================================
# Setup: Create a knowledge base
# ============================================================

documents = [
    Document(page_content="LangChain was created by Harrison Chase in October 2022. It is an open-source framework for building applications with large language models.", metadata={"source": "wiki", "page": 1}),
    Document(page_content="LCEL (LangChain Expression Language) uses the pipe operator to compose chains. Components include prompts, models, parsers, and retrievers.", metadata={"source": "docs", "page": 5}),
    Document(page_content="RAG (Retrieval Augmented Generation) enhances LLM responses by retrieving relevant documents from a knowledge base before generating answers.", metadata={"source": "docs", "page": 10}),
    Document(page_content="LangGraph is LangChain's framework for building stateful, multi-agent applications. It uses a graph-based approach with nodes and edges.", metadata={"source": "docs", "page": 15}),
    Document(page_content="LangSmith is the observability platform for LangChain. It provides tracing, evaluation, and monitoring for LLM applications.", metadata={"source": "docs", "page": 20}),
    Document(page_content="Vector stores like Chroma, FAISS, and Pinecone store document embeddings for efficient similarity search during retrieval.", metadata={"source": "tutorial", "page": 3}),
    Document(page_content="Agents in LangChain can use tools to perform actions. They decide which tool to use based on the user's question and available tools.", metadata={"source": "docs", "page": 25}),
    Document(page_content="Text splitting is crucial for RAG. RecursiveCharacterTextSplitter is the recommended default, splitting by paragraphs, sentences, then words.", metadata={"source": "tutorial", "page": 7}),
]

vectorstore = Chroma.from_documents(documents, embeddings, collection_name="rag_demo")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ============================================================
# 1. Basic RAG Chain
# ============================================================

# The core RAG pattern: retrieve context → format prompt → generate
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer questions based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have enough information to answer that."

Context:
{context}"""),
    ("human", "{question}"),
])

def format_docs(docs):
    """Combine retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,      # Retrieve → format
        "question": RunnablePassthrough(),         # Pass question through
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

print("=== Basic RAG Chain ===")
answer = rag_chain.invoke("Who created LangChain and when?")
print(f"Q: Who created LangChain and when?")
print(f"A: {answer}")
print()

answer = rag_chain.invoke("What is LCEL?")
print(f"Q: What is LCEL?")
print(f"A: {answer}")
print()

# Question not in knowledge base
answer = rag_chain.invoke("What is the capital of France?")
print(f"Q: What is the capital of France?")
print(f"A: {answer}")
print()

# ============================================================
# 2. RAG with Source Citations
# ============================================================

rag_with_sources_prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer based on the context below. Include [Source: X, Page: Y] citations.

Context:
{context}"""),
    ("human", "{question}"),
])

def format_docs_with_sources(docs):
    """Format docs with source metadata for citation."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)

rag_sources_chain = (
    {
        "context": retriever | format_docs_with_sources,
        "question": RunnablePassthrough(),
    }
    | rag_with_sources_prompt
    | llm
    | StrOutputParser()
)

print("=== RAG with Citations ===")
answer = rag_sources_chain.invoke("What is LangGraph?")
print(f"Q: What is LangGraph?")
print(f"A: {answer}")
print()

# ============================================================
# 3. Multi-Query Retriever
# ============================================================

from langchain.retrievers.multi_query import MultiQueryRetriever

# Generates multiple query variations to improve retrieval
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
)

# This generates 3 variations of the query, retrieves for each, deduplicates
print("=== Multi-Query Retriever ===")
multi_docs = multi_retriever.invoke("How do I observe my LLM app?")
print(f"Retrieved {len(multi_docs)} unique documents")
for doc in multi_docs:
    print(f"  - {doc.page_content[:70]}...")
print()

# ============================================================
# 4. RAG Chain — Return Docs + Answer Together
# ============================================================

from langchain_core.runnables import RunnableParallel

# Sometimes you need both the answer AND the source documents
rag_with_docs_chain = RunnableParallel(
    answer=rag_chain,
    source_documents=retriever,
)

print("=== RAG with Returned Documents ===")
result = rag_with_docs_chain.invoke("What vector stores does LangChain support?")
print(f"Answer: {result['answer']}")
print(f"Sources used:")
for doc in result['source_documents']:
    print(f"  - [{doc.metadata['source']}] {doc.page_content[:50]}...")
print()

# ============================================================
# 5. Conversational RAG (with follow-up questions)
# ============================================================

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Reformulate follow-up questions to be standalone
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given the chat history and a follow-up question, 
reformulate it as a standalone question that can be understood without context.
If it's already standalone, return it as-is."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

contextualize_chain = contextualize_prompt | llm | StrOutputParser()

# Full conversational RAG
def conversational_rag(question: str, chat_history: list = []):
    """RAG that handles follow-up questions."""
    # If there's history, reformulate the question
    if chat_history:
        standalone_question = contextualize_chain.invoke({
            "input": question,
            "chat_history": chat_history,
        })
    else:
        standalone_question = question
    
    # Run RAG with the standalone question
    answer = rag_chain.invoke(standalone_question)
    return answer

print("=== Conversational RAG ===")
# First question
history = []
q1 = "What is LangSmith?"
a1 = conversational_rag(q1, history)
print(f"Q1: {q1}")
print(f"A1: {a1}\n")

# Follow-up (refers to previous context)
history = [HumanMessage(content=q1), AIMessage(content=a1)]
q2 = "What features does it provide?"  # "it" = LangSmith
a2 = conversational_rag(q2, history)
print(f"Q2: {q2}")
print(f"A2: {a2}")
