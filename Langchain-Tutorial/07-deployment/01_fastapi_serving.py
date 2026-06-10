"""
Phase 7: Deployment — FastAPI Serving
=======================================
Serve LangChain applications as REST APIs with FastAPI.

Topics covered:
- Basic FastAPI + LangChain integration
- Streaming responses (SSE)
- Session management
- Health checks and metadata
- Request/response models
- Background tasks
"""

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# 1. Basic FastAPI + LangChain Setup
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import uvicorn
import asyncio

app = FastAPI(
    title="LangChain API",
    description="Production LangChain service",
    version="1.0.0",
)

# Initialize LLM and chain (once, at startup)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chat_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise."),
        ("human", "{question}"),
    ])
    | llm
    | StrOutputParser()
)

# ============================================================
# 2. Request/Response Models
# ============================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    question: str = Field(min_length=1, max_length=5000, description="User question")
    session_id: str | None = Field(default=None, description="Optional session ID")
    temperature: float = Field(default=0.7, ge=0, le=2, description="Response creativity")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    session_id: str | None = None
    tokens_used: int | None = None
    model: str = "gpt-4o-mini"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model: str

# ============================================================
# 3. Basic Chat Endpoint
# ============================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Simple chat endpoint."""
    try:
        answer = await chat_chain.ainvoke({"question": request.question})
        return ChatResponse(
            answer=answer,
            session_id=request.session_id,
            model="gpt-4o-mini",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

# ============================================================
# 4. Streaming Endpoint (Server-Sent Events)
# ============================================================

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using SSE."""
    
    async def event_generator():
        try:
            async for chunk in chat_chain.astream({"question": request.question}):
                # SSE format: "data: <content>\n\n"
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

# ============================================================
# 5. RAG Endpoint
# ============================================================

class RAGRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    context: str = Field(default="", max_length=10000, description="Retrieved context")


class RAGResponse(BaseModel):
    answer: str
    sources_used: bool


rag_chain = (
    ChatPromptTemplate.from_messages([
        ("system", """Answer based on the context. If the answer isn't in the context, say so.
        
Context: {context}"""),
        ("human", "{question}"),
    ])
    | llm
    | StrOutputParser()
)


@app.post("/rag", response_model=RAGResponse)
async def rag_query(request: RAGRequest):
    """RAG endpoint with context."""
    try:
        answer = await rag_chain.ainvoke({
            "question": request.question,
            "context": request.context,
        })
        return RAGResponse(
            answer=answer,
            sources_used=bool(request.context),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# 6. Health Check & Metadata
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check for load balancers and monitoring."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model="gpt-4o-mini",
    )


@app.get("/")
async def root():
    """API documentation redirect."""
    return {"message": "LangChain API", "docs": "/docs"}

# ============================================================
# 7. Middleware: Request Logging
# ============================================================

from fastapi import Request
import time


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring."""
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    print(f"{request.method} {request.url.path} - {response.status_code} - {duration:.2f}s")
    
    # Add timing header
    response.headers["X-Response-Time"] = f"{duration:.3f}s"
    return response

# ============================================================
# 8. Rate Limiting Middleware
# ============================================================

from collections import defaultdict

request_counts: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = 30  # requests per minute


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Simple per-IP rate limiting."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    
    # Clean old requests
    request_counts[client_ip] = [
        t for t in request_counts[client_ip] if t > now - 60
    ]
    
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again in a minute.",
        )
    
    request_counts[client_ip].append(now)
    return await call_next(request)

# ============================================================
# 9. CORS (for frontend apps)
# ============================================================

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 10. Run the Server
# ============================================================

if __name__ == "__main__":
    print("Starting LangChain API server...")
    print("Docs: http://localhost:8000/docs")
    print("Health: http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000)

# To run:
#   python 01_fastapi_serving.py
# Or for development with auto-reload:
#   uvicorn 01_fastapi_serving:app --reload --port 8000
