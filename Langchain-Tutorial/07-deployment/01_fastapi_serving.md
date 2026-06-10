# FastAPI Serving — Deep Dive Notes

## Why FastAPI for LLM Apps?

FastAPI is the standard choice for serving LangChain applications:
- **Async-native:** Handles concurrent LLM calls efficiently
- **Streaming support:** SSE (Server-Sent Events) built-in
- **Auto-docs:** Swagger UI at `/docs` for free
- **Pydantic integration:** Request/response validation
- **Production-ready:** Used by Netflix, Uber, Microsoft

---

## Architecture: API + LangChain

```
Client (browser/app)
    ↓ HTTP POST /chat
[FastAPI Server]
    ↓ invoke/astream
[LangChain Chain/Agent]
    ↓ API call
[OpenAI / LLM Provider]
    ↓ response
[FastAPI Server]
    ↓ HTTP Response / SSE Stream
Client
```

---

## Basic Structure

```python
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

# Initialize ONCE at startup (not per request!)
llm = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | llm | StrOutputParser()

@app.post("/chat")
async def chat(request: ChatRequest):
    answer = await chain.ainvoke({"question": request.question})
    return {"answer": answer}
```

**Critical:** Initialize LLM/chains at module level, not inside request handlers.

---

## Request/Response Models

Always use Pydantic models for validation:

```python
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    session_id: str | None = None
    temperature: float = Field(default=0.7, ge=0, le=2)

class ChatResponse(BaseModel):
    answer: str
    tokens_used: int | None = None
    model: str = "gpt-4o-mini"
```

Benefits:
- Auto-validates input (rejects bad requests with 422)
- Generates OpenAPI schema
- Self-documenting API

---

## Streaming with SSE

For real-time token-by-token responses:

```python
from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        async for chunk in chain.astream({"question": request.question}):
            yield f"data: {chunk}\n\n"  # SSE format
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
```

### Client-side consumption (JavaScript):
```javascript
const eventSource = new EventSource('/chat/stream');
eventSource.onmessage = (e) => {
    if (e.data === '[DONE]') { eventSource.close(); return; }
    appendToUI(e.data);
};
```

---

## Essential Middleware

### 1. Request Logging
```python
@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    print(f"{request.method} {request.url.path} - {response.status_code} - {duration:.2f}s")
    return response
```

### 2. Rate Limiting
```python
@app.middleware("http")
async def rate_limit(request, call_next):
    ip = request.client.host
    if too_many_requests(ip):
        raise HTTPException(429, "Rate limit exceeded")
    return await call_next(request)
```

### 3. CORS (for frontend apps)
```python
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"])
```

---

## Health Check Endpoint

Required for load balancers and monitoring:

```python
@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}
```

Load balancers ping this every 30s. If it fails 3 times → instance removed from rotation.

---

## Error Handling

```python
@app.exception_handler(Exception)
async def global_error_handler(request, exc):
    # Log the real error internally
    logger.error(f"Unhandled error: {exc}")
    
    # Return safe message to user
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong. Please try again."},
    )
```

**Never expose internal errors to clients.**

---

## Running in Production

```bash
# Development (auto-reload)
uvicorn main:app --reload --port 8000

# Production (multiple workers)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn (recommended)
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## Best Practices

1. **async everywhere** — use `ainvoke`, `astream` for non-blocking
2. **Initialize at startup** — LLM, chains, DB connections at module level
3. **Pydantic models** — validate all inputs and outputs
4. **Stream for chat** — always use SSE for user-facing chat
5. **Health endpoint** — required for any deployment platform
6. **Rate limit** — protect from abuse and cost spikes
7. **CORS** — enable only for your frontend origins
8. **Error handling** — never expose internals to clients

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Init LLM per request | Slow, wastes connections | Init at module level |
| No input validation | Injection, token bombs | Pydantic models |
| Sync endpoints | Blocks other requests | Use async/await |
| No health check | Can't deploy behind LB | Add /health endpoint |
| No rate limit | Cost explosion from abuse | Middleware rate limiter |
| Missing CORS | Frontend can't connect | Add CORSMiddleware |
| Exposing errors | Security risk | Generic error messages |
| No streaming | Bad chat UX | Use SSE for chat |
