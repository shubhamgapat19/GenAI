"""
Phase 5: Agents & LangGraph — Real-World Agent Patterns
=========================================================
Production-ready agent architectures.

Topics covered:
- RAG Agent (retrieval + tools)
- API Agent (calling external services)
- Agent evaluation and testing
- Rate limiting and cost control
- Agent deployment patterns
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ============================================================
# 1. RAG Agent (retrieval as a tool)
# ============================================================

from langchain_chroma import Chroma

# Create a knowledge base
docs = [
    Document(page_content="Our API rate limit is 100 requests per minute per user. Enterprise plans get 1000/min.", metadata={"source": "api_docs"}),
    Document(page_content="Authentication uses OAuth 2.0 with JWT tokens. Tokens expire after 1 hour.", metadata={"source": "auth_docs"}),
    Document(page_content="The /users endpoint supports GET, POST, PUT, DELETE. Pagination uses cursor-based approach.", metadata={"source": "api_docs"}),
    Document(page_content="Error codes: 429=Rate Limited, 401=Unauthorized, 403=Forbidden, 500=Server Error.", metadata={"source": "error_docs"}),
    Document(page_content="Webhooks are sent for events: user.created, order.completed, payment.failed. Retry 3 times with exponential backoff.", metadata={"source": "webhook_docs"}),
]

vectorstore = Chroma.from_documents(docs, embeddings, collection_name="company_docs")
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


@tool
def search_documentation(query: str) -> str:
    """Search company documentation for technical information.
    
    Args:
        query: What to search for in the docs.
    """
    results = retriever.invoke(query)
    return "\n".join(f"[{r.metadata['source']}] {r.page_content}" for r in results)


@tool
def check_api_status(service: str) -> str:
    """Check the status of an API service.
    
    Args:
        service: Service name to check (auth, users, payments, webhooks).
    """
    statuses = {
        "auth": "✅ Operational (99.9% uptime last 30d)",
        "users": "✅ Operational (99.7% uptime last 30d)",
        "payments": "⚠️ Degraded (high latency detected)",
        "webhooks": "✅ Operational (99.8% uptime last 30d)",
    }
    return statuses.get(service, f"Unknown service: {service}")


@tool
def create_support_ticket(title: str, description: str, priority: str) -> str:
    """Create a support ticket for issues that need human help.
    
    Args:
        title: Ticket title.
        description: Detailed description of the issue.
        priority: Priority level: low, medium, high, critical.
    """
    return f"Ticket created: '{title}' (Priority: {priority}) - ID: TICK-{hash(title) % 10000}"


# RAG Agent — combines knowledge search with actions
rag_agent = create_react_agent(
    llm,
    [search_documentation, check_api_status, create_support_ticket],
    prompt="""You are a technical support agent for our API platform.

Rules:
1. ALWAYS search documentation before answering technical questions
2. Check service status if the user reports issues
3. Create support tickets for problems you cannot resolve
4. Be specific and cite documentation sources
5. If unsure, say so rather than guessing""",
)

print("=== RAG Agent (Support Bot) ===")

result = rag_agent.invoke({
    "messages": [HumanMessage(content="What's the rate limit for our API? I'm getting 429 errors.")]
})
print(f"Q: Rate limit and 429 errors?")
print(f"A: {result['messages'][-1].content}")
print()

result = rag_agent.invoke({
    "messages": [HumanMessage(content="Our payment webhooks aren't being received. What should I do?")]
})
print(f"Q: Payment webhooks not received?")
print(f"A: {result['messages'][-1].content}")
print()

# ============================================================
# 2. API Agent (calling external services)
# ============================================================

import json


@tool
def http_get(url: str) -> str:
    """Make an HTTP GET request to an API endpoint.
    
    Args:
        url: The URL to send the GET request to.
    """
    # Simulated API responses
    responses = {
        "/api/users": json.dumps({"users": [{"id": 1, "name": "Shubham"}, {"id": 2, "name": "Priya"}], "total": 2}),
        "/api/orders": json.dumps({"orders": [{"id": 101, "status": "shipped"}, {"id": 102, "status": "pending"}]}),
        "/api/health": json.dumps({"status": "healthy", "version": "2.1.0", "uptime": "99.9%"}),
    }
    for path, response in responses.items():
        if path in url:
            return response
    return json.dumps({"error": "Not found"})


@tool
def http_post(url: str, body: str) -> str:
    """Make an HTTP POST request to an API endpoint.
    
    Args:
        url: The URL to send the POST request to.
        body: JSON string body for the request.
    """
    return json.dumps({"success": True, "message": f"Created resource at {url}"})


@tool
def format_json(data: str) -> str:
    """Pretty-print and format a JSON string.
    
    Args:
        data: JSON string to format.
    """
    try:
        parsed = json.loads(data)
        return json.dumps(parsed, indent=2)
    except:
        return f"Invalid JSON: {data}"


api_agent = create_react_agent(
    llm,
    [http_get, http_post, format_json],
    prompt="You are an API integration assistant. Help users interact with APIs. Format responses nicely.",
)

print("=== API Agent ===")
result = api_agent.invoke({
    "messages": [HumanMessage(content="Check the health of our API and list all users.")]
})
print(f"A: {result['messages'][-1].content}")
print()

# ============================================================
# 3. Agent Evaluation and Testing
# ============================================================

class AgentTestCase(BaseModel):
    """A test case for agent evaluation."""
    input: str
    expected_tools: list[str] = Field(description="Tools that should be called")
    expected_in_output: list[str] = Field(description="Keywords expected in final answer")


def evaluate_agent(agent, test_cases: list[AgentTestCase]) -> dict:
    """Run agent test suite and report results."""
    results = {"passed": 0, "failed": 0, "details": []}
    
    for i, tc in enumerate(test_cases, 1):
        result = agent.invoke({"messages": [HumanMessage(content=tc.input)]})
        
        # Check tool usage
        tools_used = set()
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for t in msg.tool_calls:
                    tools_used.add(t["name"])
        
        # Check output contains expected keywords
        final_output = result["messages"][-1].content.lower()
        keywords_found = [kw for kw in tc.expected_in_output if kw.lower() in final_output]
        
        # Evaluate
        tools_ok = all(t in tools_used for t in tc.expected_tools)
        output_ok = len(keywords_found) >= len(tc.expected_in_output) * 0.5  # 50% threshold
        passed = tools_ok and output_ok
        
        results["passed" if passed else "failed"] += 1
        status = "✅" if passed else "❌"
        results["details"].append(f"  {status} Test {i}: '{tc.input[:40]}...' | Tools: {tools_ok} | Output: {output_ok}")
    
    return results


# Define test suite
test_cases = [
    AgentTestCase(
        input="What's the rate limit for our API?",
        expected_tools=["search_documentation"],
        expected_in_output=["100", "rate"],
    ),
    AgentTestCase(
        input="Is the auth service working?",
        expected_tools=["check_api_status"],
        expected_in_output=["operational"],
    ),
    AgentTestCase(
        input="I can't login and nothing in docs helps. Create a ticket.",
        expected_tools=["search_documentation", "create_support_ticket"],
        expected_in_output=["ticket"],
    ),
]

print("=== Agent Evaluation ===")
results = evaluate_agent(rag_agent, test_cases)
print(f"Results: {results['passed']} passed, {results['failed']} failed")
for detail in results["details"]:
    print(detail)
print()

# ============================================================
# 4. Rate Limiting and Cost Control
# ============================================================

import time
from functools import wraps


class AgentRateLimiter:
    """Rate limiter for agent tool calls."""
    
    def __init__(self, max_tool_calls: int = 10, max_llm_calls: int = 5):
        self.max_tool_calls = max_tool_calls
        self.max_llm_calls = max_llm_calls
        self.tool_calls = 0
        self.llm_calls = 0
    
    def check_tool_limit(self) -> bool:
        self.tool_calls += 1
        if self.tool_calls > self.max_tool_calls:
            raise Exception(f"Tool call limit exceeded ({self.max_tool_calls})")
        return True
    
    def check_llm_limit(self) -> bool:
        self.llm_calls += 1
        if self.llm_calls > self.max_llm_calls:
            raise Exception(f"LLM call limit exceeded ({self.max_llm_calls})")
        return True
    
    def reset(self):
        self.tool_calls = 0
        self.llm_calls = 0
    
    def report(self) -> str:
        return f"Tool calls: {self.tool_calls}/{self.max_tool_calls}, LLM calls: {self.llm_calls}/{self.max_llm_calls}"


print("=== Rate Limiting Pattern ===")
limiter = AgentRateLimiter(max_tool_calls=10, max_llm_calls=5)
print(f"Limits: {limiter.report()}")
print("In production: wrap tool execution with limiter.check_tool_limit()")
print("Prevents runaway agents from making unlimited API calls")
print()

# ============================================================
# 5. Agent Deployment Patterns
# ============================================================

print("=== Deployment Patterns ===")
print()
print("Pattern 1: FastAPI + LangGraph")
print("  - Expose agent as REST API endpoint")
print("  - POST /chat with session_id + message")
print("  - Use Redis checkpointer for distributed state")
print()
print("Pattern 2: Streaming via SSE")
print("  - Stream agent events to frontend in real-time")
print("  - User sees: 'Searching docs...' → 'Found 3 results...' → 'Here's your answer'")
print()
print("Pattern 3: Background Jobs")
print("  - For long-running agents (research, data processing)")
print("  - Queue with Celery/Bull, webhook on completion")
print()
print("Pattern 4: Serverless")
print("  - AWS Lambda / Cloud Functions for stateless agents")
print("  - External state (DynamoDB/Redis) for memory")
print()

# Example: FastAPI endpoint structure
fastapi_example = '''
# app.py — FastAPI agent server

from fastapi import FastAPI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver

app = FastAPI()
checkpointer = PostgresSaver(conn_string=DATABASE_URL)
agent = create_react_agent(llm, tools).compile(checkpointer=checkpointer)

@app.post("/chat")
async def chat(session_id: str, message: str):
    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )
    return {"response": result["messages"][-1].content}
'''

print("FastAPI Example:")
print(fastapi_example)
