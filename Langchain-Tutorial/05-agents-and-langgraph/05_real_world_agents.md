# Real-World Agent Patterns — Deep Dive Notes

## From Demo to Production

Demo agents work on happy paths. Production agents need:
- Reliable retrieval (RAG + tools)
- External API integration
- Testing and evaluation
- Cost control
- Deployment strategies

---

## Pattern 1: RAG Agent (Documentation Bot)

Combines knowledge retrieval with action-taking:

```
User: "I'm getting 429 errors on the API"
  ↓
Agent: [search_docs("rate limit 429")] → finds docs
Agent: [check_api_status("users")] → confirms service is up
Agent: "You're hitting the rate limit (100 req/min). Here's how to fix it: [from docs]"
```

### Implementation
```python
@tool
def search_docs(query: str) -> str:
    """Search knowledge base."""
    results = retriever.invoke(query)
    return "\n".join(r.page_content for r in results)

@tool
def check_status(service: str) -> str:
    """Check service health."""
    return call_status_api(service)

@tool
def create_ticket(title: str, description: str) -> str:
    """Escalate to human support."""
    return create_jira_ticket(title, description)

agent = create_react_agent(llm, [search_docs, check_status, create_ticket],
    prompt="Always search docs first. Escalate if you can't resolve.")
```

### Key Design Decisions
- **Always search before answering** — system prompt enforces this
- **Escalation path** — create_ticket for things the agent can't solve
- **Status checks** — verify services before suggesting "it's a bug"
- **Source citation** — include doc references in answers

---

## Pattern 2: API Agent

Agent that interacts with external APIs:

```python
@tool
def api_get(endpoint: str) -> str:
    """GET request to our API."""
    response = requests.get(f"{BASE_URL}{endpoint}", headers=auth_headers)
    return response.json()

@tool
def api_post(endpoint: str, data: str) -> str:
    """POST request. Data is JSON string."""
    response = requests.post(f"{BASE_URL}{endpoint}", json=json.loads(data), headers=auth_headers)
    return response.json()
```

### Safety Considerations for API Agents
- **Read-only by default** — only add write tools when needed
- **Rate limit the agent** — not just the API
- **Validate URLs** — don't let agent call arbitrary endpoints
- **Auth scoping** — agent's API key should have minimal permissions
- **Audit log** — record every API call the agent makes

---

## Pattern 3: Agent Evaluation

### Test Case Structure
```python
class AgentTestCase:
    input: str                      # User query
    expected_tools: list[str]       # Which tools should be called
    expected_in_output: list[str]   # Keywords in final answer
    max_iterations: int             # Shouldn't take more than N loops
```

### Evaluation Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Tool Accuracy** | Called the right tools? | >90% |
| **Answer Quality** | Contains expected info? | >85% |
| **Efficiency** | Minimal tool calls? | <5 per query |
| **Safety** | No dangerous actions without approval? | 100% |
| **Cost** | Tokens used per query | <10K avg |

### Automated Testing
```python
def test_agent(agent, test_cases):
    for tc in test_cases:
        result = agent.invoke({"messages": [HumanMessage(tc.input)]})
        
        # Assert tool usage
        tools_used = extract_tool_calls(result)
        assert set(tc.expected_tools) <= set(tools_used)
        
        # Assert output quality
        output = result["messages"][-1].content
        for keyword in tc.expected_in_output:
            assert keyword.lower() in output.lower()
        
        # Assert efficiency
        assert count_iterations(result) <= tc.max_iterations
```

### Building a Test Suite
1. Start with 20-30 representative queries
2. Include edge cases (out of scope, ambiguous, multi-step)
3. Include adversarial inputs (prompt injection attempts)
4. Run weekly to catch regressions
5. Add failing cases as you discover them in production

---

## Pattern 4: Cost Control

### Token Budget per Request
```python
class CostController:
    def __init__(self, max_input_tokens=5000, max_output_tokens=2000, max_tool_calls=10):
        self.budget = {"input": max_input_tokens, "output": max_output_tokens, "tools": max_tool_calls}
        self.usage = {"input": 0, "output": 0, "tools": 0}
    
    def check(self, category: str, tokens: int) -> bool:
        if self.usage[category] + tokens > self.budget[category]:
            raise BudgetExceeded(f"{category} budget exceeded")
        self.usage[category] += tokens
        return True
```

### Cost Estimation
```
Per agent invocation (GPT-4o-mini):
  - Avg 3 LLM calls × 1000 tokens input × $0.15/1M = $0.00045
  - Avg 3 LLM calls × 500 tokens output × $0.60/1M = $0.0009
  - Total ≈ $0.001-0.005 per query

At scale (10K queries/day):
  - $10-50/day
  - $300-1500/month
```

### Cost Optimization Strategies
1. **Limit recursion** — cap at 5-10 iterations
2. **Use smaller model for routing** — GPT-4o-mini for tool decisions
3. **Cache tool results** — same query = same result
4. **Batch similar requests** — don't call the same tool repeatedly
5. **Short-circuit obvious answers** — don't use tools for "What's 2+2?"

---

## Pattern 5: Deployment Architectures

### Architecture A: Monolithic API
```
[FastAPI Server]
  ├── POST /chat → agent.invoke()
  ├── GET /chat/stream → agent.astream_events()
  └── GET /sessions/{id} → load checkpointed state

Checkpointer: PostgreSQL
State: Stored in DB per session
Scale: Vertical (bigger server)
```

### Architecture B: Microservices
```
[API Gateway]
  ├── [Chat Service] → Routes requests
  ├── [Agent Service] → Runs LangGraph agents
  ├── [Tool Service] → Executes tools (isolated)
  └── [Memory Service] → Redis for state

Scale: Horizontal (add more instances)
```

### Architecture C: Serverless
```
[API Gateway (AWS/GCP)]
  → [Lambda/Cloud Function] → Agent execution
  → [DynamoDB/Firestore] → State persistence
  → [SQS/Pub/Sub] → Async tool execution

Scale: Auto (pay per invocation)
Best for: Bursty traffic, low baseline
```

### Choosing Architecture

| Factor | Monolithic | Microservices | Serverless |
|--------|-----------|---------------|------------|
| Complexity | Low | High | Medium |
| Cost (low traffic) | Medium | High | Low |
| Cost (high traffic) | Low | Medium | Medium |
| Latency | Lowest | Low | Cold starts |
| Team size needed | 1-2 | 3+ | 1-2 |
| Best for | MVPs, < 1K users | Scale, > 10K users | Variable traffic |

---

## Production Checklist

- [ ] **Rate limiting** — per user, per session, per minute
- [ ] **Cost budgets** — per request and per user/day
- [ ] **Error handling** — graceful degradation, fallback responses
- [ ] **Logging** — every tool call, every LLM response
- [ ] **Monitoring** — latency, error rate, cost per query
- [ ] **Testing** — automated test suite, run on every deploy
- [ ] **Security** — input validation, output sanitization, no prompt injection
- [ ] **HITL for dangerous actions** — send email, delete, pay
- [ ] **Timeouts** — don't let agents run forever
- [ ] **User feedback** — thumbs up/down on answers

---

## Best Practices

1. **RAG Agent > Pure LLM** — always ground answers in retrieved docs
2. **Test before deploy** — 30+ test cases minimum
3. **Monitor costs daily** — set alerts for budget breaches
4. **Stream in production** — users need feedback
5. **Start monolithic** — split to microservices only when needed
6. **Checkpoint everything** — resume from any failure point
7. **Log tool calls** — essential for debugging production issues
8. **Human escalation** — agents can't solve everything, have a fallback path
