# Advanced LangGraph Patterns — Deep Dive Notes

## Beyond Basic Agents

Basic agent = single LLM + tools in a loop. Advanced patterns:
- Human-in-the-loop (approval gates)
- Multi-agent systems (specialized workers)
- Subgraphs (modular composition)
- Streaming events (real-time UI updates)
- Error recovery (resilient agents)

---

## Pattern 1: Human-in-the-Loop (HITL)

Some actions are **irreversible** (send email, delete data, charge payment). Add approval before execution.

```
Agent decides: "I should send an email"
  ↓
[Approval Gate] → Human reviews → Approve/Reject
  ↓                                    ↓
[Execute tool]                    [Agent informed of rejection]
```

### Implementation
```python
# Classify tools by risk
safe_tools = [search, calculate]        # No approval needed
dangerous_tools = [send_email, delete]  # Requires approval

def check_danger(state) -> str:
    tool_calls = state["messages"][-1].tool_calls
    dangerous_names = {t.name for t in dangerous_tools}
    for tc in tool_calls:
        if tc["name"] in dangerous_names:
            return "needs_approval"
    return "safe"
```

### Production HITL with `interrupt_before`
```python
# LangGraph's built-in interrupt
agent = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["dangerous_tools"],  # Pauses here
)

# Execution pauses. You notify human.
# Human approves. You resume:
agent.invoke(None, config=config)  # Continues from checkpoint
```

### When to Use HITL
- Sending external communications (email, Slack, SMS)
- Modifying/deleting data
- Financial transactions
- Deploying code
- Any action with real-world consequences

---

## Pattern 2: Multi-Agent Systems

Multiple specialized agents coordinated by a supervisor.

```
                    ┌→ [Researcher Agent] → findings
[Supervisor] ──────┤→ [Coder Agent]      → code
                    └→ [Writer Agent]     → documentation
```

### Architectures

| Architecture | How It Works | Best For |
|-------------|-------------|----------|
| **Supervisor** | One LLM routes to specialized agents | Clear task categories |
| **Hierarchical** | Supervisors managing sub-supervisors | Complex orgs |
| **Collaborative** | Agents pass work to each other | Sequential workflows |
| **Debate** | Multiple agents argue, best answer wins | Critical decisions |

### Supervisor Pattern
```python
def supervisor(state) -> dict:
    """Decide which agent should work next."""
    decision = llm.invoke(
        f"Who should handle this: researcher, coder, or done? Task: {state['messages'][-1].content}"
    )
    return {"next_agent": decision}

# Routes based on supervisor's decision
graph.add_conditional_edges("supervisor", route_to_agent)
```

### When to Use Multi-Agent
- Tasks requiring different expertise (research + code + writing)
- Complex workflows with distinct phases
- When one agent's context window isn't enough
- Parallel subtask execution

---

## Pattern 3: Subgraphs (Modular Composition)

Break large graphs into reusable subgraphs:

```python
# Define a subgraph
research_subgraph = StateGraph(ResearchState)
research_subgraph.add_node(...)
compiled_research = research_subgraph.compile()

# Use in parent graph
parent_graph.add_node("research", compiled_research)
```

Benefits:
- Reusable across projects
- Testable in isolation
- Cleaner code organization
- Independent state management

---

## Pattern 4: Streaming Events

Show users what the agent is doing in real-time:

```python
# Stream mode: "updates" — shows node-by-node
for event in agent.stream(input, stream_mode="updates"):
    for node_name, data in event.items():
        if node_name == "agent":
            # Show: "Thinking..."
        elif node_name == "tools":
            # Show: "Searching documentation..."

# Stream mode: "messages" — token-by-token
async for event in agent.astream_events(input, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

### Frontend Integration (SSE)
```python
@app.get("/chat/stream")
async def stream_chat(message: str):
    async def event_generator():
        async for event in agent.astream_events(...):
            yield f"data: {json.dumps(event)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

## Pattern 5: Error Recovery

Agents will encounter errors. Build resilience:

### Approach 1: Error as ToolMessage
```python
def safe_tool_node(state):
    try:
        result = execute_tool(...)
    except Exception as e:
        result = f"Error: {e}. Please try an alternative approach."
    return {"messages": [ToolMessage(content=result, ...)]}
```
The agent sees the error and can retry or use a different tool.

### Approach 2: Retry with backoff
```python
@tool
def api_call(endpoint: str) -> str:
    """Call external API with automatic retry."""
    for attempt in range(3):
        try:
            return make_request(endpoint)
        except Exception:
            if attempt == 2:
                return "Service unavailable after 3 retries"
            time.sleep(2 ** attempt)
```

### Approach 3: Fallback tools
```python
@tool
def web_search(query: str) -> str:
    """Primary search tool."""
    ...

@tool
def cached_search(query: str) -> str:
    """Fallback: search cached results when web search fails."""
    ...
```

---

## Pattern 6: Recursion Limit (Prevent Infinite Loops)

```python
# Global limit
config = {"configurable": {"thread_id": "x"}, "recursion_limit": 25}
result = agent.invoke(input, config=config)

# If agent exceeds limit, raises GraphRecursionError
```

**Rule of thumb:** Set recursion_limit to 2x the expected max tool calls.
- Simple Q&A: limit=10
- Research tasks: limit=25
- Complex multi-step: limit=50

---

## Combining Patterns

Real-world agents combine multiple patterns:

```
[Supervisor + HITL + Streaming + Error Recovery]

1. Supervisor routes task to researcher agent
2. Researcher calls tools (with error recovery)
3. Researcher returns findings to supervisor
4. Supervisor routes to coder agent
5. Coder wants to deploy code → HITL gate
6. Human approves → code deployed
7. All steps streamed to frontend in real-time
```

---

## Best Practices

1. **Start simple, add complexity incrementally** — basic agent first, then HITL, then multi-agent
2. **HITL for all dangerous actions** — never let agents send emails/delete without approval
3. **Set recursion limits always** — prevent runaway costs
4. **Stream for UX** — users need feedback that the agent is working
5. **Handle errors as data** — return error strings, not exceptions
6. **Test each pattern in isolation** — multi-agent bugs are hard to debug
7. **Monitor per-agent costs** — supervisor calls add up fast
8. **Use checkpointing** — resume from failures without restarting

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| No recursion limit | Agent loops 100+ times, $50 bill | Always set `recursion_limit` |
| HITL without checkpointer | Can't pause/resume | HITL requires checkpointing |
| Multi-agent without clear routing | Tasks go to wrong agent | Explicit routing rules in supervisor prompt |
| No streaming in UI | User thinks app is broken | Show "Thinking...", "Searching..." |
| Silent errors | Agent gives wrong answer silently | Log all tool errors, surface to user |
| Supervisor too complex | Slow, expensive decisions | Keep supervisor prompt simple and focused |
