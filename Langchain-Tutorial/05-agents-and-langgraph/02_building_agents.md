# Building Agents with LangGraph — Deep Dive Notes

## What Is an Agent?

An agent is an LLM that **decides its own actions** in a loop:

```
Regular chain:  Input → Process → Output  (one shot)
Agent:          Input → Think → Act → Observe → Think → Act → ... → Output  (loop)
```

The key difference: **the LLM decides what to do next** rather than following a fixed pipeline.

---

## The ReAct Pattern

**Re**asoning + **Act**ing — the most common agent architecture.

```
Step 1 (Reason):  "The user wants compound interest. I need the calculator."
Step 2 (Act):     calculator("1000 * (1.08 ** 5)")
Step 3 (Observe): "1469.33"
Step 4 (Reason):  "I have the answer. Let me format it nicely."
Step 5 (Answer):  "With 8% annual interest, $1000 becomes $1,469.33 after 5 years."
```

---

## LangGraph Agent Architecture

```
        ┌───────────────────────────────┐
        │                               │
START → [Agent Node] ──tool_calls?──→ [Tool Node]
             │                          │
             │ no tool_calls            │ (results)
             ↓                          │
            END ←───────────────────────┘
                        (loops back)
```

### Components
1. **Agent Node:** Calls LLM with tools bound. Returns tool_calls or final answer.
2. **Tool Node:** Executes requested tools. Returns ToolMessages.
3. **Router:** Checks if agent wants more tools or is done.
4. **Cycle:** tools → agent → tools → agent → ... → end

---

## Building from Scratch

### Step 1: Define State
```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

### Step 2: Agent Node
```python
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
```

### Step 3: Tool Node
```python
from langgraph.prebuilt import ToolNode
tool_node = ToolNode(tools)
```

### Step 4: Router
```python
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return "end"
```

### Step 5: Assemble Graph
```python
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")  # THE CYCLE
agent = graph.compile()
```

---

## Prebuilt Agent (One-liner)

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, tools)
# That's it! Same as building from scratch above.

result = agent.invoke({"messages": [HumanMessage("What's 2+2?")]})
```

### With System Prompt
```python
agent = create_react_agent(
    llm,
    tools,
    prompt="You are a financial analyst. Always show your work step by step.",
)
```

### With Checkpointing (memory)
```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_react_agent(llm, tools, checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "session_1"}}
agent.invoke({"messages": [...]}, config=config)
```

---

## Agent Execution Trace

When the agent runs, here's what happens internally:

```python
result = agent.invoke({"messages": [HumanMessage("What's sqrt(144) + 2^8?")]})

# Internal trace:
# 1. [Agent] → LLM says: call calculator("math.sqrt(144)")
# 2. [Tools] → Executes: 12.0
# 3. [Agent] → LLM says: call calculator("2**8")
# 4. [Tools] → Executes: 256
# 5. [Agent] → LLM says: call calculator("12 + 256")
# 6. [Tools] → Executes: 268
# 7. [Agent] → LLM says: "√144 + 2⁸ = 12 + 256 = 268" (no more tool_calls → END)
```

---

## ToolNode Details

`ToolNode` handles:
- Looking up the correct tool by name
- Executing it with the provided args
- Creating `ToolMessage` with correct `tool_call_id`
- Handling parallel tool calls (executes all)

```python
# Equivalent manual implementation:
def tool_node(state):
    tool_messages = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_fn = tools_dict[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    return {"messages": tool_messages}
```

---

## Streaming Agent Output

See the agent think in real-time:

```python
# Stream individual node updates
for event in agent.stream({"messages": [...]}, stream_mode="updates"):
    for node, data in event.items():
        print(f"[{node}] {data}")

# Stream token-by-token from LLM
for event in agent.stream({"messages": [...]}, stream_mode="messages"):
    # Each token as it's generated
    print(event.content, end="")
```

---

## Agent vs Chain — When to Use Each

| Use a Chain | Use an Agent |
|-------------|-------------|
| Fixed steps known in advance | Steps depend on input/results |
| One tool call max | Multiple tools, chosen dynamically |
| Predictable cost | Variable cost (acceptable) |
| Low latency required | Latency acceptable for quality |
| Simple Q&A | Complex research/reasoning |

---

## Best Practices

1. **Start with `create_react_agent`** — only build custom graph when needed
2. **Keep tools focused** — each tool does ONE thing well
3. **Set max iterations** — prevent infinite loops (agent calling tools forever)
4. **Add system prompt** — guide the agent's behavior and personality
5. **Use checkpointing** — for multi-turn conversations
6. **Monitor costs** — each loop iteration = LLM call = $$$
7. **Test with diverse inputs** — agents are less predictable than chains
8. **Log everything** — tool calls, arguments, results, final answers

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| No max iterations | Agent loops forever | Set recursion_limit in config |
| Vague system prompt | Agent uses tools unnecessarily | Be specific about when to use tools vs answer directly |
| Too many tools | LLM confused, calls wrong ones | Max 5-10 tools, group related tools |
| No error handling in tools | One error crashes the agent | Return error string, let agent retry |
| Not testing edge cases | Agent hallucinates for unknown queries | Test with out-of-scope questions |
| Ignoring costs | $$$$ | Log and limit tool/LLM calls per session |
