# Stateful Chains & LangGraph Intro — Deep Dive Notes

## Why State Machines for LLM Apps?

Regular chains are **linear** — input → process → output. But real apps need:
- **Cycles:** Agent loops (think → act → observe → think again)
- **Branching:** Route to different handlers based on classification
- **Persistent state:** Remember across multiple interactions
- **Checkpointing:** Save/resume long workflows

**LangGraph** solves this with a graph-based approach: nodes (functions) + edges (transitions).

---

## LangGraph Core Concepts

```
┌─────────────────────────────────────┐
│           StateGraph                 │
│                                     │
│  START → [Node A] → [Node B] → END │
│                 ↘                    │
│              [Node C] ──────────→   │
└─────────────────────────────────────┘
```

| Concept | What It Is | Analogy |
|---------|-----------|---------|
| **State** | TypedDict shared across nodes | Global variables |
| **Node** | Function that reads/modifies state | Step in a workflow |
| **Edge** | Connection between nodes | Arrow between steps |
| **Conditional Edge** | Edge chosen by a function | if/else branching |
| **Checkpointer** | Saves state after each node | Auto-save in a game |

---

## Defining State

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class MyState(TypedDict):
    messages: Annotated[list, add_messages]  # Auto-append behavior
    user_mood: str                            # Simple overwrite
    step_count: int                           # Counter
```

### Annotated[list, add_messages]
This is special — instead of overwriting, it **appends** new messages to the existing list. Without it, returning `{"messages": [new_msg]}` would replace all messages.

### State Update Rules
- Returning a key **overwrites** its value (default)
- `Annotated[list, add_messages]` **appends** instead of overwriting
- Keys not returned stay unchanged

---

## Nodes

Nodes are **functions** that:
1. Receive the current state
2. Do some work (LLM call, computation, API call)
3. Return a dict with state updates

```python
def my_node(state: MyState) -> dict:
    # Read state
    messages = state["messages"]
    mood = state["user_mood"]
    
    # Do work
    response = llm.invoke(messages)
    
    # Return updates (only what changed)
    return {
        "messages": [response],  # Appended due to add_messages
        "step_count": state["step_count"] + 1,  # Overwritten
    }
```

---

## Edges

### Static Edges (always go to next node)
```python
graph.add_edge("node_a", "node_b")  # A always → B
graph.add_edge(START, "first_node")  # Entry point
graph.add_edge("last_node", END)     # Exit point
```

### Conditional Edges (routing)
```python
def router(state: MyState) -> str:
    """Return the NAME of the next node."""
    if state["user_mood"] == "frustrated":
        return "empathy_node"
    else:
        return "normal_node"

graph.add_conditional_edges(
    "classify_node",   # Source node
    router,            # Function that decides
    # Optional: explicit mapping of return values to node names
)
```

---

## Complete Example: Build → Compile → Use

```python
from langgraph.graph import StateGraph, START, END

# 1. Define State
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    sentiment: str

# 2. Define Nodes
def analyze(state): ...
def respond(state): ...

# 3. Build Graph
graph = StateGraph(ChatState)
graph.add_node("analyze", analyze)
graph.add_node("respond", respond)
graph.add_edge(START, "analyze")
graph.add_edge("analyze", "respond")
graph.add_edge("respond", END)

# 4. Compile
app = graph.compile()

# 5. Use
result = app.invoke({"messages": [HumanMessage("Hello!")], "sentiment": ""})
```

---

## Checkpointing (Memory Across Invocations)

Without checkpointer: each `.invoke()` starts fresh.
With checkpointer: state persists between calls (conversation memory!).

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()  # In-memory (for dev)
app = graph.compile(checkpointer=checkpointer)

# Same thread_id = same persistent state
config = {"configurable": {"thread_id": "user_123"}}

# Turn 1
app.invoke({"messages": [HumanMessage("I'm Shubham")]}, config=config)

# Turn 2 — REMEMBERS Turn 1!
app.invoke({"messages": [HumanMessage("What's my name?")]}, config=config)
```

### Checkpointer Options
| Checkpointer | Persistence | Use Case |
|-------------|-------------|----------|
| `MemorySaver` | In-memory | Development/testing |
| `SqliteSaver` | SQLite file | Single-server production |
| `PostgresSaver` | PostgreSQL | Multi-server production |

---

## Branching (Conditional Routing)

```python
def route_by_type(state) -> str:
    if state["task_type"] == "code":
        return "code_node"
    elif state["task_type"] == "debug":
        return "debug_node"
    return "general_node"

graph.add_conditional_edges("classifier", route_by_type)
```

This creates a graph like:
```
                ┌→ [code_node] → END
[classifier] ──┤→ [debug_node] → END
                └→ [general_node] → END
```

---

## Cycles (Agent Loops)

The killer feature of LangGraph — loops!

```python
def should_continue(state) -> str:
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"      # Loop back
    return "end"            # Exit

graph.add_conditional_edges("agent", should_continue, {
    "tools": "tool_node",
    "end": END,
})
graph.add_edge("tool_node", "agent")  # Loop back!
```

```
START → [agent] → tool_calls? → [tools] → [agent] → no tools? → END
              ↑_______________________________________↓ (loop)
```

---

## LangGraph vs LCEL Chains

| Feature | LCEL Chains | LangGraph |
|---------|-------------|-----------|
| Flow | Linear (A → B → C) | Graph (branching, loops) |
| State | Passed through pipe | Shared TypedDict |
| Memory | RunnableWithMessageHistory | Built-in checkpointing |
| Cycles | Not possible | First-class support |
| Complexity | Simple chains | Complex agents, workflows |
| When to use | Most tasks | Agents, multi-step, stateful |

---

## When to Use LangGraph

**Use LangGraph when you need:**
- Agent loops (tool calling in a cycle)
- Complex branching logic
- State that goes beyond just messages
- Checkpointing / save-resume
- Human-in-the-loop workflows
- Multi-agent systems

**Stick with LCEL chains when:**
- Simple linear pipeline
- No branching or loops needed
- Only need message history (use RunnableWithMessageHistory)
- Prototyping quickly

---

## Best Practices

1. **Keep nodes small and focused** — one responsibility per node
2. **State is your single source of truth** — don't use globals
3. **Use checkpointing in production** — PostgresSaver for multi-server
4. **Test each node independently** — they're just functions
5. **Visualize your graph** — `app.get_graph().draw_mermaid()` to see the flow
6. **Handle errors in nodes** — a failing node breaks the whole graph
7. **Use thread_id consistently** — it's how LangGraph identifies conversations
8. **Start simple** — linear graph first, add complexity only when needed
