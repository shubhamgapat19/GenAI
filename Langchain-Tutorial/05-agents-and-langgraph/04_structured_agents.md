# Structured Agent Outputs — Deep Dive Notes

## The Problem: Unstructured Agent Responses

Agents return free-text by default. But your app often needs:
- JSON for API responses
- Typed objects for downstream processing
- Consistent format for UI rendering
- Validated data for database storage

---

## Solution: Structure the Final Step

```
Agent reasoning (free-form, messy, multi-step)
    ↓
[Structure Extraction Step]
    ↓
Clean, typed, validated output (Pydantic model)
```

---

## Approach 1: Post-Agent Extraction

```python
from pydantic import BaseModel, Field

class ResearchResult(BaseModel):
    summary: str
    key_facts: list[str]
    confidence: float

# Step 1: Agent does its thing (free-form)
result = agent.invoke({"messages": [HumanMessage("Research X")]})
agent_text = result["messages"][-1].content

# Step 2: Extract structure from the agent's output
structured_llm = llm.with_structured_output(ResearchResult)
structured = structured_llm.invoke(f"Extract structured data from: {agent_text}")
```

**Pros:** Clean separation, agent can think freely
**Cons:** Extra LLM call, added latency

---

## Approach 2: Agent with Structured Final Tool

```python
@tool
def submit_answer(summary: str, facts: list[str], confidence: float) -> str:
    """Submit the final structured answer. Call this when you have your answer ready.
    
    Args:
        summary: 2-3 sentence summary
        facts: List of key facts found
        confidence: Confidence score 0.0-1.0
    """
    return "Answer submitted."

# Add submit_answer as the "finishing" tool
agent = create_react_agent(llm, [research_tool, submit_answer],
    prompt="When you have your answer, call submit_answer with structured results.")
```

**Pros:** Single pass, no extra LLM call
**Cons:** Agent might not always call the final tool correctly

---

## Approach 3: Structured Output in Final Node

```python
def final_node(state: AgentState) -> dict:
    """After agent finishes, extract structure."""
    # Get all the agent's work
    conversation = state["messages"]
    
    # Extract structured output
    structured_llm = llm.with_structured_output(OutputSchema)
    structured = structured_llm.invoke(conversation)
    
    return {"structured_output": structured}

# Add as final node after agent loop ends
graph.add_edge("agent_done", "structure_output")
graph.add_edge("structure_output", END)
```

---

## Planning Agents (Plan → Execute)

A powerful pattern for complex tasks:

```
[Planner] → Creates step-by-step plan
    ↓
[Executor] → Executes each step with tools
    ↓
[Synthesizer] → Combines results into final answer
```

### Implementation
```python
class Plan(BaseModel):
    goal: str
    steps: list[str]
    tools_needed: list[str]

# Step 1: Plan
planner = llm.with_structured_output(Plan)
plan = planner.invoke(f"Create a plan for: {task}")

# Step 2: Execute each step
for step in plan.steps:
    result = executor_agent.invoke({"messages": [HumanMessage(step)]})
    results.append(result)

# Step 3: Synthesize
final = llm.invoke(f"Combine results: {results}")
```

### When to Use Planning
- Complex multi-step tasks
- When you want visibility into the agent's approach
- When steps can be parallelized
- When you want to validate the plan before execution

---

## Agent with Memory + Tools + Structure

The full production pattern:

```python
class FullState(TypedDict):
    messages: Annotated[list, add_messages]
    user_profile: dict          # Persistent user info
    structured_output: dict     # Final structured result

# Nodes:
# 1. agent_node: LLM with tools (reasoning)
# 2. tool_node: Execute tools
# 3. profile_update: Extract user preferences from conversation
# 4. structure_output: Format final answer

# Graph:
# START → agent ⟷ tools → profile_update → structure_output → END
```

### Profile Building Pattern
```python
@tool
def save_preference(key: str, value: str) -> str:
    """Remember a user preference."""
    return f"Saved: {key}={value}"

# After conversation, extract prefs from tool calls:
def update_profile(state):
    for msg in state["messages"]:
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "save_preference":
                    state["user_profile"][tc["args"]["key"]] = tc["args"]["value"]
```

---

## Output Schema Design Tips

### Good Schema
```python
class AnalysisResult(BaseModel):
    """Clear, focused output schema."""
    answer: str = Field(description="Direct answer to the question")
    reasoning: str = Field(description="Brief explanation of how the answer was derived")
    sources: list[str] = Field(description="Tools/data sources used")
    confidence: float = Field(ge=0, le=1, description="Confidence level")
    follow_up_questions: list[str] = Field(default=[], description="Suggested follow-ups")
```

### Schema Design Rules
1. **Be specific** — `list[str]` not `Any`
2. **Add descriptions** — LLM uses them for extraction
3. **Use constraints** — `ge=0, le=1` for confidence
4. **Default values** — for optional fields
5. **Keep flat** — avoid deeply nested structures

---

## Best Practices

1. **Separate reasoning from structure** — let agent think freely, structure at the end
2. **Use Pydantic for validation** — catch malformed outputs before they reach your app
3. **Planning for complex tasks** — plan + execute is more reliable than one-shot
4. **Profile building is valuable** — users love personalized experiences
5. **Test schema extraction** — LLMs sometimes struggle with complex schemas
6. **Keep schemas simple** — 5-10 fields max
7. **Version your schemas** — they'll evolve as your app grows

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Forcing structure too early | Agent can't reason freely | Structure only the final output |
| Complex nested schemas | LLM fails to fill correctly | Flatten to max 2 levels deep |
| No default values | Missing fields crash app | Add `default=None` or `default=[]` |
| Schema without descriptions | LLM doesn't know what to put | Add Field(description=...) always |
| Not validating | Bad data reaches DB | Use Pydantic validation |
| One giant schema | Hard to fill accurately | Split into focused sub-schemas |
