# Tools & Tool Calling — Deep Dive Notes

## What Are Tools?

Tools give LLMs **the ability to take actions** beyond generating text. The LLM doesn't execute tools — it decides WHAT to call and WITH WHAT arguments. Your code executes the tool and returns the result.

```
User: "What's the weather in Mumbai?"
  ↓
LLM: "I should call get_weather(city='Mumbai')"   ← DECISION (tool_call)
  ↓
Your code: executes get_weather("Mumbai")          ← EXECUTION
  ↓
Result: "32°C, Humid"                              ← OBSERVATION
  ↓
LLM: "The weather in Mumbai is 32°C and humid."   ← FINAL ANSWER
```

---

## The Tool Calling Flow

```
1. User sends question
2. LLM receives question + list of available tools (schemas)
3. LLM decides: answer directly OR call tool(s)
4. If tool_calls: your code executes them
5. Results are sent back to LLM as ToolMessages
6. LLM generates final answer using tool results
7. Repeat if more tool calls needed
```

---

## Defining Tools

### The @tool Decorator (simplest)
```python
from langchain_core.tools import tool

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
    """
    return a * b
```

**Critical:** The docstring IS the tool description. The LLM reads it to decide when to use the tool. Write it clearly!

### What the LLM Sees (tool schema)
```json
{
  "name": "multiply",
  "description": "Multiply two numbers together.",
  "parameters": {
    "type": "object",
    "properties": {
      "a": {"type": "number", "description": "First number"},
      "b": {"type": "number", "description": "Second number"}
    },
    "required": ["a", "b"]
  }
}
```

---

## Writing Good Tool Descriptions

| Bad | Good | Why |
|-----|------|-----|
| "Does math" | "Evaluate a mathematical expression using Python syntax" | Specific about capability |
| "Search" | "Search the company knowledge base for product documentation" | Clear scope |
| "Send message" | "Send a Slack message to a channel. Use for team notifications." | Tells WHEN to use |

### Tips
- Be specific about what the tool CAN and CANNOT do
- Mention the input format expected
- Say when the tool should be used vs alternatives
- Keep under 200 words

---

## Binding Tools to LLMs

```python
tools = [multiply, add, get_weather]
llm_with_tools = llm.bind_tools(tools)

# Now the LLM knows about these tools
response = llm_with_tools.invoke("What's 5 * 3?")
# response.tool_calls = [{"name": "multiply", "args": {"a": 5, "b": 3}, "id": "call_xxx"}]
```

### tool_calls Structure
```python
response.tool_calls = [
    {
        "name": "multiply",        # Which tool
        "args": {"a": 5, "b": 3},  # Arguments
        "id": "call_abc123",       # Unique ID (for matching results)
    }
]
```

---

## Executing Tool Calls

```python
from langchain_core.messages import ToolMessage

for tool_call in response.tool_calls:
    # Find the tool function
    tool_fn = tools_dict[tool_call["name"]]
    
    # Execute it
    result = tool_fn.invoke(tool_call["args"])
    
    # Create ToolMessage (links result to the call via ID)
    tool_msg = ToolMessage(
        content=str(result),
        tool_call_id=tool_call["id"],  # Must match!
    )
```

**The `tool_call_id` is critical** — it tells the LLM which result belongs to which call.

---

## Tool Choice (controlling tool use)

| Setting | Effect |
|---------|--------|
| Default (none set) | LLM decides whether to use tools |
| `tool_choice="auto"` | Same as default |
| `tool_choice="none"` | LLM CANNOT use tools (must answer directly) |
| `tool_choice={"type": "function", "function": {"name": "X"}}` | FORCE specific tool |
| `tool_choice="required"` | LLM MUST call at least one tool |

---

## Pydantic Schema Tools (typed, validated)

```python
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    category: str = Field(description="Category filter")
    limit: int = Field(default=5, ge=1, le=20)

@tool(args_schema=SearchInput)
def search(query: str, category: str, limit: int = 5) -> str:
    """Search with category filtering."""
    ...
```

Benefits:
- Type validation on inputs
- Default values
- Constraints (ge, le, min_length, etc.)
- Better schema for the LLM

---

## Parallel Tool Calls

Modern LLMs can call multiple tools at once:
```
User: "What's 5*3 and what's the weather in Mumbai?"
LLM response.tool_calls = [
    {"name": "multiply", "args": {"a": 5, "b": 3}},
    {"name": "get_weather", "args": {"city": "Mumbai"}},
]
```

Both are executed, both results are sent back, LLM synthesizes.

---

## Best Practices

1. **Descriptive names and docstrings** — the LLM only sees the schema, not your code
2. **Return strings** — tool results become message content (always str)
3. **Handle errors gracefully** — return error messages, don't crash
4. **Limit tool count** — 5-10 tools max per agent (more = confusion)
5. **Test without the agent** — verify tools work independently first
6. **Log tool calls** — essential for debugging agent behavior
7. **Validate inputs** — don't trust LLM-generated arguments blindly
8. **Consider side effects** — some tools are safe (read), some are dangerous (delete)

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Vague tool description | LLM uses tool incorrectly | Write specific, clear descriptions |
| Missing tool_call_id | LLM can't match results | Always include `tool_call_id` in ToolMessage |
| Tool returns complex object | LLM can't read it | Always return `str` |
| Too many tools | LLM gets confused | Keep to 5-10, use categories |
| No error handling | Agent crashes | Wrap execution in try/except |
| Dangerous tools without guard | Unintended side effects | Add human-in-the-loop for risky actions |
