"""
Phase 5: Agents & LangGraph — Tools & Tool Calling
=====================================================
Give LLMs the ability to take actions in the real world.

Topics covered:
- What are tools
- Defining custom tools (@tool decorator)
- Tool schemas and descriptions
- Binding tools to LLMs
- Invoking tools from LLM responses
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================
# 1. What Are Tools?
# ============================================================

# Tools = functions that an LLM can decide to call
# The LLM doesn't execute them — it says "I want to call X with args Y"
# YOUR CODE actually executes the tool and returns results

# Flow:
# User question → LLM → "I need to call calculator(2+2)" → You execute → Return 4 → LLM → Final answer

print("=== What Are Tools? ===")
print("LLM decides WHAT to call and WITH WHAT arguments")
print("Your code actually EXECUTES the tool")
print("Result is fed back to the LLM for final answer")
print()

# ============================================================
# 2. Defining Tools with @tool Decorator
# ============================================================

from langchain_core.tools import tool


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
    """
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    """
    return a + b


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Args:
        city: Name of the city to get weather for.
    """
    # Simulated — in production, call a real API
    weather_data = {
        "mumbai": "32°C, Humid, Partly Cloudy",
        "pune": "28°C, Clear Sky",
        "delhi": "38°C, Hazy",
        "bangalore": "24°C, Light Rain",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def search_database(query: str, limit: int = 5) -> str:
    """Search the company knowledge base for information.
    
    Args:
        query: The search query string.
        limit: Maximum number of results to return.
    """
    # Simulated database search
    return f"Found {limit} results for '{query}': [Result 1, Result 2, ...]"


print("=== Tool Definitions ===")
print(f"Tool: {multiply.name}")
print(f"Description: {multiply.description}")
print(f"Schema: {multiply.args_schema.model_json_schema()}")
print()
print(f"Tool: {get_weather.name}")
print(f"Description: {get_weather.description}")
print()

# ============================================================
# 3. Binding Tools to an LLM
# ============================================================

# Tell the LLM what tools are available
tools = [multiply, add, get_weather, search_database]
llm_with_tools = llm.bind_tools(tools)

print("=== Binding Tools ===")
print(f"LLM now has access to {len(tools)} tools")
print(f"Tools: {[t.name for t in tools]}")
print()

# ============================================================
# 4. LLM Deciding to Use Tools (tool_calls)
# ============================================================

# The LLM doesn't execute tools — it returns "tool_calls" in its response
response = llm_with_tools.invoke("What's 15 multiplied by 7?")

print("=== LLM Tool Call Decision ===")
print(f"Content: '{response.content}'")  # Usually empty when making tool calls
print(f"Tool calls: {response.tool_calls}")
print()

# Tool calls structure:
# [{"name": "multiply", "args": {"a": 15, "b": 7}, "id": "call_xxx"}]

# ============================================================
# 5. Executing Tools and Returning Results
# ============================================================

# Manual tool execution loop
def execute_tool_calls(response, tools_dict):
    """Execute tool calls from LLM response."""
    tool_messages = []
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        # Find and execute the tool
        tool_fn = tools_dict[tool_name]
        result = tool_fn.invoke(tool_args)
        
        # Create ToolMessage with result
        tool_messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_id,
        ))
    
    return tool_messages

# Build tools lookup
tools_dict = {t.name: t for t in tools}

# Execute
tool_results = execute_tool_calls(response, tools_dict)
print("=== Tool Execution ===")
for msg in tool_results:
    print(f"Result: {msg.content}")
print()

# ============================================================
# 6. Complete Tool Calling Loop
# ============================================================

def agent_loop(user_input: str) -> str:
    """Complete loop: user → LLM → tool → LLM → answer."""
    messages = [HumanMessage(content=user_input)]
    
    # First LLM call — may decide to use tools
    response = llm_with_tools.invoke(messages)
    messages.append(response)
    
    # If tool calls, execute them
    while response.tool_calls:
        tool_results = execute_tool_calls(response, tools_dict)
        messages.extend(tool_results)
        
        # Second LLM call — with tool results
        response = llm_with_tools.invoke(messages)
        messages.append(response)
    
    return response.content

print("=== Complete Agent Loop ===")
answer = agent_loop("What's 15 * 7, and then add 23 to the result?")
print(f"Q: What's 15 * 7, and then add 23?")
print(f"A: {answer}")
print()

answer = agent_loop("What's the weather in Mumbai?")
print(f"Q: What's the weather in Mumbai?")
print(f"A: {answer}")
print()

# No tool needed — LLM answers directly
answer = agent_loop("What is the capital of India?")
print(f"Q: What is the capital of India?")
print(f"A: {answer}")
print()

# ============================================================
# 7. Tools with Pydantic Schemas (typed arguments)
# ============================================================

from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    """Schema for search tool input."""
    query: str = Field(description="The search query")
    category: str = Field(description="Category: 'docs', 'code', or 'general'")
    max_results: int = Field(default=5, description="Max results to return")


@tool(args_schema=SearchInput)
def advanced_search(query: str, category: str, max_results: int = 5) -> str:
    """Search across different knowledge categories with filtering."""
    return f"Searching '{category}' for '{query}' (max {max_results} results)..."


print("=== Pydantic Schema Tool ===")
print(f"Tool: {advanced_search.name}")
print(f"Schema: {advanced_search.args_schema.model_json_schema()}")
print()

# ============================================================
# 8. Tool Choice — Force or Prevent Tool Use
# ============================================================

# Force the LLM to use a specific tool
forced = llm_with_tools.invoke(
    "Tell me a joke",
    tool_choice={"type": "function", "function": {"name": "get_weather"}},
)
print("=== Forced Tool Choice ===")
print(f"Forced to call: {forced.tool_calls[0]['name'] if forced.tool_calls else 'none'}")
print()

# Prevent tool use entirely
no_tools = llm_with_tools.invoke(
    "What's 5 * 3?",
    tool_choice="none",  # LLM must answer without tools
)
print("=== No Tools Allowed ===")
print(f"Response (no tools): {no_tools.content}")
