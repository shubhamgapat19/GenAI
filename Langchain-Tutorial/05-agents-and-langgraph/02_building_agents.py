"""
Phase 5: Agents & LangGraph — Building Agents with LangGraph
==============================================================
Create autonomous agents that reason and act in a loop.

Topics covered:
- ReAct pattern (Reason + Act)
- LangGraph agent architecture
- Tool nodes
- Agent loop with cycles
- Prebuilt agents (create_react_agent)
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================
# 1. Define Agent Tools
# ============================================================

import math
from datetime import datetime


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use Python math syntax.
    
    Args:
        expression: Math expression like '2**10' or 'math.sqrt(144)'
    """
    try:
        # Safe eval with only math operations
        allowed = {"math": math, "abs": abs, "round": round}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def web_search(query: str) -> str:
    """Search the web for current information.
    
    Args:
        query: Search query string.
    """
    # Simulated web search results
    results = {
        "langchain latest version": "LangChain v0.3.15 (released Jan 2025). Major changes: LCEL as primary API, deprecated legacy chains.",
        "python 3.13": "Python 3.13 released Oct 2024. New features: improved error messages, experimental JIT compiler.",
        "openai gpt-4o": "GPT-4o is OpenAI's multimodal model. 128K context, supports text/image/audio.",
    }
    for key, value in results.items():
        if key in query.lower():
            return value
    return f"Search results for '{query}': No specific results found. Try a more specific query."


@tool
def file_reader(filename: str) -> str:
    """Read the contents of a text file.
    
    Args:
        filename: Name of the file to read.
    """
    # Simulated file reading
    files = {
        "config.json": '{"model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 1000}',
        "notes.txt": "Meeting notes: Discussed RAG pipeline optimization. Action items: 1) Improve chunking 2) Add re-ranking",
    }
    return files.get(filename, f"File '{filename}' not found.")


tools = [calculator, get_current_time, web_search, file_reader]

# ============================================================
# 2. The ReAct Pattern (Reason + Act)
# ============================================================

# ReAct: Think → Act → Observe → Think → Act → ... → Answer
#
# Think:   "I need to calculate the compound interest..."
# Act:     calculator("1000 * (1 + 0.08)**5")
# Observe: "1469.33"
# Think:   "Now I have the answer..."
# Answer:  "The compound interest gives you $1,469.33"

print("=== ReAct Pattern ===")
print("Think → Act → Observe → Think → ... → Answer")
print("The LLM reasons about what to do, calls tools, observes results, repeats.")
print()

# ============================================================
# 3. Building an Agent with LangGraph
# ============================================================

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


# State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# The agent node — calls LLM with tools bound
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState) -> dict:
    """The agent: calls the LLM with tools."""
    system = SystemMessage(content="You are a helpful assistant. Use tools when needed. Think step by step.")
    messages = [system] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# The tool node — executes whatever tools the agent requested
tool_node = ToolNode(tools)


# Router — decides if we should call tools or finish
def should_continue(state: AgentState) -> str:
    """Check if the agent wants to use tools or is done."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"  # Agent wants to call tools → go to tool_node
    return "end"        # No tool calls → agent is done


# Build the graph
agent_graph = StateGraph(AgentState)

# Add nodes
agent_graph.add_node("agent", agent_node)
agent_graph.add_node("tools", tool_node)

# Add edges
agent_graph.add_edge(START, "agent")
agent_graph.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": END,
})
agent_graph.add_edge("tools", "agent")  # After tools → back to agent (LOOP!)

# Compile
agent = agent_graph.compile()

print("=== LangGraph Agent ===")
print("Graph: START → agent ⟷ tools → END")
print("The agent loops: think → call tool → observe → think again")
print()

# ============================================================
# 4. Running the Agent
# ============================================================

print("=== Agent Execution ===")

# Simple tool use
result = agent.invoke({
    "messages": [HumanMessage(content="What's the square root of 144 plus 2 to the power of 8?")]
})
print(f"Q: sqrt(144) + 2^8")
print(f"A: {result['messages'][-1].content}")
print()

# Multi-tool use
result = agent.invoke({
    "messages": [HumanMessage(content="What time is it now? And what's the latest version of LangChain?")]
})
print(f"Q: Current time + LangChain version")
print(f"A: {result['messages'][-1].content}")
print()

# No tools needed
result = agent.invoke({
    "messages": [HumanMessage(content="What is the capital of France?")]
})
print(f"Q: Capital of France (no tools needed)")
print(f"A: {result['messages'][-1].content}")
print()

# ============================================================
# 5. Tracing the Agent's Reasoning
# ============================================================

print("=== Agent Trace (step by step) ===")
result = agent.invoke({
    "messages": [HumanMessage(content="Read the config.json file and tell me what model is configured.")]
})

# Print all messages to see the agent's reasoning
for msg in result["messages"]:
    msg_type = msg.__class__.__name__
    if msg_type == "HumanMessage":
        print(f"  [Human] {msg.content}")
    elif msg_type == "AIMessage":
        if msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  [Agent thinks] → calling {tc['name']}({tc['args']})")
        else:
            print(f"  [Agent answer] {msg.content}")
    elif msg_type == "ToolMessage":
        print(f"  [Tool result] {msg.content}")
print()

# ============================================================
# 6. Prebuilt Agent (create_react_agent)
# ============================================================

from langgraph.prebuilt import create_react_agent

# One-liner to create a fully functional ReAct agent
simple_agent = create_react_agent(llm, tools)

print("=== Prebuilt ReAct Agent ===")
result = simple_agent.invoke({
    "messages": [HumanMessage(content="Calculate 2^10 and then search for info about Python 3.13")]
})
print(f"A: {result['messages'][-1].content}")
print()

# ============================================================
# 7. Agent with System Prompt
# ============================================================

agent_with_prompt = create_react_agent(
    llm,
    tools,
    prompt="You are a technical research assistant. Always verify information using available tools before answering. Be precise and cite your sources (which tool gave you the info).",
)

print("=== Agent with Custom Prompt ===")
result = agent_with_prompt.invoke({
    "messages": [HumanMessage(content="What are the features of GPT-4o?")]
})
print(f"A: {result['messages'][-1].content}")
