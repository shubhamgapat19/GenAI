"""
Phase 5: Agents & LangGraph — Advanced LangGraph Patterns
============================================================
Complex agent architectures with LangGraph.

Topics covered:
- Human-in-the-loop (approval before action)
- Multi-agent systems
- Subgraphs
- Streaming agent events
- Error handling in agents
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================
# 1. Human-in-the-Loop (approval before executing tools)
# ============================================================

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient.
    
    Args:
        to: Email address of the recipient.
        subject: Email subject line.
        body: Email body content.
    """
    # In production, this would actually send an email
    return f"Email sent to {to} with subject '{subject}'"


@tool
def delete_file(filename: str) -> str:
    """Delete a file from the system.
    
    Args:
        filename: Name of file to delete.
    """
    return f"File '{filename}' deleted successfully."


@tool
def safe_search(query: str) -> str:
    """Search for information (safe, no side effects).
    
    Args:
        query: What to search for.
    """
    return f"Results for '{query}': Found relevant documentation."


# Classify tools by risk level
safe_tools = [safe_search]
dangerous_tools = [send_email, delete_file]
all_tools = safe_tools + dangerous_tools

llm_with_tools = llm.bind_tools(all_tools)


class HITLState(TypedDict):
    messages: Annotated[list, add_messages]
    pending_approval: bool


def agent_node(state: HITLState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def check_danger(state: HITLState) -> str:
    """Route based on whether tool call is dangerous."""
    last_msg = state["messages"][-1]
    if not last_msg.tool_calls:
        return "end"
    
    # Check if any tool call is dangerous
    dangerous_names = {t.name for t in dangerous_tools}
    for tc in last_msg.tool_calls:
        if tc["name"] in dangerous_names:
            return "needs_approval"
    
    return "safe_tools"


def approval_node(state: HITLState) -> dict:
    """Simulate human approval (in production, this would pause and wait)."""
    last_msg = state["messages"][-1]
    print("\n  ⚠️  APPROVAL REQUIRED:")
    for tc in last_msg.tool_calls:
        print(f"    Tool: {tc['name']}")
        print(f"    Args: {tc['args']}")
    
    # Simulate approval (in production: pause, notify human, wait)
    approved = True  # Change to False to test rejection
    
    if approved:
        print("    ✅ APPROVED by human")
        return {"pending_approval": False}
    else:
        print("    ❌ REJECTED by human")
        rejection = AIMessage(content="The action was rejected by the human operator. I cannot proceed with that action.")
        return {"messages": [rejection], "pending_approval": False}


safe_tool_node = ToolNode(safe_tools)
dangerous_tool_node = ToolNode(dangerous_tools)

# Build HITL graph
hitl_graph = StateGraph(HITLState)
hitl_graph.add_node("agent", agent_node)
hitl_graph.add_node("approval", approval_node)
hitl_graph.add_node("safe_tools", safe_tool_node)
hitl_graph.add_node("dangerous_tools", dangerous_tool_node)

hitl_graph.add_edge(START, "agent")
hitl_graph.add_conditional_edges("agent", check_danger, {
    "end": END,
    "safe_tools": "safe_tools",
    "needs_approval": "approval",
})
hitl_graph.add_edge("safe_tools", "agent")
hitl_graph.add_edge("approval", "dangerous_tools")
hitl_graph.add_edge("dangerous_tools", "agent")

hitl_agent = hitl_graph.compile()

print("=== Human-in-the-Loop Agent ===")

# Safe action (no approval needed)
result = hitl_agent.invoke({
    "messages": [HumanMessage(content="Search for LangGraph documentation")],
    "pending_approval": False,
})
print(f"\nSafe action result: {result['messages'][-1].content[:100]}...")
print()

# Dangerous action (approval required)
result = hitl_agent.invoke({
    "messages": [HumanMessage(content="Send an email to team@company.com about the project update")],
    "pending_approval": False,
})
print(f"\nDangerous action result: {result['messages'][-1].content[:100]}...")
print()

# ============================================================
# 2. Multi-Agent System (Supervisor + Workers)
# ============================================================

@tool
def code_executor(code: str) -> str:
    """Execute Python code and return the output.
    
    Args:
        code: Python code to execute.
    """
    # Simulated — in production use sandboxed execution
    try:
        result = eval(code, {"__builtins__": {"len": len, "sum": sum, "range": range, "list": list}})
        return f"Output: {result}"
    except:
        return f"Executed: {code[:50]}..."


@tool
def research_tool(topic: str) -> str:
    """Research a topic and return findings.
    
    Args:
        topic: Topic to research.
    """
    findings = {
        "langgraph": "LangGraph: graph-based agent framework. Nodes=steps, Edges=transitions. Supports cycles for agent loops.",
        "fastapi": "FastAPI: modern Python web framework. Async, auto-docs, Pydantic validation. Great for LLM API servers.",
        "react pattern": "ReAct: Reasoning + Acting. Agent thinks, acts, observes, repeats. Most common agent pattern.",
    }
    for key, val in findings.items():
        if key in topic.lower():
            return val
    return f"Research on '{topic}': General information gathered."


# Define specialized agents
researcher = create_react_agent(
    llm,
    [research_tool, safe_search],
    prompt="You are a research specialist. Find information and provide detailed answers.",
)

coder = create_react_agent(
    llm,
    [code_executor, calculator],
    prompt="You are a coding specialist. Write and execute code to solve problems.",
)


class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str


def supervisor(state: SupervisorState) -> dict:
    """Supervisor decides which agent to delegate to."""
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a supervisor managing a team of specialists:
- 'researcher': For finding information, answering questions about topics
- 'coder': For writing code, calculations, programming tasks
- 'done': When the task is fully completed

Based on the conversation, decide who should work next. Reply with ONLY: researcher, coder, or done"""),
        MessagesPlaceholder("messages"),
    ])
    
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    
    chain = supervisor_prompt | llm | StrOutputParser()
    decision = chain.invoke({"messages": state["messages"]}).strip().lower()
    
    return {"next_agent": decision}


def route_supervisor(state: SupervisorState) -> str:
    return state["next_agent"]


def research_node(state: SupervisorState) -> dict:
    result = researcher.invoke({"messages": state["messages"]})
    last_msg = result["messages"][-1]
    return {"messages": [AIMessage(content=f"[Researcher]: {last_msg.content}")]}


def coder_node(state: SupervisorState) -> dict:
    result = coder.invoke({"messages": state["messages"]})
    last_msg = result["messages"][-1]
    return {"messages": [AIMessage(content=f"[Coder]: {last_msg.content}")]}


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# Build supervisor graph
sup_graph = StateGraph(SupervisorState)
sup_graph.add_node("supervisor", supervisor)
sup_graph.add_node("researcher", research_node)
sup_graph.add_node("coder", coder_node)

sup_graph.add_edge(START, "supervisor")
sup_graph.add_conditional_edges("supervisor", route_supervisor, {
    "researcher": "researcher",
    "coder": "coder",
    "done": END,
})
sup_graph.add_edge("researcher", "supervisor")
sup_graph.add_edge("coder", "supervisor")

multi_agent = sup_graph.compile()

print("=== Multi-Agent System ===")
result = multi_agent.invoke({
    "messages": [HumanMessage(content="Research what LangGraph is, then write code to calculate the sum of numbers 1 to 100.")],
    "next_agent": "",
})
for msg in result["messages"]:
    if isinstance(msg, AIMessage) and msg.content.startswith("["):
        print(f"  {msg.content[:120]}...")
print()

# ============================================================
# 3. Streaming Agent Events
# ============================================================

print("=== Streaming Agent Events ===")

simple_agent = create_react_agent(llm, [calculator, get_current_time, web_search])

# Stream events to see the agent's thinking in real-time
for event in simple_agent.stream(
    {"messages": [HumanMessage(content="What's 2^20? And what time is it?")]},
    stream_mode="updates",
):
    for node_name, updates in event.items():
        if "messages" in updates:
            last = updates["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                for tc in last.tool_calls:
                    print(f"  [{node_name}] Calling: {tc['name']}({tc['args']})")
            elif hasattr(last, "content") and last.content:
                print(f"  [{node_name}] {last.content[:80]}...")
print()

# ============================================================
# 4. Error Handling in Agents
# ============================================================

@tool
def flaky_tool(input: str) -> str:
    """A tool that sometimes fails (simulating real-world APIs).
    
    Args:
        input: Input to process.
    """
    import random
    if random.random() < 0.3:
        raise Exception("Service temporarily unavailable")
    return f"Processed: {input}"


def safe_tool_node(state: AgentState):
    """Tool node with error handling."""
    from langchain_core.messages import ToolMessage
    
    last_msg = state["messages"][-1]
    tool_messages = []
    
    tools_map = {"flaky_tool": flaky_tool, "calculator": calculator}
    
    for tc in last_msg.tool_calls:
        try:
            tool_fn = tools_map.get(tc["name"])
            if tool_fn:
                result = tool_fn.invoke(tc["args"])
            else:
                result = f"Unknown tool: {tc['name']}"
        except Exception as e:
            result = f"Tool error: {str(e)}. Please try again or use an alternative approach."
        
        tool_messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tc["id"],
        ))
    
    return {"messages": tool_messages}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


print("=== Error Handling Pattern ===")
print("Wrap tool execution in try/except")
print("Return error message as ToolMessage so agent can recover")
print("Agent sees the error and can retry or use alternative")
