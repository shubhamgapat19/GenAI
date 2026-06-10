"""
Phase 5: Agents & LangGraph — Structured Agent Outputs
========================================================
Get consistent, typed outputs from agents.

Topics covered:
- Structured output from agents
- Tool result formatting
- Agent output schemas
- Planning agents (plan → execute)
- Agent with memory + tools + structured output
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================
# 1. Structured Output from Agents
# ============================================================

# Problem: Agents return free-text. You need structured data.
# Solution: Final step extracts structured output from agent's work.

class ResearchResult(BaseModel):
    """Structured research output."""
    topic: str = Field(description="The topic researched")
    summary: str = Field(description="2-3 sentence summary of findings")
    key_facts: list[str] = Field(description="List of key facts discovered")
    confidence: float = Field(description="Confidence in accuracy, 0.0 to 1.0")
    sources_used: list[str] = Field(description="Tools/sources consulted")


@tool
def research_web(query: str) -> str:
    """Search the web for information.
    
    Args:
        query: Search query.
    """
    data = {
        "langchain agents": "LangChain agents use LLMs to decide which tools to call. They follow the ReAct pattern. Support for custom tools, multi-agent systems. Used in production by companies like Elastic, Replit.",
        "langgraph features": "LangGraph: graph-based agent framework. Features: cycles, state persistence, streaming, human-in-the-loop, multi-agent. Built on top of LangChain.",
    }
    for key, val in data.items():
        if key in query.lower():
            return val
    return f"General results for '{query}'"


@tool
def research_docs(topic: str) -> str:
    """Search official documentation.
    
    Args:
        topic: Documentation topic to look up.
    """
    docs = {
        "agents": "Agents in LangChain are autonomous systems that use LLMs for reasoning. They can call tools, maintain state, and handle complex multi-step tasks.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with graph-based workflows.",
    }
    for key, val in docs.items():
        if key in topic.lower():
            return val
    return f"Documentation for '{topic}' not found."


from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Agent does research
research_agent = create_react_agent(
    llm,
    [research_web, research_docs],
    prompt="You are a thorough research agent. Use ALL available tools to gather information. Be comprehensive.",
)

# Structured extractor
structured_llm = llm.with_structured_output(ResearchResult)

def research_with_structure(question: str) -> ResearchResult:
    """Agent researches, then output is structured."""
    # Step 1: Agent gathers information
    result = research_agent.invoke({
        "messages": [HumanMessage(content=question)]
    })
    agent_response = result["messages"][-1].content
    
    # Step 2: Extract structured output
    extraction_prompt = f"""Based on this research, create a structured summary:

Research findings:
{agent_response}

Original question: {question}"""
    
    structured = structured_llm.invoke(extraction_prompt)
    return structured


print("=== Structured Agent Output ===")
result = research_with_structure("What are LangChain agents and what is LangGraph?")
print(f"Topic: {result.topic}")
print(f"Summary: {result.summary}")
print(f"Key facts: {result.key_facts}")
print(f"Confidence: {result.confidence}")
print(f"Sources: {result.sources_used}")
print()

# ============================================================
# 2. Planning Agent (Plan → Execute → Verify)
# ============================================================

class Plan(BaseModel):
    """A plan for accomplishing a task."""
    goal: str = Field(description="The end goal")
    steps: list[str] = Field(description="Ordered steps to achieve the goal")
    tools_needed: list[str] = Field(description="Which tools each step needs")


class StepResult(BaseModel):
    """Result of executing a step."""
    step: str
    result: str
    success: bool


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression.
    
    Args:
        expression: Math expression to evaluate.
    """
    try:
        return str(eval(expression, {"__builtins__": {}}, {"__import__": None}))
    except Exception as e:
        return f"Error: {e}"


@tool
def data_lookup(key: str) -> str:
    """Look up data from the company database.
    
    Args:
        key: Data key to look up (e.g., 'revenue_2024', 'employee_count').
    """
    database = {
        "revenue_2024": "$5.2M",
        "revenue_2023": "$3.8M",
        "employee_count": "45",
        "growth_rate": "36.8%",
        "top_product": "AI Analytics Platform",
    }
    return database.get(key, f"No data for '{key}'")


planning_llm = llm.with_structured_output(Plan)

def plan_and_execute(task: str):
    """Create a plan, then execute each step."""
    # Step 1: Create plan
    plan_prompt = f"""Create a step-by-step plan to accomplish this task:
Task: {task}

Available tools: calculator (math), data_lookup (company database keys: revenue_2024, revenue_2023, employee_count, growth_rate, top_product)"""
    
    plan = planning_llm.invoke(plan_prompt)
    print(f"  Plan: {plan.goal}")
    for i, step in enumerate(plan.steps, 1):
        print(f"    Step {i}: {step}")
    print()
    
    # Step 2: Execute each step
    executor = create_react_agent(llm, [calculator, data_lookup])
    
    results = []
    for step in plan.steps:
        result = executor.invoke({
            "messages": [HumanMessage(content=f"Execute this step: {step}")]
        })
        step_result = result["messages"][-1].content
        results.append(StepResult(step=step, result=step_result, success=True))
        print(f"    ✓ {step}: {step_result[:60]}...")
    
    # Step 3: Synthesize final answer
    synthesis_prompt = f"""Task: {task}

Results from each step:
{chr(10).join(f'- {r.step}: {r.result}' for r in results)}

Provide a final comprehensive answer."""
    
    final = llm.invoke(synthesis_prompt)
    return final.content


print("=== Planning Agent ===")
answer = plan_and_execute("What is the company's revenue growth from 2023 to 2024, and what's the revenue per employee?")
print(f"\n  Final: {answer}")
print()

# ============================================================
# 3. Agent with Memory + Tools + Structure
# ============================================================

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import Annotated
from langgraph.graph.message import add_messages


class FullAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_profile: dict


@tool
def save_preference(key: str, value: str) -> str:
    """Save a user preference for future reference.
    
    Args:
        key: Preference name (e.g., 'language', 'style').
        value: Preference value.
    """
    return f"Saved preference: {key} = {value}"


@tool
def get_recommendation(category: str) -> str:
    """Get a personalized recommendation.
    
    Args:
        category: Category like 'framework', 'database', 'language'.
    """
    recs = {
        "framework": "Based on your profile: try FastAPI for APIs, or Next.js for full-stack.",
        "database": "For your use case: PostgreSQL with pgvector for RAG, Redis for caching.",
        "language": "Python for AI/ML, TypeScript for full-stack web.",
    }
    return recs.get(category, f"No recommendations for '{category}'")


memory_tools = [save_preference, get_recommendation, calculator]
llm_with_memory_tools = llm.bind_tools(memory_tools)


def memory_agent_node(state: FullAgentState) -> dict:
    system = SystemMessage(content=f"""You are a personalized assistant. 
User profile: {state.get('user_profile', {})}
Help the user and remember their preferences using save_preference tool.""")
    messages = [system] + state["messages"]
    response = llm_with_memory_tools.invoke(messages)
    return {"messages": [response]}


def update_profile(state: FullAgentState) -> dict:
    """Extract preferences from tool calls to update profile."""
    profile = state.get("user_profile", {})
    for msg in state["messages"]:
        if hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                if tc["name"] == "save_preference":
                    profile[tc["args"]["key"]] = tc["args"]["value"]
    return {"user_profile": profile}


def should_continue(state: FullAgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


# Build
full_graph = StateGraph(FullAgentState)
full_graph.add_node("agent", memory_agent_node)
full_graph.add_node("tools", ToolNode(memory_tools))
full_graph.add_node("update_profile", update_profile)

full_graph.add_edge(START, "agent")
full_graph.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": "update_profile",
})
full_graph.add_edge("tools", "agent")
full_graph.add_edge("update_profile", END)

checkpointer = MemorySaver()
full_agent = full_graph.compile(checkpointer=checkpointer)

print("=== Full Agent (Memory + Tools + Profile) ===")
config = {"configurable": {"thread_id": "user_shubham"}}

result = full_agent.invoke({
    "messages": [HumanMessage(content="I prefer Python and I'm building a logistics app. Save that.")],
    "user_profile": {},
}, config=config)
print(f"Turn 1: {result['messages'][-1].content[:100]}...")
print(f"Profile: {result['user_profile']}")
print()

result = full_agent.invoke({
    "messages": [HumanMessage(content="Based on my preferences, recommend a database.")],
    "user_profile": result["user_profile"],
}, config=config)
print(f"Turn 2: {result['messages'][-1].content[:100]}...")
