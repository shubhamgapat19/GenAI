"""
Phase 4: Memory & State — Stateful Chains & LangGraph Intro
=============================================================
Building chains that maintain and modify state over time.

Topics covered:
- Stateful chain patterns
- LangGraph basics for state management
- Checkpointing (save/resume)
- Multi-step workflows with memory
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================
# 1. Stateful Chain — Manual State Management
# ============================================================

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationState:
    """Explicit state for a conversation."""
    messages: list = field(default_factory=list)
    user_name: str = ""
    user_preferences: dict = field(default_factory=dict)
    topic: str = ""
    turn_count: int = 0


def stateful_chat(state: ConversationState, user_input: str) -> tuple[str, ConversationState]:
    """Process input, update state, return response."""
    state.turn_count += 1
    state.messages.append(HumanMessage(content=user_input))
    
    # Build prompt with state context
    system_msg = f"""You are a helpful assistant.
Known about user: Name={state.user_name or 'unknown'}, Topic={state.topic or 'general'}
Preferences: {state.user_preferences or 'none yet'}
Turn: {state.turn_count}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "history": state.messages[:-1],  # All except current
        "input": user_input,
    })
    
    state.messages.append(AIMessage(content=response))
    return response, state


print("=== Stateful Chain ===")
state = ConversationState(user_name="Shubham", topic="LangChain")

response, state = stateful_chat(state, "How do I add memory to my chains?")
print(f"Turn {state.turn_count}: {response[:100]}...")
print(f"State: name={state.user_name}, topic={state.topic}, messages={len(state.messages)}")
print()

# ============================================================
# 2. LangGraph — State Machines for LLM Apps
# ============================================================

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


# Define the state schema
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]  # Auto-appends messages
    user_mood: str
    response_style: str


# Define node functions (each step in the graph)
def analyze_mood(state: ChatState) -> dict:
    """Analyze user's mood from their last message."""
    last_message = state["messages"][-1].content
    
    mood_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "Classify the user's mood in ONE word: happy, frustrated, curious, neutral"),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )
    
    mood = mood_chain.invoke({"text": last_message})
    return {"user_mood": mood.strip().lower()}


def choose_style(state: ChatState) -> dict:
    """Choose response style based on mood."""
    mood = state.get("user_mood", "neutral")
    
    style_map = {
        "happy": "enthusiastic and encouraging",
        "frustrated": "patient and solution-focused",
        "curious": "detailed and educational",
        "neutral": "friendly and helpful",
    }
    
    style = style_map.get(mood, "friendly and helpful")
    return {"response_style": style}


def generate_response(state: ChatState) -> dict:
    """Generate response with appropriate style."""
    style = state.get("response_style", "helpful")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"Respond in a {style} manner. Be concise (2-3 sentences)."),
        MessagesPlaceholder("messages"),
    ])
    
    chain = prompt | llm
    response = chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}


# Build the graph
graph_builder = StateGraph(ChatState)

# Add nodes
graph_builder.add_node("analyze_mood", analyze_mood)
graph_builder.add_node("choose_style", choose_style)
graph_builder.add_node("generate_response", generate_response)

# Add edges (flow)
graph_builder.add_edge(START, "analyze_mood")
graph_builder.add_edge("analyze_mood", "choose_style")
graph_builder.add_edge("choose_style", "generate_response")
graph_builder.add_edge("generate_response", END)

# Compile
chat_graph = graph_builder.compile()

print("=== LangGraph Basic Chat ===")

# Test with different moods
result = chat_graph.invoke({
    "messages": [HumanMessage(content="I can't figure out why my code isn't working! So frustrated.")],
    "user_mood": "",
    "response_style": "",
})
print(f"Input: 'I can't figure out why my code isn't working! So frustrated.'")
print(f"Detected mood: {result['user_mood']}")
print(f"Style: {result['response_style']}")
print(f"Response: {result['messages'][-1].content}")
print()

result = chat_graph.invoke({
    "messages": [HumanMessage(content="Wow, I just learned about embeddings! This is so cool!")],
    "user_mood": "",
    "response_style": "",
})
print(f"Input: 'Wow, I just learned about embeddings! This is so cool!'")
print(f"Detected mood: {result['user_mood']}")
print(f"Style: {result['response_style']}")
print(f"Response: {result['messages'][-1].content}")
print()

# ============================================================
# 3. LangGraph with Checkpointing (save/resume state)
# ============================================================

from langgraph.checkpoint.memory import MemorySaver

# Add checkpointing — saves state after each node
memory_checkpointer = MemorySaver()

# Simple conversational graph with persistence
class ConvoState(TypedDict):
    messages: Annotated[list, add_messages]


def respond(state: ConvoState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be concise."),
        MessagesPlaceholder("messages"),
    ])
    chain = prompt | llm
    response = chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}


convo_graph = StateGraph(ConvoState)
convo_graph.add_node("respond", respond)
convo_graph.add_edge(START, "respond")
convo_graph.add_edge("respond", END)

# Compile WITH checkpointer
convo_app = convo_graph.compile(checkpointer=memory_checkpointer)

print("=== LangGraph with Checkpointing ===")

# Thread 1 — each thread has its own state
thread_config = {"configurable": {"thread_id": "thread_1"}}

result = convo_app.invoke(
    {"messages": [HumanMessage(content="My name is Shubham. I'm learning LangGraph.")]},
    config=thread_config,
)
print(f"Turn 1: {result['messages'][-1].content[:100]}...")

# Continue the SAME thread — it remembers!
result = convo_app.invoke(
    {"messages": [HumanMessage(content="What's my name and what am I learning?")]},
    config=thread_config,
)
print(f"Turn 2: {result['messages'][-1].content[:100]}...")
print()

# Thread 2 — completely separate state
thread2_config = {"configurable": {"thread_id": "thread_2"}}
result = convo_app.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config=thread2_config,
)
print(f"Thread 2 (no prior context): {result['messages'][-1].content[:100]}...")
print()

# ============================================================
# 4. Multi-step Workflow with Branching
# ============================================================

class TaskState(TypedDict):
    messages: Annotated[list, add_messages]
    task_type: str
    complexity: str


def classify_task(state: TaskState) -> dict:
    """Classify the user's request."""
    classifier = (
        ChatPromptTemplate.from_messages([
            ("system", "Classify this task as: code_help, explanation, or debugging. Reply with ONLY the category."),
            ("human", "{input}"),
        ])
        | llm
        | StrOutputParser()
    )
    task_type = classifier.invoke({"input": state["messages"][-1].content})
    return {"task_type": task_type.strip().lower()}


def route_task(state: TaskState) -> str:
    """Route based on task type."""
    task = state.get("task_type", "explanation")
    if "code" in task:
        return "code_response"
    elif "debug" in task:
        return "debug_response"
    else:
        return "explain_response"


def code_response(state: TaskState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a code expert. Provide clean, runnable code with brief comments."),
        MessagesPlaceholder("messages"),
    ])
    response = (prompt | llm).invoke({"messages": state["messages"]})
    return {"messages": [response]}


def debug_response(state: TaskState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a debugging expert. Identify the issue and provide a fix step by step."),
        MessagesPlaceholder("messages"),
    ])
    response = (prompt | llm).invoke({"messages": state["messages"]})
    return {"messages": [response]}


def explain_response(state: TaskState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a patient teacher. Explain clearly with examples."),
        MessagesPlaceholder("messages"),
    ])
    response = (prompt | llm).invoke({"messages": state["messages"]})
    return {"messages": [response]}


# Build branching graph
task_graph = StateGraph(TaskState)
task_graph.add_node("classify", classify_task)
task_graph.add_node("code_response", code_response)
task_graph.add_node("debug_response", debug_response)
task_graph.add_node("explain_response", explain_response)

task_graph.add_edge(START, "classify")
task_graph.add_conditional_edges("classify", route_task)
task_graph.add_edge("code_response", END)
task_graph.add_edge("debug_response", END)
task_graph.add_edge("explain_response", END)

task_app = task_graph.compile()

print("=== Multi-step Branching Workflow ===")

result = task_app.invoke({
    "messages": [HumanMessage(content="Write a Python function to reverse a linked list")],
    "task_type": "",
    "complexity": "",
})
print(f"Task: 'Write a Python function to reverse a linked list'")
print(f"Classified as: {result['task_type']}")
print(f"Response: {result['messages'][-1].content[:150]}...")
print()

result = task_app.invoke({
    "messages": [HumanMessage(content="What is the difference between async and threading in Python?")],
    "task_type": "",
    "complexity": "",
})
print(f"Task: 'What is the difference between async and threading?'")
print(f"Classified as: {result['task_type']}")
print(f"Response: {result['messages'][-1].content[:150]}...")
