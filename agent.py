import os
from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

# 1. Clear, Robust State Definition
class AgentState(MessagesState):
    next: str        # The worker node chosen to execute next
    log: str         # Hidden flag keeping track of who just finished speaking
    iter: int        # Operational guardrail counter

# 2. Initialize Model
llm = ChatOpenAI(
    model="accounts/fireworks/models/deepseek-v4-flash", 
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.fireworks.ai/inference/v1" 
)

# 3. Define the Supervisor Node
def supervisor_node(state: AgentState):
    current_iter = state.get("iter", 0)
    last_finished = state.get("log", "")
    
    # Loop boundary protection
    if current_iter >= 12:
        return {"next": "end", "iter": current_iter}

    # RULE 1: If the user just typed a new question, force the chain to kick off with the researcher
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        return {"next": "researcher", "log": "supervisor", "iter": current_iter + 1}

    # RULE 2: If the writer has completed the recommendation prose, yield control to the human input box
    if last_finished == "writer":
        return {"next": "human", "log": "supervisor"}

    system_prompt = (
        "You are the Team Supervisor managing a product research and buying advisory crew.\n"
        "You MUST communicate EXCLUSIVELY in English. Never reply in Chinese.\n\n"
        "Your task is to review the message history and guide the pipeline step-by-step:\n"
        "1. If the researcher just provided raw specs -> choose: <next>analyst</next> to contrast options.\n"
        "2. If the analyst just detailed pros/cons -> choose: <next>writer</next> to build the buying guide.\n"
        "3. If the writer finished -> choose: <next>end</next>.\n\n"
        "Output your selection wrapped in tags, for example: <next>analyst</next>"
    )
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    content_lower = response.content.lower()
    
    next_step = "researcher"  # Fallback safety default
    if "<next>" in content_lower and "</next>" in content_lower:
        parsed = content_lower.split("<next>")[1].split("</next>")[0].strip()
        if parsed in ["researcher", "analyst", "writer", "end"]:
            next_step = parsed

    # Update visual status text cleanly
    response.content = f"Supervisor: Internal workflow routing to the **{next_step}** phase."
    
    return {
        "messages": [response], 
        "next": next_step, 
        "log": "supervisor",
        "iter": current_iter + 1
    }

# 4. Define Specialized Workers
def researcher_node(state: AgentState):
    """Gathers product facts, specifications, and market availability."""
    system_prompt = (
        "You are a researcher. Gather specifications, hardware details, models, and options matching the request.\n"
        "CRITICAL: Write your response EXCLUSIVELY in English. Do not use Chinese."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    response.content = f"### 🔍 Researcher Findings\n\n{response.content}"
    return {"messages": [response], "log": "researcher"}

def analyst_node(state: AgentState):
    """Identifies patterns, value propositions, and consumer trade-offs."""
    system_prompt = (
        "You are an analyst. Identify patterns, compare feature trade-offs, and detail pros/cons of the options found.\n"
        "CRITICAL: Write your response EXCLUSIVELY in English. Do not use Chinese."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    response.content = f"### 📊 Analyst Evaluation\n\n{response.content}"
    return {"messages": [response], "log": "analyst"}

def writer_node(state: AgentState):
    """Produces the final, consumer-friendly polished buying guide advice."""
    system_prompt = (
        "You are a writer. Take the analysis and compile a beautifully polished buying suggestion guide.\n"
        "CRITICAL: Write your response EXCLUSIVELY in English. Do not use Chinese."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    response.content = f"### ✍️ Final Purchase Recommendation\n\n{response.content}"
    return {"messages": [response], "log": "writer"}

# 5. Conditional Edge Router Logic
def route_supervisor(state: AgentState):
    target = state.get("next", "human")
    if target in ["human", "end"]:
        return "end_route"
    return target

# 6. Build and Map Graph Layout
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        "end_route": END
    }
)

workflow.add_edge("researcher", "supervisor")
workflow.add_edge("analyst", "supervisor")
workflow.add_edge("writer", "supervisor")

graph = workflow.compile()