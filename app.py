import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent import graph
import os
from dotenv import load_dotenv

load_dotenv()

st.title("🔩 Hardware Agent Builder")
st.caption("Tell me what you want to build. Type 'looks good' or 'satisfied' when you love the plan to trigger datasheet downloads.")

# Store conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# User input
if user_input := st.chat_input("Explain your build or reply to questions:"):
    st.chat_message("user").write(user_input)
    
    # Append to state history
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Run through LangGraph
    inputs = {"messages": st.session_state.messages, "phase": "planning", "selected_ics": []}
    output = graph.invoke(inputs)
    
    # Capture new responses from LangGraph execution
    new_messages = output["messages"]
    
    # Filter out what we already had to find the new agent responses
    for msg in new_messages[len(st.session_state.messages):]:
        if isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMessage) and "System:" in msg.content:
            st.success(msg.content) # System updates like downloading datasheets
            
    st.session_state.messages = new_messages