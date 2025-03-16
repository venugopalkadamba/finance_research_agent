from typing import List
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from finance_ra import init_agent


finance_ra_agent = init_agent()
def get_agent_response(query: str) -> str:
    st.session_state.agent_messages.append(HumanMessage(content=query))
    agent_messages = finance_ra_agent.invoke({"messages": st.session_state.agent_messages})
    response = agent_messages["messages"][-1].content
    st.session_state.agent_messages = agent_messages["messages"]
    return response


st.set_page_config(
    page_title="KVG Finance Research Assistant",
    page_icon="📈",
    layout="centered"
)

# initialize chat session in streamlit if not already present
if "chat_history" not in st.session_state:
    init_ai_message = "Hey, how can I help you today?"
    st.session_state.chat_history = [{
        "role": "assistant",
        "content": init_ai_message
    }]
    st.session_state.agent_messages = [AIMessage(content=init_ai_message)]

# streamlit page title
st.title("📈 KVG Finance Research Assistant")

# display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# input field for user's message
user_prompt = st.chat_input()

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    with st.spinner("Thinking...", show_time=True):
        assistant_response = get_agent_response(user_prompt)
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)