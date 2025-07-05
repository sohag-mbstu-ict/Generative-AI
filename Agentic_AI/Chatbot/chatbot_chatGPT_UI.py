# Conversational Q&A Chatbot with ChatGPT-like UI
import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pathlib import Path
import os

# Load API Key
env_path = Path("/media/mtl/Volume F/PROJECTS/projects/.venv")
load_dotenv(dotenv_path=env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("GOOGLE API KEY:", GOOGLE_API_KEY)

# Setup chat model
chat = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", GOOGLE_API_KEY=GOOGLE_API_KEY)

# Streamlit config
st.set_page_config(page_title="Chatbot", layout="centered")
st.title("AI Assistant")
st.caption("Ask me anything about farming, crops, and agriculture!")

# Memory session
if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content="You are a AI assistant")
    ]

# Show message history
for msg in st.session_state['flowmessages']:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# User input area (like ChatGPT)
if prompt := st.chat_input("Ask a question..."):
    # Show user message
    st.session_state['flowmessages'].append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    response = chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=response.content))

    # Show AI response
    with st.chat_message("assistant"):
        st.markdown(response.content)
