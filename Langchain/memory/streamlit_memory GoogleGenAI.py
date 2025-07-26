# # https://python.langchain.com/api_reference/google_genai/chat_models/langchain_google_genai.chat_models.ChatGoogleGenerativeAI.html#langchain_google_genai.chat_models.ChatGoogleGenerativeAI

import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Load .env file
env_path = Path("/media/mtl/Volume F/PROJECTS/projects/.venv")
load_dotenv(dotenv_path=env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", GOOGLE_API_KEY=GOOGLE_API_KEY)

# ✅ Session state memory for persistence across reruns
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "conversation" not in st.session_state:
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=False
    )

# ✅ App UI
st.title("💬 Gemini Chatbot with Memory")
query = st.text_input("You:", key="user_input")

if query:
    response = st.session_state.conversation.invoke(query)
    st.markdown(f"**Bot:** {response['response']}")

# ✅ History Viewer
if st.checkbox("Show Conversation History"):
    st.markdown("### 🔁 Conversation History")
    for msg in st.session_state.memory.chat_memory.messages:
        role = "🧑 You" if msg.type == "human" else "🤖 Bot"
        st.markdown(f"**{role}:** {msg.content}")

# ✅ Clear Chat Option
if st.button("🗑️ Clear Chat"):
    st.session_state.memory.clear()
    st.experimental_rerun()
