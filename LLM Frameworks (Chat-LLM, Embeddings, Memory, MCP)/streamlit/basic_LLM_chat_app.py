# # https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps

# Conversational Q&A Chatbot using Mistral
import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path("/media/mtl/Volume F/PROJECTS/projects/.venv")
load_dotenv(dotenv_path=env_path)
# Now access the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("GOOGLE API KEY:", GOOGLE_API_KEY)

from langchain_google_genai import ChatGoogleGenerativeAI
client = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001",GOOGLE_API_KEY=GOOGLE_API_KEY)

st.title("ChatGPT-like clone")

# Set a default model
if "google_model" not in st.session_state:
    st.session_state["google_model"] = "gemini-2.0-flash-001"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)


# Display assistant response in chat message container
with st.chat_message("assistant"):
    stream = client(
        # model=st.session_state["google_model"],
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ],
        # stream=True,
    )
    response = st.write_stream(stream)
st.session_state.messages.append({"role": "assistant", "content": response})
