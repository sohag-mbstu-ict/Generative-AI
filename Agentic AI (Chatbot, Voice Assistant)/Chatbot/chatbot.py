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
chat = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001",GOOGLE_API_KEY=GOOGLE_API_KEY)

# Streamlit UI setup
st.set_page_config(page_title="Conversational Q&A Chatbot")
st.header("Hey, Let's Chat!")

# Memory init
if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content="You are a AI assistant.")
    ]

# Response function
def get_chatmodel_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    answer = chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))
    return answer.content

# Input + button
user_input = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

# Output
if submit and user_input:
    response = get_chatmodel_response(user_input)
    st.subheader("The Response is:")
    st.write(response)

# Optional: show past conversation
with st.expander("Conversation History"):
    for msg in st.session_state['flowmessages']:
        role = "👤 You" if isinstance(msg, HumanMessage) else ("🤖 AI" if isinstance(msg, AIMessage) else "📘 System")
        st.markdown(f"**{role}:** {msg.content}")
