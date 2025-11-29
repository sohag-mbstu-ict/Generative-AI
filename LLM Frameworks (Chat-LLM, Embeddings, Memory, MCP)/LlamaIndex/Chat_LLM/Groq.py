# https://docs.llamaindex.ai/en/stable/examples/llm/groq/

from llama_index.llms.groq import Groq
from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env from the correct file path
env_path = Path('/media/mtl/Volume F/PROJECTS/projects/.venv/')
load_dotenv(dotenv_path=env_path)
# Safely get the Groq key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("GROQ API KEY:", GROQ_API_KEY)

# Instantiate and invoke Groq LLM
llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
response = llm.complete("Explain the importance of low latency LLMs")
print(response)
print("-----------------------------------------------------------------------------")
# Call chat with a list of messages
from llama_index.core.llms import ChatMessage
messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.chat(messages)
print(resp)

print("-----------------------------------------------------------------------------")

# Using stream_chat endpoint
from llama_index.core.llms import ChatMessage
messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")
