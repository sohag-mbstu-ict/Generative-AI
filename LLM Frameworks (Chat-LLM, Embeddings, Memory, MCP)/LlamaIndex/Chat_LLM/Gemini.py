# https://docs.llamaindex.ai/en/stable/examples/llm/gemini/
import os
GOOGLE_API_KEY = "AIzaSyBExPddjZrh9VwycHjFtwEGgOaVnjJu3Yo"  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
from llama_index.llms.gemini import Gemini

llm = Gemini(
    model="models/gemini-1.5-flash",
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)

from llama_index.llms.gemini import Gemini
resp = llm.complete("Nuclear weapon")
# print(resp)
# ----------------------------------------------------------------------------------
from llama_index.core.llms import ChatMessage
messages = [
    ChatMessage(role="user", content="Hello friend!"),
    ChatMessage(role="assistant", content="Yarr what is shakin' matey?"),
    ChatMessage(
        role="user", content="Help me decide what to have for dinner."
    ),
]
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")

