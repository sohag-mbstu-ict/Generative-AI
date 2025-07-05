# # https://docs.llamaindex.ai/en/stable/examples/llm/ollama/

from llama_index.llms.ollama import Ollama
llm = Ollama(
    model="gemma:2b",
    request_timeout=120.0,
    # Manually set the context window to limit memory usage
    context_window=8000,)
resp = llm.complete("Who is Paul Graham?")
print(resp)

print("----------------------------- Call chat with a list of messages -----------------------------------------------")
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.chat(messages)
print(resp)


