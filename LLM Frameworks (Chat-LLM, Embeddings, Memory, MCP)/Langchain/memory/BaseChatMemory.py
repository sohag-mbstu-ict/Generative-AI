from dotenv import load_dotenv
from pathlib import Path
import os
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationStringBufferMemory,
)
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Load environment
env_path = Path("/media/mtl/Volume F/PROJECTS/projects/.venv")
load_dotenv(dotenv_path=env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("GOOGLE API KEY:", GOOGLE_API_KEY)

# ✅ Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", GOOGLE_API_KEY=GOOGLE_API_KEY)

# ✅ Choose memory type here (BaseChatMemory compatible)
def get_memory(memory_type: str = "buffer") -> BaseChatMemory:
    if memory_type == "buffer":
        return ConversationBufferMemory(return_messages=True)
    elif memory_type == "window":
        return ConversationBufferWindowMemory(k=3, return_messages=True)
    elif memory_type == "string":
        return ConversationStringBufferMemory()
    else:
        raise ValueError(f"Unsupported memory type: {memory_type}")

# ✅ Set memory type you want
memory = get_memory("buffer")  # change to "window" or "string" as needed

# ✅ Setup chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# ✅ Chat loop
while True:
    query = input("You: ")

    if query.lower() == "exit":
        print("End the session")
        break

    elif query.lower() == "history":
        # print("\n🔁 Conversation History:")
        if hasattr(memory, "chat_memory"):
            for msg in memory.chat_memory.messages:
                role = "You" if msg.type == "human" else "Bot"
                print(f"{role}: {msg.content}")
        elif hasattr(memory, "buffer"):
            print(memory.buffer)
        print()
        continue

    response = conversation.invoke(query)
    print("Bot:", response['response'])


