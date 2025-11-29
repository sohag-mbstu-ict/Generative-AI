from dotenv import load_dotenv
from pathlib import Path
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# ✅ Load .env
env_path = Path("/media/mtl/Volume F/PROJECTS/projects/.venv")
load_dotenv(dotenv_path=env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("GOOGLE API KEY:", GOOGLE_API_KEY)

# ✅ Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", GOOGLE_API_KEY=GOOGLE_API_KEY)

# ✅ Initialize Token Buffer Memory
memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=500,  # adjust as needed
    return_messages=True
)

# ✅ Prompt template
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
The following is a conversation between a human and an AI assistant. The assistant is helpful, creative, and friendly.
Conversation history:
{history}

User: {input}
Assistant:"""
)

# ✅ Setup conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)

# ✅ CLI loop
while True:
    query = input("You: ")

    if query.lower() == "exit":
        print("End the session.")
        break

    elif query.lower() == "history":
        print("\n🧠 Token-limited History:")
        for msg in memory.chat_memory.messages:
            role = "You" if msg.type == "human" else "Bot"
            print(f"{role}: {msg.content}")
        print()
        continue

    # Normal query response
    response = conversation.invoke(query)
    print("Bot:", response['response'])
