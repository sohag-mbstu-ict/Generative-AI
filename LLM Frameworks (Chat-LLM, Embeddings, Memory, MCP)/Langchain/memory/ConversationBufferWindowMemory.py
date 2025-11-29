from dotenv import load_dotenv
from pathlib import Path
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Load environment variables
env_path = Path("/media/mtl/Volume F/PROJECTS/projects/.venv")
load_dotenv(dotenv_path=env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("GOOGLE API KEY:", GOOGLE_API_KEY)

# ✅ Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", GOOGLE_API_KEY=GOOGLE_API_KEY)

# ✅ Initialize memory (only keep last 3 interactions)
memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# ✅ Setup conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True  # optional: shows full prompt
)

# ✅ CLI chat loop
while True:
    query = input("You: ")
    
    if query.lower() == "exit":
        print("End the session")
        break

    elif query.lower() == "history":
        print("\n🔁 Conversation History (Window Memory, last 3 interactions):")
        for msg in memory.chat_memory.messages:
            role = "You" if msg.type == "human" else "Bot"
            print(f"{role}: {msg.content}")
        print()
        continue

    # Process query
    response = conversation.invoke(query)
    print("Bot:", response['response'])

