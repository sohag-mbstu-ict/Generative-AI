# # https://python.langchain.com/api_reference/google_genai/chat_models/langchain_google_genai.chat_models.ChatGoogleGenerativeAI.html#langchain_google_genai.chat_models.ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pathlib import Path
import os
from langchain.memory import ConversationStringBufferMemory
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Load environment variables
env_path = Path("/media/mtl/Volume F/PROJECTS/projects/.venv")
load_dotenv(dotenv_path=env_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("GOOGLE API KEY:", GOOGLE_API_KEY)

# ✅ Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", GOOGLE_API_KEY=GOOGLE_API_KEY)

# ✅ Initialize ConversationStringBufferMemory
memory = ConversationStringBufferMemory()

# ✅ Create conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True  # shows prompt being sent
)

# ✅ Chat loop
while True:
    query = input("You: ")
    
    if query.lower() == "exit":
        print("End the session")
        break

    elif query.lower() == "history":
        print("\n🔁 Conversation History (String Buffer):")
        print(memory.buffer)
        print()
        continue

    # Normal query processing
    response = conversation.invoke(query)
    print("Bot:", response['response'])
