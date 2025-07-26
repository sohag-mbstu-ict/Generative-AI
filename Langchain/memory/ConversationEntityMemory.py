from dotenv import load_dotenv
from pathlib import Path
import os
from langchain.memory import ConversationEntityMemory
from langchain.memory.entity import InMemoryEntityStore
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# ✅ Load environment
env_path = Path("/media/mtl/Volume F/PROJECTS/projects/.venv")
load_dotenv(dotenv_path=env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("GOOGLE API KEY:", GOOGLE_API_KEY)

# ✅ Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", GOOGLE_API_KEY=GOOGLE_API_KEY)

# ✅ Initialize Entity Store and Entity Memory
entity_store = InMemoryEntityStore()
memory = ConversationEntityMemory(llm=llm, entity_store=entity_store)

# ✅ Custom prompt with support for 'input', 'history', and 'entities'
prompt = PromptTemplate(
    input_variables=["input", "history", "entities"],
    template="""
You are a helpful assistant. Use the conversation history and extracted entities to respond.
Entities: {entities}
History: {history}
User: {input}
Assistant:"""
)

# ✅ Setup the conversation chain
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
        print("End the session")
        break

    elif query.lower() == "history":
        print("\n🔁 Conversation History:")
        print(memory.buffer)
        print()
        continue

    elif query.lower() == "entities":
        print("\n📌 Tracked Entities:")
        if memory.entity_store.store:
            for entity, info in memory.entity_store.store.items():
                print(f"{entity}: {info}")
        else:
            print("No entities tracked yet.")
        print()
        continue

    # Normal query processing
    response = conversation.invoke(query)
    print("Bot:", response['response'])

