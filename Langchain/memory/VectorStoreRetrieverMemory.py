from dotenv import load_dotenv
from pathlib import Path
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain

# ✅ Load environment variables
env_path = Path("/media/mtl/Volume F/PROJECTS/projects/.venv")
load_dotenv(dotenv_path=env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", GOOGLE_API_KEY=GOOGLE_API_KEY)

# ✅ Embeddings model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# ✅ Create FAISS vector store
vectorstore = FAISS.from_texts(texts=[], embedding=embedding_model)

# ✅ Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ✅ Create memory
memory = VectorStoreRetrieverMemory(retriever=retriever)

# ✅ Setup conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# ✅ Interactive loop
while True:
    query = input("You: ")
    
    if query.lower() == "exit":
        print("End the session")
        break

    elif query.lower() == "history":
        print("\n🔁 Retrieved Relevant History:")
        print(memory.load_memory_variables({})["history"])
        print()
        continue

    response = conversation.invoke(query)
    print("Bot:", response['response'])


