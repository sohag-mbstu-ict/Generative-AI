from pathlib import Path
from dotenv import load_dotenv
import json
import os
import time
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# --- Configuration ---
from pathlib import Path
from dotenv import load_dotenv
env_path = Path("/workspace/Gen_AI/.venv")
load_dotenv(dotenv_path=env_path)

# --- IMPORTANT: SET YOUR PATHS AND API KEY HERE ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# JSON_PATH = "/workspace/Gen_AI/Qwen3-4B/dataset/crops.json"  # <-- path to your JSON
JSON_PATH = "/workspace/Gen_AI/Qwen3-4B/dataset/GFL.json"  # <-- path to your JSON
VECTOR_STORE_PATH = "/workspace/Gen_AI/RAG/vector_store/json_faiss_index"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")
if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"JSON file not found at '{JSON_PATH}'.")


# --- Smooth Conversation Management ---
class SmoothRAGConversation:
    def __init__(self):
        self.conversation_history = []
        self.context_window = 5
        
    def enhance_query_with_context(self, user_query):
        if len(self.conversation_history) < 2:
            return user_query
        recent_context = self.conversation_history[-4:]
        context_str = ""
        for exchange in recent_context:
            if exchange["role"] == "user":
                context_str += f"User: {exchange['content']}\n"
            else:
                context_str += f"Assistant: {exchange['content']}\n"
        return user_query
    
    def add_exchange(self, role, content):
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > self.context_window * 2:
            self.conversation_history = self.conversation_history[-self.context_window * 2:]


# --- Chatbot Functionality Class (Updated for JSON) ---
class chatbot_functionality:
    def __init__(self, api_key):
        self.llm = ChatGroq(model="qwen/qwen3-32b", api_key=api_key)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.conversation_chain = None
        self.memory = None
        self.conversation_manager = SmoothRAGConversation()

    def get_json_texts(self, json_path):
        """Reads and formats text from JSON file."""
        print("🔹 Reading JSON data...")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Error reading JSON: {e}")
            return None
        
        # Convert JSON entries into readable text chunks
        docs = []
        for item in data:
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            formatted_text = f"Question: {instruction}\nAnswer: {output}"
            docs.append(formatted_text)
        print(f"✅ Loaded {len(docs)} JSON entries.")
        return docs

    def get_text_chunks(self, docs):
        """Splits text into manageable chunks."""
        print("🔹 Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        text_chunks = text_splitter.split_text("\n".join(docs))
        print(f"✅ Created {len(text_chunks)} text chunks.")
        return text_chunks

    def create_and_save_vector_store(self, text_chunks, save_path):
        """Creates a FAISS vector store from text chunks and saves it."""
        print(f"🔹 Creating and saving vector store to '{save_path}'...")
        vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
        vector_store.save_local(save_path)
        print("✅ Vector store created and saved.")
        return vector_store

    def load_vector_store(self, load_path):
        """Loads an existing FAISS vector store."""
        print(f"🔹 Loading vector store from '{load_path}'...")
        vector_store = FAISS.load_local(load_path, self.embeddings, allow_dangerous_deserialization=True)
        print("✅ Vector store loaded.")
        return vector_store

    def get_conversational_chain(self, vector_store):
        """Sets up the conversational retrieval chain."""
        print("🔹 Setting up conversational chain...")
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(),
            memory=self.memory
        )
        print("✅ Chain is ready.")

    def ask_question(self, user_question):
        """Handles Q&A with context enhancement."""
        if not self.conversation_chain:
            return "⚠️ Error: Chain not initialized."
        
        self.conversation_manager.add_exchange("user", user_question)
        enhanced_query = self.conversation_manager.enhance_query_with_context(user_question)
        
        start_time = time.time()
        try:
            response = self.conversation_chain({'question': enhanced_query})
            end_time = time.time()
            response_time = end_time - start_time
            
            answer = response['chat_history'][-1].content
            self.conversation_manager.add_exchange("assistant", answer)
            
            return f"{answer}\n\n(Response time: {response_time:.2f}s)"
        except Exception as e:
            error_msg = f"❌ An error occurred: {e}"
            self.conversation_manager.add_exchange("assistant", error_msg)
            return error_msg


# --- Main Execution Logic ---
def main():
    print("--- JSON RAG Chatbot with Smooth Conversation ---")

    bot = chatbot_functionality(GROQ_API_KEY)
    vector_store = None

    # 1️⃣ Load or create the FAISS store
    if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        vector_store = bot.load_vector_store(VECTOR_STORE_PATH)
    else:
        print("Vector store not found. Creating new one from JSON...")
        docs = bot.get_json_texts(JSON_PATH)
        if docs:
            text_chunks = bot.get_text_chunks(docs)
            vector_store = bot.create_and_save_vector_store(text_chunks, VECTOR_STORE_PATH)
        else:
            print("❌ JSON data error. Exiting.")
            return

    # 2️⃣ Initialize conversational chain
    bot.get_conversational_chain(vector_store)

    # 3️⃣ Start chat
    print("\n🤖 Chatbot ready! Ask about your JSON knowledge base. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Chat ended.")
            break
        if not user_input.strip():
            continue
        
        print("\n🔹 Thinking...")
        answer = bot.ask_question(user_input)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
