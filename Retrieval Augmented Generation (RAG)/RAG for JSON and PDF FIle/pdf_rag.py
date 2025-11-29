from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import time
from langchain_groq import ChatGroq
import random
import string
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import CohereEmbeddings

# --- Configuration ---
# Specify the path to your .env file if you use one
env_path = Path("/media/mtl/Volume F/PROJECTS/projects/.venv")
load_dotenv(dotenv_path=env_path)

# --- IMPORTANT: SET YOUR PATHS AND API KEY HERE ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PDF_PATH = "/media/mtl/Volume F/PROJECTS/projects/Gen_AI/RAG/pdf/Short Description of Chatbot.pdf"  # <--- CHANGE THIS to your PDF file path
VECTOR_STORE_PATH = "/media/mtl/Volume F/PROJECTS/projects/Gen_AI/RAG/vectore_store/faiss_index"        # <--- Directory to save/load the vector store

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file or environment variables.")
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF file not found at '{PDF_PATH}'. Please update the PDF_PATH variable.")


# --- Smooth Conversation Management ---
class SmoothRAGConversation:
    def __init__(self):
        self.conversation_history = []
        self.context_window = 5  # Keep last 5 exchanges
        
    def enhance_query_with_context(self, user_query):
        """Enhance the query with conversation context"""
        if len(self.conversation_history) < 2:
            return user_query
            
        # Get recent conversation context
        recent_context = self.conversation_history[-4:]  # Last 2 exchanges
        
        # Create a context string
        context_str = ""
        for exchange in recent_context:
            if exchange["role"] == "user":
                context_str += f"User: {exchange['content']}\n"
            else:
                context_str += f"Assistant: {exchange['content']}\n"
        
        # For now, return the original query
        # In a more advanced implementation, you could use an LLM to rewrite the query
        return user_query
    
    def generate_response(self, enhanced_query):
        """This method will be overridden by the chatbot's actual response generation"""
        pass
    
    def add_exchange(self, role, content):
        """Add a new exchange to the conversation history"""
        self.conversation_history.append({"role": role, "content": content})
        
        # Trim history if needed
        if len(self.conversation_history) > self.context_window * 2:
            self.conversation_history = self.conversation_history[-self.context_window * 2:]
    
    def get_context_for_retrieval(self):
        """Get formatted context for retrieval"""
        if len(self.conversation_history) < 2:
            return ""
            
        # Get recent conversation context
        recent_context = self.conversation_history[-4:]  # Last 2 exchanges
        
        # Create a context string
        context_str = ""
        for exchange in recent_context:
            if exchange["role"] == "user":
                context_str += f"User: {exchange['content']}\n"
            else:
                context_str += f"Assistant: {exchange['content']}\n"
        
        return context_str


# --- Chatbot Functionality Class (Updated with Smooth Conversation) ---
class chatbot_functionality:
    def __init__(self, api_key):
        self.llm = ChatGroq(model = "qwen/qwen3-32b")
        
        # self.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        # self.embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key = "SmUYCUOyq9yhhYalnKaS84to2uhR1IdI39MdcHMr")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.conversation_chain = None
        self.memory = None
        self.conversation_manager = SmoothRAGConversation()

    def get_pdf_text(self, pdf_path):
        """Extracts text from a single PDF file."""
        print("🔹 Reading PDF...")
        text = ""
        try:
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            return None
        if len(text) == 0:
            print("❌ PDF is empty or text could not be extracted.")
            return None
        print("✅ PDF read successfully.")
        return text

    def get_text_chunks(self, text):
        """Splits text into manageable chunks."""
        print("🔹 Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        chunks = text_splitter.split_text(text)
        print(f"✅ Created {len(chunks)} text chunks.")
        return chunks

    def create_and_save_vector_store(self, text_chunks, save_path):
        """Creates a vector store from text chunks and saves it."""
        print(f"🔹 Creating and saving vector store to '{save_path}'...")
        vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
        vector_store.save_local(save_path)
        print("✅ Vector store created and saved.")
        return vector_store

    def load_vector_store(self, load_path):
        """Loads a pre-existing vector store."""
        print(f"🔹 Loading existing vector store from '{load_path}'...")
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
        """Asks a question to the chain and returns the response with enhanced conversation flow."""
        if not self.conversation_chain:
            return "Error: The conversation chain is not initialized. Please process the PDF first."
        
        # Add user question to conversation history
        self.conversation_manager.add_exchange("user", user_question)
        
        # Enhance query with context
        enhanced_query = self.conversation_manager.enhance_query_with_context(user_question)
        
        start_time = time.time()
        try:
            response = self.conversation_chain({'question': enhanced_query})
            end_time = time.time()
            response_time = end_time - start_time
            
            # The last message in the chat history is the assistant's response
            answer = response['chat_history'][-1].content
            
            # Add assistant response to conversation history
            self.conversation_manager.add_exchange("assistant", answer)
            
            return f"{answer}\n\n(Response time: {response_time:.2f}s)"
        except Exception as e:
            error_msg = f"An error occurred: {e}. Please check your query or rerun the script."
            self.conversation_manager.add_exchange("assistant", error_msg)
            return error_msg
    
    def get_conversation_history(self):
        """Returns the conversation history for display purposes"""
        return self.conversation_manager.conversation_history


# --- Main Execution Logic ---

def main():
    """Main function to run the CLI RAG application."""
    print("--- PDF RAG Chatbot with Smooth Conversation ---")
    
    # 1. Initialize the bot
    bot = chatbot_functionality(GROQ_API_KEY)
    vector_store = None

    # 2. Load or Create the Vector Store
    if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        # If the vector store exists, load it
        vector_store = bot.load_vector_store(VECTOR_STORE_PATH)
    else:
        # If not, process the PDF to create it
        print("Vector store not found. Processing PDF for the first time...")
        raw_text = bot.get_pdf_text(PDF_PATH)
        if raw_text:
            text_chunks = bot.get_text_chunks(raw_text)
            vector_store = bot.create_and_save_vector_store(text_chunks, VECTOR_STORE_PATH)
        else:
            print("Exiting due to PDF processing error.")
            return

    # 3. Set up the conversational chain
    bot.get_conversational_chain(vector_store)

    # 4. Start the interactive chat loop
    print("\n🔹 Chatbot is ready! Ask questions about your PDF. (type 'exit' to quit)\n")
    
    # Display conversation history at the beginning
    history = bot.get_conversation_history()
    if history:
        print("--- Conversation History ---")
        for exchange in history:
            if exchange["role"] == "user":
                print(f"You: {exchange['content']}")
            else:
                print(f"Assistant: {exchange['content']}")
        print("---------------------------\n")
    
    while True:
        user_question = input("You: ")
        if user_question.lower() in ['exit', 'quit']:
            print("Chat session ended.")
            break
        
        if user_question.strip() == "":
            continue

        print("\n🔹 Thinking...")
        answer = bot.ask_question(user_question)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
    
    
    