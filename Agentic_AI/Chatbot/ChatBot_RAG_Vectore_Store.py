import streamlit as st
from pathlib import Path
import os
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage


class RAGChatbot:
    def __init__(self, data_dir="data", index_dir="index_store", api_key=None):
        self.DATA_DIR = Path(data_dir)
        self.INDEX_DIR = Path(index_dir)
        self.GOOGLE_API_KEY = api_key

        self.chat_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",
            GOOGLE_API_KEY=self.GOOGLE_API_KEY)
        self.embedding_model = LangchainEmbedding(
            GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

        self._prepare_directories()

    def _prepare_directories(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    def handle_pdf_upload_st(self, uploaded_file):
        file_name = uploaded_file.name
        file_path = self.DATA_DIR / file_name
        index_subdir = self.INDEX_DIR / file_name.replace(".pdf", "")

        if index_subdir.exists():
            st.warning("✅ This PDF has already been processed. Using existing vector store.")
            return self.load_index(index_subdir)
        else:
            st.success("📄 PDF uploaded successfully. Creating vector store...")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            return self.create_index(file_path, index_subdir)

    def create_index(self, file_path, index_subdir):
        # Loads the PDF file as documents using SimpleDirectoryReader.
        # Converts the file into a format suitable for vector embedding and indexing.
        docs = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
        # This enables semantic search — allowing the chatbot to retrieve answers based on embeddings rather than keyword matching.
        index = VectorStoreIndex.from_documents(docs, embed_model=self.embedding_model)
        # Saves the created vector index to disk at the target index_subdir.
        index.storage_context.persist(persist_dir=str(index_subdir))
        return index

    def load_index(self, index_subdir):
        # Creates a StorageContext pointing to the directory where the index is saved.
        storage_context = StorageContext.from_defaults(persist_dir=str(index_subdir))
        # Loads the previously stored index from disk, and injects the correct embedding model.
        return load_index_from_storage(storage_context, embed_model=self.embedding_model)

    def ask_question(self, index, question):
        query_engine = index.as_query_engine(llm=self.chat_model)
        response = query_engine.query(question)
        return response


# === Streamlit App ===
st.set_page_config(page_title="RAG Chatbot")
st.title("📄 PDF-based RAG Chatbot (Dr. Chashi)")
st.caption("Upload a PDF and ask questions based on its content.")

# Load environment variables
load_dotenv(dotenv_path=Path("/media/mtl/Volume F/PROJECTS/projects/.venv"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the chatbot class
bot = RAGChatbot(api_key=GOOGLE_API_KEY)

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Handle PDF and build/load index
    index = bot.handle_pdf_upload_st(uploaded_file)

    # Store query engine in session state
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = index.as_query_engine(llm=bot.chat_model)

    # Initialize message history
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant answering questions from an uploaded PDF.")]

    # Display chat history
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    # Handle chat input
    if user_query := st.chat_input("Ask something about your uploaded PDF..."):
        st.session_state.messages.append(HumanMessage(content=user_query))
        with st.chat_message("user"):
            st.markdown(user_query)

        response = st.session_state.query_engine.query(user_query)
        st.session_state.messages.append(AIMessage(content=response.response))
        with st.chat_message("assistant"):
            st.markdown(response.response)

else:
    st.info("📤 Please upload a PDF file to begin.")


