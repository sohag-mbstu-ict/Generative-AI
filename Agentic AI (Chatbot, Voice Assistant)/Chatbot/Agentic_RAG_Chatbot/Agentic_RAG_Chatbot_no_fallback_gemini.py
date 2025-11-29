import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

# Ensure Bangla characters are printed correctly
sys.stdout.reconfigure(encoding='utf-8')

class AgenticRAGChatbot:
    # gemini-pro-vision
    # gemini-1.5-flash
    # gemini-1.0-pro
    # gemini-1.5-flash-latest
    # gemini-2.0-flash-001
    def __init__(self, txt_path, env_path, model_name="gemini-2.0-flash-001"):
        self.txt_path = txt_path
        self.env_path = Path(env_path)
        self.model_name = model_name

        # Load environment variables (e.g., GOOGLE_API_KEY)
        load_dotenv(dotenv_path=self.env_path)
        self.api_key = os.getenv("GOOGLE_API_KEY")

        # Initialize components
        self.llm = ChatGoogleGenerativeAI(model=self.model_name)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        self.vector_store = self._build_vector_store()
        """Converts the FAISS index into a retriever interface.
        This lets the LLM fetch relevant document chunks based on the user query."""
        # If you're confident in your vector store quality, you can relax it to "similarity".    *************************
        # self.retriever = self.vector_store.as_retriever(search_type="similarity", search_k=3)
        self.retriever = self.vector_store.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={"score_threshold": 0.5},
                        search_k=5
                    )
        self.agent = self._initialize_agent()

    def fallback_llm_response(self, query: str) -> str:
        """Fallback to direct LLM response without RAG."""
        return self.llm.invoke(query)

    def _build_vector_store(self):
        """Load and process documents, then build FAISS vector store.
        Calls a helper function to build FAISS index from the text.
        Converts your raw documents into a searchable vector store."""
        loader = TextLoader(self.txt_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=70)
        docs = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(docs, self.embeddings)
        return vector_store

    def _get_weather(self, location: str):
        return f"The current weather in {location} is sunny and 25°C."

    def _initialize_agent(self):
        """Define tools and initialize the agent."""
        # RetrievalQA chain
        retrieval_qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True
        )

        # Define tools
        tools = [
            Tool(
                name="Document Retrieval",
                func=lambda q: retrieval_qa_chain({"query": q})["result"],
                description="Retrieve knowledge from the document database."
            ),
            Tool(
                name="Weather Tool",
                func=self._get_weather,
                description="Provides weather information for a given location."
            )
        ]

        # Create the agent
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True)
        return agent

    def fallback_llm_response(self,query):
        response = self.llm.invoke(query)
        return response

    def run(self):
        print("Agentic RAG Chatbot is running! Type 'exit' to quit.")
        while True:
            user_query = input("You: ")
            if user_query.lower() == "exit":
                print("Chatbot session ended.")
                break
            try:
                response = self.agent.run(user_query)
                get_relevant_documents_len = len(self.retriever.get_relevant_documents(user_query))
                print("333333333333333333333333333333 : ",get_relevant_documents_len)
                if get_relevant_documents_len==0:
                    response = self.fallback_llm_response(user_query)
                    print("response ################ : ",response)
                    reminder = f"📄 Your question doesn't match any content in the document. I'm responding using my built-in knowledge."
                    print(f"Bot: {reminder} {response}")
                else:
                    print(f"Bot: {response}")
            except Exception as e:
                print(f"Error: {e}")


# === Run the chatbot ===
if __name__ == "__main__":
    chatbot = AgenticRAGChatbot(
        txt_path="/media/mtl/Volume F/PROJECTS/projects/Gen_AI/Chatbot/knowledge_base/BINA_insects_diseases.txt",
        env_path="/media/mtl/Volume F/PROJECTS/projects/.venv")
    chatbot.run()
# What is LangChain used for?