# # https://docs.llamaindex.ai/en/stable/examples/vector_stores/ChromaIndexDemo/

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.mistralai import MistralAI
from llama_index.core.schema import Document
from dotenv import load_dotenv
from pathlib import Path
import os
import chromadb
from typing import List

class PersistentQueryEngine:
    def __init__(
        self,
        # The path in which folder we will save the vector index collection
        db_path: str = "/media/mtl/Volume F/PROJECTS/projects/LlamaIndex/Vectore_Store/vector_index_folder",  # path to persist ChromaDB
        collection_name: str = "chromaDB",):  # collection name for storing vectors
        
        self.db_path = db_path
        self.collection_name = collection_name

        self.embed_model = None  # load embed model
        self.llm = None
        # init ChromaDB persistent client in which folder we will save the vector index collection
        self.db = chromadb.PersistentClient(path=self.db_path)  
        self.collection = self.db.get_or_create_collection(name=self.collection_name)  # create/load collection
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)  # wrap collection for LlamaIndex
        

    def MistralAIEmbeddings(self):
        # Specify the path to the .env file inside .venv
        env_path = Path("/media/mtl/Volume F/PROJECTS/projects/.venv")
        load_dotenv(dotenv_path=env_path)
        # Now access the API key
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        print("MISTRAL_API_KEY:", MISTRAL_API_KEY)

        from langchain_mistralai import MistralAIEmbeddings
        # Create the embedding model
        embed = MistralAIEmbeddings(
            model="mistral-embed",  # This is the default embedding model
            api_key=os.getenv("MISTRAL_API_KEY"))
        self.embed_model = embed

    def Mistral_LLM(self):
        self.llm = MistralAI(model="mistral-large-latest")

    def save_index(self, raw_docs: List[str]):
        documents = [Document(text=doc) for doc in raw_docs]  # convert strings to Document objects
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)  # setup storage context
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self.embed_model)  # build and save index with embeddings

    def load_index(self):
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model)  # reload index from saved vector

    def ask(self, query: str) -> str:
        # by default as_query_engine use openAI LLM model
        query_engine = self.index.as_query_engine(llm=self.llm)  # create query engine
        response = query_engine.query(query)  # ask question using vector-based retrieval
        return str(response)  # return the LLM response


if __name__ == "__main__":
    engine = PersistentQueryEngine()
    engine.MistralAIEmbeddings()
    engine.Mistral_LLM()

    # # Save embeddings (only once)   after saved we dont need to call save_index method
    # engine.save_index([
    #     "The author grew up in a small town.",
    #     "He spent his childhood exploring nature and reading books.",
    #     "He later became a writer and published novels."
    # ])

    # Later or in another session: Load from saved index
    engine.load_index()

    # Ask a question
    result = engine.ask("What did the author do growing up?")
    print(result)


