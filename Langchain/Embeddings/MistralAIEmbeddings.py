# # https://python.langchain.com/api_reference/mistralai/embeddings/langchain_mistralai.embeddings.MistralAIEmbeddings.html#langchain_mistralai.embeddings.MistralAIEmbeddings

from dotenv import load_dotenv
from pathlib import Path
import os
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
    api_key=os.getenv("MISTRAL_API_KEY")
)

# Example: get embedding for a list of texts
texts = ["LangChain makes it easy to work with LLMs.", "Mistral models are fast and powerful."]
embeddings = embed.embed_documents(texts)

# Show embeddings
print(f"Embedding for text 1 (len={len(embeddings[0])}):", embeddings[0][:10])  # print first 10 values
print(f"Embedding for text 2 (len={len(embeddings[1])}):", embeddings[1][:10])  # print first 10 values
