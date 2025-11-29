# # https://docs.llamaindex.ai/en/stable/examples/embeddings/mistralai/

from dotenv import load_dotenv
from pathlib import Path
import os
# Specify the path to the .env file inside .venv
env_path = Path("/media/mtl/Volume F/PROJECTS/projects/.venv")
load_dotenv(dotenv_path=env_path)
# Now access the API key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
print("MISTRAL_API_KEY:", MISTRAL_API_KEY)

# imports
from llama_index.embeddings.mistralai import MistralAIEmbedding
# get API key and create embeddings
api_key = MISTRAL_API_KEY
model_name = "mistral-embed"
embed_model = MistralAIEmbedding(model_name=model_name, api_key=api_key)
embeddings = embed_model.get_text_embedding("La Plateforme - The Platform")
print(" embeddings[:5] : ",embeddings[:5])
