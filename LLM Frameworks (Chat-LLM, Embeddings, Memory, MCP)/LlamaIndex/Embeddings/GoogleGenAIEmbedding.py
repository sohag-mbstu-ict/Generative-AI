# # https://docs.llamaindex.ai/en/stable/examples/embeddings/gemini/

from dotenv import load_dotenv
from pathlib import Path
import os
# Specify the path to the .env file inside .venv
env_path = Path("/media/mtl/Volume F/PROJECTS/projects/.venv")
load_dotenv(dotenv_path=env_path)
# Now access the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("GOOGLE API KEY:", GOOGLE_API_KEY)

# imports
from llama_index.embeddings.gemini import GeminiEmbedding
# get API key and create embeddings

model_name = "models/embedding-001"
embed_model = GeminiEmbedding(
    model_name=model_name, api_key=GOOGLE_API_KEY, title="this is a document")

embeddings = embed_model.get_text_embedding("Google Gemini Embeddings.")
print(f"Dimension of embeddings: {len(embeddings)}")
print("embeddings[:5] : ",embeddings[:5])

print(" --------------------------------------------------------------- ")
# List of texts to embed
texts = ["Google Gemini Embeddings.", "Google is awesome."]

# Compute embeddings for each string
embeddings = [embed_model.get_text_embedding(text) for text in texts]

# Print info
for i, emb in enumerate(embeddings):
    print(f"Embedding for text {i + 1} ({texts[i]}):")
    print(f"Length: {len(emb)}")
    print(f"Sample: {emb[:5]}")
    print("---------------------------------------------------")

print(f"Dimension of embeddings: {len(embeddings)}")
print(embeddings[0][:5])
print(embeddings[1][:5])




