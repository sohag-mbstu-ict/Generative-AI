# # https://python.langchain.com/api_reference/google_genai/embeddings/langchain_google_genai.embeddings.GoogleGenerativeAIEmbeddings.html#langchain_google_genai.embeddings.GoogleGenerativeAIEmbeddings

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path
import os
# Specify the path to the .env file inside .venv
env_path = Path("/media/mtl/Volume F/PROJECTS/projects/.venv")
load_dotenv(dotenv_path=env_path)
# Now access the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("GOOGLE API KEY:", GOOGLE_API_KEY)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",GOOGLE_API_KEY=GOOGLE_API_KEY)
ans = embeddings.embed_query("What's our Q1 revenue?")
print(len(ans))
print(ans[:8])

