# # https://python.langchain.com/api_reference/ollama/embeddings/langchain_ollama.embeddings.OllamaEmbeddings.html#langchain_ollama.embeddings.OllamaEmbeddings

from langchain_ollama import OllamaEmbeddings

embed = OllamaEmbeddings(
    model="gemma:2b")

input_text = "The meaning of life is 42"
vector = embed.embed_query(input_text)
print(len(vector))
print(vector[:3])


