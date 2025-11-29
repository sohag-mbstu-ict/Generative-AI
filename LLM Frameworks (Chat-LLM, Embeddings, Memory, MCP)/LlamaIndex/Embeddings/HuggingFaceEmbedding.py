# # https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface/

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# loads https://huggingface.co/BAAI/bge-small-en-v1.5
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
embeddings = embed_model.get_text_embedding("Hello World!")
print(len(embeddings))
print(embeddings[:5])

