import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load QA dataset
with open("/home/gflml/Chatbot/RAG/41_crops.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Initialize embeddings model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create list of questions (instructions) and answers
questions = [item['instruction'] for item in qa_data]
answers = [item['output'] for item in qa_data]

# Generate embeddings for all questions
question_embeddings = embed_model.encode(questions, convert_to_numpy=True)

# Initialize FAISS index
embedding_dim = question_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(question_embeddings)

print(f"FAISS index created with {index.ntotal} questions")

def retrieve_answer(query, top_k=1):
    # Embed the user query
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    
    # Search in FAISS index
    distances, indices = index.search(query_vec, top_k)
    print("distances : ",distances)
    
    # Retrieve top answer
    if len(indices[0]) == 0:
        return "Sorry, I don't have an answer for that yet."
    
    print("instruction : ",questions[indices[0][0]] )
    return answers[indices[0][0]]  # Return top-1 answer

while True:
    user_input = input("\nAsk your question about rooftop gardening: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = retrieve_answer(user_input)
    print("Output :", response)






