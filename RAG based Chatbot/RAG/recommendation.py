import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# No logic changed — only moved from CropExpertRAG
def get_recommendations_fn(query, embedder, instruction_embeddings, data, top_k=3, exclude_index=None):
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    sims = cosine_similarity(query_embedding, instruction_embeddings)[0]

    top_indices = np.argsort(sims)[::-1]
    if exclude_index is not None:
        top_indices = [i for i in top_indices if i != exclude_index]

    recs = []
    for idx in top_indices[:top_k]:
        recs.append({
            "instruction": data[idx]['instruction'],
            "answer": data[idx]['output'],
            "similarity": float(sims[idx])
        })
    return recs
