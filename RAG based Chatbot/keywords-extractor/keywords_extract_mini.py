import streamlit as st
from keybert import KeyBERT
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Optional: Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# -------------------------------
# 1️⃣ Load your dataset
# -------------------------------
# DATA_PATH = "/media/mtl/Volume F/PROJECTS/LLM-E-Commerce/NLP/dataset/crops.json"
DATA_PATH = "/media/mtl/Volume F/PROJECTS/LLM-E-Commerce/NLP/dataset/41_crops.json"

@st.cache_data
def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

data = load_data()

# -------------------------------
# 2️⃣ Initialize KeyBERT model
# -------------------------------
@st.cache_resource
def get_kw_model():
    return KeyBERT('all-MiniLM-L6-v2')

kw_model = get_kw_model()

# Precompute dataset keywords
@st.cache_data
def precompute_keywords(data):
    dataset_keywords = []
    for item in data:
        kws = kw_model.extract_keywords(item['instruction'], keyphrase_ngram_range=(1, 2), top_n=5)
        kws_only = [kw for kw, score in kws]
        dataset_keywords.append(kws_only)
    return dataset_keywords

dataset_keywords = precompute_keywords(data)

# -------------------------------
# 3️⃣ Answer retrieval function
# -------------------------------
def get_answer(user_query):
    query_kws = kw_model.extract_keywords(user_query, keyphrase_ngram_range=(1, 2), top_n=5)
    query_kws_only = [kw for kw, score in query_kws]

    best_score = 0
    best_index = -1
    for i, kws in enumerate(dataset_keywords):
        overlap = len(set(query_kws_only).intersection(set(kws)))
        if overlap > best_score:
            best_score = overlap
            best_index = i

    if best_index != -1 and best_score > 0:
        return {
            "answer": data[best_index]['output'],
            "matched_instruction": data[best_index]['instruction'],
            "score": best_score,
            "query_keywords": query_kws_only,
            "instruction_keywords": dataset_keywords[best_index]
        }
    else:
        return {"answer": "Sorry, I could not find an answer.", "score": 0}

# -------------------------------
# 4️⃣ Streamlit UI
# -------------------------------
st.set_page_config(page_title="Crop Expert Assistant", page_icon="🌾", layout="centered")

st.title("🌾 Crop Expert Assistant")
st.caption("Ask anything about crops, and I’ll try to find the best answer from your dataset.")

query = st.text_input("💬 Ask your question:", placeholder="e.g. How to control aphids in mustard crops?")

if query:
    with st.spinner("Analyzing your question..."):
        res = get_answer(query)

    st.markdown("### 🧠 Answer")
    st.success(res["answer"])

    st.markdown("### 🔍 Details")
    st.write(f"**Matched Instruction:** {res.get('matched_instruction', 'None')}")
    st.write(f"**Similarity Score:** {res['score']}")
    st.write(f"**Query Keywords:** {', '.join(res.get('query_keywords', []))}")
    st.write(f"**Instruction Keywords:** {', '.join(res.get('instruction_keywords', []))}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + KeyBERT")

