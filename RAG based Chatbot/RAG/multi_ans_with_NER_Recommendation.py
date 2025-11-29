# # pip install backports.lzma
import os
# Disable GPU for CPU inference
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import streamlit as st
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from text_preprocessor import TextPreprocessor

# ---------------------------------------------
# ⚙️ Protected words & Preprocessor
# ---------------------------------------------
protected_words = ['Dr','Dr.', 'Chashi', 'Brinjal']
preprocessor = TextPreprocessor(enable_spell_check=True, remove_stopwords=False, protected_words=protected_words)



# ---------------------------------------------
# 1️⃣ Load Dataset
# ---------------------------------------------
DATA_PATH = "/home/gflml/Chatbot/RAG/41_crops.json"

@st.cache_data
def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

data = load_data()

# ---------------------------------------------
# 2️⃣ Load Models
# ---------------------------------------------
@st.cache_resource
def get_ner_model():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def get_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

nlp = get_ner_model()
embedder = get_embedder()

# ---------------------------------------------
# 3️⃣ Extract NER Entities for Dataset
# ---------------------------------------------
@st.cache_data
def extract_entities(text):
    doc = nlp(text)
    # Lowercase and deduplicate entities
    entities = list(set([ent.text.lower() for ent in doc.ents]))
    return entities

@st.cache_data
def precompute_entities(data):
    dataset_entities = []
    for item in data:
        ents = extract_entities(item['instruction'])
        dataset_entities.append(ents)
    return dataset_entities

dataset_entities = precompute_entities(data)

# ---------------------------------------------
# 4️⃣ Precompute Instruction Embeddings
# ---------------------------------------------
@st.cache_data
def compute_embeddings(data):
    texts = [item['instruction'] for item in data]
    return embedder.encode(texts, convert_to_tensor=False)

instruction_embeddings = compute_embeddings(data)

# ---------------------------------------------
# 5️⃣ NER-based Answer Retrieval
# ---------------------------------------------
def get_answer(user_query, ner_weight=0.5, embed_weight=0.5,
               ner_threshold=0.10, embed_threshold=0.60):

    # ---------------------------------
    # 1️⃣ Preprocess + extract NER
    # ---------------------------------
    cleaned_query = preprocessor.clean_text(user_query)
    query_entities = extract_entities(cleaned_query)

    # Compute NER overlap scores for all items
    ner_scores = []
    for ents in dataset_entities:
        overlap = len(set(query_entities).intersection(set(ents)))
        ner_scores.append(overlap)

    # Normalize NER scores (0-1)
    if max(ner_scores) > 0:
        ner_scores_norm = [s / max(ner_scores) for s in ner_scores]
    else:
        ner_scores_norm = [0] * len(ner_scores)

    # ---------------------------------
    # 2️⃣ BERT embedding similarity
    # ---------------------------------
    query_embedding = embedder.encode([cleaned_query], convert_to_tensor=False)
    embed_sims = cosine_similarity(query_embedding, instruction_embeddings)[0]

    # Normalize embedding scores (0–1)
    embed_scores_norm = (embed_sims - embed_sims.min()) / (embed_sims.max() - embed_sims.min() + 1e-9)

    # ---------------------------------
    # 3️⃣ Final weighted score
    # ---------------------------------
    final_scores = []
    for ner, emb in zip(ner_scores_norm, embed_scores_norm):
        final_scores.append(ner_weight * ner + embed_weight * emb)

    best_index = int(np.argmax(final_scores))
    best_score = final_scores[best_index]
    best_ner = ner_scores_norm[best_index]
    best_emb = embed_sims[best_index]

    # ---------------------------------
    # 4️⃣ Decide if we know answer
    # ---------------------------------
    if best_ner < ner_threshold and best_emb < embed_threshold:
        return {
            "answer": "Sorry, I could not find an answer.",
            "score": best_score,
            "ner_score": float(best_ner),
            "embedding_score": float(best_emb),
            "index": None
        }

    # ---------------------------------
    # 5️⃣ Return matched answer
    # ---------------------------------
    return {
        "answer": data[best_index]["output"],
        "matched_instruction": data[best_index]["instruction"],
        "score": float(best_score),
        "ner_score": float(best_ner),
        "embedding_score": float(best_emb),
        "query_entities": query_entities,
        "instruction_entities": dataset_entities[best_index],
        "index": best_index
    }

crop_list = ["citrus", "banana", "brinjal", "eggplant", "tomato",
             "potato", "rice", "mango", "bean", "cucurbit", "bottle gourd"]

def detect_crops(query, crop_list):
    q = query.lower()
    return [c for c in crop_list if c in q]

def get_multi_answer_dynamic(user_query):

    # -----------------------------
    # 1️⃣ Detect crops in the query
    # -----------------------------
    crops = detect_crops(user_query, crop_list)

    # If no crop detected → fallback to single-answer mode
    if not crops:
        return {"type": "single", "data": get_answer(user_query)}

    results = []
    cleaned_query = preprocessor.clean_text(user_query)
    query_entities = extract_entities(cleaned_query)

    # -----------------------------
    # 2️⃣ For each crop, fetch all dataset rows containing that crop
    # -----------------------------
    for idx, item in enumerate(data):

        inst = item["instruction"].lower()

        # If any crop name appears, count it as relevant
        if any(crop.lower() in inst for crop in crops):

            # Hybrid scoring using your original get_answer() logic
            hybrid_res = get_answer(item["instruction"])

            # Filter low-confidence items
            if hybrid_res["embedding_score"] < 0.40:
                continue

            # Create unified detailed result
            results.append({
                "answer": item["output"],
                "matched_instruction": item["instruction"],
                "score": hybrid_res["score"],
                "ner_score": hybrid_res["ner_score"],
                "embedding_score": hybrid_res["embedding_score"],
                "query_entities": hybrid_res.get("query_entities", []),
                "instruction_entities": hybrid_res.get("instruction_entities", []),
                "index": hybrid_res["index"],
            })

    # -----------------------------
    # 3️⃣ No results found
    # -----------------------------
    if not results:
        return {"type": "none", "data": "Sorry, I could not find an answer."}

    # -----------------------------
    # 4️⃣ Sort by best embedding score
    # -----------------------------
    results = sorted(results, key=lambda x: x["embedding_score"], reverse=True)

    return {
        "type": "multi",
        "data": results
    }


# ---------------------------------------------
# 6️⃣ Semantic Recommendation System
# ---------------------------------------------
def get_recommendations(query, top_k=3, exclude_index=None):
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    sims = cosine_similarity(query_embedding, instruction_embeddings)[0]
    top_indices = np.argsort(sims)[::-1]

    if exclude_index is not None:
        top_indices = [i for i in top_indices if i != exclude_index]

    recommendations = []
    for idx in top_indices[:top_k]:
        recommendations.append({
            "instruction": data[idx]['instruction'],
            "answer": data[idx]['output'],
            "similarity": float(sims[idx])
        })
    return recommendations

# ---------------------------------------------
# 7️⃣ Streamlit UI
# ---------------------------------------------
st.set_page_config(page_title="🌾 Crop Expert Assistant (NER)", page_icon="🌾", layout="centered")

st.title("🌾 Crop Expert Assistant (NER-based)")
st.caption("Ask anything about crops — I’ll extract key entities and find the most relevant answer using NER and semantic similarity.")

query = st.text_input("💬 Ask your question:", placeholder="e.g. How to control aphids in mustard crops?")

if query:
    with st.spinner("🔍 Analyzing your question..."):
        res = get_multi_answer_dynamic(query)
        print("res : ",res)

    st.markdown("## 🧠 Best Matched Answer")

    if res["type"] == "single":
        st.success(res["data"]["answer"])
        st.markdown("### 🔍 Match Details")
        st.write(f"**Matched Instruction:** {res['data'].get('matched_instruction', 'None')}")
        st.write(f"**Entity Overlap Score:** {res['data']['score']}")
        # st.write(f"**Embedding Score:** {res['embedding_score']}")
        st.write(f"**Query Entities:** {', '.join(res['data'].get('query_entities', []))}")
        st.write(f"**Instruction Entities:** {', '.join(res['data'].get('instruction_entities', []))}")

    elif res["type"] == "multi":
        for i, item in enumerate(res["data"], start=1):
            st.markdown(f"### 🧠 Match {i}")
            st.success(item["answer"])
            st.write(f"**Matched Instruction:** {item.get('matched_instruction', 'None')}")
            st.write(f"**Entity Overlap Score:** {item['score']}")
            st.write(f"**NER Score:** {item['ner_score']}")
            st.write(f"**Embedding Score:** {item['embedding_score']}")
            st.write(f"**Query Entities:** {', '.join(item.get('query_entities', []))}")
            st.write(f"**Instruction Entities:** {', '.join(item.get('instruction_entities', []))}")

    else:  # none
        st.warning(res["data"])

    # ---------------------------------------------
    # 🌱 Recommended Related Questions
    # ---------------------------------------------
    st.markdown("## 🌱 Recommended Related Questions")
    recs = get_recommendations(query, top_k=3, exclude_index=res.get("index"))

    if recs:
        for i, rec in enumerate(recs, start=1):
            st.markdown(f"**{i}. {rec['instruction']}**")
            st.caption(f"Similarity Score: {rec['similarity']:.3f}")
    else:
        st.write("No related recommendations found.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + spaCy (NER) + SentenceTransformer")



