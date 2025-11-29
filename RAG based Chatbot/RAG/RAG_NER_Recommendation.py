import os
import time
import json
import sys
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text_preprocessor import TextPreprocessor
from NER_using_BERT_SpaCy import HybridNER
from domain_classifier.nvidia_domain_classifier import DomainClassifier
from recommendation import get_recommendations_fn
from fallback_LLM.fallback_qween0_6b import QwenChatbot

# ----------------------------------------------------
# Cache-heavy compute function outside the class
# ----------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_instruction_embeddings(_embedder, _data):
    texts = [item['instruction'] for item in _data]
    return _embedder.encode(texts, convert_to_tensor=False)

# ----------------------------
# Cached fallback LLM
# ----------------------------
@st.cache_resource(show_spinner=False, hash_funcs={QwenChatbot: lambda _: None})
def load_fallback_llm():
    return QwenChatbot()


# ----------------------------------------------------
# 🔥 Optimized CropExpertRAG (NO CODE CHANGES, ONLY CACHING)
# ----------------------------------------------------
class CropExpertRAG:
    def __init__(self, data_path):
        self.start_time = time.time()

        # -----------------------------------
        # Load dependencies (cached)
        # -----------------------------------
        self.ner_obj = self.load_cached_hybrid_ner()       # Hybrid NER cached
        self.domain_classifier = self.load_cached_dc()     # Domain classifier cached

        protected_words = ['Dr', 'Dr.', 'Chashi', 'Brinjal']
        self.preprocessor = self.load_cached_preprocessor(
            enable_spell_check=True,
            remove_stopwords=False,
            protected_words=protected_words
        )

        # -----------------------------------
        # Load dataset (cached)
        # -----------------------------------
        self.data = self.load_json(data_path)

        # -----------------------------------
        # Load models (cached resources)
        # -----------------------------------
        self.nlp = self.load_spacy_model()
        self.embedder = self.load_sentence_transformer()

        # -----------------------------------
        # Precompute embeddings (cached)
        # -----------------------------------
        self.instruction_embeddings = compute_instruction_embeddings(
            self.embedder, self.data
        )

    # ----------------------------
    # Cached object loaders
    # ----------------------------
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_cached_hybrid_ner():
        return HybridNER()

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_cached_dc():
        return DomainClassifier()

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_cached_preprocessor(enable_spell_check, remove_stopwords, protected_words):
        return TextPreprocessor(
            enable_spell_check=enable_spell_check,
            remove_stopwords=remove_stopwords,
            protected_words=protected_words
        )

    # ----------------------------
    # Dataset loader
    # ----------------------------
    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_json(_path):
        with open(_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ----------------------------
    # Load spaCy model (heavy)
    # ----------------------------
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_spacy_model(_spacy_model="en_core_web_sm"):
        return spacy.load(_spacy_model)

    # ----------------------------
    # Load SentenceTransformer (heavy)
    # ----------------------------
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_sentence_transformer(_model_name="all-MiniLM-L6-v2"):
        return SentenceTransformer(_model_name)

    # ----------------------------------------------------
    # Cached text cleaning
    # ----------------------------------------------------
    @staticmethod
    @st.cache_data(show_spinner=False)
    def cached_clean_text(_preprocessor, text):
        return _preprocessor.clean_text(text)

    # ----------------------------
    # Answer retrieval
    # ----------------------------
    def get_answer(self, user_query, embed_threshold=0.60):

        # Clean text (cached)
        cleaned_query = self.cached_clean_text(self.preprocessor, user_query)
        print("---------------cleaned_query ------------- : ", cleaned_query)

        # Embedding
        query_embedding = self.embedder.encode([cleaned_query], convert_to_tensor=False)
        embed_sims = cosine_similarity(query_embedding, self.instruction_embeddings)[0]

        # Normalize
        embed_scores_norm = (embed_sims - embed_sims.min()) / (embed_sims.max() - embed_sims.min() + 1e-9)
        best_index = int(np.argmax(embed_scores_norm))
        best_emb = embed_sims[best_index]

        # NER + domain
        query_entities_hybrid = self.ner_obj.extract_entities(cleaned_query)
        domain = self.domain_classifier.predict(cleaned_query)

        if best_emb < embed_threshold:
            qween_bot = load_fallback_llm()
            thinking, answer = qween_bot.generate(cleaned_query)
            return {
                "answer": answer,
                "query_entities": query_entities_hybrid,
                "domain": domain,
                "embedding_score": float(best_emb),
                "instruction_entities": None,
                "index": None
            }

        matched_instruction = self.data[best_index]["instruction"]
        matched_entities = self.ner_obj.extract_entities(matched_instruction)

        return {
            "answer": self.data[best_index]["output"],
            "matched_instruction": matched_instruction,
            "query_entities": query_entities_hybrid,
            "instruction_entities": matched_entities,
            "domain": domain,
            "embedding_score": float(best_emb),
            "index": best_index
        }

    # ----------------------------
    # Recommendations
    # ----------------------------
    def get_recommendations(self, query, top_k=3, exclude_index=None):
        return get_recommendations_fn(
            query,
            self.embedder,
            self.instruction_embeddings,
            self.data,
            top_k,
            exclude_index)

    # ----------------------------
    # Time elapsed
    # ----------------------------
    def get_total_time(self):
        return round(time.time() - self.start_time, 3)


# ----------------------------
# Streamlit UI
# ----------------------------
def run_ui():
    DATA_PATH = "/home/gflml/Chatbot/dataset/41_crops.json"

    rag = CropExpertRAG(DATA_PATH)
    # ------------------------- Debugging ------------------------------------
    # res = rag.get_answer("rooftop gardening")
    # ------------------------- Debugging ------------------------------------

    st.set_page_config(page_title="🌾 Crop Expert Assistant (NER)", page_icon="🌾")
    st.title("🌾 Crop Expert Assistant (NER-Based)")
    st.caption("Ask anything about crops, pests, diseases, and remedies.")

    query = st.text_input("💬 Ask your question:")

    if query:
        with st.spinner("Analyzing..."):
            res = rag.get_answer(query)

        st.markdown("## 🧠 Best Answer")
        st.success(res["answer"])

        st.markdown("### 🔍 Details")
        st.write(f"**Matched Instruction:** {res.get('matched_instruction', 'None')}")
        st.write(f"**Embedding Score:** {res['embedding_score']}")
        st.write(f"**Domain:** {res['domain']}")
        st.write(f"**Query Entities:** {res['query_entities']}")
        st.write(f"**Instruction Entities:** {res['instruction_entities']}")
        st.write(f"⏱️ **Total Time:** {rag.get_total_time()} sec")

        st.markdown("## 🌱 Recommended Questions")
        recs = rag.get_recommendations(query, top_k=3, exclude_index=res["index"])
        for i, rec in enumerate(recs, 1):
            st.markdown(f"**{i}. {rec['instruction']}**")
            st.caption(f"Similarity: {rec['similarity']:.3f}")


# Run Streamlit
run_ui()
