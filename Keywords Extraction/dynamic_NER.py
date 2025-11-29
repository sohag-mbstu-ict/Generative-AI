import os
# Disable GPU for CPU inference
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import spacy
from sentence_transformers import SentenceTransformer, util
import json

# Load NLP model
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initial crops & knowledge base
crops = ["citrus", "banana"]
kb = {
    "citrus": {"diseases": ["Citrus Greening", "Citrus Canker"], "insects": ["Asian Citrus Psyllid"]},
    "banana": {"diseases": ["Panama Disease"], "insects": ["Banana Weevil"]}
}

# Threshold for adding new entity
SIM_THRESHOLD = 0.7

def dynamic_extract(text):
    doc = nlp(text.lower())
    results = {}

    for crop in crops:
        results[crop] = {"diseases": [], "insects": []}
        for sent in doc.sents:
            if crop in sent.text:
                # Extract candidate noun phrases
                candidates = [chunk.text for chunk in sent.noun_chunks if chunk.text != crop]
                for candidate in candidates:
                    # Compare with existing entities
                    for entity_type in ["diseases", "insects"]:
                        existing = kb[crop][entity_type]
                        if existing:
                            sims = util.cos_sim(model.encode(candidate), model.encode(existing))
                            max_sim = sims.max().item()
                            if max_sim > SIM_THRESHOLD:
                                results[crop][entity_type].append(existing[sims.argmax()])
                                break
                    else:
                        # If not similar to existing, dynamically add
                        # For simplicity, classify based on keywords
                        if any(word in candidate for word in ["virus","disease","canker","spot","wilt"]):
                            kb[crop]["diseases"].append(candidate)
                            results[crop]["diseases"].append(candidate)
                        else:
                            kb[crop]["insects"].append(candidate)
                            results[crop]["insects"].append(candidate)
    return results, kb



text = """
Citrus trees suffer from Citrus Black Spot and are attacked by Red Spider Mite.
Banana plants are affected by Banana Streak Virus and Fruit Flies.
"""

mapping, updated_kb = dynamic_extract(text)
print("Mapping:", mapping)
print("Updated KB:", updated_kb)


