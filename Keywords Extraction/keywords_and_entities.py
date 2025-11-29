import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import spacy
from keybert import KeyBERT


nlp = spacy.load("en_core_web_md")
# nlp = spacy.load("/media/mtl/Volume F/PROJECTS/LLM-E-Commerce/NLP/en_core_web_sm")
kw_model = KeyBERT()

def extract_keywords_and_entities(query: str):
    doc = nlp(query)
    # keywords (using KeyBERT)
    keywords = [kw for kw, score in kw_model.extract_keywords(query, top_n=5)]
    # named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return {"keywords": keywords, "entities": entities}

# Example usage
query = "Can I use the Dr. Chashi app on my iPhone without internet?"
info = extract_keywords_and_entities(query)
print(info)





