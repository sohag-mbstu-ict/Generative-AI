import spacy
from spacy.pipeline import EntityRuler
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import streamlit as st

# ------------------------------
# Cache the heavy objects outside the class
# ------------------------------
@st.cache_resource
def load_spacy_model(_spacy_model="en_core_web_sm", _domain_patterns=None):
    nlp = spacy.load(_spacy_model)
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    if _domain_patterns is None:
        _domain_patterns = [
            {"label": "CROP", "pattern": [{"LOWER": "brinjal"}]},
            {"label": "CROP", "pattern": [{"LOWER": "eggplant"}]},
            {"label": "CROP", "pattern": [{"LOWER": "tomato"}]},
            {"label": "CROP", "pattern": [{"LOWER": "potato"}]},
            {"label": "CONTAINER", "pattern": [{"LOWER": "half drum"}]},
            {"label": "CONTAINER", "pattern": [{"LOWER": "pot"}]},
        ]
    ruler.add_patterns(_domain_patterns)
    return nlp

@st.cache_resource
def load_bert_pipeline(_bert_model="dslim/bert-base-NER"):
    tokenizer = AutoTokenizer.from_pretrained(_bert_model)
    model = AutoModelForTokenClassification.from_pretrained(_bert_model)
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )

# ------------------------------
# HybridNER class
# ------------------------------
class HybridNER:
    def __init__(self, spacy_model="en_core_web_sm", bert_model="dslim/bert-base-NER", domain_patterns=None):
        self.nlp = load_spacy_model(spacy_model, domain_patterns)
        self.ner_pipeline = load_bert_pipeline(bert_model)

    def extract_entities(self, text):
        entities = []

        # spaCy entities
        doc = self.nlp(text)
        for ent in doc.ents:
            entities.append({"name": ent.text, "entity": ent.label_})

        # BERT NER
        for res in self.ner_pipeline(text):
            entities.append({"name": res["word"], "entity": res["entity_group"]})

        return entities
