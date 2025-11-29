# # https://huggingface.co/nvidia/domain-classifier
# # https://huggingface.co/nvidia/domain-classifier
# # https://huggingface.co/nvidia/domain-classifier

# 'Adult', 'Arts_and_Entertainment', 'Autos_and_Vehicles', 'Beauty_and_Fitness', 'Books_and_Literature', 'Business_and_Industrial', 'Computers_and_Electronics', 'Finance', 'Food_and_Drink', 'Games', 'Health', 'Hobbies_and_Leisure', 'Home_and_Garden', 'Internet_and_Telecom', 'Jobs_and_Education', 'Law_and_Government', 'News', 'Online_Communities', 'People_and_Society', 'Pets_and_Animals', 'Real_Estate', 'Science', 'Sensitive_Subjects', 'Shopping', 'Sports', 'Travel_and_Transportation'
# 'Food_and_Drink', 'Hobbies_and_Leisure', 'Home_and_Garden', 'Health', 'Business_and_Industrial', 'Science'
# 'Food_and_Drink', 'Home_and_Garden'


import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import PyTorchModelHubMixin
import streamlit as st


# ----------------------------
# Cached model loader
# ----------------------------
@st.cache_resource
def load_trained_domain_classifier(model_name="nvidia/domain-classifier"):
    """
    Loads the trained DomainClassifierModel and tokenizer.
    This preserves the trained weights.
    """
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DomainClassifierModel.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, config.id2label, device


# ----------------------------
# Original DomainClassifierModel
# ----------------------------
class DomainClassifierModel(nn.Module, PyTorchModelHubMixin):
    """
    Exact same architecture as your trained model.
    """
    def __init__(self, config):
        super().__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def forward(self, input_ids, attention_mask, **kwargs):
        features = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        dropped = self.dropout(features)
        logits = self.fc(dropped)
        probs = torch.softmax(logits[:, 0, :], dim=1)
        return probs


# ----------------------------
# DomainClassifier wrapper
# ----------------------------
class DomainClassifier:
    """
    High-level wrapper for easy domain prediction.
    """
    def __init__(self, model_name="nvidia/domain-classifier"):
        self.model, self.tokenizer, self.id2label, self.device = load_trained_domain_classifier(model_name)

    def predict(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            probs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        top_idx = torch.argmax(probs, dim=1).item()
        label = self.id2label[top_idx]
        score = probs[0][top_idx].item()

        return {"label": label, "score": round(score, 4)}

    def predict_batch(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            probs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        results = []
        for i, p in enumerate(probs):
            idx = torch.argmax(p).item()
            results.append({
                "text": texts[i],
                "label": self.id2label[idx],
                "score": round(p[idx].item(), 4)
            })
        return results
