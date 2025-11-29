import torch
from transformers import BertTokenizerFast, BertForTokenClassification

# ------------------------------
# 1️⃣ Load saved model & tokenizer
# ------------------------------
save_dir = "/home/gflml/Chatbot/Agri_NER_Dataset/trained_model"

tokenizer = BertTokenizerFast.from_pretrained(save_dir)
model = BertForTokenClassification.from_pretrained(save_dir)
model.eval()  # evaluation mode

# ------------------------------
# 2️⃣ Label mapping
# ------------------------------
labels = ["O", "B-CROP", "B-PEST", "B-FERTILIZER"]
id2label = {i: label for i, label in enumerate(labels)}

# ------------------------------
# 3️⃣ Inference function
# ------------------------------
def predict_ner(sentence):
    words = sentence.split()

    # Tokenize
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    # Remove offset_mapping if exists
    if "offset_mapping" in encoding:
        encoding.pop("offset_mapping")

    # Move inputs to model
    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

    word_ids = encoding.word_ids(batch_index=0)
    predicted_labels = []
    for word_idx, pred in zip(word_ids, predictions):
        if word_idx is None:
            continue
        if len(predicted_labels) <= word_idx:
            predicted_labels.append(id2label[pred])

    return list(zip(words, predicted_labels))

# ------------------------------
# 4️⃣ Example Usage
# ------------------------------
if __name__ == "__main__":
    example_sentence = "Potato is affected by aphids"
    ner_result = predict_ner(example_sentence)
    print("✅ NER Prediction:")
    for token, tag in ner_result:
        print(f"{token} -> {tag}")
