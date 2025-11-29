import json
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import classification_report
import numpy as np
import os

# --------------------------------------
# 1️⃣ Dataset Class
# --------------------------------------
class NERDataset(Dataset):
    def __init__(self, path, tokenizer, label2id, max_len=64):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.samples = []
        self.max_len = max_len

        # Detect JSONL or JSON
        with open(path, "r") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "{":  # JSONL
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))
            else:  # JSON array
                self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]["tokens"]
        labels = self.samples[idx]["ner_tags"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        last_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)  # ignore special tokens
            elif word_id != last_word_id:
                aligned_labels.append(self.label2id[labels[word_id]])
            else:
                aligned_labels.append(-100)
            last_word_id = word_id

        item = {k: v.squeeze() for k, v in encoding.items()}
        item["labels"] = torch.tensor(aligned_labels)
        return item

# --------------------------------------
# 2️⃣ Labels
# --------------------------------------
labels = ["O", "B-CROP", "B-PEST", "B-FERTILIZER"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

# --------------------------------------
# 3️⃣ Tokenizer & Datasets
# --------------------------------------
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

train_dataset = NERDataset("data/train.jsonl", tokenizer, label2id)
eval_dataset  = NERDataset("data/valid.jsonl", tokenizer, label2id)
test_dataset  = NERDataset("data/test.jsonl", tokenizer, label2id)

# --------------------------------------
# 4️⃣ Model
# --------------------------------------
model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# --------------------------------------
# 5️⃣ Metrics
# --------------------------------------
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)

    true_labels = []
    true_preds = []

    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_labels.append(id2label[l])
                true_preds.append(id2label[p])

    report = classification_report(true_labels, true_preds, output_dict=True)
    return {
        "f1": report["weighted avg"]["f1-score"],
        "accuracy": report["accuracy"]
    }

# --------------------------------------
# 6️⃣ Training Arguments
# --------------------------------------
training_args = TrainingArguments(
    output_dir=None,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=10,
    report_to="none"
)

# --------------------------------------
# 7️⃣ Trainer
# --------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# --------------------------------------
# 8️⃣ Train
# --------------------------------------
trainer.train()

# --------------------------------------
# 9️⃣ Save Final Model
# --------------------------------------
save_dir = "/home/gflml/Chatbot/Agri_NER_Dataset/trained_model/"
# os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("✅ Training completed and final model saved!")

# --------------------------------------
# 🔟 Evaluate on Test Set
# --------------------------------------
print("📊 Evaluating on Test Set...")
predictions, labels, _ = trainer.predict(test_dataset)

preds = np.argmax(predictions, axis=-1)
true_labels = []
true_preds = []

for pred_seq, label_seq in zip(preds, labels):
    for p, l in zip(pred_seq, label_seq):
        if l != -100:
            true_labels.append(id2label[l])
            true_preds.append(id2label[p])

report = classification_report(true_labels, true_preds)
print("✅ Test Set Evaluation:\n")
print(report)
