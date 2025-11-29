from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments, AutoModelForTokenClassification, pipeline
import evaluate
import numpy as np
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, DatasetDict
from seqeval.metrics import classification_report, accuracy_score, f1_score

def read_cleanconll_file(filepath):
    tokens, labels = [], []
    sent_tokens, sent_labels = [], []

    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if sent_tokens:
                    tokens.append(sent_tokens)
                    labels.append(sent_labels)
                    sent_tokens, sent_labels = [], []
            else:
                parts = line.split()
                if len(parts) == 4:
                    token, ner = parts[0], parts[3]
                    sent_tokens.append(token)
                    sent_labels.append(ner)
    return tokens, labels

train_tokens, train_labels = read_cleanconll_file("/home/gflml/Chatbot/NER/data/conllpp_train.txt")
val_tokens, val_labels     = read_cleanconll_file("/home/gflml/Chatbot/NER/data/conllpp_valid.txt")
test_tokens, test_labels   = read_cleanconll_file("/home/gflml/Chatbot/NER/data/conllpp_test.txt")
print(len(train_tokens), len(train_labels))


for i in range(5):
    print(f"\n🔹 Sentence {i+1}:")
    print("Tokens: ", train_tokens[i])
    print("Labels: ", train_labels[i])

print(len(train_tokens), len(train_labels))
checkpoint = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

b2i = {
    "B-PER": "I-PER",
    "B-LOC": "I-LOC",
    "B-ORG": "I-ORG",
    "B-MISC": "I-MISC"
}
def allign_words(labels, word_ids):
    alligned_words = []
    last_word = None
    for word in word_ids:
        if word is None:
            label = -100
        elif word != last_word:
            label = labels[word]
        else:
            label = labels[word]
            if label in b2i:
                label = b2i[label]
        alligned_words.append(label)
        last_word = word
    return alligned_words




# ------------------------------------------------
# 1. Load your cleaned CoNLL2003 dataset
# ------------------------------------------------
def read_cleanconll_file(filepath):
    tokens, labels = [], []
    sent_tokens, sent_labels = [], []

    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if sent_tokens:
                    tokens.append(sent_tokens)
                    labels.append(sent_labels)
                    sent_tokens, sent_labels = [], []
            else:
                parts = line.split()
                if len(parts) == 4:
                    token, ner = parts[0], parts[3]
                    sent_tokens.append(token)
                    sent_labels.append(ner)
    return tokens, labels


train_tokens, train_labels = read_cleanconll_file("/home/gflml/Chatbot/NER/data/conllpp_train.txt")
val_tokens, val_labels = read_cleanconll_file("/home/gflml/Chatbot/NER/data/conllpp_valid.txt")
test_tokens, test_labels = read_cleanconll_file("/home/gflml/Chatbot/NER/data/conllpp_test.txt")

print("Train sentences:", len(train_tokens))

# ------------------------------------------------
# 2. Create HuggingFace Dataset objects
# ------------------------------------------------
train_ds = Dataset.from_dict({"tokens": train_tokens, "ner_tags": train_labels})
val_ds   = Dataset.from_dict({"tokens": val_tokens, "ner_tags": val_labels})
test_ds  = Dataset.from_dict({"tokens": test_tokens, "ner_tags": test_labels})

dataset = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds
})


# ------------------------------------------------
# 3. Prepare label mapping
# ------------------------------------------------
unique_labels = sorted(list({label for doc in train_labels for label in doc}))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

print("Labels:", unique_labels)

# ------------------------------------------------
# 4. Load tokenizer + model
# ------------------------------------------------
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForTokenClassification.from_pretrained(
    checkpoint,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)


# ------------------------------------------------
# 5. Tokenize + align labels
# ------------------------------------------------
def tokenize_and_align_labels(example):
    tokenized = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    word_ids = tokenized.word_ids()
    previous_word = None
    aligned_labels = []

    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != previous_word:
            aligned_labels.append(label2id[example["ner_tags"][word_id]])
        else:
            aligned_labels.append(-100)

        previous_word = word_id

    tokenized["labels"] = aligned_labels
    return tokenized


tokenized_ds = dataset.map(tokenize_and_align_labels, batched=False)
print("tokenized_ds : " ,tokenized_ds)

from transformers import TrainingArguments
# ------------------------------------------------
# 6. Training
# ------------------------------------------------
data_collator = DataCollatorForTokenClassification(tokenizer)
training_args = TrainingArguments(
    output_dir="/home/gflml/Chatbot/NER/output/checkpoints",
    eval_strategy="epoch",          # Instead of evaluation_strategy
    save_strategy="epoch",
    logging_strategy="steps",       # or "epoch"
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    report_to="none"                # disable wandb/tensorboard
)

def compute_metrics(pred):
    logits, labels = pred
    predictions = logits.argmax(-1)

    # Convert predictions & labels back to strings
    true_preds = [
        [id2label[p] for p, l in zip(prediction, label_row) if l != -100]
        for prediction, label_row in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for p, l in zip(prediction, label_row) if l != -100]
        for prediction, label_row in zip(predictions, labels)
    ]

    return {
        "f1": f1_score(true_labels, true_preds),
        "accuracy": accuracy_score(true_labels, true_preds)
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# ------------------------------------------------
# 7. Save trained model
# ------------------------------------------------
trainer.save_model("/home/gflml/Chatbot/NER/output/saved_model/bert-ner-mod")
tokenizer.save_pretrained("/home/gflml/Chatbot/NER/output/saved_model/bert-ner-mod")

print("Model saved successfully!")

# ------------------------------------------------
# 8. Evaluate on test set
# ------------------------------------------------
predictions, labels, _ = trainer.predict(tokenized_ds["test"])
preds = predictions.argmax(-1)

true_preds = [
    [id2label[p] for p, l in zip(pred_row, label_row) if l != -100]
    for pred_row, label_row in zip(preds, labels)
]
true_labels = [
    [id2label[l] for p, l in zip(pred_row, label_row) if l != -100]
    for pred_row, label_row in zip(preds, labels)
]

print(classification_report(true_labels, true_preds))


# ------------------------------------------------
# 9. Inference Example
# ------------------------------------------------
def ner_predict(sentence):
    model.eval()
    tokens = sentence.split()

    # Send tokenizer output to model device
    device = next(model.parameters()).device
    enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**enc).logits

    preds = logits.argmax(-1).squeeze().tolist()
    word_ids = enc.word_ids()

    results = []
    prev_word = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == prev_word:
            continue
        results.append((tokens[word_id], id2label[preds[idx]]))
        prev_word = word_id

    return results



print("\nInference:")
print(ner_predict("Apple released a new iPhone in California"))









