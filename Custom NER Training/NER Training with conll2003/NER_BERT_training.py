import os
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from seqeval.metrics import classification_report, f1_score, accuracy_score

class BertNERTrainer:
    """
    A complete class for:
    ✅ Loading CoNLL-format data
    ✅ Tokenizing and label alignment
    ✅ Training BERT for NER
    ✅ Evaluation
    ✅ Inference
    """

    def __init__(self, model_name="bert-base-cased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.label2id = {}
        self.id2label = {}

    # -------------------------
    # Load CoNLL dataset
    # -------------------------
    def read_conll(self, path):
        tokens, labels = [], []
        sent_tokens, sent_labels = [], []

        with open(path, encoding='utf-8') as f:
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

    # -------------------------
    # Prepare HF Dataset
    # -------------------------
    def prepare_dataset(self, train_path, val_path, test_path):
        train_tokens, train_labels = self.read_conll(train_path)
        val_tokens, val_labels   = self.read_conll(val_path)
        test_tokens, test_labels = self.read_conll(test_path)

        # Label mappings
        unique_labels = sorted(list({l for doc in train_labels for l in doc}))
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        train_ds = Dataset.from_dict({"tokens": train_tokens, "ner_tags": train_labels})
        val_ds   = Dataset.from_dict({"tokens": val_tokens, "ner_tags": val_labels})
        test_ds  = Dataset.from_dict({"tokens": test_tokens, "ner_tags": test_labels})

        self.dataset = DatasetDict({
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds
        })

    # -------------------------
    # Tokenize + label align
    # -------------------------
    def tokenize_and_align(self, example):
        tokenized = self.tokenizer(
            example["tokens"],
            truncation=True,
            is_split_into_words=True
        )

        word_ids = tokenized.word_ids()
        prev = None
        aligned = []

        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)
            elif word_id != prev:
                aligned.append(self.label2id[example["ner_tags"][word_id]])
            else:
                aligned.append(-100)
            prev = word_id

        tokenized["labels"] = aligned
        return tokenized

    # -------------------------
    # Train model
    # -------------------------
    def train_model(self, output_dir, epochs=3, lr=3e-5):
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )

        tokenized_ds = self.dataset.map(self.tokenize_and_align)

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        training_args = TrainingArguments(
            output_dir=None,
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=lr,
            per_device_train_batch_size=8,
            num_train_epochs=epochs,
            weight_decay=0.01,
            report_to="none"
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        self.trainer.train()

        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    # -------------------------
    # Metrics
    # -------------------------
    def compute_metrics(self, pred):
        logits, labels = pred
        predictions = logits.argmax(-1)

        true_preds = [
            [self.id2label[p] for p, l in zip(pred, lab) if l != -100]
            for pred, lab in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for p, l in zip(pred, lab) if l != -100]
            for pred, lab in zip(predictions, labels)
        ]

        return {
            "f1": f1_score(true_labels, true_preds),
            "accuracy": accuracy_score(true_labels, true_preds)
        }

    # -------------------------
    # Predict single sentence
    # -------------------------
    def predict(self, sentence):
        self.model.eval()
        tokens = sentence.split()

        device = next(self.model.parameters()).device  # <-- detect GPU/CPU

        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt"
        ).to(device)   # ✅ move to model device

        with torch.no_grad():
            logits = self.model(**enc).logits

        preds = logits.argmax(-1).squeeze().tolist()
        word_ids = enc.word_ids()

        results = []
        prev = None

        for idx, wid in enumerate(word_ids):
            if wid is None or wid == prev:
                continue
            results.append((tokens[wid], self.id2label[preds[idx]]))
            prev = wid

        return results


trainer = BertNERTrainer()

trainer.prepare_dataset(
    train_path="/home/gflml/Chatbot/NER/data/conllpp_train.txt",
    val_path="/home/gflml/Chatbot/NER/data/conllpp_valid.txt",
    test_path="/home/gflml/Chatbot/NER/data/conllpp_test.txt"
)

trainer.train_model(
    output_dir="/home/gflml/Chatbot/NER/trained_model/"
)

# Inference

print(trainer.predict("Apple released a new iPhone in California"))



