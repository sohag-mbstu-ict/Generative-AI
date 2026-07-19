import os
os.environ["UNSLOTH_DISABLE_STATS"] = "1"

import json
import numpy as np
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import (
    get_chat_template,
    standardize_data_formats,
    train_on_responses_only,
)
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback
from eval_graph import save_loss_graph_opencv, save_accuracy_graph_opencv
from perplexity_loss import save_perplexity_graph_opencv
 
import evaluate
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits: (B, T, V)
    # labels: (B, T)
    predictions = np.argmax(logits, axis=-1)
    # Shift for causal LM
    predictions = predictions[:, :-1].reshape(-1)
    labels = labels[:, 1:].reshape(-1)
    # Remove padding tokens (-100)
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    return {
        "accuracy": (predictions == labels).mean()}



class Qwen3InstructFineTuner:
    def __init__(
        self,
        json_path,
        base_model_path,
        save_path,
        max_seq_length=2048,
        seed=3407,
        early_stopping_patience=7,   # 🔹 ADDED
    ):
        self.json_path = json_path
        self.base_model_path = base_model_path
        self.save_path = save_path
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.early_stopping_patience = early_stopping_patience

        self.dataset = None
        self.model = None
        self.tokenizer = None
        self.trainer = None

    # --------------------------------------------------
    # 🔹 LOAD RAW JSON → HF DATASET
    # --------------------------------------------------
    def load_json_dataset(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.dataset = Dataset.from_list(raw_data)
        print("Loaded dataset:", self.dataset)
        return self.dataset

    # --------------------------------------------------
    # 🔹 CHATML → SHAREGPT FORMAT
    # --------------------------------------------------
    @staticmethod
    def chatml_to_sharegpt_batched(examples):
        all_convos = []

        for messages in examples["messages"]:
            conversations = []
            system_prompt = ""

            for msg in messages:
                if msg["role"] == "user":
                    text = msg["content"]
                    if system_prompt:
                        text = system_prompt + "\n\n" + text
                        system_prompt = ""
                    conversations.append({"from": "human", "value": text})

                elif msg["role"] == "assistant":
                    conversations.append({"from": "gpt", "value": msg["content"]})

            all_convos.append(conversations)

        return {"conversations": all_convos}

    def convert_dataset_format(self):
        self.dataset = self.dataset.map(
            self.chatml_to_sharegpt_batched,
            batched=True,
            batch_size=1000,
            remove_columns=self.dataset.column_names,
            num_proc=8,
        )
        print("Converted dataset:", self.dataset)
        return self.dataset

    # --------------------------------------------------
    # 🔹 LOAD BASE MODEL + APPLY LORA
    # --------------------------------------------------
    def load_model_and_tokenizer(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_path,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=8, # Larger = higher accuracy, but might overfit
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=8, # Recommended alpha == r at least
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.seed,
            use_rslora=False,
            loftq_config=None,
        )

        self.tokenizer = get_chat_template(
        self.tokenizer,
        chat_template="gemma-3",
    )

        return self.model, self.tokenizer

    # --------------------------------------------------
    # 🔹 STANDARDIZE + APPLY CHAT TEMPLATE
    # --------------------------------------------------
    def prepare_training_text(self):
        self.dataset = standardize_data_formats(self.dataset)

        def formatting_prompts_func(examples):
            convos = examples["conversations"]
            texts = [
                self.tokenizer.apply_chat_template(
                    convo,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for convo in convos
            ]
            return {"text": texts}

        self.dataset = self.dataset.map(
            formatting_prompts_func,
            batched=True,
        )

        return self.dataset

    # --------------------------------------------------
    # 🔹 SETUP TRAINER (WITH EARLY STOPPING)
    # --------------------------------------------------
    def setup_trainer(self):
        # 🔹 Train / Eval split
        if len(self.dataset) > 1:
            split = self.dataset.train_test_split(
                test_size=0.1,
                seed=self.seed,
            )
            train_dataset = split["train"]
            eval_dataset = split["test"]
        else:
            raise ValueError("Dataset too small for evaluation split.")

        self.trainer = SFTTrainer(
        model=self.model,
        tokenizer=self.tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=300,
            learning_rate=2e-4,
            logging_steps=1,
            output_dir=self.save_path,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=self.seed,
            report_to="none",

            # 🔹 FIXED
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            save_steps=500,

            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        ),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=self.early_stopping_patience,
            )])

        # 🔥 GEMMA-CORRECT MASKING
        self.trainer = train_on_responses_only(
            self.trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )


        return self.trainer

    # --------------------------------------------------
    # 🔹 TRAIN
    # --------------------------------------------------
    def train(self):
        trainer_stats = self.trainer.train()
        return trainer_stats

    # --------------------------------------------------
    # 🔹 SAVE MODEL
    # --------------------------------------------------
    def save(self):
        self.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
        print(f"Model saved to {self.save_path}")


trainer = Qwen3InstructFineTuner(
    json_path="/workspace/fine_tune/SFT_dataset/sft.json",
    base_model_path="/workspace/Gen_AI/Gemma/gemma-3-1b-it",
    save_path="/workspace/fine_tune/gemma/gemma-3-1b-it/earlyStopping_saved_model",
    early_stopping_patience=7,
)

trainer.load_json_dataset()
trainer.convert_dataset_format()
trainer.load_model_and_tokenizer()
trainer.prepare_training_text()
trainer.setup_trainer()
trainer.train()
trainer.save()
save_loss_graph_opencv(trainer.trainer.state.log_history, save_path="/workspace/fine_tune/gemma/gemma-3-1b-it/eval_matrix/train_val_loss.png")
# After training
save_accuracy_graph_opencv(trainer.trainer.state.log_history, save_path="/workspace/fine_tune/gemma/gemma-3-1b-it/eval_matrix/train_val_accuracy.png")

save_perplexity_graph_opencv(
    trainer.trainer.state.log_history,
    save_path="/workspace/fine_tune/gemma/gemma-3-1b-it/eval_matrix/train_val_perplexity.png",
)

