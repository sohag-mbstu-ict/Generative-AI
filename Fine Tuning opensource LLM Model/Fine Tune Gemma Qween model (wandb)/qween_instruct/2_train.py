import json
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import (
    get_chat_template,
    standardize_data_formats,
    train_on_responses_only,
)
from trl import SFTTrainer, SFTConfig


class Qwen3InstructFineTuner:
    def __init__(
        self,
        json_path,
        base_model_path,
        save_path,
        max_seq_length=2048,
        seed=3407,
    ):
        self.json_path = json_path
        self.base_model_path = base_model_path
        self.save_path = save_path
        self.max_seq_length = max_seq_length
        self.seed = seed

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
            r=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.seed,
            use_rslora=False,
            loftq_config=None,
        )

        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="qwen3-instruct",
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
    # 🔹 SETUP TRAINER
    # --------------------------------------------------
    def setup_trainer(self):
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            eval_dataset=None,
            args=SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=600,
                learning_rate=2e-4,
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.001,
                lr_scheduler_type="linear",
                seed=self.seed,
                report_to="none",
            ),
        )

        self.trainer = train_on_responses_only(
            self.trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
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
    base_model_path="/workspace/Gen_AI/Qwen3-4B/Base-Qwen3-4B-Instruct-2507",
    save_path="/workspace/fine_tune/qween_instruct/struc_saved_model",
)

trainer.load_json_dataset()
trainer.convert_dataset_format()
trainer.load_model_and_tokenizer()
trainer.prepare_training_text()
trainer.setup_trainer()
trainer.train()
trainer.save()
