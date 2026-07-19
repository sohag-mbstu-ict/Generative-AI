import json
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
# Add this at the top of the file, before any use of register_pytree_node
try:
    from torch.utils._pytree import register_pytree_node
except ImportError:
    from torch.utils._pytree import _register_pytree_node as register_pytree_node

print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())  # Should be >= 1
print(torch.cuda.current_device())

class QwenFineTuner:
    def __init__(self, base_model_path, save_dir, max_seq_length=2048, seed=3407):
        self.base_model_path = base_model_path
        self.save_dir = save_dir
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model_and_tokenizer(self):
        print("🔹 Loading base model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_path,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
        )

        print("🔹 Applying LoRA configuration...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.seed,
            use_rslora=False,
            loftq_config=None,
        )

        print("🔹 Applying chat template...")
        tokenizer = get_chat_template(tokenizer, chat_template="qwen3-thinking")

        self.model, self.tokenizer = model, tokenizer
        return self.model, self.tokenizer

    def load_and_prepare_dataset(self, json_file_path):
        """Load JSON dataset with system+user+assistant messages and convert to HF Dataset."""
        print("🔹 Loading JSON dataset...")
        with open(json_file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        conversations = []
        for entry in raw_data:
            msgs = entry.get("messages", [])
            if len(msgs) < 3:
                continue  # Skip if not system+user+assistant

            # Start conversation with system message
            system_msg = msgs[0]["content"]
            convo_history = []

            # Iterate through turns (support multi-turn)
            for i in range(1, len(msgs), 2):
                user_msg = msgs[i]["content"]
                assistant_msg = msgs[i + 1]["content"] if i + 1 < len(msgs) else ""
                
                # Prepend system + previous turns to user prompt
                user_prompt = system_msg + "\n" + "\n".join(
                    [f"User: {u}\nAssistant: {a}" for u, a in convo_history]
                ) + f"\nUser: {user_msg}"

                # Store for training
                conversations.append([
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_msg}
                ])

                # Update conversation history
                convo_history.append((user_msg, assistant_msg))

        hf_dataset = Dataset.from_dict({"conversations": conversations})

        # Apply tokenizer + chat template
        def formatting_prompts_func(examples):
            convos = examples["conversations"]
            texts = [
                self.tokenizer.apply_chat_template(
                    convo, tokenize=False, add_generation_prompt=False
                )
                for convo in convos
            ]
            return {"text": texts}

        hf_dataset = hf_dataset.map(formatting_prompts_func, batched=True)
        return hf_dataset

    def setup_trainer(self, dataset, max_steps=70, batch_size=12, grad_accum=4, lr=2e-4):
        print("🔹 Setting up trainer...")
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            eval_dataset=None,
            args=SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                warmup_steps=5,
                max_steps=max_steps,
                learning_rate=lr,
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=self.seed,
                report_to="none",
            ),
        )

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

        self.trainer = trainer
        return self.trainer

    def train(self):
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call setup_trainer() first.")
        print("🔹 Starting training...")
        trainer_stats = self.trainer.train()
        print("✅ Training finished!")
        return trainer_stats

    def save(self):
        print(f"🔹 Saving model and tokenizer to {self.save_dir}...")
        self.model.save_pretrained(f"{self.save_dir}/lora_model")
        self.tokenizer.save_pretrained(f"{self.save_dir}/lora_model")
        print("✅ Model and tokenizer saved!")


base_model = "/workspace/Gen_AI/Qwen3-4B/Base-Qwen3-4B-Instruct-2507"
save_dir = "/workspace/fine_tune/qween_model/trained_model"
json_file = "/workspace/fine_tune/SFT_dataset/sft.json"

qwen_trainer = QwenFineTuner(base_model, save_dir)
qwen_trainer.load_model_and_tokenizer()
dataset = qwen_trainer.load_and_prepare_dataset(json_file)
qwen_trainer.setup_trainer(dataset, max_steps=300)
qwen_trainer.train()
qwen_trainer.save()


