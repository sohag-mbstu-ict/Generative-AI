import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

class GemmaShareGPTFineTuner:
    """
    A class for fine-tuning the Gemma model using Unsloth with LoRA adapters and ShareGPT-style dataset.
    """

    def __init__(self, base_model_path, dataset_path, output_dir, max_seq_length=2048, load_in_4bit=True):
        """
        Initialize the fine-tuner with model and dataset configurations.

        Args:
            base_model_path (str): Path to the pre-trained base model.
            dataset_path (str): Path to the training dataset in ShareGPT format.
            output_dir (str): Directory to save the fine-tuned model.
            max_seq_length (int): Maximum sequence length for the model.
            load_in_4bit (bool): Whether to load the model in 4-bit precision for memory efficiency.
        """
        self.base_model_path = base_model_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.EOS_TOKEN = None

    def load_model(self):
        """Load the Gemma base model with optional 4-bit quantization."""
        print("Torch Device Capability:", torch.cuda.get_device_capability())
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_path,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )
        self.EOS_TOKEN = self.tokenizer.eos_token
        print("Model and tokenizer loaded successfully.")

    def apply_lora(self, r=16, lora_alpha=16, lora_dropout=0):
        """
        Apply LoRA (Low-Rank Adaptation) adapters to the model for parameter-efficient fine-tuning.

        Args:
            r (int): LoRA rank.
            lora_alpha (int): Scaling factor for LoRA.
            lora_dropout (float): Dropout for LoRA layers.
        """
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        print("LoRA adapters applied successfully.")

    def prepare_dataset(self):
        """Load and format the ShareGPT-style dataset for fine-tuning."""
        def formatting_prompts_func(examples):
            conversations_list = examples["conversations"]
            texts = []
            for conversation in conversations_list:
                # Convert ShareGPT conversation into a single string
                conversation_text = ""
                for msg in conversation:
                    role = msg["from"]
                    content = msg["value"]
                    if role == "system":
                        conversation_text += f"<|system|>\n{content}\n"
                    elif role == "human":
                        conversation_text += f"<|user|>\n{content}\n"
                    elif role == "gpt":
                        conversation_text += f"<|assistant|>\n{content}\n"
                conversation_text += self.EOS_TOKEN
                texts.append(conversation_text)
            return {"text": texts}

        dataset = load_dataset("json", data_files=self.dataset_path, split="train")
        print("Original ShareGPT Dataset Loaded:", dataset)
        dataset = dataset.map(formatting_prompts_func, batched=True)
        return dataset

    def configure_trainer(self, dataset, epochs=14, batch_size=2, learning_rate=2e-4):
        """
        Configure the SFTTrainer for supervised fine-tuning.

        Args:
            dataset: The processed dataset for training.
            epochs (int): Number of training epochs.
            batch_size (int): Per device batch size.
            learning_rate (float): Learning rate for training.
        """
        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # disables wandb
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )
        print("Trainer configured successfully.")

    def train(self):
        """Start the fine-tuning process."""
        if self.trainer is None:
            raise ValueError("Trainer not configured. Call configure_trainer() first.")
        print("Starting training...")
        stats = self.trainer.train()
        print("Training completed.")
        return stats

    def save_model(self):
        """Save the fine-tuned model to the specified directory."""
        self.model.save_pretrained(self.output_dir)
        print(f"Model saved at {self.output_dir}")


# ================== USAGE ==================
if __name__ == "__main__":
    base_model_path = "/home/gflmltpc/Projects/Gen_AI/base-gemma-7b-it-bnb-4bit/"
    dataset_path = "/home/gflmltpc/Projects/Gen_AI/Dataset/ShareGPT/ShareGPT.json"  # Your ShareGPT dataset
    output_dir = "/home/gflmltpc/Projects/Gen_AI/LLM_Fine_Tuning/ShareGPT-model/trained_gemma_7b_26_july_25"

    fine_tuner = GemmaShareGPTFineTuner(base_model_path, dataset_path, output_dir)
    fine_tuner.load_model()
    fine_tuner.apply_lora(r=16)
    dataset = fine_tuner.prepare_dataset()
    fine_tuner.configure_trainer(dataset, epochs=14)
    fine_tuner.train()
    fine_tuner.save_model()
