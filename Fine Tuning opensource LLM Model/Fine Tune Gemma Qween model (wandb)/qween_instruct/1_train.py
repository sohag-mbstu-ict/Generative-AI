import json
from datasets import Dataset
from unsloth.chat_templates import (
    get_chat_template,
    standardize_data_formats,
)
from unsloth import FastLanguageModel

json_path = "/workspace/fine_tune/SFT_dataset/sft.json"

with open(json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

dataset1 = Dataset.from_list(raw_data)

print("111111111111111 : ",dataset1)
def chatml_to_sharegpt_batched(examples):
    all_convos = []

    for messages in examples["messages"]:
        conversations = []
        system_prompt = ""

        for msg in messages:
            # if msg["role"] == "system":
            #     system_prompt = msg["content"]

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

dataset22 = dataset1.map(
    chatml_to_sharegpt_batched,
    batched=True,
    batch_size=1000,
    remove_columns=dataset1.column_names,
    num_proc=8
)


print("222222222222222222 : ",dataset22)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/workspace/Gen_AI/Qwen3-4B/Base-Qwen3-4B-Instruct-2507",
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen3-instruct",
)
from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset22)
def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
   return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 600,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use TrackIO/WandB etc
    ),
)

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

trainer_stats = trainer.train()
model.save_pretrained("/workspace/fine_tune/qween_instruct/saved_model")  # Local saving
tokenizer.save_pretrained("/workspace/fine_tune/qween_instruct/saved_model")
ourcesa=4

