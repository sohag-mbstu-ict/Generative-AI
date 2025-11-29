import torch
print(torch.cuda.get_device_capability())  # e.g., (8, 0)

# if False:
lora_model = "/home/gflmltpc/Projects/Gen_AI/LLM_Fine_Tuning/gemma-7b-it-bnb-4bit/"

max_seq_length = 2048 # Choose any! Llama 3 is up to 8k
dtype = None
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = lora_model, # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)



