from dotenv import load_dotenv
from pathlib import Path
import os
from unsloth import FastLanguageModel
from langchain.memory import ConversationSummaryBufferMemory
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# ✅ Load environment
env_path = Path("/home/gflmltpc/Projects/Gen_AI/.venv")
load_dotenv(dotenv_path=env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ Model paths
base_model_dir = "/home/gflmltpc/Projects/Gen_AI/base-gemma-7b-it-bnb-4bit"
trained_lora_model_path = "/home/gflmltpc/Projects/Gen_AI/LLM_Fine_Tuning/ShareGPT-model/trained_gemma_7b_26_july_25"

# ✅ Load model & LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_dir,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)
FastLanguageModel.for_inference(model)
model.load_adapter(trained_lora_model_path)

# ✅ HuggingFace pipeline
hf_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    device_map="auto"
)

# ✅ LangChain wrapper
langchain_llm = HuggingFacePipeline(pipeline=hf_pipe)

# ✅ Memory
memory = ConversationSummaryBufferMemory(
    llm=langchain_llm,
    max_token_limit=1024,
    return_messages=True
)

# ✅ Static system prompt
system_prompt = {"from": "system", "value": "You are an agricultural expert assistant."}

# ✅ ShareGPT-style prompt builder
def build_sharegpt_prompt(history, instruction):
    prompt = "<|im_start|>system\n" + system_prompt["value"] + "<|im_end|>\n"
    for msg in history:
        if msg.type == "human":
            prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
        elif msg.type == "ai":
            prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
    # Add new user message
    prompt += f"<|im_start|>user\n{instruction}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"  # Expecting model to fill this
    return prompt

# ✅ Extract response
def extract_clean_response(raw_output):
    if "<|im_start|>assistant" in raw_output:
        return raw_output.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    return raw_output.strip().split("\n")[0]

# ✅ Chat loop
while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("✅ Session ended.")
        break

    elif user_input.lower() == "history":
        print("\n🧠 Current Conversation History:")
        print(f"System: {system_prompt['value']}")
        for msg in memory.chat_memory.messages:
            role = "You" if msg.type == "human" else "Bot"
            print(f"{role}: {msg.content}")
        print()
        continue

    # ✅ Prompt
    history = memory.chat_memory.messages
    final_prompt = build_sharegpt_prompt(history, user_input)

    # ✅ Inference
    raw_result = langchain_llm(final_prompt)
    response = extract_clean_response(raw_result)

    # ✅ Save memory
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(response)

    print("Bot:", response)
