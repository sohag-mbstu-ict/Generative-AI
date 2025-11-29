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
trained_lora_model_path = "/home/gflmltpc/Projects/Gen_AI/LLM_Fine_Tuning/ChatML-Format/trained_gemma_7b_26_july_25"

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
    # do_sample=True → Enables random sampling for diverse outputs.
    do_sample=True,
    # temperature=0.7 → Controls creativity (lower = deterministic, higher = creative).
    temperature=0.7,
    device_map="auto"
)

# ✅ Convert to LangChain LLM
langchain_llm = HuggingFacePipeline(pipeline=hf_pipe)

# ✅ Memory
memory = ConversationSummaryBufferMemory(
    llm=langchain_llm,
    max_token_limit=1024,
    return_messages=True
)

# ✅ ChatML-style prompt builder
# Builds prompt with history + user_input.
def build_chatml_prompt(history, user_input):
    system_message = "<|system|>\nYou are an agricultural expert assistant.\n"
    conversation_text = system_message

    for msg in history:
        if msg.type == "human":
            conversation_text += f"<|user|>\n{msg.content}\n"
        else:
            conversation_text += f"<|assistant|>\n{msg.content}\n"
    # Add new user query
    conversation_text += f"<|user|>\n{user_input}\n<|assistant|>\n"
    return conversation_text

# ✅ Extract clean response (stop at first new line after answer)
def extract_clean_response(raw_output):
    return raw_output.split("<|assistant|>")[-1].strip().split("\n")[0]

# ✅ Chat loop
while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("✅ Session ended.")
        break

    elif user_input.lower() == "history":
        print("\n🧠 Current Conversation History:")
        for msg in memory.chat_memory.messages:
            role = "You" if msg.type == "human" else "Bot"
            print(f"{role}: {msg.content}")
        print()
        continue

    # ✅ Build ChatML prompt
    history = memory.chat_memory.messages
    # Builds prompt with history + user_input.
    final_prompt = build_chatml_prompt(history, user_input)

    # ✅ Generate raw response
    raw_result = langchain_llm(final_prompt)
    response = extract_clean_response(raw_result)

    # ✅ Save conversation
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(response)

    print("Bot:", response)
