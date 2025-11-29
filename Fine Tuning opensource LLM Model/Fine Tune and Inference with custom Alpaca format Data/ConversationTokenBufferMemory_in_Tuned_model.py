from dotenv import load_dotenv
from pathlib import Path
import os
from unsloth import FastLanguageModel
from langchain.memory import ConversationTokenBufferMemory
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# ✅ Load environment
env_path = Path("/home/gflmltpc/Projects/Gen_AI/.venv")
load_dotenv(dotenv_path=env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ Model paths
base_model_dir = "/home/gflmltpc/Projects/Gen_AI/base-gemma-7b-it-bnb-4bit"
# LoRA adapter adds fine-tuned task-specific knowledge without retraining the full model.
trained_lora_model_path = "/home/gflmltpc/Projects/Gen_AI/LLM_Fine_Tuning/Alpaca-Format/trained_gemma_7b_26_july_25"

# ✅ Load model & LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_dir,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)
FastLanguageModel.for_inference(model)
# LoRA adapter adds fine-tuned task-specific knowledge without retraining the full model.
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
memory = ConversationTokenBufferMemory(
    llm=langchain_llm,
    max_token_limit=1024,
    return_messages=True
)

# ✅ Alpaca-style prompt builder
# # Builds prompt with history + latest query.
def build_prompt(history, instruction):
    history_text = ""
    for msg in history:
        role = "Human" if msg.type == "human" else "AI"
        history_text += f"{role}: {msg.content}\n"
    
    return f"""Below is a conversation history and a new instruction. Write a helpful response.

### Conversation History:
{history_text}

### Instruction:
{instruction}

### Response:"""

# ✅ Extract clean response
def extract_clean_response(raw_output):
    if "### Response:" in raw_output:
        raw_output = raw_output.split("### Response:")[-1].strip()
    return raw_output.split("\n")[0].strip()  # take first clean sentence

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

    # ✅ Build prompt with memory
    history = memory.chat_memory.messages
    # Builds prompt with history + latest query.
    final_prompt = build_prompt(history, user_input)

    # ✅ Generate raw response
    raw_result = langchain_llm(final_prompt)
    response = extract_clean_response(raw_result)

    # ✅ Save conversation
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(response)

    print("Bot:", response)
