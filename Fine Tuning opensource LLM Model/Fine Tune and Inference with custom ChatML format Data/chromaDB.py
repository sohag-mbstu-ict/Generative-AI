from dotenv import load_dotenv
from pathlib import Path
import os
from unsloth import FastLanguageModel
from langchain.memory import ConversationSummaryBufferMemory
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

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
    do_sample=True,
    temperature=0.1,
    device_map="auto"
)

# ✅ LangChain LLM wrapper
langchain_llm = HuggingFacePipeline(pipeline=hf_pipe)

# ✅ Memory
memory = ConversationSummaryBufferMemory(
    llm=langchain_llm,
    max_token_limit=1024,
    return_messages=True
)

# ✅ Embeddings + Chroma
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_dir = "/home/gflmltpc/Projects/Gen_AI/LLM_Fine_Tuning/ChatML-Format/chroma_dir"
vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embedding_model)

# ✅ ChatML-style prompt builder
def build_chatml_prompt(history, user_input):
    system_message = "<|system|>\nYou are an agricultural expert assistant.\n"
    conversation_text = system_message

    for msg in history:
        if msg.type == "human":
            conversation_text += f"<|user|>\n{msg.content}\n"
        else:
            conversation_text += f"<|assistant|>\n{msg.content}\n"
    conversation_text += f"<|user|>\n{user_input}\n<|assistant|>\n"
    return conversation_text

# ✅ Clean response
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

    elif user_input.lower().startswith("similar:"):
        query = user_input.split("similar:", 1)[-1].strip()
        results = vectorstore.similarity_search(query, k=3)
        print("\n🔍 Most similar saved queries:")
        for i, doc in enumerate(results):
            print(f"{i+1}. {doc.page_content}")
        continue

    # ✅ Build ChatML prompt
    history = memory.chat_memory.messages
    final_prompt = build_chatml_prompt(history, user_input)

    # ✅ Generate & extract response
    raw_result = langchain_llm(final_prompt)
    response = extract_clean_response(raw_result)

    # ✅ Save to memory
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(response)

    # ✅ Save to vectorstore
    combined_doc = f"User: {user_input}\nBot: {response}"
    vectorstore.add_documents([Document(page_content=combined_doc)])

    print("Bot:", response)
    print("-----------------------------------------------------------------------------")
    # 🔍 Debug: Print stored tokens and token length
    conversation_history = memory.buffer  # Summarized history text used internally
    print("conversation_history /; ",conversation_history)
    # tokenized = tokenizer(conversation_history, return_tensors="pt")
    # token_ids = tokenized["input_ids"][0]

    # print("\n🧾 Tokenized Conversation History:")
    # print("Tokens:", token_ids.tolist())
    # print("Token Count:", len(token_ids))
    # decoded_tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
    # print("\n🧾 Decoded Tokens:", decoded_tokens)
    
    # print(f"\n[DEBUG] Total token count in memory buffer: {len(token_ids)}")
    # if len(token_ids) > 124:
    #     print("⚠️ Warning: Token count exceeds limit. Some history may be summarized or dropped.")
    print("--------------------------------------------------------------------------------")

