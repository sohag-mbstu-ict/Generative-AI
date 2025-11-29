from dotenv import load_dotenv
from pathlib import Path
import os
from unsloth import FastLanguageModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline


# ✅ Load .env
env_path = Path("/home/gflmltpc/Projects/Gen_AI/.venv")
load_dotenv(dotenv_path=env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("GOOGLE API KEY:", GOOGLE_API_KEY)

base_model_dir = "/home/gflmltpc/Projects/Gen_AI/LLM_Fine_Tuning/gemma-7b-it-bnb-4bit"
trained_lora_model_path = "/home/gflmltpc/Projects/Gen_AI/LLM_Fine_Tuning/trained-gemma-7b-4bit-model" 
# ✅ Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_dir,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
# ✅ Prepare model for inference
FastLanguageModel.for_inference(model)
# ✅ Load the LoRA adapter
model.load_adapter(trained_lora_model_path)
# Wrap the model & tokenizer into a LangChain pipeline
hf_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    do_sample=True,
    device_map="auto"
)
langchain_llm = HuggingFacePipeline(pipeline=hf_pipe)
        
# ✅ Initialize LLM
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", GOOGLE_API_KEY=GOOGLE_API_KEY)

# ✅ Initialize Token Buffer Memory
memory = ConversationTokenBufferMemory(
    llm=langchain_llm,
    max_token_limit=500,  # adjust as needed
    return_messages=True
)

# ✅ Prompt template
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
The following is a conversation between a human and an AI assistant. The assistant is helpful, creative, and friendly.
Conversation history:
{history}

User: {input}
Assistant:"""
)

# ✅ Setup conversation chain
conversation = ConversationChain(
    llm=langchain_llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)

# ✅ CLI loop
while True:
    query = input("You: ")

    if query.lower() == "exit":
        print("End the session.")
        break

    elif query.lower() == "history":
        print("\n🧠 Token-limited History:")
        for msg in memory.chat_memory.messages:
            role = "You" if msg.type == "human" else "Bot"
            print(f"{role}: {msg.content}")
        print()
        continue

    # Normal query response
    response = conversation.invoke(query)
    print("Bot:", response['response'])
