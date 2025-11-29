from unsloth import FastLanguageModel
import torch
from langchain.memory import ConversationTokenBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

class LoRAInferencer:
    def __init__(self, base_model_path, trained_lora_model_path, max_seq_length=2048, load_in_4bit=True, dtype=None):
        self.base_model_path = base_model_path
        self.trained_lora_model_path = trained_lora_model_path
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.dtype = dtype

        # ✅ Load base model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_path,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )

        # ✅ Prepare model for inference
        FastLanguageModel.for_inference(self.model)

        # ✅ Load the LoRA adapter
        self.model.load_adapter(self.trained_lora_model_path)
        # Wrap the model & tokenizer into a LangChain pipeline
        hf_pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            do_sample=True,
            device_map="auto"
        )
        langchain_llm = HuggingFacePipeline(pipeline=hf_pipe)

        # ✅ Initialize memory (max_token_limit can be adjusted)
        self.memory = ConversationTokenBufferMemory(
            llm=langchain_llm,  # Required to count tokens
            max_token_limit=3448,  # Adjust based on your context length
            return_messages=True,
            memory_key="chat_history"
        )

    def count_tokens(self, text: str) -> int:
        """Estimate token count for memory buffer."""
        return len(self.tokenizer.encode(text))

    def generate_response(self, instruction: str, input_text: str = "", max_new_tokens: int = 256) -> str:
        # ✅ Get previous conversation history
        history = self.memory.load_memory_variables({})["chat_history"]
        history_text = "\n".join([f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Bot: {msg.content}" for msg in history])

        # ✅ Construct prompt with history
        alpaca_prompt = """Below is a conversation history followed by the latest instruction. Write a concise response that continues the conversation..
                            ### Conversation History:
                            {}
                            ### Instruction:
                            {}
                            ### Input:
                            {}
                            ### Response:
                            {}"""

        prompt = alpaca_prompt.format(history_text, instruction, input_text, "")
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )

        # ✅ Decode and clean the output
        decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
        response = decoded_output.split("### Response:")[-1].split("<eos>")[0].strip()

        # ✅ Save to memory
        self.memory.save_context({"input": instruction}, {"output": response})

        return response

# ✅ Usage Example
if __name__ == "__main__":
    base_model_dir = "/home/gflmltpc/Projects/Gen_AI/LLM_Fine_Tuning/gemma-7b-it-bnb-4bit"
    trained_lora_model_path = "/home/gflmltpc/Projects/Gen_AI/LLM_Fine_Tuning/trained-gemma-7b-4bit-model"

    inferencer = LoRAInferencer(base_model_dir, trained_lora_model_path)
    while True:
        user_query = input("you : ")
        if user_query.lower() == 'exit':
            print("Chatbot session ended.")
            break
        response = inferencer.generate_response(user_query)
        print("\n🧪 Response Only:\n", response)