"""
Fallback LLM Module (Qwen3-4B Optimized with 4-bit)
---------------------------------------------------
Loads the model ONCE using Streamlit caching.
Uses 4-bit quantization for 8GB GPU to reduce memory and speed up inference.
Generates fallback answers using only the query.
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class FallbackLLM:

    def __init__(self, model_name: str = "/home/gflml/Chatbot/pretrained_model/Qwen3-4B-Instruct-2507"):
        self.model_name = model_name
        self.model, self.tokenizer = self._load_cached_model(model_name)

    # ----------------------------------------------------
    # Load Qwen Model ONCE with 4-bit quantization
    # ----------------------------------------------------
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def _load_cached_model(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit=True,          # 4-bit quantization
            torch_dtype=torch.float16,
            trust_remote_code=True      # Required for some Qwen models
        )
        return model, tokenizer

    # ----------------------------------------------------
    # Generate fallback answer (ONLY using query)
    # ----------------------------------------------------
    def generate(self, query: str, max_tokens: int = 150) -> str:
        """
        Generate a safe, practical fallback answer using Qwen.
        """
        prompt = f"""
You are an agricultural expert.
Provide a concise, practical, accurate answer in 3-6 sentences.

Question: {query}

Do NOT hallucinate chemical names. Avoid unsafe advice.
Keep the answer simple and actionable.
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9,
            do_sample=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # ----------------------------------------------------
    # Optional debug helper
    # ----------------------------------------------------
    def debug_info(self):
        return f"Qwen Fallback LLM Loaded on GPU: {torch.cuda.get_device_name(0)}, Model: {self.model_name}, 4-bit quantized"


# -------------------------------
# Manual test
# -------------------------------
if __name__ == "__main__":
    llm = FallbackLLM()
    print(llm.debug_info())

    while True:
        user_input = input("\nAsk your question : ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print(llm.generate(user_input))


