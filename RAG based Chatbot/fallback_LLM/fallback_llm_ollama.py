"""
Fallback LLM Module (Ollama)
-----------------------------
Ultra-fast fallback answers using Ollama LLM.
Ideal for instant responses when RAG fails.
"""

from ollama import Ollama

class FallbackLLM:
    """
    Fallback LLM using Ollama.
    """

    def __init__(self, model_name: str = "gemma3:270m"):
        """
        Initialize Ollama client and load the model.
        """
        self.client = Ollama()
        self.model_name = model_name

    # ----------------------------------------------------
    # Generate fallback answer
    # ----------------------------------------------------
    def generate(self, query: str) -> str:
        """
        Generate a fast fallback answer using Ollama model.
        Only uses the query text.
        """
        prompt = f"""
You are an agricultural expert.
Answer the following question in 3–6 concise sentences:

Question: {query}

Do NOT hallucinate chemical names or unsafe advice.
Keep answer practical and simple.
"""

        try:
            response = self.client.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
            return response.get("content", "Sorry, I couldn't generate an answer.")
        except Exception:
            return "Sorry, I couldn't generate an answer at the moment."

    # ----------------------------------------------------
    # Debug helper
    # ----------------------------------------------------
    def debug_info(self):
        return f"Ollama Fallback LLM loaded: {self.model_name}"


# -------------------------------
# Manual test
# -------------------------------
if __name__ == "__main__":
    llm = FallbackLLM()
    while True:
        user_input = input("\nAsk your question: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print(llm.generate(user_input))
