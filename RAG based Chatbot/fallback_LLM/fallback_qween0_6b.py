import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenChatbot:
    def __init__(self, model_path="/home/gflml/Chatbot/pretrained_model/Qwen3-0.6B"):
        print("🔄 Loading tokenizer & model...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,          # Reduce memory
            device_map="auto",             # Automatically place on GPU
            attn_implementation="eager"    # Works on all systems
        )

        # Move explicitly to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        # Disable dropout
        self.model.eval()

        # Compile model for faster inference (PyTorch 2.x)
        self.model = torch.compile(self.model)

        print("✅ Model loaded and compiled successfully!")

    def build_prompt(self, user_text):
        """Convert user text to Qwen chat-style prompt."""
        messages = [{"role": "user", "content": user_text}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

    def generate(self, user_text, max_new_tokens=256):
        """Generate answer from the model."""
        prompt = self.build_prompt(user_text)
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        with torch.no_grad():  # No gradients → faster
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,     # Greedy decoding → faster
                temperature=0.0,
            )

        # Remove prompt tokens
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # Split thinking content if </think> token is present
        THINK_ID = 151668
        try:
            index = len(output_ids) - output_ids[::-1].index(THINK_ID)
        except ValueError:
            index = 0

        thinking = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
        answer = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        return thinking, answer

    def start_chat(self):
        print("\n💬 Qwen Chatbot is ready! Type 'exit' to quit.\n")
        while True:
            user_input = input("Ask your question: ")
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Exiting chat. Bye!")
                break

            thinking, answer = self.generate(user_input)
            print("\n🧠 Thinking:", thinking)
            print("🤖 Answer:", answer)
            print("-" * 60)


if __name__ == "__main__":
    bot = QwenChatbot()
    bot.start_chat()
