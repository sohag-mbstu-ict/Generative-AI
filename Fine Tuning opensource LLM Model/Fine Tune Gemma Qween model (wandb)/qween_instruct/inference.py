import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

class QwenInference:
    def __init__(self, model_path, chat_template="qwen3-instruct", device="cuda"):
        self.model_path = model_path
        self.chat_template = chat_template
        self.device = device

        self.model = None
        self.tokenizer = None

    # ---------------- Load Model & Tokenizer ----------------
    def load_model_and_tokenizer(self):
        print("🔹 Loading model and tokenizer...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=2048,
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False
        )

        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template=self.chat_template
        )

        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded on", self.device)

    # ---------------- Generate Response ----------------
    def generate(self, messages, max_new_tokens=500, temperature=0.5, top_p=0.8, top_k=20):
        """
        messages: list of dicts [{'role': 'user', 'content': 'Your question'}, ...]
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # Must be True for generation
        )

        input_ids = self.tokenizer(text, return_tensors="pt").to(self.device)

        streamer = TextStreamer(self.tokenizer, skip_prompt=True)

        with torch.inference_mode():
            _ = self.model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                streamer=streamer
            )

# ---------------- Usage (ONLY CHANGE IS HERE) ----------------
if __name__ == "__main__":
    # model_path = "/workspace/fine_tune/qween_instruct/earlyStopping_saved_model"
    model_path = "/workspace/fine_tune/qween_instruct/Strict_SFT_RAG/strict_sft_rag_saved_model"

    infer = QwenInference(model_path)
    infer.load_model_and_tokenizer()

    while True:
        user_query = input("\n🧑 User: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break

        messages = [
            {"role": "user", "content": user_query}]

        infer.generate(
            messages,
            max_new_tokens=500,
            temperature=0.3)

