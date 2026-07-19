import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


SYSTEM_PROMPT = (
    "You are an AI assistant that provides accurate and actionable "
    "information about the Dr. Chashi application for farmers and gardeners. "
    "Answer ONLY in English. "
    "Do not guess or invent facts."
)

class QwenInference:
    def __init__(self, model_path, device="cuda", max_seq_length=2048):
        self.model_path = model_path
        self.device = device
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None

    # --------------------------------------------------
    # 🔹 LOAD MODEL
    # --------------------------------------------------
    def load(self):
        print("🔹 Loading fine-tuned Qwen3 model...")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
            full_finetuning=False,
        )

        self.model.eval()
        self.model.to(self.device)

        # 🚨 MUST MATCH TRAINING
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="qwen3-thinking"
        )

        print("✅ Model ready for inference")

    # --------------------------------------------------
    # 🔹 INFERENCE
    # --------------------------------------------------
    def infer(self, user_query, max_new_tokens=128):

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_query
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # 🚨 CRITICAL
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,     # deterministic
                temperature=0.0,     # strongest hallucination control
                top_p=1.0,
                top_k=0,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs.input_ids.shape[-1]:]

        return self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        ).strip()


# ================== RUN ==================
if __name__ == "__main__":
    model_path = "/workspace/fine_tune/qween_instruct/earlyStopping_saved_model"

    qwen = QwenInference(model_path)
    qwen.load()

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        answer = qwen.infer(query)
        if not answer.isascii():
            answer = "Please ask again. The answer is available in English only."
        print("\n🔹 Answer:", answer, "\n")
