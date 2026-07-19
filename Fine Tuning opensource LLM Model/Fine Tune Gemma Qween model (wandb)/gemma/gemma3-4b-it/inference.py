# 🔴 MUST BE FIRST
import unsloth
import time
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


class GemmaInference:
    def __init__(self, model_path, device="cuda"):
        self.model_path = model_path
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
            full_finetuning=False,
        )

        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="gemma-3",
        )

        # 🚀 IMPORTANT
        FastLanguageModel.for_inference(self.model)

        self.model.eval()
        print("✅ Model loaded")

    # ---------------- Generate Response ----------------
    def generate(
        self,
        user_query,
        max_new_tokens=300,
        temperature=0.3,
        top_p=0.9,
    ):
        # ✅ Correct Gemma-3 message format
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": user_query}
            ]
        }]

        # ✅ TOKENIZE HERE (NOT STRING)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,                  # 🔥 MUST BE TRUE
            add_generation_prompt=True,     # 🔥 REQUIRED
            return_tensors="pt",
            return_dict=True,
        )

        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )[0].strip()


# ---------------- Usage ----------------
if __name__ == "__main__":
    model_path = "/workspace/fine_tune/gemma_instruct/earlyStopping_saved_model"

    infer = GemmaInference(model_path)
    infer.load_model_and_tokenizer()

    while True:
        start = time.time()
        query = input("\n🧑 User: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Exiting...")
            break

        answer = infer.generate(query)
        print("\n🤖 Assistant:")
        print(answer)
        latency = round(time.time() - start, 3)
        print(f"\n⏱ Time: {latency}s")
        print("-" * 60)
