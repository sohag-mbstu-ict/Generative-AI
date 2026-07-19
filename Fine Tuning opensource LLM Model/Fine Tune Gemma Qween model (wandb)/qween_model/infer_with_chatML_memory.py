import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

class QwenInference:
    def __init__(self, model_path, device="cuda", max_seq_length=2048):
        self.model_path = model_path
        self.device = device
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None

    def load(self):
        print("🔹 Loading Qwen3 fine-tuned model...")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
            full_finetuning=False,
        )

        self.model.to(self.device)
        self.model.eval()

        # ✅ CORRECT for Qwen3 LoRA (ChatML)
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="chatml"
        )

        print("✅ Model loaded for inference")
        return self.model, self.tokenizer



    # --------------------------------------------------
    # 🔹 SAFE CONTEXT WINDOW
    # --------------------------------------------------
    def trim_history(self, messages, max_turns=4):
        """
        Keep system message + last N user/assistant turns
        """
        system = [m for m in messages if m["role"] == "system"]
        others = [m for m in messages if m["role"] != "system"]
        return system + others[-(max_turns * 2):]

    # --------------------------------------------------
    # 🔹 INFERENCE
    # --------------------------------------------------
    def infer(self, messages, max_new_tokens=128):
        if self.model is None:
            raise RuntimeError("Call load() first")

        messages = self.trim_history(messages)

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
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
                do_sample=False,      # ✅ FACTUAL MODE
                temperature=None,
                top_p=None,
                top_k=None,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # ✅ Decode ONLY new tokens
        new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        response = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        ).strip()

        return response


conversation_history = [
    {
        "role": "system",
        "content": (
            "You are an AI assistant that provides accurate and factual "
            "information about the Dr. Chashi application. "
            "Do not guess or invent facts."
        )
    }
]

qwen_infer = QwenInference(
    "/workspace/fine_tune/qween_model/tr_model_r_16/lora_model"
)

qwen_infer.load()   # ✅ REQUIRED

while True:
    user_query = input("You: ")
    if user_query.lower() == "exit":
        break

    conversation_history.append({"role": "user", "content": user_query})

    answer = qwen_infer.infer({"role": "user", "content": user_query})
    print("\n🔹 Answer:", answer, "\n")

    conversation_history.append({"role": "assistant", "content": answer})
