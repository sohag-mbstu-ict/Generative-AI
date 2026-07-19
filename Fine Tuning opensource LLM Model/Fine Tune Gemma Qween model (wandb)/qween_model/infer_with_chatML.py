import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

STRICT_SYSTEM_PROMPT = """
You are a STRICT RAG assistant.

Rules:
- Answer ONLY using the information provided in the CONTEXT.
- DO NOT use prior knowledge.
- DO NOT guess or invent facts.
- If the answer is not explicitly stated in the context, reply EXACTLY:
  "The information is not available in the provided data."
"""

# STATIC_CONTEXT = """
# Dr. Chashi is an AI-powered mobile Android application designed to help
# farmers and gardeners detect diseases and pests in rooftop gardens and
# field crops. It provides actionable solutions such as fertilizer
# management, irrigation adjustments, and pest control strategies.
# """

STATIC_CONTEXT = """
You are helpfull AI assistant
"""

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

        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="chatml"
        )

        print("✅ Model loaded for inference")

    def infer(self, user_query, max_new_tokens=128):
        messages = [
            {
                "role": "system",
                "content": STRICT_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
    CONTEXT:
    {STATIC_CONTEXT}

    QUESTION:
    {user_query}
    """
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False   # 🔥 CRITICAL
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
                do_sample=False,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                eos_token_id=self.tokenizer.eos_token_id
            )

        new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        ).strip()


# ================== RUN ==================
qwen = QwenInference("/workspace/fine_tune/qween_model/tr_model_r_16/lora_model")
qwen.load()

while True:
    q = input("You: ")
    if q.lower() == "exit":
        break
    print("\n🔹 Answer:", qwen.infer(q), "\n")
