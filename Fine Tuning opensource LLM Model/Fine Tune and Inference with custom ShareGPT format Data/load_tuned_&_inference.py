from unsloth import FastLanguageModel
import torch

class LoRAShareGPTInferencer:
    """
    Class to load a base model + LoRA adapter and run inference using ShareGPT-style conversation format.
    """

    def __init__(self, base_model_path, trained_lora_model_path, max_seq_length=2048, load_in_4bit=True, dtype=None):
        """
        Initialize the model and tokenizer using the base model and trained LoRA adapter.
        """
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

        # ✅ Load trained LoRA adapter
        print(f"Loading trained LoRA adapter from {self.trained_lora_model_path}...")
        self.model.load_adapter(self.trained_lora_model_path)
        print("Model loaded successfully with LoRA weights.")

    def format_sharegpt_prompt(self, system_msg: str, user_msg: str) -> str:
        """
        Format a prompt using ShareGPT roles into the same style as used during training.
        """
        sharegpt_prompt = f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"
        return sharegpt_prompt

    def generate_response(self, system_msg: str, user_msg: str, max_new_tokens: int = 256) -> str:
        """
        Generate a response for a ShareGPT-style conversation.
        """
        # ✅ Prepare the input prompt
        prompt = self.format_sharegpt_prompt(system_msg, user_msg)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        # ✅ Generate response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            use_cache=True
        )

        decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]

        # ✅ Extract text after <|assistant|>
        if "<|assistant|>" in decoded_output:
            response = decoded_output.split("<|assistant|>")[-1].strip()
        else:
            response = decoded_output

        # Remove EOS if present
        response = response.replace("</s>", "").strip()

        return response


# ✅ Usage Example
if __name__ == "__main__":
    base_model_dir = "/home/gflmltpc/Projects/Gen_AI/base-gemma-7b-it-bnb-4bit"
    trained_lora_model_path = "/home/gflmltpc/Projects/Gen_AI/LLM_Fine_Tuning/ShareGPT-model/trained_gemma_7b_26_july_25"

    inferencer = LoRAShareGPTInferencer(base_model_dir, trained_lora_model_path)

    # System message for context
    system_message = "You are an agricultural expert assistant."

    # Interactive chat
    print("\n✅ Chatbot ready! Type 'exit' to quit.\n")
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Chat session ended.")
            break
        response = inferencer.generate_response(system_message, user_query)
        print("\n🤖 Assistant:\n", response)
