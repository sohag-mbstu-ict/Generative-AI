from unsloth import FastLanguageModel
import torch

class LoRAChatMLInferencer:
    """
    Class to load a base model + LoRA adapter and run inference using ChatML-style conversations.
    """
    def __init__(self, base_model_path, trained_lora_model_path, max_seq_length=2048, load_in_4bit=True, dtype=None):
        """
        Initialize the model and tokenizer using the base and LoRA paths.
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

        # ✅ Load the LoRA adapter
        self.model.load_adapter(self.trained_lora_model_path)

    def format_chatml_prompt(self, system_msg: str, user_msg: str) -> str:
        """
        Convert system and user messages into ChatML-style prompt.
        """
        chat_prompt = f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"
        return chat_prompt

    def generate_response(self, system_msg: str, user_msg: str, max_new_tokens: int = 128) -> str:
        """
        Generate a response for a ChatML-style conversation.
        """
        prompt = self.format_chatml_prompt(system_msg, user_msg)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
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
    trained_lora_model_path = "/home/gflmltpc/Projects/Gen_AI/LLM_Fine_Tuning/ChatML-Format/trained_gemma_7b_26_july_25"

    inferencer = LoRAChatMLInferencer(base_model_dir, trained_lora_model_path)

    # Interactive Chat
    system_message = "You are a helpfull  assistant."
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Chat session ended.")
            break
        response = inferencer.generate_response(system_message, user_query)
        print("\n🧪 Response:\n", response)
