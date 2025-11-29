from dotenv import load_dotenv
from pathlib import Path
import os
import csv

from unsloth import FastLanguageModel
from langchain.memory import ConversationSummaryBufferMemory
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline


class ChatAgent:
    def __init__(self):
        self._load_environment()
        self._load_model()
        self._init_pipeline()
        self._init_langchain_memory()
        self.chat_log = []

    def _load_environment(self):
        env_path = Path("/home/gflmltpc/Projects/Gen_AI/.venv")
        load_dotenv(dotenv_path=env_path)
        self.api_key = os.getenv("GOOGLE_API_KEY")

    def _load_model(self):
        self.base_model_dir = "/home/gflmltpc/Projects/Gen_AI/base_model/base-Meta-Llama-3.1-8B-bnb-4bit"
        self.lora_path = "/home/gflmltpc/Projects/Gen_AI/LLM_Fine_Tuning/ChatML-Format/Llama-3.1-trained_4_aug_25"

        # ✅ Unsloth returns model and tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_dir,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True
        )

        FastLanguageModel.for_inference(self.model)
        self.model.load_adapter(self.lora_path)

    def _init_pipeline(self):
        hf_pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1,
            device_map="auto",
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipe)

    def _init_langchain_memory(self):
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=24,
            return_messages=True
        )

    def _build_chatml_prompt(self, user_input: str) -> str:
        system_message = "<|system|>\nYou are an agricultural expert assistant.\n"
        history = self.memory.chat_memory.messages
        conversation_text = system_message

        for msg in history:
            if msg.type == "human":
                conversation_text += f"<|user|>\n{msg.content}\n"
            else:
                conversation_text += f"<|assistant|>\n{msg.content}\n"

        conversation_text += f"<|user|>\n{user_input}\n<|assistant|>\n"
        return conversation_text

    def _extract_clean_response(self, raw_output):
        return raw_output.split("<|assistant|>")[-1].strip()

    def _export_chat_log_to_csv(self):
        csv_file = "/home/gflmltpc/Projects/Gen_AI/LLM_Fine_Tuning/ChatML-Format/chat_log.csv"
        with open(csv_file, mode="w", newline='', encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["User", "Bot"])
            writer.writeheader()
            for entry in self.chat_log:
                writer.writerow(entry)
        print(f"✅ Chat log saved to: {csv_file}")

    def _print_history(self):
        print("\n🧠 Current Conversation History:")
        for msg in self.memory.chat_memory.messages:
            role = "You" if msg.type == "human" else "Bot"
            print(f"{role}: {msg.content}")
        print()

    def _print_token_count(self, prompt: str):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        token_count = input_ids.shape[1]
        print(f"🧾 Token count in memory: {token_count}")

    def start_chat(self):
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() == "exit":
                print("✅ Session ended. Exporting chat log to CSV...")
                self._export_chat_log_to_csv()
                break

            elif user_input.lower() == "history":
                self._print_history()
                continue

            final_prompt = self._build_chatml_prompt(user_input)
            print("===== Final Prompt Sent to Model =====")
            print(final_prompt)
            print("======================================")
            self._print_token_count(final_prompt)

            raw_response = self.llm.invoke(final_prompt)
            response = self._extract_clean_response(raw_response)

            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(response)

            self.chat_log.append({"User": user_input, "Bot": response})

            print("Bot:", response)
            print("-----------------------------------------------------------------------------")


if __name__ == "__main__":
    chat = ChatAgent()
    chat.start_chat()
