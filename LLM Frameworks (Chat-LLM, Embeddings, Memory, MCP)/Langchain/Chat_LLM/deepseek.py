# https://python.langchain.com/api_reference/deepseek/chat_models/langchain_deepseek.chat_models.ChatDeepSeek.html#langchain_deepseek.chat_models.ChatDeepSeek

from langchain_deepseek import ChatDeepSeek
import os
os.environ["DEEPSEEK_API_KEY"] = "sk-d75ac3f49fb048bebd68ee5102b4ce34"

llm = ChatDeepSeek(model="deepseek-chat")
print(llm.invoke("Explain gravity").content)