from dotenv import load_dotenv
from pathlib import Path
import os

from langchain_google_genai import ChatGoogleGenerativeAI
"""
You can tweak max_token_limit in ConversationSummaryBufferMemory to control when summarization kicks in.
Best used for long chats where you want both short-term detail and long-term continuity."""


from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# ✅ Load .env for GOOGLE_API_KEY
env_path = Path("/media/mtl/Volume F/PROJECTS/projects/.venv")
load_dotenv(dotenv_path=env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("GOOGLE API KEY:", GOOGLE_API_KEY)

# ✅ Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", GOOGLE_API_KEY=GOOGLE_API_KEY)

# ✅ Initialize ConversationSummaryBufferMemory
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000,  # You can adjust depending on your use case
    return_messages=True
)

# ✅ Setup prompt template
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
The following is a conversation between a human and an AI assistant. The assistant is helpful, creative, and friendly.
Conversation history:
{history}

User: {input}
Assistant:"""
)

# ✅ Setup the conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)

# ✅ CLI chat loop
while True:
    query = input("You: ")

    if query.lower() == "exit":
        print("End the session.")
        break

    elif query.lower() == "summary":
        print("\n🧠 Current Summarized Memory:")
        print(memory.buffer)  # summary of previous messages
        print()
        continue

    elif query.lower() == "history":
        print("\n📜 Full Recent History:")
        for msg in memory.chat_memory.messages:
            role = "You" if msg.type == "human" else "Bot"
            print(f"{role}: {msg.content}")
        print()
        continue

    # Get response from the conversation chain
    response = conversation.invoke(query)
    print("Bot:", response['response'])

