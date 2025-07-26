from dotenv import load_dotenv
from pathlib import Path
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# ✅ Load environment variables
env_path = Path("/media/mtl/Volume F/PROJECTS/projects/.venv")
load_dotenv(dotenv_path=env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("GOOGLE API KEY:", GOOGLE_API_KEY)

# ✅ Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", GOOGLE_API_KEY=GOOGLE_API_KEY)

# ✅ Setup ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm)

# ✅ Custom prompt with summary placeholder
prompt = PromptTemplate(
    input_variables=["input", "history"],
    template="""
You are a helpful assistant. This is a summary of previous conversations:
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

# ✅ Chat loop
while True:
    query = input("You: ")

    if query.lower() == "exit":
        print("End the session")
        break

    elif query.lower() == "summary":
        print("\n🧠 Current Conversation Summary:")
        print(memory.buffer)
        print()
        continue

    response = conversation.invoke(query)
    print("Bot:", response['response'])


