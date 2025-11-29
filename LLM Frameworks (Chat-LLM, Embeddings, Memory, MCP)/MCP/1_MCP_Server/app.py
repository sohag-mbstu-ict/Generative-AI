# https://github.com/zhsama/duckduckgo-mcp-server
# https://github.com/microsoft/playwright-mcp
# https://github.com/openbnb-org/mcp-server-airbnb
# https://www.youtube.com/watch?v=BG4F3b5QpjM&t=476s
# https://www.youtube.com/watch?v=BG4F3b5QpjM&t=476s


import sys
import types

# --- Compatibility patch for mcp-use with modern LangChain Core ---
try:
    import langchain_core.globals as lc_globals
    sys.modules["langchain.globals"] = lc_globals
except ImportError:
    pass
# ------------------------------------------------------------------


import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# from Langchain_core.prompts import ChatPromptTemplate
from mcp_use import MCPAgent, MCPClient
import os
print("successfull")
async def run_memory_chat():
    """Run a chat using MCPAgent's built-in conversation memory."""
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    config_file = "browser_mcp.json"
    print("Initializing  chat ...........  ")
    # Create MCP client and agent with memory enabled
    client = MCPClient.from_config_file(config_file)
    llm = ChatGroq(model = "qwen/qwen3-32b")

    # Create agent with memory enabled
    agent = MCPAgent(llm = llm,
                    client = client,
                    max_steps = 15,
                    memory_enabled=True # Enable built-in conversation memory
    )

    print("\n ======= Interactive MCP Chat ==========")
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'clear' to clear conversation history")
    print("===================================== \n")
    try:
        # main chat loop
        while True:
            # Get user input
            user_input = input("\nYou: ")

            # Check for exit command
            if user_input.lower() in ["exit", "quit"]:
                print("Ending conversation .......  ")
                break

            # Check for clear history command
            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("Conversation history cleared. ")
                continue

            # Get response from agent
            print("\nAssistant: ", end="", flush=True)

            try:
                # Run the agent with the user input (memory handling is automatic)
                response = await agent.run(user_input)
                print(response)
            except Exception as e:
                print(f"\n Error: {e}")
    
    finally:
        # Clean up
        if client and client.session:
            await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(run_memory_chat())



