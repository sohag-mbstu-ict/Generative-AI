# llama3_react_agent_executor.py

import os
import dotenv

from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


groq_api_key = os.environ["GROQ_API_KEY"]

# Example tools
@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b
TOOLS = [add, multiply]

# Class to encapsulate Llama3 ReAct Agent
class Llama3ReActAgentExecutor:
    def __init__(self, api_key, base_url, model_name, tools=None, temperature=0.0):
        """
        Initialize the Llama3 ReAct AgentExecutor.
        """
        self.tools = tools or []
        # ReAct style prompt — important!
        self.prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful assistant. "
                        "You can perform reasoning step by step. "
                        "When you need to perform calculations, use the provided tools. "
                        "Available tools: {tool_names}. Think carefully before taking actions. Use tools when needed: {tools}",
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

        # Initialize the Llama3 Chat model
        self.llama3 = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            temperature=temperature,
        )
        # Create the ReAct agent
        self.agent = create_react_agent(self.llama3, self.tools, self.prompt)
        # Create the AgentExecutor
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def run(self, message: str):
        """
        Run the agent executor with a user message.
        """
        response = self.agent_executor.invoke({
            "messages": [HumanMessage(content=message)],
            "agent_scratchpad": []  # important fix!
        })
        return response['output']
# Example usage
if __name__ == "__main__":
    agent_executor_instance = Llama3ReActAgentExecutor(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1/",
        model_name="llama3-70b-8192",
        tools=TOOLS,
        temperature=0.0
    )
    # Run multi-step query!
    print(agent_executor_instance.run("First add 10 and 12, then multiply the result by 3"))
