# groq_react_math_agent.py

import os
from llama_index.core.agent import ReActAgent
from llama_index.llms.langchain import LangChainLLM
from langchain_groq import ChatGroq
import streamlit as st
import time
from llama_index.core.tools import FunctionTool

class GroqReActMathAgent:
    def __init__(self, api_key: str, model_name: str = "llama3-8b-8192", verbose: bool = True):
        """
        Initialize a ReAct agent with math tools using the Groq API.

        Args:
            api_key (str): Your Groq API key
            model_name (str): Llama3 model name ("llama3-8b-8192" or "llama3-70b-8192")
            verbose (bool): Whether to print internal steps
        """
        self.api_key = api_key
        self.model_name = model_name
        self.verbose = verbose

        # Set the environment variable for compatibility
        os.environ["GROQ_API_KEY"] = self.api_key

        # Define basic math tools
        self.tools = self._load_tools()

        # Setup the LLM via LangChain + Groq
        chat_groq = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name
        )
        llm = LangChainLLM(llm=chat_groq)

        # Initialize ReAct agent
        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=llm,
            verbose=self.verbose
        )

    def _load_tools(self):
        """
        Define and return the math tools for the agent.
        """
        def multiply(a: int, b: int) -> int:
            return a * b

        def add(a: int, b: int) -> int:
            return a + b

        def subtract(a: int, b: int) -> int:
            return a - b

        # What: Converts the functions into usable tools.
        # Why: Without this, the agent won’t know how to call your functions.
        return [
            FunctionTool.from_defaults(fn=add),
            FunctionTool.from_defaults(fn=multiply),
            FunctionTool.from_defaults(fn=subtract)
        ]

    def ask(self, query: str) -> str:
        """
        Ask a question to the agent.

        Args:
            query (str): The input question

        Returns:
            str: The agent's response
        """
        return self.agent.chat(query)

# # Example usage
# if __name__ == "__main__":
#     agent = GroqReActMathAgent(api_key=GROQ_API_KEY)
#     # Run some queries
#     print(agent.ask("Add 10 and 12, then multiply the result by 3"))
#     print(agent.ask("Multiply 8 by 6, then subtract 10"))




# Initialize the agent once (replace with your real API key or env var)

# Start timer
start_time = time.time()
@st.cache_resource
def get_agent():
    return GroqReActMathAgent(api_key=GROQ_API_KEY)
agent = get_agent()
st.title("Groq ReAct Math Agent")
query = st.text_input("Enter your math question:")
if query:
    with st.spinner("Processing..."):
        response = agent.ask(query)
    st.markdown("**Agent response:**")
    st.markdown(f"**{response.response}**")  # ✅ just show the final result

# End timer
end_time = time.time()
# Calculate elapsed time
inference_time = end_time - start_time
print(f"Inference time: {inference_time:.4f} seconds")