# https://github.com/codebasics/langgraph-crash-course/blob/main/4_tool_call.ipynb

from dotenv import load_dotenv
from pathlib import Path
import os
# Specify the path to the .env file inside .venv
env_path = Path("/media/mtl/Volume F/PROJECTS/projects/.venv")
load_dotenv(dotenv_path=env_path)
# Now access the API key
GOOGLE_API_KEY_GFL = os.getenv("GOOGLE_API_KEY_GFL")
print("GOOGLE API KEY:", GOOGLE_API_KEY_GFL)

from langchain.chat_models import init_chat_model
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

os.environ["GOOGLE_API_KEY_GFL"] = GOOGLE_API_KEY_GFL
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,)

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

@tool
def get_stock_price(symbol: str) -> float:
    '''Return the current price of a stock given the stock symbol
    :param symbol: stock symbol
    :return: current price of the stock
    '''
    return {
        "MSFT": 200.3,
        "AAPL": 100.4,
        "AMZN": 150.0,
        "RIL": 87.6
    }.get(symbol, 0.0)

tools = [get_stock_price]

llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)

builder.add_node(chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)

graph = builder.compile()

message = {"role": "user", "content": "What is the price of AAPL stock"}
# message = {"role": "user", "content": "What is the latest price of MSFT stock?"}
response = graph.invoke({"messages":[message]})
print(response["messages"])

# Save graph visualization as PNG
from langchain_core.runnables.graph import MermaidDrawMethod   # add this import

with open("/media/mtl/Volume F/PROJECTS/projects/Gen_AI/LangGraph/output_imgs/4_tool_call.png", "wb") as f:
    f.write(
        graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.PYPPETEER,  # local Chrome/Chromium
            max_retries=3,
            retry_delay=2.0
        )
    )
print("Graph saved as 4_tool_call.png – open it in VS Code to view.")

state = graph.invoke({"messages": [{"role": "user", "content": "What is the price of AAPL stock right now?"}]})
print(state["messages"][-1].content)


state = graph.invoke({"messages": [{"role": "user", "content": "Who invented theory of relativity? print person name only"}]})
print(state["messages"][-1].content)

msg = "I want to buy 20 AMZN stocks using current price. Then 15 MSFT. What will be the total cost?"

state = graph.invoke({"messages": [{"role": "user", "content": msg}]})
print(state["messages"][-1].content)





