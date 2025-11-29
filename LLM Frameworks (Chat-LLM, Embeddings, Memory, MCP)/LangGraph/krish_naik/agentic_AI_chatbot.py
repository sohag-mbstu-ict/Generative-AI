# https://www.youtube.com/watch?v=HCSPIH3I-vc&list=PLZoTAELRMXVPFd7JdvB-rnTb_5V26NYNO&index=4

from dotenv import load_dotenv
from pathlib import Path
import os
# Specify the path to the .env file inside .venv
env_path = Path("/media/mtl/Volume F/PROJECTS/projects/.venv")
load_dotenv(dotenv_path=env_path)
# Now access the API key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
print("GOOGLE API KEY:", TAVILY_API_KEY)


from langchain_community.document_loaders import ArxivLoader
from langchain_community.retrievers import ArxivRetriever
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun

api_wraper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wraper_arxiv, description="Query arxiv papers")

print(arxiv.name)
# print(arxiv.invoke("Attention all you need"))

api_wraper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wraper_wiki)
print(wiki.name)

from langchain_tavily import TavilySearch

tavily = TavilySearch(
    max_results=5,
    topic="general",)

# results = tavily.invoke("For Pest and disease detections of crops how can we get data like plantix")
# print("results : ",results)
# # combine all these tools in the list
tools = [arxiv,wiki,tavily]
from langchain_groq import ChatGroq
llm = ChatGroq(model = "qwen/qwen3-32b")
# llm.invoke("what is AI")
llm_with_tools = llm.bind_tools(tools = tools)
# print("llm_with_tools : ",llm_with_tools.invoke("What is the recent news on AI")) # tavily tool will be called
# print("llm_with_tools : ",llm_with_tools.invoke("What is the latest research on quantum computing")) # arxiv tool will be called
# print("llm_with_tools : ",llm_with_tools.invoke("What is machine learning")) # wikipedia tool will be called

# #------------------------------------ LangGraph workflow -----------------------------------------
# # State Schema
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage # # Human message or AI message
from typing import Annotated # labelling
from langgraph.graph.message import add_messages # add_messages is Reducers in LangGraph ; reducers will not overwrite the messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages] # add_messages will prevent overwite human or AI messages


# --------------------- Entire Chatbot with LangGraph ---------------------------------
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition 

# # --------------------------------   Node defination ---------------------------------------
def tool_calling_llm(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

# #  ------------------------- Build graph  ------------------------------------------------
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
## ------------------------------ edeges ----------------------------------------
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition
)
builder.add_edge("tools", "tool_calling_llm")
builder.add_edge("tools", END)
graph = builder.compile()
# Save graph visualization as PNG
with open("/media/mtl/Volume F/PROJECTS/projects/Gen_AI/LangGraph/krish_naik/output_Imgs/3_Agentic_AI_Chatbot.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())
print("Graph saved as graph.png – open it in VS Code to view.")

# ----------------------------------- Inference  ---------------------------------------
# messages = graph.invoke({"messages":"1706.03762"})
# for m in messages["messages"]:
#     m.pretty_print()

messages = graph.invoke({"messages":"Hi My name is sohag and please tell me What is the latest research on quantum computing"})
for m in messages["messages"]:
    m.pretty_print()
    
a= 4



