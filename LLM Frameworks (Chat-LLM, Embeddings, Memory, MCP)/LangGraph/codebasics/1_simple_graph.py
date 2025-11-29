# https://github.com/codebasics/langgraph-crash-course/blob/main/1_simple_graph.ipynb

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class PortfolioState(TypedDict):
    amount_usd: float
    total_usd: float
    total_inr: float
    
def calc_total(state: PortfolioState) -> PortfolioState:
    state['total_usd'] = state['amount_usd'] * 1.08
    return state

def convert_to_inr(state: PortfolioState) -> PortfolioState:
    state['total_inr'] = state['total_usd'] * 85
    return state


builder = StateGraph(PortfolioState)

builder.add_node("calc_total_node", calc_total)
builder.add_node("convert_to_inr_node", convert_to_inr)

builder.add_edge(START, "calc_total_node")
builder.add_edge("calc_total_node", "convert_to_inr_node")
builder.add_edge("convert_to_inr_node", END)

graph = builder.compile()


# Save graph visualization as PNG
with open("/media/mtl/Volume F/PROJECTS/projects/Gen_AI/LangGraph/output_imgs/1_simple_graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())
print("Graph saved as graph.png – open it in VS Code to view.")

amount = graph.invoke({"amount_usd": 100000})  # Convert USD into INR  
print("amount : ",amount)

