# https://github.com/codebasics/langgraph-crash-course/blob/main/2_graph_with_condition.ipynb

from typing import TypedDict, Literal

class PortfolioState(TypedDict):
    amount_usd: float
    total_usd: float
    target_currency: Literal["INR", "EUR"] # Choose any from this
    total: float

def calc_total(state: PortfolioState) -> PortfolioState:
    state['total_usd'] = state['amount_usd'] * 1.08
    return state

def convert_to_inr(state: PortfolioState) -> PortfolioState:
    state['total'] = state['total_usd'] * 85
    return state

def convert_to_eur(state: PortfolioState) -> PortfolioState:
    state['total'] = state['total_usd'] * 0.9
    return state

def choose_conversion(state: PortfolioState) -> str:
    return state["target_currency"]

from langgraph.graph import StateGraph, START, END

builder = StateGraph(PortfolioState)

builder.add_node("calc_total_node", calc_total)
builder.add_node("convert_to_inr_node", convert_to_inr)
builder.add_node("convert_to_eur_node", convert_to_eur)

builder.add_edge(START, "calc_total_node")
builder.add_conditional_edges(
    "calc_total_node",
    choose_conversion,
    {
        "INR": "convert_to_inr_node",
        "EUR": "convert_to_eur_node",
    }
)
builder.add_edge(["convert_to_inr_node", "convert_to_eur_node"], END)

graph = builder.compile()
# Save graph visualization as PNG
with open("/media/mtl/Volume F/PROJECTS/projects/Gen_AI/LangGraph/output_imgs/2_graph_with_condition.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())
print("Graph saved as graph.png – open it in VS Code to view.")

EUR_amount = graph.invoke({"amount_usd": 1000, "target_currency": "EUR"})
print("EUR_amount : ",EUR_amount)

INR_amount = graph.invoke({"amount_usd": 1000, "target_currency": "INR"})
print("INR_amount : ",INR_amount)  
    
    
    