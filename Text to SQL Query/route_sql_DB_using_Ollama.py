from typing import TypedDict
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, START, END
from pathlib import Path
import re

# -----------------------------
# Load SQLite Database
# -----------------------------
db_path = "/workspace/Gen_AI/Text_to_SQL/Chinook.db"
db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=0)

# -----------------------------
# Shared State
# -----------------------------
class SharedState(TypedDict):
    query: str
    intent: str
    sql_query: str
    sql_response: str
    answer: str
    model: OllamaLLM


# -----------------------------
# Helper Functions
# -----------------------------
def get_schema(_):
    return db.get_table_info()


def extract_sql(text):
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"(SELECT .*?;)", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).replace("\n", " ").strip()
    return cleaned


def run_query(model_output):
    sql = extract_sql(model_output)
    sql = re.sub(r"LIMIT\s+\d+", "", sql, flags=re.IGNORECASE).strip()
    print(f"[EXTRACTED SQL]:\n{sql}\n")
    try:
        result = db.run(sql)
        return str(result)
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        return str(e)


# -----------------------------
# Model Builder Node
# -----------------------------
def build_model(state: SharedState):
    """Initialize Ollama model."""
    state["model"] = OllamaLLM(model="qwen3:4b", temperature=0.1)
    print("[MODEL LOADED] → Qwen3:4b via Ollama")
    return state


# -----------------------------
# Router Node
# -----------------------------
def router_agent(state: SharedState):
    """Classify query as SQL or General."""
    router_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a classifier that determines whether a user query is about an SQL database query or a general question. "
         "Respond with only one word: 'sql' or 'general'."),
        ("human", """Examples:
        Q: Show me all albums by AC/DC → sql
        Q: What is SQL? → general
        Q: List all employees hired after 2020 → sql
        Q: Tell me a joke → general
        Q: Give me the name of customers from France → sql
        Q: Who founded OpenAI? → general

        Now classify this query: {query}""")
    ])

    chain = router_prompt | state["model"] | StrOutputParser()
    response = chain.invoke({"query": state["query"]}).strip().lower()

    # Simple normalization
    if "sql" in response:
        state["intent"] = "sql"
    else:
        state["intent"] = "general"

    print(f"[ROUTER DECISION] → {state['intent']}")
    return state


# -----------------------------
# SQL Handler Node
# -----------------------------
def sql_query_handler(state: SharedState):
    """Generate SQL → Run → Explain."""
    print("[INFO] Handling SQL query...")
    model = state["model"]

    # 1️⃣ Generate SQL
    sql_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert SQL generator. Convert the question to a valid SQL query."),
        ("human", "Schema:\n{schema}\n\nQuestion: {question}\nSQL Query:")
    ])
    sql_chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | sql_prompt
        | model
        | StrOutputParser()
    )
    sql_query = sql_chain.invoke({"question": state["query"]})
    state["sql_query"] = sql_query

    # 2️⃣ Execute SQL
    sql_result = run_query(sql_query)
    state["sql_response"] = sql_result

    # 3️⃣ Explain result
    explain_prompt = ChatPromptTemplate.from_messages([
        ("system", "Explain the SQL result in simple natural language."),
        ("human",
         """Schema:\n{schema}\n\nQuestion: {question}\nSQL Query: {sql_query}\nSQL Response: {response}\n\nAnswer:"""),
    ])
    explain_chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | explain_prompt
        | model
        | StrOutputParser()
    )
    answer = explain_chain.invoke({
        "question": state["query"],
        "sql_query": sql_query,
        "response": sql_result
    })
    state["answer"] = answer
    return state


# -----------------------------
# General QA Node
# -----------------------------
def general_qa_handler(state: SharedState):
    """Handle non-SQL general queries."""
    print("[INFO] Handling general question...")
    model = state["model"]
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledgeable assistant who answers general questions clearly."),
        ("human", "{query}")
    ])
    chain = qa_prompt | model | StrOutputParser()
    answer = chain.invoke({"query": state["query"]})
    state["answer"] = answer
    return state


# -----------------------------
# Router Logic
# -----------------------------
def route_intent(state: SharedState):
    if state["intent"] == "sql":
        return "sql_query_handler"
    else:
        return "general_qa_handler"


# -----------------------------
# LangGraph Workflow
# -----------------------------
def build_graph():
    workflow = StateGraph(SharedState)
    workflow.add_node("build_model", build_model)
    workflow.add_node("router_agent", router_agent)
    workflow.add_node("sql_query_handler", sql_query_handler)
    workflow.add_node("general_qa_handler", general_qa_handler)

    workflow.add_edge(START, "build_model")
    workflow.add_edge("build_model", "router_agent")
    workflow.add_conditional_edges(
        "router_agent",
        route_intent,
        {
            "sql_query_handler": "sql_query_handler",
            "general_qa_handler": "general_qa_handler",
        },
    )
    workflow.add_edge("sql_query_handler", END)
    workflow.add_edge("general_qa_handler", END)
    return workflow.compile()


# -----------------------------
# Run Interactive Example
# -----------------------------
if __name__ == "__main__":
    compiled_graph = build_graph()
    print("\n✅ Ollama-based Text-to-SQL Router is running (Qwen3:4b)\n")
    while True:
        user_question = input("\nYou: ")
        if user_question.lower() in ["exit", "quit"]:
            print("Session ended.")
            break
        if not user_question.strip():
            continue

        result = compiled_graph.invoke({"query": user_question})
        print("\n----------------------------")
        print("Intent:", result["intent"])
        print("Answer:\n", result["answer"])
        print("----------------------------")
