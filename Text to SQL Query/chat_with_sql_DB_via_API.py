from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from pathlib import Path
from dotenv import load_dotenv
import re

# -----------------------------
# Load environment variables
# -----------------------------
env_path = Path("/workspace/Gen_AI/.venv")
load_dotenv(dotenv_path=env_path)

# -----------------------------
# Load the SQLite Database
# -----------------------------
db_path = "/workspace/Gen_AI/Text_to_SQL/Chinook.db"
db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=0)

# -----------------------------
# Helper functions
# -----------------------------
def get_schema(_):
    """Fetch table schema information from the database."""
    return db.get_table_info()

def extract_sql(text):
    """
    Extract the actual SQL query from model output.
    Assumes the SQL is after the last </think> block.
    """
    # Remove <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Extract the first SELECT ...; or all statements
    match = re.search(r"(SELECT .*?;)", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).replace("\n", " ").strip()
    # fallback: return everything cleaned
    return cleaned

def run_query(model_output):
    sql = extract_sql(model_output)
    
    # Remove any LIMIT clause
    sql = re.sub(r"LIMIT\s+\d+", "", sql, flags=re.IGNORECASE).strip()
    
    print(f"[EXTRACTED SQL]:\n{sql}\n")
    try:
        result = db.run(sql)
        print(f"[INFO] Running SQL Query:\n{sql}\n")
        print(f"[INFO] Query Result:\n{result}\n")
        return result
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        return str(e)


# -----------------------------
# Initialize the LLM (Groq API)
# -----------------------------
llm = ChatGroq(model="qwen/qwen3-32b")

# -----------------------------
# SQL Query Generation Chain
# -----------------------------
def write_sql_query(llm):
    template = """Based on the table schema below, write a SQL query that would answer the user's question.
Return ONLY the SQL query, no explanation.

Schema:
{schema}

Question: {question}
SQL Query:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert SQL generator. Convert the input question into a valid SQL query for the given schema."),
        ("human", template),
    ])

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

# -----------------------------
# Response Generation Chain
# -----------------------------
def answer_user_query(question, llm):
    template = """You are an assistant that explains SQL query results in simple language.
Use the schema, question, SQL query, and SQL response to generate an answer.

Schema:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}

Answer:"""

    prompt_response = ChatPromptTemplate.from_messages([
        ("system", "Generate a natural language answer to the user's question based on the SQL response."),
        ("human", template),
    ])

    sql_chain = write_sql_query(llm)

    full_chain = (
        RunnablePassthrough.assign(query=sql_chain)
        | RunnablePassthrough.assign(
            schema=get_schema,
            response=lambda x: run_query(x["query"])
        )
        | prompt_response
        | llm
        | StrOutputParser()
    )

    return full_chain.invoke({"question": question})

# -----------------------------
# Run Example
# -----------------------------
if __name__ == "__main__":
    while True:
        user_question = input("You: ")
        if user_question.lower() in ['exit', 'quit']:
            print("Chat session ended.")
            break
        
        if user_question.strip() == "":
            continue
        # user_question = "Give some Tracks by the Artist name Audioslave"
        print("[LLM THINKING]:\n")
        final_response = answer_user_query(user_question, llm)
        print("[FINAL ANSWER]:\n", final_response)


