from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ------------------------------
#  Load the SQLite database
# ------------------------------
db = SQLDatabase.from_uri("sqlite:////workspace/Gen_AI/Text_to_SQL/Chinook.db", sample_rows_in_table_info=0)

def get_schema(_=None):
    """Return the table schema for the database."""
    return db.get_table_info()

def run_query(query: str):
    """Execute a SQL query safely."""
    print(f"\n🟦 Query being run:\n{query}\n")
    try:
        return db.run(query)
    except Exception as e:
        return f"Error executing query: {str(e)}"

# ------------------------------
#  Initialize Ollama LLM
# ------------------------------
llm = OllamaLLM(
    model="qwen3:4b",       # Ensure you've done: ollama pull qwen3:4b
    temperature=0.2,
    num_predict=4096
)


# ------------------------------
#  Step 1: SQL Query Generator Chain
# ------------------------------
def write_sql_query(llm):
    template = """You are an expert SQLite database assistant.
You will receive the database schema and a user question.
Write a valid **SQLite** SQL query (not MySQL or PostgreSQL) to answer the question.

### Important:
- Use only valid SQLite syntax.
- Do NOT use INFORMATION_SCHEMA or system tables.
- Return only the SQL query, nothing else.

Schema:
{schema}

Question: {question}
SQL Query:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a precise SQL expert for SQLite databases."),
        ("human", template),
    ])

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

# ------------------------------
#  Step 2: Response Generator Chain
# ------------------------------
def answer_user_query(user_question, llm):
    template = """You are an intelligent assistant that answers user questions
based on the database schema, SQL query, and SQL results.

Schema:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}

Write a clear, human-friendly answer:"""

    prompt_response = ChatPromptTemplate.from_messages([
        ("system", "You convert SQL query results into natural answers."),
        ("human", template),
    ])

    # Chain composition
    full_chain = (
        RunnablePassthrough.assign(query=write_sql_query(llm))
        | RunnablePassthrough.assign(
            schema=get_schema,
            response=lambda x: run_query(x["query"]),
        )
        | prompt_response
        | llm
    )

    return full_chain.invoke({"question": user_question})

# ------------------------------
#  Main chat loop
# ------------------------------
if __name__ == "__main__":
    print("🤖 SQLite + Ollama (Qwen3:4B) Chat Interface")
    print("Type 'exit' or 'quit' to end.\n")

    while True:
        user_question = input("You: ").strip()
        if user_question.lower() in ["exit", "quit"]:
            print("Chat session ended.")
            break
        if not user_question:
            continue

        response = answer_user_query(user_question, llm)
        print("\n🟩 Answer:\n", response, "\n")

# # user_question = "Give some Tracks by the Artist name Audioslave"