from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import re

# ------------------------------
#  Load the SQLite database
# ------------------------------
db = SQLDatabase.from_uri("sqlite:////workspace/Gen_AI/Text_to_SQL/Chinook.db", sample_rows_in_table_info=0)

def get_schema(_=None):
    """Return the table schema for the database."""
    return db.get_table_info()

def clean_sql_query(query: str) -> str:
    """Remove markdown-style SQL fences like ```sql ... ```."""
    return re.sub(r"```(?:sql)?|```", "", query, flags=re.IGNORECASE).strip()

def run_query(query: str):
    """Execute a SQL query safely."""
    query = clean_sql_query(query)
    print(f"\n🟦 Query being run:\n{query}\n")
    try:
        return db.run(query)
    except Exception as e:
        return f"Error executing query: {str(e)}"

# ------------------------------
#  Initialize Ollama LLM
# ------------------------------
llm = OllamaLLM(
    model="qwen2.5-coder:7b",  # Or use qwen3:4b
    temperature=0.1,
    num_predict=4096
)

# ------------------------------
#  Step 1: SQL Query Generator Chain
# ------------------------------
def write_sql_query(llm):
    template = """You are an expert SQLite database assistant.
Given the following database schema and user question,
write a valid **SQLite** SQL query that answers the question.

### Rules
- Use only valid SQLite syntax.
- Never use INFORMATION_SCHEMA or system tables.
- Return only the SQL query (no markdown, no commentary).

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

Now write a clear, human-friendly answer:"""

    prompt_response = ChatPromptTemplate.from_messages([
        ("system", "You convert SQL results into natural language answers."),
        ("human", template),
    ])

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
    print("🤖 SQLite + Ollama Chat (Qwen Models)")
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
