from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path
from dotenv import load_dotenv
env_path = Path("/home/gflml/Gen_AI/.venv")
load_dotenv(dotenv_path=env_path)

# Load DB
db = SQLDatabase.from_uri("sqlite:////home/gflml/Gen_AI/Text_to_SQL/Chinook.db", sample_rows_in_table_info=0)

def get_schema(_):
    return db.get_table_info()

def run_query(query):
    print(f'Query being run: {query} \n\n')
    return db.run(query)

# --- Load LOCAL model ---
model_path = "/home/gflml/Gen_AI/Qwen3-4B/base_model"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.1,
    do_sample=False,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=pipe)

# --- Chains ---
def write_sql_query(llm):
    template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Given an input question, convert it to a SQL query. No pre-amble. "
                   "Return ONLY the SQL query, nothing else."),
        ("human", template),
    ])
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def answer_user_query(query, llm):
    template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
    prompt_response = ChatPromptTemplate.from_messages([
        ("system", "Given an input question and SQL response, convert it to a natural language answer. No pre-amble."),
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
    return full_chain.invoke({"question": query})

# --- Run ---
query = 'Give some Tracks by the Artist name Audioslave'
response = answer_user_query(query, llm)
print(response)




