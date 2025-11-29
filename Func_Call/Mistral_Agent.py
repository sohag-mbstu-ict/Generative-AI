# # https://docs.llamaindex.ai/en/stable/examples/agent/mistral_agent/

from dotenv import load_dotenv
from pathlib import Path
import os
# Specify the path to the .env file inside .venv
env_path = Path("/media/mtl/Volume F/PROJECTS/projects/.venv")
load_dotenv(dotenv_path=env_path)
# Now access the API key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
print("MISTRAL_API_KEY:", MISTRAL_API_KEY)

def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b
from llama_index.llms.mistralai import MistralAI
import asyncio

llm = MistralAI(model="mistral-large-latest", api_key=MISTRAL_API_KEY)

# # Here we initialize a simple Mistral agent with calculator functions.
# from llama_index.core.agent.workflow import FunctionAgent

# agent = FunctionAgent(
#     tools=[multiply, add],
#     llm=llm,)

# # Define an async function
# async def Function_call():
#     response = await agent.run("What is (121 + 2) * 5?")
#     # inspect sources
#     print(response.tool_calls)
#     print("----------------------------------------------------------------------------")
#     print(response)
# # Run it
# asyncio.run(Function_call())



# print("-------------------------  Managing Context/Memory  ---------------------------------------------------")
# # Managing Context/Memory
# from llama_index.core.workflow import Context

# async def Managing_Context_Or_Memory():
#     ctx = Context(agent)  # Context allows conversation state tracking
#     await agent.run("My name is John Doe", ctx=ctx)
#     response = await agent.run("What is my name?", ctx=ctx)
#     print(response)

# asyncio.run(Managing_Context_Or_Memory())


"""
Mistral Agent over RAG Pipeline
Build a Mistral agent over a simple 10K document. We use both Mistral embeddings and mistral-medium to construct the RAG pipeline, 
and pass it to the Mistral agent as a tool."""
# Mistral Agent over RAG Pipeline


from llama_index.core.tools import QueryEngineTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI

embed_model = MistralAIEmbedding(api_key=MISTRAL_API_KEY)
query_llm = MistralAI(model="mistral-medium", api_key=MISTRAL_API_KEY)

# load data
uber_docs = SimpleDirectoryReader(
    input_files=["/media/mtl/Volume F/PROJECTS/projects/LlamaIndex/img_pdf/ReAct_vs_Function_Calling.pdf"]
).load_data()
# build index
uber_index = VectorStoreIndex.from_documents(
    uber_docs, embed_model=embed_model
)
uber_engine = uber_index.as_query_engine(similarity_top_k=3, llm=query_llm)
query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=uber_engine,
    name="uber_10k",
    description=(
        "Provides information about Function Calling and ReAct agent "
        "Use a detailed plain text question as input to the tool."),)

from llama_index.core.agent.workflow import FunctionAgent

agent = FunctionAgent(tools=[query_engine_tool], llm=llm)
async def Mistral_Agent_over_RAG_Pipeline():
    response = await agent.run(
        "Tell me both function calling and ReAct")
    print(str(response))

asyncio.run(Mistral_Agent_over_RAG_Pipeline())