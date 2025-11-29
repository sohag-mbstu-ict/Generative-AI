# # https://docs.llamaindex.ai/en/stable/examples/agent/react_agent_with_query_engine/

from llama_index.llms.mistralai import MistralAI
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.llms.mistralai import MistralAI
from llama_index.core.agent.workflow import AgentStream, ToolCallResult
from llama_index.core.tools import QueryEngineTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI
import asyncio
from dotenv import load_dotenv
from pathlib import Path
import os
from llama_index.core import Settings

# Load .env from the correct file path
env_path = Path('/media/mtl/Volume F/PROJECTS/projects/.venv/')
load_dotenv(dotenv_path=env_path)
# Safely get the Groq key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
print("MISTRAL_API_KEY :", MISTRAL_API_KEY)

Settings.llm = MistralAI(model="mistral-large-latest", api_key=MISTRAL_API_KEY)
# Settings.embed_model  = MistralAIEmbedding(api_key=MISTRAL_API_KEY)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
from llama_index.core import StorageContext, load_index_from_storage
lyft_pdf    = "/media/mtl/Volume F/PROJECTS/projects/Func_Call/pdf/lyft_2021.pdf"
uber_pdf    = "/media/mtl/Volume F/PROJECTS/projects/Func_Call/pdf/uber_2021.pdf"
lyft_folder = "/media/mtl/Volume F/PROJECTS/projects/Func_Call/index_folder/lyft_folder"
uber_folder = "/media/mtl/Volume F/PROJECTS/projects/Func_Call/index_folder/uber_folder"
try:
    storage_context = StorageContext.from_defaults(
        persist_dir=lyft_folder)
    lyft_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir=uber_folder)
    uber_index = load_index_from_storage(storage_context)
    index_loaded = True
except:
    index_loaded = False

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
if not index_loaded:
    print("----------------- if not index_loaded ----------------------------")
    # load data
    lyft_docs = SimpleDirectoryReader(
        input_files=[lyft_pdf]
    ).load_data()
    uber_docs = SimpleDirectoryReader(
        input_files=[uber_pdf]
    ).load_data()

    # build index
    lyft_index = VectorStoreIndex.from_documents(lyft_docs)
    uber_index = VectorStoreIndex.from_documents(uber_docs)

    # persist index
    lyft_index.storage_context.persist(persist_dir=lyft_folder)
    uber_index.storage_context.persist(persist_dir=uber_folder)



lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)

from llama_index.core.tools import QueryEngineTool

query_engine_tools = [
    QueryEngineTool.from_defaults(
        query_engine=lyft_engine,
        name="lyft_10k",
        description=(
            "Provides information about Lyft financials for year 2021. "
            "Use a detailed plain text question as input to the tool."),),
    QueryEngineTool.from_defaults(
        query_engine=uber_engine,
        name="uber_10k",
        description=(
            "Provides information about Uber financials for year 2021. "
            "Use a detailed plain text question as input to the tool."),),
]

"""Setup ReAct Agent
Here we setup our ReAct agent with the tools we created above.
You can optionally specify a system prompt which will be added to the core ReAct system prompt."""
# # Setup ReAct Agent
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context

agent = ReActAgent(
    tools=query_engine_tools,
    llm=MistralAI(model="mistral-large-latest", api_key=MISTRAL_API_KEY),
    # system_prompt="..."
)
# context to hold this session/state
ctx = Context(agent)
print("ctx : ",ctx)

"""Run Some Example Queries
By streaming the result, we can see the full response, including the thought process and tool calls.
If we wanted to stream only the result, we can buffer the stream and start streaming once Answer: is in the response."""
# # Run Some Example Queries

from llama_index.core.agent.workflow import ToolCallResult, AgentStream
import asyncio

async def ReAct_Agent_with_Query_Engine_RAG_tools():
    # handler = agent.run("What was Lyft's revenue growth in 2021?", ctx=ctx)
    handler = agent.run(
    "Compare and contrast the revenue growth of Uber and Lyft in 2021, then give an analysis",
    ctx=ctx,)
    async for ev in handler.stream_events():
        # if isinstance(ev, ToolCallResult):
        #     print(f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}")
        if isinstance(ev, AgentStream):
            print(f"{ev.delta}", end="", flush=True)
    response = await handler
    print("response : ",response)
asyncio.run(ReAct_Agent_with_Query_Engine_RAG_tools())



