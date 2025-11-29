# # https://docs.llamaindex.ai/en/stable/examples/llm/mistralai/

from llama_index.llms.mistralai import MistralAI
from dotenv import load_dotenv
from pathlib import Path
import os
# Load .env from the correct file path
env_path = Path('/media/mtl/Volume F/PROJECTS/projects/.venv/')
load_dotenv(dotenv_path=env_path)
# Safely get the Groq key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
print("MISTRAL_API_KEY :", MISTRAL_API_KEY)


# llm = MistralAI(api_key=MISTRAL_API_KEY)
# resp = llm.complete("Paul Graham is ")
# print("resp : ",resp)
# # print("--------------------------- Call chat with a list of messages ----------------------------------")
# # Call chat with a list of messages
# from llama_index.core.llms import ChatMessage
# from llama_index.llms.mistralai import MistralAI

# messages = [
#     ChatMessage(role="system", content="You are CEO of MistralAI."),
#     ChatMessage(role="user", content="Tell me the story about La plateforme"),
# ]
# resp = MistralAI().chat(messages)
# print("resp : ",resp)

# # print("--------------------------- Call chat with a list of messages ----------------------------------")
# from llama_index.core.llms import ChatMessage
# from llama_index.llms.mistralai import MistralAI

# messages = [
#     ChatMessage(role="system", content="You are CEO of MistralAI."),
#     ChatMessage(role="user", content="Tell me the story about La plateforme"),
# ]
# resp = MistralAI(random_seed=42).chat(messages)
# print("resp : ",resp)

# # print("--------------------------- Configure Model ----------------------------------")
# from llama_index.llms.mistralai import MistralAI
# llm = MistralAI(model="mistral-medium")
# resp = llm.stream_complete("Paul Graham is ")
# for r in resp:
#     print(r.delta, end="")
# print()

# print("---------------------------  Function Calling  ----------------------------------")
from llama_index.llms.mistralai import MistralAI
from llama_index.core.tools import FunctionTool


def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b

def mystery(a: int, b: int) -> int:
    """Mystery function on two integers."""
    return a * b + a + b


mystery_tool = FunctionTool.from_defaults(fn=mystery)
multiply_tool = FunctionTool.from_defaults(fn=multiply)

llm = MistralAI(model="mistral-large-latest")

response = llm.predict_and_call(
    [mystery_tool, multiply_tool],
    user_msg="What happens if I run the mystery function on 5 and 7",)
print(str(response))

print("------------------------------------------------------------------")
response = llm.predict_and_call(
    [mystery_tool, multiply_tool],
    user_msg=(
        """What happens if I run the mystery function on the following pairs of numbers? Generate a separate result for each row:
- 1 and 2
- 8 and 4
- 100 and 20 \
"""
    ),
    allow_parallel_tool_calls=True,)
print(str(response))

print("------------------------------------------------------------------")
for s in response.sources:
    print(f"Name: {s.tool_name}, Input: {s.raw_input}, Output: {str(s)}")

print("------------------------------------------------------------------")
import asyncio

async def run_async_llm_call(llm):
    response = await llm.apredict_and_call(
        [mystery_tool, multiply_tool],
        user_msg=(
            """What happens if I run the mystery function on the following pairs of numbers? Generate a separate result for each row:
            - 1 and 2
            - 8 and 4
            - 100 and 20"""
        ),
        allow_parallel_tool_calls=True,
    )
    print(response)

# Top-level entry point
asyncio.run(run_async_llm_call(llm))
print("------------------------------------------------------------------")

print("---------------------- Structured Prediction --------------------------------------------")
from llama_index.llms.mistralai import MistralAI
from llama_index.core.prompts import PromptTemplate
from llama_index.core.bridge.pydantic import BaseModel


class Restaurant(BaseModel):
    """A restaurant with name, city, and cuisine."""

    name: str
    city: str
    cuisine: str


llm = MistralAI(model="mistral-large-latest")
prompt_tmpl = PromptTemplate(
    "Generate a restaurant in a given city {city_name}"
)

# Option 1: Use `as_structured_llm`
restaurant_obj = (
    llm.as_structured_llm(Restaurant)
    .complete(prompt_tmpl.format(city_name="Miami"))
    .raw)

print("restaurant_obj : ",restaurant_obj)
# Option 2: Use `structured_predict`
# restaurant_obj = llm.structured_predict(Restaurant, prompt_tmpl, city_name="Miami")



