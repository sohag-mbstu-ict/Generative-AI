from llama_index.core.agent import ReActAgent
from llama_index.llms.langchain import LangChainLLM
from langchain_groq import ChatGroq
import streamlit as st
import os
import time
from llama_index.core.tools import FunctionTool

os.environ["GROQ_API_KEY"] = api_key
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    # other params...
)

messages = [
    ("system", "You are a helpful translator. Translate the usersentence to Bangali."),
    ("human", "I love you."),
]
# print(llm.invoke(messages))
for chunk in llm.stream(messages):
    print(chunk.text(), end="")
print()

from pydantic import BaseModel, Field

class GetWeather(BaseModel):
    '''Get the current weather in a given location'''
    location: str = Field(..., description="The city and state,e.g. San Francisco, CA")
class GetPopulation(BaseModel):
    '''Get the current population in a given location'''
    location: str = Field(..., description="The city and state,e.g. San Francisco, CA")
model_with_tools = llm.bind_tools([GetWeather, GetPopulation])
ai_msg = model_with_tools.invoke("What is the population of NY?")
print(ai_msg.tool_calls)


from typing import Optional
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

class AnswerWithJustification(BaseModel):
    '''An answer to the user question along with justification for the answer.'''

    answer: str
    # If we provide default values and/or descriptions for fields, these will be passed
    # to the model. This is an important part of improving a model's ability to
    # correctly return structured outputs.
    justification: Optional[str] = Field(
        default=None, description="A justification for the answer."
    )
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
structured_llm = llm.with_structured_output(AnswerWithJustification)

ans = structured_llm.invoke(
    "What weighs more a pound of bricks or a pound of feathers")
print("ans : ",ans)

