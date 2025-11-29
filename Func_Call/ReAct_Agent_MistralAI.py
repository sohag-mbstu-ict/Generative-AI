# https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/

from llama_index.llms.mistralai import MistralAI
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.llms.mistralai import MistralAI
from llama_index.core.agent.workflow import AgentStream, ToolCallResult
import asyncio
from dotenv import load_dotenv
from pathlib import Path
import os
# Load .env from the correct file path
env_path = Path('/media/mtl/Volume F/PROJECTS/projects/.venv/')
load_dotenv(dotenv_path=env_path)
# Safely get the Groq key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
print("MISTRAL_API_KEY :", MISTRAL_API_KEY)

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

llm = MistralAI(model="mistral-large-latest", api_key=MISTRAL_API_KEY)
agent = ReActAgent(tools=[multiply, add], llm=llm)
# Create a context to store the conversation history/session state
ctx = Context(agent)
print("ctx :  ",ctx)

"""Run Some Example Queries
By streaming the result, we can see the full response, including the thought process and tool calls.
If we wanted to stream only the result, we can buffer the stream and start streaming once"""
# # ------------------------------------------------- Run Some Example Queries --------------------------------------------------

# async def async_function_for_ReAct_Agent():
#     response = await agent.run("What is 20+(2*4)?", ctx=ctx)
#     print(response)

# asyncio.run(async_function_for_ReAct_Agent())

# async def ReAct_Streaming():
#     # Step 1: initiate streaming response (DON'T await here yet!)
#     handler = agent.run("What is 20+(2*4)?", ctx=ctx)

#     # Step 2: stream intermediate events
#     async for ev in handler.stream_events():
#         if isinstance(ev, ToolCallResult):
#             print(f"\n🛠️ Called tool `{ev.tool_name}` with {ev.tool_kwargs} → Result: {ev.tool_output}")
#         elif isinstance(ev, AgentStream):
#             print(ev.delta, end="", flush=True)

#     # Step 3: await final response
#     response = await handler
#     print("\n\n✅ Final Answer:", response)

# asyncio.run(ReAct_Streaming())


"""View Prompts
Let's take a look at the core system prompt powering the ReAct agent!
Within the agent, the current conversation history is dumped below this line."""
# # ------------------------------------------------- View Prompts --------------------------------------------------

# prompt_dict = agent.get_prompts()
# for k, v in prompt_dict.items():
#     print(f"Prompt: {k}\n\nValue: {v.template}")


"""Customizing the Prompt
For fun, let's try instructing the agent to output the answer along with reasoning in bullet points. See "## Additional Rules" section."""
# # ------------------------------------------------- Customizing the Prompt --------------------------------------------------

print("------------------------------------------------- Customizing the Prompt --------------------------------------------------")
from llama_index.core import PromptTemplate

react_system_header_str = """\

You are designed to help with a variety of tasks, from answering questions \
    to providing summaries to other types of analyses.

## Tools
You have access to a wide variety of tools. You are responsible for using
the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools
to complete each subtask.

You have access to the following tools:
{tool_desc}

## Output Format
To answer the question, please use the following format.

```
Thought: I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format until you have enough information
to answer the question without using any more tools. At that point, you MUST respond
in the one of the following two formats:

```
Thought: I can answer without using any more tools.
Answer: [your answer here]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: Sorry, I cannot answer your query.
```

## Additional Rules
- The answer MUST contain a sequence of bullet points that explain how you arrived at the answer. This can include aspects of the previous conversation history.
- You MUST obey the function signature of each tool. Do NOT pass in no arguments if the function expects arguments.

## Current Conversation
Below is the current conversation consisting of interleaving human and assistant messages.

"""
react_system_prompt = PromptTemplate(react_system_header_str)
agent.get_prompts()
# print("agent.get_prompts() : ",agent.get_prompts())

agent.update_prompts({"react_header": react_system_prompt})

async def Customizing_the_Prompt():
    handler = agent.run("What is 5+3+2")
    async for ev in handler.stream_events():
        # if isinstance(ev, ToolCallResult):
        #     print(f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}")
        if isinstance(ev, AgentStream):
            print(f"{ev.delta}", end="", flush=True)
    response = await handler
    print("\n Response after Customizing the Prompt : ",response)

asyncio.run(Customizing_the_Prompt())
