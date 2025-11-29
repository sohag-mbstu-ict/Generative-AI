# https://docs.llamaindex.ai/en/stable/examples/llm/llamafile/

from llama_index.llms.llamafile import Llamafile
llm = Llamafile(temperature=0, seed=0)
resp = llm.complete("Who is Octavia Butler?")
print(resp)
