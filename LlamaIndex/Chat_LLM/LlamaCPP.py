# # https://docs.llamaindex.ai/en/stable/examples/llm/llama_cpp/


model_url = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q3_k_m.gguf"

from llama_index.llms.llama_cpp import LlamaCPP
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


def messages_to_prompt(messages):
    messages = [{"role": m.role.value, "content": m.content} for m in messages]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


def completion_to_prompt(completion):
    messages = [{"role": "user", "content": completion}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=16384,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": -1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=False,
)

# response = llm.complete("Hello! Can you tell me a poem about cats and dogs?")
# print(response.text)


# ----------------------------------  Query engine set up with LlamaCPP  ---------------------------------------------------------------
# Query engine set up with LlamaCPP

from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer

set_global_tokenizer(
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct").encode
)
# use Huggingface embeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

from llama_index.core import SimpleDirectoryReader
# load documents
documents = SimpleDirectoryReader(
    input_files=["/media/mtl/Volume F/PROJECTS/projects/LlamaIndex/img_pdf/ReAct_vs_Function_Calling.pdf"]
).load_data()

from llama_index.core import VectorStoreIndex

# create vector store index
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
# set up query engine
query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("compare between React and function calling?")
print(response)


