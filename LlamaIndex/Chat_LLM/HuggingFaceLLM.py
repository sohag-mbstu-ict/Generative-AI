# https://docs.llamaindex.ai/en/stable/examples/llm/huggingface/

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from huggingface_hub import list_inference_endpoints
HF_TOKEN = ""

remotely_run = HuggingFaceInferenceAPI(
    model_name="tiiuae/falcon-7b-instruct", token=HF_TOKEN)
prompt = "Instruction: Explain the benefits of exercise.\nResponse:"
completion_response = remotely_run.complete(prompt)
print(completion_response)

# completion_response = remotely_run.complete("To infinity, and")
# print(completion_response)

