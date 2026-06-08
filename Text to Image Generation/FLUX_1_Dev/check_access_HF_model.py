from huggingface_hub import model_info
from huggingface_hub import hf_hub_download
import os


info = model_info(
    "black-forest-labs/FLUX.1-dev",
    token="")
print(info.id)
print("-"*150)


hf_hub_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    filename="README.md",
    token=os.environ["HF_TOKEN"],)
print("Access works")
print("-"*150)


hf_hub_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    filename="model_index.json", # If this pass then you will able to download the weight
    token=os.environ["HF_TOKEN"],)
print("SUCCESS")