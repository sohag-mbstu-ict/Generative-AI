# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb#scrollTo=IWUQP9wjPatu
from huggingface_hub import snapshot_download

repo_id="unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
local_dir="/home/gflmltpc/Projects/Gen_AI/base_model/Llama-3.2-3B-Instruct-bnb-4bit"
#unsloth/Meta-Llama-3.1-8B-bnb-4bit

# ✅ Download entire repository
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # Ensure actual files are copied
    revision="main")  # or specific branch/commit tag

print(f"Downloaded all files to: {local_dir}")


