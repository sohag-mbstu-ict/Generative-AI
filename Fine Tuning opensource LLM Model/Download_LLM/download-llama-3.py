
from huggingface_hub import snapshot_download

repo_id="unsloth/llama-3-8b-bnb-4bit"
local_dir="/home/gflmltpc/Projects/Gen_AI/base-llama-3-8b-bnb-4bit"

# ✅ Download entire repository
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # Ensure actual files are copied
    revision="main")  # or specific branch/commit tag

print(f"Downloaded all files to: {local_dir}")


