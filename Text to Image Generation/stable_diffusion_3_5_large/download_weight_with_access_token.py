from huggingface_hub import snapshot_download
import os
from huggingface_hub import login
# export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"

repo_id="stabilityai/stable-diffusion-3.5-large"
local_dir="/workspace/text_to_image/stable_diffusion_3_5_large/pretrained_model"


HF_TOKEN=""
login(token=os.environ["HF_TOKEN"])

# ✅ Download entire repository
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # Ensure actual files are copied
    revision="main"),  # or specific branch/commit tag


print(f"Downloaded all files to: {local_dir}")

