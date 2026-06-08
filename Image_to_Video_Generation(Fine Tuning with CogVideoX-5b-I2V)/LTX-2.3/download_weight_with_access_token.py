from huggingface_hub import snapshot_download
import os
from huggingface_hub import login
# export HF_TOKEN=""
# export HF_HUB_DISABLE_XET=1

repo_id="Lightricks/LTX-2.3"
local_dir="/workspace/image_to_image/LTX-2.3/pretrained_model"


HF_TOKEN=""
login(token=os.environ["HF_TOKEN"])

# # ✅ Download entire repository
# snapshot_download(
#     repo_id=repo_id,
#     repo_type="model",
#     local_dir=local_dir,
#     local_dir_use_symlinks=False,  # Ensure actual files are copied
#     revision="main"),  # or specific branch/commit tag

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    max_workers=1,
)

print(f"Downloaded all files to: {local_dir}")

