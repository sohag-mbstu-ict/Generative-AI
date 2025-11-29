from huggingface_hub import snapshot_download

repo_id="Qwen/Qwen3-0.6B"
local_dir="/home/gflml/Chatbot/pretrained_model/Qwen3-0.6B"

# ✅ Download entire repository
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # Ensure actual files are copied
    revision="main"  # or specific branch/commit tag
)

print(f"Downloaded all files to: {local_dir}")

