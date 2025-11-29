from huggingface_hub import snapshot_download

# 🔽 Configuration
repo_id = "Qwen/Qwen3-Embedding-0.6B"
local_dir = "/workspace/Gen_AI/RAG/Base-Qwen3-Embedding-0.6B"  # Change to your desired path

# 📥 Download the entire model repository
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # Copies actual files (no symlinks)
    revision="main",               # or a specific commit/tag if needed
    ignore_patterns=["*.msgpack", "*.h5", "*.tflite"]  # Optional: skip unused formats
)

print(f"✅ Model downloaded to: {local_dir}")