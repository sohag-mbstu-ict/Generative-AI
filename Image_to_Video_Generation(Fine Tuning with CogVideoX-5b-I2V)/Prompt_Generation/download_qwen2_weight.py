# https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

"""
pip install huggingface-hub
pip install qwen_vl_utils
"""
import torch
from huggingface_hub import snapshot_download
model_path = '/workspace/Qwen_weight/'   # The local directory to save downloaded checkpoint
# download pretrained model for donloading the weight to start fine tuning
snapshot_download("Qwen/Qwen2-VL-7B-Instruct", 
                    local_dir=model_path)

