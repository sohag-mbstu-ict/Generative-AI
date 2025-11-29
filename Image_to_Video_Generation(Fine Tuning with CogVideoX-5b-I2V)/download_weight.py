"""
pip install huggingface-hub
"""
import torch
from huggingface_hub import snapshot_download
model_path = '/workspace/pretrained_model/'   # The local directory to save downloaded checkpoint
# download pretrained model for donloading the weight to start fine tuning
snapshot_download("THUDM/CogVideoX-5b-I2V", local_dir=model_path)


# ------------------------------------------------------------------------------------

# inference_model_path = '/workspace/inference_model/'   # The local directory to save downloaded checkpoint
# download pretrained model for donloading the weight to start fine tuning
# snapshot_download("THUDM/CogVideoX-5b",  
#                 local_dir=inference_model_path)


