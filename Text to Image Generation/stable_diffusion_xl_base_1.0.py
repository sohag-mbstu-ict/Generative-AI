"""
pip uninstall torch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade transformers accelerate diffusers imageio-ffmpeg opencv-python  
pip install --upgrade peft imageio einops wandb pandas decord protobuf sentencepiece
"""
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                        torch_dtype=torch.float16, 
                                        use_safetensors=True, variant="fp16")
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "Anime manga style, light watercolor, a beautiful demones in a gorgeous dress, clear facial features, sucubus,vibrant colors."

images = pipe(prompt=prompt).images[0]

images.save("stable-diffusion-xl-base-1.0.png")


