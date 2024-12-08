"""
pip uninstall torch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade transformers accelerate diffusers imageio-ffmpeg opencv-python  
pip install --upgrade peft imageio einops wandb pandas decord protobuf sentencepiece
"""
import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "Anime manga style, light watercolor, a beautiful demones in a gorgeous dress, clear facial features, sucubus,vibrant colors."
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")


