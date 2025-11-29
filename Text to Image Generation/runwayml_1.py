"""
pip uninstall torch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade transformers accelerate diffusers imageio-ffmpeg opencv-python  
pip install --upgrade peft imageio einops wandb pandas decord protobuf sentencepiece
"""
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
                            "runwayml/stable-diffusion-v1-5", 
                            torch_dtype=torch.float16,
                            use_safetensors=True,)

pipe = pipe.to("cuda")

prompt = "Anime manga style, light watercolor, a beautiful demones in a gorgeous dress, clear facial features, sucubus,vibrant colors."
image = pipe(prompt).images[0]
image.save("character.png")
image

