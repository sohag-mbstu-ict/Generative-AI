import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
"""
pip uninstall torch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade transformers accelerate diffusers imageio-ffmpeg opencv-python 
pip install --upgrade peft imageio einops wandb pandas decord protobuf sentencepiece
"""
prompt = "Generates a video of Santa Claus on the right side of the photo presenting a gift from his hand to the person on the left, while preserving their original features and expressions."
image = load_image(image="/workspace/images/Test_image.png")

# Load the pipeline and move it to GPU
pipe = CogVideoXImageToVideoPipeline.from_pretrained("/workspace/pretrained_model", torch_dtype=torch.bfloat16)
pipe.to('cuda')  # Ensure pipeline is moved to GPU

# Enable optimizations
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

# Use GPU for generation
video = pipe(
    prompt=prompt,
    image=image,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]


# Export the video to file
export_to_video(video, "output_main_turag.mp4", fps=8)


