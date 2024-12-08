import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video, load_image
import numpy as np
import imageio

pipe = CogVideoXPipeline.from_pretrained(
    # "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
    "/workspace/inference_model", torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights("/workspace/output/cogvideox-lora__optimizer_adam__steps_100__lr-schedule_cosine_with_restarts__learning-rate_1e-4", adapter_name="cogvideox-lora")
# pipe.load_lora_weights("/workspace/output/cogvideox-lora__optimizer_adamw__steps_10__lr-schedule_cosine_with_restarts__learning-rate_1e-4/pytorch_lora_weights.safetensors", adapter_name="cogvideox-lora")
pipe.set_adapters(["cogvideox-lora"], [1.0])
frames = load_image(image="/workspace/images/Test_image.png")

# prompt = "A whimsical kitchen scene unfolds with Mickey Mouse-like animated character at its center. The character playfully interacts with kitchen utensils and appliances, showcasing a range of energetic and mischievous behavior. It juggles pans from a rack above, but ultimately loses balance and falls to the floor in a comical mishap. The kitchen remains orderly and unchanged throughout, with no changes in lighting or camera perspective, emphasizing the comedic contrast between the character's antics and the static surroundings."
prompt = "Generates a video of Santa Claus on the right side of the photo presenting a gift from his hand to the person on the left, while preserving their original features and expressions"
video = pipe(prompt).frames
# Convert frames to NumPy arrays
video = [np.array(frame) for frame in video[0]]
# Set the output file name and frames per second (fps)
output_file = "output.mp4"
fps = 8

# Write the frames to a video file
with imageio.get_writer(output_file, fps=fps, codec="libx264") as writer:
    for frame in video:
        writer.append_data(frame)

print(f"Video successfully saved to {output_file}")
video
export_to_video(video, "output__.mp4", fps=8)
video
