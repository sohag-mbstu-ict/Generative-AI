import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from io import BytesIO
import requests
import time
t0 = time.time()

from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "/workspace/Qwen_Image_Edit/pretrained_model",
    torch_dtype=torch.bfloat16,
    device_map="balanced",)
pipeline.set_progress_bar_config(disable=None)

print(pipeline.hf_device_map)
print("pipeline loaded")
print("Load:", time.time() - t0)

# pipeline.to('cuda')

t1 = time.time()
image1 = Image.open("/workspace/Qwen_Image_Edit/input_images/boy.jpg").convert("RGB")
image2 = Image.open("/workspace/Qwen_Image_Edit/input_images/girl.jpeg").convert("RGB")
prompt = "The boy is on the left, the girl is on the right, facing each other in the central park square."


print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device:", torch.cuda.get_device_name(0))
print(next(pipeline.transformer.parameters()).device)
print(next(pipeline.transformer.parameters()).dtype)

inputs = {
    "image": [image1, image2],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 15, # 40
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

output_img_path = "/workspace/Qwen_Image_Edit/output_images/boy_girl_2511.jpeg"
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save(output_img_path)
    print("image saved at", output_img_path)

print("Inference:", time.time() - t1)


