import os
import time
from datetime import datetime

import torch
import gradio as gr
from diffusers import StableDiffusion3Pipeline

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "/workspace/text_to_image/stable_diffusion_3_5_large/pretrained_model"
OUTPUT_DIR = "/workspace/text_to_image/stable_diffusion_3_5_large/output_image"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT_DEFAULT = "A cat holding a sign that says 'hello world'"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Device:", DEVICE)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("=" * 60)

# ============================================================
# LOAD MODEL ONCE
# ============================================================

print("Loading Stable Diffusion 3 model...")

pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
)

pipe = pipe.to(DEVICE)

print("✓ SD3 loaded successfully")

# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_image(prompt, steps, guidance, seed, resolution):

    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    start = time.time()

    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):

        image = pipe(
            prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            height=int(resolution),
            width=int(resolution),
            generator=generator,
        ).images[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(OUTPUT_DIR, f"sd3_{timestamp}.png")
    image.save(save_path)

    elapsed = time.time() - start
    print(f"Generated in {elapsed:.2f}s → {save_path}")

    return image, save_path

# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks() as demo:

    gr.Markdown("# 🌌 Stable Diffusion 3 Image Generator")

    prompt = gr.Textbox(value=PROMPT_DEFAULT, label="Prompt")

    with gr.Row():
        steps = gr.Slider(10, 60, value=28, step=1, label="Steps")
        guidance = gr.Slider(1.0, 10.0, value=3.5, step=0.1, label="Guidance Scale")
        seed = gr.Number(value=42, label="Seed")

    resolution = gr.Dropdown(
        choices=[512, 768, 1024, 1440],
        value=1024,
        label="Resolution"
    )

    btn = gr.Button("🚀 Generate Image")

    output_image = gr.Image(label="Generated Image")
    download_file = gr.File(label="Download PNG")

    btn.click(
        fn=generate_image,
        inputs=[prompt, steps, guidance, seed, resolution],
        outputs=[output_image, download_file]
    )

# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.queue(max_size=3)
    demo.launch(server_name="0.0.0.0", server_port=7860)

    