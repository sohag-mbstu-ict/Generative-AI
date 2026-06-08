import os
import time
from datetime import datetime

import torch
import gradio as gr
from diffusers import FluxPipeline

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "/workspace/text_to_image/FLUX_1_dev/pretrained_model"
OUTPUT_DIR = "/workspace/text_to_image/FLUX_1_dev/output_image"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT_DEFAULT = "A cat holding a sign that says 'hello world'"

assert torch.cuda.is_available(), "CUDA broken"
# ============================================================
# CUDA SETUP
# ============================================================

# Maximize Ampere Tensor Core performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("Device:", DEVICE)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("=" * 80)

# ============================================================
# LOAD MODEL ONCE
# ============================================================

print("Loading FLUX model into memory...")

pipe = FluxPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
)

# CRITICAL OPTIMIZATION: Send directly to GPU.
# You have an A6000 with 48GB VRAM—avoid 'enable_model_cpu_offload()' 
# as it slows down your generation by swapping data back and forth to RAM.
pipe = pipe.to(DEVICE)

# Native Tensor Core Optimization
try:
    pipe.transformer = pipe.transformer.to(memory_format=torch.channels_last)
    print("✓ Channels Last memory format enabled")
except Exception as e:
    print("Channels last skipped:", e)

print("FLUX loaded successfully ✅")

# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_image(prompt, steps, guidance, seed, base_resolution):
    # Use the unified GPU generator for execution speed
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    start = time.time()

    # Inference mode + native AMP
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = pipe(
            prompt=prompt,
            height=int(base_resolution),
            width=int(base_resolution),
            guidance_scale=float(guidance),
            num_inference_steps=int(steps),
            max_sequence_length=512,  # Best default setting for FLUX prompt details
            generator=generator,
        )

    image = result.images[0]

    # File saving routine
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(OUTPUT_DIR, f"flux_{timestamp}.png")
    image.save(save_path)

    elapsed = time.time() - start
    print(f"Generated in {elapsed:.2f}s → {save_path}")

    return image, save_path


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks() as demo:

    gr.Markdown("# ⚡ FLUX.1-Dev Image Generator (A6000 Optimized)")

    with gr.Row():
        prompt = gr.Textbox(value=PROMPT_DEFAULT, label="Prompt")

    with gr.Row():
        steps = gr.Slider(10, 60, value=50, step=1, label="Steps")
        guidance = gr.Slider(1.0, 10.0, value=3.5, step=0.1, label="Guidance Scale")
        seed = gr.Number(value=0, label="Seed")

    base_res = gr.Dropdown(
        choices=[512, 768, 1024, 1440],
        value=1024,
        label="Resolution (Height & Width)"
    )

    btn = gr.Button("🚀 Generate Image")

    output_image = gr.Image(label="Generated Image")
    download_file = gr.File(label="Download Full PNG")

    btn.click(
        fn=generate_image,
        inputs=[prompt, steps, guidance, seed, base_res],
        outputs=[output_image, download_file]
    )

# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.queue(max_size=3)
    demo.launch(server_name="0.0.0.0", server_port=7860)

    