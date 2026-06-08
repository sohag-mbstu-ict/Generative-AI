import os
import time
from datetime import datetime

import torch
import gradio as gr
from lens import LensPipeline

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "/workspace/text_to_image/microsoft_Lens/pretrained_model"
OUTPUT_DIR = "/workspace/text_to_image/microsoft_Lens/output_image"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT_DEFAULT = "A cat holding a sign that says 'hello world'"

# ============================================================
# CUDA SETUP
# ============================================================

# Force optimal Tensor Core settings for Ampere architecture (A6000)
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

print("Loading model (ONCE)...")

pipe = LensPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,  # Excellent choice for VRAM/Speed balance
)

pipe = pipe.to(DEVICE)

# OPTIMIZATION 1: Use Channels Last layout
# This restructures tensors natively for RTX Tensor Cores without compilation issues
try:
    pipe.transformer = pipe.transformer.to(memory_format=torch.channels_last)
    print("✓ Channels Last memory format enabled")
except Exception as e:
    print("Channels last skipped:", e)

# OPTIMIZATION 2: Enable PyTorch's native Scaled Dot Product Attention
# Ensures the model leverages FlashAttention under the hood if supported
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

print("Model loaded successfully ✅")

# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_image(prompt, steps, guidance, seed, base_resolution):
    # OPTIMIZATION 3: 'torch.cuda.empty_cache()' removed from here.
    # Wiping cache causes the A6000 to waste time reallocating blocks.

    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    start = time.time()

    # OPTIMIZATION 4: Combined inference_mode with automated mixed precision
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = pipe(
            prompt=prompt,
            base_resolution=int(base_resolution),
            aspect_ratio="1:1",
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=generator,
        )

    image = result.images[0]

    # save file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(OUTPUT_DIR, f"lens_{timestamp}.png")
    image.save(save_path)

    elapsed = time.time() - start

    print(f"Generated in {elapsed:.2f}s → {save_path}")

    # return both preview + file download
    return image, save_path


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks() as demo:

    gr.Markdown("# 🎨 Lens Image Generator (GPU Optimized)")

    with gr.Row():
        prompt = gr.Textbox(value=PROMPT_DEFAULT, label="Prompt")

    with gr.Row():
        steps = gr.Slider(5, 50, value=12, step=1, label="Steps")
        guidance = gr.Slider(1, 10, value=5.0, step=0.5, label="Guidance")
        seed = gr.Number(value=42, label="Seed")

    base_res = gr.Dropdown(
        choices=[512, 768, 1024, 1440],
        value=1024,
        label="Base Resolution"
    )

    btn = gr.Button("🚀 Generate")

    output_image = gr.Image(label="Generated Image")
    download_file = gr.File(label="Download PNG")

    btn.click(
        fn=generate_image,
        inputs=[prompt, steps, guidance, seed, base_res],
        outputs=[output_image, download_file]
    )

# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.queue(max_size=5)
    demo.launch(server_name="0.0.0.0", server_port=7860)


    