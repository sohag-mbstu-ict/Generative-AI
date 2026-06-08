import os
import gc
import time
from datetime import datetime

import torch
import gradio as gr
from diffusers import Flux2Pipeline

# ============================================================
# CONFIG & HARDWARE SETUP
# ============================================================
MODEL_PATH = "/workspace/image_to_image/FLUX_2-dev/pretrained_model" 
OUTPUT_DIR = "/workspace/image_to_image/FLUX_2-dev/output_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

assert torch.cuda.is_available(), "CUDA not available"

# Maximize H100 Tensor Core performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

print("=" * 80)
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("VRAM:", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("=" * 80)

DEVICE = "cuda:0"
TORCH_DTYPE = torch.bfloat16

# ============================================================
# LOAD MODEL ONCE 
# ============================================================
print("Loading FLUX.2 pipeline natively onto H100...")

pipe = Flux2Pipeline.from_pretrained(
    MODEL_PATH, 
    torch_dtype=TORCH_DTYPE
).to(DEVICE)

try:
    pipe.transformer = pipe.transformer.to(memory_format=torch.channels_last)
    print("✓ Channels Last memory format enabled")
except Exception as e:
    print("Channels last layout skipped:", e)

print("Pipeline completely ready and self-contained ✅")
print(f"GPU Allocated: {round(torch.cuda.memory_allocated() / 1024**3, 2)} GB")
print(f"GPU Reserved:  {round(torch.cuda.memory_reserved() / 1024**3, 2)} GB")

# ============================================================
# GENERATION FUNCTION
# ============================================================
def generate_image(prompt, steps, guidance, seed, base_resolution, progress=gr.Progress()):
    start = time.time()
    
    try:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        # Explicit tracking callback function
        def pipe_callback(pipe_instance, step_index, timestep, callback_kwargs):
            progress((step_index + 1) / int(steps), desc=f"Diffusing Step {step_index + 1}/{steps}")
            return callback_kwargs

        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=TORCH_DTYPE):
            result = pipe(
                prompt=prompt,
                height=int(base_resolution),
                width=int(base_resolution),
                generator=generator,
                num_inference_steps=int(steps), 
                guidance_scale=float(guidance),
                callback_on_step_end=pipe_callback
            )

        image = result.images[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(OUTPUT_DIR, f"flux2_{timestamp}.png")
        image.save(save_path)

        elapsed = time.time() - start
        print(f"Generated in {elapsed:.2f}s → {save_path}")

        return image, save_path

    except Exception as e:
        raise gr.Error(str(e))
        
    finally:
        gc.collect()

# ============================================================
# GRADIO UI
# ============================================================
with gr.Blocks() as demo:

    gr.Markdown("# ⚡ FLUX.2-Dev Image Generator (H100 Local Optimized)")

    with gr.Row():
        prompt = gr.Textbox(
            value="Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field.",
            label="Prompt",
            lines=4
        )

    with gr.Row():
        steps = gr.Slider(10, 60, value=50, step=1, label="Steps")
        guidance = gr.Slider(1.0, 10.0, value=4.0, step=0.1, label="Guidance Scale")
        seed = gr.Number(value=42, label="Seed")

    base_res = gr.Dropdown(
        choices=[512, 768, 1024, 1440],
        value=1024,
        label="Resolution (Height & Width)"
    )

    generate_btn = gr.Button("🚀 Generate Image", variant="primary")

    output_image = gr.Image(label="Generated Image")
    download_file = gr.File(label="Download Full PNG")

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, steps, guidance, seed, base_res],
        outputs=[output_image, download_file]
    )

# ============================================================
# LAUNCH
# ============================================================
if __name__ == "__main__":
    demo.queue(
        max_size=5,
        default_concurrency_limit=1,
    )
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False
    )