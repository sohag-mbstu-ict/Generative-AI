import os
import gc
import time
from datetime import datetime

import torch
import gradio as gr
from diffusers import Flux2Pipeline
from diffusers.utils import load_image

# ============================================================
# CONFIG & HARDWARE SETUP
# ============================================================
MODEL_PATH = "/workspace/image_to_image/FLUX_2-dev/pretrained_model" 
OUTPUT_DIR = "/workspace/image_to_image/FLUX_2-dev/output_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

assert torch.cuda.is_available(), "CUDA not available"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

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

# ============================================================
# GENERATION FUNCTION
# ============================================================
def generate_image(prompt, init_image, steps, guidance, seed, base_resolution, progress=gr.Progress()):
    start = time.time()
    
    try:
        if init_image is None:
            raise gr.Error("Please upload or provide an initial image for conditioning.")

        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        def pipe_callback(pipe_instance, step_index, timestep, callback_kwargs):
            progress((step_index + 1) / int(steps), desc=f"Diffusing Step {step_index + 1}/{steps}")
            return callback_kwargs

        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=TORCH_DTYPE):
            # Attempting optimization parameters safely
            # We add a high text guidance scale and pass image conditioning explicitly
            result = pipe(
                prompt=prompt,
                image=[init_image],  
                height=int(base_resolution),
                width=int(base_resolution),
                generator=generator,
                num_inference_steps=int(steps), 
                guidance_scale=float(guidance),  # Controls how hard the model listens to the text
                callback_on_step_end=pipe_callback
            )

        image = result.images[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(OUTPUT_DIR, f"flux2_conditioned_{timestamp}.png")
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

    gr.Markdown("# ⚡ FLUX.2-Dev Context-Conditioned Generator (H100 Local Optimized)")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                # STRUCTURAL PROMPT CHANGE: Context models respond drastically better to explicit instruction tokens
                value="[input_image] shows a house in a green field. Generate a brand new image showing an aerial view architectural photograph modifying the house style to be a modern Scandinavian-style home with clean wood lines, while keeping the wide-open green field background.",
                label="Prompt (Be explicit about what changes and what stays!)",
                lines=4
            )
            input_image = gr.Image(
                value="image_to_image/FLUX_2-dev/output_images/flux2_20260607_070043.png", 
                type="pil", 
                label="Initial Reference Image"
            )
            
        with gr.Column():
            output_image = gr.Image(label="Generated Image Output")
            download_file = gr.File(label="Download Full PNG")

    with gr.Row():
        # Raised default text guidance value from 4.0 to 7.5 to force text influence
        steps = gr.Slider(10, 60, value=50, step=1, label="Steps")
        guidance = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Text Guidance Scale (Higher = More Text Changes)")
        seed = gr.Number(value=42, label="Seed")

    base_res = gr.Dropdown(
        choices=[512, 768, 1024, 1440],
        value=1024,
        label="Resolution (Height & Width)"
    )

    generate_btn = gr.Button("🚀 Generate Image", variant="primary")

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, input_image, steps, guidance, seed, base_res],
        outputs=[output_image, download_file]
    )

if __name__ == "__main__":
    demo.queue(max_size=5, default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


    