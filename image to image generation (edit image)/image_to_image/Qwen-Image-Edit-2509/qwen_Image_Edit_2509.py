import os
import gc
import time
from datetime import datetime

import torch
import gradio as gr
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# ============================================================
# CONFIG & HARDWARE SETUP
# ============================================================
MODEL_PATH = "/workspace/image_to_image/Qwen-Image-Edit-2509/pretrained_model"
OUTPUT_DIR = "/workspace/image_to_image/Qwen-Image-Edit-2509/output_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

assert torch.cuda.is_available(), "CUDA not available"

# Maximize H100 Hopper Tensor Core speed
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

print("=" * 80)
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("VRAM:", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("Torch:", torch.__version__)
print("=" * 80)

DEVICE = "cuda"
TORCH_DTYPE = torch.bfloat16

# ============================================================
# LOAD MODEL ONCE
# ============================================================
print("Loading Qwen Image Edit Plus Pipeline natively onto H100...")
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    MODEL_PATH, 
    torch_dtype=TORCH_DTYPE
).to(DEVICE)

# Native speed optimization for spatial layouts
try:
    pipeline.transformer = pipeline.transformer.to(memory_format=torch.channels_last)
    print("✓ Channels Last memory format enabled")
except Exception as e:
    print("Channels last layout skipped:", e)

print("Qwen Pipeline completely ready ✅")

# ============================================================
# GENERATION FUNCTION
# ============================================================
def edit_image(prompt, init_image, steps, true_cfg, guidance, seed, progress=gr.Progress()):
    start_time = time.time()
    
    try:
        if init_image is None:
            raise gr.Error("Please upload or provide a source image to edit.")

        # Set up seeds cleanly
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        # Gradio custom step monitoring callback hook
        def pipe_callback(pipe_instance, step_index, timestep, callback_kwargs):
            progress((step_index + 1) / int(steps), desc=f"Editing Step {step_index + 1}/{steps}")
            return callback_kwargs

        inputs = {
            "image": [init_image], # Qwen expects images inside an iterable list
            "prompt": prompt,
            "generator": generator,
            "true_cfg_scale": float(true_cfg),
            "negative_prompt": " ",
            "num_inference_steps": int(steps),
            "guidance_scale": float(guidance),
            "num_images_per_prompt": 1,
            "callback_on_step_end": pipe_callback
        }

        print(f"Executing Qwen edit pass with prompt: '{prompt}'")
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=TORCH_DTYPE):
            output = pipeline(**inputs)
            
        output_image = output.images[0]

        # Saving process
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(OUTPUT_DIR, f"qwen_edit_{timestamp}.png")
        output_image.save(save_path)

        elapsed = time.time() - start_time
        print(f"Success! Image generated in {elapsed:.2f}s → {save_path}")

        return output_image, save_path

    except Exception as e:
        raise gr.Error(str(e))
        
    finally:
        # Prevent VRAM fragment buildup across web sessions
        gc.collect()

# ============================================================
# GRADIO UI
# ============================================================
with gr.Blocks() as demo:

    gr.Markdown("# 🤖 Qwen-Image-Edit-Plus Panel (H100 Local Optimized)")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                value="make this image as a areial view, also add green field.",
                label="Editing Command (Prompt)",
                lines=3
            )
            image_input = gr.Image(
                value="/workspace/image_to_image/FLUX_2-dev/output_images/flux2_20260607_070043.png", 
                type="pil", 
                label="Original Input Image"
            )
            
        with gr.Column():
            image_output = gr.Image(label="Edited Output Image")
            file_download = gr.File(label="Download Full Quality PNG")

    with gr.Row():
        true_cfg_scale = gr.Slider(1.0, 15.0, value=4.0, step=0.5, label="True CFG Scale (Prompt Accuracy)")
        guidance_scale = gr.Slider(1.0, 10.0, value=1.0, step=0.1, label="Guidance Scale")
        steps_input = gr.Slider(10, 60, value=40, step=1, label="Inference Steps")
        seed_input = gr.Number(value=0, label="Seed")

    generate_btn = gr.Button("🚀 Apply Edit Command", variant="primary")

    # Wire up the execution buttons
    generate_btn.click(
        fn=edit_image,
        inputs=[
            prompt_input, 
            image_input, 
            steps_input, 
            true_cfg_scale, 
            guidance_scale, 
            seed_input
        ],
        outputs=[image_output, file_download]
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

    