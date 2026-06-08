import os
import time
from datetime import datetime

import torch
import gradio as gr
from transformers import AutoModelForCausalLM

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "/workspace/text_to_image/tencent_HunyuanImage_3/pretrained_model"
OUTPUT_DIR = "/workspace/text_to_image/hunyuan_image/output_image"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

# ============================================================
# LOAD MODEL (FIXED)
# ============================================================

print("Loading HunyuanImage-3...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("✓ Model loaded")
model.load_tokenizer(MODEL_PATH)
print("Tokenizer wrapper:", getattr(model, "_tkwrapper", None))
print("Has load_tokenizer:", hasattr(model, "load_tokenizer"))
print("-"*150)
# ============================================================
# GENERATION
# ============================================================

def generate_image(prompt):

    start = time.time()

    with torch.inference_mode():
        image = model.generate_image(
            prompt=prompt,
            stream=False
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(OUTPUT_DIR, f"hunyuan_{timestamp}.png")
    image.save(save_path)

    print(f"Generated → {save_path} in {time.time() - start:.2f}s")

    return image, save_path

# ============================================================
# GRADIO
# ============================================================

with gr.Blocks() as demo:

    gr.Markdown("# 🎨 HunyuanImage-3 (Fixed Loader)")

    prompt = gr.Textbox(value="A cat holding a sign that says hello world")

    btn = gr.Button("Generate")

    out_img = gr.Image()
    out_file = gr.File()

    btn.click(generate_image, prompt, [out_img, out_file])

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)

