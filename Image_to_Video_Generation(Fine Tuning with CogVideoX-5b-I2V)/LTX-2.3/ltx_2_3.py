# Cell 0: Environment Setup (Kaggle T4)
import os, gc, psutil
import os
import time
from huggingface_hub import hf_hub_download

print("=== Kaggle T4 Environment Setup ===")
print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB total, {psutil.virtual_memory().available / 1024**3:.1f} GB available")

# Drop filesystem caches
os.system("echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1")
os.system("echo 1 | sudo tee /proc/sys/vm/overcommit_memory > /dev/null 2>&1")

gc.collect()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("✅ Environment optimized!")
print("   Kaggle has 29GB RAM — no swap needed.")

# Cell 1: Clone Wan2GP & install dependencies
import subprocess
try:
    subprocess.run(["nvidia-smi"], check=True)
    print("GPU Active!")
except Exception:
    print("WARNING: No GPU. Go to Settings → Accelerator → GPU T4 x1")


# # Cell 2: Download all required models (Kaggle disk-aware)
# # ---------------------------------------------------------------------------
# REPO = "DeepBeepMeep/LTX-2"
# MODEL_DIR = "Wan2GP"
# TMP_DIR = "TMP_DIR"
# os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs(TMP_DIR, exist_ok=True)

# # === Large files go to /kaggle/tmp, symlinked back ===
# LARGE_FILES = [
#     "ltx-2.3-22b-distilled_diffusion_model_quanto_int8.safetensors",  # 19.4 GB
#     "ltx-2.3-22b-distilled-lora-384.safetensors",                      # 7.6 GB
#     "ltx-2.3-22b_embeddings_connector.safetensors",                     # 4.0 GB
#     "ltx-2.3-22b_text_embedding_projection.safetensors",                # 2.3 GB
#     "ltx-2.3-22b_vae.safetensors",                                      # 1.5 GB
# ]

# for f in LARGE_FILES:
#     dest = os.path.join(MODEL_DIR, f)
#     if os.path.exists(dest):
#         print(f"  ✓ Already exists: {f}")
#         continue
#     print(f"Downloading {f} → /kaggle/tmp ...")
#     hf_hub_download(repo_id=REPO, filename=f, local_dir=TMP_DIR)
#     actual = os.path.join(TMP_DIR, f)
#     os.symlink(actual, dest)
#     print(f"  ✓ {f} (symlinked)")

# # === Small files download normally to /kaggle/working ===
# SMALL_FILES = [
#     "ltx-2.3-22b_audio_vae.safetensors",
#     "ltx-2.3-22b_vocoder.safetensors",
#     "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
#     "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
# ]

# for f in SMALL_FILES:
#     dest = os.path.join(MODEL_DIR, f)
#     if os.path.exists(dest):
#         print(f"  ✓ Already exists: {f}")
#         continue
#     print(f"Downloading {f}...")
#     hf_hub_download(repo_id=REPO, filename=f, local_dir=MODEL_DIR)
#     print(f"  ✓ {f}")

# # === Gemma text encoder — large, goes to /kaggle/tmp ===
# GEMMA_FOLDER = "gemma-3-12b-it-qat-q4_0-unquantized"
# GEMMA_FILES = [
#     "gemma-3-12b-it-qat-q4_0-unquantized_quanto_bf16_int8.safetensors",
#     "added_tokens.json",
#     "chat_template.json",
#     "config_light.json",
#     "generation_config.json",
#     "preprocessor_config.json",
#     "processor_config.json",
#     "special_tokens_map.json",
#     "tokenizer.json",
#     "tokenizer.model",
#     "tokenizer_config.json",
# ]

# # Download gemma to /kaggle/tmp, symlink the whole folder
# gemma_dest = os.path.join(MODEL_DIR, GEMMA_FOLDER)
# gemma_tmp = os.path.join(TMP_DIR, GEMMA_FOLDER)

# if os.path.exists(gemma_dest):
#     print(f"  ✓ Already exists: {GEMMA_FOLDER}/")
# else:
#     os.makedirs(gemma_tmp, exist_ok=True)
#     for gf in GEMMA_FILES:
#         tmp_file = os.path.join(gemma_tmp, gf)
#         if os.path.exists(tmp_file):
#             print(f"  ✓ Already exists: gemma/{gf}")
#             continue
#         print(f"Downloading gemma/{gf} → /kaggle/tmp ...")
#         hf_hub_download(
#             repo_id=REPO,
#             filename=f"{GEMMA_FOLDER}/{gf}",
#             local_dir=TMP_DIR,
#         )
#         print(f"  ✓ gemma/{gf}")
#     # Symlink the whole gemma folder
#     os.symlink(gemma_tmp, gemma_dest)
#     print(f"  ✓ {GEMMA_FOLDER}/ (symlinked)")

# # Clean up HF download cache to free disk
# import shutil
# cache_dir = os.path.join(MODEL_DIR, ".cache")
# if os.path.exists(cache_dir):
#     shutil.rmtree(cache_dir)
# tmp_cache = os.path.join(TMP_DIR, ".cache")
# if os.path.exists(tmp_cache):
#     shutil.rmtree(tmp_cache)

# os.system("df -h /kaggle/working video_generation/TMP_DIR")
# print("\n✅ All downloads complete!")
# # --------------------------------------------------------------------------------

# Cell 3: Gradio Setup
import gc
import os
import sys
import json
import random
import tempfile
import glob
import traceback
import numpy as np
import subprocess
import psutil
from PIL import Image

# ---- bootstrap Wan2GP ----
WAN2GP_DIR = os.path.abspath("/workspace/video_generation/Wan2GP")
sys.path.insert(0, WAN2GP_DIR)
os.chdir(WAN2GP_DIR)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.5"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import gradio as gr
from Wan2GP.shared.utils.audio_video import save_video

# ==== GPU INFO ====
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Compute Capability: {torch.cuda.get_device_capability()}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
ram = psutil.virtual_memory()
print(f"RAM: {ram.total / 1024**3:.1f} GB total, {ram.available / 1024**3:.1f} GB available")
sys.stdout.flush()

# ==== Force attention backends for T4/P100 ====
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ==== LOAD MODEL VIA WAN2GP ====
print("\nLoading LTX-2.3 22B Distilled (quanto int8)...")
sys.stdout.flush()

from mmgp import offload
from Wan2GP.shared.utils import files_locator as fl

fl.set_checkpoints_paths(["models", "ckpts", "."])

from Wan2GP.models.ltx2.ltx2_handler import family_handler

base_model_type = "ltx2_22B"
model_def = {"ltx2_pipeline": "distilled"}
extra = family_handler.query_model_def(base_model_type, model_def)
model_def.update(extra)

gemma_folder = "/workspace/video_generation/Wan2GP/gemma-3-12b-it-qat-q4_0-unquantized"
gemma_files = sorted(glob.glob(os.path.join(gemma_folder, "*.safetensors")))
# gemma_files = "/workspace/video_generation/TMP_DIR/gemma-3-12b-it-qat-q4_0-unquantized/gemma-3-12b-it-qat-q4_0-unquantized_quanto_bf16_int8.safetensors"
quanto_files = [f for f in gemma_files if "quanto" in f]
print("quanto_files : ", quanto_files)
text_encoder_file = quanto_files[0] if quanto_files else (gemma_files[0] if gemma_files else None)
print("text_encoder_file : ", text_encoder_file)
if not text_encoder_file:
    raise FileNotFoundError(f"No .safetensors in {gemma_folder}. Check Cell 2.")
print(f"  Text encoder: {os.path.basename(text_encoder_file)}")

tmp_path = "/workspace/video_generation/Wan2GP" 
transformer_path = os.path.join(tmp_path, "ltx-2.3-22b-distilled_diffusion_model_quanto_int8.safetensors")
if not os.path.isfile(transformer_path):
    raise FileNotFoundError(f"Transformer not found at {transformer_path}. Check Cell 2.")
print(f"  Transformer : {os.path.basename(transformer_path)}")
sys.stdout.flush()

ltx2_model, pipe = family_handler.load_model(
    model_filename=transformer_path,
    model_type="ltx2_22B_distilled",
    base_model_type=base_model_type,
    model_def=model_def,
    dtype=torch.bfloat16,
    VAE_dtype=torch.float32,
    text_encoder_filename=text_encoder_file,
)

# ==== Verify pipeline components ====
print("\n--- Pipeline Components ---")
for name, component in pipe.items():
    if component is not None:
        ctype = type(component).__name__
        if hasattr(component, 'parameters'):
            try:
                p = next(component.parameters())
                print(f"  {name}: {ctype} (dtype={p.dtype})")
            except StopIteration:
                print(f"  {name}: {ctype} (no params)")
        else:
            print(f"  {name}: {ctype}")
    else:
        print(f"  {name}: None")

has_upscaler = pipe.get("spatial_upsampler") is not None
print(f"\n  Spatial Upscaler: {'✅ LOADED' if has_upscaler else '❌ MISSING'}")
print(f"  Note: Distilled LoRA is baked into the quanto int8 checkpoint")
sys.stdout.flush()

# ==== Apply mmgp Profile 4 with upscaler budgets ====
print("\nApplying mmgp Profile 4 with per-model budgets...")
sys.stdout.flush()

offload.profile(
    pipe,
    profile_no=4,
    quantizeTransformer=False,
    convertWeightsFloatTo=torch.bfloat16,
    budgets={
        "transformer":  7000,
        "text_encoder": 1500,
        "vae":          2500,
        "spatial_upsampler": 1500,
        "video_encoder": 1500,
        "*":             500,
    },
)
print("✅ mmgp offloading ready!")
sys.stdout.flush()

offload.shared_state["_attention"] = "sdpa"

print("\n✅ Setup complete! Two-stage distilled pipeline active.")
sys.stdout.flush()

# ==== HELPER FUNCTIONS ====
def get_resolution(base_res_str, aspect_ratio_str):
    base_resolutions = {
        "1080p": 1088,
        "720p": 704,
        "540p": 544,
        "480p": 480,
    }
    ratios = {
        "16:9 Landscape": 16/9,
        "4:3 Standard": 4/3,
        "1:1 Square": 1.0,
        "3:4 Portrait": 3/4,
        "9:16 Portrait": 9/16,
    }
    base = base_resolutions.get(base_res_str, 704)
    ratio = ratios.get(aspect_ratio_str, 16/9)
    if ratio >= 1.0:
        height = base
        width = int(base * ratio)
    else:
        width = base
        height = int(base / ratio)
    width = (width // 32) * 32
    height = (height // 32) * 32
    return width, height

def get_vae_tile_size(height, width):
    vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    effective_vram = vram_mb / 1.5
    if effective_vram >= 24000:
        vae_config = 1
    elif effective_vram >= 8000:
        vae_config = 2
    else:
        vae_config = 3
    ref_size = max(height, width)
    if ref_size > 480:
        vae_config += 1
    if vae_config <= 1:
        tile_size = 0
    elif vae_config == 2:
        tile_size = 512
    elif vae_config == 3:
        tile_size = 256
    else:
        tile_size = 128
    return tile_size, vae_config

DEVICE = torch.device("cuda")

@torch.inference_mode()
def Video_Generation(prompt, input_image_start, input_image_end, seed, duration_dropdown,
                     resolution_dropdown, aspect_ratio_dropdown, progress=gr.Progress()):
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        progress(0, desc="Starting...")

        duration_map = {
            "2 Seconds (49 frames)": 49,
            "3 Seconds (73 frames)": 73,
            "5 Seconds (121 frames)": 121,
            "10 Seconds (241 frames)": 241,
            "15 Seconds (361 frames)": 361,
            "20 Seconds (481 frames)": 481,
        }
        num_frames = duration_map.get(duration_dropdown, 73)
        frame_rate = 24.0

        width, height = get_resolution(resolution_dropdown, aspect_ratio_dropdown)

        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed)

        image_start = None
        image_end = None
        if input_image_start is not None:
            image_start = Image.open(input_image_start).convert("RGB")
        if input_image_end is not None:
            image_end = Image.open(input_image_end).convert("RGB")

        free_vram = torch.cuda.mem_get_info()[0] / 1024**3
        ram = psutil.virtual_memory()
        print(f"\n{'='*60}")
        print(f"Generating: {width}x{height}, {num_frames} frames, seed={seed}")
        print(f"Prompt: {prompt[:100]}...")
        print(f"  VRAM free: {free_vram:.2f} GB | RAM free: {ram.available / 1024**3:.1f} GB")
        print(f"  Stage 1 at {width//2}x{height//2} → 2x upscale → Stage 2 at {width}x{height}")
        print(f"{'='*60}")
        sys.stdout.flush()

        vae_tile_size, vae_config = get_vae_tile_size(height, width)
        print(f"  VAE tile size: {vae_tile_size} (auto-calculated, vae_config={vae_config})")
        sys.stdout.flush()

        total_steps = [8]
        current_step = [0]
        current_pass = [1]

        def cb(step, latent, is_start, override_num_inference_steps=None, pass_no=None, **kwargs):
            if is_start:
                if override_num_inference_steps is not None:
                    total_steps[0] = override_num_inference_steps
                if pass_no is not None:
                    current_pass[0] = pass_no
                current_step[0] = 0
                return
            current_step[0] += 1
            stage_name = "Stage 1 (low-res)" if current_pass[0] == 1 else "Stage 2 (full-res refine)"
            free_vram = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"  [{stage_name}] step {current_step[0]}/{total_steps[0]} | VRAM free: {free_vram:.2f} GB")
            sys.stdout.flush()
            frac = current_step[0] / max(total_steps[0], 1)
            if current_pass[0] == 2:
                frac = 0.7 + 0.3 * frac
            else:
                frac = frac * 0.7
            progress(min(frac, 0.95), desc=f"{stage_name}: {current_step[0]}/{total_steps[0]}")

        gen_kwargs = dict(
            input_prompt=prompt,
            image_start=image_start,
            height=height,
            width=width,
            frame_num=num_frames,
            fps=frame_rate,
            seed=seed,
            callback=cb,
            VAE_tile_size=vae_tile_size,
            enhance_prompt=True,
            # ✅ ADD THIS FIX
            input_video_strength=0.7
        )
        if image_end is not None:
            gen_kwargs["image_end"] = image_end

        result = ltx2_model.generate(**gen_kwargs)

        if result is None:
            return None, "Generation failed or was interrupted."

        # ==== Parse result dict ====
        # Result is a dict with keys: 'x' (video tensor), 'audio' (numpy), 'audio_sampling_rate' (int)
        audio_data = None
        audio_sr = None

        if isinstance(result, dict):
            video_tensor = result.get("x")
            audio_data = result.get("audio")
            audio_sr = result.get("audio_sampling_rate", 24000)
            print(f"  Result dict: x={type(video_tensor)}, audio={type(audio_data)}, sr={audio_sr}")
        elif isinstance(result, tuple):
            video_tensor = result[0]
            if len(result) > 1:
                audio_data = result[1]
            if len(result) > 2:
                audio_sr = result[2]
        else:
            video_tensor = result

        if video_tensor is None or not torch.is_tensor(video_tensor):
            return None, f"❌ No video tensor found. Got: {type(video_tensor)}"

        # video_tensor shape: [C, T, H, W] uint8
        print(f"  Video tensor shape: {video_tensor.shape}, dtype: {video_tensor.dtype}")
        sys.stdout.flush()

        video_tensor = video_tensor.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # ==== Save video ====
        # out_path = tempfile.mktemp(suffix=".mp4")

        output_dir = "/workspace/video_generation/output_video"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"video_{int(time.time())}.mp4"
        out_path = os.path.join(output_dir, filename)

        # Convert [C, T, H, W] uint8 -> [B, C, T, H, W] float [-1, 1] for save_video
        video_for_save = video_tensor.unsqueeze(0).float()
        video_for_save = video_for_save / 127.5 - 1.0

        save_video(
            tensor=video_for_save,
            save_file=out_path,
            fps=frame_rate,
            normalize=True,
            value_range=(-1, 1),
        )
        print(f"  ✅ Video saved: {out_path}")

        # ==== Mux audio if available ====
        if audio_data is not None:
            try:
                import soundfile as sf

                audio_tmp = tempfile.mktemp(suffix=".wav")

                if isinstance(audio_data, np.ndarray):
                    # numpy audio: shape could be (samples,) or (channels, samples) or (samples, channels)
                    audio_np = audio_data
                    if audio_np.ndim == 1:
                        audio_np = audio_np  # mono
                    elif audio_np.ndim == 2:
                        # soundfile expects (samples, channels)
                        if audio_np.shape[0] <= 2:
                            audio_np = audio_np.T
                    sr = int(audio_sr) if audio_sr else 24000
                    sf.write(audio_tmp, audio_np, sr)
                    print(f"  Audio: numpy shape={audio_data.shape}, sr={sr}")
                elif torch.is_tensor(audio_data):
                    import torchaudio
                    audio_cpu = audio_data.cpu().float()
                    if audio_cpu.dim() == 1:
                        audio_cpu = audio_cpu.unsqueeze(0)
                    if audio_cpu.dim() == 3:
                        audio_cpu = audio_cpu.squeeze(0)
                    sr = int(audio_sr) if audio_sr else 24000
                    torchaudio.save(audio_tmp, audio_cpu, sr)
                    print(f"  Audio: tensor shape={audio_cpu.shape}, sr={sr}")
                else:
                    raise ValueError(f"Unknown audio type: {type(audio_data)}")

                final_path = out_path.replace(".mp4", "_with_audio.mp4")
                subprocess.run([
                    "ffmpeg", "-y", "-i", out_path, "-i", audio_tmp,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", final_path
                ], check=True, capture_output=True)

                if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                    out_path = final_path
                    print(f"  ✅ Audio muxed: {out_path}")
                else:
                    print(f"  ⚠️ Audio mux produced empty file, using video-only")
            except Exception as e:
                print(f"  ⚠️ Audio mux failed: {e}")
                traceback.print_exc()

        del video_tensor, video_for_save
        gc.collect()
        torch.cuda.empty_cache()

        progress(1.0, desc="Done!")
        print(f"  ✅ Final output: {out_path}")
        sys.stdout.flush()
        return out_path, f"✅ Done! Seed: {seed} | {width}x{height} | {num_frames} frames"

    except Exception as e:
        traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()
        return None, f"❌ Error: {str(e)}"

# ==== GRADIO UI (AIQUEST BRANDED) ====
CSS = """@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 900px !important; margin: auto !important; }
.brand-header { text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; margin-bottom: 25px; box-shadow: 0 10px 25px rgba(0,0,0,0.15); }
.brand-title { color: white; font-size: 2.2em; font-weight: 700; margin: 0 0 10px 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }
.brand-subtitle { color: #f0f0f0; font-size: 1.1em; margin-bottom: 15px; }
.social-buttons { display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; }
.social-btn { padding: 10px 24px; border-radius: 8px; font-weight: 700; font-size: 15px; text-decoration: none; display: inline-block; color: white; transition: all 0.3s; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
.social-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,0.3); }
.youtube-btn { background: linear-gradient(135deg, #FF0000 0%, #CC0000 100%); }
.x-btn { background: linear-gradient(135deg, #0000 0%, #3333 100%); }
button.primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; font-weight: 600 !important; border-radius: 12px !important; }
.footer { text-align: center; padding: 20px; margin-top: 30px; border-top: 2px solid #e5e7eb; color: #6b7280; }
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft(), title="LTX-2.3 22B Video Generator | AIQUEST") as demo:
    gr.HTML('<div class="brand-header"><div class="brand-title">🎬 LTX-2.3 22B Distilled — Kaggle P100</div><div class="brand-subtitle">Created by <strong>AIQuest Academy</strong> | AI-Powered Video Generation</div><div class="social-buttons"><a href="https://youtube.com/@aiquestacademy" target="_blank" class="social-btn youtube-btn">▶️ Subscribe on YouTube</a><a href="https://x.com/aiquestacademy" target="_blank" class="social-btn x-btn">𝕏 Follow on X</a></div></div>')

    gr.Markdown(
        "**Two-stage distilled pipeline** (8 + 3 steps) with spatial 2x upscaler\n\n"
        "**Recommended:** 720p / 3-5 sec ✅ &nbsp;&nbsp; 540p / 3-10 sec ✅ &nbsp;&nbsp; "
        "480p ⚠️ (Stage 1 at 240p = artifacts)\n\n"
        "**Kaggle advantage:** 29GB RAM = faster mmgp offloading vs Colab"
    )

    with gr.Column():
        prompt = gr.Textbox(label="🎬 Prompt", lines=3,
                   placeholder="A cinematic shot of a red fox walking through a snowy forest...")

        with gr.Accordion("🖼️ Image to Video (Optional)", open=False):
            with gr.Row():
                input_image_start = gr.Image(type="filepath", label="Start Frame (optional)")
                input_image_end = gr.Image(type="filepath", label="End Frame (optional)")
            gr.Markdown("*Upload start and/or end frames. The model will generate video between them.*")

        with gr.Row():
            seed = gr.Number(label="🎲 Seed (-1 for Random)", value=-1, precision=0)
            duration_dropdown = gr.Dropdown(
                label="⏱️ Duration",
                choices=[
                   "2 Seconds (49 frames)",
                   "3 Seconds (73 frames)",
                   "5 Seconds (121 frames)",
                   "10 Seconds (241 frames)",
                   "15 Seconds (361 frames)",
                   "20 Seconds (481 frames)",
                ],
                value="3 Seconds (73 frames)",
            )

        with gr.Row():
            resolution_dropdown = gr.Dropdown(
                label="📐 Base Resolution Quality",
                choices=["1080p", "720p", "540p", "480p"],
                value="720p",
            )
            aspect_ratio_dropdown = gr.Dropdown(
                label="📏 Aspect Ratio",
                choices=["16:9 Landscape", "4:3 Standard", "1:1 Square", "3:4 Portrait", "9:16 Portrait"],
                value="16:9 Landscape",
            )

        gen_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
        video_out = gr.Video(label="🎥 Output")
        status_out = gr.Textbox(label="ℹ️ Status", interactive=False)

        gen_btn.click(
            fn=Video_Generation,
            inputs=[prompt, input_image_start, input_image_end, seed, duration_dropdown,
                    resolution_dropdown, aspect_ratio_dropdown],
            outputs=[video_out, status_out],
        )

    gr.HTML('<div class="footer"><p style="font-size: 16px; margin: 5px 0;">🎬 Created by <strong>AIQuest Academy</strong></p><p style="font-size: 14px; margin: 5px 0; color: #9ca3af;">Free & Open Source | LTX-2.3 22B Distilled | Kaggle P100 GPU</p><p style="font-size: 13px; margin: 10px 0;"><a href="https://youtube.com/@aiquestacademy" target="_blank" style="color: #667eea; text-decoration: none; margin: 0 10px;">YouTube</a> | <a href="https://x.com/aiquestacademy" target="_blank" style="color: #667eea; text-decoration: none; margin: 0 10px;">X (Twitter)</a></p></div>')

print("\nLaunching Gradio...")
sys.stdout.flush()
demo.queue()
demo.launch(
    share=True,
    inline=False,
    debug=True,
    show_error=True,
    max_threads=1,
    ssr_mode=False,
)