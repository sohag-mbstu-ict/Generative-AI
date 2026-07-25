import io
import os
import torch
import streamlit as st
from PIL import Image
from diffusers import QwenImageEditPipeline

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Qwen Image Edit Studio",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Qwen Image Edit Studio")
st.write("Upload an image, describe your edit, and let the model transform it.")

# -----------------------------------------------------------------------------
# 2. Model & LoRA Loading (Cached to prevent reloading on rerun)
# -----------------------------------------------------------------------------
MODEL_PATH = "/workspace/Qwen_Image_Edit/pretrained_model"
LORA_PATH = "/workspace/flymyai-lora-trainer/test_lora_saves_edit/checkpoint-480"

@st.cache_resource(show_spinner="Loading model and LoRA weights into GPU...")
def load_pipeline():
    pipe = QwenImageEditPipeline.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16
    )
    
    # Load LoRA weights if they exist
    if os.path.exists(LORA_PATH):
        pipe.load_lora_weights(LORA_PATH)
    else:
        st.warning(f"LoRA path not found: {LORA_PATH}. Running with base model.")
        
    pipe.to("cuda")
    return pipe

# Load model pipeline
try:
    pipeline = load_pipeline()
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Sidebar Parameters
# -----------------------------------------------------------------------------
st.sidebar.header("Generation Settings")

num_inference_steps = st.sidebar.slider(
    "Inference Steps", min_value=10, max_value=100, value=50, step=5
)

true_cfg_scale = st.sidebar.slider(
    "True CFG Scale", min_value=1.0, max_value=15.0, value=4.0, step=0.5
)

seed = st.sidebar.number_input("Random Seed", value=0, step=1)

negative_prompt = st.sidebar.text_input("Negative Prompt", value=" ")

# -----------------------------------------------------------------------------
# 4. Main Interface
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

# Left Column: Inputs
with col1:
    st.subheader("Input")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    # Fallback to local default image if no file is uploaded
    default_img_path = "/workspace/flymyai-lora-trainer/input_image/image_27.png"
    
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
    elif os.path.exists(default_img_path):
        input_image = Image.open(default_img_path).convert("RGB")
        st.info(f"Using default image: `{default_img_path}`")
    else:
        input_image = None
        st.warning("Please upload an image to proceed.")

    if input_image:
        st.image(input_image, caption="Input Image", use_container_width=True)

    prompt = st.text_area(
        "Editing Prompt",
        value="Replace it with a black dog.",
        height=100,
        placeholder="e.g., Change the background to a beach sunset..."
    )
    
    generate_btn = st.button("✨ Generate Edit", type="primary", use_container_width=True)

# Right Column: Output
with col2:
    st.subheader("Output")
    
    if generate_btn:
        if not input_image:
            st.error("Please provide an image first!")
        elif not prompt.strip():
            st.error("Please provide an editing prompt!")
        else:
            # Create a progress bar element in the Streamlit UI
            progress_bar = st.progress(0, text="Starting image generation...")

            # Callback function invoked after every denoising step
            def progress_callback(pipe, step_index, timestep, callback_kwargs):
                current_step = step_index + 1
                progress_percentage = min(current_step / num_inference_steps, 1.0)
                status_text = f"Denoising step {current_step}/{num_inference_steps} ({int(progress_percentage * 100)}%)"
                progress_bar.progress(progress_percentage, text=status_text)
                return callback_kwargs

            generator = torch.manual_seed(seed)
            
            inputs = {
                "image": input_image,
                "prompt": prompt,
                "generator": generator,
                "true_cfg_scale": true_cfg_scale,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "callback_on_step_end": progress_callback,
            }
            
            with torch.inference_mode():
                output = pipeline(**inputs)
                edited_image = output.images[0]

            # Clear progress bar when complete
            progress_bar.empty()

            # Save output locally
            output_dir = "/workspace/flymyai-lora-trainer/output_image"
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "edited_image.png")
            edited_image.save(save_path)

            # Store in session state to persist preview
            st.session_state["edited_image"] = edited_image

    # Display result if available in session state
    if "edited_image" in st.session_state:
        st.image(st.session_state["edited_image"], caption="Edited Image", use_container_width=True)
        
        # Download button
        buf = io.BytesIO()
        st.session_state["edited_image"].save(buf, format="PNG")
        st.download_button(
            label="💾 Download Image",
            data=buf.getvalue(),
            file_name="edited_image.png",
            mime="image/png",
            use_container_width=True
        )