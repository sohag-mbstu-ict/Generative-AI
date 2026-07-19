import time
import torch
import streamlit as st
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

MODEL_PATH = "/workspace/Qwen_Image_Edit/pretrained_model"

st.set_page_config(
    page_title="Qwen Image Edit 2511",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 Qwen Image Edit 2511")

##########################################################################
# Load Model Only Once
##########################################################################

@st.cache_resource
def load_pipeline():
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )

    pipeline.set_progress_bar_config(disable=False)

    return pipeline


with st.spinner("Loading model... (only once)"):
    pipeline = load_pipeline()

st.success("✅ Model Loaded")

##########################################################################
# Sidebar
##########################################################################

st.sidebar.header("Generation Settings")

num_steps = st.sidebar.slider(
    "Inference Steps",
    min_value=5,
    max_value=50,
    value=15,
)

true_cfg = st.sidebar.slider(
    "True CFG Scale",
    1.0,
    10.0,
    4.0,
)

guidance_scale = st.sidebar.slider(
    "Guidance Scale",
    0.0,
    10.0,
    1.0,
)

seed = st.sidebar.number_input(
    "Seed",
    value=0,
    step=1,
)

##########################################################################
# Main UI
##########################################################################

uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
)

prompt = st.text_area(
    "Prompt",
    value="Remove the background and replace it with a clean white studio background.",
    height=120,
)

##########################################################################
# Generate
##########################################################################

if uploaded_image is not None:

    image = Image.open(uploaded_image).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    if st.button("🚀 Generate", use_container_width=True):

        progress = st.progress(0)

        start = time.time()

        progress.progress(10)

        generator = torch.Generator().manual_seed(seed)

        with st.spinner("Generating image..."):

            progress.progress(25)

            output = pipeline(
                image=[image],
                prompt=prompt,
                generator=generator,
                true_cfg_scale=true_cfg,
                negative_prompt=" ",
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
            )

            progress.progress(90)

        output_image = output.images[0]

        end = time.time()

        progress.progress(100)

        with col2:
            st.subheader("Generated Image")
            st.image(output_image, use_container_width=True)

        st.success("Generation Complete!")

        st.metric(
            "⏱ Total Inference Time",
            f"{end-start:.2f} seconds",
        )

        output_image.save("generated.png")
        print("Image Generation has been completed ................................................ ")
        with open("generated.png", "rb") as f:
            st.download_button(
                "⬇ Download Image",
                data=f,
                file_name="generated.png",
                mime="image/png",
            )

            