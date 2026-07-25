import io
import time
import torch
import streamlit as st
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "/workspace/Qwen_Image_Edit/pretrained_model"

st.set_page_config(
    page_title="Qwen Image Edit 2511",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 Qwen Image Edit 2511")

# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_pipeline():

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )

    pipe.set_progress_bar_config(disable=True)

    return pipe


with st.spinner("Loading Qwen Image Edit..."):

    pipeline = load_pipeline()

st.success("✅ Model Loaded")

# ============================================================
# GPU Information
# ============================================================

st.sidebar.header("GPU")

st.sidebar.write(
    f"**Device:** {torch.cuda.get_device_name(0)}"
)

st.sidebar.write(
    f"**CUDA:** {torch.cuda.is_available()}"
)

# ============================================================
# Parameters
# ============================================================

st.sidebar.header("Generation Settings")

steps = st.sidebar.slider(
    "Inference Steps",
    5,
    50,
    40,
)

true_cfg = st.sidebar.slider(
    "True CFG Scale",
    1.0,
    10.0,
    4.0,
)

guidance = st.sidebar.slider(
    "Guidance Scale",
    0.0,
    10.0,
    1.0,
)

seed = st.sidebar.number_input(
    "Seed",
    value=0,
)

# ============================================================
# Upload Images
# ============================================================

uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

prompt = st.text_area(
    "Prompt",
    height=120,
    placeholder="Describe how you want to edit the uploaded image(s)...",
)

# ============================================================
# Preview
# ============================================================

images = []

if uploaded_files:

    cols = st.columns(len(uploaded_files))

    for idx, file in enumerate(uploaded_files):

        img = Image.open(file).convert("RGB")

        images.append(img)

        cols[idx].image(
            img,
            caption=file.name,
            use_container_width=True,
        )

# ============================================================
# Generate
# ============================================================

if st.button("🚀 Generate Image", use_container_width=True):

    if len(images) == 0:

        st.error("Please upload at least one image.")

        st.stop()

    if prompt.strip() == "":

        st.error("Please enter a prompt.")

        st.stop()

    progress = st.progress(0)

    status = st.empty()

    start = time.time()

    status.info("Preparing generation...")

    progress.progress(10)

    generator = torch.manual_seed(seed)

    with torch.inference_mode():

        status.info("Running Qwen Image Edit...")

        progress.progress(35)

        output = pipeline(

            image=images,

            prompt=prompt,

            generator=generator,

            true_cfg_scale=true_cfg,

            negative_prompt=" ",

            num_inference_steps=steps,

            guidance_scale=guidance,

            num_images_per_prompt=1,

        )

    progress.progress(95)

    output_image = output.images[0]

    end = time.time()

    total_time = end - start

    progress.progress(100)

    status.success("Generation Complete!")

    st.divider()

    st.subheader("Generated Image")

    st.image(
        output_image,
        use_container_width=True,
    )

    st.metric(
        "⏱ Total Inference Time",
        f"{total_time:.2f} seconds",
    )

    buf = io.BytesIO()

    output_image.save(buf, format="PNG")
    print("Image Generation has been completed ................................................ ")
    st.download_button(
        "⬇ Download Image",
        data=buf.getvalue(),
        file_name="generated.png",
        mime="image/png",
    )



