import io
import base64
import streamlit as st
import torch
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from huggingface_hub import hf_hub_download
from model.networks import Generator

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Monument Image Restoration",
    page_icon="🏛️",
    layout="wide"
)

# Mobile touch CSS
st.markdown(
    """
    <style>
    iframe[title="streamlit_drawable_canvas.drawable_canvas"] {
        touch-action: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# LOAD MODEL (CACHED)
# ==========================================
@st.cache_resource
def load_model():
    checkpoint_path = hf_hub_download(
        repo_id="amanicorn/monument-deepfillv2",
        filename="states.pth",
        repo_type="model"
    )
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    generator = Generator(cnum_in=5, cnum_out=3, cnum=48, return_flow=False)
    generator.load_state_dict(checkpoint["G"])
    generator.to(DEVICE)
    generator.eval()
    return generator

generator = load_model()

# ==========================================
# INFERENCE LOGIC
# ==========================================
@torch.no_grad()
def restore_image(image_np, mask_np):
    image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
    image_tensor = (image_tensor * 2 - 1).unsqueeze(0).to(DEVICE)

    mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    incomplete = image_tensor * (1 - mask_tensor)
    ones = torch.ones_like(incomplete[:, 0:1])
    x = torch.cat([incomplete, ones, ones * mask_tensor], dim=1)

    _, stage2 = generator(x, mask_tensor)
    completed = stage2 * mask_tensor + incomplete * (1 - mask_tensor)
    completed = (completed + 1) / 2
    completed = completed.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return completed

# ==========================================
# UI
# ==========================================
st.title("🏛️ Monument Image Restoration")
st.write("Upload a damaged monument photo, brush over the damaged areas, and restore it using fine-tuned DeepFill v2.")

st.sidebar.header("Brush Controls")
brush_size = st.sidebar.slider("Brush Size", min_value=5, max_value=60, value=20)
st.sidebar.info("💡 **Instructions**:\n1. Upload an image.\n2. Draw over damaged regions in white.\n3. Click **Restore Monument**.")

uploaded_file = st.file_uploader("Upload Damaged Monument Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Ensure RGB, resized to standard 256x256
    image = Image.open(uploaded_file).convert("RGB").resize((256, 256))

    col_draw, col_mask = st.columns(2)

    with col_draw:
        st.subheader("1. Paint Over Damage")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1.0)",
            stroke_width=brush_size,
            stroke_color="#FFFFFF",
            background_image=image,
            background_color="#000000",
            update_streamlit=True,
            height=256,
            width=256,
            drawing_mode="freedraw",
            key=f"canvas_{uploaded_file.name}",  # Dynamic key prevents stale black cache
        )

    mask_np = np.zeros((256, 256), dtype=np.float32)
    has_mask = False

    if canvas_result.image_data is not None:
        alpha = canvas_result.image_data[:, :, 3]
        mask_np = (alpha > 0).astype(np.float32)
        if mask_np.sum() > 0:
            has_mask = True

    with col_mask:
        st.subheader("2. Mask Preview")
        st.image(mask_np, clamp=True, width=256, caption="Binary Inpainting Mask")

    st.write("---")
    if st.button("✨ Restore Monument", type="primary"):
        if not has_mask:
            st.warning("⚠️ Please paint over the damaged areas on the canvas first.")
        else:
            with st.spinner("Restoring monument..."):
                img_normalized = np.array(image).astype(np.float32) / 255.0
                restored_img = restore_image(img_normalized, mask_np)

            st.subheader("3. Restoration Results")
            res1, res2, res3 = st.columns(3)
            with res1:
                st.image(image, caption="Uploaded Damaged Image", use_container_width=True)
            with res2:
                st.image(mask_np, caption="Applied Mask", use_container_width=True)
            with res3:
                st.image(restored_img, caption="Restored Monument Output", use_container_width=True)
