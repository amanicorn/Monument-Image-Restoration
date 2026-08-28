
import streamlit as st
import torch
import numpy as np

from PIL import Image, ImageDraw

from model.networks import Generator


# ==========================================
# SETTINGS
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

from huggingface_hub import hf_hub_download

CHECKPOINT = hf_hub_download(
    repo_id="amanicorn/monument-deepfillv2",
    filename="states.pth",
    repo_type="model"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    generator = Generator(
        cnum_in=5,
        cnum_out=3,
        cnum=48,
        return_flow=False
    )

    generator.load_state_dict(checkpoint["G"])
    generator.to(DEVICE)
    generator.eval()

    return generator


generator = load_model()


# ==========================================
# CREATE DAMAGE MASK
# ==========================================

def create_damage_mask():

    mask = Image.new(
        "L",
        (256, 256),
        0
    )

    draw = ImageDraw.Draw(mask)

    # Fixed seed for consistent results
    np.random.seed(42)

    cx = 128
    cy = 128

    for _ in range(7):

        x = np.random.randint(
            cx - 35,
            cx + 35
        )

        y = np.random.randint(
            cy - 35,
            cy + 35
        )

        w = np.random.randint(7, 18)
        h = np.random.randint(7, 18)

        draw.ellipse(
            [
                x - w,
                y - h,
                x + w,
                y + h
            ],
            fill=255
        )

    return np.array(mask).astype(
        np.float32
    ) / 255.0


# ==========================================
# RESTORE IMAGE
# ==========================================

@torch.no_grad()
def restore_image(image, mask):

    original = np.array(
        image
    ).astype(
        np.float32
    ) / 255.0

    image_tensor = torch.from_numpy(
        original.transpose(2, 0, 1)
    ).float()

    image_tensor = (
        image_tensor * 2 - 1
    )

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    mask_tensor = torch.from_numpy(
        mask
    ).float()

    mask_tensor = (
        mask_tensor
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    incomplete = (
        image_tensor
        * (1 - mask_tensor)
    )

    ones = torch.ones_like(
        incomplete[:, 0:1]
    )

    x = torch.cat(
        [
            incomplete,
            ones,
            ones * mask_tensor
        ],
        dim=1
    )

    _, stage2 = generator(
        x,
        mask_tensor
    )

    completed = (
        stage2 * mask_tensor
        + incomplete * (1 - mask_tensor)
    )

    completed = (
        completed + 1
    ) / 2

    completed = completed.clamp(
        0, 1
    )

    completed = (
        completed
        .squeeze(0)
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    return completed


# ==========================================
# STREAMLIT UI
# ==========================================

st.title(
    "🏛️ Monument Image Restoration"
)

st.write(
    "Restore damaged monument images using "
    "a DeepFill v2 model fine-tuned on "
    "Indian monument imagery."
)


uploaded_file = st.file_uploader(
    "Upload a monument image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image = image.resize(
        (256, 256)
    )

    st.subheader("Original Image")

    st.image(
        image,
        width=500
    )


    if st.button(
        "Restore Image"
    ):

        with st.spinner(
            "Restoring monument..."
        ):

            mask = create_damage_mask()

            original = np.array(
                image
            ).astype(
                np.float32
            ) / 255.0

            damaged = (
                original
                * (1 - mask[:, :, None])
            )

            restored = restore_image(
                image,
                mask
            )

        st.subheader(
            "Restoration Results"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.image(
                damaged,
                caption="Damaged",
                use_container_width=True
            )

        with col2:

            st.image(
                restored,
                caption="Restored",
                use_container_width=True
            )

        with col3:

            st.image(
                original,
                caption="Original",
                use_container_width=True
            )
