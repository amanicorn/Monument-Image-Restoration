import sys
from pathlib import Path

import numpy as np

# Compatibility for older LaMa dependencies with NumPy 2.x
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, bytes, str],
    }

import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

LAMA_REPO_ID = "amanicorn/monument-lama"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load LaMa's implementation from the cloned repository.
LAMA_CODE = Path("/content/lama")
if str(LAMA_CODE) not in sys.path:
    sys.path.insert(0, str(LAMA_CODE))

from saicinpainting.training.trainers import load_checkpoint


def load_lama():
    config_path = hf_hub_download(
        repo_id=LAMA_REPO_ID,
        filename="config.yaml",
        repo_type="model",
    )
    checkpoint_path = hf_hub_download(
        repo_id=LAMA_REPO_ID,
        filename="models/best.ckpt",
        repo_type="model",
    )

    config = OmegaConf.load(config_path)
    config.training_model.predict_only = True
    config.visualizer.kind = "noop"

    model = load_checkpoint(
        config,
        checkpoint_path,
        map_location="cpu",
        strict=False,
    )
    model.freeze()
    model.to(DEVICE)
    model.eval()
    return model


def restore_lama(image, mask, model):
    image = image.convert("RGB")
    mask = mask.convert("L").resize(image.size)

    image_np = np.asarray(image)
    mask_np = (np.asarray(mask) > 0).astype(np.float32)

    image_tensor = (
        torch.from_numpy(image_np)
        .permute(2, 0, 1)
        .float()
        .div(255)
        .unsqueeze(0)
        .to(DEVICE)
    )
    mask_tensor = (
        torch.from_numpy(mask_np)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        result = model({"image": image_tensor, "mask": mask_tensor})

    output = result["inpainted"][0].permute(1, 2, 0).cpu().numpy()
    output = (np.clip(output, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(output)
