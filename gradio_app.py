import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import label
from huggingface_hub import hf_hub_download
from model.networks import Generator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = hf_hub_download(
    repo_id="amanicorn/monument-deepfillv2",
    filename="states.pth",
    repo_type="model"
)

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
generator = Generator(cnum_in=5, cnum_out=3, cnum=48, return_flow=False)
generator.load_state_dict(checkpoint["G"])
generator.to(DEVICE)
generator.eval()

def run_deepfill_patch(img_patch_pil, mask_patch_np):
    """Processes a 256x256 patch with DeepFill v2."""
    img_np = np.array(img_patch_pil.convert("RGB")).astype(np.float32) / 255.0
    mask_np = mask_patch_np.astype(np.float32)

    image_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
    image_tensor = (image_tensor * 2 - 1).unsqueeze(0).to(DEVICE)
    mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    incomplete = image_tensor * (1 - mask_tensor)
    ones = torch.ones_like(incomplete[:, 0:1])
    x = torch.cat([incomplete, ones, ones * mask_tensor], dim=1)

    _, stage2 = generator(x, mask_tensor)
    completed = stage2 * mask_tensor + incomplete * (1 - mask_tensor)
    completed = (completed + 1) / 2
    completed = completed.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
    return (completed * 255).astype(np.uint8)

@torch.no_grad()
def restore_monument(editor_data):
    if editor_data is None:
        return None, None, "Please upload an image."

    bg_image = editor_data.get("background")
    if bg_image is None:
        return None, None, "No background image found."

    orig_rgb = bg_image.convert("RGB")
    orig_w, orig_h = orig_rgb.size

    layers = editor_data.get("layers", [])
    if not layers:
        return None, None, "Please paint over the damaged area."

    full_mask_np = np.zeros((orig_h, orig_w), dtype=np.float32)
    for layer in layers:
        layer_resized = layer.resize((orig_w, orig_h))
        layer_np = np.array(layer_resized)
        if layer_np.shape[-1] == 4:
            alpha = layer_np[:, :, 3]
            full_mask_np = np.maximum(full_mask_np, (alpha > 0).astype(np.float32))

    if full_mask_np.sum() == 0:
        return None, None, "No brush strokes detected."

    # Identify distinct disconnected painted regions
    labeled_mask, num_features = label(full_mask_np > 0)
    current_image = orig_rgb.copy()

    for i in range(1, num_features + 1):
        component_mask = (labeled_mask == i).astype(np.float32)
        y_indices, x_indices = np.where(component_mask > 0)
        
        min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
        min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))

        bbox_w = max_x - min_x + 1
        bbox_h = max_y - min_y + 1

        # Case 1: Damage fits in 256x256 window -> 1:1 pixel crop with native sharpness
        if bbox_w <= 256 and bbox_h <= 256 and orig_w >= 256 and orig_h >= 256:
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2

            crop_x1 = max(0, min(orig_w - 256, center_x - 128))
            crop_y1 = max(0, min(orig_h - 256, center_y - 128))
            crop_x2 = crop_x1 + 256
            crop_y2 = crop_y1 + 256

            img_crop = current_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            mask_crop = component_mask[crop_y1:crop_y2, crop_x1:crop_x2]

            restored_patch = run_deepfill_patch(img_crop, mask_crop)

            # Paste back seamlessly
            img_arr = np.array(current_image)
            crop_arr = img_arr[crop_y1:crop_y2, crop_x1:crop_x2]
            m3 = np.expand_dims(mask_crop, axis=2)

            blended_crop = (crop_arr * (1.0 - m3) + restored_patch * m3).astype(np.uint8)
            img_arr[crop_y1:crop_y2, crop_x1:crop_x2] = blended_crop
            current_image = Image.fromarray(img_arr)

        # Case 2: Damage is larger than 256px -> Local bounding box resize & seamless blend
        else:
            pad = 20
            crop_x1 = max(0, min_x - pad)
            crop_y1 = max(0, min_y - pad)
            crop_x2 = min(orig_w, max_x + pad)
            crop_y2 = min(orig_h, max_y + pad)

            patch_w = crop_x2 - crop_x1
            patch_h = crop_y2 - crop_y1

            sub_img = current_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            sub_mask = component_mask[crop_y1:crop_y2, crop_x1:crop_x2]

            sub_img_256 = sub_img.resize((256, 256), Image.LANCZOS)
            sub_mask_256 = Image.fromarray((sub_mask * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST)
            sub_mask_np = (np.array(sub_mask_256) > 0).astype(np.float32)

            restored_256 = run_deepfill_patch(sub_img_256, sub_mask_np)
            restored_hd = Image.fromarray(restored_256).resize((patch_w, patch_h), Image.LANCZOS)

            img_arr = np.array(current_image)
            crop_arr = img_arr[crop_y1:crop_y2, crop_x1:crop_x2]
            m3 = np.expand_dims(sub_mask, axis=2)

            blended_crop = (crop_arr * (1.0 - m3) + np.array(restored_hd) * m3).astype(np.uint8)
            img_arr[crop_y1:crop_y2, crop_x1:crop_x2] = blended_crop
            current_image = Image.fromarray(img_arr)

    mask_preview_pil = Image.fromarray((full_mask_np * 255).astype(np.uint8))
    return mask_preview_pil, current_image, f"Restored {num_features} regions with original camera quality preserved!"

with gr.Blocks(title="Monument Image Restoration") as demo:
    gr.Markdown("# 🏛️ Monument Image Restoration")
    gr.Markdown("Upload a damaged monument photo, paint over the damage with the brush, and click **Restore Monument**.")

    with gr.Row():
        with gr.Column():
            editor = gr.ImageEditor(
                label="Damaged Monument (Draw Mask)",
                type="pil",
                sources=["upload"],
                brush=gr.Brush(colors=["#FFFFFF"], default_size=20),
                eraser=gr.Eraser(default_size=20),
                height=400
            )
            restore_btn = gr.Button("✨ Restore Monument", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False)

        with gr.Column():
            mask_output = gr.Image(label="Extracted Mask", type="pil", height=180)
            restored_output = gr.Image(label="Restored Image (Original HD Preserved)", type="pil", height=400)

    restore_btn.click(
        fn=restore_monument,
        inputs=[editor],
        outputs=[mask_output, restored_output, status_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
