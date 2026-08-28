import gradio as gr
import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from model.networks import Generator

# Use GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load fine-tuned weights
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

@torch.no_grad()
def restore_monument(editor_data):
    if editor_data is None:
        return None, None, "Please upload an image."

    bg_image = editor_data.get("background")
    if bg_image is None:
        return None, None, "No background image found."

    # Resize input to 256x256
    img = bg_image.convert("RGB").resize((256, 256))
    img_np = np.array(img).astype(np.float32) / 255.0

    layers = editor_data.get("layers", [])
    if not layers:
        return None, None, "Please paint over the damaged area."

    mask_accum = np.zeros((256, 256), dtype=np.float32)
    for layer in layers:
        layer_resized = layer.resize((256, 256))
        layer_np = np.array(layer_resized)
        if layer_np.shape[-1] == 4:
            alpha = layer_np[:, :, 3]
            mask_accum = np.maximum(mask_accum, (alpha > 0).astype(np.float32))

    if mask_accum.sum() == 0:
        return None, None, "No brush strokes detected."

    image_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
    image_tensor = (image_tensor * 2 - 1).unsqueeze(0).to(DEVICE)

    mask_tensor = torch.from_numpy(mask_accum).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    incomplete = image_tensor * (1 - mask_tensor)
    ones = torch.ones_like(incomplete[:, 0:1])
    x = torch.cat([incomplete, ones, ones * mask_tensor], dim=1)

    _, stage2 = generator(x, mask_tensor)
    completed = stage2 * mask_tensor + incomplete * (1 - mask_tensor)
    completed = (completed + 1) / 2
    completed = completed.clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    restored_img = (completed * 255).astype(np.uint8)

    mask_preview = (mask_accum * 255).astype(np.uint8)

    return Image.fromarray(mask_preview), Image.fromarray(restored_img), "Restoration complete!"

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
            )
            restore_btn = gr.Button("✨ Restore Monument", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False)

        with gr.Column():
            mask_output = gr.Image(label="Extracted Mask", type="pil")
            restored_output = gr.Image(label="Restored Image", type="pil")

    restore_btn.click(
        fn=restore_monument,
        inputs=[editor],
        outputs=[mask_output, restored_output, status_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
