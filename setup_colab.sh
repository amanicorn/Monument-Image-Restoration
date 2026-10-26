#!/usr/bin/env bash
set -euo pipefail

PROJECT="/content/Monument-Image-Restoration"
LAMA="/content/lama"
LAMA_REPO="https://github.com/advimman/lama.git"

# Install compatible dependencies; preserve the existing CUDA PyTorch.
python -m pip install \
  "numpy==2.1.3" \
  "gradio==6.30.0" \
  "huggingface-hub==1.33.0" \
  "albumentations==1.3.1" \
  "imgaug==0.4.0" \
  "omegaconf==2.3.1" \
  "hydra-core==1.3.8" \
  "pytorch-lightning==2.6.6" \
  "kornia==0.9.0" \
  "webdataset==1.0.2" \
  "scikit-image==0.26.0" \
  "scikit-learn==1.6.1" \
  "scipy==1.16.3" \
  "pillow==11.3.0" \
  "easydict==1.9.0"

# Clone LaMa if needed.
if [ ! -d "$LAMA/.git" ]; then
  git clone "$LAMA_REPO" "$LAMA"
fi

# Apply the compatibility file saved in this project.
test -f "$PROJECT/compat/lama_aug.py"
cp "$PROJECT/compat/lama_aug.py" \
   "$LAMA/saicinpainting/training/data/aug.py"

# Verify required files.
test -f "$PROJECT/gradio_app.py"
test -f "$PROJECT/lama_inference.py"
grep -q 'def to_tuple' "$LAMA/saicinpainting/training/data/aug.py"

echo "Setup completed."
echo "Launch with:"
echo "cd $PROJECT && python gradio_app.py"
