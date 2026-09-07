#!/usr/bin/env bash
# Bootstrap a rented Linux/CUDA box for the SmolVLA fine-tune.
# Deliberately does NOT install LIBERO/robosuite: evaluation happens on the Mac,
# so the cloud box only ever needs to train. That skips the whole EGL swamp.
set -euo pipefail

python3 -m venv .venv || uv venv --python 3.12 .venv
.venv/bin/pip install -U pip
.venv/bin/pip install "lerobot[smolvla,training]"

# Auth first — the dataset is 35 GB and unauthenticated pulls are throttled to ~6 MB/s.
if [ -n "${HF_TOKEN:-}" ]; then
  .venv/bin/hf auth login --token "$HF_TOKEN"
else
  echo "!! Set HF_TOKEN or run: .venv/bin/hf auth login"
fi

.venv/bin/python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "Now: DEVICE=cuda BATCH=64 STEPS=20000 ./train_libero.sh"
