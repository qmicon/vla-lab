#!/usr/bin/env bash
# Fine-tune SmolVLA on LIBERO. Defaults are the cloud (CUDA) recipe;
# override DEVICE/BATCH/STEPS for a local MPS smoke test.
set -euo pipefail
cd "$(dirname "$0")"

DEVICE=${DEVICE:-cuda}
BATCH=${BATCH:-64}
STEPS=${STEPS:-20000}
WORKERS=${WORKERS:-8}
RUN=${RUN:-smolvla_libero}

# smolvla_base was pretrained with cameras named camera1/2/3; the LIBERO dataset
# uses image/image2. Without this the run dies on a feature-mismatch assert.
RENAME='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'

MUJOCO_GL=${MUJOCO_GL:-egl} TOKENIZERS_PARALLELISM=false \
.venv/bin/lerobot-train \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --policy.path=lerobot/smolvla_base \
  --policy.device=$DEVICE --policy.push_to_hub=false \
  --rename_map="$RENAME" \
  --batch_size=$BATCH --steps=$STEPS --num_workers=$WORKERS \
  --save_freq=5000 --log_freq=100 \
  --wandb.enable=false \
  --output_dir=outputs/$RUN --job_name=$RUN
