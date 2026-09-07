set -euo pipefail
export HF_HOME=/workspace/hf TOKENIZERS_PARALLELISM=false
ulimit -n 65536              # image default is 1024; DataLoader workers abort without this
BATCH=${BATCH:-64}; STEPS=${STEPS:-20000}; WORKERS=${WORKERS:-32}
PREFETCH=${PREFETCH:-6}; RUN=${RUN:-smolvla_libero}; SAVE=${SAVE:-5000}; LOGF=${LOGF:-100}
RENAME='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
rm -rf /workspace/vla-lab/outputs/$RUN
lerobot-train \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda --policy.push_to_hub=false \
  --rename_map="$RENAME" \
  --batch_size=$BATCH --steps=$STEPS \
  --num_workers=$WORKERS --prefetch_factor=$PREFETCH --persistent_workers=true \
  --save_freq=$SAVE --log_freq=$LOGF \
  --wandb.enable=false \
  --output_dir=/workspace/vla-lab/outputs/$RUN --job_name=$RUN
