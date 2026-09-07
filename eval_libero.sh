#!/usr/bin/env bash
# Evaluate a policy in the LIBERO simulator. Runs natively on macOS.
#   ./eval_libero.sh                       -> base (un-finetuned) control
#   POLICY=outputs/smolvla_libero/checkpoints/last/pretrained_model ./eval_libero.sh
set -euo pipefail
cd "$(dirname "$0")"

POLICY=${POLICY:-lerobot/smolvla_base}
SUITE=${SUITE:-libero_object}
N=${N:-20}
DEVICE=${DEVICE:-mps}
TASK_IDS=${TASK_IDS:-}          # e.g. "0,7,20" to evaluate a subset of the suite
TAG=${TAG:-$(basename $POLICY)}

RENAME='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'

MUJOCO_GL=${MUJOCO_GL:-cgl} TOKENIZERS_PARALLELISM=false \
.venv/bin/lerobot-eval \
  --policy.path=$POLICY --policy.device=$DEVICE \
  --env.type=libero --env.task=$SUITE \
  ${TASK_IDS:+--env.task_ids="[$TASK_IDS]"} \
  --eval.n_episodes=$N --eval.batch_size=1 --eval.use_async_envs=false \
  --rename_map="$RENAME" \
  --output_dir=outputs/eval_${SUITE}_${TAG}
