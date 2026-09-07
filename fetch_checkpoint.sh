#!/usr/bin/env bash
# Pull a finished checkpoint back from the rented box for local evaluation.
#   ./fetch_checkpoint.sh user@1.2.3.4:/root/vla-lab/outputs/smolvla_libero
set -euo pipefail
SRC=${1:?usage: fetch_checkpoint.sh user@host:/path/to/outputs/RUN}
DST=outputs/$(basename "$SRC")
mkdir -p "$DST"
# Only the last checkpoint's weights are needed to evaluate — skip optimizer state.
rsync -avP --exclude 'training_state' "$SRC/checkpoints/last/pretrained_model/" "$DST/pretrained_model/"
echo "Now: POLICY=$DST/pretrained_model ./eval_libero.sh"
