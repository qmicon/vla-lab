#!/usr/bin/env bash
# Full evaluation profile for one checkpoint, all on the Mac.
set -uo pipefail
cd "$(dirname "$0")"
CKPT=${CKPT:-outputs/ckpt_010000/pretrained_model}
run () { echo "=== $1 (N=$2) ==="; POLICY=$CKPT SUITE=$1 N=$2 TAG=${3:-final} ./eval_libero.sh 2>&1 \
         | tr '\r' '\n' | grep -a "'pc_success'" | tail -1 | grep -o "'pc_success': [0-9.]*, 'n_episodes': [0-9]*"; }
run libero_object 10
run libero_goal   10
run libero_10      5
echo "ALL EVALS DONE"
