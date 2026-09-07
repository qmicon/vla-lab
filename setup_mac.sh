#!/usr/bin/env bash
# Reproducible LeRobot + LIBERO install on Apple Silicon. Verified on M4 Pro / macOS 26.5.2.
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.12 .venv          # lerobot 0.6.1 requires >=3.12
uv pip install --python .venv/bin/python "lerobot[smolvla,pusht,training,evaluation]"

# --- LIBERO. Three macOS-specific traps, in order: -------------------------
# 1. `lerobot[libero]` is a no-op if lerobot is already installed (uv "audits"
#    the extra away). Install hf-libero explicitly.
# 2. hf-egl-probe builds a C extension and needs cmake on PATH. lerobot already
#    pip-installs one into .venv/bin, so just put that on PATH.
# 3. hf-libero pulls robomimic==0.2.0, which depends on the ORIGINAL egl-probe.
#    EGL is Linux-only and it will not build here. Install robomimic --no-deps
#    and hand-install the rest. (Same trap as roomsim.)
export PATH="$PWD/.venv/bin:$PATH"
uv pip install --python .venv/bin/python --no-deps robomimic==0.2.0 hf-libero==0.1.4
uv pip install --python .venv/bin/python \
  "robosuite==1.4.0" "bddl==1.0.1" hydra-core easydict thop future \
  cloudpickle matplotlib "mujoco<3.9" hf-egl-probe

# 4. LIBERO blocks on an interactive input() the first time it is imported and
#    ~/.libero/config.yaml does not exist. Feed it an N.
echo "N" | .venv/bin/python -c "import libero.libero" >/dev/null 2>&1 || true

echo "OK. Remember: MUJOCO_GL=cgl on macOS (not egl)."
