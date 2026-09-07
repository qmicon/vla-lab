set -euo pipefail
export HF_HOME=/workspace/hf
export PIP_BREAK_SYSTEM_PACKAGES=1 UV_BREAK_SYSTEM_PACKAGES=1
mkdir -p /workspace/hf /workspace/vla-lab
grep -q HF_HOME /root/.bashrc || echo "export HF_HOME=/workspace/hf" >> /root/.bashrc

pip install -q -U pip uv 2>&1 | tail -2
# NB: the image already ships torch 2.9.1+cu128, which satisfies lerobot's
# torch>=2.7,<2.12 — the resolver leaves it alone, so we keep the CUDA build.
uv pip install --system -q "lerobot[smolvla,training]" 2>&1 | tail -5

python3 -c "import torch, lerobot; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('lerobot ok')"
