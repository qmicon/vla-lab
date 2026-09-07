# vla-lab — LeRobot + LIBERO, natively on a Mac

Week-1 rig for the VLA on-ramp: fine-tune SmolVLA on somebody else's data, and get a
**success-rate** number (not just a loss curve) by evaluating in a simulator you own.

Verified 2026-08-21 on M4 Pro / 48 GB / macOS 26.5.2.

## Status

| | |
|---|---|
| LIBERO sim on Apple Silicon | ✅ runs natively, no Linux, no CUDA |
| Raw robosuite throughput | 30.6 steps/s @ 2×256 px offscreen |
| Full eval loop (`lerobot-eval`) | ✅ 7.0 s per episode @ 360 px |
| Baseline: `smolvla_base`, libero_object, n=20 | **0.0 % success** ← the control |
| Training loop | ✅ policy builds, 450 M params / 100 M trainable |
| Fine-tune | pending — dataset downloading, run goes on a rented GPU |

`pc_success = 0.0` for the un-finetuned base model is the number that matters right now:
it proves the harness is not accidentally scoring successes. Any eval that can't produce a
trustworthy zero is not an eval.

## Layout

```
setup_mac.sh      reproducible install (every macOS trap commented inline)
eval_libero.sh    run a policy in the LIBERO sim, get pc_success
train_libero.sh   fine-tune SmolVLA (DEVICE=cuda default, DEVICE=mps to smoke test)
smoke_libero.py   bare robosuite/LIBERO render + step benchmark, no lerobot
```

## Versions that work together

```
python 3.12.12   lerobot 0.6.1    torch 2.11.0     numpy 2.2.6
hf-libero 0.1.4  robosuite 1.4.0  robomimic 0.2.0  bddl 1.0.1  mujoco 3.8.1
transformers 5.5.4   accelerate 1.14.0   gymnasium 1.3.0
```

## The traps, in the order you hit them

1. **lerobot 0.6.1 requires Python ≥ 3.12.** A 3.11 venv resolves to an ancient version.
2. **`uv pip install "lerobot[libero]"` silently does nothing** if lerobot is already
   installed — uv "audits" the extra away (`Audited 1 package`). Install `hf-libero` by name.
3. **`hf-egl-probe` builds a C extension and needs `cmake` on `PATH`.** lerobot already
   pip-installs one into `.venv/bin`; just prepend it.
4. **`robomimic==0.2.0` depends on the original `egl-probe`, which cannot build on macOS**
   (EGL is Linux-only). Install `robomimic --no-deps` and hand-install the rest. Same trap
   as `../roomsim`; note HF's `hf-egl-probe` fork *does* build, it's the transitive
   original that breaks.
5. **LIBERO blocks forever on an interactive `input()`** the first time it's imported with
   no `~/.libero/config.yaml`. Under an agent or CI it looks like a hang at 0 % CPU.
   Fix: `echo "N" | python -c "import libero.libero"`.
6. **`MUJOCO_GL=cgl`, not `egl`.**
7. **`smolvla_base` expects cameras named `camera1/2/3`; LIBERO datasets use
   `image`/`image2`.** Without `--rename_map` the run dies on a feature-mismatch assert.
   The same map is needed at eval time, because the env uses the dataset's names too.
8. **`--policy.push_to_hub=false`** or training refuses to start ("'repo_id' argument missing").
9. **`--dataset.streaming=true` yields empty batches** — `ValueError: Batch does not contain
   any data (None)` on the first step. Don't use it to avoid the 35 GB download.
10. **`--dataset.episodes='[0,1,2,3]'` crashes the sampler** with
    `KeyError` in `_absolute_to_relative`. Episode subsetting is broken in 0.6.1; you have
    to take the whole dataset.
11. **`eval.n_episodes` is per *task*, not total.** `N=2` on a 10-task suite runs 20 episodes.
12. **Rendered frames are 180°-rotated** (text reads mirrored) by `LiberoProcessorStep`.
    That is deliberate — it matches the `HuggingFaceVLA/libero` storage convention, so
    training and eval agree. It only looks wrong to a human.

## Dataset

`HuggingFaceVLA/libero` — 1693 episodes, 273 465 frames, 10 fps, 40 tasks, **35 GB**.
Feature names line up with LeRobot's LIBERO env out of the box (`observation.images.image`,
`observation.images.image2`, `observation.state` (8,), `action` (7,)), which the per-suite
`lerobot/libero_*_image` datasets do not — those name the wrist camera `wrist_image`.

Set `HF_TOKEN` before downloading; unauthenticated pulls are rate-limited to ~6 MB/s
(≈80 min for this dataset).

## Run it

```bash
./setup_mac.sh
N=2 ./eval_libero.sh                       # baseline control, ~2.5 min, expect 0.0%
DEVICE=cuda BATCH=64 STEPS=20000 ./train_libero.sh
POLICY=outputs/smolvla_libero/checkpoints/last/pretrained_model ./eval_libero.sh
```

SmolVLA fine-tunes `train_expert_only=True` with a frozen vision encoder: 100 M of 450 M
parameters actually get gradients. That is why this is a ~$10–30 job and not a ~$400 one.

## One thing that looks broken but isn't

The dataset advertises `fps: 10.0` while `LiberoEnv` runs robosuite at `control_freq=20`.
That mismatch would normally wreck a delta-action policy. It doesn't here: the whole
pipeline is **step-indexed, not time-indexed** — one dataset frame is one env step — and
LeRobot calibrated the episode budget against these exact demos:

```python
TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 280,  # longest training demo has 193 steps
    "libero_object":  280,  # longest training demo has 254 steps
    "libero_goal":    300,  # longest training demo has 270 steps
    "libero_10":      520,  # longest training demo has 505 steps
}
```

So the `fps` field is nominal metadata. Don't "fix" it by setting `--env.fps=10`.

## Cloud side: RunPod A100 (verified 2026-08-21)

Training runs on a rented A100 80 GB; evaluation stays on the Mac. The pod therefore
needs **no LIBERO, no robosuite, no EGL** — just `lerobot[smolvla,training]`.

**Measured on 1× A100 80 GB PCIe ($1.39/hr):**

| | |
|---|---|
| Dataset pull (35 GB, datacenter link) | ~3 min @ ~220 MB/s |
| Throughput, tuned | 1.05 steps/s @ batch 64 (~68 smp/s) |
| 20 000 steps | **~5.2 h ≈ $7.30** |
| Peak VRAM | **15.3 GB of 80 GB** |
| GPU utilization | 86 % |

### RunPod traps

1. **`runpodctl pod create --ssh` does not publish port 22.** It only injects your public
   key, so `--wait` sits on "ssh port not allocated yet" until it times out — while the pod
   bills. Pass `--ports 22/tcp` at create time, or fix after the fact with
   `runpodctl pod update <id> --ports 22/tcp`.
2. **`ulimit -n` is 1024 in the base image.** DataLoader workers die with
   `terminate called without an active exception` / `worker ... killed by signal: Aborted`.
   This is *not* `/dev/shm` (that's 55 GB here). `ulimit -n 65536` before training.
3. **PEP 668** blocks pip: set `PIP_BREAK_SYSTEM_PACKAGES=1 UV_BREAK_SYSTEM_PACKAGES=1`.
   Do this rather than building a venv — the image's torch is already the cu128 build, and
   lerobot's `torch>=2.7,<2.12` accepts it, so the resolver leaves CUDA torch alone.
4. **torchcodec fails to load and falls back to pyav.** Harmless here: this dataset stores
   images, not video.

### The dataloader is the bottleneck, not the GPU

Out of the box the A100 sits idle. Watch `data_s` against `updt_s` in the log:

| config | `updt_s` (GPU) | `data_s` (input) | steps/s | 20k steps |
|---|---|---|---|---|
| `--num_workers=8` | 0.78 | **1.1 – 3.6** | ~0.45 | ~12 h ≈ $17 |
| `--num_workers=32 --prefetch_factor=6 --persistent_workers=true` | 0.93 | **0.009** | 1.05 | ~5.2 h ≈ $7 |

Same GPU, same batch, **2.3× cheaper**. The tell is stalls arriving every `num_workers`
steps. The pod has 252 vCPUs; using 8 of them is the whole problem.

### Don't turn on AMP here

`SmolVLAConfig` has no `dtype` field, so `--policy.use_amp=true` falls through to
`torch.get_autocast_dtype("cuda")`, which is **float16** — not bf16. fp16 through a VLM plus
a flow-matching action expert risks overflow, and a silently degraded 5-hour run costs far
more than the ~$3 saved. Left at fp32.

## Evaluating the fine-tuned policy

Evaluation runs **on the Mac** — no GPU, no dataset, just the simulator and a checkpoint.

```bash
./fetch_checkpoint.sh root@<pod-ip>:/workspace/vla-lab/outputs/smolvla_libero
POLICY=outputs/smolvla_libero/pretrained_model SUITE=libero_object N=20 ./eval_libero.sh
python review_rollouts.py outputs/eval_libero_object_pretrained_model
```

`lerobot-eval` writes `eval_info.json` plus one MP4 per episode under
`videos/<suite>_<task_id>/eval_episode_<n>.mp4`. `review_rollouts.py` turns that into
things you can actually look at:

| output | what it is |
|---|---|
| `summary.md` | per-task success table |
| `grid_failures.mp4` | up to 9 failed rollouts tiled, red-framed |
| `grid_successes.mp4` | successes tiled, green-framed |
| `contact_sheet.png` | one 6-frame filmstrip per task |

**Watch the failures, not the success rate.** A single percentage hides the thing you need:
*how* the policy is wrong. Wrong object (language grounding), grasp slips (control), reaches
then stalls (horizon), ignores the instruction entirely (conditioning). Those four have
completely different fixes, and only the video distinguishes them.

### Which suites, and what "complex" means

The fine-tune covers all 40 tasks, so any suite can be evaluated:

| suite | character | step budget | 20 eps/task |
|---|---|---|---|
| `libero_object` | pick named object → basket | 280 | ~23 min |
| `libero_spatial` | same object, spatial reference | 280 | ~23 min |
| `libero_goal` | fixed objects, varying goal | 300 | ~25 min |
| **`libero_10`** | **long-horizon, multi-step** | **520** | **~45 min** |

`libero_10` is the complex-task test — *"turn on the stove and put the moka pot on it"*,
*"put both the alphabet soup and the cream cheese box in the basket"*. Two chained subtasks,
nearly double the step budget, and the suite where partial competence shows up as "does the
first half, never the second."

### Read the number honestly

`n=20` per task is the community convention, but 20 Bernoulli trials at p≈0.5 carry a
standard error of ~11 points. **A 55% vs 60% difference between two checkpoints is noise.**
Treat per-task rates as indicative and the 200-episode suite aggregate as the real number.
This is the same trap the field is loud about — see Levine's *"80% is not that good"*.
