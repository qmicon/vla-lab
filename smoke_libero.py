import os, time, numpy as np
os.environ.setdefault("MUJOCO_GL", "cgl")          # macOS: CGL, not EGL
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bd = benchmark.get_benchmark_dict()
print("suites:", sorted(bd.keys()))

suite = bd["libero_object"]()
print("tasks in libero_object:", suite.n_tasks)
task = suite.get_task(0)
print("task 0 language:", repr(task.language))

bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)
obs = env.reset()
print("obs keys:", sorted(obs.keys()))
print("agentview_image:", obs["agentview_image"].shape, obs["agentview_image"].dtype)

t0 = time.time()
N = 100
for _ in range(N):
    obs, rew, done, info = env.step(np.zeros(7))
dt = time.time() - t0
print(f"{N} steps in {dt:.2f}s -> {N/dt:.1f} steps/s (with 2x256px offscreen render)")

import imageio
imageio.imwrite("libero_smoke.png", obs["agentview_image"][::-1])
print("wrote libero_smoke.png")
env.close()
