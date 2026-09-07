#!/usr/bin/env python
"""Turn a lerobot-eval output directory into something you can actually watch.

    python review_rollouts.py outputs/eval_libero_object_smolvla_base

Produces, next to eval_info.json:
  summary.md          per-task success table
  grid_failures.mp4   tiled failures  — where the information is
  grid_successes.mp4  tiled successes — check they are real, not lucky
  contact_sheet.png   one filmstrip row per task, green/red framed

A success rate is a single number that hides everything. The failures tell you
*how* the policy is wrong: wrong object, grasp slips, reaches and stalls,
ignores the language. That distinction decides what you fix next.
"""

import argparse
import json
import math
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

TILE = 360
PAD = 4


def _label(img: np.ndarray, text: str, ok: bool) -> np.ndarray:
    """Draw a colored border + caption. Green = success, red = failure."""
    im = Image.fromarray(img).convert("RGB").resize((TILE, TILE))
    d = ImageDraw.Draw(im)
    color = (34, 168, 84) if ok else (208, 48, 48)
    d.rectangle([0, 0, TILE - 1, TILE - 1], outline=color, width=6)
    d.rectangle([0, TILE - 26, TILE, TILE], fill=color)
    d.text((8, TILE - 20), text[:52], fill=(255, 255, 255))
    return np.asarray(im)


def _language_lookup(suite_name: str) -> dict[int, str]:
    """task_id -> natural-language instruction, straight from the LIBERO suite.

    Worth the import: a tile captioned 'open the top drawer of the cabinet' is
    self-explanatory in a shared video; 'libero_90_12 ep0' is not.
    """
    try:
        import os

        os.environ.setdefault("MUJOCO_GL", "cgl")
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()[suite_name]()
        return {i: suite.get_task(i).language for i in range(suite.n_tasks)}
    except Exception:
        return {}


def _episodes(info: dict):
    """Flatten eval_info.json into (task_id, caption, success, path)."""
    out = []
    langs: dict[str, dict[int, str]] = {}
    for entry in info["per_task"]:
        m = entry["metrics"]
        group = entry["task_group"]
        if group not in langs:
            langs[group] = _language_lookup(group)
        tid = entry["task_id"]
        caption = langs[group].get(tid) or f"{group}_{tid}"
        vids = m.get("video_paths", [])
        succ = m.get("successes", [])
        for i, v in enumerate(vids):
            ok = bool(succ[i]) if i < len(succ) else False
            out.append((tid, caption, ok, Path(v)))
    return out


def _read(path: Path, stride: int = 1) -> np.ndarray:
    frames = iio.imread(path, plugin="pyav")
    return frames[::stride]


def build_grid(eps, out_path: Path, max_tiles: int = 9, fps: int = 20):
    """Tile up to max_tiles rollouts into one video, padding short ones by freezing."""
    eps = eps[:max_tiles]
    if not eps:
        return None
    clips = []
    for _, name, ok, path in eps:
        if not path.exists():
            continue
        f = _read(path)
        clips.append(([_label(x, name, ok) for x in f], name))
    if not clips:
        return None

    n = len(clips)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    longest = max(len(c) for c, _ in clips)

    canvas = []
    for t in range(longest):
        cells = []
        for c, _ in clips:
            # Freeze on the last frame so a short episode does not truncate the grid.
            cells.append(c[min(t, len(c) - 1)])
        while len(cells) < rows * cols:
            cells.append(np.zeros((TILE, TILE, 3), np.uint8))
        grid = np.concatenate(
            [np.concatenate(cells[r * cols : (r + 1) * cols], axis=1) for r in range(rows)], axis=0
        )
        canvas.append(grid)
    iio.imwrite(out_path, np.stack(canvas), fps=fps, codec="libx264")
    return out_path


def build_contact_sheet(eps, out_path: Path, per_task: int = 1, n_frames: int = 6):
    """One filmstrip row per task: n_frames sampled across the episode."""
    by_task: dict[int, list] = {}
    for e in eps:
        by_task.setdefault(e[0], []).append(e)
    rows = []
    for tid in sorted(by_task):
        for _, name, ok, path in by_task[tid][:per_task]:
            if not path.exists():
                continue
            f = _read(path)
            idx = np.linspace(0, len(f) - 1, n_frames).astype(int)
            rows.append(np.concatenate([_label(f[i], name if i == idx[0] else "", ok) for i in idx], axis=1))
    if not rows:
        return None
    iio.imwrite(out_path, np.concatenate(rows, axis=0))
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("--max-tiles", type=int, default=9)
    args = ap.parse_args()

    d = args.eval_dir
    info = json.loads((d / "eval_info.json").read_text())
    eps = _episodes(info)
    # video_paths in eval_info.json are relative to the cwd of the eval run
    eps = [(t, n, o, p if p.exists() else d / Path(*p.parts[1:])) for t, n, o, p in eps]

    wins = [e for e in eps if e[2]]
    losses = [e for e in eps if not e[2]]

    lines = ["# Rollout review", "", f"**{len(wins)}/{len(eps)} episodes succeeded "
             f"({100 * len(wins) / max(len(eps), 1):.1f}%)**", "", "| task | success rate | episodes |", "|---|---|---|"]
    by_task: dict[int, list] = {}
    for e in eps:
        by_task.setdefault(e[0], []).append(e)
    for tid in sorted(by_task):
        g = by_task[tid]
        k = sum(1 for x in g if x[2])
        lines.append(f"| {tid} | {100 * k / len(g):.0f}% | {k}/{len(g)} |")
    (d / "summary.md").write_text("\n".join(lines) + "\n")

    made = [d / "summary.md"]
    for tag, sel in (("failures", losses), ("successes", wins)):
        p = build_grid(sel, d / f"grid_{tag}.mp4", args.max_tiles)
        if p:
            made.append(p)
    p = build_contact_sheet(eps, d / "contact_sheet.png")
    if p:
        made.append(p)

    print(f"{len(wins)}/{len(eps)} succeeded ({100 * len(wins) / max(len(eps), 1):.1f}%)")
    for m in made:
        print("  wrote", m)


if __name__ == "__main__":
    main()
