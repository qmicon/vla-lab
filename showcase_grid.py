#!/usr/bin/env python
"""Build a showcase grid of successful rollouts across maximally different tasks.

    python showcase_grid.py

Picks one success per distinct *behavior type* — knob turning, articulated pulls,
non-prehensile pushing, precision insertion, compound two-step — rather than nine
variations of pick-and-place, and tags the zero-shot one so the honest caveat
travels with the video.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "cgl")

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TILE = 420
BAR = 46
COLS = 3
OUT = Path("outputs/showcase_grid.mp4")
POSTER = Path("outputs/showcase_poster.png")

# (eval dir, suite, exact instruction, short behavior label, is_unseen)
PICKS = [
    ("eval_libero_goal_final", "libero_goal", "turn on the stove", "knob · no grasp", False),
    ("eval_libero_goal_final", "libero_goal", "open the middle drawer of the cabinet", "articulated pull", False),
    ("eval_libero_goal_final", "libero_goal", "push the plate to the front of the stove", "non-prehensile push", False),
    ("eval_libero_goal_final", "libero_goal", "put the wine bottle on the rack", "precision insert", False),
    ("eval_libero_goal_final", "libero_goal", "put the bowl on top of the cabinet", "place at height", False),
    ("eval_libero_10_final", "libero_10", "put the black bowl in the bottom drawer of the cabinet and close it", "2-step compound", False),
    ("eval_libero_10_final", "libero_10", "turn on the stove and put the moka pot on it", "2-step compound", False),
    ("eval_libero_10_final", "libero_10", "pick up the book and place it in the back compartment of the caddy", "precision insert", False),
    ("eval_libero_90_unseen", "libero_90", "pick up the alphabet soup and put it in the tray", "NEVER TRAINED ON", True),
]


def _font(size: int):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


F_MAIN, F_SUB = _font(17), _font(14)


def _wrap(draw, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if draw.textlength(t, font=font) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:2]


def _decorate(frame, instruction, label, unseen):
    im = Image.fromarray(frame).convert("RGB").resize((TILE, TILE))
    canvas = Image.new("RGB", (TILE, TILE + BAR), (17, 17, 19))
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    accent = (232, 150, 40) if unseen else (34, 168, 84)
    d.rectangle([0, 0, TILE - 1, TILE - 1], outline=accent, width=5)

    # Behavior tag as a pill ON the frame, so it can never collide with the
    # instruction — long instructions used to run straight through it.
    pad, tw = 7, d.textlength(label, font=F_SUB)
    d.rectangle([9, 9, 9 + tw + 2 * pad, 9 + 24], fill=accent)
    d.text((9 + pad, 13), label, fill=(15, 15, 15), font=F_SUB)

    # Bottom bar belongs entirely to the instruction.
    y = TILE + 5
    for ln in _wrap(d, instruction, F_MAIN, TILE - 16):
        d.text((8, y), ln, fill=(242, 242, 242), font=F_MAIN)
        y += 19
    return np.asarray(canvas)


def _find_success(eval_dir, suite, instruction):
    """Locate the video of one successful episode for an exact instruction."""
    from libero.libero import benchmark

    s = benchmark.get_benchmark_dict()[suite]()
    langs = {i: s.get_task(i).language for i in range(s.n_tasks)}
    info = json.loads(Path(f"outputs/{eval_dir}/eval_info.json").read_text())
    for e in info["per_task"]:
        if langs.get(e["task_id"]) != instruction:
            continue
        m = e["metrics"]
        for i, ok in enumerate(m["successes"]):
            if ok:
                p = Path(m["video_paths"][i])
                if not p.exists():
                    p = Path(f"outputs/{eval_dir}") / Path(*p.parts[1:])
                if p.exists():
                    return p
    return None


def main():
    clips = []
    for eval_dir, suite, instr, label, unseen in PICKS:
        p = _find_success(eval_dir, suite, instr)
        if p is None:
            print(f"  !! no success video for: {instr}")
            continue
        frames = iio.imread(p, plugin="pyav")
        clips.append([_decorate(f, instr, label, unseen) for f in frames])
        print(f"  ok  {len(frames):4d} frames  {instr[:58]}")

    if not clips:
        raise SystemExit("no clips")

    rows = (len(clips) + COLS - 1) // COLS
    longest = max(len(c) for c in clips)
    cell_h = TILE + BAR
    blank = np.full((cell_h, TILE, 3), 17, np.uint8)

    out = []
    for t in range(longest):
        cells = [c[min(t, len(c) - 1)] for c in clips]      # freeze short clips on last frame
        while len(cells) < rows * COLS:
            cells.append(blank)
        out.append(np.concatenate(
            [np.concatenate(cells[r * COLS:(r + 1) * COLS], axis=1) for r in range(rows)], axis=0))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(OUT, np.stack(out), fps=20, codec="libx264")
    iio.imwrite(POSTER, out[int(len(out) * 0.6)])
    print(f"\nwrote {OUT}  ({len(clips)} tasks, {longest} frames, {rows}x{COLS})")
    print(f"wrote {POSTER}")


if __name__ == "__main__":
    main()
