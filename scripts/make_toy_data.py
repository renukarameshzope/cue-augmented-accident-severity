"""Generate a small synthetic image set so the full pipeline can be run without real data.

The images are cartoons (grey road, coloured boxes for vehicles, red blobs,
dark streaks). They exist only to check that the code runs end to end.
Numbers obtained on this toy set mean nothing and must not be reported.

The red-region cue is computed with the real heuristic; the other cue scores
are simulated from the generator's ground truth because no trained YOLO
weights are shipped.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import _path  # noqa: F401
from casev.cues.red_region import red_region
from casev.cues.scoring import LevelThresholds, level_to_score, value_to_level, WEATHER_SCORES

SEV = ["low", "moderate", "high"]


def draw_scene(rng, sev: int, size: int, base_seed: int, frame: int):
    r2 = np.random.default_rng(base_seed)  # scene layout shared by frames of one "video"
    img = np.full((size, size, 3), (90, 90, 90), np.uint8)
    cv2.rectangle(img, (0, 0), (size, size // 3), (200, 180, 150), -1)  # sky
    n_veh = int(r2.integers(1, 3 + 4 * sev))
    for _ in range(n_veh):
        x, y = r2.integers(0, size - 20), r2.integers(size // 3, size - 12)
        col = tuple(int(c) for c in r2.integers(40, 200, 3))
        cv2.rectangle(img, (int(x + frame), int(y)), (int(x + 18 + frame), int(y + 10)), col, -1)
    n_debris = int(r2.integers(0, 2 + 8 * sev))
    for _ in range(n_debris):
        x, y = r2.integers(0, size), r2.integers(size // 3, size)
        cv2.circle(img, (int(x), int(y)), int(r2.integers(1, 3)), (30, 30, 30), -1)
    tire = sev >= 1 and r2.random() < 0.3 + 0.3 * sev
    if tire:
        x = int(r2.integers(10, size - 10))
        cv2.line(img, (x, size), (x + 15, size // 2), (20, 20, 20), 2)
    n_red = int(r2.integers(0, 1 + 3 * (sev == 2)))
    for _ in range(n_red):
        x, y = r2.integers(10, size - 10), r2.integers(size // 2, size - 5)
        cv2.circle(img, (int(x), int(y)), int(r2.integers(2, 4 + 2 * sev)), (0, 0, 200), -1)
    noise = rng.normal(0, 6, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, dict(n_veh=n_veh, n_debris=n_debris, tire=tire)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/toy")
    ap.add_argument("--videos", type=int, default=90)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True)
    th = LevelThresholds()
    rows = []
    for v in range(a.videos):
        sev = v % 3
        weather = rng.choice(list(WEATHER_SCORES))
        for f in range(a.frames):
            img, gt = draw_scene(rng, sev, a.size, a.seed * 1000 + v, f)
            name = f"images/vid{v:03d}_f{f}.png"
            cv2.imwrite(str(out / name), img)
            rr = red_region(img, min_blob_frac=1e-4)
            red_level = 4 if rr.n_large_blobs >= 2 else value_to_level(rr.area_fraction, th.red_area)
            rows.append({
                "image_path": name,
                "label": SEV[sev],
                "group_id": f"vid{v:03d}",
                "source": "toy",
                "tire_marks": float(gt["tire"]) * float(rng.uniform(0.4, 1.0)),
                "debris": level_to_score(min(4, gt["n_debris"] // 3)),
                "scattered_objects": level_to_score(min(4, gt["n_debris"] // 4)),
                "traffic_load": level_to_score(value_to_level(gt["n_veh"], [1, 4, 7, 10])),
                "weather": WEATHER_SCORES[weather],
                "red_region": level_to_score(red_level),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out / "manifest.csv", index=False)
    print(f"wrote {len(df)} images from {a.videos} toy videos to {out}")


if __name__ == "__main__":
    main()
