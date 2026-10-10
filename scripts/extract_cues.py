"""Run the six cue detectors over every image in a manifest and write the cue scores.

Input manifest needs: image_path, label, group_id. Output adds the six cue columns.

Example:
    python scripts/extract_cues.py --manifest data/labels.csv --image-root data/images \
        --tire weights/tire.pt --debris weights/debris.pt --objects weights/objects.pt \
        --weather weights/weather.pth --out data/manifest_with_cues.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

import _path  # noqa: F401
from casev.cues.detectors import CueConfig, CueExtractor
from casev.cues.scoring import CUE_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tire", default=None)
    ap.add_argument("--debris", default=None)
    ap.add_argument("--objects", default=None)
    ap.add_argument("--obstructive-ids", type=int, nargs="*", default=[])
    ap.add_argument("--vehicles", default="yolov8n.pt")
    ap.add_argument("--weather", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    ext = CueExtractor(CueConfig(a.tire, a.debris, a.objects, a.obstructive_ids, a.vehicles, a.weather), device=a.device)
    df = pd.read_csv(a.manifest)
    root = Path(a.image_root) if a.image_root else None
    scores = []
    for i, p in enumerate(df["image_path"]):
        path = root / p if root and not Path(p).is_absolute() else Path(p)
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        scores.append(ext(img))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(df)}")
    df[CUE_NAMES] = pd.DataFrame(scores)[CUE_NAMES].values
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
