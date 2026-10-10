"""Generate the synthetic Mumbai–Pune scenario database, evaluate the clearance-time
estimator against the segment-mean baseline, and show one diversion decision.

    python scripts/run_clearance_demo.py --out results/clearance
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import _path  # noqa: F401
from casev.clearance import (
    ClearanceEstimator,
    EstimatorConfig,
    SynthConfig,
    diversion_decision,
    evaluate_holdout,
    generate,
    load_seed_records,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/clearance")
    ap.add_argument("--n", type=int, default=3100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=60.0, help="diversion threshold in minutes")
    ap.add_argument("--seed-records", default="data/seed/seed_incidents.csv")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    db = generate(SynthConfig(n_records=a.n, seed=a.seed))
    db.to_csv(out / "synthetic_incidents.csv", index=False)
    print(f"generated {len(db)} synthetic records (seed {a.seed})")

    metrics = evaluate_holdout(db, test_frac=0.2, seed=a.seed)
    (out / "clearance_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    # Example query: one seed record from public reports, estimated from the synthetic database only
    seed = load_seed_records(a.seed_records).iloc[0]
    est = ClearanceEstimator(db, EstimatorConfig()).estimate(seed.lat, seed.lon, seed.incident_type, seed.severity)
    decision = diversion_decision(est, seed.segment, threshold_min=a.threshold)
    decision["reported_clearance_min_public_account"] = float(seed.clearance_min)
    print(json.dumps(decision, indent=2))
    (out / "example_decision.json").write_text(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
