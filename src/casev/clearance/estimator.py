"""Location-aware clearance-time estimator and diversion decision (paper Section 4.7).

For a new incident the estimator
  1. retrieves past incidents within ``radius_km`` of the GPS location
     (widening the radius until at least ``min_neighbours`` are found),
  2. weights each one by spatial proximity and similarity in incident type
     and severity,
  3. returns the similarity-weighted median clearance time and an empirical
     interval (weighted quantiles),
and recommends diversion when the estimate exceeds a threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

SEV_INDEX = {"low": 0, "moderate": 1, "high": 2}
EARTH_R_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return float(np.interp(q, cw, v))


@dataclass
class EstimatorConfig:
    radius_km: float = 5.0
    min_neighbours: int = 15
    max_radius_km: float = 40.0
    distance_scale_km: float = 3.0   # exp(-d / scale)
    type_mismatch_weight: float = 0.2
    severity_step_weight: float = 0.3  # weight multiplier per severity level of difference
    interval: tuple = (0.1, 0.9)


@dataclass
class ClearanceEstimate:
    median_min: float
    lower_min: float
    upper_min: float
    n_neighbours: int
    radius_used_km: float


class ClearanceEstimator:
    def __init__(self, records: pd.DataFrame, cfg: EstimatorConfig = EstimatorConfig()):
        need = {"lat", "lon", "incident_type", "severity", "clearance_min"}
        if need - set(records.columns):
            raise ValueError(f"records missing {need - set(records.columns)}")
        self.db = records.reset_index(drop=True)
        self.cfg = cfg
        self._lat = self.db["lat"].to_numpy()
        self._lon = self.db["lon"].to_numpy()
        self._sev = self.db["severity"].map(SEV_INDEX).to_numpy()
        self._type = self.db["incident_type"].to_numpy()
        self._y = self.db["clearance_min"].to_numpy(dtype=float)

    def estimate(self, lat: float, lon: float, incident_type: str, severity: str) -> ClearanceEstimate:
        c = self.cfg
        d = haversine_km(lat, lon, self._lat, self._lon)
        r = c.radius_km
        mask = d <= r
        while mask.sum() < c.min_neighbours and r < c.max_radius_km:
            r = min(c.max_radius_km, r * 1.5)
            mask = d <= r
        if mask.sum() == 0:
            raise ValueError("no past incidents found within max_radius_km")
        w = np.exp(-d[mask] / c.distance_scale_km)
        w *= np.where(self._type[mask] == incident_type, 1.0, c.type_mismatch_weight)
        w *= c.severity_step_weight ** np.abs(self._sev[mask] - SEV_INDEX[severity])
        y = self._y[mask]
        lo, hi = c.interval
        return ClearanceEstimate(
            median_min=weighted_quantile(y, w, 0.5),
            lower_min=weighted_quantile(y, w, lo),
            upper_min=weighted_quantile(y, w, hi),
            n_neighbours=int(mask.sum()),
            radius_used_km=float(r),
        )


class SegmentMeanBaseline:
    """Baseline: historical mean clearance time of the same road segment."""

    def __init__(self, records: pd.DataFrame):
        self.means = records.groupby("segment")["clearance_min"].mean().to_dict()
        self.global_mean = float(records["clearance_min"].mean())

    def estimate(self, segment: str) -> float:
        return float(self.means.get(segment, self.global_mean))


# Alternate routes keyed by segment (demonstration only)
ALTERNATE_ROUTES: Dict[str, str] = {
    "default": "Old Mumbai–Pune Highway (NH-48)",
}


def diversion_decision(est: ClearanceEstimate, segment: str, threshold_min: float = 60.0,
                       use_upper_bound: bool = False) -> Dict[str, object]:
    """Recommend diversion when the (median or upper-bound) estimate exceeds the threshold.

    Returns a plain dict. Formatting it as a V2X message (e.g. ETSI DENM) is
    outside the scope of this reference implementation.
    """
    value = est.upper_min if use_upper_bound else est.median_min
    divert = value > threshold_min
    return {
        "segment": segment,
        "estimated_clearance_min": round(est.median_min, 1),
        "interval_min": [round(est.lower_min, 1), round(est.upper_min, 1)],
        "threshold_min": threshold_min,
        "divert": bool(divert),
        "alternate_route": ALTERNATE_ROUTES.get(segment, ALTERNATE_ROUTES["default"]) if divert else None,
    }


def evaluate_holdout(records: pd.DataFrame, test_frac: float = 0.2, seed: int = 42,
                     cfg: Optional[EstimatorConfig] = None) -> Dict[str, float]:
    """MAE of the estimator vs the segment-mean baseline on held-out records."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))
    n_test = int(len(records) * test_frac)
    test, train = records.iloc[idx[:n_test]], records.iloc[idx[n_test:]]
    est = ClearanceEstimator(train, cfg or EstimatorConfig())
    base = SegmentMeanBaseline(train)
    pred, lo, hi, bpred = [], [], [], []
    for row in test.itertuples():
        e = est.estimate(row.lat, row.lon, row.incident_type, row.severity)
        pred.append(e.median_min); lo.append(e.lower_min); hi.append(e.upper_min)
        bpred.append(base.estimate(row.segment))
    y = test["clearance_min"].to_numpy()
    pred, lo, hi, bpred = map(np.asarray, (pred, lo, hi, bpred))
    return {
        "n_train": len(train),
        "n_test": len(test),
        "mae_estimator_min": float(np.mean(np.abs(pred - y))),
        "median_ae_estimator_min": float(np.median(np.abs(pred - y))),
        "mae_segment_mean_min": float(np.mean(np.abs(bpred - y))),
        "interval_coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_clearance_min": float(y.mean()),
    }
