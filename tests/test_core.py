import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from casev.clearance import ClearanceEstimator, SynthConfig, diversion_decision, evaluate_holdout, generate, weighted_quantile
from casev.cues.red_region import red_region
from casev.cues.scoring import CUE_NAMES, level_to_score, tire_marks_to_score, value_to_level, weather_to_score
from casev.data import assert_no_group_leakage, grouped_holdout_split, grouped_kfold
from casev.evaluate import classification_metrics, mcnemar
from casev.model import CueFusionNet, score_to_class, severity_score


def test_scoring_tables():
    assert [level_to_score(i) for i in range(5)] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert weather_to_score("Clear") == 1.0 and weather_to_score("stormy") == 0.0
    assert weather_to_score("partly cloudy") == 0.78
    assert value_to_level(0, [1, 3, 6, 10]) == 0 and value_to_level(7, [1, 3, 6, 10]) == 3
    assert tire_marks_to_score(0, 0.05) == 0 and tire_marks_to_score(1, 0.05) == 1


def test_red_region_detects_red_blob_and_ignores_grey():
    img = np.full((100, 100, 3), 120, np.uint8)
    assert red_region(img).area_fraction == 0
    img[40:60, 40:60] = (0, 0, 220)  # BGR red
    assert red_region(img).area_fraction > 0.03


def test_model_variants_forward():
    x, c = torch.randn(2, 3, 64, 64), torch.rand(2, 6)
    for kw in [{}, {"use_cues": False}, {"cue_mask": [1, 0, 1, 1, 1, 1]}]:
        m = CueFusionNet(pretrained=False, n_frozen_blocks=0, **kw)
        assert m(x, c).shape == (2, 3)
    m = CueFusionNet(pretrained=False, use_image=False)
    assert m(None, c).shape == (2, 3)


def test_cue_mask_zeroes_cue():
    m = CueFusionNet(pretrained=False, use_image=False, cue_mask=[0, 1, 1, 1, 1, 1]).eval()
    c1, c2 = torch.rand(1, 6), torch.rand(1, 6)
    c2[0, 1:] = c1[0, 1:]
    assert torch.allclose(m(None, c1), m(None, c2))


def test_severity_score_and_thresholds():
    p = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    assert severity_score(p).tolist() == [0.0, 0.5, 1.0]
    assert [score_to_class(s) for s in (0.2, 0.5, 0.7)] == [0, 1, 2]


def test_group_splits_have_no_leakage():
    df = pd.DataFrame({"label": np.repeat([0, 1, 2], 40), "group_id": np.repeat(np.arange(30), 4)})
    tr, te = grouped_holdout_split(df, 0.4)
    assert_no_group_leakage(df, tr, te)
    for tr, te in grouped_kfold(df, k=5):
        assert_no_group_leakage(df, tr, te)


def test_metrics_and_mcnemar():
    y = np.array([0, 1, 2, 2])
    p = np.eye(3)[y] * 0.9 + 0.033
    m = classification_metrics(y, p)
    assert m["accuracy"] == 1 and m["macro_f1"] == 1
    assert mcnemar(y, y, y)["p_value"] == 1.0


def test_clearance_estimator():
    db = generate(SynthConfig(n_records=400, seed=1))
    assert len(db) == 400 and db["clearance_min"].min() >= 5
    e = ClearanceEstimator(db).estimate(18.72, 73.395, "tanker overturn", "high")
    assert e.lower_min <= e.median_min <= e.upper_min
    d = diversion_decision(e, "Khandala Ghat", threshold_min=10)
    assert d["divert"] is True
    r = evaluate_holdout(db, 0.2, 1)
    assert r["mae_estimator_min"] < r["mae_segment_mean_min"]


def test_weighted_quantile():
    assert weighted_quantile(np.array([1.0, 2, 3]), np.ones(3), 0.5) == pytest.approx(2.0)
    assert weighted_quantile(np.array([1.0, 2, 3]), np.array([0.01, 0.01, 10]), 0.5) > 2.5
