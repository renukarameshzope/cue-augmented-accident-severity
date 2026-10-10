"""Hold-out evaluation, grouped k-fold cross-validation, baselines and cue ablation.

All splits are stratified AND grouped by ``group_id`` so frames of one video
never appear in both training and test data.

Outputs (in --out):
  holdout_metrics.json, holdout_confusion.csv, holdout_predictions.csv
  cv_folds.csv, cv_summary.json, cv_curves.npz
  ablation.csv            (one row per variant, with McNemar p-value vs full model)

Example:
    python scripts/run_experiments.py --manifest data/manifest_with_cues.csv \
        --image-root data/images --config configs/default.yaml --out results/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import _path  # noqa: F401
from casev.config import load_config
from casev.cues.scoring import CUE_NAMES
from casev.data import (
    SeverityDataset,
    assert_no_group_leakage,
    build_transforms,
    grouped_holdout_split,
    grouped_kfold,
    load_manifest,
)
from casev.evaluate import (
    bootstrap_ci,
    classification_metrics,
    confusion,
    curves,
    mcnemar,
    severe_vs_nonsevere_auc,
    summarize_folds,
)
from casev.model import CLASS_NAMES, count_parameters, severity_score
from casev.train import predict, train_model


def make_ds(df, idx, root, cfg, train):
    return SeverityDataset(df.iloc[idx], root, build_transforms(cfg.image_size, train), load_images=cfg.use_image)


@torch.no_grad()
def latency_ms(model, cfg, n=30):
    model.eval()
    img = torch.randn(1, 3, cfg.image_size, cfg.image_size, device=cfg.device) if cfg.use_image else None
    cues = torch.rand(1, len(CUE_NAMES), device=cfg.device)
    for _ in range(5):
        model(img, cues)
    if cfg.device.startswith("cuda"):
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        model(img, cues)
    if cfg.device.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def run_holdout(df, root, cfg, out: Path, log):
    tr, te = grouped_holdout_split(df, test_size=0.4, seed=cfg.seed)
    assert_no_group_leakage(df, tr, te)
    log(f"hold-out: {len(tr)} train / {len(te)} test images; test class counts "
        f"{np.bincount(df.iloc[te]['label'], minlength=3).tolist()}")
    model = train_model(cfg, make_ds(df, tr, root, cfg, True), log=log)
    torch.save(model.state_dict(), out / "model.pt")
    res = predict(model, DataLoader(make_ds(df, te, root, cfg, False), batch_size=cfg.batch_size), cfg.device)
    y, p = res["y_true"], res["proba"]
    groups = df.iloc[te]["group_id"].to_numpy()
    metrics = classification_metrics(y, p)
    metrics["severe_vs_nonsevere"] = severe_vs_nonsevere_auc(y, p)
    metrics["macro_f1_95ci_grouped"] = bootstrap_ci(y, p, "macro_f1", groups=groups, seed=cfg.seed)
    metrics["accuracy_95ci_grouped"] = bootstrap_ci(y, p, "accuracy", groups=groups, seed=cfg.seed)
    (out / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    pd.DataFrame(confusion(y, p), index=[f"true_{c}" for c in CLASS_NAMES],
                 columns=[f"pred_{c}" for c in CLASS_NAMES]).to_csv(out / "holdout_confusion.csv")
    pred_df = df.iloc[te][["image_path", "group_id", "label", *CUE_NAMES]].copy()
    pred_df[[f"p_{c}" for c in CLASS_NAMES]] = p
    pred_df["pred"] = p.argmax(1)
    pred_df["severity_score"] = severity_score(torch.from_numpy(p)).numpy()
    pred_df.to_csv(out / "holdout_predictions.csv", index=False)
    log(f"hold-out accuracy {metrics['accuracy']:.3f}  macro F1 {metrics['macro_f1']:.3f}")
    return tr, te


def run_cv(df, root, cfg, k, out: Path, log):
    rows, all_curves = [], {}
    for fold, (tr, te) in enumerate(grouped_kfold(df, k=k, seed=cfg.seed), start=1):
        assert_no_group_leakage(df, tr, te)
        model = train_model(cfg, make_ds(df, tr, root, cfg, True), log=lambda *_: None)
        res = predict(model, DataLoader(make_ds(df, te, root, cfg, False), batch_size=cfg.batch_size), cfg.device)
        m = classification_metrics(res["y_true"], res["proba"])
        sv = severe_vs_nonsevere_auc(res["y_true"], res["proba"])
        row = {"fold": fold, "n_test": len(te), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
               "roc_auc_ovr_macro": m.get("roc_auc_ovr_macro", np.nan), "pr_auc_ovr_macro": m.get("pr_auc_ovr_macro", np.nan),
               "roc_auc_severe_vs_rest": sv["roc_auc"], "pr_auc_severe_vs_rest": sv["pr_auc"]}
        rows.append(row)
        for cls, c in curves(res["y_true"], res["proba"]).items():
            for key, arr in c.items():
                all_curves[f"fold{fold}_{cls}_{key}"] = arr
        log(f"fold {fold}/{k}: acc {row['accuracy']:.3f}  macro F1 {row['macro_f1']:.3f}  ROC-AUC {row['roc_auc_ovr_macro']:.3f}")
    pd.DataFrame(rows).to_csv(out / "cv_folds.csv", index=False)
    summary = summarize_folds([{k_: v for k_, v in r.items() if k_ not in ("fold", "n_test")} for r in rows])
    (out / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out / "cv_curves.npz", **all_curves)
    log("CV summary: " + json.dumps(summary))


def ablation_variants(base_cfg):
    from dataclasses import replace

    v = {
        "proposed_efficientnet_b1_plus_cues": replace(base_cfg),
        "efficientnet_b1_image_only": replace(base_cfg, use_cues=False),
        "resnet50_image_only": replace(base_cfg, backbone="resnet50", use_cues=False),
        "vit_b_16_image_only": replace(base_cfg, backbone="vit_b_16", use_cues=False, image_size=224),
        "cue_scores_only_mlp": replace(base_cfg, use_image=False),
    }
    for j, name in enumerate(CUE_NAMES):
        mask = [1.0] * len(CUE_NAMES)
        mask[j] = 0.0
        v[f"without_{name}"] = replace(base_cfg, cue_mask=mask)
    return v


def run_ablation(df, root, base_cfg, tr, te, seeds, out: Path, log, skip=()):
    from dataclasses import replace

    rows, full_preds = [], {}
    for name, cfg0 in ablation_variants(base_cfg).items():
        if any(s in name for s in skip):
            continue
        per_seed = []
        for s in seeds:
            cfg = replace(cfg0, seed=s)
            model = train_model(cfg, make_ds(df, tr, root, cfg, True), log=lambda *_: None)
            res = predict(model, DataLoader(make_ds(df, te, root, cfg, False), batch_size=cfg.batch_size), cfg.device)
            m = classification_metrics(res["y_true"], res["proba"])
            pred = res["proba"].argmax(1)
            if name.startswith("proposed"):
                full_preds[s] = pred
            per_seed.append((m, pred, res["y_true"]))
        accs = [m["accuracy"] for m, _, _ in per_seed]
        f1s = [m["macro_f1"] for m, _, _ in per_seed]
        aucs = [m.get("roc_auc_ovr_macro", np.nan) for m, _, _ in per_seed]
        row = {"variant": name, "accuracy_mean": np.mean(accs), "accuracy_sd": np.std(accs, ddof=1) if len(seeds) > 1 else 0.0,
               "macro_f1_mean": np.mean(f1s), "macro_f1_sd": np.std(f1s, ddof=1) if len(seeds) > 1 else 0.0,
               "roc_auc_mean": np.mean(aucs), "params": count_parameters(model), "latency_ms_batch1": latency_ms(model, cfg),
               "n_seeds": len(seeds)}
        if full_preds and not name.startswith("proposed"):
            s0 = seeds[0]
            row["mcnemar_p_vs_proposed_seed0"] = mcnemar(per_seed[0][2], full_preds[s0], per_seed[0][1])["p_value"]
        rows.append(row)
        log(f"{name:40s} acc {row['accuracy_mean']:.3f}  macro F1 {row['macro_f1_mean']:.3f}")
    df_out = pd.DataFrame(rows)
    full_f1 = df_out.loc[df_out.variant.str.startswith("proposed"), "macro_f1_mean"]
    if len(full_f1):
        df_out["delta_macro_f1_vs_proposed"] = df_out["macro_f1_mean"] - float(full_f1.iloc[0])
    df_out.to_csv(out / "ablation.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--steps", nargs="+", default=["holdout", "cv", "ablation"], choices=["holdout", "cv", "ablation"])
    ap.add_argument("--skip-variants", nargs="*", default=[], help="substring filter, e.g. vit resnet")
    a = ap.parse_args()

    cfg = load_config(a.config)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")

    def log(msg):
        print(msg); logf.write(msg + "\n"); logf.flush()

    df = load_manifest(a.manifest)
    log(f"{len(df)} images, {df['group_id'].nunique()} groups, class counts {np.bincount(df['label'], minlength=3).tolist()}")
    tr, te = grouped_holdout_split(df, 0.4, cfg.seed)
    if "holdout" in a.steps:
        tr, te = run_holdout(df, a.image_root, cfg, out, log)
    if "cv" in a.steps:
        run_cv(df, a.image_root, cfg, a.k, out, log)
    if "ablation" in a.steps:
        run_ablation(df, a.image_root, cfg, tr, te, a.seeds, out, log, skip=a.skip_variants)


if __name__ == "__main__":
    main()
