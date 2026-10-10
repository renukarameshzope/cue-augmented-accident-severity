# Cue-Augmented Accident Severity Estimation

Reference implementation of the method in

> R. Zope, S. Bhoite. *Cue-Augmented Deep Learning for Interpretable Accident Severity Estimation, with a Location-Aware Clearance-Time Module for Proactive Route Diversion.* (under review)

A single scene image is turned into six interpretable **cue scores** (tire marks, debris, scattered objects, traffic load, weather, red-region injury cue). These are concatenated with an **EfficientNet-B1** embedding and classified into **low / moderate / high** severity. Every prediction therefore comes with the visible evidence behind it. A separate **location-aware clearance-time module** retrieves similar past incidents near the GPS position, estimates how long the road will stay blocked, and recommends diversion when the estimate exceeds a threshold.

```
image ──► cue detectors (YOLOv8, tracker, weather CNN, red-region heuristic) ──► 6 cue scores ─┐
   └────► EfficientNet-B1 embedding ───────────────────────────────────────────────────────────┴► FC ► low / moderate / high
severity + GPS ──► similar-incident retrieval ──► weighted-median clearance time ──► divert? ──► alternate route
```

> **Scope of this repository.** This is an independent, clean reference implementation written from the method description in the paper. It is **not** the code used to produce the numbers in the paper, and it ships **no trained weights and no images**. The red-region cue is a colour heuristic, and the clearance-time module is demonstrated on **synthetic** data only (see [Limitations](#limitations)).

## Repository layout

| Path | Paper section | What it does |
|---|---|---|
| `src/casev/cues/scoring.py` | 4.4, Table 2 | Maps detector outputs to normalised scores in [0, 1] |
| `src/casev/cues/red_region.py` | 4.3 | Red-region heuristic (channel dominance + morphology) |
| `src/casev/cues/detectors.py` | 4.3 | Runs all six cue detectors on an image or clip |
| `src/casev/cues/weather.py` | 4.3 | 7-class weather classifier |
| `src/casev/model.py` | 4.5, Table 3–4 | Fusion model, baselines, cue masking for ablation, scalar severity score |
| `src/casev/data.py` | 4.1, 4.6 | Dataset, **video-grouped** stratified splits, leakage check |
| `src/casev/train.py` | 4.5 | Training loop |
| `src/casev/evaluate.py` | 4.6, 5 | Metrics, ROC/PR curves, bootstrap CIs, McNemar test |
| `src/casev/explain.py` | 5.4, 6 | Cue permutation importance and per-image cue contributions |
| `src/casev/clearance/synth.py` | 4.7 | Synthetic Mumbai–Pune Expressway incident generator (seed 42) |
| `src/casev/clearance/estimator.py` | 4.7, 5.5 | Location-aware clearance estimator, segment-mean baseline, diversion decision |
| `scripts/run_experiments.py` | 5.1–5.3 | Hold-out, k-fold CV, baselines and ablation in one run |
| `scripts/run_clearance_demo.py` | 5.5 | Clearance-time demonstration |

## Installation

```bash
git clone https://github.com/<your-account>/cue-augmented-accident-severity.git
cd cue-augmented-accident-severity
pip install -r requirements.txt          # add `ultralytics` for the YOLO cue detectors
python -m pytest -q                      # unit tests
```

## Quick start (no data needed)

A toy generator draws cartoon road scenes so that every step can be run on a CPU in a few minutes. **Numbers from the toy set are meaningless and must not be reported.**

```bash
python scripts/make_toy_data.py --out data/toy
python scripts/run_experiments.py --manifest data/toy/manifest.csv --image-root data/toy \
       --config configs/toy.yaml --out results/toy --k 3 --seeds 42 --skip-variants vit resnet
python scripts/run_clearance_demo.py --out results/clearance
```

## Using your own data

1. **Collect images** from the public datasets listed in the paper (Table 1: DAD, A3D, CADP, UCF-Crime, RDD2022, etc.). Respect each dataset's licence; images are not redistributed here.
2. **Write a label file** with columns `image_path, label, group_id` (and optionally `source`).
   `group_id` must identify the source video or scene. All frames of one video share one `group_id`; the splitting code guarantees they never appear in both train and test.
3. **Train the cue detectors** (YOLOv8 for tire marks, debris and scattered objects; a weather classifier) on your own box and class annotations, and report their mAP / accuracy.
4. **Extract cues**
   ```bash
   python scripts/extract_cues.py --manifest labels.csv --image-root images/ \
          --tire w/tire.pt --debris w/debris.pt --objects w/objects.pt --weather w/weather.pth \
          --out manifest_with_cues.csv
   ```
   The level thresholds in `LevelThresholds` (`scoring.py`) are defaults; calibrate and report the values you use.
5. **Run all experiments**
   ```bash
   python scripts/run_experiments.py --manifest manifest_with_cues.csv --image-root images/ \
          --config configs/default.yaml --out results/ --k 10 --seeds 42 43 44
   ```

### Outputs

| File | Content |
|---|---|
| `holdout_metrics.json` | Accuracy, per-class and macro P/R/F1, one-vs-rest ROC-AUC and PR-AUC (with class prevalence as the PR chance level), severe-vs-non-severe AUC, video-level bootstrap 95% CIs |
| `holdout_confusion.csv`, `holdout_predictions.csv` | Confusion matrix; per-image probabilities, severity score and cue scores |
| `cv_folds.csv`, `cv_summary.json`, `cv_curves.npz` | Per-fold metrics, mean ± SD, ROC/PR curve points for every fold |
| `ablation.csv` | Proposed model, image-only EfficientNet-B1 / ResNet-50 / ViT-B/16, cue-only MLP, and each "without <cue>" variant; mean ± SD over seeds, parameters, batch-1 latency, McNemar p-value vs the proposed model |
| `model.pt` | Weights of the hold-out model (used by `scripts/predict.py`) |

Latency in `ablation.csv` is for the classifier only. End-to-end latency including the six cue detectors must be measured separately on the deployment hardware.

## Method details

**Cue scores** (Table 2). Debris, scattered objects, traffic load and the red-region cue use five ordinal levels → 0, 0.25, 0.5, 0.75, 1. Weather uses seven classes: clear 1.00, partly cloudy 0.78, cloudy 0.56, foggy 0.33, rainy 0.22, snowy 0.11, stormy 0.00. Tire marks are continuous: 0 = none, 1 = prominent.

**Fusion model.** EfficientNet-B1 (ImageNet, stem + first three stages frozen) → global embedding ⊕ six cue scores → Dropout → FC(256) → ReLU → Dropout → FC(3). Adam, learning rate 1e-4 with cosine decay, batch 32, 20 epochs, input 240 × 240 (`configs/default.yaml`).

**Scalar severity score.** `score = 0·P(low) + 0.5·P(moderate) + 1·P(high)`, mapped to classes by the thresholds 0.30 / 0.60 (Table 4). The reported class is the arg-max of the softmax; the scalar score is supplementary.

**Interpretability evidence.** `explain.cue_permutation_importance` measures the drop in macro F1 when one cue is shuffled across the test set; `explain.cue_occlusion` gives each cue's contribution to a single prediction.

**Clearance-time module.** Past incidents within 5 km are retrieved (radius widened until ≥ 15 are found). Each is weighted by `exp(-d/3 km) × type-match × 0.3^|severity difference|`. The estimate is the weighted median, with the weighted 10th–90th percentiles as the interval. Diversion is recommended when the estimate exceeds the threshold (default 60 min). The baseline is the historical mean of the same road segment.

## Limitations

* **Red-region cue** is a colour heuristic. It also fires on red vehicles, brake lights and signs.
* **Clearance-time data are synthetic.** The estimator is evaluated on records produced by a known scenario model, so the results show that the module works as designed, not how accurate it is on real incidents. Segment coordinates are approximate. The three seed records come from public accounts, not official records.
* **V2X broadcast is not implemented.** `diversion_decision` returns a plain dictionary; formatting it as a standard V2X message (e.g. ETSI DENM) is left to the integrator.
* **No trained weights or images are included.**

## Citation

```bibtex
@unpublished{zope_cue_severity,
  author = {Zope, Renuka and Bhoite, Sachin},
  title  = {Cue-Augmented Deep Learning for Interpretable Accident Severity Estimation, with a Location-Aware Clearance-Time Module for Proactive Route Diversion},
  note   = {Under review},
  year   = {2026}
}
```

## Licence

The datasets referenced in the paper have their own licences.
