
from pathlib import Path
import pandas as pd
import numpy as np

input_csv = Path(
    r"D:\PhDWork\cue-augmented-accident-severity"
    r"\cue-augmented-accident-severity\data\manifests"
    r"\dataset_manifest.csv"
)

output_csv = input_csv.with_name("dataset_manifest_demo_labels.csv")

df = pd.read_csv(input_csv)

# Demo labels only — NOT real severity ground truth
ratios = {"LOW": 0.35, "Moderate": 0.35, "High": 0.30}

n = len(df)
n_low = round(n * ratios["LOW"])
n_moderate = round(n * ratios["Moderate"])
n_high = n - n_low - n_moderate

labels = (
    ["LOW"] * n_low
    + ["Moderate"] * n_moderate
    + ["High"] * n_high
)

rng = np.random.default_rng(42)
rng.shuffle(labels)

df["severity_label_demo"] = labels
df["severity_label_is_synthetic"] = True

df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"Rows: {n}")
print(df["severity_label_demo"].value_counts())
print("\nActual proportions:")
print(df["severity_label_demo"].value_counts(normalize=True).round(3))
print(f"\nSaved demo CSV: {output_csv}")
