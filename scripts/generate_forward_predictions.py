"""
Generate forward predictions for 2024 — unverified (no ground-truth labels yet).

These rows have complete monthly features but fall outside the 12-month forward
window needed to construct verified transition labels. They represent the most
current model signal available to an investor.

Outputs:
  data/processed/model_predictions_forward_2024.csv
  data/processed/model_predictions_multiclass_forward_2024.csv
"""
import json, pathlib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

BASE = pathlib.Path(__file__).parent.parent
DATA = BASE / "data" / "processed"

# ── Load inputs ────────────────────────────────────────────────────────────────
print("Loading data ...")
features_df = pd.read_csv(DATA / "features_engineered.csv")
features_df["month"] = pd.to_datetime(features_df["month"])
features_df["zip"]   = features_df["zip"].astype(str).str.zfill(5)

model_ds = pd.read_csv(DATA / "modeling_dataset_with_clusters.csv")
model_ds["month"] = pd.to_datetime(model_ds["month"])
model_ds["zip"]   = model_ds["zip"].astype(str).str.zfill(5)

zip_labels = pd.read_csv(DATA / "zip_cluster_labels.csv", index_col=0)
zip_labels.index = zip_labels.index.astype(str).str.zfill(5)

with open(DATA / "model_comparison.json") as f:
    comp = json.load(f)

selected_monthly = comp["monthly"]["features"]
print(f"Model features: {len(selected_monthly)}")

# ── Forward data: 2024 rows with complete features ─────────────────────────────
labeled_pairs = set(zip(model_ds["zip"], model_ds["month"].astype(str)))
features_df["month_str"] = features_df["month"].astype(str)

fwd = features_df[
    (features_df["month"] >= "2024-01-01") &
    (~features_df.apply(lambda r: (r["zip"], r["month_str"]) in labeled_pairs, axis=1))
].copy()

# Only rows with complete model features
fwd_complete = fwd[fwd[selected_monthly].notna().all(axis=1)].copy()
print(f"2024 forward rows (complete features): {fwd_complete.shape}")

# Attach cluster labels
fwd_complete["cluster_name"] = fwd_complete["zip"].map(zip_labels["cluster_name"])
fwd_complete["cluster"]      = fwd_complete["zip"].map(zip_labels["cluster"])

missing_cluster = fwd_complete["cluster_name"].isna().sum()
if missing_cluster > 0:
    print(f"  Warning: {missing_cluster} rows have no cluster assignment (new ZIPs) — will be labeled 'unknown'")
    fwd_complete["cluster_name"] = fwd_complete["cluster_name"].fillna("unknown")
    fwd_complete["cluster"]      = fwd_complete["cluster"].fillna(-1)

# ── Train final model (same as run_monthly_model.py Fold C) ───────────────────
print("Training final GB model on Jun 2019–Dec 2022 ...")
TARGET_BIN   = "transition_next_12m"
TARGET_MULTI = "transition_direction"

train_mask = (model_ds["month"] >= "2019-06-01") & (model_ds["month"] <= "2022-12-31")

X_train = model_ds.loc[train_mask, selected_monthly].values
y_train = model_ds.loc[train_mask, TARGET_BIN].values

gb_final = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, min_samples_leaf=20, random_state=42
)
gb_final.fit(X_train, y_train)
print("  Binary model trained.")

# Multi-class
y_train_mc = model_ds.loc[train_mask, TARGET_MULTI].values.astype(int)
gb_multi = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, min_samples_leaf=20, random_state=42
)
gb_multi.fit(X_train, y_train_mc)
print("  Multi-class model trained.")

# ── Score 2024 forward rows ────────────────────────────────────────────────────
X_fwd = fwd_complete[selected_monthly].values

prob_fwd = gb_final.predict_proba(X_fwd)[:, 1]

class_order = gb_multi.classes_
proba_mc    = gb_multi.predict_proba(X_fwd)
idx_d = list(class_order).index(-1)
idx_s = list(class_order).index(0)
idx_u = list(class_order).index(1)

# ── Save binary predictions ────────────────────────────────────────────────────
out_bin = fwd_complete[["zip", "month", "cluster", "cluster_name"]].copy().reset_index(drop=True)
out_bin["prob_transition"] = prob_fwd
out_bin["verified"] = False
out_bin.to_csv(DATA / "model_predictions_forward_2024.csv", index=False)
print(f"Saved model_predictions_forward_2024.csv  {out_bin.shape}")

# ── Save multi-class predictions ───────────────────────────────────────────────
out_mc = fwd_complete[["zip", "month", "cluster", "cluster_name"]].copy().reset_index(drop=True)
out_mc["prob_down"]      = proba_mc[:, idx_d]
out_mc["prob_stable"]    = proba_mc[:, idx_s]
out_mc["prob_up"]        = proba_mc[:, idx_u]
out_mc["pred_direction"] = gb_multi.predict(X_fwd)
out_mc["net_direction"]  = out_mc["prob_up"] - out_mc["prob_down"]
out_mc["verified"] = False
out_mc.to_csv(DATA / "model_predictions_multiclass_forward_2024.csv", index=False)
print(f"Saved model_predictions_multiclass_forward_2024.csv  {out_mc.shape}")

print("\nDone. Summary:")
print(f"  Period:   {out_bin['month'].min()} → {out_bin['month'].max()}")
print(f"  ZIPs:     {out_bin['zip'].nunique()}")
print(f"  Avg opportunity score: {prob_fwd.mean():.3f}")
print(f"  % with prob > 0.6:     {(prob_fwd > 0.6).mean()*100:.1f}%")
