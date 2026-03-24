"""
Validate 2024 forward predictions against actual 2025 outcomes.

The 2024 forward predictions (model_predictions_forward_2024.csv) were generated
from Dec-2024 features using the validated monthly-only model. The 12-month forward
window for those predictions is Jan–Dec 2025. As of early 2026, Zillow covers all of
2025 and Redfin covers Jan–Nov 2025 (11/12 months).

Strategy
--------
- Rebuild the three composite-score signals (home_value_yoy_pct, inventory_mom_pct,
  pct_sold_above_list) from raw Zillow + Redfin data, extended through 2025.
- For each 2024 prediction row (month t), compute:
    forward_signal  = mean of signal[t+1 : t+12]   min_periods=11
    trailing_signal = mean of signal[t-11 : t]      (12-month trailing)
    delta = forward - trailing
- Z-score all three deltas, average them into a composite score.
- Apply fresh p25/p75 thresholds on the 2025 validation set (fair ranking-based eval).
- Join against forward predictions, compute AUC + F1 + per-cluster breakdown.

Outputs
-------
  data/processed/validation_2024_predictions.csv   — per-ZIP-month validation table
  data/processed/validation_summary.json           — overall + per-cluster metrics
"""

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

BASE = pathlib.Path(__file__).parent.parent
RAW  = BASE / "data" / "raw"
DATA = BASE / "data" / "processed"

# ── 1. Load model ZIPs and forward predictions ────────────────────────────────
print("Loading forward predictions ...")
fwd_bin = pd.read_csv(DATA / "model_predictions_forward_2024.csv")
fwd_bin["month"] = pd.to_datetime(fwd_bin["month"])
fwd_bin["zip"]   = fwd_bin["zip"].astype(str).str.zfill(5)

fwd_mc = pd.read_csv(DATA / "model_predictions_multiclass_forward_2024.csv")
fwd_mc["month"] = pd.to_datetime(fwd_mc["month"])
fwd_mc["zip"]   = fwd_mc["zip"].astype(str).str.zfill(5)

model_zips = set(fwd_bin["zip"].unique())
print(f"  {len(model_zips)} model ZIPs  |  {len(fwd_bin)} prediction rows")

# ── 2. Build Zillow home_value_yoy_pct through Dec 2025 ───────────────────────
print("Processing Zillow data ...")
zhvi = pd.read_csv(RAW / "zillow" / "zillow_zhvi_zip_monthly.csv")
zhvi["zip"] = zhvi["RegionName"].astype(str).str.zfill(5)
zhvi = zhvi[zhvi["zip"].isin(model_zips)].copy()

date_cols = [c for c in zhvi.columns if c[:4].isdigit()]
zhvi_long = zhvi[["zip"] + date_cols].melt(
    id_vars="zip", var_name="month", value_name="home_value_index"
)
zhvi_long["month"] = pd.to_datetime(zhvi_long["month"])
zhvi_long = zhvi_long.sort_values(["zip", "month"])

# home_value_yoy_pct = pct change vs 12 months ago
zhvi_long["home_value_yoy_pct"] = (
    zhvi_long.groupby("zip")["home_value_index"]
    .pct_change(12) * 100
)

zillow_signals = zhvi_long[["zip", "month", "home_value_yoy_pct"]].dropna()
print(f"  Zillow: {zillow_signals['month'].min().date()} → {zillow_signals['month'].max().date()}")

# ── 3. Build Redfin inventory_mom_pct and pct_sold_above_list through Nov 2025 ─
print("Processing Redfin data ...")
redfin_raw = pd.read_csv(
    RAW / "redfin" / "redfin_zip_market_tracker_zip.tsv000",
    sep="\t",
    usecols=["PERIOD_BEGIN", "REGION", "PROPERTY_TYPE", "INVENTORY", "SOLD_ABOVE_LIST"],
    low_memory=False,
)
# Keep only "All Residential" — one row per ZIP per month (same as original pipeline)
redfin_raw = redfin_raw[redfin_raw["PROPERTY_TYPE"] == "All Residential"].copy()
redfin_raw["zip"] = redfin_raw["REGION"].str.extract(r"(\d{5})")
redfin_raw["zip"] = redfin_raw["zip"].str.zfill(5)
redfin_raw = redfin_raw[redfin_raw["zip"].isin(model_zips)].copy()
redfin_raw["month"] = pd.to_datetime(redfin_raw["PERIOD_BEGIN"])
redfin_raw = redfin_raw.sort_values(["zip", "month"])

# inventory_mom_pct = MoM % change in inventory
redfin_raw["inventory_mom_pct"] = (
    redfin_raw.groupby("zip")["INVENTORY"]
    .pct_change(1) * 100
)

redfin_signals = redfin_raw[["zip", "month", "inventory_mom_pct", "SOLD_ABOVE_LIST"]].copy()
redfin_signals = redfin_signals.rename(columns={"SOLD_ABOVE_LIST": "pct_sold_above_list"})
# Redfin PERIOD_BEGIN is first-of-month; snap to end-of-month to align with Zillow
redfin_signals["month"] = redfin_signals["month"] + pd.offsets.MonthEnd(0)
redfin_signals = redfin_signals.dropna(subset=["pct_sold_above_list"])
print(f"  Redfin: {redfin_signals['month'].min().date()} → {redfin_signals['month'].max().date()}")

# ── 4. Merge three signals into one panel ─────────────────────────────────────
print("Merging signals ...")
signals = zillow_signals.merge(redfin_signals, on=["zip", "month"], how="outer")
signals = signals.sort_values(["zip", "month"]).reset_index(drop=True)
print(f"  Combined panel: {signals.shape}  |  {signals['zip'].nunique()} ZIPs")
print(f"  Date range: {signals['month'].min().date()} → {signals['month'].max().date()}")

# ── 5. Compute forward/trailing means for each 2024 prediction month ──────────
print("Computing forward/trailing windows ...")

SIGNAL_COLS = ["home_value_yoy_pct", "inventory_mom_pct", "pct_sold_above_list"]
MIN_PERIODS = 11   # allows Dec-2024 rows where Redfin ends Nov-2025 (11/12 months)

records = []
for zip_id, grp in signals.groupby("zip"):
    grp = grp.set_index("month").sort_index()

    for sig in SIGNAL_COLS:
        s = grp[sig]
        # forward mean: t+1 to t+12 (shift back 12, rolling 12)
        grp[f"fwd_{sig}"] = (
            s[::-1].rolling(12, min_periods=MIN_PERIODS).mean().shift(1)[::-1]
        )
        # trailing mean: t-11 to t (rolling 12, no shift)
        grp[f"trail_{sig}"] = s.rolling(12, min_periods=MIN_PERIODS).mean()

    grp["zip"] = zip_id
    records.append(grp.reset_index())

panel = pd.concat(records, ignore_index=True)

# Keep only 2024 months that appear in the forward predictions
pred_months = fwd_bin[["zip", "month"]].drop_duplicates()
panel = panel.merge(pred_months, on=["zip", "month"], how="inner")
print(f"  Panel after filtering to prediction months: {panel.shape}")

# ── 6. Compute delta, z-score, composite score ────────────────────────────────
print("Computing composite transition scores ...")

for sig in SIGNAL_COLS:
    panel[f"delta_{sig}"] = panel[f"fwd_{sig}"] - panel[f"trail_{sig}"]

delta_cols = [f"delta_{s}" for s in SIGNAL_COLS]
valid_mask = panel[delta_cols].notna().all(axis=1)
print(f"  Rows with valid deltas: {valid_mask.sum():,} / {len(panel):,}")

# Z-score computed on the validation set (fair: model ranks within period)
def zscore(s):
    return (s - s.mean()) / s.std()

valid_idx = panel[valid_mask].index
panel.loc[valid_idx, "z_price"]           =  zscore(panel.loc[valid_idx, "delta_home_value_yoy_pct"])
panel.loc[valid_idx, "z_inventory"]       = -zscore(panel.loc[valid_idx, "delta_inventory_mom_pct"])
panel.loc[valid_idx, "z_competitiveness"] =  zscore(panel.loc[valid_idx, "delta_pct_sold_above_list"])

panel.loc[valid_idx, "composite_score"] = (
    panel.loc[valid_idx, ["z_price", "z_inventory", "z_competitiveness"]].mean(axis=1)
)

# ── 7. Apply p25/p75 thresholds to get actual 2025 labels ─────────────────────
score_valid = panel.loc[valid_idx, "composite_score"]
lower_thresh = score_valid.quantile(0.25)
upper_thresh = score_valid.quantile(0.75)

print(f"  Thresholds — lower (p25): {lower_thresh:.4f}  upper (p75): {upper_thresh:.4f}")

panel["actual_transition"] = np.nan
panel.loc[valid_idx, "actual_transition"] = (
    (panel.loc[valid_idx, "composite_score"] >= upper_thresh).astype(int)
)

panel["actual_direction"] = np.nan
panel.loc[valid_idx, "actual_direction"] = 0
up_idx   = valid_idx[panel.loc[valid_idx, "composite_score"] >= upper_thresh]
down_idx = valid_idx[panel.loc[valid_idx, "composite_score"] <= lower_thresh]
panel.loc[up_idx,   "actual_direction"] =  1
panel.loc[down_idx, "actual_direction"] = -1

actual_counts = panel.loc[valid_idx, "actual_transition"].value_counts()
print(f"  Actual up: {actual_counts.get(1,0):,}  ({actual_counts.get(1,0)/valid_mask.sum():.1%})")

# ── 8. Join predictions and evaluate ─────────────────────────────────────────
print("Joining predictions and evaluating ...")

result = panel[["zip", "month", "composite_score", "actual_transition", "actual_direction"]].copy()
result = result.merge(
    fwd_bin[["zip", "month", "prob_transition", "cluster", "cluster_name"]],
    on=["zip", "month"], how="inner"
)
result = result.merge(
    fwd_mc[["zip", "month", "prob_down", "prob_stable", "prob_up", "net_direction"]],
    on=["zip", "month"], how="left"
)

eval_df = result.dropna(subset=["actual_transition", "composite_score"])
print(f"  Evaluable rows: {len(eval_df):,}  ZIPs: {eval_df['zip'].nunique()}")

# Binary metrics
y_true  = eval_df["actual_transition"].astype(int)
y_prob  = eval_df["prob_transition"]
y_pred  = (y_prob >= 0.5).astype(int)

auc     = roc_auc_score(y_true, y_prob)
f1_mac  = f1_score(y_true, y_pred, average="macro")
f1_bin  = f1_score(y_true, y_pred, average="binary")
acc     = (y_true == y_pred).mean()

print(f"\n{'='*50}")
print(f"  Binary validation (AUC)")
print(f"  AUC:          {auc:.4f}")
print(f"  Accuracy:     {acc:.4f}")
print(f"  F1 (binary):  {f1_bin:.4f}")
print(f"  F1 (macro):   {f1_mac:.4f}")
print(f"{'='*50}")
print()
print(classification_report(y_true, y_pred, target_names=["not_up", "up"]))

# Multi-class metrics (direction)
mc_df = eval_df.dropna(subset=["actual_direction", "prob_up"])
if len(mc_df) > 0:
    y_dir_true = mc_df["actual_direction"].astype(int)
    y_dir_pred = mc_df[["prob_down","prob_stable","prob_up"]].values
    classes    = [-1, 0, 1]
    y_dir_pred_label = [classes[np.argmax(row)] for row in y_dir_pred]
    f1_dir = f1_score(y_dir_true, y_dir_pred_label, average="macro")
    print(f"  Multi-class macro F1: {f1_dir:.4f}")
    print()
    print(classification_report(y_dir_true, y_dir_pred_label, target_names=["down","stable","up"]))

# ── 9. Per-cluster breakdown ──────────────────────────────────────────────────
print("\nPer-cluster AUC:")
cluster_metrics = {}
for cname, cdf in eval_df.groupby("cluster_name"):
    if cdf["actual_transition"].nunique() < 2:
        print(f"  {cname:40s}  (only one class — skipped)")
        continue
    c_auc = roc_auc_score(cdf["actual_transition"].astype(int), cdf["prob_transition"])
    c_acc = (cdf["actual_transition"].astype(int) == (cdf["prob_transition"] >= 0.5).astype(int)).mean()
    n     = len(cdf)
    print(f"  {cname:40s}  AUC={c_auc:.4f}  acc={c_acc:.3f}  n={n}")
    cluster_metrics[cname] = {"auc": round(c_auc, 4), "accuracy": round(c_acc, 4), "n": n}

# ── 10. Save results ──────────────────────────────────────────────────────────
print("\nSaving results ...")

out_cols = [
    "zip", "month", "cluster", "cluster_name",
    "prob_transition", "prob_down", "prob_stable", "prob_up", "net_direction",
    "composite_score", "actual_transition", "actual_direction",
]
result[out_cols].to_csv(DATA / "validation_2024_predictions.csv", index=False)
print(f"  Saved validation_2024_predictions.csv  {result.shape}")

summary = {
    "evaluable_rows":   len(eval_df),
    "evaluable_zips":   eval_df["zip"].nunique(),
    "data_coverage": {
        "zillow_through":  "2026-01-31",
        "redfin_through":  "2025-11-01",
        "min_periods_used": MIN_PERIODS,
    },
    "binary": {
        "auc":      round(auc, 4),
        "accuracy": round(acc, 4),
        "f1_binary": round(f1_bin, 4),
        "f1_macro":  round(f1_mac, 4),
    },
    "multiclass": {
        "f1_macro": round(f1_dir, 4) if len(mc_df) > 0 else None,
    },
    "thresholds": {
        "lower_p25": round(lower_thresh, 4),
        "upper_p75": round(upper_thresh, 4),
    },
    "per_cluster": cluster_metrics,
}

with open(DATA / "validation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("  Saved validation_summary.json")

print("\nDone.")
print(f"  Key result — AUC on 2025 actual outcomes: {auc:.4f}")
print(f"  (Training AUC on 2023 held-out data was 0.8652)")
