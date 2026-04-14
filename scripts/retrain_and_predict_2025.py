"""
Retrain on Jun 2019 – Dec 2024 and generate forward predictions from Nov 2025 features.

Changes vs the original model:
  - Training extended by 2 years (2023 + 2024 market dynamics included)
  - 2024 transition labels come from validate_2024_predictions.py output
  - Feature point for forward predictions: Nov 2025 (last month with full Redfin coverage)
  - Forward window: Dec 2025 – Nov 2026 (unverifiable until late 2026)

The 18 RFECV-selected features are recomputed fresh from raw Zillow + Redfin data
for the full 2018–2025 panel to ensure consistency.

Outputs
-------
  data/processed/model_predictions_forward_2025.csv
  data/processed/model_predictions_multiclass_forward_2025.csv
  data/processed/model_comparison_v2.json   (updated with retrained metrics)
"""

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")

BASE = pathlib.Path(__file__).parent.parent
RAW  = BASE / "data" / "raw"
DATA = BASE / "data" / "processed"

FEATURE_MONTH = "2025-11-30"   # last month with full Redfin coverage

# ── Load model config ─────────────────────────────────────────────────────────
with open(DATA / "model_comparison.json") as f:
    comp = json.load(f)
SELECTED = comp["monthly"]["features"]
print(f"Model features ({len(SELECTED)}): {SELECTED}")

# ── 1. Build Zillow panel ─────────────────────────────────────────────────────
print("\nBuilding Zillow panel ...")
zhvi_wide = pd.read_csv(RAW / "zillow" / "zillow_zhvi_zip_monthly.csv")
zhvi_wide["zip"] = zhvi_wide["RegionName"].astype(str).str.zfill(5)

date_cols = [c for c in zhvi_wide.columns if c[:4].isdigit()]
zhvi = zhvi_wide[["zip"] + date_cols].melt(
    id_vars="zip", var_name="month", value_name="home_value_index"
)
zhvi["month"] = pd.to_datetime(zhvi["month"])
zhvi = zhvi.sort_values(["zip", "month"]).reset_index(drop=True)

# Derived Zillow features (within each ZIP)
zhvi["home_value_mom_pct"] = (
    zhvi.groupby("zip")["home_value_index"]
    .transform(lambda s: s.pct_change(1, fill_method=None) * 100)
)
zhvi["home_value_yoy_pct"] = (
    zhvi.groupby("zip")["home_value_index"]
    .transform(lambda s: s.pct_change(12, fill_method=None) * 100)
)

print(f"  Zillow: {zhvi['month'].min().date()} to {zhvi['month'].max().date()} | {zhvi['zip'].nunique()} ZIPs")

# ── 2. Build Redfin panel ─────────────────────────────────────────────────────
print("Building Redfin panel ...")
redfin_raw = pd.read_csv(
    RAW / "redfin" / "redfin_zip_market_tracker_zip.tsv000",
    sep="\t",
    usecols=[
        "PERIOD_BEGIN", "REGION", "PROPERTY_TYPE",
        "HOMES_SOLD", "NEW_LISTINGS", "INVENTORY", "MEDIAN_DOM",
        "OFF_MARKET_IN_TWO_WEEKS", "SOLD_ABOVE_LIST", "AVG_SALE_TO_LIST",
    ],
    low_memory=False,
)
redfin_raw = redfin_raw[redfin_raw["PROPERTY_TYPE"] == "All Residential"].copy()
redfin_raw["zip"]   = redfin_raw["REGION"].str.extract(r"(\d{5})").iloc[:, 0].str.zfill(5)
redfin_raw["month"] = pd.to_datetime(redfin_raw["PERIOD_BEGIN"]) + pd.offsets.MonthEnd(0)
redfin_raw = redfin_raw.rename(columns={
    "HOMES_SOLD":             "homes_sold",
    "NEW_LISTINGS":           "new_listings",
    "INVENTORY":              "inventory",
    "MEDIAN_DOM":             "median_days_on_market",
    "OFF_MARKET_IN_TWO_WEEKS":"pct_off_market_in_2wks",
    "SOLD_ABOVE_LIST":        "pct_sold_above_list",
    "AVG_SALE_TO_LIST":       "avg_sale_to_list_ratio",
})
redfin = redfin_raw[["zip","month","homes_sold","new_listings","inventory",
                      "median_days_on_market","pct_off_market_in_2wks",
                      "pct_sold_above_list","avg_sale_to_list_ratio"]].copy()
redfin = redfin.sort_values(["zip", "month"]).reset_index(drop=True)

# Derived Redfin features
redfin["inventory_mom_pct"] = (
    redfin.groupby("zip")["inventory"]
    .transform(lambda s: s.pct_change(1, fill_method=None) * 100)
)
redfin["inventory_yoy_pct"] = (
    redfin.groupby("zip")["inventory"]
    .transform(lambda s: s.pct_change(12, fill_method=None) * 100)
)

print(f"  Redfin: {redfin['month'].min().date()} to {redfin['month'].max().date()} | {redfin['zip'].nunique()} ZIPs")

# ── 3. Merge into a single panel ──────────────────────────────────────────────
print("Merging panels ...")
panel = zhvi.merge(redfin, on=["zip", "month"], how="inner")
panel = panel.sort_values(["zip", "month"]).reset_index(drop=True)
print(f"  Merged panel: {panel.shape} | {panel['zip'].nunique()} ZIPs")

# ── 4. Compute engineered features (within each ZIP) ─────────────────────────
print("Engineering features ...")

def zip_rolling(df, col, window, func="mean"):
    return (
        df.groupby("zip")[col]
        .transform(lambda s: getattr(s.rolling(window, min_periods=window), func)())
    )

def zip_lag(df, col, periods):
    return df.groupby("zip")[col].transform(lambda s: s.shift(periods))

def zip_relative_departure(df, col, window=12):
    rm = df.groupby("zip")[col].transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    return (df[col] / rm.replace(0, np.nan)) - 1

# Rolling means
panel["home_value_mom_3m_avg"]  = zip_rolling(panel, "home_value_mom_pct",  3)
panel["home_value_mom_12m_avg"] = zip_rolling(panel, "home_value_mom_pct",  12)
panel["inventory_mom_12m_avg"]  = zip_rolling(panel, "inventory_mom_pct",   12)

# Acceleration (3m avg minus 12m avg)
panel["home_value_accel"] = panel["home_value_mom_3m_avg"] - panel["home_value_mom_12m_avg"]

# ZIP-relative departures from trailing baseline
panel["home_value_vs_baseline"] = zip_relative_departure(panel, "home_value_index")
panel["dom_vs_baseline"]        = zip_relative_departure(panel, "median_days_on_market")

# Lag features
panel["home_value_mom_pct_lag6"]      = zip_lag(panel, "home_value_mom_pct",      6)
panel["pct_sold_above_list_lag6"]     = zip_lag(panel, "pct_sold_above_list",     6)
panel["avg_sale_to_list_ratio_lag6"]  = zip_lag(panel, "avg_sale_to_list_ratio",  6)
panel["home_value_vs_baseline_lag6"]  = zip_lag(panel, "home_value_vs_baseline",  6)

print(f"  Features ready. Panel: {panel.shape}")

# Verify all 18 selected features exist
missing_feats = [f for f in SELECTED if f not in panel.columns]
if missing_feats:
    raise ValueError(f"Missing features in panel: {missing_feats}")
print(f"  All {len(SELECTED)} selected features present.")

# ── 5. Load training labels ───────────────────────────────────────────────────
print("\nLoading training labels ...")

# Existing labels: Jun 2019 – Dec 2023 (from modeling_dataset_with_clusters.csv)
existing = pd.read_csv(DATA / "modeling_dataset_with_clusters.csv",
                       usecols=["zip","month","transition_next_12m","transition_direction",
                                "cluster","cluster_name"])
existing["month"] = pd.to_datetime(existing["month"])
existing["zip"]   = existing["zip"].astype(str).str.zfill(5)
print(f"  Existing labels: {len(existing):,} rows | {existing['month'].min().date()} to {existing['month'].max().date()}")

# 2024 labels from validation script
val = pd.read_csv(DATA / "validation_2024_predictions.csv",
                  usecols=["zip","month","actual_transition","actual_direction",
                            "cluster","cluster_name"])
val["month"] = pd.to_datetime(val["month"])
val["zip"]   = val["zip"].astype(str).str.zfill(5)
val = val.rename(columns={
    "actual_transition": "transition_next_12m",
    "actual_direction":  "transition_direction",
})
# Keep one row per zip+month (take first month within each zip — all 2024 months are valid)
val = val.dropna(subset=["transition_next_12m"])
print(f"  2024 labels: {len(val):,} rows | {val['month'].min().date()} to {val['month'].max().date()}")

# Combine: no overlap (existing ends Dec 2023, val starts Jan 2024)
all_labels = pd.concat([existing, val], ignore_index=True)
all_labels = all_labels.drop_duplicates(subset=["zip","month"])
print(f"  Combined labels: {len(all_labels):,} rows | {all_labels['month'].min().date()} to {all_labels['month'].max().date()}")

# ── 6. Build modeling dataset (features + labels) ─────────────────────────────
print("\nBuilding modeling dataset ...")
model_df = panel.merge(
    all_labels[["zip","month","transition_next_12m","transition_direction","cluster","cluster_name"]],
    on=["zip","month"], how="inner"
)

# Filter to Jun 2019 – Dec 2024 and rows with complete features + labels
train_mask = (
    (model_df["month"] >= "2019-06-01") &
    (model_df["month"] <= "2024-12-31") &
    model_df[SELECTED].notna().all(axis=1) &
    model_df["transition_next_12m"].notna()
)
train_df = model_df[train_mask].copy()
print(f"  Training rows: {len(train_df):,} | ZIPs: {train_df['zip'].nunique()} | "
      f"{train_df['month'].min().date()} to {train_df['month'].max().date()}")

binary_dist = train_df["transition_next_12m"].value_counts(normalize=True)
print(f"  Class split: up={binary_dist.get(1,0):.1%}  not-up={binary_dist.get(0,0):.1%}")

# ── 7. Walk-forward validation on new data (Fold D: train to Dec 2023, test 2024) ──
print("\nFold D validation (train Jun2019-Dec2023, test 2024) ...")
fold_d_train = train_df[train_df["month"] <= "2023-12-31"]
fold_d_test  = train_df[train_df["month"].dt.year == 2024]

X_d_train = fold_d_train[SELECTED].values
y_d_train = fold_d_train["transition_next_12m"].values.astype(int)
X_d_test  = fold_d_test[SELECTED].values
y_d_test  = fold_d_test["transition_next_12m"].values.astype(int)

gb_d = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, min_samples_leaf=20, random_state=42
)
gb_d.fit(X_d_train, y_d_train)
fold_d_auc = roc_auc_score(y_d_test, gb_d.predict_proba(X_d_test)[:, 1])
print(f"  Fold D AUC (test 2024): {fold_d_auc:.4f}")
print(f"  (Compare: Fold C test 2023 = 0.8652 | 2025 actuals = 0.7747)")

# ── 8. Train final model on full Jun 2019 – Dec 2024 ─────────────────────────
print("\nTraining final model on Jun 2019 – Dec 2024 ...")
X_train = train_df[SELECTED].values
y_train = train_df["transition_next_12m"].values.astype(int)
y_train_mc = train_df["transition_direction"].values.astype(int)

gb_final = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, min_samples_leaf=20, random_state=42
)
gb_final.fit(X_train, y_train)
print("  Binary model trained.")

gb_multi = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, min_samples_leaf=20, random_state=42
)
gb_multi.fit(X_train, y_train_mc)
print("  Multi-class model trained.")

# ── 9. Score Nov 2025 feature rows ────────────────────────────────────────────
print(f"\nScoring {FEATURE_MONTH} rows ...")
zip_labels = pd.read_csv(DATA / "zip_cluster_labels.csv", index_col=0)
zip_labels.index = zip_labels.index.astype(str).str.zfill(5)
model_zips = set(zip_labels.index)

score_df = panel[
    (panel["month"] == FEATURE_MONTH) &
    (panel["zip"].isin(model_zips))
].copy()
score_df = score_df[score_df[SELECTED].notna().all(axis=1)].copy()
score_df["cluster_name"] = score_df["zip"].map(zip_labels["cluster_name"])
score_df["cluster"]      = score_df["zip"].map(zip_labels["cluster"])
score_df["cluster_name"] = score_df["cluster_name"].fillna("unknown")
score_df["cluster"]      = score_df["cluster"].fillna(-1)

print(f"  Rows with complete features at {FEATURE_MONTH}: {len(score_df)}")

X_score = score_df[SELECTED].values
prob_bin = gb_final.predict_proba(X_score)[:, 1]

class_order = gb_multi.classes_
proba_mc    = gb_multi.predict_proba(X_score)
idx_d = list(class_order).index(-1)
idx_s = list(class_order).index(0)
idx_u = list(class_order).index(1)

# ── 10. Save outputs ──────────────────────────────────────────────────────────
print("\nSaving outputs ...")

out_bin = score_df[["zip","month","cluster","cluster_name"]].copy().reset_index(drop=True)
out_bin["prob_transition"] = prob_bin
out_bin["verified"]        = False
out_bin.to_csv(DATA / "model_predictions_forward_2025.csv", index=False)
print(f"  Saved model_predictions_forward_2025.csv  {out_bin.shape}")

out_mc = score_df[["zip","month","cluster","cluster_name"]].copy().reset_index(drop=True)
out_mc["prob_down"]      = proba_mc[:, idx_d]
out_mc["prob_stable"]    = proba_mc[:, idx_s]
out_mc["prob_up"]        = proba_mc[:, idx_u]
out_mc["pred_direction"] = gb_multi.predict(X_score)
out_mc["net_direction"]  = out_mc["prob_up"] - out_mc["prob_down"]
out_mc["verified"]       = False
out_mc.to_csv(DATA / "model_predictions_multiclass_forward_2025.csv", index=False)
print(f"  Saved model_predictions_multiclass_forward_2025.csv  {out_mc.shape}")

# Update model_comparison.json
comp_out = dict(comp)
comp_out["retrained"] = {
    "label":        "Retrained Monthly-Only (Jun 2019 – Dec 2024, predict Nov 2025)",
    "train_range":  "Jun 2019 – Dec 2024",
    "feature_point": FEATURE_MONTH,
    "forward_window": "Dec 2025 – Nov 2026",
    "n_features":   len(SELECTED),
    "fold_d_auc":   round(fold_d_auc, 4),
    "features":     SELECTED,
}
with open(DATA / "model_comparison_v2.json", "w") as f:
    json.dump(comp_out, f, indent=2)
print("  Saved model_comparison_v2.json")

print("\nDone.")
print(f"  Fold D AUC (test 2024):              {fold_d_auc:.4f}")
print(f"  Fold C AUC (test 2023, original):    0.8652")
print(f"  2025 actuals AUC (forward val):      0.7747")
print(f"  ZIPs scored at {FEATURE_MONTH}:   {len(out_bin)}")
print(f"  Avg opportunity score:               {prob_bin.mean():.3f}")
print(f"  % with prob > 0.6:                   {(prob_bin > 0.6).mean()*100:.1f}%")
