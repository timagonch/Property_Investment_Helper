"""
Score all months with complete features (Jun 2019 - Nov 2025) using the
retrained model (trained on Jun 2019 - Dec 2024).

This extends the app time series beyond the 2023 test set so investors
can see how each ZIP's opportunity score has evolved up to the present.

Outputs
-------
  data/processed/model_predictions_full.csv
      Binary opportunity score for every scoreable month per ZIP
  data/processed/model_predictions_multiclass_full.csv
      Multiclass (prob_down/stable/up, net_direction) for every month
  data/processed/features_full.csv
      Full feature panel (Jun 2019 - Nov 2025) for Deep Dive feature charts
"""

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")

BASE = pathlib.Path(__file__).parent.parent
RAW  = BASE / "data" / "raw"
DATA = BASE / "data" / "processed"

# ── Load model config ─────────────────────────────────────────────────────────
with open(DATA / "model_comparison.json") as f:
    comp = json.load(f)
SELECTED = comp["monthly"]["features"]
print(f"Features ({len(SELECTED)}): {SELECTED}")

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
REDFIN_COLS = [
    "PERIOD_BEGIN", "REGION", "PROPERTY_TYPE",
    "HOMES_SOLD", "NEW_LISTINGS", "INVENTORY", "MEDIAN_DOM",
    "OFF_MARKET_IN_TWO_WEEKS", "SOLD_ABOVE_LIST", "AVG_SALE_TO_LIST",
]
chunks = []
for chunk in pd.read_csv(
    RAW / "redfin" / "redfin_zip_market_tracker_zip.tsv000",
    sep="\t",
    usecols=REDFIN_COLS,
    chunksize=50_000,
    low_memory=False,
):
    chunks.append(chunk[chunk["PROPERTY_TYPE"] == "All Residential"])
redfin_raw = pd.concat(chunks, ignore_index=True)
redfin_raw["zip"]   = redfin_raw["REGION"].str.extract(r"(\d{5})").iloc[:, 0].str.zfill(5)
redfin_raw["month"] = pd.to_datetime(redfin_raw["PERIOD_BEGIN"]) + pd.offsets.MonthEnd(0)
redfin_raw = redfin_raw.rename(columns={
    "HOMES_SOLD":              "homes_sold",
    "NEW_LISTINGS":            "new_listings",
    "INVENTORY":               "inventory",
    "MEDIAN_DOM":              "median_days_on_market",
    "OFF_MARKET_IN_TWO_WEEKS": "pct_off_market_in_2wks",
    "SOLD_ABOVE_LIST":         "pct_sold_above_list",
    "AVG_SALE_TO_LIST":        "avg_sale_to_list_ratio",
})
redfin = redfin_raw[["zip", "month", "homes_sold", "new_listings", "inventory",
                      "median_days_on_market", "pct_off_market_in_2wks",
                      "pct_sold_above_list", "avg_sale_to_list_ratio"]].copy()
redfin = redfin.sort_values(["zip", "month"]).reset_index(drop=True)

redfin["inventory_mom_pct"] = (
    redfin.groupby("zip")["inventory"]
    .transform(lambda s: s.pct_change(1, fill_method=None) * 100)
)
redfin["inventory_yoy_pct"] = (
    redfin.groupby("zip")["inventory"]
    .transform(lambda s: s.pct_change(12, fill_method=None) * 100)
)
print(f"  Redfin: {redfin['month'].min().date()} to {redfin['month'].max().date()} | {redfin['zip'].nunique()} ZIPs")

# ── 3. Merge and engineer features ───────────────────────────────────────────
print("Merging and engineering features ...")
panel = zhvi.merge(redfin, on=["zip", "month"], how="inner")
panel = panel.sort_values(["zip", "month"]).reset_index(drop=True)

def zip_rolling(df, col, window, func="mean"):
    return df.groupby("zip")[col].transform(
        lambda s: getattr(s.rolling(window, min_periods=window), func)()
    )

def zip_lag(df, col, periods):
    return df.groupby("zip")[col].transform(lambda s: s.shift(periods))

def zip_relative_departure(df, col, window=12):
    rm = df.groupby("zip")[col].transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    return (df[col] / rm.replace(0, np.nan)) - 1

panel["home_value_mom_3m_avg"]  = zip_rolling(panel, "home_value_mom_pct", 3)
panel["home_value_mom_12m_avg"] = zip_rolling(panel, "home_value_mom_pct", 12)
panel["inventory_mom_12m_avg"]  = zip_rolling(panel, "inventory_mom_pct",  12)
panel["home_value_accel"]       = panel["home_value_mom_3m_avg"] - panel["home_value_mom_12m_avg"]
panel["home_value_vs_baseline"] = zip_relative_departure(panel, "home_value_index")
panel["dom_vs_baseline"]        = zip_relative_departure(panel, "median_days_on_market")
panel["home_value_mom_pct_lag6"]     = zip_lag(panel, "home_value_mom_pct",     6)
panel["pct_sold_above_list_lag6"]    = zip_lag(panel, "pct_sold_above_list",    6)
panel["avg_sale_to_list_ratio_lag6"] = zip_lag(panel, "avg_sale_to_list_ratio", 6)
panel["home_value_vs_baseline_lag6"] = zip_lag(panel, "home_value_vs_baseline", 6)

print(f"  Panel: {panel.shape} | {panel['zip'].nunique()} ZIPs")

# ── 4. Attach cluster labels ──────────────────────────────────────────────────
zip_labels = pd.read_csv(DATA / "zip_cluster_labels.csv", index_col=0)
zip_labels.index = zip_labels.index.astype(str).str.zfill(5)
model_zips = set(zip_labels.index)

panel = panel[panel["zip"].isin(model_zips)].copy()
panel["cluster_name"] = panel["zip"].map(zip_labels["cluster_name"])
panel["cluster"]      = panel["zip"].map(zip_labels["cluster"])

# ── 5. Load training labels and train final model ─────────────────────────────
print("\nLoading training labels ...")
existing = pd.read_csv(DATA / "modeling_dataset_with_clusters.csv",
                       usecols=["zip", "month", "transition_next_12m", "transition_direction"])
existing["month"] = pd.to_datetime(existing["month"])
existing["zip"]   = existing["zip"].astype(str).str.zfill(5)

val = pd.read_csv(DATA / "validation_2024_predictions.csv",
                  usecols=["zip", "month", "actual_transition", "actual_direction"])
val["month"] = pd.to_datetime(val["month"])
val["zip"]   = val["zip"].astype(str).str.zfill(5)
val = val.rename(columns={"actual_transition": "transition_next_12m",
                           "actual_direction":  "transition_direction"})
val = val.dropna(subset=["transition_next_12m"])

all_labels = pd.concat([existing, val], ignore_index=True).drop_duplicates(subset=["zip", "month"])
print(f"  Labels: {len(all_labels):,} rows | {all_labels['month'].min().date()} to {all_labels['month'].max().date()}")

model_df = panel.merge(all_labels, on=["zip", "month"], how="left")

train_mask = (
    (model_df["month"] >= "2019-06-01") &
    (model_df["month"] <= "2024-12-31") &
    model_df[SELECTED].notna().all(axis=1) &
    model_df["transition_next_12m"].notna()
)
train_df = model_df[train_mask].copy()
print(f"  Training rows: {len(train_df):,} | {train_df['month'].min().date()} to {train_df['month'].max().date()}")

print("\nTraining models ...")
X_train    = train_df[SELECTED].values
y_train    = train_df["transition_next_12m"].values.astype(int)
y_train_mc = train_df["transition_direction"].values.astype(int)

GB_PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=4,
                 subsample=0.8, min_samples_leaf=20, random_state=42)

gb_bin   = GradientBoostingClassifier(**GB_PARAMS).fit(X_train, y_train)
gb_multi = GradientBoostingClassifier(**GB_PARAMS).fit(X_train, y_train_mc)
print("  Done.")

# ── 6. Score all months with complete features ────────────────────────────────
print("\nScoring all months ...")
score_mask = (
    (panel["month"] >= "2019-06-01") &
    panel[SELECTED].notna().all(axis=1)
)
score_df = panel[score_mask].copy()
print(f"  Scoreable rows: {len(score_df):,} | {score_df['month'].min().date()} to {score_df['month'].max().date()}")

X_score    = score_df[SELECTED].values
prob_bin   = gb_bin.predict_proba(X_score)[:, 1]

class_order = gb_multi.classes_
proba_mc    = gb_multi.predict_proba(X_score)
idx_d = list(class_order).index(-1)
idx_s = list(class_order).index(0)
idx_u = list(class_order).index(1)

# ── 7. Save outputs ───────────────────────────────────────────────────────────
print("\nSaving outputs ...")

out_bin = score_df[["zip", "month", "cluster", "cluster_name"]].copy().reset_index(drop=True)
out_bin["prob_transition"] = prob_bin
out_bin.to_csv(DATA / "model_predictions_full.csv", index=False)
print(f"  Saved model_predictions_full.csv  {out_bin.shape}")

out_mc = score_df[["zip", "month", "cluster", "cluster_name"]].copy().reset_index(drop=True)
out_mc["prob_down"]      = proba_mc[:, idx_d]
out_mc["prob_stable"]    = proba_mc[:, idx_s]
out_mc["prob_up"]        = proba_mc[:, idx_u]
out_mc["pred_direction"] = gb_multi.predict(X_score)
out_mc["net_direction"]  = out_mc["prob_up"] - out_mc["prob_down"]
out_mc.to_csv(DATA / "model_predictions_multiclass_full.csv", index=False)
print(f"  Saved model_predictions_multiclass_full.csv  {out_mc.shape}")

# Save feature panel for Deep Dive charts
feat_cols = ["zip", "month", "cluster", "cluster_name"] + SELECTED
features_out = score_df[feat_cols].copy().reset_index(drop=True)
features_out.to_csv(DATA / "features_full.csv", index=False)
print(f"  Saved features_full.csv  {features_out.shape}")

# ── 8. Compute SHAP values for full scored panel ──────────────────────────────
print("\nComputing SHAP values ...")
explainer = shap.TreeExplainer(gb_bin)
shap_explanation = explainer(X_score)

vals = shap_explanation.values
if vals.ndim == 3:
    vals = vals[:, :, 1]   # positive-class slice if 3D

base_val = shap_explanation.base_values
if hasattr(base_val, "__len__"):
    base_val = float(base_val[0])
else:
    base_val = float(base_val)

shap_out = score_df[["zip", "month", "cluster", "cluster_name"]].copy().reset_index(drop=True)
for i, feat in enumerate(SELECTED):
    shap_out[f"shap_{feat}"] = vals[:, i]
shap_out["shap_base"] = base_val

shap_out.to_csv(DATA / "shap_values_full.csv", index=False)
print(f"  Saved shap_values_full.csv  {shap_out.shape}")

print("\nDone.")
print(f"  Date range scored: {out_bin['month'].min().date()} to {out_bin['month'].max().date()}")
print(f"  ZIPs scored: {out_bin['zip'].nunique()}")
print(f"  Avg opportunity score (Nov 2025): "
      f"{out_bin[out_bin['month']=='2025-11-30']['prob_transition'].mean():.3f}")
