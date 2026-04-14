"""
Recluster ZIP codes using Strategy C: 18 RFECV model features as means, Jun 2019–Dec 2024.
Silhouette = 0.163 vs. original 0.110.

Outputs:
  data/processed/zip_cluster_labels.csv  (updated)
  data/processed/modeling_dataset_with_clusters.csv  (updated cluster + cluster_name)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

RANDOM_STATE = 42
K = 5

MODEL_18 = [
    "homes_sold", "new_listings", "inventory", "median_days_on_market", "pct_off_market_in_2wks",
    "home_value_mom_pct", "home_value_yoy_pct", "inventory_yoy_pct",
    "home_value_mom_3m_avg", "home_value_mom_12m_avg", "inventory_mom_12m_avg",
    "home_value_accel", "home_value_vs_baseline", "dom_vs_baseline",
    "home_value_mom_pct_lag6", "pct_sold_above_list_lag6",
    "avg_sale_to_list_ratio_lag6", "home_value_vs_baseline_lag6",
]

# Cluster ID → internal name (assigned after examining full profiles including home_value_index)
# 0 (211, $266K): highest above-list (37.6%), fastest DOM (45), tightest competition → competitive_mid_market
# 1 (172, $216K): highest YoY (12.8%), rising inventory (+10.6%)                    → affordable_high_demand
# 2 (154, $213K): rising inventory (8% YoY), slow DOM (75.6), softest demand        → moderate_supply_growing
# 3  (44, $162K): lowest HV, weakest appreciation (7.1%), highest inv growth (14.2%) → low_activity_cooling
# 4  (74, $318K): highest HV, ONLY cluster with declining inventory (-8%), high demand → high_value_appreciating
CLUSTER_NAMES = {
    0: "competitive_mid_market",
    1: "affordable_high_demand",
    2: "moderate_supply_growing",
    3: "low_activity_cooling",
    4: "high_value_appreciating",
}

# ── Load data ────────────────────────────────────────────────────────────────
zip_labels_orig = pd.read_csv("data/processed/zip_cluster_labels.csv", index_col=0)
zip_labels_orig.index = zip_labels_orig.index.astype(str).str.zfill(5)
MODEL_ZIPS = set(zip_labels_orig.index)

ds_ext = pd.read_csv("data/processed/features_engineered.csv")
ds_ext["month"] = pd.to_datetime(ds_ext["month"])
ds_ext["zip"]   = ds_ext["zip"].astype(str).str.zfill(5)

ds_b = ds_ext[
    (ds_ext["month"] >= "2019-06-01") &
    (ds_ext["month"] <= "2024-12-31") &
    ds_ext["zip"].isin(MODEL_ZIPS)
].dropna(subset=MODEL_18)

mat_c = ds_b.groupby("zip")[MODEL_18].mean().dropna()
print(f"Feature matrix: {mat_c.shape}  ({len(mat_c)} ZIPs)")

# ── Fit clustering ───────────────────────────────────────────────────────────
sc = StandardScaler()
X_c = sc.fit_transform(mat_c)

km = KMeans(n_clusters=K, n_init=50, max_iter=500, random_state=RANDOM_STATE)
labels = km.fit_predict(X_c)
mat_c["cluster"]      = labels
mat_c["cluster_name"] = mat_c["cluster"].map(CLUSTER_NAMES)

# ── Print profiles for verification ─────────────────────────────────────────
key = [
    "home_value_yoy_pct", "home_value_vs_baseline", "inventory_yoy_pct", "inventory_mom_12m_avg",
    "pct_sold_above_list_lag6", "pct_off_market_in_2wks", "median_days_on_market",
    "home_value_accel", "avg_sale_to_list_ratio_lag6",
]
profile = mat_c.groupby("cluster_name")[key].mean()
profile.insert(0, "n_zips", mat_c.groupby("cluster_name").size())
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 220)
print("\nCluster profiles:")
print(profile.to_string())

# ── Save zip_cluster_labels.csv ──────────────────────────────────────────────
zip_labels_new = mat_c[["cluster", "cluster_name"]].copy()
zip_labels_new.index.name = "zip"
zip_labels_new.to_csv("data/processed/zip_cluster_labels.csv")
print(f"\nSaved zip_cluster_labels.csv  ({len(zip_labels_new)} ZIPs)")

# ── Update modeling_dataset_with_clusters.csv ────────────────────────────────
mds = pd.read_csv("data/processed/modeling_dataset_with_clusters.csv")
mds["zip"] = mds["zip"].astype(str).str.zfill(5)

# Drop old cluster columns, join new ones
mds = mds.drop(columns=[c for c in ["cluster", "cluster_name"] if c in mds.columns])
mds = mds.merge(zip_labels_new.reset_index()[["zip", "cluster", "cluster_name"]], on="zip", how="left")
mds.to_csv("data/processed/modeling_dataset_with_clusters.csv", index=False)
print(f"Updated modeling_dataset_with_clusters.csv  ({mds.shape})")
print(mds["cluster_name"].value_counts().sort_values(ascending=False).to_string())

# ── Also update any forward prediction files that carry cluster_name ─────────
for fname in [
    "model_predictions_forward_2025.csv",
    "model_predictions_multiclass_forward_2025.csv",
]:
    try:
        df = pd.read_csv(f"data/processed/{fname}")
        df["zip"] = df["zip"].astype(str).str.zfill(5)
        if "cluster" in df.columns or "cluster_name" in df.columns:
            df = df.drop(columns=[c for c in ["cluster", "cluster_name"] if c in df.columns])
        df = df.merge(zip_labels_new.reset_index()[["zip", "cluster", "cluster_name"]], on="zip", how="left")
        df.to_csv(f"data/processed/{fname}", index=False)
        print(f"Updated {fname}")
    except FileNotFoundError:
        print(f"Skipped {fname} (not found)")

# Also update validation output if it exists
for fname in ["validation_2024_predictions.csv"]:
    try:
        df = pd.read_csv(f"data/processed/{fname}")
        df["zip"] = df["zip"].astype(str).str.zfill(5)
        if "cluster" in df.columns or "cluster_name" in df.columns:
            df = df.drop(columns=[c for c in ["cluster", "cluster_name"] if c in df.columns])
        df = df.merge(zip_labels_new.reset_index()[["zip", "cluster", "cluster_name"]], on="zip", how="left")
        df.to_csv(f"data/processed/{fname}", index=False)
        print(f"Updated {fname}")
    except FileNotFoundError:
        pass

print("\nDone.")
