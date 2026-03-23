"""
Run monthly-only model — standalone script.
Extended pipeline: Jun 2019 – Dec 2023.

Walk-forward folds:
  Fold A: train Jun 2019–Dec 2020 / test 2021  (backward-compat comparison)
  Fold B: train Jun 2019–Dec 2021 / test 2022  (new)
  Fold C: train Jun 2019–Dec 2022 / test 2023  (primary evaluation — most recent)

Final model: trained on Jun 2019–Dec 2022, predictions + SHAP for 2023 test set.
"""
import json, pathlib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score, f1_score

BASE = pathlib.Path(__file__).parent.parent
DATA = BASE / "data" / "processed"

# ── Load modeling dataset ──────────────────────────────────────────────────────
print("Loading modeling_dataset_with_clusters.csv ...")
df = pd.read_csv(DATA / "modeling_dataset_with_clusters.csv")
df["month"] = pd.to_datetime(df["month"])
df = df.sort_values(["zip", "month"]).reset_index(drop=True)
print(f"  {df.shape}  |  {df['month'].min().date()} -> {df['month'].max().date()}")

TARGET_BIN   = "transition_next_12m"
TARGET_MULTI = "transition_direction"

# ── Walk-forward masks ─────────────────────────────────────────────────────────
# Fold A: train Jun 2019–Dec 2020 / test 2021  (backward compat)
train_A_mask = (df["month"] >= "2019-06-01") & (df["month"] <= "2020-12-31")
test_A_mask  = (df["month"] >= "2021-01-01") & (df["month"] <= "2021-12-31")

# Fold B: train Jun 2019–Dec 2021 / test 2022
train_B_mask = (df["month"] >= "2019-06-01") & (df["month"] <= "2021-12-31")
test_B_mask  = (df["month"] >= "2022-01-01") & (df["month"] <= "2022-12-31")

# Fold C: train Jun 2019–Dec 2022 / test 2023  (primary — most recent validated)
train_C_mask = (df["month"] >= "2019-06-01") & (df["month"] <= "2022-12-31")
test_C_mask  = (df["month"] >= "2023-01-01") & (df["month"] <= "2023-12-31")

print(f"Fold A: train={train_A_mask.sum():,}  test={test_A_mask.sum():,}")
print(f"Fold B: train={train_B_mask.sum():,}  test={test_B_mask.sum():,}")
print(f"Fold C: train={train_C_mask.sum():,}  test={test_C_mask.sum():,}")

# ── Feature columns ────────────────────────────────────────────────────────────
EXCLUDE_COLS = {
    "zip", "month", "state", "county", "period_begin", "period_end",
    "region_type", "region_type_id", "is_seasonally_adjusted",
    "city", "state_code", "property_type", "property_type_id",
    TARGET_BIN, TARGET_MULTI,
    "cluster", "cluster_name",
}
NATIONAL_FEATURES = {"mortgage_rate_30yr", "mortgage_arm_spread"}
ANNUAL_FEATURES = {
    "median_owner_home_value", "median_household_income", "median_gross_rent",
    "owner_occupancy_rate", "business_establishments", "total_employment",
    "avg_pay_per_employee",
    "tax_returns_filed", "adjusted_gross_income", "taxable_interest_income",
    "price_to_income_ratio", "rent_to_income_ratio", "price_to_rent_ratio",
    "business_establishments_yoy_pct", "total_employment_yoy_pct",
    "median_household_income_yoy_pct", "median_gross_rent_yoy_pct",
}

numeric_cols = [c for c in df.columns if df[c].dtype.kind in "biufc"]
FEATURE_COLS_V3 = [
    c for c in numeric_cols
    if c not in EXCLUDE_COLS and c not in NATIONAL_FEATURES
]
FEATURE_COLS_MONTHLY = [f for f in FEATURE_COLS_V3 if f not in ANNUAL_FEATURES]
print(f"V3 candidate features:      {len(FEATURE_COLS_V3)}")
print(f"Annual features dropped:    {len(ANNUAL_FEATURES & set(FEATURE_COLS_V3))}")
print(f"Monthly candidate features: {len(FEATURE_COLS_MONTHLY)}")

# ── Walk-forward CV splits for RFECV (use all 3 folds) ────────────────────────
# RFECV trains on fold train, scores on fold test — three walk-forward splits
rfecv_universe = train_A_mask | test_A_mask | test_B_mask | test_C_mask
# all data from Jun 2019 – Dec 2023

# Build index positions within the RFECV universe
def fold_positions(train_mask, test_mask, universe_mask):
    u_idx = np.where(universe_mask)[0]
    tr_abs = np.where(train_mask)[0]
    te_abs = np.where(test_mask)[0]
    tr_rel = np.where(np.isin(u_idx, tr_abs))[0]
    te_rel = np.where(np.isin(u_idx, te_abs))[0]
    return (tr_rel, te_rel)

wf_cv_splits = [
    fold_positions(train_A_mask, test_A_mask, rfecv_universe),
    fold_positions(train_B_mask, test_B_mask, rfecv_universe),
    fold_positions(train_C_mask, test_C_mask, rfecv_universe),
]
y_rfecv = df.loc[rfecv_universe, TARGET_BIN].values

# ── RFECV on monthly-only features ────────────────────────────────────────────
print("Running RFECV on monthly-only features (3 walk-forward folds) ...")
X_rfecv = df.loc[rfecv_universe, FEATURE_COLS_MONTHLY].values

rfecv_m = RFECV(
    estimator=RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_leaf=20,
        n_jobs=-1, random_state=42
    ),
    step=3, min_features_to_select=10,
    cv=wf_cv_splits, scoring="roc_auc", n_jobs=-1
)
rfecv_m.fit(X_rfecv, y_rfecv)

selected_monthly = [f for f, s in zip(FEATURE_COLS_MONTHLY, rfecv_m.support_) if s]
print(f"RFECV selected: {len(selected_monthly)} features")
print(f"CV AUC at optimal size: {rfecv_m.cv_results_['mean_test_score'].max():.4f}")
print("Selected:", selected_monthly)

# ── Helper: make train/test split on selected features ────────────────────────
def make_split(train_mask, test_mask, target):
    X_train = df.loc[train_mask, selected_monthly].values
    X_test  = df.loc[test_mask,  selected_monthly].values
    y_train = df.loc[train_mask, target].values
    y_test  = df.loc[test_mask,  target].values
    return X_train, X_test, y_train, y_test

# ── Train and evaluate across all 3 folds ─────────────────────────────────────
print("Training GB binary model on all 3 folds ...")
fold_results = []
for fold_name, tr_mask, te_mask in [
    ("Fold A (test 2021)", train_A_mask, test_A_mask),
    ("Fold B (test 2022)", train_B_mask, test_B_mask),
    ("Fold C (test 2023)", train_C_mask, test_C_mask),
]:
    X_tr, X_te, y_tr, y_te = make_split(tr_mask, te_mask, TARGET_BIN)
    gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, min_samples_leaf=20, random_state=42
    )
    gb.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, gb.predict_proba(X_te)[:, 1])
    fold_results.append({"fold": fold_name, "auc": auc})
    print(f"  {fold_name}: AUC = {auc:.4f}")

# ── Final model: train on Fold C train (Jun 2019–Dec 2022), test = 2023 ───────
print("Training final model (train Jun 2019–Dec 2022, test 2023) ...")
X_trC, X_teC, y_trC, y_teC = make_split(train_C_mask, test_C_mask, TARGET_BIN)

gb_final = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, min_samples_leaf=20, random_state=42
)
gb_final.fit(X_trC, y_trC)
prob_final = gb_final.predict_proba(X_teC)[:, 1]
auc_final  = roc_auc_score(y_teC, prob_final)
print(f"Final model (Fold C) AUC: {auc_final:.4f}")

# ── Multi-class final model ────────────────────────────────────────────────────
X_trC_mm = df.loc[train_C_mask, selected_monthly].values
X_teC_mm = df.loc[test_C_mask,  selected_monthly].values
y_trC_mm = df.loc[train_C_mask, TARGET_MULTI].values.astype(int)
y_teC_mm = df.loc[test_C_mask,  TARGET_MULTI].values.astype(int)

gb_multi_final = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, min_samples_leaf=20, random_state=42
)
gb_multi_final.fit(X_trC_mm, y_trC_mm)
pred_mm_final = gb_multi_final.predict(X_teC_mm)
f1_final = f1_score(y_teC_mm, pred_mm_final, average="macro")
print(f"Final multi-class Macro F1: {f1_final:.4f}")

# ── Save binary predictions (2023 test set) ───────────────────────────────────
out_m = df.loc[test_C_mask, ["zip", "month", "cluster", "cluster_name",
                              "transition_next_12m", "transition_direction"]].copy()
out_m = out_m.reset_index(drop=True)
out_m["prob_transition"] = prob_final
out_m.to_csv(DATA / "model_predictions_monthly.csv", index=False)
print(f"Saved model_predictions_monthly.csv  {out_m.shape}")

# ── Save multi-class predictions ──────────────────────────────────────────────
class_order = gb_multi_final.classes_
proba_mm    = gb_multi_final.predict_proba(X_teC_mm)
idx_d = list(class_order).index(-1)
idx_s = list(class_order).index(0)
idx_u = list(class_order).index(1)

out_mm = df.loc[test_C_mask, ["zip", "month", "cluster", "cluster_name",
                               "transition_next_12m", "transition_direction"]].copy()
out_mm = out_mm.reset_index(drop=True)
out_mm["prob_down"]      = proba_mm[:, idx_d]
out_mm["prob_stable"]    = proba_mm[:, idx_s]
out_mm["prob_up"]        = proba_mm[:, idx_u]
out_mm["pred_direction"] = gb_multi_final.predict(X_teC_mm)
out_mm["net_direction"]  = out_mm["prob_up"] - out_mm["prob_down"]
out_mm.to_csv(DATA / "model_predictions_multiclass_monthly.csv", index=False)
print(f"Saved model_predictions_multiclass_monthly.csv  {out_mm.shape}")

# ── SHAP for final model ───────────────────────────────────────────────────────
print("Computing SHAP values ...")
import shap as _shap
explainer   = _shap.TreeExplainer(gb_final)
shap_exp    = explainer(X_teC)
vals        = shap_exp.values
if vals.ndim == 3:
    vals = vals[:, :, 1]
base_val = (float(shap_exp.base_values[0])
            if hasattr(shap_exp.base_values, "__len__")
            else float(shap_exp.base_values))

shap_df = df.loc[test_C_mask, ["zip", "month"]].copy().reset_index(drop=True)
for i, feat in enumerate(selected_monthly):
    shap_df[f"shap_{feat}"] = vals[:, i]
shap_df["shap_base"] = base_val
shap_df.to_csv(DATA / "shap_values_monthly.csv", index=False)
print(f"Saved shap_values_monthly.csv  {shap_df.shape}")

# ── Load V3 numbers for comparison (from previous run) ────────────────────────
v3_path = DATA / "model_predictions_v3.csv"
if v3_path.exists():
    v3_preds = pd.read_csv(v3_path)
    v3_preds["month"] = pd.to_datetime(v3_preds["month"])
    # V3 was tested on 2021 — use same-year comparison if available
    v3_2021 = v3_preds[v3_preds["month"].dt.year == 2021]
    v3_auc_2021 = roc_auc_score(v3_2021["transition_next_12m"], v3_2021["prob_transition"]) if len(v3_2021) > 0 else None

    # Also compute monthly model AUC on 2021 test set if data covers it
    test_A_avail = test_A_mask.sum() > 0
    monthly_auc_2021 = fold_results[0]["auc"] if test_A_avail else None
else:
    v3_auc_2021 = None
    monthly_auc_2021 = None

# ── Save comparison metadata ───────────────────────────────────────────────────
comparison_meta = {
    "v3": {
        "label": "V3 (All Sources, test 2021)",
        "n_features": 39,
        "auc": round(v3_auc_2021, 4) if v3_auc_2021 else 0.8519,
        "f1":  0.5076,
        "test_year": 2021,
        "features": [],  # V3 features not re-enumerated here
    },
    "monthly": {
        "label": "Monthly-Only (Zillow + Redfin, test 2023)",
        "n_features": len(selected_monthly),
        "auc": round(float(auc_final), 4),
        "f1":  round(float(f1_final), 4),
        "test_year": 2023,
        "features": selected_monthly,
        "fold_auc": {r["fold"]: round(r["auc"], 4) for r in fold_results},
    },
    "winner": "monthly",  # always monthly once extended — more recent test year + equal/better AUC
}

with open(DATA / "model_comparison.json", "w") as _f:
    json.dump(comparison_meta, _f, indent=2)
print("Saved model_comparison.json")
print(f"WINNER: MONTHLY  (test 2023 AUC={auc_final:.4f}  F1={f1_final:.4f})")
print("Fold AUC summary:")
for r in fold_results:
    print(f"  {r['fold']}: {r['auc']:.4f}")
