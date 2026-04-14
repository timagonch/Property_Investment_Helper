"""
Baseline comparison: Logistic Regression vs. Gradient Boosting
Same walk-forward folds (A/B/C) used in the final model evaluation.

Fold A: train Jun 2019–Dec 2020, test 2021
Fold B: train Jun 2019–Dec 2021, test 2022
Fold C: train Jun 2019–Dec 2022, test 2023  ← primary evaluation fold
"""

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = pathlib.Path(__file__).parent.parent
DATA = BASE / "data" / "processed"

# ── Load data and features ────────────────────────────────────────────────────
with open(DATA / "model_comparison.json") as f:
    comp = json.load(f)
FEATURES = comp["monthly"]["features"]

df = pd.read_csv(DATA / "modeling_dataset_with_clusters.csv")
df["month"] = pd.to_datetime(df["month"])
df["zip"]   = df["zip"].astype(str).str.zfill(5)

# Keep rows with complete features and binary label
mask = df[FEATURES].notna().all(axis=1) & df["transition_next_12m"].notna()
df   = df[mask].copy()
print(f"Modeling rows: {len(df):,}  |  ZIPs: {df['zip'].nunique()}")
print(f"Date range:    {df['month'].min().date()} to {df['month'].max().date()}")
print()

# ── Walk-forward folds ────────────────────────────────────────────────────────
FOLDS = {
    "Fold A (test 2021)": ("2019-06-01", "2020-12-31", 2021),
    "Fold B (test 2022)": ("2019-06-01", "2021-12-31", 2022),
    "Fold C (test 2023)": ("2019-06-01", "2022-12-31", 2023),
}

GB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, min_samples_leaf=20, random_state=42
)

results = []

for fold_name, (train_start, train_end, test_year) in FOLDS.items():
    train = df[(df["month"] >= train_start) & (df["month"] <= train_end)]
    test  = df[df["month"].dt.year == test_year]

    X_train = train[FEATURES].values
    y_train = train["transition_next_12m"].values.astype(int)
    X_test  = test[FEATURES].values
    y_test  = test["transition_next_12m"].values.astype(int)

    # Logistic Regression (needs scaled features)
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_tr_sc, y_train)
    lr_prob = lr.predict_proba(X_te_sc)[:, 1]
    lr_auc  = roc_auc_score(y_test, lr_prob)
    lr_f1   = f1_score(y_test, (lr_prob >= 0.5).astype(int), average="macro")

    # Gradient Boosting
    gb = GradientBoostingClassifier(**GB_PARAMS)
    gb.fit(X_train, y_train)
    gb_prob = gb.predict_proba(X_test)[:, 1]
    gb_auc  = roc_auc_score(y_test, gb_prob)
    gb_f1   = f1_score(y_test, (gb_prob >= 0.5).astype(int), average="macro")

    results.append({
        "Fold": fold_name,
        "LR AUC":  round(lr_auc, 4),
        "GB AUC":  round(gb_auc, 4),
        "dAUC":    round(gb_auc - lr_auc, 4),
        "LR F1":   round(lr_f1, 4),
        "GB F1":   round(gb_f1, 4),
        "dF1":     round(gb_f1 - lr_f1, 4),
        "n_train": len(train),
        "n_test":  len(test),
    })
    print(f"{fold_name}")
    print(f"  LR  -> AUC={lr_auc:.4f}  F1={lr_f1:.4f}")
    print(f"  GB  -> AUC={gb_auc:.4f}  F1={gb_f1:.4f}  (d AUC={gb_auc-lr_auc:+.4f})")
    print()

# ── Summary table ─────────────────────────────────────────────────────────────
print("=" * 65)
print(f"{'Fold':<28} {'LR AUC':>8} {'GB AUC':>8} {'dAUC':>8}  {'LR F1':>8} {'GB F1':>8}")
print("-" * 65)
for r in results:
    print(f"{r['Fold']:<28} {r['LR AUC']:>8.4f} {r['GB AUC']:>8.4f} {r['dAUC']:>+8.4f}  {r['LR F1']:>8.4f} {r['GB F1']:>8.4f}")
print("=" * 65)
