# Modeling Iteration Log

Tracks every significant modeling decision, what changed, why, and what the results were.
Useful for presenting the evolution of the project and justifying final choices.

---

## Iteration 1 — Baseline: V1 with All Features (inc. Mortgage)
**Date:** Early Spring 2026
**Dataset:** 540 NC/SC ZIPs · Jun 2019–Dec 2021 · 16,740 rows
**Model:** Gradient Boosting, Fold B (train 2019–2020, test 2021)

| Metric | Score |
|---|---|
| Binary AUC | 0.8607 |
| Macro F1 | — |
| Features | 71 |

**What happened:** First full model trained on all available features including mortgage rate.

**Finding:** `mortgage_rate_30yr` was the #1 feature by permutation importance — but it has **zero variance across ZIPs within any given month** (it's a national rate). The model was learning the macro calendar (low rates 2020–2021 = more transitions), not ZIP-specific signals. This is temporal leakage.

**Decision:** Remove mortgage features.

---

## Iteration 2 — V2: Removed Mortgage Features
**Date:** Early Spring 2026
**Dataset:** Same 540 ZIPs
**Model:** Gradient Boosting, Fold B

| Metric | Score |
|---|---|
| Binary AUC | 0.8473 |
| Macro F1 | — |
| Features | 69 |

**What happened:** Dropped `mortgage_rate_30yr` and `mortgage_arm_spread`.

**Result:** AUC dropped by 0.013 — negligible. Confirms the mortgage signal was spurious temporal leakage, not real ZIP-level predictive power.

**Decision:** Proceed without mortgage features.

---

## Iteration 3 — V3: RFECV Feature Selection (540 ZIPs)
**Date:** Early Spring 2026
**Dataset:** 540 ZIPs
**Model:** Gradient Boosting + RFECV with walk-forward CV

| Metric | Score |
|---|---|
| Binary AUC | 0.858 |
| Macro F1 | 0.555 |
| Features | 10 (down from 69) |

**What happened:** Applied Recursive Feature Elimination with Cross-Validation (RFECV) using the walk-forward CV splits to select the minimum feature set that preserved AUC.

**Result:** 69 → 10 features with negligible AUC loss. All 10 selected features were housing momentum signals (home value dynamics + inventory). No demographic or business features survived.

**Decision:** This is the V3 model. Use it in production.

---

## Iteration 4 — ZIP Recovery: 540 → 666 ZIPs
**Date:** 2026-03-18
**Change:** Modified streak filter in `exploration.ipynb` (cell 10)

**Problem discovered:** The streak filter checked for consecutive missing months across ALL 39+ numeric columns. But only 5 columns actually feed the final model features (`zhvi`, `HOMES_SOLD`, `INVENTORY_mom_pct`, `INVENTORY_yoy_pct`, `B25077_001E`). ZIPs were being excluded because of gaps in columns the model never used.

**Fix:** Changed `MODEL_COLS` to only the 5 base columns before applying the streak filter.

**Result:**
- Before: 675 ZIPs → 540 kept (135 dropped unnecessarily)
- After: 675 ZIPs → 666 kept (83 dropped with genuine gaps)
- Recovered 126 ZIPs

**Full pipeline re-run:** All 6 notebooks re-executed on the 666-ZIP dataset.

**V3 re-run results on 666 ZIPs:**

| Metric | 540 ZIPs | 666 ZIPs |
|---|---|---|
| Binary AUC | 0.858 | 0.852 |
| Macro F1 | 0.555 | 0.508 |
| Features (RFECV) | 10 | 39 |

The RFECV selected 39 features on the larger dataset (was 10 on 540). AUC barely changed; more ZIPs = more training rows = RFECV found more features useful before hitting the diminishing-returns threshold.

**Decision:** Accept 39-feature V3 on 666 ZIPs as new baseline. The feature count increase is RFECV working correctly, not overfitting.

---

## Iteration 5 — Monthly-Only Model: Drop Annual ACS/CBP/IRS Features
**Date:** 2026-03-18 → 2026-03-19
**Change:** New model trained without the 17 annual ACS/CBP/IRS features

**Motivation:** Of the 39 V3 features, 17 came from annual sources (ACS 5-year estimates, County Business Patterns, IRS Statistics of Income). These features:
- Ranked #12–#27 in permutation importance (0.0008–0.0024 vs 0.046 for top feature)
- Introduce a data lag (annual surveys lag real conditions by 12–24 months)
- Block future data extension (ACS/CBP/IRS data currently only available through 2022)

**Hypothesis:** Dropping them forces RFECV to find better monthly substitutes, may improve generalization, and opens the door to training on Redfin data back to 2012.

**Implementation:** `scripts/run_monthly_model.py` — standalone script that loads the modeling dataset, drops 17 annual features, runs RFECV on the remaining 51, trains GB binary + multi-class, saves all output files + `model_comparison.json`.

**Results (Fold B — test year 2021):**

| Model | Features | Binary AUC | Macro F1 |
|---|---|---|---|
| V3 (All Sources) | 39 | 0.8519 | 0.5076 |
| **Monthly-Only** | **33** | **0.8650** | **0.5512** |

Monthly-Only **wins on both metrics**. Dropping noisy annual features improved generalization.

**App update:** Added Model Comparison page (page 4). Monthly-Only model wired as default for all analysis pages. Winner controlled by `model_comparison.json` — auto-updates if models are retrained.

**Open question raised by user:** The monthly-only model was still trained and tested on the same 2019–2021 window. The actual benefit — more training data from 2012+ and validated predictions for 2022/2023 — was NOT yet implemented.

---

## Iteration 6 — Extended Date Range: 2018–2024
**Date:** 2026-03-19
**Change:** Extend pipeline from 2022-end to 2024-end; add 2022 and 2023 test folds

**Motivation:** App was showing 2021 predictions — 3+ years stale. Raw data already contained Redfin through Nov 2025 and Zillow through Jan 2026 (no re-pull needed). The only pipeline blockers were:
1. A hardcoded date filter in `exploration.ipynb` (`2022-12-31` → `2024-12-31`)
2. `B25077_001E` (ACS annual) in the streak-check columns — causes 24-month streak for 2023–2024 since ACS only covers through 2022. Fixed by removing it from `BASE_MODEL_COLS` (we're monthly-only, so it's irrelevant anyway).

**Pipeline changes:**
- `exploration.ipynb`: end date `2022-12-31` → `2024-12-31`; removed `B25077_001E` from streak-check cols
- All 6 notebooks re-executed
- `run_monthly_model.py` redesigned with 3 explicit date-based walk-forward folds

**Dataset after pipeline re-run:**
- 655 ZIPs (was 666 — 11 dropped due to gaps in 2022–2024 Redfin data)
- 36,025 rows · Jun 2019–Dec 2023 (55 months per ZIP)

**Results:**

| Fold | Train window | Test year | AUC |
|---|---|---|---|
| A | Jun 2019–Dec 2020 | 2021 | 0.8443 |
| B | Jun 2019–Dec 2021 | 2022 | 0.8153 |
| C (primary) | Jun 2019–Dec 2022 | 2023 | 0.8652 |

Final model: RFECV selected 18 features · AUC 0.8652 · Macro F1 0.3838 (test year 2023)

**Interpretation of fold AUC pattern:**
- 2022 dip (0.8153): Fed began aggressive rate hikes in March 2022, creating an abrupt market inflection not seen in 2019–2021 training data. Many momentum signals that predicted upward transitions reversed sharply. The model underperforms on an out-of-distribution market shift.
- 2023 recovery (0.8652): Market normalized. Housing momentum signals resumed their predictive relationship. The model's features (home value MoM, inventory trends, sale-to-list ratios) again capture the leading dynamics.

**Note on F1 drop (0.38 vs 0.55 before):** Multi-class F1 measures direction prediction (up/stable/down). In 2023, the distribution of actual transitions may differ from the 2021 class balance the model was calibrated for. This warrants investigation — the binary AUC is strong but the directional signal is weaker in this period.

**App:** Now shows 2023 predictions by default. Fold AUC breakdown added to Model Comparison page.

---

## Notes on Walk-Forward Design

Walk-forward (no random splits) is mandatory for panel time-series data. Random splits allow future months to appear in the training set, inflating AUC by leaking forward-looking information.

The "year-based" split approach in `modeling.ipynb` (`df['month'].dt.year == 2021`) is fragile because it relies on data coincidentally starting in mid-2019. This was replaced with explicit date-based splits in `run_monthly_model.py`.

---

## Feature Selection Philosophy

RFECV with walk-forward splits (not random CV) is used so feature selection itself doesn't leak future data. The RFECV uses the same train/test structure as the final model evaluation.

Permutation importance (not Gini/split-based) is used for interpretability: it measures how much AUC drops when each feature is shuffled, directly quantifying the feature's contribution to held-out predictive performance.

---

## Iteration 7 — Cluster Re-labeling
**Date:** 2026-03-19
**Change:** All 5 cluster names corrected after discovering labels were wrong

**What happened:** The CLUSTER_NAMES mapping in `clustering.ipynb` was hardcoded based on centroid analysis from the OLD 540-ZIP, 2019–2021 clustering. After re-running with 655 ZIPs × 2019–2023, K-means reorganized the clusters but the hardcoded integer-to-name mapping was never re-evaluated. The labels were wrong.

**Evidence:**
- "luxury_appreciating" had the 2nd-lowest median home value ($208K) — far from luxury
- "hot_competitive" had the lowest above-list rate (19.7%) and slowest DOM (84 days) — the opposite of competitive
- "slow_market_rising" had the highest above-list rate (35.4%) and tightest inventory — the most competitive cluster

**Root cause of original labels:** Assigned manually by visual inspection of a centroid heatmap in a single session. No documentation of the reasoning. Not re-verified after data changed.

**New labels (derived from current centroid data):**

| Old name | New name | Key differentiators |
|---|---|---|
| hot_competitive | high_value_appreciating | $376K values, 84-day DOM (slowest), 12.2% YoY (highest), accelerating appreciation slope |
| stagnant_low_activity | low_activity_cooling | $181K (lowest), 8.5% YoY (lowest), most negative appreciation slope |
| slow_market_rising | competitive_mid_market | 35.4% above-list (highest), tightest inventory, 50-day DOM — actually the most competitive |
| affordable_heating | affordable_high_demand | 48-day DOM (fastest), 45.5% off-market (highest), $218K — peak activity at entry-level price |
| luxury_appreciating | moderate_supply_growing | $208K (2nd lowest), +6.5% inventory MoM (highest of any cluster) — supply loosening |

---

## Iteration 8 — Forward Predictions (2024, Unverified)
**Date:** 2026-03-19
**Change:** Added investor-facing 2024 predictions with no ground-truth labels

**Motivation:** The validated predictions only cover 2023 (the test year). An investor making decisions today needs signals for 2024, not 2023. The raw data already covers Jan–Dec 2024 with complete Zillow + Redfin features — the model can score these rows even though the 12-month forward outcomes won't be verifiable until late 2025.

**Implementation:**
- `scripts/generate_forward_predictions.py`: trains the final GB model on Jun 2019–Dec 2022 (same as run_monthly_model.py Fold C), applies it to all 7,992 rows where Jan–Dec 2024 features are complete (666 ZIPs)
- Saves `model_predictions_forward_2024.csv` and `model_predictions_multiclass_forward_2024.csv`
- New app page "2024 Forward Predictions" — warning banner makes unverified status explicit, shows opportunity score map, net direction map, top heating/cooling ZIP tables, interpretation guide

**Key distinction maintained in the app:**
- Pages 1–3 (Market Transition Map, Archetypes, ZIP Deep Dive): show 2023 **validated** predictions
- Page 4 (Forward Predictions): shows 2024 **unverified** signal — clearly labeled as such
- Model Comparison and Performance pages: show methodology and validation metrics

**Note on 132 rows with no cluster assignment:** 11 ZIPs present in features_engineered.csv for 2024 but not in the 655-ZIP clustering dataset (they passed the streak filter but were dropped later in target_engineering due to insufficient labeled rows). These are labeled "unknown" cluster in the forward predictions file. A future improvement would be to assign them to the nearest cluster centroid.
