# CLAUDE.md — Project Context for Claude Code

## Package management
Always use `uv add <package>` to install dependencies. Never use `pip install`.
The project uses `uv` with `pyproject.toml` + `uv.lock` for reproducibility.

## Running notebooks
```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb --ExecutePreprocessor.timeout=600
```
joblib KeyError warnings during execution are benign Windows temp-file cleanup noise — not actual errors.

## Running the Streamlit app
```bash
uv run streamlit run app.py
```

---

## Project overview

**Neighborhood Market Early-Warning System** — a Business Intelligence group project (Spring 2026).
Goal: identify NC/SC ZIP codes showing early signs of neighborhood transition and predict the direction of change over the next 12 months.

**Final analytic dataset:** 16,740 monthly observations × 540 ZIP codes × Jun 2019–Dec 2021

### Core questions
1. Which ZIPs show the strongest signs of emerging transition pressure?
2. Are there distinct neighborhood market archetypes across ZIP codes?
3. Can past indicators predict whether a ZIP will enter a transition state within 12 months?
4. Which leading indicators contribute most to that prediction?

---

## Data sources (6 integrated)

| Source | What it provides |
|---|---|
| Zillow Research | ZHVI (home value index), ZIP universe |
| Redfin Market Tracker | Monthly housing activity (sale/list price, inventory, DOM, sale-to-list, etc.) |
| ACS 5-year estimates | Median income, rent, home value, owner occupancy rate (annual) |
| County Business Patterns | Establishments, employment, annual payroll (annual) |
| IRS Statistics of Income | ZIP-level income and tax return indicators (annual) |
| Freddie Mac PMMS | 30yr/15yr/5-1 ARM mortgage rates (removed from final model — national signal) |

---

## Target variables

Both are engineered from a composite transition score (z-scored deltas of home value growth, inventory change, and market competitiveness over the forward 12-month window vs. each ZIP's own trailing baseline).

| Target | Type | Definition |
|---|---|---|
| `transition_next_12m` | Binary | 1 if composite score ≥ p75 (upward transition) |
| `transition_direction` | 3-class | +1 up / 0 stable / -1 down (p25/p75 thresholds) |

Thresholds computed on the final 16,740-row modeling subset → clean 25/50/25 class split.

---

## Notebook pipeline (run in order)

1. `notebooks/exploration.ipynb` — data quality audit (already run)
2. `notebooks/target_engineering.ipynb` → `data/processed/modeling_dataset.csv` (16,740 × 74)
3. `notebooks/clustering.ipynb` → `data/processed/modeling_dataset_with_clusters.csv` (16,740 × 76)
4. `notebooks/modeling.ipynb` → model predictions + feature importance CSVs

---

## Key modeling decisions

- **Walk-forward validation only** — no random splits on time-series panel data
  - Fold A: train Jun–Dec 2019, test Jan–Dec 2020
  - Fold B: train Jun 2019–Dec 2020, test Jan–Dec 2021 (primary evaluation)
- **Mortgage features removed** — `mortgage_rate_30yr` and `mortgage_arm_spread` have zero cross-ZIP variance per month; they capture temporal trends, not ZIP-specific signal
- **RFECV feature selection** — reduced from 69 → 10 features with negligible AUC loss (0.8598 → 0.8580)
- **V3 final features (10):** `home_value_mom_pct`, `home_value_accel`, `home_value_vs_baseline`, `home_value_vs_baseline_lag6`, `home_value_mom_3m_avg`, `home_value_index`, `median_owner_home_value`, `inventory_mom_12m_avg`, `inventory_yoy_pct`, `homes_sold`

## Model performance (Fold B — test year 2021)

| Model | Task | Metric | Score |
|---|---|---|---|
| Gradient Boosting V3 | Binary (upward) | AUC | 0.858 |
| Gradient Boosting multi-class V3 | 3-class direction | Macro F1 | 0.555 |

---

## Clustering

K-means k=5 on 22 ZIP-level features (16 means + 6 slopes). Silhouette = 0.1105.

| Cluster | Name | ZIPs |
|---|---|---|
| 0 | `hot_competitive` | 138 |
| 1 | `stagnant_low_activity` | 95 |
| 2 | `slow_market_rising` | 103 |
| 3 | `affordable_heating` | 149 |
| 4 | `luxury_appreciating` | 55 |

---

## Key processed files

| File | Description |
|---|---|
| `data/processed/modeling_dataset.csv` | 16,740 × 74, both targets, no clusters |
| `data/processed/modeling_dataset_with_clusters.csv` | 16,740 × 76, adds cluster + cluster_name |
| `data/processed/model_predictions_v3.csv` | 6,480 rows, GB V3 binary predictions (2021 test set) |
| `data/processed/model_predictions_multiclass.csv` | 6,480 rows, GB multi-class predictions with prob_down/stable/up + net_direction |
| `data/processed/feature_importance.csv` | 65 features ranked by permutation importance |
| `data/processed/zip_cluster_labels.csv` | 540 ZIPs with cluster assignments |
| `data/geo/nc_sc_zips.geojson` | Simplified Census ZCTA boundaries for the 540 ZIPs |
| `data/geo/nc_sc_states.geojson` | NC/SC state boundary lines for map overlay |

---

## Terminology

- **Opportunity score** = predicted probability of upward transition (buy signal for investors)
- **Net direction score** = prob_up − prob_down, range −1 to +1 (diverging map)
- **Cooling risk** = downward transition probability (avoid/sell signal)
- Do NOT use "risk" for upward transition predictions — "risk" implies downside in finance

---

## Important constraints

- Respect time-ordered structure — no random splits
- The `data/processed/` files are large; do not commit them to git without checking `.gitignore`
- `data/geo/` GeoJSON files are committed (small, needed for the app)
- Python 3.12, pandas < 3 (streamlit constraint), scikit-learn ≥ 1.8 (removed `multi_class` param from LogisticRegression)
