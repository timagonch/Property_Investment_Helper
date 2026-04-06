# NC/SC Neighborhood Market Transition — Early-Warning System

A Business Intelligence group project (Spring 2026) that identifies ZIP codes in **North Carolina and South Carolina** showing early signs of neighborhood transition and predicts the direction of change over the next 12 months.

---

## What it does

The system answers four questions for **655 NC/SC ZIP codes**:

1. Which ZIPs are showing the strongest signals of emerging transition pressure?
2. Are there distinct neighborhood market tendencies across the region?
3. Will a ZIP enter an upward or downward transition within the next 12 months?
4. Which leading indicators drive that prediction most?

Results are delivered through an interactive Streamlit dashboard with choropleth maps, cluster profiles, ZIP-level time series, SHAP explanations, and model performance summaries.

---

## Live Demo

**[https://huggingface.co/spaces/timagonch/nc-sc-market-monitor](https://huggingface.co/spaces/timagonch/nc-sc-market-monitor)**

## Run Locally

```bash
uv sync
uv pip install shap   # install separately — cross-platform build conflict
uv run streamlit run app.py
```

The app has two modes selectable from the sidebar:

**Investor mode** (default):

| Page | Description |
|---|---|
| **Home** | Project overview, methodology summary, and navigation guide |
| **2025 Signal** | Forward-looking predictions for Dec 2025–Nov 2026 — retrained model, use for prospecting |
| **Archetypes** | Map colored by cluster archetype with profile cards |
| **Compare** | Side-by-side comparison of 2–4 ZIPs |
| **Deep Dive** | Select any ZIP — monthly opportunity score, feature trends vs. cluster average, SHAP waterfall |

**Educator mode** (adds methodology pages):

| Page | Description |
|---|---|
| **Validated Map** | 2023 test-set choropleth — opportunity score or net market direction |
| **Model Comparison** | V3 (all sources) vs. Monthly-Only head-to-head with side-by-side maps |
| **Performance** | Walk-forward fold metrics, version history, top-15 features by mean \|SHAP\| |

---

## Data sources

| Source | Coverage | What it provides |
|---|---|---|
| **Zillow Research** | Monthly | Home Value Index (ZHVI) by ZIP |
| **Redfin Market Tracker** | Monthly | Sale/list price, inventory, days on market, sale-to-list ratio, homes sold, off-market speed |
| **ACS 5-year estimates** | Annual | Median income, rent, home value, owner occupancy (used in V3; dropped from winning model) |
| **County Business Patterns** | Annual | Establishments, employment, payroll (used in V3; dropped from winning model) |
| **IRS Statistics of Income** | Annual | ZIP-level income and tax indicators (used in V3; dropped from winning model) |
| **Freddie Mac PMMS** | Monthly | Mortgage rates (excluded — national signal with zero cross-ZIP variance per month) |

**Final analytic dataset:** 655 ZIPs · Jun 2019–Dec 2024 (features) · Jun 2019–Dec 2024 (labeled) · 2025 forward signal

---

## Methodology

### 1. Target engineering
Two supervised targets are engineered from a composite transition score — z-scored deltas of home value growth, inventory tightening, and market competitiveness over the forward 12-month window vs. each ZIP's own trailing baseline:

- `transition_next_12m` (binary): 1 = upward transition (composite score ≥ p75)
- `transition_direction` (3-class): +1 up / 0 stable / −1 down

Class split: 25% up / 50% stable / 25% down.

### 2. Clustering (unsupervised)
K-means (k=5) on the same 18 features used in the prediction model (means over Jun 2019–Dec 2024) identifies five neighborhood market tendencies. Using the model's own feature set improved silhouette from 0.11 to 0.16. These are **overlapping archetypes on a spectrum**, not hard mutually-exclusive categories. Cluster labels are static — they represent a ZIP's long-run market character, separate from short-term directional predictions.

| Archetype | ZIPs | Character |
|---|---|---|
| 💎 High-Value, Tight Supply | 74 | Highest median values (~$318K), only cluster with declining inventory (−8% YoY), high off-market rate (41%) |
| 🧊 Low-Activity, Cooling | 44 | Lowest values (~$162K), weakest appreciation (7.1% YoY), highest inventory growth (+14% YoY) |
| 🔥 Competitive Mid-Market | 211 | Mid-range values (~$266K), highest above-list rate (37.6%), fastest DOM (45 days) |
| ⚡ Affordable, High Appreciation | 172 | Entry-level (~$216K), strongest YoY appreciation (12.8%), rising inventory signals growing interest |
| 📦 Moderate Value, Supply Growing | 154 | Mid-range (~$213K), rising inventory (+8% YoY), slow DOM (76 days), softest competition |

### 3. Supervised modeling
Walk-forward validation only — no random splits on time-ordered panel data:

| Fold | Train | Test | AUC | Notes |
|---|---|---|---|---|
| A | Jun 2019–Dec 2020 | Jan–Dec 2021 | 0.8443 | |
| B | Jun 2019–Dec 2021 | Jan–Dec 2022 | 0.8153 | Fed rate-hike year |
| C (primary) | Jun 2019–Dec 2022 | Jan–Dec 2023 | **0.8652** | Primary validation |
| D (retrain) | Jun 2019–Dec 2023 | Jan–Dec 2024 | 0.7240 | Powers 2025 signal |
| Actuals | Jun 2019–Dec 2022 | 2025 real outcomes | 0.7747 | 2024 preds vs. actual 2025 transitions |

Two models were evaluated:

| Model | Features | Test year | AUC | Macro F1 |
|---|---|---|---|---|
| V3 (All Sources) | 39 (Zillow + Redfin + ACS + CBP + IRS) | 2021 | 0.8519 | 0.5076 |
| **Monthly-Only** ✓ | **18 (Zillow + Redfin only)** | **2023** | **0.8652** | **0.3838** |

The monthly-only model wins on AUC. Removing the 17 annual features (ranked #12–#27 in permutation importance) forced RFECV to find better monthly substitutes and eliminated data-lag noise. It also enables future extension to Redfin data back to 2012 (4× more training data) and can be updated whenever new Redfin/Zillow data is available.

### 4. Key features (RFECV-selected, monthly-only model)
All 18 features are Zillow and Redfin monthly signals — home value momentum, inventory dynamics, and sales activity. No demographic or business data survived selection.

| Feature | What it captures |
|---|---|
| `home_value_vs_baseline_lag6` | Price level vs. ZIP's own 6-month-lagged norm — strongest single signal (SHAP rank 1) |
| `home_value_mom_pct` | Month-over-month % change in home value — current price momentum (SHAP rank 2) |
| `home_value_mom_pct_lag6` | Home value MoM change 6 months prior — lagged momentum confirmation (SHAP rank 3) |
| `pct_sold_above_list_lag6` | Lagged demand heat — were homes recently selling above asking? (SHAP rank 4) |
| `dom_vs_baseline` | Days on market vs. ZIP's own historical norm — unusually fast/slow turnover (SHAP rank 5) |

### 5. Forward predictions (2025)
The model was retrained on Jun 2019–Dec 2024 (incorporating validated 2024 transition labels) and scored against November 2025 Zillow and Redfin features. The forward window is **Dec 2025–Nov 2026**. The prior 2024 signal was retrospectively validated against actual 2025 outcomes (AUC 0.7747), confirming the model generalises across market cycles. The 2025 predictions cannot be verified until late 2026 — use for prospecting and early positioning.

---

## Repository structure

```
Property_Investment_Helper/
├── app.py                              # Streamlit dashboard (8 pages, investor + educator modes)
├── notebooks/
│   ├── exploration.ipynb               # Data quality audit and ZIP coverage filter
│   ├── data_prep_and_eda.ipynb         # Consolidation, EDA, modeling-ready dataset
│   ├── feature_engineering.ipynb       # Lag features, rolling averages, baseline deltas
│   ├── target_engineering.ipynb        # Composite transition score + binary/3-class targets
│   ├── clustering.ipynb                # K-means neighborhood archetypes (k=5)
│   └── modeling.ipynb                  # Walk-forward GB models, RFECV, SHAP
├── scripts/
│   ├── run_monthly_model.py            # Monthly-only model training + model_comparison.json
│   ├── generate_forward_predictions.py # 2024 forward signal generation (historical)
│   ├── validate_2024_predictions.py    # Validates 2024 predictions against actual 2025 outcomes
│   ├── retrain_and_predict_2025.py     # Retrains on 2019–2024, generates 2025 forward signal
│   └── ...                             # Data ingestion scripts
├── data/
│   ├── processed/                      # Modeling datasets and predictions (not committed — see data_samples/)
│   └── geo/                            # GeoJSON boundaries for 655 NC/SC ZIPs and state outlines
├── data_samples/                       # 50-row samples of all processed files for reference
├── pyproject.toml
├── uv.lock
└── CLAUDE.md                           # Project context for Claude Code
```

---

## Setup

Requires Python 3.12. Uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Install shap separately (cross-platform build conflict prevents uv add)
uv pip install shap

# Run the dashboard
uv run streamlit run app.py

# Re-execute a notebook
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb --ExecutePreprocessor.timeout=600
```

---

## Limitations

- ZIP code analysis is aggregated — does not capture block-level variation within a ZIP
- Transition labels are engineered proxies, not pre-labeled ground truth
- Neighborhood archetypes (clusters) are tendencies on a spectrum, not hard categories — silhouette score of 0.16 (improved from 0.11 by using the same 18 model features) still reflects that real estate markets don't fall into discrete buckets
- The 2022 AUC dip (0.815) is expected — the Fed rate-hike cycle created abrupt market conditions outside the 2019–2021 training distribution; the 2023 recovery (0.865) shows the model captures durable signals, not COVID-era noise
- The 2024 predictions were validated against actual 2025 outcomes (AUC 0.7747) — a graceful degradation into an unseen rate-cycle year, confirming durable signal
- The 2025 forward predictions (Dec 2025–Nov 2026 window) cannot be verified until late 2026
- Results should be interpreted as **probabilistic early-warning signals**, not causal proof of neighborhood change

---

## Why this framing matters

We deliberately avoid calling the upward transition probability a "risk score." In finance, *risk* implies downside. A high opportunity score means a neighborhood is likely to heat up — which is a **buy signal** for investors, not a warning. The directional model separates the two: the blue end of the direction map represents genuine cooling risk; the red end represents opportunity.
