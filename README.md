# Property Investment Helper

An early-warning analytics project for identifying ZIP codes that may be entering a neighborhood transition phase.

This repository is being built as a **data-driven business intelligence and predictive modeling project** focused on North Carolina and South Carolina ZIP codes. The current goal is to combine housing-market signals with demographic, business, tax, and mortgage-rate context to answer a practical question:

> Which ZIP codes are showing early warning signs of neighborhood transition, and which ones are most likely to continue changing over the next 12 months?

---

## Project status

**Work in progress.**

The project foundation is already in place:
- raw Zillow sample data is in the repo
- scripts are being developed for data ingestion and processing
- the integrated modeling dataset has already been narrowed to a high-quality final ZIP universe
- the modeling and reporting layer is still being built

This README is meant to document the project direction, current structure, and next steps so the repo stays organized as development continues.

---

## Problem statement

Neighborhood change usually does not happen all at once. It tends to appear through a combination of signals such as:
- rising home values
- stronger sale prices
- lower inventory
- faster sales
- higher sale-to-list ratios
- improving local income or employment context

Rather than trying to claim a perfect prediction of "gentrification," this project frames the task as a **short-horizon early-warning problem**.

The main idea is to use historical ZIP-level data to:
1. identify neighborhoods currently under market pressure,
2. group ZIP codes into meaningful change patterns,
3. predict whether a ZIP is likely to enter a **transition-like state in the next 12 months**.

---

## Main objectives

The project is built around four core objectives:

### 1. Build an integrated ZIP-month modeling table
Combine monthly and annual public datasets into one reproducible panel dataset for NC/SC ZIP codes.

### 2. Engineer leading indicators of neighborhood transition
Create trend, momentum, affordability, and market-competition features that can act as early warning signals.

### 3. Discover neighborhood change typologies
Use unsupervised learning to group ZIP codes into interpretable categories based on their multi-year trajectories.

### 4. Predict near-term transition risk
Use supervised learning to estimate whether a ZIP is likely to transition within the next 12 months.

---

## Data sources

The current project design uses six primary sources:

### 1. Zillow Research Data
Used for:
- ZIP-level home value trends
- the NC/SC ZIP universe reference table

### 2. Redfin Market Tracker
Used for monthly housing-market activity and competitiveness measures such as:
- median sale price
- median list price
- median price per square foot
- homes sold
- pending sales
- new listings
- inventory
- median days on market
- average sale-to-list ratio
- sold above list share
- off-market-in-two-weeks share

### 3. American Community Survey (ACS) 5-year estimates
Used for annual ZIP-level demographic and housing context such as:
- median household income
- median gross rent
- median owner-occupied home value
- owner occupancy rate

### 4. County Business Patterns (CBP)
Used for annual business and employment structure such as:
- establishments
- employment
- annual payroll
- first-quarter payroll
- derived pay-per-employee measures

### 5. IRS Statistics of Income (SOI) ZIP code data
Used for annual ZIP-level income and tax-return indicators.

### 6. Freddie Mac Primary Mortgage Market Survey (PMMS)
Used for mortgage-rate context and market-wide financing conditions.

---

## Current data scope

The working project scope is currently:
- **Geography:** North Carolina and South Carolina ZIP codes
- **Final analytic ZIP universe:** 540 ZIPs
- **Time span:** January 2018 through December 2022
- **Time grain:** monthly panel structure

This final filtered sample is intentionally smaller than the original ZIP universe because the project keeps only ZIPs with enough usable coverage across the selected data sources to support modeling and comparison.

---

## Target variable

The main supervised target is planned as:

### `transition_next_12m`
A binary label indicating whether a ZIP shows a **transition-like pattern over the following 12 months**.

This target is not taken directly from a raw dataset. It is **engineered** from future observed behavior using a composite of history-based signals such as:
- sustained home-value growth relative to the ZIP's own recent history
- sustained sale-price growth relative to the ZIP's own recent history
- tightening inventory or shorter time on market
- stronger sale-to-list behavior or related market pressure indicators

This makes the problem more realistic and more defensible than pretending there is a perfect pre-labeled "transition" column.

---

## Planned methodology

### 1. Data integration and preparation
The pipeline will:
- standardize ZIP identifiers and dates
- align monthly housing data with annual socioeconomic and business indicators
- create a clean ZIP-month panel for analysis

### 2. Feature engineering
Existing and planned feature families include:
- month-over-month and year-over-year growth rates
- rolling averages and rolling volatility
- trend acceleration measures
- affordability pressure ratios
- inventory tightening indicators
- competitiveness indicators
- relative change versus each ZIP's own history

### 3. Unsupervised learning
Unsupervised methods will be used to identify **types of neighborhood change**, not just predict outcomes.

Planned use:
- cluster ZIPs by multi-year trend summaries
- identify neighborhood archetypes
- compare stable, emerging, fast-heating, or cooling ZIP patterns

Candidate methods:
- K-means
- hierarchical clustering
- possibly PCA/UMAP for visualization

### 4. Supervised learning
Supervised models will be used to predict `transition_next_12m`.

Planned model path:
- logistic regression baseline
- random forest
- gradient boosting / XGBoost style model if appropriate

### 5. Time-aware validation
Because this is a time-based panel dataset, the project will **not** use a random row split.

Instead, it will use **walk-forward validation**, for example:
- train on earlier years
- validate on the next year
- test on the final year

This is important to avoid leakage from the future into the past.

---

## Repository structure

Current repo structure:

```text
Property_Investment_Helper/
├── data_samples/
│   └── raw/
│       └── zillow/
├── scripts/
├── README.md
├── pyproject.toml
└── uv.lock
```

Planned structure as the project grows:

```text
Property_Investment_Helper/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── data_samples/
│   └── raw/
│       └── zillow/
├── notebooks/
├── scripts/
│   ├── pull/
│   ├── process/
│   ├── features/
│   └── modeling/
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── maps/
├── README.md
├── pyproject.toml
└── uv.lock
```

---

## Example outputs

Planned outputs include:
- a clean integrated modeling table at the ZIP-month level
- a transition risk score or probability for each ZIP
- neighborhood typology clusters
- feature importance summaries
- charts and maps showing where pressure is rising
- a final report or dashboard for interpretation

---

## Why this project matters

This project is meant to be useful for more than just a class grade.

A system like this could help:
- investors identify emerging areas earlier
- analysts compare neighborhood trajectories more systematically
- planners monitor housing pressure and market shifts
- decision-makers build watchlists rather than rely on guesswork

It also creates a strong real-world analytics portfolio piece because it combines:
- multi-source data integration
- panel/time-based feature engineering
- unsupervised learning
- supervised modeling
- business interpretation

---

## Current limitations

This project still has important limitations:
- ZIP code analysis is aggregated and does not capture block-level variation
- transition is a constructed proxy, not a perfect ground-truth label
- annual public indicators may lag fast local change
- results should be interpreted as early warning signals, not causal proof

---

## Next steps

Planned next development steps:
- finalize the processed modeling dataset
- define the exact target engineering rules for `transition_next_12m`
- build the clustering pipeline
- train the first baseline predictive models
- evaluate with walk-forward validation
- generate maps, tables, and model interpretation outputs

---

## Getting started

This section will be expanded as the repo matures. A likely workflow will look like:

```bash
uv sync
uv run python scripts/<your_script>.py
```

As the project is cleaned up, this README will be updated with:
- exact setup instructions
- data pipeline order
- script entry points
- model training steps
- reproducibility notes

---

## Notes

This repository is under active development. Some file paths, scripts, and outputs will change as the project moves from data assembly to modeling and reporting.
