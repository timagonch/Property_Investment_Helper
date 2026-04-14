"""
NC/SC Neighborhood Market Transition Monitor
Walk-forward validated Gradient Boosting model (AUC 0.865, test year 2023).
655 ZIPs · Jun 2019–Nov 2025 · Dec 2025–Nov 2026 forward window.
"""

import json
import os

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NC/SC Neighborhood Market Transition",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data", "processed")
GEO  = os.path.join(BASE, "data", "geo")

# ── Constants ───────────────────────────────────────────────────────────────────
CLUSTER_COLORS = {
    "high_value_appreciating": "#9b2226",
    "low_activity_cooling":    "#457b9d",
    "competitive_mid_market":  "#e63946",
    "affordable_high_demand":  "#e9c46a",
    "moderate_supply_growing": "#2a9d8f",
}
CLUSTER_LABELS = {
    "high_value_appreciating": "💎 High-Value, Tight Supply",
    "low_activity_cooling":    "🧊 Low-Activity, Cooling",
    "competitive_mid_market":  "🔥 Competitive Mid-Market",
    "affordable_high_demand":  "⚡ Affordable, High Appreciation",
    "moderate_supply_growing": "📦 Moderate Value, Supply Growing",
}
CLUSTER_DESCRIPTIONS = {
    "high_value_appreciating": (
        "74 ZIPs · Highest median home values (~$318K) with the only meaningfully declining "
        "inventory of any cluster (-8% YoY). High off-market rate (41%), moderate DOM (56 days). "
        "Premium ZIPs where demand consistently absorbs supply — durable appreciation signal."
    ),
    "low_activity_cooling": (
        "44 ZIPs · Lowest home values (~$162K) and weakest appreciation (7.1% YoY). "
        "Highest inventory growth (+14% YoY) and slowest sales pace (76-day DOM, 22% above-list). "
        "Momentum is softest here — lowest upside for investors, most buyer-friendly market."
    ),
    "competitive_mid_market": (
        "211 ZIPs · Mid-range values (~$266K) with the highest above-list rate (37.6%) "
        "and fastest DOM (45 days). Strongest competition of any cluster — homes move quickly "
        "and inventory is stable. A balanced hot market with broad geographic representation."
    ),
    "affordable_high_demand": (
        "172 ZIPs · Entry-level prices (~$216K) with the strongest YoY appreciation (12.8%). "
        "Moderate transaction pace (63-day DOM) but rapidly rising inventory signals growing "
        "developer and investor interest. Best combination of affordability and momentum."
    ),
    "moderate_supply_growing": (
        "154 ZIPs · Mid-range values (~$213K) with rising inventory (+8% YoY) and slow sales "
        "pace (76-day DOM, 24% above-list). Supply is outpacing demand — appreciation is "
        "holding (10.1% YoY) but buyers have more options than in tighter clusters."
    ),
}

ZIP_COMPARISON_COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"]

# Features available in features_full.csv (18 model features from score_full_panel.py)
SELECTED_FEATURES = [
    "home_value_mom_pct", "home_value_accel", "home_value_vs_baseline",
    "home_value_mom_3m_avg", "home_value_mom_12m_avg", "inventory_mom_12m_avg",
    "avg_sale_to_list_ratio_lag6", "pct_sold_above_list_lag6", "inventory_yoy_pct",
    "dom_vs_baseline",
]

FEATURE_LABELS = {
    # Home value momentum
    "home_value_index":            "Home Value Index",
    "home_value_mom_pct":          "Home Value MoM %",
    "home_value_yoy_pct":          "Home Value YoY %",
    "home_value_accel":            "Home Value Acceleration",
    "home_value_mom_3m_avg":       "Home Value MoM 3m Avg",
    "home_value_mom_12m_avg":      "Home Value MoM 12m Avg",
    "home_value_mom_3m_std":       "Home Value MoM Volatility",
    "home_value_mom_pct_lag3":     "Home Value MoM % (3m lag)",
    "home_value_mom_pct_lag6":     "Home Value MoM % (6m lag)",
    "home_value_vs_baseline":      "Value vs Baseline",
    "home_value_vs_baseline_lag3": "Value vs Baseline (3m lag)",
    "home_value_vs_baseline_lag6": "Value vs Baseline (6m lag)",
    # Inventory
    "inventory":                   "Inventory",
    "inventory_yoy_pct":           "Inventory YoY %",
    "inventory_mom_3m_avg":        "Inventory MoM 3m Avg",
    "inventory_mom_12m_avg":       "Inventory Trend (12m avg)",
    "inventory_vs_baseline":       "Inventory vs Baseline",
    "inventory_vs_baseline_lag3":  "Inventory vs Baseline (3m lag)",
    # Sales activity
    "homes_sold":                  "Homes Sold",
    "new_listings":                "New Listings",
    "median_days_on_market":       "Median Days on Market",
    "dom_vs_baseline":             "Days on Market vs Baseline",
    "avg_sale_to_list_ratio":      "Sale-to-List Ratio",
    "avg_sale_to_list_ratio_lag6": "Sale-to-List Ratio (6m lag)",
    "sale_to_list_3m_avg":         "Sale-to-List 3m Avg",
    "pct_sold_above_list":         "% Sold Above List",
    "pct_sold_above_list_lag3":    "% Sold Above List (3m lag)",
    "pct_sold_above_list_lag6":    "% Sold Above List (6m lag)",
    "pct_off_market_in_2wks":      "% Off-Market in 2 Weeks",
    "off_market_2wk_3m_avg":       "% Off-Market 2wk 3m Avg",
    # Price levels
    "median_sale_price":           "Median Sale Price",
    "median_list_price":           "Median List Price",
    "median_price_per_sqft":       "Median Price per Sq Ft",
    # Annual features (V3 only)
    "median_owner_home_value":     "Median Owner Home Value",
    "median_household_income":     "Median Household Income",
    "median_gross_rent":           "Median Gross Rent",
    "owner_occupancy_rate":        "Owner Occupancy Rate",
    "business_establishments":     "Business Establishments",
    "total_employment":            "Total Employment",
    "avg_pay_per_employee":        "Avg Pay per Employee",
    "tax_returns_filed":           "Tax Returns Filed",
    "adjusted_gross_income":       "Adjusted Gross Income",
    "taxable_interest_income":     "Taxable Interest Income",
    "price_to_income_ratio":       "Price-to-Income Ratio",
    "price_to_rent_ratio":         "Price-to-Rent Ratio",
    "rent_to_income_ratio":        "Rent-to-Income Ratio",
}


# ── Data loaders ────────────────────────────────────────────────────────────────
@st.cache_data
def load_model_comparison():
    path = os.path.join(DATA, "model_comparison.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _winner_prefix():
    comp = load_model_comparison()
    if comp is None:
        return "v3"
    return comp.get("winner", "v3")


@st.cache_data
def load_predictions():
    prefix = _winner_prefix()
    fname = "model_predictions_monthly.csv" if prefix == "monthly" else "model_predictions_v3.csv"
    df = pd.read_csv(os.path.join(DATA, fname))
    df["month"] = pd.to_datetime(df["month"])
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    return df


@st.cache_data
def load_multiclass_predictions():
    prefix = _winner_prefix()
    fname = "model_predictions_multiclass_monthly.csv" if prefix == "monthly" else "model_predictions_multiclass.csv"
    df = pd.read_csv(os.path.join(DATA, fname))
    df["month"] = pd.to_datetime(df["month"])
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    return df


@st.cache_data
def load_shap():
    prefix = _winner_prefix()
    fname = "shap_values_monthly.csv" if prefix == "monthly" else "shap_values.csv"
    df = pd.read_csv(os.path.join(DATA, fname))
    df["month"] = pd.to_datetime(df["month"])
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    return df


@st.cache_data
def load_v3_predictions():
    df = pd.read_csv(os.path.join(DATA, "model_predictions_v3.csv"))
    df["month"] = pd.to_datetime(df["month"])
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    return df


@st.cache_data
def load_v3_multiclass():
    df = pd.read_csv(os.path.join(DATA, "model_predictions_multiclass.csv"))
    df["month"] = pd.to_datetime(df["month"])
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    return df


@st.cache_data
def load_monthly_predictions():
    df = pd.read_csv(os.path.join(DATA, "model_predictions_monthly.csv"))
    df["month"] = pd.to_datetime(df["month"])
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    return df


@st.cache_data
def load_monthly_multiclass():
    df = pd.read_csv(os.path.join(DATA, "model_predictions_multiclass_monthly.csv"))
    df["month"] = pd.to_datetime(df["month"])
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    return df


@st.cache_data
def load_forward_predictions():
    df = pd.read_csv(os.path.join(DATA, "model_predictions_forward_2025.csv"))
    df["month"] = pd.to_datetime(df["month"])
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    return df


@st.cache_data
def load_forward_multiclass():
    df = pd.read_csv(os.path.join(DATA, "model_predictions_multiclass_forward_2025.csv"))
    df["month"] = pd.to_datetime(df["month"])
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    return df


@st.cache_data
def load_features():
    df = pd.read_csv(os.path.join(DATA, "modeling_dataset_with_clusters.csv"))
    df["month"] = pd.to_datetime(df["month"])
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    return df


@st.cache_data
def load_full_predictions():
    """Full scored panel Jun 2019 – Nov 2025."""
    df = pd.read_csv(os.path.join(DATA, "model_predictions_full.csv"))
    df["month"] = pd.to_datetime(df["month"])
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    return df


@st.cache_data
def load_full_multiclass():
    """Full multiclass scored panel Jun 2019 – Nov 2025."""
    df = pd.read_csv(os.path.join(DATA, "model_predictions_multiclass_full.csv"))
    df["month"] = pd.to_datetime(df["month"])
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    return df


@st.cache_data
def load_full_features():
    """Full feature panel Jun 2019 – Nov 2025 (18 model features)."""
    df = pd.read_csv(os.path.join(DATA, "features_full.csv"))
    df["month"] = pd.to_datetime(df["month"])
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    return df


@st.cache_data
def load_full_shap():
    """Full SHAP values Jun 2019 – Nov 2025 for all scored rows."""
    df = pd.read_csv(os.path.join(DATA, "shap_values_full.csv"))
    df["month"] = pd.to_datetime(df["month"])
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    return df


@st.cache_data
def build_validation_table():
    """Per-ZIP validation summary for the 2023 test year (Fold C, primary evaluation).

    Returns one row per ZIP with:
      - avg predicted opportunity score
      - actual upward transition rate (fraction of months that were labeled 1)
      - predicted label (1 if avg score >= 0.50)
      - actual label (majority vote across the 12 test months)
      - correct (predicted == actual majority)
    """
    df = pd.read_csv(os.path.join(DATA, "model_predictions_monthly.csv"))
    df["month"] = pd.to_datetime(df["month"])
    df["zip"]   = df["zip"].astype(str).str.zfill(5)

    # Fold C test year is 2023
    df = df[df["month"].dt.year == 2023]

    cities = load_city_lookup()

    summary = (
        df.groupby(["zip", "cluster_name"])
        .agg(
            avg_score=("prob_transition", "mean"),
            peak_score=("prob_transition", "max"),
            actual_any=("transition_next_12m", "max"),   # 1 if ANY month was upward
            cluster=("cluster", "first"),
        )
        .reset_index()
    )
    # Predicted: did the model think this ZIP was high-opportunity on average?
    summary["predicted_label"] = (summary["avg_score"] >= 0.50).astype(int)
    # Actual: did ANY month in 2023 cross the upward-transition threshold?
    summary["actual_label"]    = summary["actual_any"].fillna(0).astype(int)
    summary["correct"]         = summary["predicted_label"] == summary["actual_label"]
    summary["cluster_label"]   = summary["cluster_name"].map(CLUSTER_LABELS)
    summary["state"]           = summary["zip"].apply(_state_label)
    summary["city"]            = summary["zip"].map(cities).fillna("")
    summary["Avg Opp Score (%)"] = (summary["avg_score"] * 100).round(1)
    summary["Peak Opp Score (%)"]= (summary["peak_score"] * 100).round(1)
    summary["Predicted"]       = summary["predicted_label"].map({1: "Up", 0: "Stable/Down"})
    summary["Actual"]          = summary["actual_label"].map({1: "Up ✓", 0: "Stable/Down"})
    summary["Match"]           = summary["correct"].map({True: "✓", False: "✗"})
    return summary


@st.cache_data
def load_city_lookup():
    df = pd.read_csv(os.path.join(DATA, "zip_city_lookup.csv"))
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    # Returns dict: zip -> "City, ST"
    return {row["zip"]: f"{row['city']}, {row['State']}" for _, row in df.iterrows()}


@st.cache_data
def load_geojson():
    with open(os.path.join(GEO, "nc_sc_zips.geojson")) as f:
        return json.load(f)


@st.cache_data
def load_state_geojson():
    with open(os.path.join(GEO, "nc_sc_states.geojson")) as f:
        return json.load(f)


def state_border_color():
    try:
        theme = st.get_option("theme.base")
        return "#ffffff" if theme == "dark" else "#222222"
    except Exception:
        return "#222222"


def add_state_overlays(fig):
    state_geo = load_state_geojson()
    border_color = state_border_color()
    fig.add_trace(go.Choropleth(
        geojson=state_geo,
        locations=["37", "45"],
        featureidkey="properties.STATEFP",
        z=[0, 0],
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
        marker_line_color=border_color,
        marker_line_width=2.5,
        showscale=False,
        hoverinfo="skip",
        name="",
    ))
    return fig


def _state_label(z):
    return "NC" if z[:3] in [str(x) for x in range(270, 290)] else "SC"


@st.cache_data
def build_forward_zip_summary():
    """ZIP-level summary built from the 2025 forward predictions (Nov 2025 snapshot)."""
    preds = load_forward_predictions()
    preds = preds[preds["month"] == preds["month"].max()]
    zip_df = (
        preds.groupby("zip")
        .agg(
            prob_transition=("prob_transition", "mean"),
            cluster=("cluster", "first"),
            cluster_name=("cluster_name", "first"),
        )
        .reset_index()
    )
    zip_df["risk_pct"]      = (zip_df["prob_transition"] * 100).round(1)
    zip_df["cluster_label"] = zip_df["cluster_name"].map(CLUSTER_LABELS)
    zip_df["state"]         = zip_df["zip"].apply(_state_label)
    cities = load_city_lookup()
    zip_df["city"] = zip_df["zip"].map(cities).fillna("")
    return zip_df


@st.cache_data
def build_forward_zip_multiclass_summary():
    """ZIP-level multiclass summary from the 2025 forward predictions."""
    preds = load_forward_multiclass()
    preds = preds[preds["month"] == preds["month"].max()]
    zip_df = (
        preds.groupby("zip")
        .agg(
            prob_down=("prob_down", "mean"),
            prob_stable=("prob_stable", "mean"),
            prob_up=("prob_up", "mean"),
            net_direction=("net_direction", "mean"),
            cluster=("cluster", "first"),
            cluster_name=("cluster_name", "first"),
        )
        .reset_index()
    )
    zip_df["cluster_label"] = zip_df["cluster_name"].map(CLUSTER_LABELS)
    zip_df["state"]         = zip_df["zip"].apply(_state_label)
    cities = load_city_lookup()
    zip_df["city"] = zip_df["zip"].map(cities).fillna("")
    return zip_df


@st.cache_data
def build_zip_summary(model: str = "winner"):
    if model == "v3":
        preds = load_v3_predictions()
    elif model == "monthly":
        preds = load_monthly_predictions()
    else:
        preds = load_predictions()

    preds = preds[preds["month"] == preds["month"].max()]

    zip_df = (
        preds.groupby("zip")
        .agg(
            prob_transition=("prob_transition", "mean"),
            actual_transitions=("transition_next_12m", "mean"),
            cluster=("cluster", "first"),
            cluster_name=("cluster_name", "first"),
            n_months=("month", "count"),
        )
        .reset_index()
    )
    zip_df["risk_pct"] = (zip_df["prob_transition"] * 100).round(1)
    zip_df["cluster_label"] = zip_df["cluster_name"].map(CLUSTER_LABELS)
    zip_df["state"] = zip_df["zip"].apply(_state_label)
    cities = load_city_lookup()
    zip_df["city"] = zip_df["zip"].map(cities).fillna("")
    return zip_df


@st.cache_data
def build_zip_multiclass_summary(model: str = "winner"):
    if model == "v3":
        preds = load_v3_multiclass()
    elif model == "monthly":
        preds = load_monthly_multiclass()
    else:
        preds = load_multiclass_predictions()

    preds = preds[preds["month"] == preds["month"].max()]

    zip_df = (
        preds.groupby("zip")
        .agg(
            prob_down=("prob_down", "mean"),
            prob_stable=("prob_stable", "mean"),
            prob_up=("prob_up", "mean"),
            net_direction=("net_direction", "mean"),
            cluster=("cluster", "first"),
            cluster_name=("cluster_name", "first"),
            actual_direction=("transition_direction", "mean"),
        )
        .reset_index()
    )
    zip_df["cluster_label"] = zip_df["cluster_name"].map(CLUSTER_LABELS)
    zip_df["state"] = zip_df["zip"].apply(_state_label)
    cities = load_city_lookup()
    zip_df["city"] = zip_df["zip"].map(cities).fillna("")
    return zip_df


# ── Global CSS ──────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* Hide Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ── Metric cards ─────────────────────────── */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        border: 1px solid #e9ecef !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        color: #6c757d !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        color: #212529 !important;
    }

    /* ── Headings ─────────────────────────────── */
    h1 { font-weight: 800 !important; letter-spacing: -0.5px !important; }
    h2 { font-weight: 700 !important; }
    h3 { font-weight: 600 !important; }

    /* ── Dividers ─────────────────────────────── */
    hr { border: none !important; border-top: 1.5px solid #f0f2f5 !important; margin: 1.5rem 0 !important; }

    /* ── DataFrames ───────────────────────────── */
    [data-testid="stDataFrame"] iframe { border-radius: 8px !important; }

    /* ── Tabs (if used) ───────────────────────── */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important;
        font-weight: 600 !important;
    }

    /* ── Streamlit info/warning/success boxes ─── */
    [data-testid="stAlert"] { border-radius: 10px !important; }

    /* ── Segmented control (mode switcher) ───── */
    [data-testid="stSegmentedControl"] {
        width: 100% !important;
        background: rgba(255,255,255,0.08) !important;
        border-radius: 20px !important;
        padding: 3px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }
    [data-testid="stSegmentedControl"] button {
        border-radius: 16px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        flex: 1 !important;
        transition: all 0.18s ease !important;
    }
    [data-testid="stSegmentedControl"] button[aria-selected="true"] {
        background: #e63946 !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(230,57,70,0.45) !important;
    }

    /* ── Sidebar polish ───────────────────────── */
    section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem !important; }

    /* ── Archetype cards ──────────────────────── */
    .archetype-card {
        border-radius: 12px;
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        border: 1px solid rgba(0,0,0,0.07);
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s ease;
    }
    .archetype-card:hover { box-shadow: 0 6px 18px rgba(0,0,0,0.11) !important; }

    /* ── Page header strip ────────────────────── */
    .page-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 100%);
        padding: 1.25rem 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.75rem;
    }

    /* ── Live badge pulse ─────────────────────── */
    @keyframes pulse-ring {
        0%   { box-shadow: 0 0 0 0 rgba(244,162,97,0.55); }
        70%  { box-shadow: 0 0 0 7px rgba(244,162,97,0); }
        100% { box-shadow: 0 0 0 0 rgba(244,162,97,0); }
    }
    .live-badge { animation: pulse-ring 1.8s ease infinite; }
    </style>
    """, unsafe_allow_html=True)


# ── Page header helper ──────────────────────────────────────────────────────────
def page_header(icon, title, subtitle, accent="#e63946", badge=None):
    badge_html = ""
    if badge:
        badge_html = (
            f'<span class="live-badge" style="display:inline-block; background:{accent}; '
            f'color:#fff; font-size:0.62rem; font-weight:800; padding:3px 9px; '
            f'border-radius:20px; letter-spacing:1.5px; vertical-align:middle; '
            f'margin-left:10px; text-transform:uppercase;">{badge}</span>'
        )
    st.markdown(f"""
    <div class="page-header" style="border-left: 5px solid {accent};">
        <div style="display:flex; align-items:center; gap:1rem;">
            <span style="font-size:2rem; line-height:1; flex-shrink:0;">{icon}</span>
            <div>
                <div style="font-size:1.3rem; font-weight:800; color:#fff; letter-spacing:-0.3px; line-height:1.2;">
                    {title}{badge_html}
                </div>
                <div style="font-size:0.8rem; color:#a8b2c1; margin-top:0.3rem;">{subtitle}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Home Page ───────────────────────────────────────────────────────────────────
def page_home():
    comp = load_model_comparison()
    winner = comp.get("winner", "monthly") if comp else "monthly"
    m = comp[winner] if comp else {}

    # ── Hero banner ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 55%, #243447 100%);
        padding: 3rem 2.8rem 2.6rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute; top: -60px; right: -60px;
            width: 260px; height: 260px; border-radius: 50%;
            background: rgba(230,57,70,0.12); pointer-events: none;
        "></div>
        <div style="
            font-size: 0.78rem; letter-spacing: 2.5px; text-transform: uppercase;
            color: #e63946; font-weight: 700; margin-bottom: 0.6rem; text-align: center;
        ">NC / SC Real Estate Intelligence &nbsp;·&nbsp; Spring 2026</div>
        <h1 style="
            color: #ffffff; font-size: 2.6rem; font-weight: 800;
            margin: 0 0 1rem; letter-spacing: -1px; line-height: 1.15; text-align: center;
        ">Neighborhood Market<br>Transition Monitor</h1>
        <p style="
            color: #a8b2c1; font-size: 1.05rem; margin: 0 auto;
            max-width: 600px; line-height: 1.7; text-align: center;
        ">
            An early-warning system that identifies NC &amp; SC ZIP codes showing leading signs
            of neighborhood transition — months before they appear in headline metrics.
            Built with Zillow, Redfin, and a walk-forward validated Gradient Boosting model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Key stats ────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ZIP codes analyzed", "655")
    c2.metric("Validation AUC (2023 test)", f"{m.get('auc', 0.865):.3f}")
    c3.metric("Data coverage", "Jun 2019 – Nov 2025")
    c4.metric("Forward window", "Dec 2025 – Nov 2026")

    st.divider()

    # ── What is a neighborhood transition? ──────────────────────────────────────
    st.subheader("What is a neighborhood transition?")
    st.markdown(
        "A neighborhood transition is a **sustained, directional shift** in a ZIP code's housing "
        "market dynamics — typically visible in momentum data months before it shows up in prices."
    )

    col1, col2, col3 = st.columns(3)
    cards = [
        (col1, "#e63946", "#fff5f5", "#ffe3e3", "📈", "Upward Transition",
         "Prices accelerating, inventory tightening, buyers competing above list price. "
         "Strong momentum signal — opportunity for early entry."),
        (col2, "#6c757d", "#f8f9fa", "#e9ecef", "➡️", "Stable",
         "Normal seasonal variation. No clear directional momentum. "
         "Market is in equilibrium — hold or monitor."),
        (col3, "#457b9d", "#f0f7ff", "#dbeafe", "📉", "Cooling / Downward",
         "Prices softening, inventory growing, homes sitting longer. "
         "Demand is retreating — exercise caution."),
    ]
    for col, border, bg1, bg2, icon, title, desc in cards:
        with col:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {bg1}, {bg2});
                border-left: 5px solid {border};
                padding: 1.4rem 1.5rem;
                border-radius: 10px;
                min-height: 170px;
            ">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                <h4 style="color: {border}; margin: 0 0 0.5rem; font-size: 1rem;">{title}</h4>
                <p style="margin: 0; color: #4a5568; font-size: 0.875rem; line-height: 1.6;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── How the target is built ──────────────────────────────────────────────────
    st.subheader("How the target variable is built")
    st.markdown(
        "The model predicts whether a ZIP will enter an upward transition **over the next 12 months**, "
        "defined by a composite of three forward-looking signals. A ZIP is labeled **upward** if its "
        "composite score exceeds the 75th percentile across all ZIPs."
    )

    tc1, tc2, tc3 = st.columns(3)
    for col, icon, title, sub in [
        (tc1, "🏠", "Home Value Growth",
         "12-month forward avg YoY growth vs. each ZIP's own trailing baseline"),
        (tc2, "📦", "Inventory Tightening",
         "Falling supply relative to baseline — negated so tightening = positive signal"),
        (tc3, "⚔️", "Market Competitiveness",
         "% of homes sold above list price — forward 12-month average"),
    ]:
        with col:
            st.markdown(f"""
            <div style="
                text-align: center; padding: 1.2rem 1rem;
                background: #f8fafc; border-radius: 10px;
                border: 1px solid #e9ecef; border-top: 3px solid #e63946;
            ">
                <div style="font-size: 1.8rem; margin-bottom: 0.4rem;">{icon}</div>
                <div style="font-weight: 700; color: #212529; margin-bottom: 0.35rem; font-size: 0.95rem;">{title}</div>
                <div style="font-size: 0.82rem; color: #6c757d; line-height: 1.55;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.caption(
        "Each signal is z-scored relative to all ZIPs in the same month, then equally weighted. "
        "Top 25% composite = upward (1), bottom 25% = downward (−1), middle 50% = stable (0)."
    )

    st.divider()

    # ── How to use the tool ──────────────────────────────────────────────────────
    st.subheader("How to use this tool")

    guide = [
        ("🔮", "2025 Signal",
         "Start here. Retrained model (Jun 2019–Dec 2024) scored against Nov 2025 features — forward window Dec 2025–Nov 2026. The 2024 predictions were validated at AUC 0.7747 against actual 2025 outcomes."),
        ("🏘️", "Archetypes",
         "The 5 distinct market types identified by K-means clustering. Understand each ZIP's long-run market character — not just its current trend."),
        ("⚖️", "Compare",
         "Select 2–4 ZIPs for a side-by-side view: opportunity score and net direction history (Jun 2019–Nov 2025), current forward signal, and SHAP feature drivers."),
        ("🔍", "Deep Dive",
         "Select any ZIP to see its full opportunity score history (Jun 2019–Nov 2025), key feature trends vs. cluster average, and a SHAP waterfall explaining exactly why the model scored it the way it did."),
        ("🗺️", "Validated Map (Educator)",
         "Choropleth of the 2023 walk-forward test-set predictions — useful for understanding model validation and how scores distribute across NC/SC."),
        ("📊", "Performance (Educator)",
         "Walk-forward validation metrics across all folds, model version history, and top-15 features ranked by mean absolute SHAP impact across Jun 2019–Nov 2025."),
    ]

    for i in range(0, len(guide), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j >= len(guide):
                break
            icon, title, desc = guide[i + j]
            with col:
                st.markdown(f"""
                <div style="
                    background: #ffffff; border: 1px solid #e9ecef;
                    border-top: 3px solid #e63946;
                    border-radius: 10px; padding: 1.3rem 1.5rem;
                    margin-bottom: 0.8rem;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
                ">
                    <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">{icon}</div>
                    <div style="font-weight: 700; font-size: 0.975rem; color: #212529; margin-bottom: 0.4rem;">{title}</div>
                    <div style="font-size: 0.855rem; color: #6c757d; line-height: 1.6;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()
    st.caption(
        "**Data:** Zillow ZHVI · Redfin Market Tracker &nbsp;|&nbsp; "
        "**Model:** Gradient Boosting · RFECV feature selection · Walk-forward validation &nbsp;|&nbsp; "
        "**Coverage:** 655 NC/SC ZIPs · Jun 2019 – Nov 2025 · Dec 2025–Nov 2026 forward signal"
    )


# ── Sidebar ─────────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("""
    <div style="
        background: linear-gradient(135deg, #0d1b2a, #1b2838);
        padding: 1.1rem 1.2rem 0.9rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    ">
        <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff; letter-spacing: -0.3px;">
            🏘️ Market Monitor
        </div>
        <div style="font-size: 0.72rem; color: #a8b2c1; margin-top: 0.2rem; letter-spacing: 0.5px;">
            NC / SC · 655 ZIPs · 2019–2025
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Mode switcher — investor vs educator
    try:
        mode = st.sidebar.segmented_control(
            "mode_switch",
            options=["🎯 Investor", "🎓 Educator"],
            default="🎯 Investor",
            label_visibility="collapsed",
        )
    except AttributeError:
        mode = st.sidebar.radio(
            "mode_switch",
            options=["🎯 Investor", "🎓 Educator"],
            horizontal=True,
            label_visibility="collapsed",
        )
    if mode is None:
        mode = "🎯 Investor"

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filters**")
    states = st.sidebar.multiselect(
        "State", options=["NC", "SC"], default=["NC", "SC"]
    )

    all_clusters = list(CLUSTER_LABELS.keys())
    clusters = st.sidebar.multiselect(
        "Cluster archetype",
        options=all_clusters,
        default=all_clusters,
        format_func=lambda x: CLUSTER_LABELS[x],
    )

    threshold = st.sidebar.slider(
        "Highlight ZIPs above opportunity threshold (%)",
        min_value=0, max_value=100, value=60, step=5,
    )

    comp = load_model_comparison()
    if comp:
        winner = comp.get("winner", "monthly")
        winner_short = "Monthly-Only" if winner == "monthly" else "V3 (All Sources)"
        winner_auc = comp[winner]["auc"]
        st.sidebar.markdown("---")
        st.sidebar.caption(
            f"**Model:** {winner_short}  \n"
            f"**AUC:** {winner_auc:.3f} · test 2023  \n"
            f"**Coverage:** 655 ZIPs · Jun 2019–Nov 2025"
        )

    return states, clusters, threshold, mode


# ── Page 1: Risk Map ─────────────────────────────────────────────────────────────
def page_risk_map(states, clusters, threshold):
    comp = load_model_comparison()
    winner_label = comp["monthly"]["label"] if comp and comp.get("winner") == "monthly" else "V3 (All Sources)"
    winner_auc   = comp["monthly"]["auc"]   if comp and comp.get("winner") == "monthly" else comp["v3"]["auc"] if comp else 0.852
    winner_f1    = comp["monthly"]["f1"]    if comp and comp.get("winner") == "monthly" else comp["v3"]["f1"]  if comp else 0.508
    winner_nfeat = comp["monthly"]["n_features"] if comp and comp.get("winner") == "monthly" else comp["v3"]["n_features"] if comp else 39

    map_mode = st.radio(
        "Map view",
        options=["upward_risk", "direction"],
        format_func=lambda x: "🔴 Opportunity Score (upward transition probability)"
                               if x == "upward_risk" else "🔵↔🔴 Net Direction (heating vs. cooling)",
        horizontal=True,
        label_visibility="collapsed",
    )

    geojson = load_geojson()

    if map_mode == "upward_risk":
        page_header(
            "🗺️", "2023 Validated Map — NC & SC",
            f"Opportunity score · {winner_nfeat} features · Walk-forward validated · AUC {winner_auc:.3f}",
            accent="#6a4c93",
        )

        zip_df = build_zip_summary()
        filtered = zip_df[zip_df["state"].isin(states) & zip_df["cluster_name"].isin(clusters)]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ZIPs shown", len(filtered))
        high_risk = filtered[filtered["prob_transition"] >= threshold / 100]
        col2.metric(f"High-opportunity ZIPs (≥{threshold}%)", len(high_risk))
        col3.metric("Avg opportunity score", f"{filtered['prob_transition'].mean()*100:.1f}%")
        col4.metric("Median opportunity score", f"{filtered['prob_transition'].median()*100:.1f}%")

        st.divider()

        fig = px.choropleth(
            filtered,
            geojson=geojson,
            locations="zip",
            featureidkey="properties.ZCTA5CE20",
            color="prob_transition",
            color_continuous_scale=[
                [0.0,  "#1a9641"],
                [0.35, "#a6d96a"],
                [0.55, "#ffffbf"],
                [0.70, "#fdae61"],
                [1.0,  "#d7191c"],
            ],
            range_color=(0, 1),
            hover_name="zip",
            hover_data={
                "prob_transition": ":.1%",
                "city": True,
                "cluster_label": True,
                "state": True,
                "zip": False,
            },
            labels={
                "prob_transition": "Opportunity score",
                "city": "City",
                "cluster_label": "Cluster",
                "state": "State",
            },
        )
        fig.update_traces(marker_line_width=0.3, marker_line_color="white")
        fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
        fig.update_layout(
            height=580,
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(title="Opportunity<br>Score", tickformat=".0%", len=0.6),
            paper_bgcolor="rgba(0,0,0,0)",
            geo=dict(bgcolor="rgba(0,0,0,0)"),
        )
        add_state_overlays(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Predicted vs. Actual — 2023 Test Year (Fold C)")
        val = build_validation_table()
        val_filtered = val[val["state"].isin(states) & val["cluster_name"].isin(clusters)]
        n_correct = val_filtered["correct"].sum()
        n_total   = len(val_filtered)
        accuracy  = n_correct / n_total if n_total > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("ZIPs evaluated", n_total)
        c2.metric("Correctly predicted", n_correct)
        c3.metric("ZIP-level accuracy", f"{accuracy:.1%}")

        display_cols = ["zip", "city", "Avg Opp Score (%)", "Peak Opp Score (%)",
                        "Predicted", "Actual", "Match", "cluster_label", "state"]
        display_cols = [c for c in display_cols if c in val_filtered.columns]
        table = (
            val_filtered[display_cols]
            .sort_values("Avg Opp Score (%)", ascending=False)
            .rename(columns={"zip": "ZIP", "city": "City",
                              "cluster_label": "Cluster", "state": "State"})
        )
        st.dataframe(table.reset_index(drop=True), use_container_width=True, height=320)
        st.caption(
            "**Predicted** = avg 2023 opportunity score ≥ 50% · "
            "**Actual** = ZIP had at least one month in the top-quartile transition threshold during 2023 · "
            "**Match ✓** = prediction agreed with outcome"
        )

        st.subheader("Opportunity Score by Cluster")
        cluster_stats = (
            filtered.groupby("cluster_name")
            .agg(avg_risk=("prob_transition", "mean"), n_zips=("zip", "count"))
            .reset_index().sort_values("avg_risk", ascending=False)
        )
        cluster_stats["label"] = cluster_stats["cluster_name"].map(CLUSTER_LABELS)
        cluster_stats["color"] = cluster_stats["cluster_name"].map(CLUSTER_COLORS)
        fig2 = go.Figure(go.Bar(
            x=cluster_stats["label"], y=cluster_stats["avg_risk"] * 100,
            marker_color=cluster_stats["color"],
            text=cluster_stats["avg_risk"].map("{:.1%}".format), textposition="outside",
        ))
        fig2.update_layout(height=300, yaxis_title="Opportunity Score (%)",
                           xaxis_title="", margin=dict(t=20, b=40), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    else:
        page_header(
            "🗺️", "2023 Market Direction Map — NC & SC",
            f"Net direction = P(up) − P(down) · 🔴 Heating · ⚪ Stable · 🔵 Cooling · Macro F1 {winner_f1:.3f}",
            accent="#6a4c93",
        )

        mc_df = build_zip_multiclass_summary()
        filtered = mc_df[mc_df["state"].isin(states) & mc_df["cluster_name"].isin(clusters)]

        n_heating = (filtered["net_direction"] >  0.15).sum()
        n_cooling = (filtered["net_direction"] < -0.15).sum()
        n_stable  = len(filtered) - n_heating - n_cooling
        avg_net   = filtered["net_direction"].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔴 Heating ZIPs (net > 0.15)",  n_heating)
        col2.metric("🔵 Cooling ZIPs (net < −0.15)", n_cooling)
        col3.metric("⚪ Stable ZIPs",                 n_stable)
        col4.metric("Avg net direction",              f"{avg_net:+.3f}")

        st.divider()

        _dir_plot = (
            filtered[["zip", "net_direction", "prob_up", "prob_stable",
                       "prob_down", "city", "cluster_label", "state"]]
            .copy()
        )
        for _c in ["net_direction", "prob_up", "prob_stable", "prob_down"]:
            _dir_plot[_c] = _dir_plot[_c].fillna(0)
        _dir_plot["city"]          = _dir_plot["city"].fillna("")
        _dir_plot["cluster_label"] = _dir_plot["cluster_label"].fillna("Unknown")
        _dir_plot["state"]         = _dir_plot["state"].fillna("")
        fig = px.choropleth(
            _dir_plot,
            geojson=geojson,
            locations="zip",
            featureidkey="properties.ZCTA5CE20",
            color="net_direction",
            color_continuous_scale=[
                [0.0,  "#2166ac"],
                [0.35, "#92c5de"],
                [0.5,  "#f7f7f7"],
                [0.65, "#f4a582"],
                [1.0,  "#b2182b"],
            ],
            color_continuous_midpoint=0,
            range_color=(-1, 1),
            hover_name="zip",
            hover_data={
                "net_direction": ":.3f",
                "prob_up": ":.1%",
                "prob_stable": ":.1%",
                "prob_down": ":.1%",
                "city": True,
                "cluster_label": True,
                "state": True,
                "zip": False,
            },
            labels={
                "net_direction": "Net direction",
                "prob_up": "P(heating up)",
                "prob_stable": "P(stable)",
                "prob_down": "P(cooling down)",
                "city": "City",
                "cluster_label": "Cluster",
                "state": "State",
            },
        )
        fig.update_traces(marker_line_width=0.3, marker_line_color="white")
        fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
        fig.update_layout(
            height=580,
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(
                title="Net direction<br>(up − down)",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["−1 Cooling", "−0.5", "0 Stable", "+0.5", "+1 Heating"],
                len=0.6,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            geo=dict(bgcolor="rgba(0,0,0,0)"),
        )
        add_state_overlays(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Net Direction Score by Cluster")
        cluster_stats = (
            filtered.groupby("cluster_name")
            .agg(avg_net=("net_direction", "mean"), n_zips=("zip", "count"))
            .reset_index().sort_values("avg_net", ascending=False)
        )
        cluster_stats["label"] = cluster_stats["cluster_name"].map(CLUSTER_LABELS)
        bar_colors = ["#b2182b" if v > 0 else "#2166ac" for v in cluster_stats["avg_net"]]
        fig2 = go.Figure(go.Bar(
            x=cluster_stats["label"], y=cluster_stats["avg_net"],
            marker_color=bar_colors,
            text=cluster_stats["avg_net"].map("{:+.3f}".format), textposition="outside",
        ))
        fig2.add_hline(y=0, line_color="grey", line_dash="dash", line_width=1)
        fig2.update_layout(
            height=300,
            yaxis_title="Avg net direction score (up − down)",
            xaxis_title="", margin=dict(t=20, b=40), showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)


# ── Page 2: Archetypes ───────────────────────────────────────────────────────────
def page_archetypes(states, clusters, threshold):
    page_header(
        "🏘️", "Neighborhood Archetypes",
        "K-means clustering on 18 model features · 5 market tendencies across 655 ZIPs",
        accent="#2a9d8f",
    )

    zip_df = build_zip_summary()
    filtered = zip_df[zip_df["state"].isin(states) & zip_df["cluster_name"].isin(clusters)]
    geojson = load_geojson()

    fig = px.choropleth(
        filtered,
        geojson=geojson,
        locations="zip",
        featureidkey="properties.ZCTA5CE20",
        color="cluster_name",
        color_discrete_map=CLUSTER_COLORS,
        hover_name="zip",
        hover_data={
            "cluster_label": True,
            "city": True,
            "state": True,
            "risk_pct": True,
            "zip": False,
            "cluster_name": False,
        },
        labels={"cluster_label": "Archetype", "city": "City", "state": "State", "risk_pct": "Opportunity score (%)"},
    )
    fig.update_traces(marker_line_width=0.3, marker_line_color="white")
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=0, b=0),
        legend_title_text="Archetype",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    add_state_overlays(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Archetype Profiles")
    col_pairs = [
        ("high_value_appreciating", "low_activity_cooling"),
        ("competitive_mid_market",  "affordable_high_demand"),
        ("moderate_supply_growing", None),
    ]
    for left, right in col_pairs:
        cols = st.columns(2)
        for col, cname in zip(cols, [left, right]):
            if cname is None:
                continue
            if cname not in clusters:
                continue
            cdata = zip_df[zip_df["cluster_name"] == cname]
            avg_risk = cdata["prob_transition"].mean() * 100
            n = len(cdata)
            color = CLUSTER_COLORS[cname]
            with col:
                bar_w = min(avg_risk, 100)
                st.markdown(
                    f"""
                    <div class="archetype-card" style="border-top: 4px solid {color};">
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.6rem;">
                            <h4 style="margin:0; color:#212529; font-size:0.97rem;">{CLUSTER_LABELS[cname]}</h4>
                            <span style="background:{color}1a; color:{color}; font-size:0.72rem; font-weight:700;
                                         padding:3px 10px; border-radius:20px; white-space:nowrap;">{n} ZIPs</span>
                        </div>
                        <p style="color:#4a5568; margin:0 0 1rem; font-size:0.865rem; line-height:1.65;">
                            {CLUSTER_DESCRIPTIONS[cname]}
                        </p>
                        <div style="display:flex; align-items:center; gap:0.6rem;">
                            <div style="flex:1; height:6px; border-radius:3px; background:#e9ecef; overflow:hidden;">
                                <div style="height:100%; width:{bar_w:.1f}%;
                                            background:linear-gradient(90deg,{color}88,{color});
                                            border-radius:3px;"></div>
                            </div>
                            <span style="font-size:0.78rem; font-weight:700; color:{color}; white-space:nowrap;">
                                {avg_risk:.1f}% avg opp.
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown(
        """
        <div style="margin-top:1.8rem; padding:1rem 1.2rem; background:#f8f9fa;
                    border-left:4px solid #2a9d8f; border-radius:6px; font-size:0.84rem; color:#495057; line-height:1.7;">
            <strong>📊 How to read these archetypes</strong><br>
            These five groups were identified by K-means clustering on the same 18 features used in the prediction model (silhouette score = 0.16),
            which reflects that real estate markets exist on a <em>spectrum</em> rather than falling into hard buckets.
            Think of each archetype as a <strong>market tendency</strong>, not a rigid category — a ZIP near a boundary
            shares characteristics of both neighbors. Cluster assignments are <strong>static long-run labels</strong>
            representing a ZIP's baseline market character; the <em>directional prediction</em> (opportunity score) is a
            separate, time-varying signal.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Page 3: ZIP Deep Dive ────────────────────────────────────────────────────────
def page_zip_dive(states, clusters, threshold):
    page_header(
        "🔍", "ZIP Code Deep Dive",
        "Monthly opportunity score · feature trends vs. cluster average · SHAP waterfall explanation",
    )

    features    = load_full_features()
    fwd_zip_df  = build_forward_zip_summary()
    fwd_mc_df   = build_forward_zip_multiclass_summary()

    available_zips = sorted(
        fwd_zip_df[fwd_zip_df["state"].isin(states) & fwd_zip_df["cluster_name"].isin(clusters)]["zip"].tolist()
    )
    if not available_zips:
        st.warning("No ZIPs match the current sidebar filters.")
        return

    selected_zip = st.selectbox(
        "Select ZIP code", available_zips,
        format_func=lambda z: (
            f"{z} — {fwd_zip_df.loc[fwd_zip_df['zip']==z, 'city'].values[0]}  |  "
            f"{fwd_zip_df.loc[fwd_zip_df['zip']==z, 'cluster_label'].values[0]}  |  "
            f"2025 Signal: {fwd_zip_df.loc[fwd_zip_df['zip']==z, 'risk_pct'].values[0]:.1f}%"
        )
    )

    zip_info    = fwd_zip_df[fwd_zip_df["zip"] == selected_zip].iloc[0]
    mc_row_rows = fwd_mc_df[fwd_mc_df["zip"] == selected_zip]
    mc_row      = mc_row_rows.iloc[0] if len(mc_row_rows) > 0 else None
    cname = zip_info["cluster_name"]
    color = CLUSTER_COLORS[cname]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ZIP Code", selected_zip)
    c2.metric("City", zip_info["city"])
    c3.metric("2025 Opportunity Score", f"{zip_info['risk_pct']:.1f}%")
    c4.metric("Cluster", CLUSTER_LABELS.get(cname, cname))

    if mc_row is not None:
        net = mc_row["net_direction"]
        dir_lbl = "Heating" if net > 0.15 else ("Cooling" if net < -0.15 else "Stable")
        st.caption(
            f"**Dec 2025–Nov 2026 forward signal** &nbsp;|&nbsp; "
            f"Net direction: {net:+.3f} ({dir_lbl}) &nbsp;|&nbsp; "
            f"P(Up): {mc_row['prob_up']:.1%} &nbsp; P(Stable): {mc_row['prob_stable']:.1%} &nbsp; P(Down): {mc_row['prob_down']:.1%}"
        )

    st.divider()

    # Full opportunity score history Jun 2019 – Nov 2025
    full_preds = load_full_predictions()
    zip_preds  = full_preds[full_preds["zip"] == selected_zip].sort_values("month")
    fig_risk = go.Figure()
    fig_risk.add_trace(go.Scatter(
        x=zip_preds["month"], y=zip_preds["prob_transition"] * 100,
        mode="lines+markers", name="Opportunity Score",
        line=dict(color=color, width=2),
        marker=dict(size=4),
    ))
    fig_risk.add_hline(
        y=threshold, line_dash="dash", line_color="grey",
        annotation_text=f"Threshold ({threshold}%)",
        annotation_position="bottom right",
    )
    fig_risk.update_layout(
        title=f"Opportunity Score History (Jun 2019 – Nov 2025) — {selected_zip}",
        yaxis_title="Opportunity Score (%)", xaxis_title="",
        height=280, margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    # Feature time series
    zip_feat = features[features["zip"] == selected_zip].sort_values("month")
    # Only show features that exist in the features dataframe
    avail_features = [f for f in SELECTED_FEATURES if f in features.columns]
    cluster_avg = (
        features[features["cluster_name"] == cname]
        .groupby("month")[avail_features]
        .mean()
        .reset_index()
    )

    st.subheader("Key Features vs. Cluster Average")
    feature_options = {FEATURE_LABELS.get(f, f): f for f in avail_features}
    chosen_labels = st.multiselect(
        "Select features to plot",
        list(feature_options.keys()),
        default=list(feature_options.keys())[:4],
    )
    chosen = [feature_options[l] for l in chosen_labels]

    if chosen:
        ncols = 2
        rows = [chosen[i:i+ncols] for i in range(0, len(chosen), ncols)]
        for row in rows:
            cols = st.columns(ncols)
            for col, feat in zip(cols, row):
                with col:
                    fig_f = go.Figure()
                    fig_f.add_trace(go.Scatter(
                        x=zip_feat["month"], y=zip_feat[feat],
                        mode="lines", name=selected_zip,
                        line=dict(color=color, width=2),
                    ))
                    fig_f.add_trace(go.Scatter(
                        x=cluster_avg["month"], y=cluster_avg[feat],
                        mode="lines", name=f"{cname} avg",
                        line=dict(color="#adb5bd", width=1.5, dash="dot"),
                    ))
                    fig_f.update_layout(
                        title=FEATURE_LABELS.get(feat, feat),
                        height=220,
                        margin=dict(t=35, b=20, l=40, r=10),
                        legend=dict(orientation="h", y=1.15, font_size=10),
                        showlegend=True,
                    )
                    st.plotly_chart(fig_f, use_container_width=True)

    # SHAP explanation
    st.divider()
    st.subheader("Why this prediction? — SHAP Feature Contributions")

    shap_df  = load_full_shap()
    zip_shap = shap_df[shap_df["zip"] == selected_zip].sort_values("month")

    if zip_shap.empty:
        st.info("SHAP values not available for this ZIP.")
    else:
        shap_months = zip_shap["month"].dt.strftime("%b %Y").tolist()
        selected_month_label = st.select_slider(
            "Month", options=shap_months, value=shap_months[-1]
        )
        selected_month_idx = shap_months.index(selected_month_label)
        row = zip_shap.iloc[selected_month_idx]

        shap_cols  = [c for c in shap_df.columns if c.startswith("shap_") and c != "shap_base"]
        feat_names = [FEATURE_LABELS.get(c.replace("shap_", ""), c.replace("shap_", "")) for c in shap_cols]
        shap_vals  = [row[c] for c in shap_cols]

        pairs = sorted(zip(feat_names, shap_vals), key=lambda x: abs(x[1]))
        labels, values = zip(*pairs)
        bar_colors = ["#e63946" if v > 0 else "#457b9d" for v in values]

        fig_shap = go.Figure(go.Bar(
            x=list(values),
            y=list(labels),
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
        ))
        fig_shap.add_vline(x=0, line_color="grey", line_width=1)
        fig_shap.update_layout(
            height=max(360, len(shap_cols) * 22),
            margin=dict(l=10, r=80, t=30, b=20),
            xaxis_title="SHAP value (log-odds contribution)",
            yaxis_title="",
            xaxis=dict(zeroline=False),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption(
            "**Red bars** push the prediction toward upward transition · "
            "**Blue bars** push it toward stable or cooling · "
            "Values are log-odds contributions — the sum plus base value equals the model's raw score."
        )


# ── ZIP Comparison ───────────────────────────────────────────────────────────────
def page_zip_comparison(states, clusters, threshold):
    page_header(
        "⚖️", "ZIP Code Comparison",
        "Select 2–4 ZIPs · side-by-side opportunity score, net direction, and SHAP drivers · sorted highest first",
        accent="#457b9d",
    )

    zip_df = build_forward_zip_summary()
    mc_df  = build_forward_zip_multiclass_summary()

    available_df = zip_df[
        zip_df["state"].isin(states) & zip_df["cluster_name"].isin(clusters)
    ].sort_values("prob_transition", ascending=False)

    if len(available_df) < 2:
        st.warning("Not enough ZIPs match current sidebar filters.")
        return

    available_zips = available_df["zip"].tolist()

    selected_zips = st.multiselect(
        "Select ZIPs to compare (2–4)",
        options=available_zips,
        default=available_zips[:2],
        max_selections=4,
        format_func=lambda z: (
            f"{z} — {zip_df.loc[zip_df['zip']==z, 'city'].values[0]}  ·  "
            f"{CLUSTER_LABELS.get(zip_df.loc[zip_df['zip']==z, 'cluster_name'].values[0], '')}  ·  "
            f"{zip_df.loc[zip_df['zip']==z, 'risk_pct'].values[0]:.1f}%"
        ),
    )

    if len(selected_zips) < 2:
        st.info("Select at least 2 ZIPs to compare.")
        return

    colors = {z: ZIP_COMPARISON_COLORS[i] for i, z in enumerate(selected_zips)}

    # ── Summary cards ────────────────────────────────────────────────────────────
    card_cols = st.columns(len(selected_zips))
    for col, z in zip(card_cols, selected_zips):
        info    = zip_df[zip_df["zip"] == z].iloc[0]
        mc_rows = mc_df[mc_df["zip"] == z]
        mc_row  = mc_rows.iloc[0] if len(mc_rows) > 0 else None
        color   = colors[z]
        net     = mc_row["net_direction"] if mc_row is not None else 0
        dir_lbl = "↑ Heating" if net > 0.15 else ("↓ Cooling" if net < -0.15 else "→ Stable")
        dir_col = "#e63946"  if net > 0.15 else ("#457b9d"  if net < -0.15 else "#6c757d")
        cname   = info["cluster_name"]
        with col:
            st.markdown(f"""
            <div style="
                border-top: 5px solid {color};
                padding: 1.2rem 1.4rem 1.1rem;
                border-radius: 10px;
                background: #f8fafc;
                border: 1px solid #e9ecef;
                border-top-color: {color};
            ">
                <div style="font-size: 1.9rem; font-weight: 800; color: #212529; letter-spacing: -0.5px; line-height: 1;">{z}</div>
                <div style="color: #212529; font-size: 0.92rem; font-weight: 600; margin: 0.15rem 0 0.1rem;">{info['city']}</div>
                <div style="color: #6c757d; font-size: 0.78rem; margin: 0 0 0.9rem;">{CLUSTER_LABELS.get(cname, cname)}</div>
                <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                    <div>
                        <div style="font-size: 0.65rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600;">Opportunity</div>
                        <div style="font-size: 1.7rem; font-weight: 800; color: {color}; line-height: 1.1;">{info['risk_pct']:.1f}%</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.65rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600;">Direction</div>
                        <div style="font-size: 1rem; font-weight: 700; color: {dir_col};">{dir_lbl}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Time-series charts (full history Jun 2019 – Nov 2025) ────────────────────
    preds    = load_full_predictions()
    mc_preds = load_full_multiclass()
    zip_preds = preds[preds["zip"].isin(selected_zips)].copy()
    zip_mc    = mc_preds[mc_preds["zip"].isin(selected_zips)].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Opportunity Score — Jun 2019 to Nov 2025")
        fig = go.Figure()
        for z in selected_zips:
            zd = zip_preds[zip_preds["zip"] == z].sort_values("month")
            fig.add_trace(go.Scatter(
                x=zd["month"], y=zd["prob_transition"] * 100,
                mode="lines+markers", name=z,
                line=dict(color=colors[z], width=2.5),
                marker=dict(size=4),
            ))
        fig.add_hline(
            y=threshold, line_dash="dash", line_color="#adb5bd",
            annotation_text=f"Threshold ({threshold}%)", annotation_position="bottom right",
        )
        fig.update_layout(
            height=300, margin=dict(t=10, b=20, l=40, r=20),
            yaxis_title="Opportunity Score (%)", xaxis_title="",
            yaxis_range=[0, 100],
            legend=dict(orientation="h", y=1.12, font_size=11),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Net Market Direction — Jun 2019 to Nov 2025")
        fig2 = go.Figure()
        for z in selected_zips:
            zd = zip_mc[zip_mc["zip"] == z].sort_values("month")
            fig2.add_trace(go.Scatter(
                x=zd["month"], y=zd["net_direction"],
                mode="lines+markers", name=z,
                line=dict(color=colors[z], width=2.5),
                marker=dict(size=4),
            ))
        fig2.add_hline(y=0, line_color="#adb5bd", line_dash="dash", line_width=1)
        fig2.update_layout(
            height=300, margin=dict(t=10, b=20, l=40, r=20),
            yaxis_title="Net Direction (P↑ − P↓)", xaxis_title="",
            yaxis_range=[-1, 1],
            legend=dict(orientation="h", y=1.12, font_size=11),
            hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Summary table ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Head-to-Head Summary — Dec 2025–Nov 2026 Forward Signal")
    rows = []
    for z in selected_zips:
        info    = zip_df[zip_df["zip"] == z].iloc[0]
        mc_rows = mc_df[mc_df["zip"] == z]
        mc_row  = mc_rows.iloc[0] if len(mc_rows) > 0 else None
        rows.append({
            "ZIP":              z,
            "City":             info["city"],
            "Cluster":          CLUSTER_LABELS.get(info["cluster_name"], info["cluster_name"]),
            "Opportunity Score":f"{info['risk_pct']:.1f}%",
            "Net Direction":    f"{mc_row['net_direction']:+.3f}" if mc_row is not None else "—",
            "P(Heating ↑)":     f"{mc_row['prob_up']:.1%}"       if mc_row is not None else "—",
            "P(Stable →)":      f"{mc_row['prob_stable']:.1%}"   if mc_row is not None else "—",
            "P(Cooling ↓)":     f"{mc_row['prob_down']:.1%}"     if mc_row is not None else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── SHAP comparison ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("What drove each prediction? — SHAP (Nov 2025)")

    shap_df   = load_full_shap()
    shap_cols = [c for c in shap_df.columns if c.startswith("shap_") and c != "shap_base"]
    latest_mo = shap_df["month"].max()

    avail_shap   = [z for z in selected_zips
                    if not shap_df[(shap_df["zip"] == z) & (shap_df["month"] == latest_mo)].empty]
    missing_shap = [z for z in selected_zips if z not in avail_shap]
    if missing_shap:
        st.caption(f"SHAP values not available for: {', '.join(missing_shap)}.")

    if avail_shap:
        shap_rows = {
            z: shap_df[(shap_df["zip"] == z) & (shap_df["month"] == latest_mo)].iloc[0]
            for z in avail_shap
        }
        # Top 8 features by mean |SHAP| across selected ZIPs
        mean_abs  = pd.Series({
            c: np.mean([abs(shap_rows[z][c]) for z in avail_shap]) for c in shap_cols
        }).sort_values(ascending=False)
        top_cols   = mean_abs.head(8).index.tolist()
        top_labels = [FEATURE_LABELS.get(c.replace("shap_", ""), c.replace("shap_", "")) for c in top_cols]

        shap_fig = go.Figure()
        for z in avail_shap:
            vals = [shap_rows[z][c] for c in top_cols]
            shap_fig.add_trace(go.Bar(
                name=z, x=top_labels, y=vals,
                marker_color=colors[z], opacity=0.88,
            ))
        shap_fig.add_hline(y=0, line_color="#adb5bd", line_width=1)
        shap_fig.update_layout(
            barmode="group",
            height=400,
            margin=dict(t=10, b=100, l=20, r=20),
            yaxis_title="SHAP value (log-odds)",
            xaxis_tickangle=-30,
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(shap_fig, use_container_width=True)
        st.caption(
            "Positive = pushes toward upward transition · Negative = pushes toward stable/cooling. "
            "Top 8 features ranked by mean |SHAP| across selected ZIPs."
        )


# ── Page 4: Model Comparison ─────────────────────────────────────────────────────
def page_model_comparison():
    comp = load_model_comparison()
    if comp is None:
        st.warning("model_comparison.json not found. Run scripts/run_monthly_model.py first.")
        return

    winner  = comp["winner"]
    v3      = comp["v3"]
    monthly = comp["monthly"]
    winner_label = monthly["label"] if winner == "monthly" else v3["label"]

    page_header(
        "⚡", "Model Comparison",
        "V3 (all sources) vs. Monthly-Only (Zillow + Redfin) · the winner drives all other pages",
        accent="#6a4c93",
    )

    # Winner banner
    delta_auc = monthly["auc"] - v3["auc"]
    delta_f1  = monthly["f1"]  - v3["f1"]
    banner_color = "#2a9d8f" if winner == "monthly" else "#e63946"
    st.markdown(
        f"""<div style="background:{banner_color}22; border-left:5px solid {banner_color};
                        padding:12px 20px; border-radius:4px; margin-bottom:16px">
        <h3 style="margin:0">🏆 Winner: {winner_label}</h3>
        <p style="margin:4px 0 0 0; color:#555">
        AUC {monthly['auc']:.4f} vs {v3['auc']:.4f} ({delta_auc:+.4f}) &nbsp;|&nbsp;
        Macro F1 {monthly['f1']:.4f} vs {v3['f1']:.4f} ({delta_f1:+.4f})
        </p></div>""",
        unsafe_allow_html=True,
    )

    # Metrics table
    st.subheader("Side-by-Side Metrics")
    st.caption(
        "Note: V3 was evaluated on the 2021 test set. Monthly-Only was evaluated on 2023. "
        "These are different test years — the monthly-only model is more recent, not directly AUC-comparable."
    )
    metrics_df = pd.DataFrame([
        {
            "Model": v3["label"],
            "Features": v3["n_features"],
            "Test year": v3.get("test_year", 2021),
            "Binary AUC": v3["auc"],
            "Macro F1": v3["f1"],
            "Data sources": "Zillow + Redfin + ACS + CBP + IRS",
        },
        {
            "Model": monthly["label"],
            "Features": monthly["n_features"],
            "Test year": monthly.get("test_year", 2023),
            "Binary AUC": monthly["auc"],
            "Macro F1": monthly["f1"],
            "Data sources": "Zillow + Redfin only",
        },
    ])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Fold AUC breakdown for monthly model
    fold_auc = monthly.get("fold_auc", {})
    if fold_auc:
        st.subheader("Monthly-Only Model: AUC Across Walk-Forward Folds")
        rows = [
            {"Fold": k, "AUC": v, "Note": "COVID-era hot market" if "2021" in k
             else "Rate-hike year — market inflection" if "2022" in k
             else "Post-normalization" if "2023" in k else ""}
            for k, v in fold_auc.items()
        ]
        retrained = comp.get("retrained", {}) if comp else {}
        rows.append({
            "Fold": "Fold D (test 2024)",
            "AUC": retrained.get("fold_d_auc", 0.7240),
            "Note": "Retrained on Jun 2019–Dec 2023 · powers 2025 forward signal",
        })
        rows.append({
            "Fold": "Actuals (2025 outcomes)",
            "AUC": 0.7747,
            "Note": "2024 predictions validated against real 2025 transitions",
        })
        fold_df = pd.DataFrame(rows)
        st.dataframe(fold_df, use_container_width=True, hide_index=True)
        st.caption(
            "The 2022 dip is expected: the Fed rate-hike cycle created abrupt market conditions "
            "outside the 2019–2021 training distribution. The 2023 recovery (0.8652) confirms "
            "the model captures durable signals. Fold D (0.7240) and actuals validation (0.7747) "
            "show the model generalises across unseen post-2023 market conditions."
        )

    st.info(
        "**Why Monthly-Only:** The 17 annual ACS/CBP/IRS features removed rank #12–#27 in "
        "permutation importance (0.0008–0.0024 vs 0.046 for the top feature). Dropping them "
        "forces RFECV to select better monthly substitutes and removes data-lag noise. "
        "It also unlocks future extension to Redfin data back to 2012 (4× more training data) "
        "and can be updated whenever new Redfin/Zillow data is available — no annual survey lag."
    )

    st.divider()

    # Side-by-side opportunity score maps
    st.subheader("Opportunity Score Maps — V3 vs Monthly-Only")
    geojson = load_geojson()

    col1, col2 = st.columns(2)

    for col, model_key, model_info in [
        (col1, "v3",      v3),
        (col2, "monthly", monthly),
    ]:
        with col:
            st.markdown(f"**{model_info['label']}** — AUC {model_info['auc']:.4f}")
            zip_df = build_zip_summary(model=model_key)
            fig = px.choropleth(
                zip_df,
                geojson=geojson,
                locations="zip",
                featureidkey="properties.ZCTA5CE20",
                color="prob_transition",
                color_continuous_scale=[
                    [0.0, "#1a9641"], [0.35, "#a6d96a"],
                    [0.55, "#ffffbf"], [0.70, "#fdae61"], [1.0, "#d7191c"],
                ],
                range_color=(0, 1),
                hover_name="zip",
                hover_data={"prob_transition": ":.1%", "cluster_label": True, "zip": False},
                labels={"prob_transition": "Opportunity score", "cluster_label": "Cluster"},
            )
            fig.update_traces(marker_line_width=0.3, marker_line_color="white")
            fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
            fig.update_layout(
                height=420,
                margin=dict(l=0, r=0, t=0, b=0),
                coloraxis_colorbar=dict(title="Score", tickformat=".0%", len=0.5),
                paper_bgcolor="rgba(0,0,0,0)",
                geo=dict(bgcolor="rgba(0,0,0,0)"),
            )
            add_state_overlays(fig)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Feature lists
    st.subheader("Selected Features Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{v3['label']} — {v3['n_features']} features**")
        for f in sorted(v3["features"]):
            st.markdown(f"- {FEATURE_LABELS.get(f, f)}")
    with col2:
        st.markdown(f"**{monthly['label']} — {monthly['n_features']} features**")
        for f in sorted(monthly["features"]):
            st.markdown(f"- {FEATURE_LABELS.get(f, f)}")


# ── Page 5: Model Performance ────────────────────────────────────────────────────
def page_model_performance():
    page_header(
        "📊", "Model Performance & Feature Importance",
        "Walk-forward validation · 3 folds · version history · top-15 features by mean |SHAP|",
        accent="#6a4c93",
    )

    comp = load_model_comparison()
    winner = comp["winner"] if comp else "v3"
    winner_auc = comp[winner]["auc"] if comp else 0.865
    winner_f1  = comp[winner]["f1"]  if comp else 0.551

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Model Version Comparison")
        perf = pd.DataFrame([
            {"Version": "V1 — All features (incl. mortgage)", "Features": 71,
             "AUC (test 2021)": 0.8607, "AUC (test 2023)": "—", "AUC (test 2024)": "—", "AUC vs 2025 actuals": "—",
             "Note": "Mortgage rate was #1 feature — national signal, not ZIP-specific"},
            {"Version": "V2 — Removed mortgage features",     "Features": 69,
             "AUC (test 2021)": 0.8473, "AUC (test 2023)": "—", "AUC (test 2024)": "—", "AUC vs 2025 actuals": "—",
             "Note": "ΔAUC = −0.013 (negligible)"},
            {"Version": "V3 — RFECV-selected",                "Features": comp["v3"]["n_features"] if comp else 39,
             "AUC (test 2021)": comp["v3"]["auc"] if comp else 0.8519, "AUC (test 2023)": "—", "AUC (test 2024)": "—", "AUC vs 2025 actuals": "—",
             "Note": "Annual features blocked extension — ACS/CBP/IRS lag prevents 2022+ labels"},
            {"Version": "Monthly-Only — RFECV-selected ✓",    "Features": comp["monthly"]["n_features"] if comp else 18,
             "AUC (test 2021)": 0.8443, "AUC (test 2023)": comp["monthly"]["auc"] if comp else 0.8652,
             "AUC (test 2024)": comp.get("retrained", {}).get("fold_d_auc", 0.7240) if comp else 0.7240,
             "AUC vs 2025 actuals": 0.7747,
             "Note": "Zillow + Redfin only — updatable monthly, validated through 2025"},
        ])
        st.dataframe(perf, use_container_width=True, hide_index=True)

        st.subheader("Walk-Forward Validation Folds")
        folds = pd.DataFrame([
            {"Fold": "A",           "Train": "Jun 2019–Dec 2020", "Test": "Jan–Dec 2021", "AUC": 0.8443, "Note": ""},
            {"Fold": "B",           "Train": "Jun 2019–Dec 2021", "Test": "Jan–Dec 2022", "AUC": 0.8153, "Note": "Fed rate-hike year — out of training dist."},
            {"Fold": "C (primary)", "Train": "Jun 2019–Dec 2022", "Test": "Jan–Dec 2023", "AUC": 0.8652, "Note": "Primary validation fold"},
            {"Fold": "D (retrain)", "Train": "Jun 2019–Dec 2023", "Test": "Jan–Dec 2024", "AUC": 0.7240, "Note": "Retrained model — powers 2025 signal"},
            {"Fold": "Actuals",     "Train": "Jun 2019–Dec 2022", "Test": "2025 real outcomes", "AUC": 0.7747, "Note": "2024 preds vs. actual 2025 transitions"},
        ])
        st.dataframe(folds, use_container_width=True, hide_index=True)

        st.info(
            "**Why walk-forward?**  Random train/test splits leak future data in time-series panels. "
            "Walk-forward trains only on past months and tests on future months — matching real deployment."
        )

        st.divider()
        st.subheader("Why Monthly-Only Won")
        st.info(
            "The 17 annual ACS/CBP/IRS features removed ranked #12–#27 in permutation importance "
            "(0.0008–0.0024 vs 0.046 for the top monthly feature). Dropping them forces RFECV to find "
            "better monthly substitutes and removes data-lag noise. More importantly, annual data "
            "publication lag (1–2 years) capped V3 at Fold B (test 2021) — Monthly-Only extends "
            "validation to 2023, 2024, and retrospective 2025 actuals, and can be updated monthly "
            "as new Redfin/Zillow data is released."
        )

    with col2:
        st.subheader("Feature Importance — Mean |SHAP| (Winning Model)")
        shap_df = load_full_shap()
        shap_cols = [c for c in shap_df.columns if c.startswith("shap_") and c != "shap_base"]
        mean_abs  = shap_df[shap_cols].abs().mean().sort_values(ascending=False)
        top_n = mean_abs.head(15)
        feat_labels = [FEATURE_LABELS.get(c.replace("shap_", ""), c.replace("shap_", "")) for c in top_n.index]

        fig_imp = go.Figure(go.Bar(
            x=top_n.values[::-1],
            y=feat_labels[::-1],
            orientation="h",
            marker_color="#e63946",
            text=[f"{v:.4f}" for v in top_n.values[::-1]],
            textposition="outside",
        ))
        fig_imp.update_layout(
            height=480,
            margin=dict(l=10, r=70, t=20, b=20),
            xaxis_title="Mean |SHAP| (log-odds)",
            yaxis_title="",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.subheader("What the top features mean")
        explanations = [
            ("Value vs Baseline (6m lag)",  "ZIP's price level vs. its own 6-month-lagged norm — sustained structural drift is the single strongest signal."),
            ("Home Value MoM %",           "Month-over-month % change in ZIP home value index — current price momentum."),
            ("Home Value MoM % (6m lag)",  "Home value MoM change 6 months prior — lagged momentum confirmation that direction is persistent."),
            ("% Sold Above List (6m lag)", "% of homes sold above list price 6 months ago — sustained buyer demand heat, not a one-month spike."),
            ("Days on Market vs Baseline", "Days on market vs. this ZIP's own historical norm — whether homes are selling unusually fast or slow."),
        ]
        for feat, desc in explanations:
            st.markdown(f"**{feat}:** {desc}")


# ── Page: Forward Predictions (2025) ─────────────────────────────────────────────
def page_forward_predictions(states, clusters, threshold):
    page_header(
        "🔮", "2025 Market Signal",
        "Retrained model (Jun 2019–Dec 2024) · scored on Nov 2025 features · forward window Dec 2025–Nov 2026",
        accent="#f4a261",
        badge="LIVE",
    )
    st.info(
        "**Retrained on Jun 2019–Dec 2024, scored on Nov 2025 features.** "
        "The 2024 predictions were retrospectively validated against actual 2025 outcomes (AUC 0.7747), "
        "confirming the model captures durable signals across market regimes. "
        "This 2025 signal covers the forward window **Dec 2025–Nov 2026** and cannot be verified until late 2026."
    )

    geojson = load_geojson()
    # Use the full-panel files (same Nov 2025 snapshot, but generated fresh and
    # known-clean — the old forward_2025 CSVs had serialization issues on HF)
    fwd_bin = load_full_predictions()
    fwd_mc  = load_full_multiclass()

    # ── Mode toggle ────────────────────────────────────────────────────────────
    fwd_mode = st.radio(
        "Map view",
        options=["opportunity", "direction"],
        format_func=lambda x: "Opportunity Score (upward transition probability)"
                               if x == "opportunity" else "Net Direction (heating vs. cooling)",
        horizontal=True,
    )

    # Always use the latest available month (Nov 2025 — last full Redfin coverage)
    fwd_bin_p = fwd_bin[fwd_bin["month"] == fwd_bin["month"].max()]
    fwd_mc_p  = fwd_mc[fwd_mc["month"] == fwd_mc["month"].max()]

    zip_fwd = (
        fwd_bin_p.groupby("zip")
        .agg(
            prob_transition=("prob_transition", "mean"),
            cluster=("cluster", "first"),
            cluster_name=("cluster_name", "first"),
        )
        .reset_index()
    )
    zip_fwd["risk_pct"]      = (zip_fwd["prob_transition"] * 100).round(1)
    zip_fwd["cluster_label"] = zip_fwd["cluster_name"].map(CLUSTER_LABELS)
    zip_fwd["state"]         = zip_fwd["zip"].apply(_state_label)
    cities = load_city_lookup()
    zip_fwd["city"]          = zip_fwd["zip"].map(cities).fillna("")

    zip_mc = (
        fwd_mc_p.groupby("zip")
        .agg(
            prob_down=("prob_down", "mean"),
            prob_stable=("prob_stable", "mean"),
            prob_up=("prob_up", "mean"),
            net_direction=("net_direction", "mean"),
            cluster=("cluster", "first"),
            cluster_name=("cluster_name", "first"),
        )
        .reset_index()
    )
    zip_mc["cluster_label"] = zip_mc["cluster_name"].map(CLUSTER_LABELS)
    zip_mc["state"]         = zip_mc["zip"].apply(_state_label)
    zip_mc["city"]          = zip_mc["zip"].map(cities).fillna("")

    # Apply sidebar filters
    filtered_bin = zip_fwd[zip_fwd["state"].isin(states) & zip_fwd["cluster_name"].isin(clusters)]
    filtered_mc  = zip_mc[zip_mc["state"].isin(states)   & zip_mc["cluster_name"].isin(clusters)]

    # ── Metrics row ───────────────────────────────────────────────────────────
    if fwd_mode == "opportunity":
        col1, col2, col3, col4 = st.columns(4)
        high = filtered_bin[filtered_bin["prob_transition"] >= threshold / 100]
        col1.metric("ZIPs shown", len(filtered_bin))
        col2.metric(f"High-opportunity (≥{threshold}%)", len(high))
        col3.metric("Avg opportunity score", f"{filtered_bin['prob_transition'].mean()*100:.1f}%")
        col4.metric("Median opportunity score", f"{filtered_bin['prob_transition'].median()*100:.1f}%")
    else:
        col1, col2, col3, col4 = st.columns(4)
        n_heat = (filtered_mc["net_direction"] > 0.15).sum()
        n_cool = (filtered_mc["net_direction"] < -0.15).sum()
        col1.metric("Heating ZIPs (net > 0.15)", n_heat)
        col2.metric("Cooling ZIPs (net < -0.15)", n_cool)
        col3.metric("Avg net direction", f"{filtered_mc['net_direction'].mean():+.3f}")
        col4.metric("Stable ZIPs", len(filtered_mc) - n_heat - n_cool)

    st.divider()

    # ── Map ───────────────────────────────────────────────────────────────────
    if fwd_mode == "opportunity":
        _bin_plot = filtered_bin[["zip", "prob_transition", "city", "cluster_label", "cluster_name", "state"]].copy()
        _bin_plot["prob_transition"] = (
            pd.to_numeric(_bin_plot["prob_transition"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .clip(0, 1)
        )
        _bin_plot["city"]          = _bin_plot["city"].fillna("")
        _bin_plot["cluster_label"] = _bin_plot["cluster_label"].fillna("Unknown")
        _bin_plot["state"]         = _bin_plot["state"].fillna("")
        fig = px.choropleth(
            _bin_plot,
            geojson=geojson, locations="zip", featureidkey="properties.ZCTA5CE20",
            color="prob_transition",
            color_continuous_scale=[
                [0.0, "#1a9641"], [0.35, "#a6d96a"],
                [0.55, "#ffffbf"], [0.70, "#fdae61"], [1.0, "#d7191c"],
            ],
            range_color=(0, 1),
            hover_name="zip",
            hover_data={"prob_transition": ":.1%", "city": True, "cluster_label": True, "state": True, "zip": False},
            labels={"prob_transition": "Opportunity score", "city": "City", "cluster_label": "Cluster", "state": "State"},
        )
        fig.update_layout(
            coloraxis_colorbar=dict(title="Opportunity<br>Score", tickformat=".0%", len=0.6)
        )
    else:
        # Subset to only the columns Plotly needs — extra columns (prob_stable,
        # cluster, cluster_name, etc.) can leak into customdata and serialize as
        # bare NaN/Infinity literals, causing a JSON.parse failure in the frontend.
        _mc_plot = (
            filtered_mc[["zip", "net_direction", "prob_up", "prob_down",
                          "city", "cluster_label", "state"]]
            .copy()
        )
        for _col, _lo, _hi in [("net_direction", -1, 1), ("prob_up", 0, 1), ("prob_down", 0, 1)]:
            _mc_plot[_col] = (
                pd.to_numeric(_mc_plot[_col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .clip(_lo, _hi)
            )
        _mc_plot["city"]           = _mc_plot["city"].fillna("")
        _mc_plot["cluster_label"]  = _mc_plot["cluster_label"].fillna("Unknown")
        _mc_plot["state"]          = _mc_plot["state"].fillna("")
        fig = px.choropleth(
            _mc_plot,
            geojson=geojson, locations="zip", featureidkey="properties.ZCTA5CE20",
            color="net_direction",
            color_continuous_scale=[
                [0.0, "#2166ac"], [0.35, "#92c5de"],
                [0.5, "#f7f7f7"], [0.65, "#f4a582"], [1.0, "#b2182b"],
            ],
            color_continuous_midpoint=0, range_color=(-1, 1),
            hover_name="zip",
            hover_data={
                "net_direction": ":.3f", "prob_up": ":.1%",
                "prob_down": ":.1%", "city": True, "cluster_label": True, "state": True, "zip": False,
            },
            labels={
                "net_direction": "Net direction", "prob_up": "P(heating)",
                "prob_down": "P(cooling)", "city": "City", "cluster_label": "Cluster", "state": "State",
            },
        )
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Net direction<br>(up - down)",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1 Cooling", "-0.5", "0 Stable", "+0.5", "+1 Heating"],
                len=0.6,
            )
        )

    fig.update_traces(marker_line_width=0.3, marker_line_color="white")
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
        height=560, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", geo=dict(bgcolor="rgba(0,0,0,0)"),
    )
    add_state_overlays(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ── Top opportunities table ────────────────────────────────────────────────
    if fwd_mode == "opportunity":
        st.subheader(f"Top Opportunity ZIPs for 2025–2026 (≥ {threshold}%)")
        top = (
            filtered_bin[filtered_bin["prob_transition"] >= threshold / 100]
            [["zip", "city", "risk_pct", "cluster_label", "state"]]
            .sort_values("risk_pct", ascending=False)
            .rename(columns={
                "zip": "ZIP", "city": "City", "risk_pct": "Opportunity Score (%)",
                "cluster_label": "Cluster", "state": "State",
            })
        )
        if len(top) > 0:
            st.dataframe(top.reset_index(drop=True), use_container_width=True, height=320)
        else:
            st.info("No ZIPs above the selected threshold.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Heating ZIPs")
            top_heat = (
                filtered_mc.nlargest(15, "net_direction")
                [["zip", "net_direction", "prob_up", "cluster_label", "state"]]
                .rename(columns={
                    "zip": "ZIP", "net_direction": "Net Direction",
                    "prob_up": "P(heating)", "cluster_label": "Cluster", "state": "State",
                })
            )
            st.dataframe(top_heat.reset_index(drop=True), use_container_width=True, height=320)
        with col2:
            st.subheader("Top Cooling ZIPs")
            top_cool = (
                filtered_mc.nsmallest(15, "net_direction")
                [["zip", "net_direction", "prob_down", "cluster_label", "state"]]
                .rename(columns={
                    "zip": "ZIP", "net_direction": "Net Direction",
                    "prob_down": "P(cooling)", "cluster_label": "Cluster", "state": "State",
                })
            )
            st.dataframe(top_cool.reset_index(drop=True), use_container_width=True, height=320)

    # ── Context ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("How to interpret these predictions")
    st.markdown(
        """
        | Signal | What it means | Investor implication |
        |---|---|---|
        | **High opportunity score (≥ 0.6)** | Model predicts upward housing transition in the next 12 months | Potential buy signal — monitor for entry point |
        | **Net direction > 0.15** | More likely heating than cooling | Market momentum is positive |
        | **Net direction < -0.15** | More likely cooling than stable | Exercise caution; oversupply or demand drop likely |
        | **Net direction near 0** | Market likely stable | Hold or wait-and-see |

        The model was **retrained on Jun 2019–Dec 2024** (Fold D AUC on 2024 test data: 0.7240) and scored
        on Nov 2025 Zillow + Redfin features. The forward window is **Dec 2025–Nov 2026** — unverifiable
        until late 2026. The prior 2024 signal was validated at AUC 0.7747 against actual 2025 outcomes,
        confirming the model generalises across market cycles.
        """
    )


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    inject_css()
    states, clusters, threshold, mode = render_sidebar()

    educator_mode = mode and "Educator" in mode

    nav_options = ["Home", "2025 Signal", "Archetypes", "Compare", "Deep Dive"]
    nav_icons   = ["house-fill", "graph-up-arrow", "buildings", "arrow-left-right", "search"]
    if educator_mode:
        nav_options += ["Validated Map", "Performance"]
        nav_icons   += ["map-fill", "speedometer2"]

    selected = option_menu(
        menu_title=None,
        options=nav_options,
        icons=nav_icons,
        orientation="horizontal",
        default_index=0,
        key="top_nav",
        styles={
            "container": {
                "padding": "6px 10px",
                "background-color": "#0d1b2a",
                "border-radius": "12px",
                "margin-bottom": "1.75rem",
            },
            "icon": {"color": "#a8b2c1", "font-size": "15px"},
            "nav-link": {
                "font-size": "0.83rem",
                "font-weight": "600",
                "color": "#a8b2c1",
                "padding": "10px 16px",
                "border-radius": "8px",
                "--hover-color": "#1b2838",
            },
            "nav-link-selected": {
                "background-color": "#e63946",
                "color": "#ffffff",
                "font-weight": "700",
            },
        },
    )

    page_dispatch = {
        "Home":             page_home,
        "2025 Signal":      lambda: page_forward_predictions(states, clusters, threshold),
        "Archetypes":       lambda: page_archetypes(states, clusters, threshold),
        "Compare":          lambda: page_zip_comparison(states, clusters, threshold),
        "Deep Dive":        lambda: page_zip_dive(states, clusters, threshold),
        "Validated Map":    lambda: page_risk_map(states, clusters, threshold),
        "Performance":      page_model_performance,
    }

    page_dispatch[selected]()


if __name__ == "__main__":
    main()
