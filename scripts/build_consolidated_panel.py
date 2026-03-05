#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config / Patterns
# -----------------------------
DATE_COL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")  # Zillow wide date columns
ZIP_RE = re.compile(r"(\d{5})")


# -----------------------------
# Utility functions
# -----------------------------
def to_month_end(x: pd.Series) -> pd.Series:
    dt = pd.to_datetime(x, errors="coerce")
    return (dt + pd.offsets.MonthEnd(0)).dt.normalize()


def month_end_range(min_month: pd.Timestamp, max_month: pd.Timestamp) -> pd.DatetimeIndex:
    # pandas >= 3.0: "M" removed, use "ME" (month-end)
    return pd.date_range(min_month, max_month, freq="ME").normalize()


def zfill_zip(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.zfill(5)


def assert_unique(df: pd.DataFrame, keys: List[str], name: str) -> None:
    dup = df.duplicated(keys).sum()
    if dup:
        raise ValueError(f"[{name}] Found {dup:,} duplicate rows for keys={keys}")


def load_zip_universe(unique_zip_csv: Path) -> pd.Index:
    z = pd.read_csv(unique_zip_csv)
    if "RegionName" in z.columns:
        s = z["RegionName"]
    elif "zip" in z.columns:
        s = z["zip"]
    else:
        raise ValueError(f"ZIP universe file must have RegionName or zip: {unique_zip_csv}")

    s = zfill_zip(s)
    s = s[s.str.fullmatch(r"\d{5}", na=False)]
    return pd.Index(sorted(s.unique()))


def melt_zillow_wide(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    date_cols = [c for c in df.columns if DATE_COL_RE.match(str(c))]
    if not date_cols:
        raise ValueError(f"No date columns like YYYY-MM-DD found in Zillow file: {path.name}")

    if "RegionName" not in df.columns:
        raise ValueError(f"Zillow file missing RegionName: {path.name}")

    out = (
        df.melt(
            id_vars=["RegionName"],
            value_vars=date_cols,
            var_name="date",
            value_name=value_name,
        )
        .assign(
            zip=lambda d: zfill_zip(d["RegionName"]),
            month=lambda d: to_month_end(d["date"]),
        )
        .drop(columns=["RegionName", "date"])
    )

    # enforce types
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")

    out = out[["zip", "month", value_name]].sort_values(["zip", "month"])
    assert_unique(out, ["zip", "month"], f"zillow_{value_name}")
    return out


def read_redfin_long(path: Path, property_type: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # Required columns from your prepared file
    # (you already extracted zip in prep script)
    if "zip" not in df.columns or "PERIOD_BEGIN" not in df.columns:
        raise ValueError("Redfin file must include columns: zip, PERIOD_BEGIN")

    df = df.assign(
        zip=lambda d: zfill_zip(d["zip"]),
        month=lambda d: to_month_end(d["PERIOD_BEGIN"]),
    )

    if "PROPERTY_TYPE" in df.columns:
        df = df[df["PROPERTY_TYPE"].astype(str) == property_type].copy()

    # Keep a clean “pressure + price anchors” set
    keep = [
        "zip", "month",
        "MEDIAN_SALE_PRICE", "MEDIAN_LIST_PRICE",
        "MEDIAN_PPSF", "MEDIAN_LIST_PPSF",
        "HOMES_SOLD", "PENDING_SALES", "NEW_LISTINGS", "INVENTORY",
        "MONTHS_OF_SUPPLY", "MEDIAN_DOM",
        "AVG_SALE_TO_LIST", "SOLD_ABOVE_LIST",
        "PRICE_DROPS", "OFF_MARKET_IN_TWO_WEEKS",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # numeric coercion
    for c in df.columns:
        if c not in ("zip", "month"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["zip", "month"])
    assert_unique(df, ["zip", "month"], "redfin")
    return df


def read_pmms_monthly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # FRED-style PMMS exports sometimes use DATE, sometimes date
    date_col = "date" if "date" in df.columns else ("DATE" if "DATE" in df.columns else None)
    if not date_col:
        raise ValueError("PMMS file must contain 'date' or 'DATE' column")

    df = df.assign(date=pd.to_datetime(df[date_col], errors="coerce")).dropna(subset=["date"])
    df = df.assign(month=to_month_end(df["date"]))

    # Keep known rate columns if present
    candidates = ["pmms30", "pmms15", "pmms51", "pmms51spread", "PMMS", "MORTGAGE30US"]
    cols = [c for c in candidates if c in df.columns]

    if cols:
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        out = df.groupby("month", as_index=False)[cols].mean(numeric_only=True)
    else:
        # If unknown format, average all numeric cols
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        out = df.groupby("month", as_index=False)[num_cols].mean(numeric_only=True)

    out = out.sort_values("month")
    assert_unique(out, ["month"], "pmms_monthly")
    return out


def read_acs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "zip" not in df.columns or "year" not in df.columns:
        raise ValueError("ACS long file must contain columns: zip, year")

    df = df.assign(
        zip=lambda d: zfill_zip(d["zip"]),
        year=lambda d: pd.to_numeric(d["year"], errors="coerce").astype("Int64"),
    )

    # convert sentinel to NaN
    df = df.replace(-666666666, np.nan)

    # Ensure numeric
    for c in df.columns:
        if c not in ("zip", "year"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # derive owner_rate if possible
    if "B25003_002E" in df.columns and "B25003_003E" in df.columns:
        occ_total = df["B25003_002E"] + df["B25003_003E"]
        df["owner_rate"] = df["B25003_002E"] / occ_total.replace(0, np.nan)

    keep = ["zip", "year", "B19013_001E", "B25077_001E", "B25064_001E", "owner_rate"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    assert_unique(out, ["zip", "year"], "acs")
    return out


def read_cbp(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "zip" not in df.columns or "year" not in df.columns:
        raise ValueError("CBP long file must contain columns: zip, year")

    df = df.assign(
        zip=lambda d: zfill_zip(d["zip"]),
        year=lambda d: pd.to_numeric(d["year"], errors="coerce").astype("Int64"),
    )

    for c in df.columns:
        if c not in ("zip", "year"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "PAYANN" in df.columns and "EMP" in df.columns:
        df["avg_annual_pay_per_emp"] = df["PAYANN"] / df["EMP"].replace(0, np.nan)

    keep = ["zip", "year", "ESTAB", "EMP", "PAYANN", "PAYQTR1", "avg_annual_pay_per_emp"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    assert_unique(out, ["zip", "year"], "cbp")
    return out


def read_soi_totals(soi_dir: Path) -> pd.DataFrame:
    files = sorted(soi_dir.glob("soi_*.csv"))
    if not files:
        return pd.DataFrame(columns=["zip", "year"])

    out_parts = []
    for f in files:
        m = re.search(r"soi_(\d{4})\.csv$", f.name)
        if not m:
            continue
        year = int(m.group(1))

        d = pd.read_csv(f, low_memory=False)
        if "zipcode" not in d.columns:
            continue

        # assign zip/year in one go (avoids fragmentation)
        d = d.assign(
            zip=zfill_zip(d["zipcode"]),
            year=year,
        )

        # If agi_stub exists, keep total stub when available; else sum across stubs
        if "agi_stub" in d.columns:
            if (d["agi_stub"] == 0).any():
                d = d[d["agi_stub"] == 0].copy()
            else:
                num_cols = d.select_dtypes(include=[np.number]).columns.tolist()
                d = d.groupby("zip", as_index=False)[num_cols].sum(numeric_only=True)
                d["year"] = year
                d = d.copy()

        # compact feature set
        keep = ["zip", "year", "N1", "A00100", "A00200", "A00300", "A00600"]
        keep = [c for c in keep if c in d.columns]
        d = d[keep].copy()

        # numeric
        for c in d.columns:
            if c not in ("zip", "year"):
                d[c] = pd.to_numeric(d[c], errors="coerce")

        out_parts.append(d)

    if not out_parts:
        return pd.DataFrame(columns=["zip", "year"])

    out = pd.concat(out_parts, ignore_index=True)
    out = out.sort_values(["zip", "year"])
    assert_unique(out, ["zip", "year"], "soi")
    return out


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data", help="Use 'data' or 'data_samples'")
    ap.add_argument("--out", default="data/processed/consolidated_zip_month.csv")
    ap.add_argument("--property-type", default="All Residential", help="Redfin PROPERTY_TYPE to keep")
    ap.add_argument("--start", default=None, help="Optional start month (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="Optional end month (YYYY-MM-DD)")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    DATA_ROOT = (ROOT / args.data_root).resolve()

    RAW = DATA_ROOT / "raw"
    PROCESSED = DATA_ROOT / "processed"

    # ---- input paths
    ZIPS_PATH = RAW / "zillow" / "zip_universe_nc_sc.csv.csv"

    ZORI_PATH = RAW / "zillow" / "zillow_zori_zip_monthly.csv.csv"
    ZHVI_PATH = RAW / "zillow" / "zillow_zhvi_zip_monthly.csv.csv"

    PMMS_PATH = RAW / "mortgage_rate" / "freddie_mac_pmms_weekly.csv"

    ACS_PATH = PROCESSED / "acs5_filtered_long_2013_2022.csv"
    CBP_PATH = PROCESSED / "cbp_zip_filtered_long_2018_2022.csv"
    REDFIN_PATH = PROCESSED / "redfin_market_tracker_zip_nc_sc_long.csv"
    SOI_DIR = RAW / "soi"

    out_path = (ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- load universe + sources
    zips = load_zip_universe(ZIPS_PATH)

    zhvi = melt_zillow_wide(ZHVI_PATH, "zhvi")
    zori = melt_zillow_wide(ZORI_PATH, "zori")
    zillow = zhvi.merge(zori, on=["zip", "month"], how="outer").sort_values(["zip", "month"])
    zillow = zillow[zillow["zip"].isin(zips)].copy()
    assert_unique(zillow, ["zip", "month"], "zillow_merged")

    redfin = read_redfin_long(REDFIN_PATH, property_type=args.property_type)
    redfin = redfin[redfin["zip"].isin(zips)].copy()
    assert_unique(redfin, ["zip", "month"], "redfin_filtered")

    pmms_m = read_pmms_monthly(PMMS_PATH)

    acs = read_acs(ACS_PATH)
    acs = acs[acs["zip"].isin(zips)].copy()

    cbp = read_cbp(CBP_PATH)
    cbp = cbp[cbp["zip"].isin(zips)].copy()

    soi = read_soi_totals(SOI_DIR)
    if not soi.empty:
        soi = soi[soi["zip"].isin(zips)].copy()

    # ---- master monthly grid
    # use Zillow + Redfin to set bounds (then clamp with args.start/end if provided)
    min_month = pd.concat([zillow["month"], redfin["month"]]).min()
    max_month = pd.concat([zillow["month"], redfin["month"]]).max()

    if args.start:
        min_month = max(min_month, to_month_end(pd.Series([args.start])).iloc[0])
    if args.end:
        max_month = min(max_month, to_month_end(pd.Series([args.end])).iloc[0])

    months = month_end_range(min_month, max_month)

    base = pd.MultiIndex.from_product([zips, months], names=["zip", "month"]).to_frame(index=False)
    base = base.assign(year=base["month"].dt.year.astype("Int64"))
    assert_unique(base, ["zip", "month"], "base_grid")

    # ---- merge everything (audit after each step)
    df = base.merge(zillow, on=["zip", "month"], how="left")
    assert_unique(df, ["zip", "month"], "after_zillow")

    df = df.merge(redfin, on=["zip", "month"], how="left")
    assert_unique(df, ["zip", "month"], "after_redfin")

    df = df.merge(pmms_m, on="month", how="left")
    assert_unique(df, ["zip", "month"], "after_pmms")

    df = df.merge(acs, on=["zip", "year"], how="left")
    assert_unique(df, ["zip", "month"], "after_acs")

    df = df.merge(cbp, on=["zip", "year"], how="left")
    assert_unique(df, ["zip", "month"], "after_cbp")

    if not soi.empty:
        df = df.merge(soi, on=["zip", "year"], how="left")
        assert_unique(df, ["zip", "month"], "after_soi")

    # ---- derived features
    if "zhvi" in df.columns and "zori" in df.columns:
        df["zori_to_zhvi_ratio"] = df["zori"] / df["zhvi"].replace(0, np.nan)

    # optional: growth rates (1m, 12m) for key indices
    for col in ["zhvi", "zori", "MEDIAN_SALE_PRICE", "INVENTORY", "MONTHS_OF_SUPPLY"]:
        if col in df.columns:
            df = df.sort_values(["zip", "month"])
            df[f"{col}_mom_pct"] = df.groupby("zip")[col].pct_change(1)
            df[f"{col}_yoy_pct"] = df.groupby("zip")[col].pct_change(12)

    df = df.sort_values(["zip", "month"]).reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"✅ Saved consolidated panel: {out_path}")
    print(f"   shape={df.shape}  zips={df['zip'].nunique():,}  months={df['month'].nunique():,}")


if __name__ == "__main__":
    main()