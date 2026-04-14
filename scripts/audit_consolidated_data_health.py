#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


CORE_GROUPS = {
    "zillow": ["zhvi", "zori", "zori_to_zhvi_ratio"],
    "redfin": [
        "MEDIAN_SALE_PRICE", "INVENTORY", "NEW_LISTINGS", "PENDING_SALES",
        "MONTHS_OF_SUPPLY", "MEDIAN_DOM", "AVG_SALE_TO_LIST", "SOLD_ABOVE_LIST",
        "PRICE_DROPS", "OFF_MARKET_IN_TWO_WEEKS"
    ],
    "acs": ["B19013_001E", "B25077_001E", "B25064_001E", "owner_rate"],
    "cbp": ["EMP", "ESTAB", "PAYANN", "PAYQTR1", "avg_annual_pay_per_emp"],
    "soi": ["A00100", "A00200", "A00300", "A00600", "N1"],
    "pmms": ["pmms30", "pmms15", "pmms51", "pmms51spread", "PMMS", "MORTGAGE30US"],
}


def pct(x: float) -> str:
    return f"{x*100:.1f}%"


def ensure_month_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "month" not in df.columns:
        raise ValueError("Expected column 'month' in consolidated file.")
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    if df["month"].isna().any():
        bad = df[df["month"].isna()].head(5)
        raise ValueError(f"Could not parse some month values. Examples:\n{bad}")
    return df


def missing_rate_by(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    # Returns a dataframe with group columns + missing rate for each column
    grouped = df.groupby(group_cols, dropna=False)
    out = grouped.apply(lambda g: g.isna().mean(numeric_only=False)).reset_index()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="data/processed/consolidated_zip_month.csv")
    ap.add_argument("--outdir", default="data/processed/audit")
    ap.add_argument("--min-nonnull", type=float, default=0.60,
                    help="Columns with non-null rate below this are flagged (default 0.60).")
    ap.add_argument("--core-window", default="2018-01-01:2022-12-31",
                    help="YYYY-MM-DD:YYYY-MM-DD window for 'full-feature' audit (default 2018-01-01:2022-12-31)")
    args = ap.parse_args()

    infile = Path(args.infile)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile, low_memory=False)
    df = ensure_month_datetime(df)

    # Basic checks
    required = {"zip", "month"}
    missing_required = required - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    df["zip"] = df["zip"].astype(str).str.zfill(5)
    df["year"] = df["month"].dt.year.astype("Int64")

    # 1) Uniqueness / instances
    dup = df.duplicated(["zip", "month"]).sum()
    n_rows = len(df)
    n_zips = df["zip"].nunique()
    n_months = df["month"].nunique()

    # 2) Metadata attributes count (>= 5)
    n_cols = df.shape[1]
    meta_cols = [c for c in df.columns if c not in ("zip", "month")]
    n_meta = len(meta_cols)

    # 3) Overall missingness by column
    col_missing = df.isna().mean().sort_values(ascending=False)
    col_report = (
        col_missing.rename("missing_rate")
        .to_frame()
        .assign(nonnull_rate=lambda d: 1 - d["missing_rate"])
        .assign(missing_pct=lambda d: (d["missing_rate"] * 100).round(1))
        .assign(nonnull_pct=lambda d: (d["nonnull_rate"] * 100).round(1))
    )
    col_report.to_csv(outdir / "missingness_by_column.csv")

    # Flag sparse columns
    sparse_cols = col_report[col_report["nonnull_rate"] < args.min_nonnull].copy()
    sparse_cols.to_csv(outdir / "columns_below_nonnull_threshold.csv")

    # 4) Missingness by month (overall)
    month_missing = df.groupby("month").apply(lambda g: g.isna().mean()).reset_index()
    # Summarize overall missing rate as mean across columns (excluding zip/month)
    feature_cols = [c for c in df.columns if c not in ("zip", "month")]
    month_missing["avg_missing_rate"] = df.groupby("month")[feature_cols].apply(lambda g: g.isna().mean().mean()).values
    month_missing = month_missing[["month", "avg_missing_rate"]].sort_values("avg_missing_rate", ascending=False)
    month_missing.to_csv(outdir / "missingness_by_month_avg.csv", index=False)

    # 5) Missingness by zip (overall)
    zip_missing = df.groupby("zip")[feature_cols].apply(lambda g: g.isna().mean().mean()).reset_index(name="avg_missing_rate")
    zip_missing = zip_missing.sort_values("avg_missing_rate", ascending=False)
    zip_missing.to_csv(outdir / "missingness_by_zip_avg.csv", index=False)

    # 6) Coverage flags by source group
    flags = df[["zip", "month", "year"]].copy()

    for group, cols in CORE_GROUPS.items():
        use = [c for c in cols if c in df.columns]
        if not use:
            # Create empty False flag if none exist
            flags[f"has_{group}"] = False
        else:
            flags[f"has_{group}"] = df[use].notna().any(axis=1)

    flags.to_csv(outdir / "coverage_flags_rowlevel.csv", index=False)

    # Summaries of coverage by ZIP and by month
    cov_by_zip = flags.groupby("zip")[[c for c in flags.columns if c.startswith("has_")]].mean().reset_index()
    cov_by_zip.to_csv(outdir / "coverage_by_zip.csv", index=False)

    cov_by_month = flags.groupby("month")[[c for c in flags.columns if c.startswith("has_")]].mean().reset_index()
    cov_by_month.to_csv(outdir / "coverage_by_month.csv", index=False)

    # Identify ZIPs with no Redfin at all (common issue)
    if "has_redfin" in flags.columns:
        redfin_zip = flags.groupby("zip")["has_redfin"].max().reset_index()
        no_redfin = redfin_zip[redfin_zip["has_redfin"] == 0].copy()
        no_redfin.to_csv(outdir / "zips_with_no_redfin_coverage.csv", index=False)

    # 7) Core-window audit (2018–2022 default)
    start_s, end_s = args.core_window.split(":")
    start = pd.to_datetime(start_s)
    end = pd.to_datetime(end_s)

    core = df[(df["month"] >= start) & (df["month"] <= end)].copy()
    core_flags = flags[(flags["month"] >= start) & (flags["month"] <= end)].copy()

    core_col_missing = core[feature_cols].isna().mean().sort_values(ascending=False)
    core_col_report = (
        core_col_missing.rename("missing_rate")
        .to_frame()
        .assign(nonnull_rate=lambda d: 1 - d["missing_rate"])
        .assign(missing_pct=lambda d: (d["missing_rate"] * 100).round(1))
        .assign(nonnull_pct=lambda d: (d["nonnull_rate"] * 100).round(1))
    )
    core_col_report.to_csv(outdir / "missingness_by_column_core_window.csv")

    core_cov_by_zip = core_flags.groupby("zip")[[c for c in core_flags.columns if c.startswith("has_")]].mean().reset_index()
    core_cov_by_zip.to_csv(outdir / "coverage_by_zip_core_window.csv", index=False)

    core_cov_by_month = core_flags.groupby("month")[[c for c in core_flags.columns if c.startswith("has_")]].mean().reset_index()
    core_cov_by_month.to_csv(outdir / "coverage_by_month_core_window.csv", index=False)

    # 8) Summary TXT (human-readable)
    summary_lines = []
    summary_lines.append("=== Consolidated Data Health Summary ===")
    summary_lines.append(f"File: {infile.resolve()}")
    summary_lines.append(f"Rows (instances): {n_rows:,}")
    summary_lines.append(f"Unique ZIPs: {n_zips:,}")
    summary_lines.append(f"Unique months: {n_months:,}")
    summary_lines.append(f"Columns: {n_cols:,} (metadata attributes excluding keys: {n_meta:,})")
    summary_lines.append(f"Duplicate (zip,month) rows: {dup:,}")
    summary_lines.append("")
    summary_lines.append("Top 15 most-missing columns (overall):")
    for col, rate in col_missing.head(15).items():
        summary_lines.append(f"  - {col}: missing {pct(rate)} (nonnull {pct(1-rate)})")
    summary_lines.append("")
    summary_lines.append(f"Columns below nonnull threshold {pct(args.min_nonnull)}: {len(sparse_cols):,}")
    summary_lines.append("")
    summary_lines.append(f"Core window: {start.date()} to {end.date()}  (rows={len(core):,})")
    summary_lines.append("Top 15 most-missing columns (core window):")
    for col, rate in core_col_missing.head(15).items():
        summary_lines.append(f"  - {col}: missing {pct(rate)} (nonnull {pct(1-rate)})")

    (outdir / "SUMMARY.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"\n✅ Wrote audit outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()