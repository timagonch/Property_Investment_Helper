# scripts/prep_redfin_zip_market_tracker.py
# ------------------------------------------------------------
# Reads Redfin Market Tracker TSV (.tsv000), extracts ZIP from REGION,
# filters to your Zillow NC/SC ZIP universe, and writes:
#  - a smaller CSV for exploration
#  - a feature coverage report
# Also prints debug stats so you can validate the parsing.
# ------------------------------------------------------------

from pathlib import Path
import pandas as pd


def main():
    # ----------------------------
    # Project paths
    # ----------------------------
    ROOT = Path(__file__).resolve().parents[1]
    DATA_RAW = ROOT / "data" / "raw"
    DATA_PROCESSED = ROOT / "data" / "processed"
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Inputs (adjust filenames if needed)
    REDFIN_PATH = DATA_RAW / "redfin" / "redfin_zip_market_tracker_zip.tsv000"
    ZILLOW_ZIPS_PATH = DATA_RAW / "zillow" / "zip_universe_nc_sc.csv"

    if not REDFIN_PATH.exists():
        raise FileNotFoundError(f"Missing Redfin file: {REDFIN_PATH}")
    if not ZILLOW_ZIPS_PATH.exists():
        raise FileNotFoundError(f"Missing Zillow ZIP universe file: {ZILLOW_ZIPS_PATH}")

    # Outputs
    OUT_CSV = DATA_PROCESSED / "redfin_market_tracker_zip_nc_sc_long.csv"
    OUT_REPORT = DATA_PROCESSED / "redfin_market_tracker_feature_report.csv"
    OUT_NONMATCH_SAMPLE = DATA_PROCESSED / "redfin_market_tracker_region_nonmatch_sample.csv"

    # ----------------------------
    # Load Zillow ZIP universe
    # ----------------------------
    zdf = pd.read_csv(ZILLOW_ZIPS_PATH)

    if "RegionName" in zdf.columns:
        zdf["zip"] = zdf["RegionName"].astype(str).str.strip().str.zfill(5)
    elif "zip" in zdf.columns:
        zdf["zip"] = zdf["zip"].astype(str).str.strip().str.zfill(5)
    else:
        raise ValueError("Zillow ZIP file must contain 'RegionName' or 'zip' column")

    zdf = zdf[zdf["zip"].str.fullmatch(r"\d{5}", na=False)].copy()
    zip_universe = set(zdf["zip"].tolist())
    print(f"✅ Zillow ZIP universe loaded: {len(zip_universe)} ZIPs")

    # ----------------------------
    # Load Redfin TSV
    # ----------------------------
    print("Loading Redfin TSV (this can take a while)...")
    rdf = pd.read_csv(REDFIN_PATH, sep="\t", low_memory=False)
    print("Raw Redfin shape:", rdf.shape)

    # Required columns
    required = {"REGION", "REGION_TYPE"}
    missing = required - set(rdf.columns)
    if missing:
        print("Columns present (first 60):", list(rdf.columns)[:60])
        raise ValueError(f"Missing required columns in Redfin file: {missing}")

    # ----------------------------
    # Debug: inspect REGION_TYPE values
    # ----------------------------
    rt_counts = rdf["REGION_TYPE"].astype(str).value_counts(dropna=False).head(20)
    print("\nTop REGION_TYPE values:")
    print(rt_counts.to_string())

    # Keep only ZIP-like REGION_TYPE rows (case-insensitive contains 'zip')
    rdf["REGION_TYPE"] = rdf["REGION_TYPE"].astype(str)
    rdf_zip = rdf[rdf["REGION_TYPE"].str.lower().str.contains("zip")].copy()
    print("\nZIP-like REGION_TYPE rows:", rdf_zip.shape)

    # ----------------------------
    # Extract ZIP from REGION robustly
    # REGION examples often include text; we extract first 5-digit sequence.
    # ----------------------------
    region_str = rdf_zip["REGION"].astype(str).str.strip()

    rdf_zip["zip"] = region_str.str.extract(r"(\d{5})", expand=False)

    extracted = rdf_zip["zip"].notna().sum()
    print(f"Extracted 5-digit ZIP from REGION for {extracted} rows out of {len(rdf_zip)}")

    # Show a few REGION examples for sanity
    print("\nSample REGION values (first 10):")
    print(region_str.head(10).to_string(index=False))

    # Drop rows where we couldn't extract a 5-digit ZIP
    rdf_zip = rdf_zip[rdf_zip["zip"].notna()].copy()

    # Ensure 5-digit formatting
    rdf_zip["zip"] = rdf_zip["zip"].astype(str).str.zfill(5)

    # ----------------------------
    # Filter to your Zillow ZIP universe
    # ----------------------------
    before = len(rdf_zip)
    rdf_zip = rdf_zip[rdf_zip["zip"].isin(zip_universe)].copy()
    after = len(rdf_zip)

    print(f"\nRows with extracted ZIP (pre-universe filter): {before}")
    print(f"Rows after filtering to Zillow ZIP universe: {after}")

    if after == 0:
        # Save a small sample of extracted ZIPs that didn't match, for diagnosis
        nonmatch = rdf_zip.copy()
        # We can only do this if we kept non-matching; so reconstruct quickly:
        tmp = rdf[rdf["REGION_TYPE"].str.lower().str.contains("zip")].copy()
        tmp_region = tmp["REGION"].astype(str).str.strip()
        tmp["zip_extracted"] = tmp_region.str.extract(r"(\d{5})", expand=False)
        tmp = tmp[tmp["zip_extracted"].notna()].copy()
        tmp["zip_extracted"] = tmp["zip_extracted"].astype(str).str.zfill(5)
        tmp_nonmatch = tmp[~tmp["zip_extracted"].isin(zip_universe)].copy()
        tmp_nonmatch[["REGION_TYPE", "REGION", "zip_extracted"]].head(200).to_csv(
            OUT_NONMATCH_SAMPLE, index=False
        )
        print(f"\n⚠️ Still 0 matches. Saved non-match sample to: {OUT_NONMATCH_SAMPLE}")
        print("Open it and confirm what ZIPs Redfin contains vs your Zillow universe file.")
        # Still continue to write empty outputs (but report will be useless)

    # ----------------------------
    # Parse dates (useful later)
    # ----------------------------
    for c in ["PERIOD_BEGIN", "PERIOD_END", "LAST_UPDATED"]:
        if c in rdf_zip.columns:
            rdf_zip[c] = pd.to_datetime(rdf_zip[c], errors="coerce")

    # ----------------------------
    # Feature coverage report
    # ----------------------------
    n = len(rdf_zip)
    report_rows = []
    for c in rdf_zip.columns:
        non_null = int(rdf_zip[c].notna().sum())
        non_null_pct = (non_null / n) if n else 0
        nunique = int(rdf_zip[c].nunique(dropna=True))
        report_rows.append({
            "column": c,
            "dtype": str(rdf_zip[c].dtype),
            "non_null_pct": round(non_null_pct, 4),
            "n_unique": nunique,
        })

    report = pd.DataFrame(report_rows).sort_values("non_null_pct", ascending=False)
    report.to_csv(OUT_REPORT, index=False)
    print(f"\n✅ Saved feature report: {OUT_REPORT}")

    # ----------------------------
    # Save filtered long file
    # ----------------------------
    rdf_zip.to_csv(OUT_CSV, index=False)
    print(f"✅ Saved filtered Redfin CSV: {OUT_CSV}")
    print("Final filtered shape:", rdf_zip.shape)

    # Console summaries
    if n > 0:
        print("\nTop 25 columns by non-null % (most complete):")
        print(report.head(25).to_string(index=False))


if __name__ == "__main__":
    main()