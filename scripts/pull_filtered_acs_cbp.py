# scripts/pull_filtered_acs_cbp.py
# ------------------------------------------------------------
# Pulls & filters Census ACS (ZCTA) and CBP ZIP to your Zillow ZIP universe.
# - ACS 5-year ZCTA: 2013–2022
# - CBP ZIP via /cbp: 2018–2022 (ZIP geography unsupported before 2018)
# Also reports how many Zillow ZIPs are missing from ACS each year.
# ------------------------------------------------------------

from pathlib import Path
import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv


def census_get(url: str, params: dict, timeout: int = 60):
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code != 200:
        print("\n--- HTTP ERROR ---")
        print("URL:", r.url)
        print("Status:", r.status_code)
        print("Response:", r.text[:1200])
        r.raise_for_status()
    return r.json()


def to_df(data):
    return pd.DataFrame(data[1:], columns=data[0])


def numericize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_zillow_zip_universe(path: Path) -> set[str]:
    df = pd.read_csv(path)
    if "RegionName" in df.columns:
        s = df["RegionName"]
    elif "zip" in df.columns:
        s = df["zip"]
    else:
        raise ValueError("Zillow ZIP universe file must have 'RegionName' or 'zip' column")

    s = s.astype(str).str.strip().str.zfill(5)
    s = s[s.str.fullmatch(r"\d{5}", na=False)]
    return set(s.tolist())


def pull_acs_zcta_filtered(year: int, zip_universe: set[str], acs_vars: list[str], key: str) -> pd.DataFrame:
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "NAME," + ",".join(acs_vars),
        "for": "zip code tabulation area:*",
        "key": key,
    }
    data = census_get(url, params)
    df = numericize(to_df(data), acs_vars)

    # Standardize column names
    if "zip code tabulation area" in df.columns and "zcta" not in df.columns:
        df = df.rename(columns={"zip code tabulation area": "zcta"})
    if "zcta" not in df.columns:
        raise ValueError(f"ACS response missing ZCTA column for year {year}")

    df["zcta"] = df["zcta"].astype(str).str.strip().str.zfill(5)

    # "Convert" ZCTA5 -> ZIP5 by identity and filter to Zillow ZIP universe
    df["zip"] = df["zcta"]
    df["year"] = year

    # Filter
    df = df[df["zip"].isin(zip_universe)].copy()
    return df


def pull_cbp_zip_filtered(year: int, zip_universe: set[str], cbp_vars: list[str], key: str) -> pd.DataFrame:
    url = f"https://api.census.gov/data/{year}/cbp"

    # For 2018–2022, NAICS2017 works for total all-sectors (00)
    params = {
        "get": "NAME," + ",".join(cbp_vars),
        "for": "zip code:*",
        "NAICS2017": "00",
        "key": key,
    }
    data = census_get(url, params)
    df = numericize(to_df(data), cbp_vars)

    # Standardize column names
    if "zip code" in df.columns and "zip" not in df.columns:
        df = df.rename(columns={"zip code": "zip"})
    if "zip" not in df.columns:
        raise ValueError(f"CBP response missing ZIP column for year {year}")

    df["zip"] = df["zip"].astype(str).str.strip().str.zfill(5)
    df["year"] = year

    df = df[df["zip"].isin(zip_universe)].copy()
    return df


def main():
    # ----------------------------
    # Paths (scripts/ -> project root)
    # ----------------------------
    ROOT = Path(__file__).resolve().parents[1]
    DATA_RAW = ROOT / "data" / "raw"
    DATA_PROCESSED = ROOT / "data" / "processed"
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Inputs
    ZILLOW_ZIPS_PATH = DATA_RAW / "zillow" / "unique_zipcodes_nc_sc.csv"
    if not ZILLOW_ZIPS_PATH.exists():
        raise FileNotFoundError(f"Missing Zillow ZIP universe: {ZILLOW_ZIPS_PATH}")

    # Load API key
    load_dotenv(ROOT / ".env")
    CENSUS_KEY = os.getenv("CENSUS_API_KEY") or os.getenv("CENSUS_KEY")
    if not CENSUS_KEY:
        raise SystemExit("Missing CENSUS_API_KEY (or CENSUS_KEY) in .env")

    # Years (based on what you verified)
    ACS_YEARS = list(range(2013, 2023))       # 2013–2022
    CBP_ZIP_YEARS = list(range(2018, 2023))   # 2018–2022 (ZIP geo supported)

    # Variables
    ACS_VARS = [
        "B19013_001E",  # Median household income
        "B25077_001E",  # Median home value
        "B25064_001E",  # Median gross rent
        "B25003_002E",  # Owner occupied
        "B25003_003E",  # Renter occupied
    ]
    CBP_VARS = ["ESTAB", "EMP", "PAYANN", "PAYQTR1"]

    # Timing
    SLEEP_SEC = 0.35

    # Output folders
    OUT_ACS_YR = DATA_PROCESSED / "census_acs_filtered_by_year"
    OUT_CBP_YR = DATA_PROCESSED / "census_cbp_filtered_by_year"
    OUT_ACS_YR.mkdir(parents=True, exist_ok=True)
    OUT_CBP_YR.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load Zillow ZIP universe
    # ----------------------------
    zip_universe = load_zillow_zip_universe(ZILLOW_ZIPS_PATH)
    print(f"✅ Loaded Zillow ZIP universe: {len(zip_universe)} ZIPs")

    # ----------------------------
    # Pull ACS filtered
    # ----------------------------
    acs_all = []
    missing_rows = []

    for y in ACS_YEARS:
        print(f"\n=== ACS {y} (ZCTA -> ZIP identity, filtered) ===")
        df = pull_acs_zcta_filtered(y, zip_universe, ACS_VARS, CENSUS_KEY)

        # Coverage / missing
        zips_seen = set(df["zip"].unique().tolist())
        missing = sorted(list(zip_universe - zips_seen))

        missing_rows.append({"year": y, "zillow_zip_count": len(zip_universe),
                             "acs_matched_zip_count": len(zips_seen),
                             "acs_missing_zip_count": len(missing)})

        print(f"Matched ZIPs in ACS {y}: {len(zips_seen)} / {len(zip_universe)}")
        print(f"Missing ZIPs in ACS {y}: {len(missing)}")

        out = OUT_ACS_YR / f"acs5_filtered_{y}.csv"
        df.to_csv(out, index=False)
        print(f"✅ Saved {out} shape={df.shape}")

        acs_all.append(df)
        time.sleep(SLEEP_SEC)

    # Save missing report
    missing_df = pd.DataFrame(missing_rows)
    missing_report_path = DATA_PROCESSED / "acs_missing_zip_report_2013_2022.csv"
    missing_df.to_csv(missing_report_path, index=False)
    print(f"\n✅ Saved ACS missing ZIP report: {missing_report_path}")

    # Save ACS long
    acs_long = pd.concat(acs_all, ignore_index=True)
    acs_long_path = DATA_PROCESSED / "acs5_filtered_long_2013_2022.csv"
    acs_long.to_csv(acs_long_path, index=False)
    print(f"✅ Saved ACS long: {acs_long_path} shape={acs_long.shape}")

    # ----------------------------
    # Pull CBP ZIP filtered (2018+ only)
    # ----------------------------
    cbp_all = []
    for y in CBP_ZIP_YEARS:
        print(f"\n=== CBP ZIP {y} (filtered) ===")
        df = pull_cbp_zip_filtered(y, zip_universe, CBP_VARS, CENSUS_KEY)

        out = OUT_CBP_YR / f"cbp_zip_filtered_{y}.csv"
        df.to_csv(out, index=False)
        print(f"✅ Saved {out} shape={df.shape}")

        cbp_all.append(df)
        time.sleep(SLEEP_SEC)

    cbp_long = pd.concat(cbp_all, ignore_index=True)
    cbp_long_path = DATA_PROCESSED / "cbp_zip_filtered_long_2018_2022.csv"
    cbp_long.to_csv(cbp_long_path, index=False)
    print(f"\n✅ Saved CBP long: {cbp_long_path} shape={cbp_long.shape}")

    print("\n✅ DONE")


if __name__ == "__main__":
    main()