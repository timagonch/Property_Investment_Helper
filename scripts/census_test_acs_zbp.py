# census_test_acs_zipbp.py
import os
import sys
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CENSUS_KEY = os.getenv("CENSUS_API_KEY") or os.getenv("CENSUS_KEY")
if not CENSUS_KEY:
    raise SystemExit("Missing CENSUS_API_KEY in .env")

def census_get(url: str, params: dict, timeout: int = 30):
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code != 200:
        print("\n--- HTTP ERROR ---")
        print("URL:", r.url)
        print("Status:", r.status_code)
        print("Response:", r.text[:1000])
        r.raise_for_status()
    return r.json()

def to_df(data):
    return pd.DataFrame(data[1:], columns=data[0])

def numericize(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

print("✅ Loaded Census API key from .env")

# ------------------------------------------------------------
# 1) ACS TEST (ACS 5-year, ZCTA geography)
# ------------------------------------------------------------
ACS_YEAR = 2022
ACS_URL = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
TEST_ZCTA = "28202"  # ZCTA

ACS_VARS = [
    "B19013_001E",  # Median household income
    "B25077_001E",  # Median home value
    "B25064_001E",  # Median gross rent
    "B25003_002E",  # Owner-occupied count
    "B25003_003E",  # Renter-occupied count
]

print("\n--- ACS (5-year) quick test ---")
acs_params = {
    "get": "NAME," + ",".join(ACS_VARS),
    "for": f"zip code tabulation area:{TEST_ZCTA}",
    "key": CENSUS_KEY,
}
acs_json = census_get(ACS_URL, acs_params)
acs_df = numericize(to_df(acs_json), ACS_VARS)
print("✅ ACS test query succeeded")
print(acs_df)

# ------------------------------------------------------------
# 2) ZIP BUSINESS PATTERNS TEST
# IMPORTANT: 2019+ ZBP is delivered through the CBP API (/cbp)
# ------------------------------------------------------------
CBP_YEAR = 2022
CBP_URL = f"https://api.census.gov/data/{CBP_YEAR}/cbp"
TEST_ZIP = "28202"  # USPS ZIP

# These are the standard CBP/ZBP-style measures:
# - ESTAB: establishments
# - EMP: employment
# - PAYANN: annual payroll
# - PAYQTR1: first quarter payroll
# NOTE: Some vintages use EMP rather than EMPP.
CBP_ZIP_VARS = ["ESTAB", "EMP", "PAYANN", "PAYQTR1"]

print("\n--- ZIP Business Patterns via CBP quick test ---")

# ZIP-level query, all sectors (NAICS=00)
cbp_zip_params = {
    "get": "NAME," + ",".join(CBP_ZIP_VARS),
    "for": f"zip code:{TEST_ZIP}",
    "NAICS2017": "00",   # total for all sectors
    "key": CENSUS_KEY,
}

try:
    cbp_zip_json = census_get(CBP_URL, cbp_zip_params)
except requests.HTTPError:
    # Fallback: some years use NAICS2022 / NAICS2012 etc.
    # If you get a NAICS param error, comment the NAICS line and retry,
    # then we’ll inspect variables.
    print("\nIf this failed due to NAICS parameter naming, try these alternatives:")
    print(" - Replace NAICS2017 with NAICS2022 (or NAICS2012) depending on vintage.")
    print(" - Or remove the NAICS param entirely for a first test.")
    raise

cbp_zip_df = to_df(cbp_zip_json)
cbp_zip_df = numericize(cbp_zip_df, CBP_ZIP_VARS)
print("✅ CBP ZIP-level test query succeeded")
print(cbp_zip_df)

# ------------------------------------------------------------
# 3) OPTIONAL: Pull NC/SC ZIP sample (small) to prove scaling
# ------------------------------------------------------------
PULL_NC_SC_SAMPLE = True

if PULL_NC_SC_SAMPLE:
    print("\n--- Pulling NC/SC ZIP sample (first ~50 rows) ---")
    # NOTE: CBP API does not let you filter by state directly at ZIP level.
    # We'll pull all ZIPs and then filter using ZIP prefix logic as a quick test.
    # For production, use a ZIP->State crosswalk (HUD/USPS) to filter exactly.
    params_all = {
        "get": "NAME," + ",".join(CBP_ZIP_VARS),
        "for": "zip code:*",
        "NAICS2017": "00",
        "key": CENSUS_KEY,
    }
    all_json = census_get(CBP_URL, params_all)
    all_df = numericize(to_df(all_json), CBP_ZIP_VARS)
    all_df.rename(columns={"zip code": "zip"}, inplace=True)

    # QUICK test filter (NOT perfect): NC often 27xxx–28xxx, SC 29xxx
    all_df["zip"] = all_df["zip"].astype(str).str.zfill(5)
    approx_nc_sc = all_df[
        all_df["zip"].str.startswith(("27", "28", "29"))
    ].copy()

    print("All ZIP rows:", len(all_df))
    print("Approx NC/SC rows:", len(approx_nc_sc))
    print(approx_nc_sc.head(10))

print("\n✅ Done.")