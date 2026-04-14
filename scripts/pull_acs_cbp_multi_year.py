import os
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# ----------------------------
# CONFIG
# ----------------------------
load_dotenv()
CENSUS_KEY = os.getenv("CENSUS_API_KEY") or os.getenv("CENSUS_KEY")
if not CENSUS_KEY:
    raise SystemExit("Missing CENSUS_API_KEY in .env")

OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

YEARS = list(range(2013, 2023))  # 2013–2022 inclusive (edit as needed)
SLEEP_SEC = 0.4  # be polite to the API

# ACS 5-year (ZCTA-level) variables (edit/expand later)
ACS_VARS = [
    "B19013_001E",  # median household income
    "B25077_001E",  # median home value
    "B25064_001E",  # median gross rent
    "B25003_002E",  # owner-occupied count
    "B25003_003E",  # renter-occupied count
]

# CBP ZIP-level variables (ZIP Business Patterns via /cbp)
CBP_VARS = ["ESTAB", "EMP", "PAYANN", "PAYQTR1"]

# ----------------------------
# HELPERS
# ----------------------------
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

# ----------------------------
# PULL ACS (ZCTA)
# ----------------------------
def pull_acs_zcta(year: int) -> pd.DataFrame:
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "NAME," + ",".join(ACS_VARS),
        "for": "zip code tabulation area:*",
        "key": CENSUS_KEY,
    }
    data = census_get(url, params)
    df = numericize(to_df(data), ACS_VARS)
    df.rename(columns={"zip code tabulation area": "zcta"}, inplace=True)
    df["year"] = year
    return df

# ----------------------------
# PULL CBP ZIP (ZIP business patterns)
# ----------------------------
def pull_cbp_zip(year: int) -> pd.DataFrame:
    url = f"https://api.census.gov/data/{year}/cbp"
    params = {
        "get": "NAME," + ",".join(CBP_VARS),
        "for": "zip code:*",
        "NAICS2017": "00",  # total, all sectors
        "key": CENSUS_KEY,
    }
    data = census_get(url, params)
    df = numericize(to_df(data), CBP_VARS)
    df.rename(columns={"zip code": "zip"}, inplace=True)
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    df["year"] = year
    return df

# ----------------------------
# MAIN LOOP
# ----------------------------
def main():
    all_acs_files = []
    all_cbp_files = []

    for y in YEARS:
        print(f"\n=== Year {y} ===")

        # ACS
        try:
            print("Pulling ACS (ZCTA)...")
            acs_df = pull_acs_zcta(y)
            acs_path = OUT_DIR / f"acs5_zcta_{y}.csv"
            acs_df.to_csv(acs_path, index=False)
            print(f"✅ Saved {acs_path}  shape={acs_df.shape}")
            all_acs_files.append(str(acs_path))
        except Exception as e:
            print(f"❌ ACS failed for {y}: {e}")

        time.sleep(SLEEP_SEC)

        # CBP ZIP
        try:
            print("Pulling CBP (ZIP)...")
            cbp_df = pull_cbp_zip(y)
            cbp_path = OUT_DIR / f"cbp_zip_{y}.csv"
            cbp_df.to_csv(cbp_path, index=False)
            print(f"✅ Saved {cbp_path}  shape={cbp_df.shape}")
            all_cbp_files.append(str(cbp_path))
        except Exception as e:
            print(f"❌ CBP failed for {y}: {e}")

        time.sleep(SLEEP_SEC)

    print("\nDONE.")
    print("ACS files:", len(all_acs_files))
    print("CBP files:", len(all_cbp_files))

if __name__ == "__main__":
    main()