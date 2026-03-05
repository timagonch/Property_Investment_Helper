#!/usr/bin/env python3
"""
Replicate a data folder tree into a new folder, but only copy CSVs as "head" samples.

Behavior:
- If dst exists: DELETE it entirely, then recreate (true overwrite)
- Replicate directory structure under src into dst
- For each .csv file: create a copy with ALL columns but only first N rows (default 50)
- Skip any file whose name contains ".tsv000"
- Skip all non-CSV files

Usage:
  python scripts/make_data_samples_tree_simple.py --src data --dst data_samples --rows 50
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


SKIP_DIRS = {".venv", "venv", "__pycache__", ".git"}
SKIP_NAME_CONTAINS = [".tsv000"]


def should_skip_path(path: Path) -> bool:
    # Skip if any parent folder is in SKIP_DIRS
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    # Skip if filename contains forbidden substring(s)
    name_lower = path.name.lower()
    return any(s in name_lower for s in SKIP_NAME_CONTAINS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a sampled replica of a data folder tree (CSV head only).")
    parser.add_argument("--src", type=str, default="data", help="Source folder (default: data)")
    parser.add_argument("--dst", type=str, default="data_samples", help="Destination folder (default: data_samples)")
    parser.add_argument("--rows", type=int, default=50, help="Number of rows to keep per CSV (default: 50)")
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Source folder not found: {src_root}")

    # True overwrite: wipe dst completely
    if dst_root.exists():
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)

    for src_path in src_root.rglob("*"):
        if should_skip_path(src_path):
            continue

        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel

        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue

        # Only process CSV files
        if src_path.suffix.lower() != ".csv":
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        df_head = pd.read_csv(src_path, nrows=args.rows, low_memory=False)
        df_head.to_csv(dst_path, index=False)

        print(f"Wrote sample: {dst_path}  (rows={len(df_head)})")

    print(f"\n✅ Done. Sample tree rebuilt at: {dst_root}")


if __name__ == "__main__":
    main()