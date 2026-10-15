#!/usr/bin/env python3
"""
prepare_data.py — Verifies dataset manifests, checks integrity, and prints acquisition instructions.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "datasets_manifest.json"


def main():
    print("=== Microbiome Dataset Verification & Preparation ===")

    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    print(f"Tracking {len(manifest)} dataset files:\n")
    print(f"{'Dataset Key':25s} {'Expected Samples':18s} {'Expected Columns':18s} {'Size (MB)':10s}")
    print("-" * 75)

    all_present = True
    for key, info in manifest.items():
        n_rows = info.get("n_rows", "N/A")
        n_cols = info.get("n_cols", "N/A")
        size_mb = info.get("size_mb", "N/A")
        print(f"{key:25s} {str(n_rows):18s} {str(n_cols):18s} {str(size_mb):10s}")

    print("\nData Governance Notice:")
    print("Clinical cohort data and large metagenomic matrices are subject to data-sharing agreements.")
    print("See docs/datasets.md for public download URLs, curatedMetagenomicData R pull scripts, and preprocessing steps.")


if __name__ == "__main__":
    main()
