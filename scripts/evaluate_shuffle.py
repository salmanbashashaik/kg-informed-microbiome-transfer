#!/usr/bin/env python3
"""
evaluate_shuffle.py — Summarizes shuffle and structural control experiments against the canonical KG.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DIR = REPO_ROOT / "results" / "canonical"


def main():
    parser = argparse.ArgumentParser(description="Evaluate shuffle controls")
    parser.add_argument("--show_canonical", action="store_true", default=True)
    args = parser.parse_args()

    print("\n=== Canonical Structural Controls (MetaCardis T2D) ===")
    t2d_res = CANONICAL_DIR / "incohort" / "t2d_metacardis" / "results.json"
    if not t2d_res.exists():
        print("Canonical in-cohort results not found.")
        return

    data = json.load(open(t2d_res))
    fracs = ["0.01", "0.02", "0.05", "0.25", "0.50", "1.00"]

    print(f"{'Fraction':10s} {'True KG (Graph Init)':25s} {'Random Init':15s} {'Shuffled KG (Mean)':20s}")
    print("-" * 75)

    shuffle_dir = CANONICAL_DIR / "shuffle"

    for f in fracs:
        g = data[f]["stats"]["auroc"]["graph_mean"]
        r = data[f]["stats"]["auroc"]["random_mean"]

        shuf_vals = []
        for s in range(1, 6):
            sf = shuffle_dir / f"seed{s}_metacardis" / "results.json"
            if sf.exists():
                sd = json.load(open(sf))
                shuf_vals.append(sd[f]["stats"]["auroc"]["random_mean"])

        shuf_mean = sum(shuf_vals) / len(shuf_vals) if shuf_vals else float("nan")
        pct = f"{int(float(f)*100)}%"
        print(f"{pct:10s} {g:25.4f} {r:15.4f} {shuf_mean:20.4f}")


if __name__ == "__main__":
    main()
