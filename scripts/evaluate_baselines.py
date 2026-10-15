#!/usr/bin/env python3
"""
evaluate_baselines.py — Command-line interface for classical Random Forest and CORAL baselines.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from microbiome_kg.baselines.random_forest import run_intersection_rf, run_zerofill_rf, run_taxonomy_rf
from microbiome_kg.baselines.coral import run_coral_rf


def main():
    parser = argparse.ArgumentParser(description="Evaluate classical machine learning baselines")
    parser.add_argument("--source_file", type=str, default=None, help="Source cohort TSV")
    parser.add_argument("--target_file", type=str, default=None, help="Target cohort TSV")
    parser.add_argument("--taxonomy_map", type=str, default="results/canonical/baselines/genus_taxonomy_map.json", help="Genus to family map")
    parser.add_argument("--method", type=str, choices=["all", "intersection", "zerofill", "taxonomy", "coral"], default="all")
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--show_canonical", action="store_true", help="Display verified canonical baseline results")

    args = parser.parse_args()

    if args.show_canonical or (args.source_file is None and args.target_file is None):
        print("\n=== Canonical IEEE BIBM 2026 Classical Baseline Results ===")
        repo_root = Path(__file__).resolve().parent.parent
        tbl_path = repo_root / "results" / "tables" / "table3_classical_baselines.md"
        if tbl_path.exists():
            print(open(tbl_path).read())
        else:
            print("Run scripts/generate_tables.py to compile canonical tables.")
        return

    source_df = pd.read_csv(args.source_file, sep="\t", index_col=0)
    target_df = pd.read_csv(args.target_file, sep="\t", index_col=0)

    print(f"Evaluating transfer: {args.source_file} -> {args.target_file}")

    if args.method in ("all", "intersection"):
        res = run_intersection_rf(source_df, target_df, n_seeds=args.n_seeds)
        print(f"  Intersection RF AUROC: {res['auroc_mean']:.4f} ± {res['auroc_std']:.4f} (Shared features: {res['shared_features_count']})")

    if args.method in ("all", "zerofill"):
        res = run_zerofill_rf(source_df, target_df, n_seeds=args.n_seeds)
        print(f"  Zero-Fill RF AUROC:    {res['auroc_mean']:.4f} ± {res['auroc_std']:.4f}")

    if args.method in ("all", "taxonomy"):
        if Path(args.taxonomy_map).exists():
            with open(args.taxonomy_map) as f:
                tax_map = json.load(f)
            res = run_taxonomy_rf(source_df, target_df, genus_to_family=tax_map, n_seeds=args.n_seeds)
            print(f"  Taxonomy RF AUROC:     {res['auroc_mean']:.4f} ± {res['auroc_std']:.4f}")

    if args.method in ("all", "coral"):
        res = run_coral_rf(source_df, target_df, n_seeds=args.n_seeds)
        print(f"  CORAL RF AUROC:        {res['auroc_mean']:.4f} ± {res['auroc_std']:.4f}")


if __name__ == "__main__":
    main()
