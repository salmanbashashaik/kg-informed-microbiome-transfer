#!/usr/bin/env python3
"""
evaluate_loco.py — Command-line interface for Leave-One-Cohort-Out (LOCO) evaluation.
Supports either running evaluation on provided cohort files or displaying canonical evaluation results.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import torch
from microbiome_kg.transformer.config import ModelConfig
from microbiome_kg.transformer.model import MicrobiomeTransformer
from microbiome_kg.evaluation.loco import evaluate_loco_pair


def main():
    parser = argparse.ArgumentParser(description="Evaluate LOCO cross-cohort transfer")
    parser.add_argument("--source_file", type=str, default=None, help="Path to source cohort TSV")
    parser.add_argument("--target_file", type=str, default=None, help="Path to target cohort TSV")
    parser.add_argument("--vocab_file", type=str, default="data/vocab/genus_vocab.txt", help="Path to genus vocabulary")
    parser.add_argument("--graph_ckpt", type=str, default=None, help="Path to graph-init transformer checkpoint")
    parser.add_argument("--random_ckpt", type=str, default=None, help="Path to random-init transformer checkpoint")
    parser.add_argument("--n_seeds", type=int, default=20, help="Number of evaluation seeds")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for training linear head")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_file", type=str, default=None, help="Output JSON path")
    parser.add_argument("--show_canonical", action="store_true", help="Display verified canonical paper results")

    args = parser.parse_args()

    if args.show_canonical or (args.source_file is None and args.target_file is None):
        print("\n=== Canonical IEEE BIBM 2026 LOCO Transfer Results ===")
        repo_root = Path(__file__).resolve().parent.parent
        tbl_path = repo_root / "results" / "tables" / "table2_loco_transfer.md"
        if tbl_path.exists():
            print(open(tbl_path).read())
        else:
            print("Run scripts/generate_tables.py to compile canonical tables.")
        return

    print(f"Loading source: {args.source_file}")
    print(f"Loading target: {args.target_file}")
    source_df = pd.read_csv(args.source_file, sep="\t", index_col=0)
    target_df = pd.read_csv(args.target_file, sep="\t", index_col=0)

    cfg = ModelConfig()
    g_model = MicrobiomeTransformer(cfg, graph_init=True)
    r_model = MicrobiomeTransformer(cfg, graph_init=False)

    if args.graph_ckpt and os.path.exists(args.graph_ckpt):
        ckpt_g = torch.load(args.graph_ckpt, map_location="cpu", weights_only=False)
        g_model.load_state_dict(ckpt_g["model"], strict=False)
        print(f"Loaded graph-init checkpoint from {args.graph_ckpt}")

    if args.random_ckpt and os.path.exists(args.random_ckpt):
        ckpt_r = torch.load(args.random_ckpt, map_location="cpu", weights_only=False)
        r_model.load_state_dict(ckpt_r["model"], strict=False)
        print(f"Loaded random-init checkpoint from {args.random_ckpt}")

    results = evaluate_loco_pair(
        graph_model=g_model,
        random_model=r_model,
        source_df=source_df,
        target_df=target_df,
        vocab_file=args.vocab_file,
        n_seeds=args.n_seeds,
        epochs=args.epochs,
        device=args.device,
    )

    stats = results["stats"]["auroc"]
    print("\nEvaluation Complete:")
    print(f"  Graph AUROC:  {stats['graph_mean']:.4f} ± {stats['graph_std']:.4f}")
    print(f"  Random AUROC: {stats['random_mean']:.4f} ± {stats['random_std']:.4f}")
    print(f"  Delta:        {stats['delta_mean']:+.4f} (95% CI: [{stats['delta_ci95'][0]:.4f}, {stats['delta_ci95'][1]:.4f}])")
    print(f"  Wilcoxon p:   {stats['wilcoxon_p']:.4e}")
    print(f"  Wins:         {results['wins']}/{args.n_seeds}")

    if args.out_file:
        Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.out_file}")


if __name__ == "__main__":
    main()
