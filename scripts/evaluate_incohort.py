#!/usr/bin/env python3
"""
evaluate_incohort.py — Evaluates in-cohort classification across label subsampling fractions.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
from microbiome_kg.data.dataset import TaskDataset
from microbiome_kg.transformer.config import ModelConfig
from microbiome_kg.transformer.model import MicrobiomeTransformer
from microbiome_kg.evaluation.probe import extract_cls_features, train_linear_head
from microbiome_kg.evaluation.metrics import compute_paired_stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate in-cohort classification")
    parser.add_argument("--task_file", type=str, default=None, help="Path to cohort task TSV")
    parser.add_argument("--vocab_file", type=str, default="data/vocab/genus_vocab.txt", help="Path to genus vocabulary")
    parser.add_argument("--graph_ckpt", type=str, default=None, help="Path to graph-init transformer checkpoint")
    parser.add_argument("--random_ckpt", type=str, default=None, help="Path to random-init transformer checkpoint")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.02, 0.05, 0.25, 0.50, 1.00])
    parser.add_argument("--n_seeds", type=int, default=20, help="Number of random seeds")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_file", type=str, default=None, help="Output JSON path")
    parser.add_argument("--show_canonical", action="store_true", help="Display verified canonical paper results")

    args = parser.parse_args()

    if args.show_canonical or args.task_file is None:
        print("\n=== Canonical IEEE BIBM 2026 In-Cohort Classification Results ===")
        repo_root = Path(__file__).resolve().parent.parent
        tbl_path = repo_root / "results" / "tables" / "table1_incohort_fractions.md"
        if tbl_path.exists():
            print(open(tbl_path).read())
        else:
            print("Run scripts/generate_tables.py to compile canonical tables.")
        return

    print(f"Loading task data: {args.task_file}")
    df = pd.read_csv(args.task_file, sep="\t", index_col=0)

    cfg = ModelConfig()
    g_model = MicrobiomeTransformer(cfg, graph_init=True)
    r_model = MicrobiomeTransformer(cfg, graph_init=False)

    if args.graph_ckpt and os.path.exists(args.graph_ckpt):
        ckpt_g = torch.load(args.graph_ckpt, map_location="cpu", weights_only=False)
        g_model.load_state_dict(ckpt_g["model"], strict=False)

    if args.random_ckpt and os.path.exists(args.random_ckpt):
        ckpt_r = torch.load(args.random_ckpt, map_location="cpu", weights_only=False)
        r_model.load_state_dict(ckpt_r["model"], strict=False)

    ds = TaskDataset(df, vocab_file=args.vocab_file)
    g_feat, labels = extract_cls_features(g_model, ds, device=args.device)
    r_feat, _ = extract_cls_features(r_model, ds, device=args.device)

    n_samples = len(labels)
    indices = np.arange(n_samples)

    results = {}
    for frac in args.fractions:
        g_aurocs, r_aurocs = [], []
        frac_str = f"{frac:.2f}"
        for s in range(args.n_seeds):
            rng = np.random.RandomState(s)
            shuffled = rng.permutation(indices)
            n_train = max(4, int(n_samples * 0.8 * frac))
            train_idx = shuffled[:n_train]
            test_idx = shuffled[int(n_samples * 0.8):]

            _, g_m = train_linear_head(
                g_feat[train_idx], labels[train_idx],
                g_feat[test_idx], labels[test_idx],
                d_model=cfg.d_model, seed=s, device=args.device
            )
            _, r_m = train_linear_head(
                r_feat[train_idx], labels[train_idx],
                r_feat[test_idx], labels[test_idx],
                d_model=cfg.d_model, seed=s, device=args.device
            )
            g_aurocs.append(g_m["auroc"])
            r_aurocs.append(r_m["auroc"])

        stats = compute_paired_stats(g_aurocs, r_aurocs)
        results[frac_str] = {
            "stats": {
                "auroc": {
                    "graph_mean": stats["a_mean"],
                    "random_mean": stats["b_mean"],
                    "delta_mean": stats["delta_mean"],
                    "wilcoxon_p": stats["wilcoxon_p"],
                }
            }
        }
        print(f"Fraction {frac*100:.0f}%: Graph={stats['a_mean']:.4f}, Random={stats['b_mean']:.4f}, Delta={stats['delta_mean']:+.4f}")

    if args.out_file:
        Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.out_file}")


if __name__ == "__main__":
    main()
