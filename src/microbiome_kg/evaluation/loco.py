"""
loco.py — Leave-One-Cohort-Out (LOCO) cross-cohort transfer evaluation protocol.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch

from microbiome_kg.data.dataset import TaskDataset
from microbiome_kg.transformer.model import MicrobiomeTransformer
from microbiome_kg.evaluation.probe import extract_cls_features, train_linear_head
from microbiome_kg.evaluation.metrics import compute_paired_stats


def evaluate_loco_pair(
    graph_model: MicrobiomeTransformer,
    random_model: MicrobiomeTransformer,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    vocab_file: str,
    n_seeds: int = 20,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Evaluates cross-cohort transfer from source_df to target_df across paired seeds.
    Protocol:
      - Frozen backbone
      - Frozen embedding table & projection
      - Linear head trained on source cohort
      - Evaluated on held-out target cohort without adaptation
    """
    source_ds = TaskDataset(source_df, vocab_file=vocab_file)
    target_ds = TaskDataset(target_df, vocab_file=vocab_file)

    # Extract frozen CLS features
    g_train_x, train_y = extract_cls_features(graph_model, source_ds, device=device)
    g_test_x, test_y = extract_cls_features(graph_model, target_ds, device=device)

    r_train_x, _ = extract_cls_features(random_model, source_ds, device=device)
    r_test_x, _ = extract_cls_features(random_model, target_ds, device=device)

    per_seed = []
    g_aurocs = []
    r_aurocs = []

    for seed in range(n_seeds):
        _, g_metrics = train_linear_head(
            g_train_x, train_y, g_test_x, test_y,
            d_model=graph_model.d_model, epochs=epochs, lr=lr, seed=seed, device=device
        )
        _, r_metrics = train_linear_head(
            r_train_x, train_y, r_test_x, test_y,
            d_model=random_model.d_model, epochs=epochs, lr=lr, seed=seed, device=device
        )

        per_seed.append({"graph": g_metrics, "random": r_metrics})
        g_aurocs.append(g_metrics["auroc"])
        r_aurocs.append(r_metrics["auroc"])

    stats = compute_paired_stats(g_aurocs, r_aurocs)

    return {
        "per_seed": per_seed,
        "stats": {
            "auroc": {
                "graph_mean": stats["a_mean"],
                "graph_std": stats["a_std"],
                "random_mean": stats["b_mean"],
                "random_std": stats["b_std"],
                "delta_mean": stats["delta_mean"],
                "delta_std": stats["delta_std"],
                "delta_ci95": stats["delta_ci95"],
                "wilcoxon_p": stats["wilcoxon_p"],
                "cohens_d": stats["cohens_d"],
            }
        },
        "wins": stats["wins"],
    }
