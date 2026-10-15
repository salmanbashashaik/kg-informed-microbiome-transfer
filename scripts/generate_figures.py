#!/usr/bin/env python3
"""
generate_figures.py — Generates publication-quality figures directly from canonical JSON results.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DIR = REPO_ROOT / "results" / "canonical"
FIGS_DIR = REPO_ROOT / "results" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

LOCO_PAIRS = [
    ("Qin→Karlsson", "loco_qin_to_karlsson.json", "qin_to_karlsson"),
    ("Karlsson→Qin", "loco_karlsson_to_qin.json", "karlsson_to_qin"),
    ("Qin→MetaCardis", "loco_qin_to_metacardis.json", "qin_to_metacardis"),
    ("MetaCardis→Qin", "loco_metacardis_to_qin.json", "metacardis_to_qin"),
    ("Karlsson→MetaCardis", "loco_karlsson_to_metacardis.json", "karlsson_to_metacardis"),
    ("MetaCardis→Karlsson", "loco_metacardis_to_karlsson.json", "metacardis_to_karlsson"),
    ("ACVD→MetaCardis-CVD", "loco_acvd_to_metacardis_cvd.json", "acvd_to_metacardis_cvd"),
    ("MetaCardis-CVD→ACVD", "loco_metacardis_cvd_to_acvd.json", "metacardis_cvd_to_acvd"),
]


def make_fig_loco():
    loco_dir = CANONICAL_DIR / "loco"
    rf_file = loco_dir / "rf_loco_results.json"
    rf_data = json.load(open(rf_file)) if rf_file.exists() else {}

    labels = []
    g_means, r_means, rf_means = [], [], []

    for label, fname, key in LOCO_PAIRS:
        fpath = loco_dir / fname
        if not fpath.exists():
            continue
        data = json.load(open(fpath))
        stats = data.get("stats", {}).get("auroc", {})
        labels.append(label)
        g_means.append(stats.get("graph_mean", 0.0))
        r_means.append(stats.get("random_mean", 0.0))
        rf_means.append(rf_data.get(key, {}).get("auroc_mean", np.nan))

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5.5))
    rects1 = ax.bar(x - width, g_means, width, label="Graph-Init Transformer", color="#1f77b4")
    rects2 = ax.bar(x, r_means, width, label="Random-Init Transformer", color="#aec7e8")
    rects3 = ax.bar(x + width, rf_means, width, label="Random Forest Baseline", color="#ff7f0e")

    ax.set_ylabel("Cross-Cohort AUROC")
    ax.set_title("Leave-One-Cohort-Out (LOCO) Generalization Across Clinical Cohorts")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Chance Level")
    ax.set_ylim(0.40, 0.78)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    out_path = FIGS_DIR / "fig3_loco_transfer.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")


def make_fig_incohort():
    incohort_dir = CANONICAL_DIR / "incohort"
    cohorts = [
        ("T2D — Qin", "t2d_qin"),
        ("T2D — Karlsson", "t2d_karlsson"),
        ("T2D — MetaCardis", "t2d_metacardis"),
        ("CVD — MetaCardis", "cvd_metacardis"),
    ]
    fracs = [0.01, 0.02, 0.05, 0.25, 0.50, 1.00]
    frac_strs = ["0.01", "0.02", "0.05", "0.25", "0.50", "1.00"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, (title, cname) in enumerate(cohorts):
        ax = axes[idx]
        res_file = incohort_dir / cname / "results.json"
        if not res_file.exists():
            continue
        data = json.load(open(res_file))
        g_vals, r_vals = [], []
        for fs in frac_strs:
            if fs in data:
                stats = data[fs].get("stats", {}).get("auroc", {})
                g_vals.append(stats.get("graph_mean", np.nan))
                r_vals.append(stats.get("random_mean", np.nan))
            else:
                g_vals.append(np.nan)
                r_vals.append(np.nan)

        ax.plot(fracs, g_vals, "o-", color="#1f77b4", label="Graph Init", linewidth=2)
        ax.plot(fracs, r_vals, "s--", color="#e377c2", label="Random Init", linewidth=2)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xticks(fracs)
        ax.set_xticklabels(["1%", "2%", "5%", "25%", "50%", "100%"])
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx % 2 == 0:
            ax.set_ylabel("In-Cohort AUROC")
        if idx >= 2:
            ax.set_xlabel("Label Fraction")
        if idx == 0:
            ax.legend(loc="lower right")

    fig.suptitle("In-Cohort Classification AUROC vs. Label Fraction", fontsize=15)
    fig.tight_layout()
    out_path = FIGS_DIR / "fig4_incohort_fractions.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")


def make_fig_shuffle():
    # Shuffle control comparison
    incohort_dir = CANONICAL_DIR / "incohort"
    shuffle_dir = CANONICAL_DIR / "shuffle"

    frac_strs = ["0.01", "0.02", "0.05", "0.25", "0.50", "1.00"]
    fracs = [0.01, 0.02, 0.05, 0.25, 0.50, 1.00]

    # MetaCardis T2D
    t2d_res = incohort_dir / "t2d_metacardis" / "results.json"
    if not t2d_res.exists():
        return
    data_t2d = json.load(open(t2d_res))

    g_vals = [data_t2d[fs]["stats"]["auroc"]["graph_mean"] for fs in frac_strs if fs in data_t2d]
    r_vals = [data_t2d[fs]["stats"]["auroc"]["random_mean"] for fs in frac_strs if fs in data_t2d]

    # Collect 5 shuffle seeds
    shuf_matrix = []
    for s in range(1, 6):
        sf = shuffle_dir / f"seed{s}_metacardis" / "results.json"
        if sf.exists():
            s_data = json.load(open(sf))
            s_vals = [s_data[fs]["stats"]["auroc"]["random_mean"] for fs in frac_strs if fs in s_data]
            shuf_matrix.append(s_vals)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fracs, g_vals, "o-", color="#1f77b4", label="Graph Init (True KG)", linewidth=2.5)
    ax.plot(fracs, r_vals, "s--", color="#aec7e8", label="Random Init", linewidth=2)

    if shuf_matrix:
        shuf_mean = np.mean(shuf_matrix, axis=0)
        ax.plot(fracs, shuf_mean, "^:", color="#d62728", label="Shuffled KG Init (5 Seeds)", linewidth=2)

    ax.set_xscale("log")
    ax.set_xticks(fracs)
    ax.set_xticklabels(["1%", "2%", "5%", "25%", "50%", "100%"])
    ax.set_xlabel("Label Fraction")
    ax.set_ylabel("AUROC (MetaCardis T2D)")
    ax.set_title("Structural Control: True KG vs. Shuffled KG vs. Random Init")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    out_path = FIGS_DIR / "fig2_shuffle_control.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Generated {out_path}")


if __name__ == "__main__":
    make_fig_loco()
    make_fig_incohort()
    make_fig_shuffle()
