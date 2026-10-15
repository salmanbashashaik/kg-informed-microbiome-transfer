#!/usr/bin/env python3
"""
generate_tables.py — Generates publication-ready Markdown and LaTeX tables directly from canonical JSON results.
"""

import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DIR = REPO_ROOT / "results" / "canonical"
TABLES_DIR = REPO_ROOT / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

LOCO_PAIRS = [
    ("Qin -> Karlsson", "loco_qin_to_karlsson.json", "qin_to_karlsson"),
    ("Karlsson -> Qin", "loco_karlsson_to_qin.json", "karlsson_to_qin"),
    ("Qin -> MetaCardis", "loco_qin_to_metacardis.json", "qin_to_metacardis"),
    ("MetaCardis -> Qin", "loco_metacardis_to_qin.json", "metacardis_to_qin"),
    ("Karlsson -> MetaCardis", "loco_karlsson_to_metacardis.json", "karlsson_to_metacardis"),
    ("MetaCardis -> Karlsson", "loco_metacardis_to_karlsson.json", "metacardis_to_karlsson"),
    ("ACVD -> MetaCardis-CVD", "loco_acvd_to_metacardis_cvd.json", "acvd_to_metacardis_cvd"),
    ("MetaCardis-CVD -> ACVD", "loco_metacardis_cvd_to_acvd.json", "metacardis_cvd_to_acvd"),
]


def generate_loco_table():
    loco_dir = CANONICAL_DIR / "loco"
    rf_file = loco_dir / "rf_loco_results.json"
    rf_data = json.load(open(rf_file)) if rf_file.exists() else {}

    md_rows = [
        "| Transfer Pair | Disease | Graph AUROC | Random AUROC | Delta | Cohen's d | Wilcoxon p | Wins | RF Baseline | Graph > RF |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    latex_rows = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cross-cohort transfer performance (AUROC) across eight directed same-disease pairs (20 paired seeds).}",
        r"\label{tab:loco_transfer}",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"Source $\to$ Target & Disease & Graph Init & Random Init & $\Delta$ & Cohen's $d$ & $p$-value & Wins & RF Baseline & Beat RF? \\",
        r"\midrule",
    ]

    for label, fname, key in LOCO_PAIRS:
        fpath = loco_dir / fname
        if not fpath.exists():
            continue
        data = json.load(open(fpath))
        stats = data.get("stats", {}).get("auroc", {})
        g = stats.get("graph_mean", 0.0)
        r = stats.get("random_mean", 0.0)
        delta = stats.get("delta_mean", 0.0)
        d = stats.get("cohens_d", 0.0)
        p = stats.get("wilcoxon_p", 1.0)
        wins = data.get("wins", 0)

        disease = "CVD" if "CVD" in label or "ACVD" in label else "T2D"
        rf_val = rf_data.get(key, {}).get("auroc_mean", np.nan)
        beat_rf = "YES" if (not np.isnan(rf_val) and g > rf_val) else "no"

        md_rows.append(
            f"| {label} | {disease} | {g:.4f} | {r:.4f} | {delta:+.4f} | {d:.2f} | {p:.2e} | {wins}/20 | {rf_val:.4f} | {beat_rf} |"
        )
        latex_rows.append(
            f"{label} & {disease} & \\textbf{{{g:.4f}}} & {r:.4f} & {delta:+.4f} & {d:.2f} & {p:.2e} & {wins}/20 & {rf_val:.4f} & {beat_rf} \\\\"
        )

    latex_rows.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    md_content = "\n".join(md_rows)
    latex_content = "\n".join(latex_rows)

    with open(TABLES_DIR / "table2_loco_transfer.md", "w") as f:
        f.write(md_content + "\n")
    with open(TABLES_DIR / "table2_loco_transfer.tex", "w") as f:
        f.write(latex_content + "\n")

    print(f"Generated {TABLES_DIR / 'table2_loco_transfer.md'}")
    print(f"Generated {TABLES_DIR / 'table2_loco_transfer.tex'}")


def generate_incohort_table():
    incohort_dir = CANONICAL_DIR / "incohort"
    cohorts = [
        ("T2D — Qin", "t2d_qin"),
        ("T2D — Karlsson", "t2d_karlsson"),
        ("T2D — MetaCardis", "t2d_metacardis"),
        ("CVD — MetaCardis", "cvd_metacardis"),
        ("ACVD", "acvd"),
    ]
    fractions = ["0.01", "0.02", "0.05", "0.25", "0.50", "1.00"]

    md_rows = [
        "| Cohort | Label Fraction | Graph AUROC | Random AUROC | Delta | Graph AUPRC | Random AUPRC |",
        "|---|---|---|---|---|---|---|",
    ]

    for label, cname in cohorts:
        res_file = incohort_dir / cname / "results.json"
        if not res_file.exists():
            continue
        data = json.load(open(res_file))
        for frac in fractions:
            if frac not in data:
                continue
            s = data[frac].get("stats", {})
            auroc_g = s.get("auroc", {}).get("graph_mean", 0.0)
            auroc_r = s.get("auroc", {}).get("random_mean", 0.0)
            auroc_d = s.get("auroc", {}).get("delta_mean", 0.0)
            auprc_g = s.get("auprc", {}).get("graph_mean", 0.0)
            auprc_r = s.get("auprc", {}).get("random_mean", 0.0)
            frac_pct = f"{int(float(frac)*100)}%"
            md_rows.append(
                f"| {label} | {frac_pct} | {auroc_g:.4f} | {auroc_r:.4f} | {auroc_d:+.4f} | {auprc_g:.4f} | {auprc_r:.4f} |"
            )

    md_content = "\n".join(md_rows)
    with open(TABLES_DIR / "table1_incohort_fractions.md", "w") as f:
        f.write(md_content + "\n")
    print(f"Generated {TABLES_DIR / 'table1_incohort_fractions.md'}")


def generate_baselines_table():
    base_dir = CANONICAL_DIR / "baselines"
    inter_f = base_dir / "intersection_rf_loco_results.json"
    tax_f = base_dir / "taxonomy_rf_loco_results.json"
    zero_f = base_dir / "zerofill_rf_loco_results.json"
    coral_f = base_dir / "coral_rf_loco_results.json"

    inter_d = json.load(open(inter_f)) if inter_f.exists() else {}
    tax_d = json.load(open(tax_f)) if tax_f.exists() else {}
    zero_d = json.load(open(zero_f)) if zero_f.exists() else {}
    coral_d = json.load(open(coral_f)) if coral_f.exists() else {}

    md_rows = [
        "| Transfer Pair | Intersection RF | Taxonomy RF | Zero-Fill RF | CORAL RF |",
        "|---|---|---|---|---|",
    ]

    for label, _, key in LOCO_PAIRS:
        i_val = inter_d.get(key, {}).get("auroc_mean", np.nan)
        t_val = tax_d.get(key, {}).get("auroc_mean", np.nan)
        z_val = zero_d.get(key, {}).get("auroc_mean", np.nan)
        c_val = coral_d.get(key, {}).get("auroc_mean", np.nan)
        md_rows.append(f"| {label} | {i_val:.4f} | {t_val:.4f} | {z_val:.4f} | {c_val:.4f} |")

    with open(TABLES_DIR / "table3_classical_baselines.md", "w") as f:
        f.write("\n".join(md_rows) + "\n")
    print(f"Generated {TABLES_DIR / 'table3_classical_baselines.md'}")


if __name__ == "__main__":
    generate_loco_table()
    generate_incohort_table()
    generate_baselines_table()
