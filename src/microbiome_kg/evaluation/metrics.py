"""
metrics.py — Statistical evaluation metrics: AUROC, AUPRC, Wilcoxon signed-rank, Cohen's d, bootstrap CIs.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from scipy.stats import wilcoxon


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """
    Computes AUROC, AUPRC, sensitivity, specificity, and F1 at optimal threshold.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if len(np.unique(y_true)) < 2:
        return {
            "auroc": 0.5,
            "auprc": 0.0,
            "sensitivity": 0.0,
            "specificity": 1.0,
            "f1": 0.0,
        }

    auroc = float(roc_auc_score(y_true, y_score))
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    auprc = float(auc(recall, precision))

    # Optimal F1 threshold search
    f1_scores = []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        f1_scores.append(f1_score(y_true, pred, zero_division=0))
    best_idx = np.argmax(f1_scores) if len(f1_scores) > 0 else 0
    best_thresh = thresholds[best_idx] if len(thresholds) > 0 else 0.5

    pred = (y_score >= best_thresh).astype(int)
    tp = np.sum((pred == 1) & (y_true == 1))
    tn = np.sum((pred == 0) & (y_true == 0))
    fp = np.sum((pred == 1) & (y_true == 0))
    fn = np.sum((pred == 0) & (y_true == 1))

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    best_f1 = float(f1_scores[best_idx]) if len(f1_scores) > 0 else 0.0

    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": best_f1,
    }


def compute_paired_stats(scores_a: List[float], scores_b: List[float]) -> Dict[str, Any]:
    """
    Computes mean, std, 95% bootstrap CI, delta, paired Wilcoxon p-value, and Cohen's d.
    scores_a: typically Graph-init
    scores_b: typically Random-init or Baseline
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    delta = a - b

    mean_a = float(np.mean(a))
    std_a = float(np.std(a))
    mean_b = float(np.mean(b))
    std_b = float(np.std(b))
    delta_mean = float(np.mean(delta))
    delta_std = float(np.std(delta))

    # Bootstrap 95% CI on delta
    rng = np.random.RandomState(42)
    boot_means = []
    n = len(delta)
    for _ in range(2000):
        idx = rng.randint(0, n, size=n)
        boot_means.append(np.mean(delta[idx]))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # Paired Wilcoxon signed-rank test
    try:
        w_stat, p_val = wilcoxon(a, b)
        p_val = float(p_val)
    except Exception:
        p_val = 1.0

    # Cohen's d for paired samples
    cohens_d = float(delta_mean / (delta_std + 1e-8))

    return {
        "a_mean": mean_a,
        "a_std": std_a,
        "b_mean": mean_b,
        "b_std": std_b,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "delta_ci95": [ci_lower, ci_upper],
        "wilcoxon_p": p_val,
        "cohens_d": cohens_d,
        "wins": int(np.sum(delta > 0)),
        "n": len(a),
    }
