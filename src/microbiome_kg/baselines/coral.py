"""
coral.py — Correlation Alignment (CORAL) domain adaptation baseline (Sun et al., AAAI 2016).
Whitens source feature covariance and recolors with target feature covariance before fitting classifier.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from microbiome_kg.evaluation.metrics import compute_binary_metrics


def coral_transform(X_source: np.ndarray, X_target: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    CORAL alignment: whitens source covariance to identity and recolors with target covariance.
    Uses eigendecomposition for stability when n < d.
    """
    d = X_source.shape[1]
    cov_source = np.cov(X_source, rowvar=False) + eps * np.eye(d)
    cov_target = np.cov(X_target, rowvar=False) + eps * np.eye(d)

    def sqrt_inv_and_fwd(cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, eps, None)
        sqrt_inv = vecs @ np.diag(vals ** -0.5) @ vecs.T
        sqrt_fwd = vecs @ np.diag(vals ** 0.5) @ vecs.T
        return sqrt_inv, sqrt_fwd

    source_sqrt_inv, _ = sqrt_inv_and_fwd(cov_source)
    _, target_sqrt_fwd = sqrt_inv_and_fwd(cov_target)

    X_source_centered = X_source - X_source.mean(axis=0, keepdims=True)
    X_whitened = X_source_centered @ source_sqrt_inv
    X_recolored = X_whitened @ target_sqrt_fwd
    return X_recolored + X_target.mean(axis=0, keepdims=True)


def run_coral_rf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "label",
    n_seeds: int = 20,
    n_estimators: int = 500,
) -> Dict[str, Any]:
    """
    Runs CORAL domain adaptation Random Forest on the shared genus intersection.
    """
    train_features = [c for c in train_df.columns if c != label_col]
    test_features = [c for c in test_df.columns if c != label_col]
    shared = sorted(set(train_features) & set(test_features))

    if not shared:
        raise ValueError("No shared features found between train and test cohorts.")

    X_train = train_df[shared].values.astype(np.float64)
    y_train = train_df[label_col].values.astype(int)
    X_test = test_df[shared].values.astype(np.float64)
    y_test = test_df[label_col].values.astype(int)

    # Apply CORAL alignment to train features using test covariance
    X_train_coral = coral_transform(X_train, X_test)

    aurocs, auprcs = [], []
    for s in range(n_seeds):
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=s, n_jobs=-1)
        rf.fit(X_train_coral, y_train)
        probs = rf.predict_proba(X_test)[:, 1]
        m = compute_binary_metrics(y_test, probs)
        aurocs.append(m["auroc"])
        auprcs.append(m["auprc"])

    return {
        "shared_features_count": len(shared),
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
        "per_seed_auroc": aurocs,
    }
