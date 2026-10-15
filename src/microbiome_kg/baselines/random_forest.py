"""
random_forest.py — Classical tree-based baselines for cross-cohort transfer and in-cohort classification.
Includes:
- Intersection Random Forest (train & test on shared genus intersection)
- Zero-Fill Random Forest (train on source genus set, test-time missing features zero-filled)
- Taxonomy-collapsed Random Forest (features aggregated to family rank)
"""

from typing import Dict, Any, List, Tuple, Set, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from microbiome_kg.evaluation.metrics import compute_binary_metrics


def get_feature_cols(df: pd.DataFrame, label_col: str = "label") -> List[str]:
    metadata_cols = {label_col.lower(), "sample_id", "sample", "subject_id", "upf_proxy_score", "index", "id"}
    cols = []
    for c in df.columns:
        c_str = str(c).strip()
        if not c_str or c_str.lower() in metadata_cols or c_str.lower().startswith("unnamed"):
            continue
        cols.append(c)
    return cols


def run_intersection_rf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "label",
    n_seeds: int = 20,
    n_estimators: int = 500,
) -> Dict[str, Any]:
    """
    Evaluates Random Forest strictly on the shared genus intersection between train and test cohorts.
    """
    train_features = get_feature_cols(train_df, label_col)
    test_features = get_feature_cols(test_df, label_col)
    shared = sorted(set(train_features) & set(test_features))

    if not shared:
        raise ValueError("No shared features found between training and testing cohorts.")

    X_train = train_df[shared].values.astype(np.float64)
    y_train = train_df[label_col].values.astype(int)
    X_test = test_df[shared].values.astype(np.float64)
    y_test = test_df[label_col].values.astype(int)

    aurocs, auprcs = [], []
    for s in range(n_seeds):
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=s, n_jobs=-1)
        rf.fit(X_train, y_train)
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


def run_zerofill_rf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "label",
    n_seeds: int = 20,
    n_estimators: int = 500,
) -> Dict[str, Any]:
    """
    Trains RF on full source feature space; zero-fills unmeasured features at test time.
    """
    train_features = get_feature_cols(train_df, label_col)

    X_train = train_df[train_features].values.astype(np.float64)
    y_train = train_df[label_col].values.astype(int)

    # Align test dataframe to train_features, zero-filling missing ones
    aligned_test = pd.DataFrame(0.0, index=test_df.index, columns=train_features)
    common = [c for c in train_features if c in test_df.columns]
    aligned_test[common] = test_df[common].values.astype(np.float64)

    X_test = aligned_test.values
    y_test = test_df[label_col].values.astype(int)

    aurocs, auprcs = [], []
    for s in range(n_seeds):
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=s, n_jobs=-1)
        rf.fit(X_train, y_train)
        probs = rf.predict_proba(X_test)[:, 1]
        m = compute_binary_metrics(y_test, probs)
        aurocs.append(m["auroc"])
        auprcs.append(m["auprc"])

    return {
        "train_features_count": len(train_features),
        "common_features_count": len(common),
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
        "per_seed_auroc": aurocs,
    }


def run_taxonomy_rf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    genus_to_family: Dict[str, str],
    label_col: str = "label",
    n_seeds: int = 20,
    n_estimators: int = 500,
) -> Dict[str, Any]:
    """
    Collapses genus abundances to family rank and evaluates RF transfer on shared families.
    """
    def collapse_to_family(df: pd.DataFrame) -> pd.DataFrame:
        features = get_feature_cols(df, label_col)

        family_cols = {}
        for feat in features:
            fam = genus_to_family.get(feat, "Unknown_Family")
            if fam not in family_cols:
                family_cols[fam] = df[feat].copy()
            else:
                family_cols[fam] += df[feat]
        res = pd.DataFrame(family_cols, index=df.index)
        res[label_col] = df[label_col]
        return res

    train_fam = collapse_to_family(train_df)
    test_fam = collapse_to_family(test_df)

    return run_intersection_rf(train_fam, test_fam, label_col=label_col, n_seeds=n_seeds, n_estimators=n_estimators)
