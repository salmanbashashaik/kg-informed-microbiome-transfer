"""Evaluation protocols and statistical metrics."""

from microbiome_kg.evaluation.metrics import compute_binary_metrics, compute_paired_stats
from microbiome_kg.evaluation.probe import extract_cls_features, train_linear_head
from microbiome_kg.evaluation.loco import evaluate_loco_pair
