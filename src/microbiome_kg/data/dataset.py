"""
dataset.py — Data loading, taxonomic normalization, and abundance binning for microbiome profiles.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from microbiome_kg.transformer.config import (
    PAD_ID, MASK_ID, CLS_ID, UNK_ID, VOCAB_OFFSET,
    BIN_EDGES, NUM_BINS
)


def load_vocab(vocab_file: str) -> Tuple[Dict[str, int], List[str]]:
    """Loads vocabulary file returning lowercase token-to-ID dict and token list."""
    with open(vocab_file, "r") as f:
        tokens = [line.strip() for line in f if line.strip()]
    tok2id = {t.lower(): i for i, t in enumerate(tokens)}
    return tok2id, tokens


def normalize_genus(col: str) -> str:
    """Extracts normalized genus name from raw taxonomic string."""
    col = col.strip()
    if not col:
        return ""
    if "|g__" in col:
        genus = col.split("|g__")[-1].split("|")[0]
    elif "|" in col:
        genus = col.split("|")[0].strip()
        if genus.startswith("g__"):
            genus = genus[3:]
    elif col.startswith("g__"):
        genus = col[3:]
    elif col[0:1].isupper() and col[0:1].isalpha():
        genus = re.split(r"[_\s]", col)[0]
    else:
        return ""
    return re.sub(r"[^A-Za-z0-9].*$", "", genus).capitalize()


def bin_abundance(x: np.ndarray) -> np.ndarray:
    """
    Converts relative abundance values (0-1 proportion or 0-100 percentage)
    into categorical bins 0-4 based on BIN_EDGES.
    """
    x = np.asarray(x, dtype=float).copy()
    if x.max() > 1.5:  # Auto-convert percentage scale to proportion
        x /= 100.0
    bins = np.zeros(x.shape, dtype=np.int64)
    for i, thresh in enumerate(BIN_EDGES):
        bins[x > thresh] = i + 1
    return bins


def is_already_binned(arr: np.ndarray) -> bool:
    """True if values are already integer bins in {0, 1, 2, 3, 4}."""
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return True
    int_arr = arr.astype(int)
    return bool(np.all(arr == int_arr) and int_arr.max() <= 4 and int_arr.min() >= 0)


def map_df_to_vocab(df: pd.DataFrame, tok2id: Dict[str, int]) -> Dict[int, List[str]]:
    """Maps columns of a cohort DataFrame to vocabulary genus IDs."""
    vocab_cols: Dict[int, List[str]] = defaultdict(list)
    for col in df.columns:
        if col.lower() in ("label", "sample_id", "upf_proxy_score"):
            continue
        norm = normalize_genus(col)
        tok_id = tok2id.get(norm.lower(), None)
        if tok_id is not None and tok_id >= VOCAB_OFFSET:
            vocab_cols[tok_id].append(col)
    return vocab_cols


class MicrobiomeDataset(Dataset):
    """
    Dataset for pretraining on pooled unlabelled microbiome samples.
    Factored sequence construction with stochastic masking.
    """

    def __init__(
        self,
        samples_df: pd.DataFrame,
        vocab_file: str,
        mask_nonzero_p: float = 0.30,
        mask_zero_p: float = 0.05,
        n_zero_per_sample: int = 50,
        seed: int = 42,
    ):
        self.tok2id, self.id2tok = load_vocab(vocab_file)
        self.mask_nonzero_p = mask_nonzero_p
        self.mask_zero_p = mask_zero_p
        self.n_zero_per_sample = n_zero_per_sample
        self.rng = np.random.RandomState(seed)

        self.vocab_cols = map_df_to_vocab(samples_df, self.tok2id)
        self.mapped_vocab_ids = sorted(self.vocab_cols.keys())

        # Precompute per-sample binned abundance vectors for mapped vocabulary
        n_samples = len(samples_df)
        self.binned_matrix = np.zeros((n_samples, len(self.mapped_vocab_ids)), dtype=np.int64)

        for col_idx, vid in enumerate(self.mapped_vocab_ids):
            cols = self.vocab_cols[vid]
            vals = samples_df[cols].values
            if vals.shape[1] > 1:
                vals = vals.sum(axis=1)
            else:
                vals = vals[:, 0]
            if is_already_binned(vals):
                self.binned_matrix[:, col_idx] = vals.astype(np.int64)
            else:
                self.binned_matrix[:, col_idx] = bin_abundance(vals)

    def __len__(self) -> int:
        return len(self.binned_matrix)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.binned_matrix[idx]
        pos_indices = np.where(row > 0)[0]
        zero_indices = np.where(row == 0)[0]

        if len(zero_indices) > self.n_zero_per_sample:
            sampled_zeros = self.rng.choice(zero_indices, size=self.n_zero_per_sample, replace=False)
        else:
            sampled_zeros = zero_indices

        selected_indices = np.concatenate([pos_indices, sampled_zeros])
        # Sort non-zeros by abundance descending, followed by zeros
        selected_indices = sorted(selected_indices, key=lambda i: (-row[i], i))

        genus_ids = [self.mapped_vocab_ids[i] for i in selected_indices]
        bin_ids = [row[i] for i in selected_indices]

        # Masking
        input_genus = list(genus_ids)
        input_bin = list(bin_ids)
        presence_targets = [-100] * len(genus_ids)
        bin_targets = [-100] * len(genus_ids)

        for i, (g, b) in enumerate(zip(genus_ids, bin_ids)):
            p_mask = self.mask_nonzero_p if b > 0 else self.mask_zero_p
            if self.rng.rand() < p_mask:
                input_genus[i] = MASK_ID
                input_bin[i] = 0
                presence_targets[i] = 1 if b > 0 else 0
                if b > 0:
                    bin_targets[i] = b - 1  # 0..3 for non-zero bins 1..4

        return {
            "genus_ids": torch.tensor(input_genus, dtype=torch.long),
            "bin_ids": torch.tensor(input_bin, dtype=torch.long),
            "presence_targets": torch.tensor(presence_targets, dtype=torch.long),
            "bin_targets": torch.tensor(bin_targets, dtype=torch.long),
        }


class TaskDataset(Dataset):
    """
    Dataset for downstream supervised classification (in-cohort and LOCO transfer).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vocab_file: str,
        n_zero_per_sample: int = 50,
        seed: int = 42,
    ):
        self.tok2id, self.id2tok = load_vocab(vocab_file)
        self.n_zero_per_sample = n_zero_per_sample
        self.rng = np.random.RandomState(seed)

        self.labels = df["label"].values.astype(np.int64) if "label" in df.columns else np.zeros(len(df), dtype=np.int64)

        self.vocab_cols = map_df_to_vocab(df, self.tok2id)
        self.mapped_vocab_ids = sorted(self.vocab_cols.keys())

        n_samples = len(df)
        self.binned_matrix = np.zeros((n_samples, len(self.mapped_vocab_ids)), dtype=np.int64)

        for col_idx, vid in enumerate(self.mapped_vocab_ids):
            cols = self.vocab_cols[vid]
            vals = df[cols].values
            if vals.shape[1] > 1:
                vals = vals.sum(axis=1)
            else:
                vals = vals[:, 0]
            if is_already_binned(vals):
                self.binned_matrix[:, col_idx] = vals.astype(np.int64)
            else:
                self.binned_matrix[:, col_idx] = bin_abundance(vals)

    def __len__(self) -> int:
        return len(self.binned_matrix)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.binned_matrix[idx]
        pos_indices = np.where(row > 0)[0]
        zero_indices = np.where(row == 0)[0]

        if len(zero_indices) > self.n_zero_per_sample:
            sampled_zeros = self.rng.choice(zero_indices, size=self.n_zero_per_sample, replace=False)
        else:
            sampled_zeros = zero_indices

        selected_indices = np.concatenate([pos_indices, sampled_zeros])
        selected_indices = sorted(selected_indices, key=lambda i: (-row[i], i))

        genus_ids = [self.mapped_vocab_ids[i] for i in selected_indices]
        bin_ids = [row[i] for i in selected_indices]

        return {
            "genus_ids": torch.tensor(genus_ids, dtype=torch.long),
            "bin_ids": torch.tensor(bin_ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_microbiome_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collates variable-length pretraining sequences with padding."""
    max_len = max(item["genus_ids"].size(0) for item in batch)
    B = len(batch)

    genus_ids = torch.full((B, max_len), PAD_ID, dtype=torch.long)
    bin_ids = torch.zeros((B, max_len), dtype=torch.long)
    pad_mask = torch.ones((B, max_len), dtype=torch.bool)
    presence_targets = torch.full((B, max_len), -100, dtype=torch.long)
    bin_targets = torch.full((B, max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        L = item["genus_ids"].size(0)
        genus_ids[i, :L] = item["genus_ids"]
        bin_ids[i, :L] = item["bin_ids"]
        pad_mask[i, :L] = False
        presence_targets[i, :L] = item["presence_targets"]
        bin_targets[i, :L] = item["bin_targets"]

    return {
        "genus_ids": genus_ids,
        "bin_ids": bin_ids,
        "pad_mask": pad_mask,
        "presence_targets": presence_targets,
        "bin_targets": bin_targets,
    }


def collate_task_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collates variable-length task classification sequences with padding."""
    max_len = max(item["genus_ids"].size(0) for item in batch)
    B = len(batch)

    genus_ids = torch.full((B, max_len), PAD_ID, dtype=torch.long)
    bin_ids = torch.zeros((B, max_len), dtype=torch.long)
    pad_mask = torch.ones((B, max_len), dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        L = item["genus_ids"].size(0)
        genus_ids[i, :L] = item["genus_ids"]
        bin_ids[i, :L] = item["bin_ids"]
        pad_mask[i, :L] = False
        labels[i] = item["label"]

    return {
        "genus_ids": genus_ids,
        "bin_ids": bin_ids,
        "pad_mask": pad_mask,
        "label": labels,
    }
