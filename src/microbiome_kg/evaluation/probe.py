"""
probe.py — Frozen linear probe evaluation protocol across subsampled label fractions.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from microbiome_kg.data.dataset import TaskDataset, collate_task_batch
from microbiome_kg.transformer.model import MicrobiomeTransformer, ClassificationHead
from microbiome_kg.evaluation.metrics import compute_binary_metrics, compute_paired_stats


def extract_cls_features(
    model: MicrobiomeTransformer,
    dataset: TaskDataset,
    batch_size: int = 64,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts frozen CLS embeddings and labels from a task dataset."""
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_task_batch)

    cls_list = []
    labels_list = []

    with torch.no_grad():
        for batch in loader:
            g = batch["genus_ids"].to(device)
            b = batch["bin_ids"].to(device)
            m = batch["pad_mask"].to(device)
            cls_repr, _ = model(g, b, pad_mask=m)
            cls_list.append(cls_repr.cpu().numpy())
            labels_list.append(batch["label"].numpy())

    return np.concatenate(cls_list, axis=0), np.concatenate(labels_list, axis=0)


def train_linear_head(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    d_model: int = 256,
    num_classes: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 50,
    seed: int = 42,
    device: str = "cpu"
) -> Tuple[ClassificationHead, Dict[str, float]]:
    """Trains a logistic/linear classification head on frozen features."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    head = ClassificationHead(d_model=d_model, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    tx = torch.tensor(train_x, dtype=torch.float32, device=device)
    ty = torch.tensor(train_y, dtype=torch.long, device=device)
    vx = torch.tensor(val_x, dtype=torch.float32, device=device)

    head.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        out = head(tx)
        loss = criterion(out, ty)
        loss.backward()
        optimizer.step()

    head.eval()
    with torch.no_grad():
        val_logits = head(vx)
        probs = torch.softmax(val_logits, dim=-1)[:, 1].cpu().numpy()

    metrics = compute_binary_metrics(val_y, probs)
    return head, metrics
