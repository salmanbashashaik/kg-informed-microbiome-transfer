"""
config.py — Configuration parameters and constants for the Microbiome Transformer.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Special token IDs (matching vocabulary order)
PAD_ID: int = 0
MASK_ID: int = 1
CLS_ID: int = 2
UNK_ID: int = 3
VOCAB_OFFSET: int = 4

# Relative abundance binning edges (proportions 0.0 to 1.0)
# bin 0 = absent (== 0)
# bin 1 = trace  (0, 1e-4]
# bin 2 = low    (1e-4, 1e-3]
# bin 3 = medium (1e-3, 1e-2]
# bin 4 = high   (> 1e-2)
BIN_EDGES: List[float] = [0.0, 1e-4, 1e-3, 1e-2]
NUM_BINS: int = 5


@dataclass
class ModelConfig:
    """Microbiome Transformer architectural hyperparameters."""
    vocab_size: int = 1041
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    n_bins: int = NUM_BINS
    rgcn_dim: int = 256
    abundance_emb: str = "bin_emb"  # "bin_emb" (canonical)


@dataclass
class PretrainConfig:
    """Pretraining hyperparameters and paths."""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    n_bins: int = NUM_BINS
    rgcn_dim: int = 256
    abundance_emb: str = "bin_emb"

    # Training schedule
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    grad_clip: float = 1.0

    # Factored MMP masking
    mask_nonzero_p: float = 0.30
    mask_zero_p: float = 0.05
    n_zero_per_sample: int = 50

    # Factored loss parameters
    presence_weight: float = 0.3
    focal_gamma: float = 2.0
    bin_freq: List[float] = field(default_factory=lambda: [0.286, 0.286, 0.214, 0.214])

    # Initialization & Prior preservation
    graph_init: bool = True
    embed_transform: str = "none"  # "none", "rotation", "reflection"
    transform_seed: int = 0
    preservation_mode: str = "frozen"  # "frozen", "l2sp", "free"
    l2sp_lambda: float = 0.1
    freeze_proj: bool = False

    # Evaluation & checkpointing
    eval_every: int = 5
    save_every: int = 10
    seed: int = 42


@dataclass
class FinetuneConfig:
    """Downstream linear-probe in-cohort classification parameters."""
    d_model: int = 256
    rgcn_dim: int = 256
    abundance_emb: str = "bin_emb"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 32
    fractions: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05, 0.25, 0.50, 1.00])
    n_seeds: int = 20
    seed: int = 42


@dataclass
class LOCOConfig:
    """Leave-one-cohort-out (LOCO) cross-cohort transfer parameters."""
    d_model: int = 256
    rgcn_dim: int = 256
    abundance_emb: str = "bin_emb"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 32
    n_seeds: int = 20
    seed: int = 42
