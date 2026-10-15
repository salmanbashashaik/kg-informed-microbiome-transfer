"""
model.py — MicrobiomeTransformer architecture with Knowledge Graph embedding integration.
"""

import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from microbiome_kg.transformer.config import (
    PAD_ID, MASK_ID, CLS_ID, UNK_ID, VOCAB_OFFSET, ModelConfig
)

# Scale normalizer for R-GCN vectors to match learnable initialization:
# 0.02 * sqrt(256) = 0.32
_LEARNABLE_INIT_NORM = 0.32


class GenusEmbedding(nn.Module):
    """
    Embedding module bridging Knowledge Graph entity embeddings with the Transformer vocabulary.
    - Matched genera (graph_init=True): frozen R-GCN vector (scale-normalized) -> linear projection
    - Special tokens and unmatched genera: learnable nn.Embedding
    - Random initialization baseline: all tokens learnable with identity projection kept for parameter parity.
    """

    def __init__(
        self,
        vocab_size: int = 1041,
        d_model: int = 256,
        rgcn_dim: int = 256,
        rgcn_emb_file: Optional[str] = None,
        bridge_file: Optional[str] = None,
        graph_init: bool = True,
        embed_transform: str = "none",
        transform_seed: int = 0,
        freeze_proj: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.rgcn_dim = rgcn_dim
        self.graph_init = graph_init
        self.embed_transform = embed_transform
        self.transform_seed = transform_seed
        self.freeze_proj = freeze_proj

        # Learnable fallback embedding table
        self.learnable = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        nn.init.normal_(self.learnable.weight, std=0.02)
        with torch.no_grad():
            self.learnable.weight[PAD_ID].zero_()

        # Geometry channel: linear projection from R-GCN space into model space
        self.proj = nn.Linear(rgcn_dim, d_model)
        with torch.no_grad():
            eye_size = min(d_model, rgcn_dim)
            self.proj.weight.zero_()
            self.proj.weight[:eye_size, :eye_size] = torch.eye(eye_size)
            self.proj.bias.zero_()

        if self.freeze_proj:
            self.proj.weight.requires_grad_(False)
            self.proj.bias.requires_grad_(False)

        # Buffer to store frozen R-GCN representations
        self.register_buffer("graph_frozen", torch.zeros(vocab_size, rgcn_dim))
        self.register_buffer("is_bridged", torch.zeros(vocab_size, dtype=torch.bool))

        if graph_init and rgcn_emb_file and bridge_file:
            self._load_graph_prior(rgcn_emb_file, bridge_file)

    def _load_graph_prior(self, rgcn_emb_file: str, bridge_file: str):
        p_emb = Path(rgcn_emb_file)
        p_bridge = Path(bridge_file)
        if not p_emb.exists() or not p_bridge.exists():
            return

        with open(p_bridge, "r") as f:
            bridge_dict = json.load(f)
        bridge = {int(k): int(v) for k, v in bridge_dict.items()}

        emb_matrix = np.load(p_emb)

        # Optional geometry transform control (rotation or reflection isometry)
        if self.embed_transform in ("rotation", "reflection"):
            rng = np.random.RandomState(self.transform_seed)
            Q, _ = np.linalg.qr(rng.randn(self.rgcn_dim, self.rgcn_dim))
            det = np.linalg.det(Q)
            if self.embed_transform == "rotation" and det < 0:
                Q[:, 0] = -Q[:, 0]
            elif self.embed_transform == "reflection" and det > 0:
                Q[:, 0] = -Q[:, 0]
            emb_matrix = emb_matrix @ Q

        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normed_emb = (emb_matrix / norms) * _LEARNABLE_INIT_NORM

        for tok_id, rgcn_row in bridge.items():
            if tok_id < self.vocab_size and rgcn_row < len(normed_emb):
                self.graph_frozen[tok_id] = torch.from_numpy(normed_emb[rgcn_row]).float()
                self.is_bridged[tok_id] = True

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if not self.graph_init:
            return self.learnable(token_ids)

        bridged_mask = self.is_bridged[token_ids]
        out = torch.zeros(
            *token_ids.shape, self.d_model,
            device=token_ids.device, dtype=torch.float32
        )

        if bridged_mask.any():
            frozen_vecs = self.graph_frozen[token_ids[bridged_mask]]
            projected = self.proj(frozen_vecs)
            out[bridged_mask] = projected

        unbridged_mask = ~bridged_mask
        if unbridged_mask.any():
            out[unbridged_mask] = self.learnable(token_ids[unbridged_mask])

        return out


class MicrobiomeTransformer(nn.Module):
    """
    Biologically informed orderless sequence transformer for microbiome profiling.
    """

    def __init__(
        self,
        config: ModelConfig,
        rgcn_emb_file: Optional[str] = None,
        bridge_file: Optional[str] = None,
        graph_init: bool = True,
        embed_transform: str = "none",
        transform_seed: int = 0,
        freeze_proj: bool = False,
    ):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        # Token embedding (KG-bridged or random)
        self.genus_emb = GenusEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            rgcn_dim=config.rgcn_dim,
            rgcn_emb_file=rgcn_emb_file,
            bridge_file=bridge_file,
            graph_init=graph_init,
            embed_transform=embed_transform,
            transform_seed=transform_seed,
            freeze_proj=freeze_proj,
        )

        # Abundance bin embedding
        self.bin_emb = nn.Embedding(config.n_bins, config.d_model, padding_idx=0)
        nn.init.normal_(self.bin_emb.weight, std=0.02)
        with torch.no_grad():
            self.bin_emb.weight[0].zero_()

        # CLS token representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # LayerNorm and Dropout
        self.input_ln = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Pre-LN Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

    def forward(
        self,
        genus_ids: torch.Tensor,
        bin_ids: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        genus_ids: (B, L)
        bin_ids:   (B, L)
        pad_mask:  (B, L) True where token is padding
        Returns:
          cls_repr: (B, d_model)
          seq_repr: (B, L + 1, d_model)
        """
        B = genus_ids.size(0)

        x_g = self.genus_emb(genus_ids)
        x_b = self.bin_emb(bin_ids)
        x = x_g + x_b

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, L + 1, d_model)

        if pad_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)
            key_padding_mask = torch.cat([cls_mask, pad_mask], dim=1)
        else:
            key_padding_mask = None

        x = self.dropout(self.input_ln(x))
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)

        cls_repr = h[:, 0, :]
        return cls_repr, h


class FactoredMMPHead(nn.Module):
    """
    Factored Masked Microbiome Prediction Head:
    - Presence head: 2-way classification (absent vs present) on all masked tokens.
    - Bin head: (n_bins - 1)-way classification over non-zero bins on present masked tokens.
    """

    def __init__(self, d_model: int, n_bins: int = 5):
        super().__init__()
        self.presence_head = nn.Linear(d_model, 2)
        self.bin_head = nn.Linear(d_model, n_bins - 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pres_logits = self.presence_head(h)
        bin_logits = self.bin_head(h)
        return pres_logits, bin_logits


class ClassificationHead(nn.Module):
    """Linear classification head for frozen probe and LOCO transfer."""

    def __init__(self, d_model: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, cls_repr: torch.Tensor) -> torch.Tensor:
        return self.linear(cls_repr)
