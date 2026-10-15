"""
model.py — 2-layer Relational Graph Convolutional Network (R-GCN) encoder with ComplEx decoder.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


class RGCNLayer(nn.Module):
    """
    R-GCN message-passing layer with basis decomposition, residual connection, and LayerNorm.
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int, num_bases: int = 8, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases

        # Shared basis matrices + per-relation mixing coefficients
        self.basis = nn.Parameter(torch.empty(num_bases, in_dim, out_dim))
        self.coeff = nn.Parameter(torch.empty(num_relations, num_bases))

        # Independent self-loop transformation
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)

        # Residual projection
        self.residual_proj = (
            nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        )

        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(p=dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.basis.view(self.num_bases, -1)).reshape_as(self.basis)
        nn.init.xavier_uniform_(self.coeff)
        if self.in_dim == self.out_dim:
            nn.init.eye_(self.self_loop.weight)
        else:
            nn.init.xavier_uniform_(self.self_loop.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        src = edge_index[0]
        dst = edge_index[1]

        # Per-relation weight matrices: (R, in_dim, out_dim)
        W = torch.einsum("rb,bio->rio", self.coeff, self.basis)

        deg = degree(dst, num_nodes=N, dtype=x.dtype).clamp(min=1.0)
        agg = torch.zeros(N, self.out_dim, device=x.device, dtype=x.dtype)

        for r in range(self.num_relations):
            mask = (edge_type == r).nonzero(as_tuple=True)[0]
            if mask.numel() == 0:
                continue
            s_r = src[mask]
            d_r = dst[mask]
            msg = x[s_r] @ W[r]
            norm = deg[d_r].unsqueeze(-1)
            agg.scatter_add_(0, d_r.unsqueeze(-1).expand_as(msg), msg / norm)

        out = self.ln(agg + self.self_loop(x) + self.residual_proj(x))
        return self.dropout(F.relu(out))


class RGCN(nn.Module):
    """
    Two-layer R-GCN encoder producing 256-dimensional entity representations.
    """

    def __init__(self, num_nodes: int, hidden_dim: int = 256, out_dim: int = 256, num_relations: int = 23, num_bases: int = 8):
        super().__init__()
        assert out_dim % 2 == 0, "out_dim must be even for ComplEx real/imag splitting"
        self.num_nodes = num_nodes
        self.entity_emb = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)

        self.conv1 = RGCNLayer(hidden_dim, hidden_dim, num_relations, num_bases=num_bases)
        self.conv2 = RGCNLayer(hidden_dim, out_dim, num_relations, num_bases=num_bases)

    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        x = self.entity_emb.weight
        h = self.conv1(x, edge_index, edge_type)
        h = self.conv2(h, edge_index, edge_type)
        return h


class ComplExDecoder(nn.Module):
    """
    ComplEx link-prediction scoring function for asymmetric directed relations.
    Score = Re(<h, r, conj(t)>)
    """

    def __init__(self, num_relations: int, emb_dim: int = 256):
        super().__init__()
        self.num_relations = num_relations
        self.emb_dim = emb_dim
        self.half_dim = emb_dim // 2

        self.rel_real = nn.Parameter(torch.empty(num_relations, self.half_dim))
        self.rel_imag = nn.Parameter(torch.empty(num_relations, self.half_dim))
        nn.init.xavier_uniform_(self.rel_real)
        nn.init.xavier_uniform_(self.rel_imag)

    def forward(self, head: torch.Tensor, rel_idx: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        h_re, h_im = head[:, :self.half_dim], head[:, self.half_dim:]
        t_re, t_im = tail[:, :self.half_dim], tail[:, self.half_dim:]
        r_re = self.rel_real[rel_idx]
        r_im = self.rel_imag[rel_idx]

        score = (
            (h_re * r_re - h_im * r_im) * t_re +
            (h_re * r_im + h_im * r_re) * t_im
        ).sum(dim=-1)
        return score
