"""
data.py — Graph loading and relation-frequency capping for R-GCN training.
"""

import os
from typing import Dict, Any, Tuple
import pandas as pd
import torch

# Canonical relation caps to prevent training dominance by huge relation types
DEFAULT_CAPS: Dict[str, int] = {
    "encodes": 5000,
    "associates_with": 5000,
    "interacts_with": 5000,
}


def load_graph(
    data_dir: str,
    node_file: str = "nodes_v06.tsv",
    edge_file: str = "edges_v06.tsv",
    cap_relations: bool = True,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Loads nodes and edges TSVs into PyTorch Geometric tensors.
    Applies frequency capping to dominant relations.
    """
    nodes_path = os.path.join(data_dir, node_file)
    edges_path = os.path.join(data_dir, edge_file)

    nodes_df = pd.read_csv(nodes_path, sep="\t")
    edges_df = pd.read_csv(edges_path, sep="\t")

    node_id_to_idx = {nid: i for i, nid in enumerate(nodes_df["id"])}
    idx_to_node_id = {i: nid for nid, i in node_id_to_idx.items()}
    node_types = nodes_df["type"].tolist()
    node_type_list = sorted(set(node_types))
    node_type_to_idx = {t: i for i, t in enumerate(node_type_list)}

    relation_list = sorted(edges_df["relation"].unique().tolist())
    relation_to_idx = {r: i for i, r in enumerate(relation_list)}
    idx_to_relation = {i: r for r, i in relation_to_idx.items()}

    # Drop dangling edges if any
    valid = edges_df["head"].isin(node_id_to_idx) & edges_df["tail"].isin(node_id_to_idx)
    edges_df = edges_df[valid].reset_index(drop=True)

    head_idx = [node_id_to_idx[h] for h in edges_df["head"]]
    tail_idx = [node_id_to_idx[t] for t in edges_df["tail"]]
    rel_idx = [relation_to_idx[r] for r in edges_df["relation"]]

    edge_index = torch.tensor([head_idx, tail_idx], dtype=torch.long)
    edge_type = torch.tensor(rel_idx, dtype=torch.long)

    # Apply relation capping
    if cap_relations:
        torch.manual_seed(seed)
        for rel_name, cap in DEFAULT_CAPS.items():
            if rel_name in relation_to_idx:
                r_idx = relation_to_idx[rel_name]
                r_mask = (edge_type == r_idx).nonzero(as_tuple=True)[0]
                if r_mask.size(0) > cap:
                    perm = torch.randperm(r_mask.size(0))[:cap]
                    keep = r_mask[perm]
                    other = (edge_type != r_idx).nonzero(as_tuple=True)[0]
                    keep_all = torch.cat([other, keep])
                    edge_index = edge_index[:, keep_all]
                    edge_type = edge_type[keep_all]

    metadata = {
        "num_nodes": len(nodes_df),
        "num_edges": edge_index.shape[1],
        "node_id_to_idx": node_id_to_idx,
        "idx_to_node_id": idx_to_node_id,
        "relation_list": relation_list,
        "relation_to_idx": relation_to_idx,
        "idx_to_relation": idx_to_relation,
        "node_type_list": node_type_list,
        "node_type_to_idx": node_type_to_idx,
    }

    return edge_index, edge_type, metadata
