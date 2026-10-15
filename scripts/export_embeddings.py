#!/usr/bin/env python3
"""
export_embeddings.py — Extracts 256-dimensional entity embeddings from a trained R-GCN checkpoint.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from microbiome_kg.graph_models.data import load_graph
from microbiome_kg.graph_models.model import RGCN


def main():
    parser = argparse.ArgumentParser(description="Extract R-GCN entity embeddings")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to R-GCN best_model.pt")
    parser.add_argument("--kg_dir", type=str, default="data/kg")
    parser.add_argument("--node_file", type=str, default="nodes_v06.tsv")
    parser.add_argument("--edge_file", type=str, default="edges_v06.tsv")
    parser.add_argument("--out_dir", type=str, default="data/kg")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    print(f"Loading graph from {args.kg_dir}...")
    edge_index, edge_type, meta = load_graph(args.kg_dir, args.node_file, args.edge_file)

    num_nodes = meta["num_nodes"]
    num_relations = len(meta["relation_list"])

    print(f"Loading model checkpoint from {args.checkpoint}...")
    model = RGCN(num_nodes=num_nodes, hidden_dim=256, out_dim=256, num_relations=num_relations)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    model.to(args.device)
    model.eval()

    with torch.no_grad():
        edge_index_dev = edge_index.to(args.device)
        edge_type_dev = edge_type.to(args.device)
        embeddings = model(edge_index_dev, edge_type_dev).cpu().numpy()

    out_p = Path(args.out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    emb_file = out_p / "entity_embeddings.npy"
    np.save(emb_file, embeddings)
    print(f"Saved entity embeddings: {embeddings.shape} to {emb_file}")

    id_file = out_p / "node_id_to_row.json"
    with open(id_file, "w") as f:
        json.dump(meta["node_id_to_idx"], f, indent=2)
    print(f"Saved node-to-row mapping to {id_file}")


if __name__ == "__main__":
    main()
