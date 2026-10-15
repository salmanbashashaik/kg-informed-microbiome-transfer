#!/usr/bin/env python3
"""
train_rgcn.py — Trains 2-layer Relational Graph Convolutional Network (R-GCN) via ComplEx link prediction.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from microbiome_kg.graph_models.data import load_graph
from microbiome_kg.graph_models.model import RGCN, ComplExDecoder


def main():
    parser = argparse.ArgumentParser(description="Train R-GCN on biological Knowledge Graph")
    parser.add_argument("--kg_dir", type=str, default="data/kg")
    parser.add_argument("--node_file", type=str, default="nodes_v06.tsv")
    parser.add_argument("--edge_file", type=str, default="edges_v06.tsv")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--out_dir", type=str, default="checkpoints/rgcn_test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1 quick step on CPU")

    args = parser.parse_args()

    print(f"Loading Knowledge Graph from {args.kg_dir}...")
    edge_index, edge_type, meta = load_graph(args.kg_dir, args.node_file, args.edge_file)
    num_nodes = meta["num_nodes"]
    num_relations = len(meta["relation_list"])
    num_edges = edge_index.size(1)

    print(f"Loaded {num_nodes:,} nodes, {num_edges:,} edges, {num_relations} relations.")

    model = RGCN(num_nodes=num_nodes, hidden_dim=256, out_dim=256, num_relations=num_relations).to(args.device)
    decoder = ComplExDecoder(num_relations=num_relations, emb_dim=256).to(args.device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    edge_index_dev = edge_index.to(args.device)
    edge_type_dev = edge_type.to(args.device)

    print(f"Starting R-GCN training on {args.device} for {args.epochs} epoch(s)...")
    model.train()
    decoder.train()

    perm = torch.randperm(num_edges)
    train_edges = edge_index_dev[:, perm]
    train_types = edge_type_dev[perm]

    for ep in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        for i in range(0, min(num_edges, 2048 if args.smoke_test else num_edges), args.batch_size):
            h_idx = train_edges[0, i:i + args.batch_size]
            t_idx = train_edges[1, i:i + args.batch_size]
            r_idx = train_types[i:i + args.batch_size]
            B = h_idx.size(0)

            # Negative sampling (random tail replacement)
            neg_t_idx = torch.randint(0, num_nodes, (B,), device=args.device)

            optimizer.zero_grad()
            emb = model(edge_index_dev, edge_type_dev)

            pos_score = decoder(emb[h_idx], r_idx, emb[t_idx])
            neg_score = decoder(emb[h_idx], r_idx, emb[neg_t_idx])

            # BCE loss with logits
            pos_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
            neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
            loss = pos_loss + neg_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            if args.smoke_test and n_batches >= 2:
                break

        print(f"Epoch {ep + 1}/{args.epochs}: Link Prediction Loss = {total_loss / max(1, n_batches):.4f}")

    out_p = Path(args.out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_p / "best_model.pt"
    torch.save({
        "epoch": args.epochs,
        "model": model.state_dict(),
        "decoder": decoder.state_dict(),
        "num_nodes": num_nodes,
        "num_relations": num_relations,
    }, ckpt_path)
    print(f"Saved R-GCN checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
