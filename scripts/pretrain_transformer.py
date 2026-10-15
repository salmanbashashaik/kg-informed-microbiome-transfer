#!/usr/bin/env python3
"""
pretrain_transformer.py — Pretrains MicrobiomeTransformer via Factored Masked Microbiome Prediction (MMP).
Supports full GPU pretraining or fast CPU smoke testing.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from microbiome_kg.transformer.config import PretrainConfig, ModelConfig
from microbiome_kg.transformer.model import MicrobiomeTransformer, FactoredMMPHead
from microbiome_kg.data.dataset import MicrobiomeDataset, collate_microbiome_batch


def main():
    parser = argparse.ArgumentParser(description="Pretrain Microbiome Transformer")
    parser.add_argument("--data_file", type=str, default="data/examples/sample_pretrain.tsv", help="Pretraining dataset TSV")
    parser.add_argument("--vocab_file", type=str, default="data/vocab/genus_vocab.txt")
    parser.add_argument("--bridge_file", type=str, default="data/vocab/bridge.json")
    parser.add_argument("--rgcn_emb_file", type=str, default="data/kg/entity_embeddings.npy")
    parser.add_argument("--rgcn_id_file", type=str, default="data/kg/node_id_to_row.json")
    parser.add_argument("--init", type=str, choices=["graph", "random"], default="graph")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="checkpoints/pretrain_test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1 quick epoch on small batch")

    args = parser.parse_args()

    print(f"Loading pretraining data from {args.data_file}...")
    df = pd.read_csv(args.data_file, sep="\t", index_col=0)
    print(f"Loaded {len(df)} samples, {len(df.columns)} taxa.")

    dataset = MicrobiomeDataset(df, vocab_file=args.vocab_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_microbiome_batch)

    cfg = ModelConfig()
    graph_init = (args.init == "graph")

    model = MicrobiomeTransformer(
        cfg,
        rgcn_emb_file=args.rgcn_emb_file if graph_init else None,
        bridge_file=args.bridge_file if graph_init else None,
        graph_init=graph_init,
    ).to(args.device)

    head = FactoredMMPHead(d_model=cfg.d_model, n_bins=cfg.n_bins).to(args.device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr,
        weight_decay=1e-2,
    )
    criterion_pres = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_bin = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"Starting pretraining ({args.init}-init) for {args.epochs} epoch(s) on {args.device}...")
    model.train()
    head.train()

    for ep in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            g = batch["genus_ids"].to(args.device)
            b = batch["bin_ids"].to(args.device)
            m = batch["pad_mask"].to(args.device)
            p_targets = batch["presence_targets"].to(args.device)
            b_targets = batch["bin_targets"].to(args.device)

            optimizer.zero_grad()
            _, h = model(g, b, pad_mask=m)
            # Remove CLS token representation from token predictions
            token_h = h[:, 1:, :]

            pres_logits, bin_logits = head(token_h)

            loss_pres = criterion_pres(pres_logits.reshape(-1, 2), p_targets.reshape(-1))
            loss_bin = criterion_bin(bin_logits.reshape(-1, cfg.n_bins - 1), b_targets.reshape(-1))
            loss = 0.3 * loss_pres + loss_bin

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            if args.smoke_test and n_batches >= 2:
                break

        avg_loss = total_loss / max(1, n_batches)
        print(f"Epoch {ep + 1}/{args.epochs}: Average MMP Loss = {avg_loss:.4f}")

    out_p = Path(args.out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_p / f"{args.init}_init_model.pt"
    torch.save({
        "epoch": args.epochs,
        "model": model.state_dict(),
        "head": head.state_dict(),
        "vocab_size": cfg.vocab_size,
        "graph_init": graph_init,
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
