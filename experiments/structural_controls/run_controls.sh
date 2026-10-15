#!/usr/bin/env bash
# Runs structural graph controls (co-occurrence graph and degree-preserved shuffled KG)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "Evaluating Co-Occurrence Graph Control..."
python scripts/train_rgcn.py \
    --edges-file data/kg/edges_v06_cooccurrence.tsv \
    --nodes-file data/kg/nodes_v06.tsv \
    --output-dir checkpoints/rgcn_cooccurrence

echo "Evaluating Shuffled-KG Edge Control..."
python scripts/train_rgcn.py \
    --edges-file data/kg/edges_v06_shuffled_seed1.tsv \
    --nodes-file data/kg/nodes_v06.tsv \
    --output-dir checkpoints/rgcn_shuffled
