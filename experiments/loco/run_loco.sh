#!/usr/bin/env bash
# Runs Leave-One-Cohort-Out (LOCO) cross-study transfer evaluation over 8 directed pairs
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python scripts/evaluate_loco.py \
    --config configs/evaluation/loco_transfer.yaml \
    --output results/canonical/loco_transfer_results.json
