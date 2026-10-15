#!/usr/bin/env bash
# Runs derangement shuffle controls (5 random seeds)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python scripts/evaluate_shuffle.py \
    --seeds 1 2 3 4 5 \
    --output results/canonical/shuffle_control_results.json
