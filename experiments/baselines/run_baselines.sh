#!/usr/bin/env bash
# Runs classical machine learning baselines (Intersection RF, Zero-Fill RF, Taxonomy RF, CORAL RF)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python scripts/evaluate_baselines.py \
    --config configs/baselines/classical_baselines.yaml \
    --output results/canonical/classical_baselines_results.json
