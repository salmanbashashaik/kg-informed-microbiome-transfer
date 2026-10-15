#!/usr/bin/env bash
# Runs in-cohort linear probing across subsampled label fractions (1%, 2%, 5%, 25%, 50%, 100%)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python scripts/evaluate_incohort.py \
    --config configs/evaluation/incohort_probing.yaml \
    --output results/canonical/incohort_fractions_results.json
