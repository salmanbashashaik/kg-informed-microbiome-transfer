# Reproducibility Guide

This guide provides step-by-step instructions for reproducing all numerical tables, experimental figures, statistical tests, and computational pipelines from our IEEE BIBM 2026 paper (*Paper ID: B1518*).

---

## 1. Environment Setup

### Option A: Using Conda / Mamba (Recommended)
```bash
# Create and activate environment
conda env create -f environment.yml
conda activate kg-informed-microbiome

# Install package in editable mode
pip install -e .
```

### Option B: Using Pip & Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 2. Verification and Smoke Testing

Verify the complete repository installation, tensor operations, model architectures, and data pipelines on CPU:
```bash
python tests/run_all.py
```
*Expected Output*:
```text
Ran 10 tests in ~15s
OK
```

---

## 3. Paper Results Regeneration

All canonical experimental results across the 20 paired seeds are archived in `results/canonical/`.

### 3.1 Regenerate All Numerical Tables
Generate Table 1 (In-Cohort Probing), Table 2 (LOCO Transfer), and Table 3 (Classical Baselines) in Markdown and LaTeX formats:
```bash
python scripts/generate_tables.py
```
*Generated Files*:
- `results/tables/table1_incohort_fractions.md`
- `results/tables/table2_loco_transfer.md`
- `results/tables/table2_loco_transfer.tex`
- `results/tables/table3_classical_baselines.md`

### 3.2 Regenerate All Paper Figures
Regenerate publication-ready figures (Figure 2: Shuffle Controls, Figure 3: LOCO Transfer Gains, Figure 4: In-Cohort Fractions):
```bash
python scripts/generate_figures.py
```
*Generated Files*:
- `results/figures/fig2_shuffle_control.png`
- `results/figures/fig3_loco_transfer.png`
- `results/figures/fig4_incohort_fractions.png`

---

## 4. Running Experimental Pipelines

### 4.1 Cross-Cohort LOCO Transfer Probing
Run cross-cohort transfer between two cohorts:
```bash
python scripts/evaluate_loco.py \
    --source data/examples/sample_cohort_a.tsv \
    --target data/examples/sample_cohort_b.tsv \
    --vocab data/vocab/genus_vocab.txt \
    --n-seeds 20 \
    --epochs 50 \
    --lr 0.001
```

### 4.2 Classical Baseline Evaluations
Run Random Forest transfer baselines (Intersection RF and Zero-Fill RF):
```bash
python scripts/evaluate_baselines.py \
    --source data/examples/sample_cohort_a.tsv \
    --target data/examples/sample_cohort_b.tsv \
    --n-seeds 20
```

### 4.3 Knowledge Graph Construction & Relation Capping
Reconstruct the capped training graph from raw edge definitions:
```bash
python scripts/build_kg.py \
    --nodes-file data/kg/nodes_v06.tsv \
    --edges-file data/kg/edges_v06.tsv \
    --cap-encodes 5000 \
    --cap-associates 5000 \
    --cap-interacts 5000 \
    --seed 42 \
    --output-edges data/kg/edges_v06_capped.tsv
```

### 4.4 R-GCN Entity Embedding Training
Train the 2-layer R-GCN with basis decomposition and ComplEx decoder:
```bash
python scripts/train_rgcn.py \
    --nodes-file data/kg/nodes_v06.tsv \
    --edges-file data/kg/edges_v06.tsv \
    --hidden-dim 256 \
    --out-dim 256 \
    --num-bases 30 \
    --epochs 375 \
    --lr 0.001 \
    --output-dir checkpoints/rgcn
```

### 4.5 Transformer Pretraining (Factored MMP)
Pretrain the orderless microbiome transformer on unlabelled metagenomes:
```bash
python scripts/pretrain_transformer.py \
    --data data/examples/sample_pretrain.tsv \
    --vocab data/vocab/genus_vocab.txt \
    --rgcn-emb data/kg/entity_embeddings.npy \
    --bridge data/vocab/bridge.json \
    --d-model 256 \
    --n-heads 8 \
    --n-layers 6 \
    --epochs 10 \
    --batch-size 32 \
    --output-dir checkpoints/pretrain
```

---

## 5. Hardware Requirements and Estimated Runtimes

| Task | Device | Approximate Runtime |
| :--- | :---: | :---: |
| Full Test Suite (`tests/run_all.py`) | CPU (single core) | ~15 seconds |
| Table & Figure Generation (`scripts/generate_*.py`) | CPU | < 5 seconds |
| Downstream Probing (`evaluate_loco.py`) on example cohorts | CPU | ~30 seconds |
| Full R-GCN Training (375 epochs) | 1x NVIDIA GPU | ~45 minutes |
| Transformer Pretraining (15,255 samples, 10 epochs) | 1x NVIDIA A100 GPU | ~6 hours |
