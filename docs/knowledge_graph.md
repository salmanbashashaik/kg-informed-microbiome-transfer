# Multi-Relational Knowledge Graph

This document details the curation, schema, relation-frequency capping, R-GCN representation learning, and structural controls of the biological knowledge graph (KG) used in our IEEE BIBM 2026 paper (*Paper ID: B1518*).

---

## 1. Graph Summary and Canonical Statistics

The biological knowledge graph connects gut microbial taxa with metabolic pathways, enzymatic functions, dietary components, cardiovascular phenotypes, and host cardiometabolic disease mechanisms.

| Metric | Raw Expanded Graph | Capped Training Graph (Canonical) |
| :--- | :---: | :---: |
| **Total Unique Nodes** | 2,865 | **2,856** |
| **Total Directed Edges** | 165,161 | **19,642** |
| **Entity Node Types** | 15 | **15** |
| **Relation Types** | 23 | **23** (all $\ge 50$ edges) |
| **Node TSV Artifact** | `data/kg/nodes_v06.tsv` | `data/kg/nodes_v06.tsv` |
| **Edge TSV Artifact** | `kg/edges_v06.tsv` | Generated via reproducible frequency capping |

---

## 2. Entity Node Schema (15 Types)

The 2,856 nodes span 15 biological categories:
1. `Microbe`: Human gut microbial genera and species.
2. `Metabolite`: Small-molecule biochemical intermediates (e.g., Short-Chain Fatty Acids [butyrate, acetate, propionate], TMA, TMAO, bile acids).
3. `Pathway`: KEGG and MetaCyc biological and metabolic pathways.
4. `Disease`: Cardiometabolic conditions (e.g., Type 2 Diabetes, Coronary Artery Disease, Atherosclerosis, Hypertension).
5. `Gene` / `Protein`: Microbial and host enzymes and functional receptors.
6. `Food`: Dietary nutrients, fiber classes, and ultra-processed food (UPF) additives.
7. `BiologicalProcess`: Gene Ontology biological processes.
8. `MolecularFunction`: Gene Ontology molecular functions.
9. `CellularComponent`: Cellular compartments.
10. `ChemicalSubstance`: Pharmacological compounds and therapeutic drugs.
11. `Drug`: Approved medications affecting host metabolism (e.g., metformin, statins).
12. `Symptom`: Clinical markers and phenotypes.
13. `AnatomicalEntity`: Human gut compartments and vascular tissues.
14. `Phenotype`: Observable cardiovascular biomarkers (e.g., lipid profiles, blood pressure).
15. `Taxon`: Higher-level phylogenetic groupings.

---

## 3. Relation-Frequency Capping

In biological knowledge bases, certain generic functional annotations (such as gene-protein encoding or ubiquitous co-occurrence) generate hundreds of thousands of edges, disproportionately dominating graph convolutions.

To prevent representation collapse, relations with counts exceeding 5,000 edges are capped using uniform random subsampling (`seed=42`):
- `encodes`: Capped from 85,241 to **5,000** edges
- `associates_with`: Capped from 41,209 to **5,000** edges
- `interacts_with`: Capped from 18,924 to **5,000** edges

All 23 relation types in the resulting graph maintain at least 50 edges, providing balanced multi-hop biological paths.

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

---

## 4. Relational Graph Convolutional Network (R-GCN) Embeddings

### 4.1 Architecture & Training
- **Model**: 2-layer R-GCN with basis decomposition ($B = 30$ basis matrices)
- **Decoder**: ComplEx scoring decoder in complex embedding space
- **Hidden / Output Dimension**: 256
- **Regularization**: Self-adversarial negative sampling, $L_2$ weight regularization, and anti-anisotropy regularizers.
- **Convergence & Performance**: Checkpoint at epoch 375 (`rgcn_v06_verified_epoch375`) achieved:
  - Validation Mean Reciprocal Rank (MRR): **0.0569**
  - Validation Hits@10: **0.0937**

### 4.2 Scale Normalization for Transformer Injection
To ensure that injected frozen R-GCN embeddings match the initial variance scale of randomly initialized weights ($\mathcal{N}(0, 0.02^2)$):
$$\mathbf{e}_g^{\text{init}} = \gamma \cdot \frac{\mathbf{e}_g^{\text{RGCN}}}{\|\mathbf{e}_g^{\text{RGCN}}\|_2}$$
where $\gamma = 0.02 \times \sqrt{256} = 0.32$. The normalized 256-dimensional embeddings for all 2,856 entities are saved in `data/kg/entity_embeddings.npy`.

---

## 5. Structural Experimental Controls

To confirm that downstream transfer performance derives specifically from *multi-relational biological semantics* rather than graph density or numerical regularization, the codebase includes three structural controls:

1. **Derangement Shuffle Control (`shuffle_init`)**:
   Preserves the exact geometric distribution of the 256-dimensional R-GCN embeddings, but randomly deranges the genus-to-embedding assignment (zero fixed points across 5 seeds).
2. **Shuffled-KG Edge Control (`edges_v06_shuffled_seed1.tsv`)**:
   Permutes the edge endpoints while preserving node degrees and relation frequencies, training an R-GCN on topological noise.
3. **Co-occurrence Graph Control (`edges_v06_cooccurrence.tsv`)**:
   Replaces multi-relational biological edges with undirected taxonomic co-occurrence edges computed directly from pretraining metagenomic abundance profiles.
