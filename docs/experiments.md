# Experimental Design and Results

This document describes the three empirical evaluation regimes and experimental controls reported in our IEEE BIBM 2026 paper (*Paper ID: B1518*).

---

## 1. Overview of Experimental Framework

The central hypothesis of this work is that initializing a microbiome foundation model with structured, multi-relational biological knowledge improves:
1. **Out-of-Distribution Generalizability**: Transferring learned disease classifiers across distinct clinical cohorts (LOCO).
2. **Label Efficiency**: Achieving superior diagnostic accuracy when only tiny fractions of labelled clinical samples are available (in-cohort probing).
3. **Biological Alignment**: Outperforming classical tree baselines and randomized graph controls that lack relational biomedical semantics.

---

## 2. Leave-One-Cohort-Out (LOCO) Cross-Study Transfer

### 2.1 Protocol
In LOCO cross-cohort transfer:
- The transformer backbone, embedding table, and projection layer remain **strictly frozen**.
- A linear classification head is trained on the source cohort (50 epochs, AdamW, $\text{lr}=10^{-3}$).
- Evaluated on the held-out target cohort without any fine-tuning, domain adaptation, or target-label exposure.
- Evaluated across 20 paired random seeds.
- Evaluated over 8 directed same-disease transfer tasks across T2D and CVD cohorts.

### 2.2 Canonical LOCO Transfer Results

| Task ID | Source Cohort $\to$ Target Cohort | Disease | Graph-Init AUROC | Random-Init AUROC | $\Delta$ AUROC | Transfer Outcome |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Qin $\to$ Karlsson | T2D | **0.6271** | 0.6015 | +0.0255 | Graph Win |
| 2 | Karlsson $\to$ Qin | T2D | **0.5931** | 0.5184 | +0.0747 | Graph Win |
| 3 | Qin $\to$ MetaCardis | T2D | **0.6383** | 0.5279 | +0.1104 | Graph Win |
| 4 | MetaCardis $\to$ Qin | T2D | **0.5966** | 0.5135 | +0.0831 | Graph Win |
| 5 | Karlsson $\to$ MetaCardis | T2D | **0.6195** | 0.4927 | +0.1268 | Graph Win |
| 6 | MetaCardis $\to$ Karlsson | T2D | **0.6699** | 0.5394 | +0.1305 | Graph Win |
| 7 | ACVD $\to$ MetaCardis-CVD | CVD | **0.5267** | 0.5101 | +0.0166 | Graph Win |
| 8 | MetaCardis-CVD $\to$ ACVD | CVD | 0.4427 | **0.5361** | -0.0934 | Known Counterexample |

**Key Takeaways**:
- Knowledge graph initialization outperforms random initialization in **7 of 8 directed cohort transfers**.
- In T2D transfers, KG initialization delivers substantial gains up to **+0.1305 AUROC** (MetaCardis $\to$ Karlsson).
- Task 8 represents a documented negative transfer counterexample, arising from technical sequencing differences and severe cross-study batch effects between European and Chinese CVD cohorts (see [limitations.md](limitations.md)).

---

## 3. In-Cohort Probing Across Label Fractions

### 3.1 Protocol
In-cohort probing evaluates sample-efficiency by training frozen linear probes across subsampled training label fractions:
$$\text{Fractions} \in \{1\%, 2\%, 5\%, 25\%, 50\%, 100\%\}$$
Evaluated across 20 paired random seeds on 5 independent clinical cohorts.

### 3.2 Canonical Results Summary (100% Label Fraction)

| Cohort | Condition | Graph-Init AUROC | Random-Init AUROC | $\Delta$ AUROC |
| :--- | :---: | :---: | :---: | :---: |
| **Qin et al.** | T2D | **0.7812** | 0.7420 | +0.0392 |
| **Karlsson et al.** | T2D | **0.7104** | 0.6841 | +0.0263 |
| **MetaCardis T2D** | T2D | **0.7455** | 0.7198 | +0.0257 |
| **ACVD** | CVD | **0.8142** | 0.7935 | +0.0207 |
| **MetaCardis CVD** | CVD | **0.7621** | 0.7389 | +0.0232 |

In low-data regimes (1% to 5% label fractions), graph initialization provides even larger relative performance advantages, demonstrating rapid convergence on small clinical cohorts.

---

## 4. Experimental Controls

To confirm that performance gains derive from biological relational knowledge rather than trivial graph density artifacts, we evaluated four structural controls:

1. **Random Initialization**: Randomly initialized embeddings ($\mathcal{N}(0, 0.02^2)$).
2. **Shuffle Initialization (`shuffle_init`)**: 5 derangement seeds permuting genus identities while preserving the 256-dimensional R-GCN embedding distribution.
3. **Co-Occurrence Graph Control**: R-GCN trained strictly on taxonomic co-occurrence graphs derived from abundance correlations.
4. **Shuffled-KG Edge Control**: R-GCN trained on degree-preserved edge permutations.

The canonical graph model significantly outperforms both the derangement controls and co-occurrence graphs, establishing that multi-relational biological semantics drive transfer generalizability.

---

## 5. Comparison to Classical Machine Learning Baselines

We evaluated standard and domain-adapted tree baselines across the 8 LOCO transfer pairs:
1. **Intersection Random Forest**: Restricts features to the shared genus intersection between source and target cohorts.
2. **Zero-Fill Random Forest**: Trains on source genus features; fills unmeasured features with zeros at test time.
3. **Taxonomy-Collapsed Random Forest**: Collapses genus-level abundances to family rank before transfer.
4. **CORAL Random Forest**: Subspace covariance alignment (whitening and recoloring) between source and target feature spaces.

Classical tree baselines degrade severely under cross-cohort feature distribution shifts (often yielding near-random transfer AUROCs between 0.48 and 0.54), whereas the knowledge-informed foundation model successfully transfers learned representations.
