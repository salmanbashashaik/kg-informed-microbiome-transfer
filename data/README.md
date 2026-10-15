# Data Directory Structure

This directory contains knowledge graph definitions, vocabulary artifacts, dataset manifests, and reproducible demonstration cohorts for the IEEE BIBM 2026 paper (*Paper ID: B1518*).

---

## Directory Organization

```
data/
├── kg/
│   ├── nodes_v06.tsv                 # Canonical 2,856 entity nodes across 15 biological types
│   ├── edges_v06.tsv                 # Raw multi-relational edges before capping
│   ├── edges_v06_capped.tsv          # Capped training graph (19,642 edges, 23 relation types)
│   ├── entity_embeddings.npy         # Dense 256-dim R-GCN embeddings (scale normalized)
│   ├── node_id_to_row.json           # Node identifier to R-GCN row index mapping
│   ├── edges_v06_shuffled_seed1.tsv  # Structural control: permuted edge topology
│   └── edges_v06_cooccurrence.tsv    # Structural control: abundance co-occurrence graph
├── vocab/
│   ├── genus_vocab.txt               # Canonical 1,041-token vocabulary
│   └── bridge.json                   # Token ID to R-GCN row mapping (991 bridged genera)
├── manifests/
│   └── datasets_manifest.json        # Checksums, row/col counts, and accession IDs for all cohorts
└── examples/
    ├── sample_cohort_a.tsv           # Demonstration source cohort for testing (N=50 samples)
    ├── sample_cohort_b.tsv           # Demonstration target cohort for testing (N=50 samples)
    └── sample_pretrain.tsv           # Demonstration pretraining samples for testing (N=20 samples)
```

---

## Accessing Full Patient Cohorts

Due to human genomic privacy regulations (HIPAA/GDPR), raw clinical sequencing profiles are subject to controlled data access:
- **Qin et al. (2012)**: NCBI SRA ERP002469 / BioProject PRJNA422434
- **Karlsson et al. (2013)**: NCBI BioProject PRJEB1982
- **MetaCardis (T2D & CVD)**: EBI ArrayExpress E-MTAB-9488
- **ACVD (Jie et al., 2017)**: NCBI BioProject PRJNA342217 / SRA SRP090684

To test and execute pipelines immediately without downloading gigabytes of patient data, use the demonstration cohorts in `data/examples/`.

