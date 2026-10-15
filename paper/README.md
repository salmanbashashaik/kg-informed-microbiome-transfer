# Paper Overview & Scientific Artifact Mapping

**Paper Title**: Knowledge Graph-Informed Foundation Model for Microbiome-Cardiovascular Disease Classification  
**Conference**: IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2026)  
**Paper ID**: B1518  
**Acceptance Status**: Accepted  

### Authors:
1. **Shaik Salman Basha** (University of Waterloo)
2. **Ziting Shen** (University of Waterloo)
3. **Anita Layton** (University of Waterloo)
4. **Sun Sun** (University of Waterloo)
5. **Helen Hong Chen** (University of Waterloo)

---

## Abstract

Microbiome-based disease classification models typically suffer from poor cross-cohort generalizability due to technical batch effects, study-specific sequencing protocols, geographic variation in taxonomic distributions, and sparse genus representation. We introduce a biologically grounded foundation model for human gut microbiome data that integrates a multi-relational biomedical knowledge graph into transformer pretraining via Relational Graph Convolutional Networks (R-GCN). Across cross-cohort transfer evaluations spanning Type 2 Diabetes and Cardiovascular Disease, knowledge graph pretraining improves cross-study transfer over random initialization in 7 of 8 cohort transfers. The learned representations demonstrate strong label efficiency, yielding significant diagnostic accuracy improvements in extreme low-data regimes (1%–5% label fractions) while outperforming derangement-based structural controls and classical tree baselines.

---

## Claim-to-Artifact Mapping

Every quantitative claim and experimental result presented in the paper maps directly to an audited code implementation and canonical result file in this repository:

| Paper Claim / Component | Description | Canonical Repository Artifacts |
| :--- | :--- | :--- |
| **Knowledge Graph Schema** | 2,856 nodes, 19,642 directed edges, 15 node types, 23 relations | `data/kg/nodes_v06.tsv`<br>`scripts/build_kg.py`<br>`docs/knowledge_graph.md` |
| **R-GCN Embeddings** | 256-dim embeddings trained with ComplEx decoder (Val MRR: 0.0569) | `data/kg/entity_embeddings.npy`<br>`src/microbiome_kg/graph_models/model.py`<br>`scripts/train_rgcn.py` |
| **Vocabulary & Bridging** | 1,041 tokens (1,037 genera); 991 bridged genera, 46 unbridged | `data/vocab/genus_vocab.txt`<br>`data/vocab/bridge.json`<br>`src/microbiome_kg/bridging/bridge.py` |
| **Pretraining Corpus** | 15,255 metagenomic profiles (AGP, HMP, MGnify, cMD) | `data/manifests/datasets_manifest.json`<br>`src/microbiome_kg/data/dataset.py`<br>`scripts/pretrain_transformer.py` |
| **Transformer Backbone** | 6 layers, 8 heads, 256 hidden dim, ~5.1M parameters | `src/microbiome_kg/transformer/model.py`<br>`src/microbiome_kg/transformer/config.py`<br>`checkpoints/manifests/checkpoints_manifest.json` |
| **LOCO Transfer (7/8 Wins)** | Cross-cohort transfer across 8 directed T2D and CVD pairs | `results/canonical/loco_transfer_results.json`<br>`results/tables/table2_loco_transfer.md`<br>`results/figures/fig3_loco_transfer.png`<br>`scripts/evaluate_loco.py` |
| **Label Efficiency (Probing)** | Frozen linear probing across 1%, 2%, 5%, 25%, 50%, 100% fractions | `results/canonical/incohort_fractions_results.json`<br>`results/tables/table1_incohort_fractions.md`<br>`results/figures/fig4_incohort_fractions.png`<br>`scripts/evaluate_incohort.py` |
| **Structural Controls** | Derangement shuffle (5 seeds), co-occurrence graph, shuffled KG | `results/canonical/shuffle_control_results.json`<br>`results/figures/fig2_shuffle_control.png`<br>`data/kg/edges_v06_shuffled_seed1.tsv`<br>`data/kg/edges_v06_cooccurrence.tsv` |
| **Classical Baselines** | Intersection RF, Zero-Fill RF, Taxonomy-collapsed RF, CORAL | `results/canonical/classical_baselines_results.json`<br>`results/tables/table3_classical_baselines.md`<br>`src/microbiome_kg/baselines/` |
| **Negative Transfer Analysis** | Documented MetaCardis-CVD $\to$ ACVD counterexample ($\Delta -0.0934$) | `docs/limitations.md`<br>`results/tables/table2_loco_transfer.md` |
