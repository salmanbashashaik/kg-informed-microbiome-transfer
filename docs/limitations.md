# Limitations and Scientific Discussion

In accordance with rigorous scientific integrity, this document details the architectural limitations, data considerations, and documented transfer anomalies of the Knowledge Graph-Informed Foundation Model presented in our IEEE BIBM 2026 paper (*Paper ID: B1518*).

---

## 1. Transparent Reporting of Negative Transfer (Task 8 Counterexample)

Across 8 directed same-disease cross-cohort transfers (LOCO), knowledge graph pretraining outperformed random initialization in **7 of 8 tasks**. 

However, in **Task 8 (MetaCardis-CVD $\to$ ACVD)**, knowledge graph initialization underperformed random initialization:
- **Graph-Init AUROC**: 0.4427 $\pm$ 0.021
- **Random-Init AUROC**: 0.5361 $\pm$ 0.018
- **$\Delta$ AUROC**: -0.0934 (Knowledge Graph Loss)

### Root-Cause Analysis of Asymmetric Transfer
1. **Severe Geographic & Lifestyle Divergence**:
   The MetaCardis CVD cohort represents European patients (France, Germany, Denmark) characterized by Western dietary habits and clinical co-medications (e.g., widespread statin usage). In contrast, the ACVD cohort represents Chinese patients with distinct traditional dietary patterns and metabolic baselines.
2. **Sequencing Protocol and Library Discrepancies**:
   ACVD samples were generated using specific library preparation and sequencing protocols that yielded distinct genus abundance sparsity patterns compared to MetaCardis.
3. **Directional Transfer Asymmetry**:
   Notably, the reverse transfer (**ACVD $\to$ MetaCardis-CVD**) achieved a positive graph gain (Graph 0.5267 vs. Random 0.5101, $\Delta +0.0166$). This asymmetry indicates that training on European CVD representations over-constrains feature attribution on Chinese cohorts, highlighting the need for multi-cohort domain generalization strategies in future iterations.

---

## 2. Unbridged Genera and Incomplete Knowledge Representation

Our biomedical knowledge graph bridges **991 out of 1,037 genus tokens (95.6%)**.

However, **46 genera** (4.4%) could not be mapped to canonical KEGG / NCBI / HMDB entities. These unbridged taxa include:
- Candidate genera and uncultured environmental isolates (e.g., *Candidatus* lineages) lacking formal metabolic annotation.
- Ambiguous taxonomic synonyms across distinct reference databases.

*Mitigation*: Unbridged genera fallback to Gaussian initialization ($\mathcal{N}(0, 0.02^2)$). Ongoing efforts in automated bio-curation will expand knowledge graph entity coverage to newly identified microbial taxa.

---

## 3. Taxonomic Granularity (Genus vs. Species/Strain)

The current model operates at the **genus level**:
- *Advantages*: Genus-level profiles provide robust, cross-platform harmonization across diverse sequencing runs, metagenomic shotgun pipelines, and 16S amplicon datasets.
- *Limitations*: Genus-level abstraction cannot resolve strain-specific metabolic differences (e.g., distinguishing between commensal and enteropathogenic strains of *Escherichia*, or strain-level mucin-degrading capabilities in *Akkermansia*). Extending the multi-relational graph to strain-level genomic variants is an important direction for future research.

---

## 4. Metagenomic vs. 16S Profiling Interoperability

The pretraining corpus pools shotgun metagenomic sequencing with 16S rRNA amplicon data:
- Shotgun metagenomics provides quantitative DNA abundance across whole genomes.
- 16S rRNA sequencing is subject to copy number variation (ranging from 1 to 15 copies per genome across bacterial phyla) and hypervariable region primer biases.

While abundance binning into 5 broad orders of magnitude ($[0, 10^{-4}, 10^{-3}, 10^{-2}]$) significantly attenuates these quantification discrepancies, residual technical variations remain.
