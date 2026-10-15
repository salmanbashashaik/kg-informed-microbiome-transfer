# Datasets and Cohorts

This document describes the pretraining corpora, disease evaluation cohorts, taxonomic harmonization, and dataset accession information for our IEEE BIBM 2026 paper (*Paper ID: B1518*).

---

## 1. Pretraining Corpus (15,255 Metagenomic Profiles)

To train the foundation model backbone without disease-specific bias, we pooled 15,255 human stool metagenomic and 16S amplicon profiles across four major open-access consortia:

| Dataset Source | Modality | Sample Count | Primary Cohort Description |
| :--- | :--- | :---: | :--- |
| **American Gut Project (AGP)** | 16S rRNA / Metagenomic | 9,511 | Large citizen-science cohort spanning diverse North American/UK lifestyles |
| **Human Microbiome Project (HMP)** | Metagenomic shotgun | 748 | Healthy adult baseline multi-site human microbiome profiles |
| **MGnify Repository** | Metagenomic shotgun | 2,000 | Curated European and global human gut metagenomes |
| **curatedMetagenomicData (cMD)** | Metagenomic shotgun | 2,996 | Harmonized, healthy control profiles across international metagenomic studies |
| **Total Pretraining Pool** | Pooled | **15,255** | **Orderless, unlabelled foundation pretraining corpus** |

All pretraining samples were normalized and tokenized according to the canonical 1,041-token genus vocabulary (`data/vocab/genus_vocab.txt`).

---

## 2. Downstream Disease Evaluation Cohorts

Cross-cohort generalization (Leave-One-Cohort-Out, LOCO) and sample-efficiency (in-cohort probing) were rigorously evaluated across five international cohorts spanning two major cardiometabolic diseases:

### 2.1 Type 2 Diabetes (T2D) Cohorts
1. **Qin et al. (2012)** (*Nature* 490, 55–60):
   - Population: Han Chinese
   - Samples: $N = 344$ (170 T2D cases, 174 non-diabetic controls)
   - Accession: NCBI SRA ERP002469 / BioProject PRJNA422434
2. **Karlsson et al. (2013)** (*Nature* 498, 99–103):
   - Population: European (Swedish postmenopausal women)
   - Samples: $N = 145$ (53 T2D cases, 92 non-diabetic controls)
   - Accession: NCBI BioProject PRJEB1982
3. **MetaCardis T2D Sub-cohort** (*Nature Medicine* 28, 545–555, 2022):
   - Population: European (France, Germany, Denmark)
   - Samples: $N = 158$ (79 T2D cases, 79 balanced non-diabetic controls)
   - Accession: EBI ArrayExpress E-MTAB-9488

### 2.2 Cardiovascular Disease (CVD) Cohorts
1. **Atherosclerotic Cardiovascular Disease (ACVD)** (Jie et al., *Nature Communications* 8:845, 2017):
   - Population: Chinese cohort
   - Samples: $N = 385$ (214 ACVD cases, 171 healthy controls)
   - Accession: NCBI BioProject PRJNA342217 / SRA SRP090684
   - *Note on Data Fix*: Evaluated strictly with the verified July 5, 2026 data correction.
2. **MetaCardis CVD Cohort** (*Nature Medicine* 28, 545–555, 2022):
   - Population: European (France, Germany, Denmark)
   - Samples: $N = 414$ (207 CVD cases, 207 healthy controls)
   - Accession: EBI ArrayExpress E-MTAB-9488

---

## 3. Taxonomic Harmonization & Vocabulary Bridging

### 3.1 Genus Extraction and Filtering
Taxonomic profiles were processed through MetaPhlAn / NCBI taxonomy pipelines. Microbial taxa were mapped to the genus rank:
- Strings matching NCBI / MetaPhlAn taxpaths (e.g., `k__Bacteria|...|g__Bacteroides`) are parsed to isolate the genus token.
- Typographical prefixes (`g__`), capitalization, and non-alphanumeric trailing strings are standardized using `normalize_genus()`.

### 3.2 Canonical Vocabulary
- Total vocabulary size: **1,041 tokens**
  - 4 special reserved tokens: `[PAD]` (0), `[MASK]` (1), `[CLS]` (2), `[UNK]` (3)
  - 1,037 microbial genera (IDs 4 through 1040)
- Vocabulary bridge:
  - 991 genera bridge directly to nodes in the Knowledge Graph.
  - 46 genera remain unbridged and are initialized via $\mathcal{N}(0, 0.02^2)$.

---

## 4. Abundance Discretization

Relative abundance values are converted into 5 discrete ordinal bins:
$$\text{Bin}(x) = \begin{cases} 
0 & \text{if } x = 0 \quad (\text{absent}) \\
1 & \text{if } 0 < x \le 10^{-4} \quad (\text{trace}) \\
2 & \text{if } 10^{-4} < x \le 10^{-3} \quad (\text{low}) \\
3 & \text{if } 10^{-3} < x \le 10^{-2} \quad (\text{medium}) \\
4 & \text{if } x > 10^{-2} \quad (\text{high})
\end{cases}$$

Percentage scale inputs ($x > 1.5$) are automatically divided by 100 before applying bin thresholds.

---

## 5. Privacy, Ethics, and Example Datasets

In accordance with institutional ethics guidelines and human data privacy laws (GDPR / HIPAA), individual patient-level sequencing FASTQ files cannot be bundled directly within public GitHub repositories. 

To enable full zero-friction reproducibility:
- Public download instructions and accession IDs are provided above.
- Cryptographic hashes and dimensions for all datasets are cataloged in `data/manifests/datasets_manifest.json`.
- Fully functional demonstration benchmark cohorts formatted identically to real cohorts are provided under `data/examples/` (`sample_cohort_a.tsv`, `sample_cohort_b.tsv`, `sample_pretrain.tsv`).
