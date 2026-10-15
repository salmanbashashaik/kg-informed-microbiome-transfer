# Checkpoints Directory

This directory manages the foundation model weights and R-GCN link prediction checkpoints used in our IEEE BIBM 2026 paper (*Paper ID: B1518*).

---

## Model Checkpoint Manifests

All pretraining model weights are versioned and cataloged with their exact SHA-256 cryptographic hashes in `checkpoints/manifests/checkpoints_manifest.json`:

| Checkpoint Name | Initialization Arm | Parameter Count | Model Dimension | SHA-256 Checksum |
| :--- | :---: | :---: | :---: | :--- |
| **`graph_init/best_model.pt`** | Knowledge Graph (R-GCN) | 5,148,829 (~5.1M) | $d_{\text{model}}=256$ | `3aa820cc8567119ff92d19f50e70c6347c6999b844ca15f16e4ee640ca2df61a` |
| **`random_init/best_model.pt`** | Random Gaussian | 5,148,829 (~5.1M) | $d_{\text{model}}=256$ | `21994cd3c0a0c242c74d081e7d2358055628b05fcfece5336fbf18868840ca88` |
| **`shuffle_init_seed{1..5}`** | Derangement Shuffle | 5,148,829 (~5.1M) | $d_{\text{model}}=256$ | Documented in manifest |

---

## Checkpoint Loading and Verification

To verify that model weights match the reported architecture and tensor shapes:
```python
import torch
from microbiome_kg.transformer.config import ModelConfig
from microbiome_kg.transformer.model import MicrobiomeTransformer

cfg = ModelConfig(vocab_size=1041, d_model=256, n_heads=8, n_layers=6)
model = MicrobiomeTransformer(cfg, graph_init=True)
# model.load_state_dict(torch.load("path/to/best_model.pt", map_location="cpu"))
```
Binary checkpoint `.pt` files (> 20 MB) are managed via Git LFS / persistent archive links per conference repository guidelines.
