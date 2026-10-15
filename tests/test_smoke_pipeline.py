import unittest
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from microbiome_kg.transformer.config import ModelConfig, PAD_ID, MASK_ID
from microbiome_kg.transformer.model import MicrobiomeTransformer, FactoredMMPHead, ClassificationHead
from microbiome_kg.data.dataset import MicrobiomeDataset, TaskDataset, collate_microbiome_batch, collate_task_batch
from microbiome_kg.evaluation.metrics import compute_binary_metrics
from microbiome_kg.baselines.random_forest import run_intersection_rf, run_zerofill_rf

REPO_ROOT = Path(__file__).resolve().parent.parent
VOCAB_FILE = str(REPO_ROOT / "data" / "vocab" / "genus_vocab.txt")
SAMPLE_PRETRAIN = str(REPO_ROOT / "data" / "examples" / "sample_pretrain.tsv")
SAMPLE_COHORT_A = str(REPO_ROOT / "data" / "examples" / "sample_cohort_a.tsv")
SAMPLE_COHORT_B = str(REPO_ROOT / "data" / "examples" / "sample_cohort_b.tsv")


class TestSmokePipeline(unittest.TestCase):
    def test_end_to_end_pretrain_step(self):
        """Smoke test verifying a single optimization step of masked microbiome pretraining on CPU."""
        torch.manual_seed(42)
        cfg = ModelConfig(vocab_size=1041, d_model=64, n_heads=4, n_layers=2)
        model = MicrobiomeTransformer(cfg, graph_init=False)
        head = FactoredMMPHead(d_model=64, n_bins=5)

        df_pretrain = pd.read_csv(SAMPLE_PRETRAIN, sep="\t")
        dataset = MicrobiomeDataset(df_pretrain, VOCAB_FILE, mask_nonzero_p=0.5, mask_zero_p=0.2)
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_microbiome_batch,
        )

        batch = next(iter(loader))
        input_genus = batch["genus_ids"]
        input_bins = batch["bin_ids"]
        pres_targets = batch["presence_targets"]
        bin_targets = batch["bin_targets"]

        optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-3)
        optimizer.zero_grad()

        _, seq_repr = model(input_genus, input_bins)
        # Exclude CLS token (index 0) from sequence representations
        pres_logits, bin_logits = head(seq_repr[:, 1:, :])

        mask_pos = (pres_targets != -100)
        if mask_pos.sum() > 0:
            loss_pres = F.cross_entropy(pres_logits[mask_pos], pres_targets[mask_pos])
        else:
            loss_pres = torch.tensor(0.0, requires_grad=True)

        nonzero_mask = (bin_targets != -100)
        if nonzero_mask.sum() > 0:
            loss_bin = F.cross_entropy(bin_logits[nonzero_mask], bin_targets[nonzero_mask])
            total_loss = 0.3 * loss_pres + 0.7 * loss_bin
        else:
            total_loss = loss_pres

        total_loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(total_loss).item(), "Pretraining loss is not finite.")
        self.assertGreater(total_loss.item(), 0.0, "Pretraining loss should be positive.")

    def test_end_to_end_loco_step(self):
        """Smoke test verifying frozen backbone representation extraction and transfer probing on CPU."""
        torch.manual_seed(42)
        cfg = ModelConfig(vocab_size=1041, d_model=64, n_heads=4, n_layers=2)
        backbone = MicrobiomeTransformer(cfg, graph_init=False)
        backbone.eval()

        df_a = pd.read_csv(SAMPLE_COHORT_A, sep="\t")
        df_b = pd.read_csv(SAMPLE_COHORT_B, sep="\t")

        ds_a = TaskDataset(df_a, VOCAB_FILE)
        ds_b = TaskDataset(df_b, VOCAB_FILE)

        loader_a = DataLoader(ds_a, batch_size=len(ds_a), shuffle=False, collate_fn=collate_task_batch)
        loader_b = DataLoader(ds_b, batch_size=len(ds_b), shuffle=False, collate_fn=collate_task_batch)

        batch_a = next(iter(loader_a))
        batch_b = next(iter(loader_b))

        # Extract frozen representations
        with torch.no_grad():
            emb_a, _ = backbone(batch_a["genus_ids"], batch_a["bin_ids"])
            emb_b, _ = backbone(batch_b["genus_ids"], batch_b["bin_ids"])

        y_a = batch_a["label"]
        y_b = batch_b["label"].numpy()

        # Train linear probe on source cohort A
        clf = ClassificationHead(d_model=64, num_classes=2)
        optimizer = torch.optim.AdamW(clf.parameters(), lr=1e-2)

        for _ in range(5):
            optimizer.zero_grad()
            logits_a = clf(emb_a)
            loss = F.cross_entropy(logits_a, y_a)
            loss.backward()
            optimizer.step()

        # Evaluate zero-shot transfer on target cohort B
        clf.eval()
        with torch.no_grad():
            logits_b = clf(emb_b)
            probs_b = F.softmax(logits_b, dim=1)[:, 1].numpy()

        metrics = compute_binary_metrics(y_b, probs_b)
        self.assertIn("auroc", metrics)
        self.assertIn("auprc", metrics)
        self.assertTrue(0.0 <= metrics["auroc"] <= 1.0)
        self.assertTrue(0.0 <= metrics["auprc"] <= 1.0)

    def test_end_to_end_baseline_step(self):
        """Smoke test verifying classical baseline random forest execution on example cohorts."""
        df_a = pd.read_csv(SAMPLE_COHORT_A, sep="\t")
        df_b = pd.read_csv(SAMPLE_COHORT_B, sep="\t")

        metrics_inter = run_intersection_rf(df_a, df_b, n_seeds=2, n_estimators=10)
        self.assertIn("auroc_mean", metrics_inter)
        self.assertIn("auprc_mean", metrics_inter)
        self.assertTrue(0.0 <= metrics_inter["auroc_mean"] <= 1.0)

        metrics_zero = run_zerofill_rf(df_a, df_b, n_seeds=2, n_estimators=10)
        self.assertIn("auroc_mean", metrics_zero)
        self.assertIn("auprc_mean", metrics_zero)
        self.assertTrue(0.0 <= metrics_zero["auroc_mean"] <= 1.0)



if __name__ == "__main__":
    unittest.main()
