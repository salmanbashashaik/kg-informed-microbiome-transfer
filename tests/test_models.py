import unittest
from pathlib import Path
import torch
from microbiome_kg.transformer.config import ModelConfig
from microbiome_kg.transformer.model import MicrobiomeTransformer, FactoredMMPHead, ClassificationHead
from microbiome_kg.graph_models.model import RGCN, ComplExDecoder

REPO_ROOT = Path(__file__).resolve().parent.parent
RGCN_EMB = str(REPO_ROOT / "data" / "kg" / "entity_embeddings.npy")
BRIDGE_FILE = str(REPO_ROOT / "data" / "vocab" / "bridge.json")


class TestModels(unittest.TestCase):
    def test_rgcn_forward(self):
        num_nodes = 50
        num_relations = 5
        model = RGCN(num_nodes=num_nodes, hidden_dim=64, out_dim=64, num_relations=num_relations)
        decoder = ComplExDecoder(num_relations=num_relations, emb_dim=64)

        edge_index = torch.randint(0, num_nodes, (2, 100))
        edge_type = torch.randint(0, num_relations, (100,))

        emb = model(edge_index, edge_type)
        self.assertEqual(emb.shape, (num_nodes, 64))

        scores = decoder(emb[edge_index[0]], edge_type, emb[edge_index[1]])
        self.assertEqual(scores.shape, (100,))

    def test_transformer_forward(self):
        cfg = ModelConfig(vocab_size=1041, d_model=256, n_heads=8, n_layers=2)

        # Test random init
        model_rand = MicrobiomeTransformer(cfg, graph_init=False)
        # Test graph init with canonical embeddings
        model_graph = MicrobiomeTransformer(
            cfg,
            rgcn_emb_file=RGCN_EMB,
            bridge_file=BRIDGE_FILE,
            graph_init=True,
        )

        B, L = 4, 32
        g_ids = torch.randint(4, 1041, (B, L))
        b_ids = torch.randint(0, 5, (B, L))

        for m in [model_rand, model_graph]:
            cls_repr, seq_repr = m(g_ids, b_ids)
            self.assertEqual(cls_repr.shape, (B, 256))
            self.assertEqual(seq_repr.shape, (B, L + 1, 256))  # Includes prepended CLS token

        # Test heads
        mmp_head = FactoredMMPHead(d_model=256, n_bins=5)
        pres_logits, bin_logits = mmp_head(seq_repr[:, 1:, :])
        self.assertEqual(pres_logits.shape, (B, L, 2))
        self.assertEqual(bin_logits.shape, (B, L, 4))

        clf_head = ClassificationHead(d_model=256, num_classes=2)
        logits = clf_head(cls_repr)
        self.assertEqual(logits.shape, (B, 2))


if __name__ == "__main__":
    unittest.main()

