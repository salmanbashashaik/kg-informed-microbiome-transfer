import unittest
from pathlib import Path
from microbiome_kg.data.dataset import load_vocab
from microbiome_kg.transformer.config import PAD_ID, MASK_ID, CLS_ID, UNK_ID, VOCAB_OFFSET

REPO_ROOT = Path(__file__).resolve().parent.parent
VOCAB_FILE = str(REPO_ROOT / "data" / "vocab" / "genus_vocab.txt")


class TestVocabulary(unittest.TestCase):
    def test_vocab_structure(self):
        tok2id, id2tok = load_vocab(VOCAB_FILE)

        self.assertEqual(len(id2tok), 1041, f"Expected 1,041 tokens, got {len(id2tok)}")
        self.assertEqual(id2tok[PAD_ID], "[PAD]")
        self.assertEqual(id2tok[MASK_ID], "[MASK]")
        self.assertEqual(id2tok[CLS_ID], "[CLS]")
        self.assertEqual(id2tok[UNK_ID], "[UNK]")

        genus_tokens = id2tok[VOCAB_OFFSET:]
        self.assertEqual(len(genus_tokens), 1037, f"Expected 1,037 genus tokens, got {len(genus_tokens)}")

        # Ensure all genus tokens are unique
        self.assertEqual(len(set(genus_tokens)), 1037)


if __name__ == "__main__":
    unittest.main()

