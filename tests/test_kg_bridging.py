import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BRIDGE_FILE = str(REPO_ROOT / "data" / "vocab" / "bridge.json")
NODE_ID_FILE = str(REPO_ROOT / "data" / "kg" / "node_id_to_row.json")


class TestKGBridging(unittest.TestCase):
    def test_bridge_counts(self):
        with open(BRIDGE_FILE, "r") as f:
            bridge = json.load(f)

        with open(NODE_ID_FILE, "r") as f:
            node_to_row = json.load(f)

        # Canonical paper bridging: exactly 991 bridged genera out of 1037 genus tokens
        self.assertEqual(len(bridge), 991, f"Expected 991 bridged genera, got {len(bridge)}")
        unbridged = 1037 - len(bridge)
        self.assertEqual(unbridged, 46, f"Expected 46 unbridged genera, got {unbridged}")

        # Verify that all bridged row indices are valid node rows
        num_kg_nodes = len(node_to_row)
        self.assertEqual(num_kg_nodes, 2856, f"Expected 2,856 KG nodes, got {num_kg_nodes}")

        for tok_id_str, row_idx in bridge.items():
            tok_id = int(tok_id_str)
            self.assertTrue(4 <= tok_id < 1041, f"Invalid token id: {tok_id}")
            self.assertTrue(0 <= row_idx < num_kg_nodes, f"Invalid RGCN row index: {row_idx}")


if __name__ == "__main__":
    unittest.main()

