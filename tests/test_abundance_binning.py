import unittest
import numpy as np
from microbiome_kg.data.dataset import bin_abundance, is_already_binned
from microbiome_kg.transformer.config import BIN_EDGES


class TestAbundanceBinning(unittest.TestCase):
    def test_bin_abundance_edges(self):
        # Exact edge checks: [0.0, 1e-4, 1e-3, 1e-2]
        # 0.0 -> bin 0 (absent)
        # 5e-5 -> bin 1 (trace: 0 to 1e-4]
        # 5e-4 -> bin 2 (low: 1e-4 to 1e-3]
        # 5e-3 -> bin 3 (medium: 1e-3 to 1e-2]
        # 5e-2 -> bin 4 (high: > 1e-2)
        test_vals = np.array([0.0, 5e-5, 5e-4, 5e-3, 5e-2])
        bins = bin_abundance(test_vals)
        expected = np.array([0, 1, 2, 3, 4])
        np.testing.assert_array_equal(bins, expected)

    def test_percentage_scale_autodetect(self):
        # 5.0% = 0.05 -> bin 4 (> 1e-2)
        # 0.05% = 5e-4 -> bin 2 (1e-4 to 1e-3]
        pct_vals = np.array([0.0, 0.005, 0.05, 0.5, 5.0])
        bins = bin_abundance(pct_vals)
        expected = np.array([0, 1, 2, 3, 4])
        np.testing.assert_array_equal(bins, expected)

    def test_is_already_binned(self):
        binned = np.array([0, 1, 4, 2, 3])
        self.assertTrue(is_already_binned(binned))

        continuous = np.array([0.0, 0.05, 0.12])
        self.assertFalse(is_already_binned(continuous))


if __name__ == "__main__":
    unittest.main()

