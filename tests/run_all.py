"""
Test runner for the full test suite.
Executes all unit and integration smoke tests.
"""

import sys
import unittest
from pathlib import Path

# Add src to python path so package imports succeed without installation
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

if __name__ == "__main__":
    test_dir = str(REPO_ROOT / "tests")
    suite = unittest.defaultTestLoader.discover(test_dir, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
