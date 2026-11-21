#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["cryptography", "zstandard", "httpx", "frida==17.26.3", "numpy"]
# ///
"""Run the offline unit test suite: `uv run tests/run.py [-v] [pattern]`."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    args = sys.argv[1:]
    verbosity = 2 if "-v" in args else 1
    patterns = [a for a in args if a != "-v"]
    loader = unittest.TestLoader()
    if patterns:
        suite = unittest.TestSuite(loader.loadTestsFromName(p) for p in patterns)
    else:
        suite = loader.discover(os.path.dirname(os.path.abspath(__file__)), pattern="test_*.py")
    res = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    sys.exit(0 if res.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
