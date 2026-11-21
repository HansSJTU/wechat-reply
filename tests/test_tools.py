"""Unit tests for tools/private_io.py (owner-only files for key material)."""
import json
import os
import stat
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import private_io  # noqa: E402


def mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


class TestPrivateIO(unittest.TestCase):
    def test_write_private_json_is_0600_atomic_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "keys.json")
            with open(p + ".tmp", "w") as f:  # a stale, world-readable tmp from an earlier run
                f.write("old")
            os.chmod(p + ".tmp", 0o644)
            private_io.write_private_json(p, {"raw_key": "ab"})
            self.assertEqual(mode(p), 0o600)
            with open(p) as f:
                self.assertEqual(json.load(f), {"raw_key": "ab"})
            self.assertFalse(os.path.exists(p + ".tmp"))

            with self.assertRaises(TypeError):  # not JSON-serializable: old file kept, no tmp left
                private_io.write_private_json(p, {"bad": object()})
            with open(p) as f:
                self.assertEqual(json.load(f), {"raw_key": "ab"})
            self.assertFalse(os.path.exists(p + ".tmp"))

    def test_create_private_tightens_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "captures.tsv")
            os.close(private_io.create_private(p))
            self.assertEqual(mode(p), 0o600)
            os.chmod(p, 0o644)
            fd = private_io.create_private(p)
            os.write(fd, b"aes\tff\n")
            os.close(fd)
            self.assertEqual(mode(p), 0o600)
            with open(p, "rb") as f:
                self.assertEqual(f.read(), b"aes\tff\n")


if __name__ == "__main__":
    unittest.main()
