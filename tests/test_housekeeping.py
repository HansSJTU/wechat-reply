"""Unit tests for wxbot/housekeeping.py: launchd log rotation and data/media expiry."""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

from wxbot import housekeeping

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestRotateLogs(unittest.TestCase):
    def test_rotates_inherited_descriptor_in_a_child_process(self):
        """Like launchd: the log is opened outside the process and passed as stdout / stderr."""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "monitor.log")
            for i in (1, 2, 3):  # older generations: .3 must fall off the end
                with open(f"{p}.{i}", "w") as f:
                    f.write(f"gen{i}\n")
            code = ("import sys, os; sys.path.insert(0, sys.argv[1]); from wxbot import housekeeping as h\n"
                    "print('x' * 2000, flush=True)\n"
                    "assert h.rotate_std_logs(1000, keep=3) == [sys.argv[2]], 'not rotated'\n"
                    "assert h.rotate_std_logs(1000, keep=3) == [], 'rotated twice'\n"
                    "print('after', flush=True); print('err', file=sys.stderr, flush=True)\n")
            with open(p, "a") as log:
                subprocess.run([sys.executable, "-c", code, ROOT, os.path.realpath(p)],
                               stdout=log, stderr=log, check=True, timeout=60)
            with open(p) as f:
                self.assertEqual(f.read(), "after\nerr\n")
            with open(p + ".1") as f:
                self.assertEqual(f.read(), "x" * 2000 + "\n")
            with open(p + ".2") as f:
                self.assertEqual(f.read(), "gen1\n")
            with open(p + ".3") as f:
                self.assertEqual(f.read(), "gen2\n")
            self.assertFalse(os.path.exists(p + ".4"))

    def test_terminal_or_small_files_are_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "small.log")
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
            try:
                os.write(fd, b"tiny\n")
                self.assertEqual(housekeeping.rotate_std_logs(1000, fds=(fd,)), [])
                r, w = os.pipe()
                try:
                    self.assertEqual(housekeeping.rotate_std_logs(0, fds=(w,)), [])  # not a regular file
                finally:
                    os.close(r), os.close(w)
            finally:
                os.close(fd)
            self.assertFalse(os.path.exists(p + ".1"))


class TestCleanMedia(unittest.TestCase):
    def test_expiry_rules(self):
        now = time.time()
        with tempfile.TemporaryDirectory() as d:
            def make(name, age_days):
                path = os.path.join(d, name)
                with open(path, "wb") as f:
                    f.write(b"x")
                os.utime(path, (now - age_days * 86400, now - age_days * 86400))
                return name
            old_img, new_img = make("1_aa.jpg", 40), make("2_bb.png", 5)
            old_voice, labeled_voice = make("3_cc_voice.wav", 400), make("4_dd_voice.wav", 400)
            scrap, fresh_scrap = make("5_ee_voice.silk", 2), make("6_ff_voice.pcm", 0.1)
            os.mkdir(os.path.join(d, "sub"))

            labels = os.path.join(d, "labels.jsonl")
            with open(labels, "w", encoding="utf-8") as f:
                f.write(json.dumps({"key": labeled_voice, "status": "skip"}) + "\n{half-writ")
            protected = housekeeping.labeled_keys(labels)
            self.assertEqual(protected, {labeled_voice})

            # default: voices are kept, old images and stale scraps go
            n = housekeeping.clean_media(d, image_days=30, voice_days=0, protected=protected, now=now)
            self.assertEqual(n, 2)
            left = set(os.listdir(d))
            self.assertEqual(left, {new_img, old_voice, labeled_voice, fresh_scrap, "sub", "labels.jsonl"})
            self.assertNotIn(old_img, left)
            self.assertNotIn(scrap, left)

            # voice_days set: unlabeled old voices go, labeled ones never do
            n = housekeeping.clean_media(d, image_days=0, voice_days=365, protected=protected, now=now)
            self.assertEqual(n, 1)
            self.assertEqual(set(os.listdir(d)), {new_img, labeled_voice, fresh_scrap, "sub", "labels.jsonl"})

    def test_missing_dir(self):
        self.assertEqual(housekeeping.clean_media("/nonexistent/media", image_days=1), 0)
        self.assertEqual(housekeeping.labeled_keys("/nonexistent/labels.jsonl"), set())


if __name__ == "__main__":
    unittest.main()
