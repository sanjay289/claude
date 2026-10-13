#!/usr/bin/env python3
"""Tests for brain.py's capture and search.

Each test points brain.VAULT at a temp directory. Search tests need
ripgrep and are skipped without it. Run with: python3 -m unittest discover tests
"""

import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import brain


class VaultTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self.tmp.name) / "vault"
        patcher = patch.object(brain, "VAULT", self.vault)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)


class AddTests(VaultTestCase):
    def test_creates_para_folders_and_daily_note(self):
        note = brain.add("first", now=datetime(2026, 10, 13, 9, 5))
        for folder in brain.FOLDERS:
            self.assertTrue((self.vault / folder).is_dir())
        self.assertEqual(note, self.vault / "Inbox" / "2026-10-13.md")
        self.assertEqual(note.read_text(), "- 09:05 first\n")

    def test_second_add_appends(self):
        brain.add("first", now=datetime(2026, 10, 13, 9, 5))
        note = brain.add("second", now=datetime(2026, 10, 13, 9, 6))
        self.assertEqual(note.read_text(), "- 09:05 first\n- 09:06 second\n")

    def test_main_add_joins_words(self):
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(brain.main(["add", "two", "words"]), 0)
        self.assertIn("added to Inbox", out.getvalue())
        note = next((self.vault / "Inbox").glob("*.md"))
        self.assertTrue(note.read_text().rstrip().endswith("two words"))


@unittest.skipIf(shutil.which("rg") is None, "ripgrep not installed")
class FindTests(VaultTestCase):
    def test_finds_case_insensitive_match(self):
        brain.add("Proot fixes DNS", now=datetime(2026, 10, 13, 9, 5))
        matches = brain.find("proot")
        self.assertEqual(len(matches), 1)
        self.assertIn("2026-10-13.md:1:", matches[0])

    def test_no_match_and_missing_vault(self):
        self.assertEqual(brain.find("anything"), [])
        brain.add("hello", now=datetime(2026, 10, 13, 9, 5))
        self.assertEqual(brain.find("absent"), [])

    def test_main_find_prints_when_not_a_tty(self):
        brain.add("hello world", now=datetime(2026, 10, 13, 9, 5))
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(brain.main(["find", "hello"]), 0)
        self.assertIn("hello world", out.getvalue())
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(brain.main(["find", "absent"]), 1)
        self.assertEqual(out.getvalue(), "no matches\n")


class UsageTests(unittest.TestCase):
    def test_bad_args_print_usage(self):
        with patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(brain.main([]), 2)
            self.assertEqual(brain.main(["add"]), 2)
        self.assertIn("brain add TEXT", err.getvalue())


if __name__ == "__main__":
    unittest.main()
