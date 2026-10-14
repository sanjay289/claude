#!/usr/bin/env python3
"""Tests for brain.py's capture, search, and ask.

Each test points brain.VAULT at a temp directory. Search tests need
ripgrep and are skipped without it. Ollama is stubbed out for ask, so
no server or model is needed. Run with: python3 -m unittest discover tests
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
from types import SimpleNamespace
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


WORDS = ["tomato", "oil", "router"]


def fake_embed(model, input):
    """One dimension per keyword, so similarity is keyword overlap."""
    vectors = [[float(w in text.lower()) for w in WORDS] + [0.01] for text in input]
    return SimpleNamespace(embeddings=vectors)


def fake_chat(model, messages, stream):
    yield SimpleNamespace(message=SimpleNamespace(content="Sept 3 "))
    yield SimpleNamespace(message=SimpleNamespace(content="[[Garden]]"))


class AskTests(VaultTestCase):
    def setUp(self):
        super().setUp()
        self.embed = patch.object(brain.ollama, "embed", side_effect=fake_embed)
        self.embed_mock = self.embed.start()
        self.addCleanup(self.embed.stop)
        for folder, name, text in [
            ("Projects", "Garden", "Tomato planted Sept 3.\n\nWatered daily."),
            ("Areas", "Car", "Oil change due at 48,500 km."),
        ]:
            (self.vault / folder).mkdir(parents=True, exist_ok=True)
            (self.vault / folder / f"{name}.md").write_text(text)

    def test_index_chunks_by_paragraph_and_names_notes(self):
        index = brain.load_index()
        self.assertEqual(
            [(name, chunk) for name, chunk, _ in index],
            [("Car", "Oil change due at 48,500 km."),
             ("Garden", "Tomato planted Sept 3."),
             ("Garden", "Watered daily.")],
        )
        texts = self.embed_mock.call_args_list[0].kwargs["input"]
        self.assertTrue(texts[0].startswith("search_document: "))

    def test_cache_reembeds_only_changed_notes(self):
        brain.load_index()
        self.assertEqual(self.embed_mock.call_count, 2)
        brain.load_index()
        self.assertEqual(self.embed_mock.call_count, 2)
        (self.vault / "Areas" / "Car.md").write_text("Oil change at 50,000 km.")
        index = brain.load_index()
        self.assertEqual(self.embed_mock.call_count, 3)
        self.assertIn(("Car", "Oil change at 50,000 km."), [i[:2] for i in index])

    def test_cache_drops_deleted_notes(self):
        brain.load_index()
        (self.vault / "Areas" / "Car.md").unlink()
        self.assertEqual({i[0] for i in brain.load_index()}, {"Garden"})

    def test_retrieve_ranks_by_similarity(self):
        hits = brain.retrieve("when was the tomato planted?", brain.load_index(), k=1)
        self.assertEqual(hits, [("Garden", "Tomato planted Sept 3.")])

    def test_ask_sends_retrieved_notes_and_streams_answer(self):
        with patch.object(brain.ollama, "chat", side_effect=fake_chat) as chat, \
                redirect_stdout(io.StringIO()) as out:
            brain.ask("tomato?")
        self.assertEqual(out.getvalue(), "Sept 3 [[Garden]]\n")
        prompt = chat.call_args.kwargs["messages"][1]["content"]
        self.assertIn("[[Garden]]\nTomato planted Sept 3.", prompt)
        self.assertIn("Question: tomato?", prompt)

    def test_ask_with_empty_vault(self):
        shutil.rmtree(self.vault)
        self.vault.mkdir()
        with redirect_stdout(io.StringIO()) as out:
            brain.ask("anything")
        self.assertEqual(out.getvalue(), "no notes to search\n")

    def test_main_ask_starts_server(self):
        with patch.object(brain, "ensure_server_running") as server, \
                patch.object(brain, "ask") as ask:
            self.assertEqual(brain.main(["ask", "two", "words"]), 0)
        server.assert_called_once()
        ask.assert_called_once_with("two words")


class UsageTests(unittest.TestCase):
    def test_bad_args_print_usage(self):
        with patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(brain.main([]), 2)
            self.assertEqual(brain.main(["add"]), 2)
        self.assertIn("brain add TEXT", err.getvalue())


if __name__ == "__main__":
    unittest.main()
