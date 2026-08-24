#!/usr/bin/env python3
"""Tests for council.py's graph_engine-based deliberation flow.

Ollama and subprocess are stubbed out so these run offline — no server,
no models pulled. Run with: python3 -m unittest discover tests
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import council


class FakeMsg:
    def __init__(self, content):
        self.content = content


class FakeResp:
    def __init__(self, content):
        self.message = FakeMsg(content)


class AskNodeTests(unittest.TestCase):
    def setUp(self):
        self._real_chat = council.ollama.chat
        self.addCleanup(lambda: setattr(council.ollama, "chat", self._real_chat))

    def test_returns_answer_content(self):
        council.ollama.chat = lambda model, messages: FakeResp("some answer")
        ask = council.make_ask_node("some-model")
        self.assertEqual(ask({"question": "Q"}), "some answer")

    def test_empty_response_falls_back(self):
        council.ollama.chat = lambda model, messages: FakeResp(None)
        ask = council.make_ask_node("some-model")
        self.assertEqual(ask({"question": "Q"}), "(empty response)")

    def test_exception_is_captured_not_raised(self):
        def raising_chat(model, messages):
            raise RuntimeError("boom")
        council.ollama.chat = raising_chat
        ask = council.make_ask_node("some-model")
        self.assertEqual(ask({"question": "Q"}), "(error: boom)")


class SynthesizeNodeTests(unittest.TestCase):
    def setUp(self):
        self._real_chat = council.ollama.chat
        self.addCleanup(lambda: setattr(council.ollama, "chat", self._real_chat))

    def test_builds_verdict_from_branch_results(self):
        council.ollama.chat = lambda model, messages: FakeResp("synthesized answer")
        state = {
            "question": "What's best?",
            "branch_results": {
                "ask::model-a": "answer A",
                "ask::model-b": "answer B",
            },
        }
        self.assertEqual(council.synthesize_node(state), {"verdict": "synthesized answer"})

    def test_exception_is_captured_not_raised(self):
        def raising_chat(model, messages):
            raise RuntimeError("boom")
        council.ollama.chat = raising_chat
        state = {"question": "Q", "branch_results": {}}
        self.assertEqual(council.synthesize_node(state), {"verdict": "(error: boom)"})


class BuildCouncilGraphTests(unittest.TestCase):
    def test_has_start_ask_and_synthesize_nodes(self):
        graph = council.build_council_graph()
        expected = {"start", "synthesize"} | {f"ask::{m}" for m in council.MEMBERS}
        self.assertEqual(set(graph._nodes), expected)
        self.assertEqual(graph._entry, "start")


class RunCouncilTests(unittest.TestCase):
    def setUp(self):
        self._real_chat = council.ollama.chat
        self.addCleanup(lambda: setattr(council.ollama, "chat", self._real_chat))

    def test_full_flow_returns_chair_verdict(self):
        def fake_chat(model, messages):
            if model == council.CHAIR:
                return FakeResp("final verdict")
            return FakeResp(f"answer from {model}")
        council.ollama.chat = fake_chat
        self.assertEqual(council.run_council("What's best?"), "final verdict")


class EnsureServerRunningTests(unittest.TestCase):
    def setUp(self):
        self._real_list = council.ollama.list
        self.addCleanup(lambda: setattr(council.ollama, "list", self._real_list))

    def test_returns_immediately_when_server_already_up(self):
        council.ollama.list = lambda: True
        with patch("council.subprocess.Popen") as popen:
            council.ensure_server_running()
        popen.assert_not_called()

    def test_starts_server_when_down_then_waits_for_it(self):
        calls = {"n": 0}

        def flaky_list():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("down")
            return True

        council.ollama.list = flaky_list
        with patch("council.subprocess.Popen") as popen, \
             patch("council.time.sleep") as sleep:
            council.ensure_server_running()
        popen.assert_called_once_with(
            ["ollama", "serve"],
            stdout=council.subprocess.DEVNULL,
            stderr=council.subprocess.DEVNULL,
        )
        self.assertTrue(sleep.called)

    def test_raises_if_server_never_comes_up(self):
        council.ollama.list = lambda: (_ for _ in ()).throw(ConnectionError("down"))
        with patch("council.subprocess.Popen"), patch("council.time.sleep"):
            with self.assertRaises(RuntimeError):
                council.ensure_server_running(timeout=0.05)


if __name__ == "__main__":
    unittest.main()
