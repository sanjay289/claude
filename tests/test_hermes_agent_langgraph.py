#!/usr/bin/env python3
"""Tests for hermes_agent_langgraph's LangGraph-based agent loop.

hermes_agent_langgraph re-exports TOOLS/TOOL_MAP/call_tool/
looks_like_fake_tool_call/discover_skills etc. unchanged from hermes_agent
(same objects) — those are already covered by test_hermes_agent.py, so
these tests focus on what's actually different here: build_agent_graph/
agent_loop on LangGraph, and main().

Ollama is stubbed out with fake chat responses so these run offline — no
server, no model pulled. Run with: python3 -m unittest discover tests
"""

import os
import runpy
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hermes_agent_langgraph as hlg


class FakeFn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name, arguments):
        self.function = FakeFn(name, arguments)


class FakeMsg:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeResp:
    def __init__(self, content, tool_calls=None):
        self.message = FakeMsg(content, tool_calls)


def make_chat(sequence):
    """Returns a fake ollama.chat(model, messages, tools) that replays
    `sequence`, repeating the last response once exhausted."""
    calls = {"n": 0}

    def chat(model, messages, tools):
        i = calls["n"]
        calls["n"] += 1
        return sequence[min(i, len(sequence) - 1)]

    return chat


class HermesAgentLangGraphTests(unittest.TestCase):
    def setUp(self):
        self._real_chat = hlg.ollama.chat
        self.addCleanup(lambda: setattr(hlg.ollama, "chat", self._real_chat))

    def test_plain_answer_ends_immediately(self):
        hlg.ollama.chat = make_chat([FakeResp("hello there")])
        self.assertEqual(hlg.agent_loop("hi"), "hello there")

    def test_real_tool_call_routes_through_execute_tools(self):
        hlg.ollama.chat = make_chat([
            FakeResp(None, [FakeToolCall("get_datetime", {})]),
            FakeResp("done using tool"),
        ])
        self.assertEqual(hlg.agent_loop("what time is it"), "done using tool")

    def test_fake_tool_call_retries_then_falls_back_to_raw_text(self):
        fake_text = 'I will call {"name": "get_datetime"}'
        hlg.ollama.chat = make_chat([FakeResp(fake_text)])
        self.assertEqual(hlg.agent_loop("do the thing"), fake_text)

    def test_fake_tool_call_succeeds_after_retry(self):
        fake_text = 'I will call {"name": "get_datetime"}'
        hlg.ollama.chat = make_chat([FakeResp(fake_text), FakeResp("real answer")])
        self.assertEqual(hlg.agent_loop("do the thing"), "real answer")

    def test_iteration_cap_enforced(self):
        hlg.ollama.chat = make_chat([
            FakeResp(None, [FakeToolCall("get_datetime", {})]),
        ])
        self.assertEqual(
            hlg.agent_loop("loop forever", max_iterations=2), "Max iterations reached."
        )

    def test_multiple_tool_calls_in_one_turn_all_execute(self):
        hlg.ollama.chat = make_chat([
            FakeResp(None, [
                FakeToolCall("get_datetime", {}),
                FakeToolCall("list_dir", {"path": "."}),
            ]),
            FakeResp("used both tools"),
        ])
        self.assertEqual(hlg.agent_loop("do two things"), "used both tools")


class MainTests(unittest.TestCase):
    def test_quit_exits_immediately(self):
        with patch("builtins.input", return_value="quit"):
            hlg.main()

    def test_empty_input_is_skipped_then_quit(self):
        with patch("builtins.input", side_effect=["", "exit"]):
            hlg.main()

    def test_eof_breaks_loop(self):
        with patch("builtins.input", side_effect=EOFError):
            hlg.main()

    def test_processes_input_via_agent_loop(self):
        with patch("builtins.input", side_effect=["hello", "q"]), \
             patch.object(hlg, "agent_loop", return_value="answer") as agent_loop_mock:
            hlg.main()
        agent_loop_mock.assert_called_once_with("hello")


class MainGuardTests(unittest.TestCase):
    def test_running_as_script_calls_main(self):
        with patch("builtins.input", return_value="quit"):
            runpy.run_module("hermes_agent_langgraph", run_name="__main__")


if __name__ == "__main__":
    unittest.main()
