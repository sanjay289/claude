#!/usr/bin/env python3
"""Tests for hermes_agent's graph_engine-based agent loop.

Ollama is stubbed out with fake chat responses so these run offline —
no server, no model pulled. Run with: python3 -m unittest discover tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hermes_agent as ha


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


class HermesAgentGraphTests(unittest.TestCase):
    def setUp(self):
        self._real_chat = ha.ollama.chat
        self.addCleanup(lambda: setattr(ha.ollama, "chat", self._real_chat))

    def test_plain_answer_ends_immediately(self):
        ha.ollama.chat = make_chat([FakeResp("hello there")])
        self.assertEqual(ha.agent_loop("hi"), "hello there")

    def test_real_tool_call_routes_through_execute_tools(self):
        ha.ollama.chat = make_chat([
            FakeResp(None, [FakeToolCall("get_datetime", {})]),
            FakeResp("done using tool"),
        ])
        self.assertEqual(ha.agent_loop("what time is it"), "done using tool")

    def test_fake_tool_call_retries_then_falls_back_to_raw_text(self):
        fake_text = 'I will call {"name": "get_datetime"}'
        ha.ollama.chat = make_chat([FakeResp(fake_text)])
        self.assertEqual(ha.agent_loop("do the thing"), fake_text)

    def test_fake_tool_call_succeeds_after_retry(self):
        fake_text = 'I will call {"name": "get_datetime"}'
        ha.ollama.chat = make_chat([FakeResp(fake_text), FakeResp("real answer")])
        self.assertEqual(ha.agent_loop("do the thing"), "real answer")

    def test_iteration_cap_enforced(self):
        ha.ollama.chat = make_chat([
            FakeResp(None, [FakeToolCall("get_datetime", {})]),
        ])
        self.assertEqual(ha.agent_loop("loop forever", max_iterations=2), "Max iterations reached.")


if __name__ == "__main__":
    unittest.main()
