#!/usr/bin/env python3
"""Tests for hermes_agent's graph_engine-based agent loop.

Ollama is stubbed out with fake chat responses so these run offline —
no server, no model pulled. Run with: python3 -m unittest discover tests
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

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


class ParseSkillFileTests(unittest.TestCase):
    def test_parses_frontmatter_and_body(self):
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write("---\nname: custom-name\ndescription: does a thing\n---\nBody text here.\n")
            path = f.name
        try:
            meta = ha.parse_skill_file(path)
            self.assertEqual(meta["name"], "custom-name")
            self.assertEqual(meta["description"], "does a thing")
            self.assertEqual(meta["body"], "Body text here.")
        finally:
            os.remove(path)

    def test_no_frontmatter_uses_filename_as_name(self):
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write("Just body, no frontmatter.")
            path = f.name
        try:
            meta = ha.parse_skill_file(path)
            self.assertEqual(meta["name"], os.path.splitext(os.path.basename(path))[0])
            self.assertEqual(meta["description"], "")
            self.assertEqual(meta["body"], "Just body, no frontmatter.")
        finally:
            os.remove(path)


class DiscoverSkillsTests(unittest.TestCase):
    def test_missing_dir_returns_empty(self):
        with patch.object(ha, "SKILLS_DIR", "/nonexistent/path/xyz"):
            self.assertEqual(ha.discover_skills(), {})

    def test_discovers_md_files_only(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.md"), "w") as f:
                f.write("---\ndescription: skill A\n---\nbody A")
            with open(os.path.join(d, "b.md"), "w") as f:
                f.write("---\ndescription: skill B\n---\nbody B")
            with open(os.path.join(d, "ignore.txt"), "w") as f:
                f.write("not a skill")
            with patch.object(ha, "SKILLS_DIR", d):
                skills = ha.discover_skills()
            self.assertEqual(set(skills), {"a", "b"})
            self.assertEqual(skills["a"]["description"], "skill A")


class RunShellTests(unittest.TestCase):
    def test_returns_stripped_stdout(self):
        self.assertEqual(ha.run_shell("echo hello"), "hello")

    def test_no_output_returns_placeholder(self):
        self.assertEqual(ha.run_shell("true"), "(no output)")

    def test_timeout_returns_error_message(self):
        with patch.object(
            ha.subprocess, "run",
            side_effect=ha.subprocess.TimeoutExpired(cmd="x", timeout=30),
        ):
            self.assertEqual(ha.run_shell("sleep 100"), "Error: command timed out")

    def test_exception_returns_error_message(self):
        with patch.object(ha.subprocess, "run", side_effect=RuntimeError("boom")):
            self.assertEqual(ha.run_shell("whatever"), "Error: boom")


class ReadWriteFileTests(unittest.TestCase):
    def test_read_file_returns_contents(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("file contents")
            path = f.name
        try:
            self.assertEqual(ha.read_file(path), "file contents")
        finally:
            os.remove(path)

    def test_read_file_missing_returns_error(self):
        result = ha.read_file("/nonexistent/path/xyz.txt")
        self.assertTrue(result.startswith("Error:"))

    def test_write_file_writes_and_confirms(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.txt")
            result = ha.write_file(path, "hello")
            self.assertEqual(result, f"Written to {path}")
            with open(path) as f:
                self.assertEqual(f.read(), "hello")

    def test_write_file_error_returns_message(self):
        result = ha.write_file("/nonexistent_dir_xyz/out.txt", "hello")
        self.assertTrue(result.startswith("Error:"))


class ListDirTests(unittest.TestCase):
    def test_lists_entries(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "file1"), "w").close()
            open(os.path.join(d, "file2"), "w").close()
            entries = set(ha.list_dir(d).splitlines())
            self.assertEqual(entries, {"file1", "file2"})

    def test_empty_path_defaults_to_cwd(self):
        result = ha.list_dir("")
        self.assertFalse(result.startswith("Error:"))

    def test_missing_dir_returns_error(self):
        result = ha.list_dir("/nonexistent/path/xyz")
        self.assertTrue(result.startswith("Error:"))


class ListSkillsTests(unittest.TestCase):
    def test_no_skills_found(self):
        with patch.object(ha, "discover_skills", return_value={}):
            self.assertEqual(ha.list_skills(), "(no skills found)")

    def test_lists_name_and_description(self):
        with patch.object(ha, "discover_skills", return_value={
            "a": {"description": "desc A"},
            "b": {"description": "desc B"},
        }):
            self.assertEqual(ha.list_skills(), "a: desc A\nb: desc B")


class UseSkillTests(unittest.TestCase):
    def test_returns_skill_body(self):
        with patch.object(ha, "discover_skills", return_value={"a": {"body": "the body"}}):
            self.assertEqual(ha.use_skill("a"), "the body")

    def test_unknown_skill_lists_available(self):
        with patch.object(ha, "discover_skills", return_value={"a": {}, "b": {}}):
            result = ha.use_skill("missing")
            self.assertIn("Unknown skill 'missing'", result)
            self.assertIn("a", result)

    def test_unknown_skill_when_none_available(self):
        with patch.object(ha, "discover_skills", return_value={}):
            result = ha.use_skill("missing")
            self.assertIn("(none)", result)


class LooksLikeFakeToolCallTests(unittest.TestCase):
    def test_empty_content_is_false(self):
        self.assertFalse(ha.looks_like_fake_tool_call(""))
        self.assertFalse(ha.looks_like_fake_tool_call(None))

    def test_no_braces_is_false(self):
        self.assertFalse(ha.looks_like_fake_tool_call("just plain text"))

    def test_invalid_json_is_false(self):
        self.assertFalse(ha.looks_like_fake_tool_call("here's a { broken json }"))

    def test_json_that_is_not_a_dict_is_false(self):
        with patch.object(ha.json, "loads", return_value=[1, 2, 3]):
            self.assertFalse(ha.looks_like_fake_tool_call("{anything}"))


class CallToolTests(unittest.TestCase):
    def test_unknown_tool_returns_message(self):
        self.assertEqual(ha.call_tool("nonexistent_tool", {}), "Unknown tool: nonexistent_tool")

    def test_known_tool_executes_and_returns_result(self):
        result = ha.call_tool("get_datetime", {})
        self.assertRegex(result, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


class MainTests(unittest.TestCase):
    def test_quit_exits_immediately(self):
        with patch("builtins.input", return_value="quit"):
            ha.main()

    def test_empty_input_is_skipped_then_quit(self):
        with patch("builtins.input", side_effect=["", "exit"]):
            ha.main()

    def test_eof_breaks_loop(self):
        with patch("builtins.input", side_effect=EOFError):
            ha.main()

    def test_processes_input_via_agent_loop(self):
        with patch("builtins.input", side_effect=["hello", "q"]), \
             patch.object(ha, "agent_loop", return_value="answer") as agent_loop_mock:
            ha.main()
        agent_loop_mock.assert_called_once_with("hello")


if __name__ == "__main__":
    unittest.main()
