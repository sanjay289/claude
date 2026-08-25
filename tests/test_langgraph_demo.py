#!/usr/bin/env python3
"""Tests for langgraph_demo.py.

Everything that would touch a real model (ChatOllama.invoke, the
create_react_agent-based subagent) is mocked, so these run offline — no
ollama server, no model pulled. Importing langgraph_demo itself is safe
without a server: ChatOllama's constructor and create_react_agent don't
make network calls, only .invoke() does.

No runpy-based "running as script calls main()" guard test here, unlike
test_hermes_agent.py/test_hermes_agent_langgraph.py: those work because
`import ollama; ollama.chat = fake` patches a shared module object that
survives a runpy re-exec of the whole file. langgraph_demo builds `llm` and
`subagent` (and computes DB_PATH from __file__) at import time, so a fresh
runpy re-execution would reconstruct all of that from scratch, pointing at
the real repo DB_PATH and requiring class-level mocking gymnastics for
little extra coverage over calling main() directly (below) — not worth the
fragility for a demo script.

langgraph requires Python >=3.10, so this whole module is skipped on older
interpreters (matching requirements.txt's environment marker on the
langgraph line) rather than failing to import.

Run with: python3 -m unittest discover tests
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.version_info < (3, 10):
    def setUpModule():
        raise unittest.SkipTest("langgraph_demo requires langgraph, which needs Python 3.10+")
else:
    from typing import TypedDict

    from langchain_core.messages import AIMessage
    from langgraph.graph import END, StateGraph

    import langgraph_demo as lgd


class TempDbTestCase(unittest.TestCase):
    """Base class for tests that need a scratch sqlite file, cleaned up
    after the test regardless of outcome."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))


class SimpleSqliteSaverTests(TempDbTestCase):
    """Exercises SimpleSqliteSaver through a real (tiny, unrelated) compiled
    graph rather than calling put/get_tuple by hand, since the actual
    contract that matters is "does LangGraph's own checkpoint/resume
    machinery work against this saver" — which is also how the real bug
    (thread-safety, see the class docstring in langgraph_demo.py) surfaced."""

    def _build_counter_app(self, saver):
        class CounterState(TypedDict):
            count: int

        builder = StateGraph(CounterState)
        builder.add_node("add_one", lambda s: {"count": s["count"] + 1})
        builder.add_node("add_two", lambda s: {"count": s["count"] + 2})
        builder.add_edge("__start__", "add_one")
        builder.add_edge("add_one", "add_two")
        builder.add_edge("add_two", END)
        return builder.compile(checkpointer=saver)

    def test_persists_across_separate_saver_instances(self):
        """The actual behavior asked for: state survives a process restart,
        not just staying alive within one Python process."""
        config = {"configurable": {"thread_id": "t1"}}
        with lgd.SimpleSqliteSaver(self.db_path) as saver:
            app = self._build_counter_app(saver)
            result = app.invoke({"count": 0}, config=config)
        self.assertEqual(result["count"], 3)

        with lgd.SimpleSqliteSaver(self.db_path) as saver2:
            app2 = self._build_counter_app(saver2)
            state = app2.get_state(config)
        self.assertEqual(state.values["count"], 3)

    def test_get_tuple_returns_none_for_unknown_thread(self):
        with lgd.SimpleSqliteSaver(self.db_path) as saver:
            config = {"configurable": {"thread_id": "nope"}}
            self.assertIsNone(saver.get_tuple(config))

    def test_list_returns_checkpoints_most_recent_first(self):
        config = {"configurable": {"thread_id": "t2"}}
        with lgd.SimpleSqliteSaver(self.db_path) as saver:
            app = self._build_counter_app(saver)
            app.invoke({"count": 0}, config=config)
            checkpoint_ids = [t.checkpoint["id"] for t in saver.list(config)]
        self.assertGreaterEqual(len(checkpoint_ids), 2)
        self.assertEqual(checkpoint_ids, sorted(checkpoint_ids, reverse=True))

    def test_list_respects_limit(self):
        config = {"configurable": {"thread_id": "t3"}}
        with lgd.SimpleSqliteSaver(self.db_path) as saver:
            app = self._build_counter_app(saver)
            app.invoke({"count": 0}, config=config)
            limited = list(saver.list(config, limit=1))
        self.assertEqual(len(limited), 1)

    def test_get_tuple_with_explicit_checkpoint_id_returns_that_checkpoint(self):
        config = {"configurable": {"thread_id": "t4"}}
        with lgd.SimpleSqliteSaver(self.db_path) as saver:
            app = self._build_counter_app(saver)
            app.invoke({"count": 0}, config=config)
            all_tuples = list(saver.list(config))
            earliest, latest = all_tuples[-1], all_tuples[0]
            fetched = saver.get_tuple(earliest.config)
        self.assertEqual(fetched.checkpoint["id"], earliest.checkpoint["id"])
        self.assertNotEqual(fetched.checkpoint["id"], latest.checkpoint["id"])

    def test_list_before_excludes_newer_checkpoints(self):
        config = {"configurable": {"thread_id": "t5"}}
        with lgd.SimpleSqliteSaver(self.db_path) as saver:
            app = self._build_counter_app(saver)
            app.invoke({"count": 0}, config=config)
            latest = list(saver.list(config, limit=1))[0]
            older = list(saver.list(config, before=latest.config))
        self.assertNotIn(latest.checkpoint["id"], [t.checkpoint["id"] for t in older])


class ResearchStrategyTests(unittest.TestCase):
    def test_invokes_subagent_and_returns_content(self):
        fake_result = {"messages": [AIMessage(content="LRU explanation")]}
        with patch.object(lgd.subagent, "invoke", return_value=fake_result) as mock_invoke:
            result = lgd.research_strategy("LRU")
        self.assertEqual(result, "LRU explanation")
        sent_messages = mock_invoke.call_args[0][0]["messages"]
        self.assertIn("LRU", sent_messages[-1].content)


class InitPlanTests(unittest.TestCase):
    def test_creates_one_pending_entry_per_strategy(self):
        result = lgd.init_plan({"messages": [], "plan": [], "notes": {}})
        self.assertEqual(
            result["plan"],
            [{"task": s, "status": "pending"} for s in lgd.STRATEGIES],
        )


class DelegateTests(unittest.TestCase):
    def test_researches_all_pending_tasks(self):
        state = {
            "messages": [],
            "plan": [{"task": s, "status": "pending"} for s in lgd.STRATEGIES],
            "notes": {},
        }
        with patch.object(
            lgd, "research_strategy", side_effect=lambda s: f"about {s}"
        ) as mock_research:
            result = lgd.delegate(state)
        self.assertEqual(mock_research.call_count, len(lgd.STRATEGIES))
        self.assertTrue(all(p["status"] == "done" for p in result["plan"]))
        self.assertEqual(result["notes"], {s: f"about {s}" for s in lgd.STRATEGIES})

    def test_skips_already_done_tasks(self):
        state = {
            "messages": [],
            "plan": [
                {"task": "LRU", "status": "done"},
                {"task": "LFU", "status": "pending"},
                {"task": "write-through", "status": "pending"},
            ],
            "notes": {"LRU": "already researched"},
        }
        with patch.object(
            lgd, "research_strategy", side_effect=lambda s: f"about {s}"
        ) as mock_research:
            result = lgd.delegate(state)
        self.assertEqual(mock_research.call_count, 2)
        mock_research.assert_any_call("LFU")
        mock_research.assert_any_call("write-through")
        self.assertEqual(result["notes"]["LRU"], "already researched")
        self.assertEqual(result["notes"]["LFU"], "about LFU")
        self.assertTrue(all(p["status"] == "done" for p in result["plan"]))


class SummarizeTests(unittest.TestCase):
    def test_invokes_llm_with_notes_and_returns_message(self):
        fake_response = AIMessage(content="comparison text")
        state = {"messages": [], "plan": [], "notes": {"LRU": "lru info", "LFU": "lfu info"}}
        with patch.object(type(lgd.llm), "invoke", return_value=fake_response) as mock_invoke:
            result = lgd.summarize(state)
        self.assertEqual(result["messages"], [fake_response])
        prompt = mock_invoke.call_args[0][0][0].content
        self.assertIn("lru info", prompt)
        self.assertIn("lfu info", prompt)


class FullGraphIntegrationTests(TempDbTestCase):
    def test_full_run_then_resume_without_rework(self):
        config = {"configurable": {"thread_id": "test-thread"}}
        fake_answer = AIMessage(content="final comparison")

        with patch.object(
            lgd, "research_strategy", side_effect=lambda s: f"about {s}"
        ) as mock_research, patch.object(type(lgd.llm), "invoke", return_value=fake_answer):
            with lgd.SimpleSqliteSaver(self.db_path) as saver:
                app = lgd.builder.compile(checkpointer=saver)
                result = app.invoke({"messages": [], "plan": [], "notes": {}}, config=config)

        self.assertEqual(mock_research.call_count, len(lgd.STRATEGIES))
        self.assertEqual(result["messages"][-1].content, "final comparison")
        self.assertTrue(all(p["status"] == "done" for p in result["plan"]))

        with lgd.SimpleSqliteSaver(self.db_path) as saver2:
            app2 = lgd.builder.compile(checkpointer=saver2)
            state = app2.get_state(config)
        self.assertEqual(set(state.values["notes"]), set(lgd.STRATEGIES))


class MainTests(TempDbTestCase):
    def test_first_run_executes_then_second_run_resumes_without_rework(self):
        fake_answer = AIMessage(content="final comparison")
        with patch.object(lgd, "DB_PATH", self.db_path), \
             patch.object(
                 lgd, "research_strategy", side_effect=lambda s: f"about {s}"
             ) as mock_research, \
             patch.object(type(lgd.llm), "invoke", return_value=fake_answer):
            lgd.main()
            self.assertEqual(mock_research.call_count, len(lgd.STRATEGIES))

            lgd.main()
            self.assertEqual(mock_research.call_count, len(lgd.STRATEGIES))


if __name__ == "__main__":
    unittest.main()
