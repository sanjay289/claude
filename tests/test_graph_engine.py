#!/usr/bin/env python3
"""Tests for graph_engine.py's Graph runner.

Run with: python3 -m unittest discover tests
"""

import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_engine import Graph, END


class LinearRunTests(unittest.TestCase):
    def test_runs_nodes_in_order_and_returns_final_state(self):
        graph = (
            Graph()
            .add_node("a", lambda state: {"a_ran": True})
            .add_node("b", lambda state: {"b_ran": True})
            .set_entry("a")
            .add_edge("a", "b")
            .add_edge("b", END)
        )
        result = graph.run({"x": 1})
        self.assertEqual(result, {"x": 1, "a_ran": True, "b_ran": True})

    def test_in_place_mutation_is_preserved(self):
        def mutate(state):
            state["seen"] = True

        graph = Graph().add_node("a", mutate).set_entry("a").add_edge("a", END)
        result = graph.run({})
        self.assertEqual(result, {"seen": True})

    def test_returned_update_wins_over_in_place_mutation(self):
        def conflicting(state):
            state["key"] = "from mutation"
            return {"key": "from return"}

        graph = Graph().add_node("a", conflicting).set_entry("a").add_edge("a", END)
        result = graph.run({})
        self.assertEqual(result["key"], "from return")

    def test_initial_state_is_not_mutated(self):
        initial = {"x": 1}
        graph = (
            Graph()
            .add_node("a", lambda state: {"x": 2})
            .set_entry("a")
            .add_edge("a", END)
        )
        graph.run(initial)
        self.assertEqual(initial, {"x": 1})


class ConditionalEdgeTests(unittest.TestCase):
    def test_routes_to_mapped_node(self):
        graph = (
            Graph()
            .add_node("start", lambda state: None)
            .add_node("left", lambda state: {"branch": "left"})
            .add_node("right", lambda state: {"branch": "right"})
            .set_entry("start")
            .add_conditional_edges(
                "start", lambda state: "go_left", {"go_left": "left", "go_right": "right"}
            )
            .add_edge("left", END)
            .add_edge("right", END)
        )
        self.assertEqual(graph.run({})["branch"], "left")

    def test_unmapped_router_key_raises(self):
        graph = (
            Graph()
            .add_node("start", lambda state: None)
            .add_node("left", lambda state: None)
            .set_entry("start")
            .add_conditional_edges("start", lambda state: "nope", {"go_left": "left"})
            .add_edge("left", END)
        )
        with self.assertRaises(ValueError):
            graph.run({})


class FanOutTests(unittest.TestCase):
    def test_collects_branch_results_keyed_by_name(self):
        def make_branch(name):
            return lambda state: f"{name}-result"

        graph = (
            Graph()
            .add_node("start", lambda state: None)
            .add_node("branch_a", make_branch("a"))
            .add_node("branch_b", make_branch("b"))
            .add_node("join", lambda state: None)
            .set_entry("start")
            .add_fan_out("start", ["branch_a", "branch_b"], "join")
            .add_edge("join", END)
        )
        result = graph.run({})
        self.assertEqual(
            result["branch_results"], {"branch_a": "a-result", "branch_b": "b-result"}
        )

    def test_uses_custom_result_key(self):
        graph = (
            Graph()
            .add_node("start", lambda state: None)
            .add_node("branch_a", lambda state: "a")
            .add_node("join", lambda state: None)
            .set_entry("start")
            .add_fan_out("start", ["branch_a"], "join", result_key="custom")
            .add_edge("join", END)
        )
        result = graph.run({})
        self.assertEqual(result["custom"], {"branch_a": "a"})

    def test_branches_run_on_isolated_deep_copies(self):
        def mutating_branch(state):
            state["nested"]["touched"] = True
            return None

        graph = (
            Graph()
            .add_node("start", lambda state: None)
            .add_node("branch_a", mutating_branch)
            .add_node("join", lambda state: None)
            .set_entry("start")
            .add_fan_out("start", ["branch_a"], "join")
            .add_edge("join", END)
        )
        result = graph.run({"nested": {}})
        self.assertNotIn("touched", result["nested"])


class ErrorHandlingTests(unittest.TestCase):
    def test_run_without_entry_raises(self):
        with self.assertRaises(ValueError):
            Graph().run({})

    def test_unknown_node_raises(self):
        graph = Graph().set_entry("missing")
        with self.assertRaises(ValueError):
            graph.run({})

    def test_node_without_outgoing_edge_raises(self):
        graph = Graph().add_node("a", lambda state: None).set_entry("a")
        with self.assertRaises(ValueError):
            graph.run({})

    def test_unknown_edge_kind_raises(self):
        graph = Graph().add_node("a", lambda state: None).set_entry("a")
        graph._edges["a"] = ("bogus_kind",)
        with self.assertRaises(ValueError):
            graph.run({})


class VerboseModeTests(unittest.TestCase):
    def test_verbose_prints_node_names(self):
        graph = Graph().add_node("a", lambda state: None).set_entry("a").add_edge("a", END)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            graph.run({}, verbose=True)
        self.assertIn("[graph] -> a", out.getvalue())


if __name__ == "__main__":
    unittest.main()
