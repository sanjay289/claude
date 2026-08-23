#!/usr/bin/env python3
"""Minimal dependency-free graph engine for agent workflows.

A Graph is nodes (state -> state-update) connected by edges. Edges can be
plain, conditional (branch on a router function), or fan-out (run several
nodes in parallel on copies of the state, then join into one node with the
per-branch results collected).

State is a single dict threaded through the whole run. A regular node
returns a dict of updates that gets merged into that state. A fan-out
branch's return value is *not* merged automatically — it's collected into
state[result_key] keyed by branch name, since branches run concurrently and
merging into shared state directly would race.
"""

import concurrent.futures

END = "__END__"


class Graph:
    def __init__(self):
        self._nodes = {}
        self._edges = {}
        self._entry = None

    def add_node(self, name: str, fn) -> "Graph":
        self._nodes[name] = fn
        return self

    def set_entry(self, name: str) -> "Graph":
        self._entry = name
        return self

    def add_edge(self, from_name: str, to_name: str) -> "Graph":
        self._edges[from_name] = ("edge", to_name)
        return self

    def add_conditional_edges(self, from_name: str, router, mapping: dict) -> "Graph":
        """router(state) -> key; mapping[key] -> next node name (or END)."""
        self._edges[from_name] = ("cond", router, mapping)
        return self

    def add_fan_out(self, from_name: str, branches: list, join_name: str,
                     result_key: str = "branch_results", max_workers: int = None) -> "Graph":
        """Run each node in `branches` in parallel on its own copy of state,
        then continue at `join_name` with state[result_key] = {branch: result}."""
        self._edges[from_name] = ("fan_out", branches, join_name, result_key, max_workers)
        return self

    def run(self, initial_state: dict, verbose: bool = False) -> dict:
        if self._entry is None:
            raise ValueError("Graph has no entry point; call set_entry() first")

        state = dict(initial_state)
        current = self._entry

        while current != END:
            if current not in self._nodes:
                raise ValueError(f"Unknown node: {current!r}")
            if verbose:
                print(f"[graph] -> {current}")

            fn = self._nodes[current]
            update = fn(state)
            if update:
                state.update(update)

            if current not in self._edges:
                raise ValueError(
                    f"Node {current!r} has no outgoing edge; "
                    f"add an edge to another node or to END"
                )
            edge = self._edges[current]
            kind = edge[0]

            if kind == "edge":
                current = edge[1]
            elif kind == "cond":
                _, router, mapping = edge
                key = router(state)
                if key not in mapping:
                    raise ValueError(f"Router at {current!r} returned unmapped key: {key!r}")
                current = mapping[key]
            elif kind == "fan_out":
                _, branches, join_name, result_key, max_workers = edge
                results = self._run_fan_out(branches, state, max_workers)
                state[result_key] = results
                current = join_name
            else:
                raise ValueError(f"Unknown edge kind: {kind!r}")

        return state

    def _run_fan_out(self, branches: list, state: dict, max_workers: int) -> dict:
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers or len(branches)) as pool:
            futures = {
                pool.submit(self._nodes[b], dict(state)): b for b in branches
            }
            for future in concurrent.futures.as_completed(futures):
                branch = futures[future]
                results[branch] = future.result()
        return results
