# claude

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Built with Ollama](https://img.shields.io/badge/built%20with-Ollama-000000.svg)](https://ollama.com)

Small Ollama-backed agent experiments, built on a shared dependency-free graph engine.

## Components

- **`graph_engine.py`** — a minimal node/edge graph runner for agent workflows.
  A `Graph` is nodes (`state -> state-update`) connected by edges: plain,
  conditional (branch on a router function), or fan-out (run several nodes in
  parallel on isolated copies of state, then join into one node with the
  per-branch results collected). No dependencies beyond the standard library.

- **`hermes_agent.py`** — a tool-using chat agent against a local/cloud
  `hermes3` model via [Ollama](https://ollama.com). Built as a graph: call
  the model, route to real tool execution or a bounded retry when the model
  writes a tool call as plain text instead of using the tool-calling
  interface, loop until it produces a final answer or hits the iteration
  cap. Tools include shell execution, file read/write, and a skill hub
  (`skills/*.md`) the model can discover and load before doing a task a
  skill covers.

- **`council.py`** — parallel multi-model deliberation. Several member
  models answer the same question independently and concurrently (a
  fan-out), then a separate chair model reads every answer and synthesizes
  a verdict, noting agreement and disagreement across the council.

- **`skills/`** — markdown skill files (`code-review.md`, `shell-safety.md`,
  `summarize.md`) that `hermes_agent.py` can load on demand via its
  `list_skills` / `use_skill` tools.

- **`tests/`** — unit tests for the agent loop, with `ollama.chat` stubbed
  out so they run offline (no server, no model pulled).

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) running locally, with the models referenced
  in `hermes_agent.py` (`hermes3`) and `council.py` (`MEMBERS`, `CHAIR`)
  pulled or available as cloud models
- The `ollama` Python package (`pip install ollama`)

## Running

```bash
python3 hermes_agent.py   # interactive tool-using chat agent
python3 council.py        # interactive multi-model council
```

## Testing

```bash
python3 -m unittest discover tests
```
