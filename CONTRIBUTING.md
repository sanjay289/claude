# Contributing

## Setup

```bash
pip install -r requirements.txt
```

Running the agents (`hermes_agent.py`, `council.py`) also needs
[Ollama](https://ollama.com) installed and running locally, with the models
they reference pulled or available as cloud models. Tests don't need any of
that — `ollama.chat` is stubbed out so `tests/` runs fully offline.

## Running tests

```bash
python3 -m unittest discover tests -v
```

CI (`.github/workflows/tests.yml`) runs the same suite on Python 3.9 and
3.13 for every push and PR against `main`/`master`.

## Making changes

- Keep `graph_engine.py` dependency-free (standard library only) — it's meant
  to be a small, reusable primitive that `hermes_agent.py` and `council.py`
  both build on.
- Add or update tests under `tests/` for any behavior change, following the
  existing pattern in `test_hermes_agent.py` of stubbing `ollama.chat` with a
  fake response sequence rather than hitting a real model.
- To add a new skill for `hermes_agent.py`'s skill hub, drop a markdown file
  into `skills/`; it's picked up automatically via `list_skills`/`use_skill`.
- Match the existing style: no docstrings/comments beyond what's needed to
  explain a non-obvious decision.

## Pull requests

Fill out the PR template's checklist — CI must be green
(`python3 -m unittest discover tests`) before merging.
