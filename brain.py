#!/usr/bin/env python3
"""brain: capture, search, and Q&A over a markdown second-brain vault.

  brain add TEXT    append a timestamped line to Inbox/<today>.md
  brain find TERM   search notes with ripgrep; pick one with fzf and open it
  brain ask QUESTION  answer from your notes with a local Ollama model (RAG)

The vault is a plain folder of .md files (Obsidian-compatible) at
$BRAIN_VAULT, defaulting to shared storage so the Obsidian app can see it.
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import ollama

from council import ensure_server_running

VAULT = Path(
    os.environ.get("BRAIN_VAULT", "~/storage/shared/Documents/SecondBrain")
).expanduser()
FOLDERS = ["Inbox", "Projects", "Areas", "Resources", "Archives"]
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"
TOP_K = 5
SYSTEM_PROMPT = (
    "Answer the question using only the notes provided. Cite each note you "
    "use as [[Note Name]]. If the notes don't contain the answer, say so "
    "instead of guessing."
)
# llama3.2:3b ignores the citation instruction unless reminded with an example.
CITE_HINT = 'End your answer with the [[Note Name]] it came from, e.g. "...on Friday [[Calendar]]."'


def add(text, now=None):
    now = now or datetime.now()
    for folder in FOLDERS:
        (VAULT / folder).mkdir(parents=True, exist_ok=True)
    note = VAULT / "Inbox" / f"{now:%Y-%m-%d}.md"
    with note.open("a") as fh:
        fh.write(f"- {now:%H:%M} {text}\n")
    return note


def find(term):
    if not VAULT.is_dir():
        return []
    result = subprocess.run(
        ["rg", "--line-number", "--no-heading", "--ignore-case",
         "--glob", "*.md", "--", term, "."],
        cwd=VAULT, capture_output=True, text=True,
    )
    return result.stdout.splitlines()


def embed(texts):
    return ollama.embed(model=EMBED_MODEL, input=texts).embeddings


def load_index():
    """Return [(note name, chunk text, vector)], re-embedding only changed notes.

    Chunks are blank-line-separated paragraphs. Vectors are cached in
    .brain/cache.json, keyed by note path and content hash; Obsidian hides
    dot-folders, so the cache stays out of the vault view.
    """
    cache_path = VAULT / ".brain" / "cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    fresh = {}
    for path in sorted(VAULT.rglob("*.md")):
        rel = path.relative_to(VAULT)
        if rel.parts[0].startswith("."):
            continue
        text = path.read_text(errors="replace")
        digest = hashlib.sha1(text.encode()).hexdigest()
        entry = cache.get(str(rel))
        if not entry or entry["hash"] != digest:
            chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
            vectors = embed(["search_document: " + c for c in chunks]) if chunks else []
            entry = {"hash": digest, "chunks": chunks, "vectors": vectors}
        fresh[str(rel)] = entry
    if fresh != cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(fresh))
    return [
        (Path(rel).stem, chunk, vector)
        for rel, entry in fresh.items()
        for chunk, vector in zip(entry["chunks"], entry["vectors"])
    ]


def retrieve(question, index, k=TOP_K):
    if not index:
        return []
    q = np.array(embed(["search_query: " + question])[0])
    m = np.array([vector for _, _, vector in index])
    scores = m @ q / (np.linalg.norm(m, axis=1) * np.linalg.norm(q))
    return [index[i][:2] for i in np.argsort(-scores)[:k]]


def ask(question):
    hits = retrieve(question, load_index())
    if not hits:
        print("no notes to search")
        return
    notes = "\n\n".join(f"[[{name}]]\n{chunk}" for name, chunk in hits)
    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Notes:\n\n{notes}\n\nQuestion: {question}\n\n{CITE_HINT}"},
        ],
        stream=True,
    )
    for part in stream:
        print(part.message.content, end="", flush=True)
    print()


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) >= 2 and args[0] == "add":
        note = add(" ".join(args[1:]))
        print(f"added to {note.relative_to(VAULT)}")
        return 0
    if len(args) >= 2 and args[0] == "find":
        matches = find(" ".join(args[1:]))
        if not matches:
            print("no matches")
            return 1
        if not sys.stdout.isatty():
            print("\n".join(matches))
            return 0
        picked = subprocess.run(
            ["fzf"], input="\n".join(matches), stdout=subprocess.PIPE, text=True,
        ).stdout.strip()
        if picked:
            path, line = picked.split(":", 2)[:2]
            editor = os.environ.get("EDITOR", "nano")
            subprocess.run([editor, f"+{line}", path], cwd=VAULT)
        return 0
    if len(args) >= 2 and args[0] == "ask":
        ensure_server_running()
        ask(" ".join(args[1:]))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
