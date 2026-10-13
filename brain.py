#!/usr/bin/env python3
"""brain: capture and search for a markdown second-brain vault.

  brain add TEXT    append a timestamped line to Inbox/<today>.md
  brain find TERM   search notes with ripgrep; pick one with fzf and open it

The vault is a plain folder of .md files (Obsidian-compatible) at
$BRAIN_VAULT, defaulting to shared storage so the Obsidian app can see it.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

VAULT = Path(
    os.environ.get("BRAIN_VAULT", "~/storage/shared/Documents/SecondBrain")
).expanduser()
FOLDERS = ["Inbox", "Projects", "Areas", "Resources", "Archives"]


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
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
