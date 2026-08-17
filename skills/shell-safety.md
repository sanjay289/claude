---
name: shell-safety
description: Guidelines to follow before calling run_shell for any destructive or irreversible command.
---
Before running a shell command via run_shell:
- Never run rm -rf, git reset --hard, git push --force, or DROP/TRUNCATE without
  first stating what will be destroyed and asking the user to confirm.
- Prefer read-only inspection (ls, cat, git status, git diff) before any mutating command.
- If a command's output could contain secrets (env dumps, cat on credential files),
  do not echo it back verbatim to the user — summarize instead.
- Quote paths and variables; never interpolate raw user input into a shell string.
