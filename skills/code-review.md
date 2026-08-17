---
name: code-review
description: Review a file or diff for bugs, unsafe patterns, and unnecessary complexity.
---
When reviewing code:
- Read the whole file with read_file before commenting, not just the diff.
- Flag correctness bugs first (crashes, wrong logic, unhandled errors at boundaries).
- Flag unsafe patterns: shell injection, unvalidated paths, secrets in code.
- Only then note simplification opportunities — don't bury real bugs under style nits.
- For each finding, give the file, line if known, and a concrete failure scenario.
- Do not rewrite the file yourself unless explicitly asked to fix it.
