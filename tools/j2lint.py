#!/usr/bin/env python3
"""Lint Jinja2 (.j2) template files.

Checks:
- Jinja2 syntax (via jinja2.Environment.parse)
- Leading whitespace on each non-blank line is a multiple of INDENT_SIZE

Report-only: never modifies files. Exits non-zero on any violation.
"""

import sys
from pathlib import Path

import jinja2

INDENT_SIZE = 4


def lint_file(path: Path) -> int:
    """Return the number of violations found in `path`."""
    text = path.read_text()
    errors = 0

    try:
        jinja2.Environment().parse(text)
    except jinja2.TemplateSyntaxError as e:
        print(f"{path}:{e.lineno}: jinja2 syntax error: {e.message}", file=sys.stderr)
        errors += 1

    # Track multi-line Jinja `{# ... #}` comment blocks so their contents
    # are exempt from the indent rule (free-form prose, not yaml content).
    in_comment_block = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if in_comment_block:
            if "#}" in line:
                in_comment_block = False
            continue

        stripped = line.lstrip(" ")
        if not stripped:
            continue

        # A `{#` that isn't closed on the same line opens a comment block.
        if "{#" in line and "#}" not in line.split("{#", 1)[1]:
            in_comment_block = True
            continue
        # Single-line `{# ... #}` comment: skip indent check on it.
        if stripped.startswith("{#") and stripped.rstrip().endswith("#}"):
            continue

        leading = len(line) - len(stripped)
        if leading > 0 and leading % INDENT_SIZE != 0:
            print(
                f"{path}:{lineno}: leading indent {leading} is not a multiple of {INDENT_SIZE}",
                file=sys.stderr,
            )
            errors += 1

    return errors


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: j2lint.py <file.j2> [<file.j2> ...]", file=sys.stderr)
        return 2

    total = 0
    for arg in argv:
        total += lint_file(Path(arg))

    if total:
        print(f"\n{total} issue(s) found in {len(argv)} file(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
