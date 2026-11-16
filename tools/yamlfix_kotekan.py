#!/usr/bin/env python3
"""Round-trip yaml via ruamel.yaml: fixes indent and adds `---`.

Raises on duplicate keys / syntax errors. A pre-pass relocates `---` to
line 1. Regions wrapped in `# yamlfix:off` / `# yamlfix:on` comments are
preserved byte-for-byte (useful for hand-laid tables).
"""

import sys
from io import StringIO
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

EXCLUDED_DIRS = {"build", "external", ".venv", "scratch", "julia", ".git"}
EXCLUDED_PREFIXES = ("build-",)
EXCLUDED_PATHS = ("lib/cuda/generated", ".github/workflows")

OFF_MARKER = "# yamlfix:off"
ON_MARKER = "# yamlfix:on"


def excluded(rel: Path) -> bool:
    for part in rel.parts:
        if part in EXCLUDED_DIRS or any(part.startswith(p) for p in EXCLUDED_PREFIXES):
            return True
    rel_str = str(rel)
    return any(rel_str == ep or rel_str.startswith(ep + "/") for ep in EXCLUDED_PATHS)


def ensure_doc_start_at_top(text: str) -> str:
    """Move `---` to line 1 (or insert if missing). Multi-doc yaml is
    left alone."""
    if not text.strip():
        return text
    lines = text.splitlines(keepends=True)
    dashes = [i for i, line in enumerate(lines) if line.rstrip() == "---"]
    if len(dashes) > 1:
        return text
    if not dashes:
        return "---\n" + text
    if dashes[0] == 0:
        return text
    del lines[dashes[0]]
    return "---\n" + "".join(lines)


def extract_protected_regions(text: str) -> list[str]:
    """Return the verbatim text (including markers) of each
    `# yamlfix:off` ... `# yamlfix:on` region, in source order. Raises
    on stray `:on`, nested `:off`, or unterminated `:off`."""
    lines = text.splitlines(keepends=True)
    regions = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == OFF_MARKER:
            start = i
            j = i + 1
            while j < len(lines):
                s2 = lines[j].strip()
                if s2 == OFF_MARKER:
                    raise ValueError(f"nested {OFF_MARKER} at line {j + 1}")
                if s2 == ON_MARKER:
                    regions.append("".join(lines[start : j + 1]))
                    i = j + 1
                    break
                j += 1
            else:
                raise ValueError(f"unterminated {OFF_MARKER} at line {start + 1}")
        elif s == ON_MARKER:
            raise ValueError(
                f"stray {ON_MARKER} at line {i + 1} (no preceding {OFF_MARKER})"
            )
        else:
            i += 1
    return regions


def _shift_block(text: str, shift: int) -> str:
    """Add `shift` leading spaces to each line (or strip them if shift < 0)."""
    if shift == 0:
        return text
    if shift > 0:
        pad = " " * shift
        return "".join(pad + line for line in text.splitlines(keepends=True))
    n = -shift
    return "".join(
        (line[n:] if line.startswith(" " * n) else line)
        for line in text.splitlines(keepends=True)
    )


def _structural_indent_at(lines: list[str], idx: int, direction: int) -> int | None:
    """Walk in `direction` (+1 or -1) from `idx`, return indent of first
    non-empty, non-comment line. Returns None if none found."""
    step = direction
    j = idx + step
    while 0 <= j < len(lines):
        stripped = lines[j].strip()
        if stripped and not stripped.startswith("#"):
            return len(lines[j]) - len(lines[j].lstrip(" "))
        j += step
    return None


def _expected_indent(lines: list[str], off_idx: int, on_idx: int, current: int) -> int:
    """Pick the smallest of {before, after} candidates that is >= current.
    Falls back to whichever is available, or 0 if none. The reformat-
    increases-indent assumption means the right answer is the smallest
    indent at or above the protected block's existing indent."""
    before = _structural_indent_at(lines, off_idx, -1)
    after = _structural_indent_at(lines, on_idx, +1)
    candidates = [x for x in (before, after) if x is not None and x >= current]
    if candidates:
        return min(candidates)
    return next((x for x in (before, after) if x is not None), 0)


def restore_protected_regions(text: str, originals: list[str]) -> str:
    """Replace each `# yamlfix:off`..`# yamlfix:on` block with its
    matching `originals` entry, re-indenting it to match the protected
    block's sibling context (looking both before the off and after the on
    marker in ruamel's output). Without this, ruamel's re-indent of
    mapping keys would leave protected blocks at the old shallower indent
    and break the yaml structure (ruamel doesn't re-indent comments)."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    it = iter(originals)
    i = 0
    while i < len(lines):
        if lines[i].strip() == OFF_MARKER:
            j = i + 1
            while j < len(lines) and lines[j].strip() != ON_MARKER:
                j += 1
            if j >= len(lines):
                # ruamel ate the closing marker; keep what we have.
                out.append(lines[i])
                i += 1
                continue
            try:
                original = next(it)
            except StopIteration:
                original = "".join(lines[i : j + 1])
            orig_lines = original.splitlines(keepends=True)
            if orig_lines:
                orig_indent = len(orig_lines[0]) - len(orig_lines[0].lstrip(" "))
                target = _expected_indent(lines, i, j, orig_indent)
                original = _shift_block(original, target - orig_indent)
            out.append(original)
            i = j + 1
        else:
            out.append(lines[i])
            i += 1
    return "".join(out)


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: yamlfix_kotekan.py <root>", file=sys.stderr)
        return 2

    root = Path(argv[0]).resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.indent(mapping=4, sequence=6, offset=4)
    yaml.explicit_start = True
    yaml.width = 2 ** 31 - 1  # disable width-based wrapping

    errors = 0
    fixed = 0
    for ext in ("*.yaml", "*.yml"):
        for path in sorted(root.rglob(ext)):
            if not path.is_file() or excluded(path.relative_to(root)):
                continue
            src = path.read_text()
            staged = ensure_doc_start_at_top(src)
            try:
                protected = extract_protected_regions(staged)
            except ValueError as e:
                print(f"{path}: marker error: {e}", file=sys.stderr)
                errors += 1
                continue
            try:
                data = yaml.load(staged)
            except YAMLError as e:
                print(f"{path}: {type(e).__name__}: {e}", file=sys.stderr)
                errors += 1
                continue
            buf = StringIO()
            yaml.dump(data, buf)
            out = restore_protected_regions(buf.getvalue(), protected)
            if out != src:
                path.write_text(out)
                fixed += 1

    if fixed:
        print(f"yamlfix_kotekan: reformatted {fixed} file(s)", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
