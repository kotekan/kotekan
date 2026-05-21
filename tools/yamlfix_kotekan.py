#!/usr/bin/env python3
"""Round-trip yaml via ruamel.yaml: fixes indent and adds `---`.

Raises on duplicate keys / syntax errors. A pre-pass relocates any
existing `---` to line 1 (or inserts one if missing) so leading comments
end up between `---` and the first key, where ruamel preserves them.
"""

import sys
from io import StringIO
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

EXCLUDED_DIRS = {"build", "external", ".venv", "scratch", "julia", ".git"}
EXCLUDED_PREFIXES = ("build-",)
EXCLUDED_PATHS = ("lib/cuda/generated", ".github/workflows")


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
    yaml.width = 2**31 - 1  # disable width-based wrapping

    errors = 0
    fixed = 0
    for ext in ("*.yaml", "*.yml"):
        for path in sorted(root.rglob(ext)):
            if not path.is_file() or excluded(path.relative_to(root)):
                continue
            src = path.read_text()
            staged = ensure_doc_start_at_top(src)
            try:
                data = yaml.load(staged)
            except YAMLError as e:
                print(f"{path}: {type(e).__name__}: {e}", file=sys.stderr)
                errors += 1
                continue
            buf = StringIO()
            yaml.dump(data, buf)
            out = buf.getvalue()
            if out != src:
                path.write_text(out)
                fixed += 1

    if fixed:
        print(f"yamlfix_kotekan: reformatted {fixed} file(s)", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
