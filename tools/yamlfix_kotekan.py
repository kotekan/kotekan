#!/usr/bin/env python3
"""Round-trip yaml via ruamel.yaml: fixes indent and adds `---`.

Raises on duplicate keys / syntax errors. Drops comments above `---`
(ruamel.yaml limitation).
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

    errors = 0
    fixed = 0
    for ext in ("*.yaml", "*.yml"):
        for path in sorted(root.rglob(ext)):
            if not path.is_file() or excluded(path.relative_to(root)):
                continue
            src = path.read_text()
            try:
                data = yaml.load(src)
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
