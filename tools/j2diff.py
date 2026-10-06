#!/usr/bin/env python3
"""Render two config templates the way kotekan does and compare the parsed dicts.

For config refactors that must not change what kotekan sees, e.g.

    git show develop:config/chord_pathfinder.j2 > /tmp/old/chord_pathfinder.j2
    PYTHONPATH=python tools/j2diff.py /tmp/old/chord_pathfinder.j2 config/chord/pathfinder.j2

(the old file goes in its own directory so its includes, if any, resolve there).

Renders with kotekan.config.render_jinja, but rejects duplicate keys where kotekan keeps
the last.  Exit 0 if the parsed dicts are equal, 1 with a key-sorted JSON diff otherwise.
"""
import difflib
import json
import sys

import yaml
from kotekan.config import render_jinja


class _NoDupLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of keeping the last."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for k_node, _ in node.value:
            k = self.construct_object(k_node, deep=deep)
            if k in seen:
                raise yaml.YAMLError(f"duplicate key {k!r} at {k_node.start_mark}")
            seen.add(k)
        return super().construct_mapping(node, deep)


def render(path: str) -> dict:
    return yaml.load(render_jinja(path), Loader=_NoDupLoader)


def main(a: str, b: str) -> int:
    da, db = render(a), render(b)
    if da == db:
        print(f"IDENTICAL: {a} == {b} ({len(da)} top-level keys)")
        return 0
    ja = json.dumps(da, indent=1, sort_keys=True).splitlines()
    jb = json.dumps(db, indent=1, sort_keys=True).splitlines()
    for line in difflib.unified_diff(ja, jb, a, b, lineterm="", n=2):
        print(line)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
