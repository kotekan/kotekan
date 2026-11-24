#!/usr/bin/env python3
"""Render each .j2 with an empty context and require valid yaml output.

Mirrors python/kotekan/config.py: FileSystemLoader rooted at the template's
parent so `{% include %}` resolves, default Undefined behaviour. Also rejects
duplicate keys in the rendered output (via ruamel, matching yamlfix_kotekan.py;
the runtime PyYAML loader silently keeps the last duplicate). Report-only.
"""

import sys
from pathlib import Path

import jinja2
import yaml
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError as RuamelYAMLError


def lint_file(path: Path) -> int:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(path.parent)),
        autoescape=jinja2.select_autoescape(),
    )
    try:
        rendered = env.get_template(path.name).render({})
    except jinja2.TemplateError as e:
        print(f"{path}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    try:
        yaml.safe_load(rendered)
    except yaml.YAMLError as e:
        print(f"{path}: rendered output is not valid yaml: {e}", file=sys.stderr)
        return 1
    try:
        YAML().load(rendered)
    except RuamelYAMLError as e:
        print(f"{path}: rendered output: {e}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: j2lint.py <file.j2> [<file.j2> ...]", file=sys.stderr)
        return 2
    errors = sum(lint_file(Path(p)) for p in argv)
    if errors:
        print(f"\n{errors} issue(s) found in {len(argv)} file(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
