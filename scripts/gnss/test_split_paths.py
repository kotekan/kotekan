#!/usr/bin/env python3
"""split_paths.txt keeps every repo file the manifest and the chains config name.

    python3 scripts/gnss/test_split_paths.py

scripts/gnss/split_paths.txt is the git filter-repo --paths-from-file list that extracts
gnss-broker from kotekan. A config file the generator or the broker reads and the list leaves
out is missing from gnss-broker, and nothing says so until a run there fails (10-23:
config/elem_shared_ref_20261023_repoint.json, which the manifest's elem-shared-ref names).

The list is read as filter-repo reads it: '#' lines and blank lines ignored; plain lines are
literal path prefixes (a directory or an exact file); 'regex:' lines use re.search; 'glob:'
lines use fnmatch, where '*' also matches '/'; a line with '==>' renames, and the lines after
it test the renamed path. A path is kept when any filter line matches it.

The names checked are the repo-relative paths (config/..., python/..., scripts/...) in the
string values of config/gnss_fleet_chord.yaml and config/gnss_chains_chord.yaml that are
tracked files, and the fixed paths gnss_arraymap.py reads. Comments are not values, and a
generated or untracked file is not in the history the split rewrites.

The list is append-only from the commit that says so: filter-repo filters every commit with
the list as it is now, so a removed or reordered line changes the extraction of earlier
commits and the next push of gnss-broker stops fast-forwarding (10-25, 211a18037). Every
filter line of that commit must still be here, in the same order.
"""
import fnmatch
import os
import re
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
PATHS = os.path.join(HERE, "split_paths.txt")
CONFIGS = ["config/gnss_fleet_chord.yaml", "config/gnss_chains_chord.yaml"]
APPEND_ONLY = "APPEND-ONLY from here on"
PATH_RE = re.compile(r"(?<![\w/.])((?:config|python|scripts)/[\w./-]+)")

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def parse(text):
    """[(kind, pattern, replacement)] in file order; kind is literal, regex or glob, and
    replacement is None on a filter line."""
    rules = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        kind = "literal"
        for k in ("regex", "glob"):
            if line.startswith(k + ":"):
                kind, line = k, line[len(k) + 1 :]
        repl = None
        if "==>" in line:
            line, repl = line.split("==>", 1)
        rules.append((kind, line, repl))
    return rules


def _match(kind, pat, name):
    if kind == "regex":
        return re.search(pat, name) is not None
    if kind == "glob":
        return fnmatch.fnmatchcase(name, pat)
    return name == pat or name.startswith(pat if pat.endswith("/") else pat + "/")


def kept(rules, path):
    name, keep = path, False
    for kind, pat, repl in rules:
        if repl is None:
            keep = keep or _match(kind, pat, name)
        elif kind == "regex":
            name = re.sub(pat, repl, name)
        elif _match(kind, pat, name):
            name = repl + name[len(pat) :]
    return keep


def tracked():
    """The tracked files, or None outside a git checkout (then a file on disk counts)."""
    try:
        out = subprocess.run(
            ["git", "-C", REPO, "ls-files", "-z"],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return set(out.decode().split("\0")) - {""}


def append_only_base():
    """The list as the commit that made it append-only left it, or None outside a git
    checkout. The commit is found by the marker line, so the same check runs in gnss-broker,
    whose SHAs differ."""
    try:
        sha = subprocess.run(
            ["git", "-C", REPO, "log", "--reverse", "--format=%H", "-S", APPEND_ONLY]
            + ["--", "scripts/gnss/split_paths.txt"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.split()
        if not sha:
            return ""
        return subprocess.run(
            ["git", "-C", REPO, "show", sha[0] + ":scripts/gnss/split_paths.txt"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def in_order(old, new):
    """The first of old's lines missing from new taken in order, or None if old is a
    subsequence of new."""
    i = 0
    for line in old:
        while i < len(new) and new[i] != line:
            i += 1
        if i == len(new):
            return line
        i += 1
    return None


def strings(node):
    if isinstance(node, dict):
        for v in node.values():
            yield from strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from strings(v)
    elif isinstance(node, str):
        yield node


def referenced(files):
    out = {}
    for cfg in CONFIGS:
        with open(os.path.join(REPO, cfg)) as fh:
            doc = yaml.safe_load(fh)
        for s in strings(doc):
            for p in PATH_RE.findall(s):
                p = p.rstrip(".")
                ok = (p in files) if files is not None else os.path.isfile(
                    os.path.join(REPO, p)
                )
                if ok:
                    out.setdefault(p, cfg)
    return out


def main():
    with open(PATHS) as fh:
        rules = parse(fh.read())

    print("the parser reads the list as filter-repo does")
    for path, want in (
        ("python/scripts/gnss/gnss_broker/cli.py", True),  # regex
        ("python/scripts/gnss/l2oc_code_check.py", False),  # regex lookahead
        ("scripts/gnss/trim_sign_gate.py", False),  # stays in kotekan
        ("scripts/gnss/split_paths.txt", True),  # the list lists itself
        ("config/gnss_fleet_chord.yaml", True),  # literal file
        ("config/gnss_fleet_chord.yaml.bak", False),  # a literal is not a string prefix
        ("config/base/x.yaml", True),  # literal directory
        ("config/chord/pathfinder.j2", False),  # the stock template stays in kotekan
        ("docs/CHORD_LIVE_PRN_RECONFIG.md", True),  # glob
    ):
        check(kept(rules, path) == want, "%s %s" % (path, "kept" if want else "dropped"))
    check(
        any(r[2] is not None for r in rules) and rules[-1][2] is not None,
        "the rename line is parsed, and it is the last line",
    )

    print("\nthe list only grows after the append-only marker")
    with open(PATHS) as fh:
        now = [ln for ln in fh.read().splitlines() if ln.strip() and ln[0] != "#"]
    base = append_only_base()
    if base is None:
        print("  [SKIP] not a git checkout: no history to compare with")
    else:
        old = [ln for ln in base.splitlines() if ln.strip() and ln[0] != "#"]
        check(bool(old), "the commit that wrote %r is in the history" % APPEND_ONLY)
        lost = in_order(old, now)
        check(
            lost is None,
            "its %d filter lines are all here, in order%s"
            % (len(old), "" if lost is None else " (first missing: %s)" % lost),
        )
    check("==>" in now[-1], "the rename line is the file's last line")

    print("\nthe fixed paths gnss_arraymap.py reads are kept")
    sys.path.insert(0, os.path.join(REPO, "python", "scripts", "gnss"))
    import gnss_arraymap

    for path in (
        gnss_arraymap.CONFIG,
        gnss_arraymap.NODE_CONFIG,
        gnss_arraymap.BAD_INPUTS,
    ):
        rel = os.path.relpath(path, REPO)
        check(kept(rules, rel), rel)

    print("\nevery file the manifest and the chains config name is kept")
    refs = referenced(tracked())
    check(bool(refs), "the configs name at least one tracked file (%d)" % len(refs))
    for p, cfg in sorted(refs.items()):
        check(kept(rules, p), "%s (named in %s)" % (p, cfg))

    print("\n%s (%d check(s) failed)" % ("FAIL" if _fails else "PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
