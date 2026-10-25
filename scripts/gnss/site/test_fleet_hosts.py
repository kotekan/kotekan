#!/usr/bin/env python3
"""No hand-typed host lists: the scripts take their hosts from config/gnss_fleet.py.

    python3 scripts/gnss/site/test_fleet_hosts.py

The fleet's node list is `nodes:` in config/gnss_fleet_chord.yaml, and the site's hosts are in
config/chord_gnss_node.yaml. Every other file reads them through config/gnss_fleet.py, so adding
or removing a node is one line there. This fails on a code line (comments and docstrings aside)
under scripts/gnss, python/scripts/gnss, config/*.py or config/gnss_chains_chord.yaml that names
two or more distinct cxNN hosts, on a cxNN URL anywhere in a site script, and on a cxNN URL
given as a default elsewhere. Tests, selftests, gates, telem_e2e.py and fixtures keep their
literal hosts. It also checks that the shell and Python defaults fail, saying why, when the
fleet list cannot be read, rather than running over no nodes.
"""
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.realpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "config"))
import gnss_fleet  # noqa: E402

SCAN_DIRS = ("scripts/gnss", "python/scripts/gnss")
SCAN_EXT = (".py", ".sh")
CONFIG_FILES = ("config/gnss_chains_chord.yaml",)
HOST = re.compile(r"\bcx\d\d\b")
URL = re.compile(r"https?://cx\d\d\b")
# A URL as a default value: ${VAR:-URL}, default=URL, os.environ.get(NAME, URL).
DEFAULT_URL = re.compile(
    r"""(:-|:=|default\s*=\s*|\.get\([^,()]*,\s*)["']?https?://cx\d\d\b"""
)

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def exempt(name):
    return (
        (name.startswith("test_") and name.endswith(".py"))
        or "selftest" in name
        or name.endswith("_gate.py")
        or name == "telem_e2e.py"
    )


def scanned_files():
    """Repo-relative paths of the files the rule covers."""
    out = []
    for d in SCAN_DIRS:
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, d)):
            dirnames[:] = [x for x in dirnames if x not in ("__pycache__", "fixtures")]
            for f in filenames:
                if f.endswith(SCAN_EXT) and not exempt(f):
                    out.append(os.path.relpath(os.path.join(dirpath, f), ROOT))
    cfg = os.path.join(ROOT, "config")
    for f in os.listdir(cfg):
        if f.endswith(".py") and not exempt(f):
            out.append(os.path.join("config", f))
    out.extend(CONFIG_FILES)
    return sorted(out)


def python_code_lines(text):
    """[(line number, code)] per logical line, comments dropped and bare string statements
    (docstrings) skipped."""
    out, toks = [], []
    skip = (tokenize.COMMENT, tokenize.NL, tokenize.INDENT, tokenize.DEDENT)
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type in skip:
            continue
        if tok.type in (tokenize.NEWLINE, tokenize.ENDMARKER):
            if toks and not all(t.type == tokenize.STRING for t in toks):
                out.append((toks[0].start[0], " ".join(t.string for t in toks)))
            toks = []
            continue
        toks.append(tok)
    return out


def shell_code_lines(text):
    """[(line number, code)] per line, backslash continuations joined and # comments dropped.
    Also used for yaml, whose comments are the same."""
    out, buf, start = [], "", 0
    for i, line in enumerate(text.split("\n"), 1):
        if not buf:
            start = i
        if line.lstrip().startswith("#"):
            line = ""
        line = re.sub(r"(^|\s)#.*$", "", line)
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        buf += line
        if buf.strip():
            out.append((start, buf))
        buf = ""
    return out


def violations(path, text):
    """['path:line: why'] for one file's text; `path` picks the language and the URL rule."""
    lines = python_code_lines(text) if path.endswith(".py") else shell_code_lines(text)
    site = path.startswith("scripts/gnss/site/")
    out = []
    for n, code in lines:
        hosts = sorted(set(HOST.findall(code)))
        if len(hosts) >= 2:
            out.append("%s:%d: names hosts %s" % (path, n, " ".join(hosts)))
        elif site and URL.search(code):
            out.append("%s:%d: a host URL in a site script" % (path, n))
        elif DEFAULT_URL.search(code):
            out.append("%s:%d: a host URL as a default" % (path, n))
    return out


def test_rule_catches_lists():
    """The rule itself: hand-typed lists and default URLs fail; comments, docstrings and one
    host as an example pass."""
    bad = {
        "scripts/gnss/site/a.sh": 'NODES=${GNSS_NODES:-"cx19 cx27"}\n',
        "scripts/gnss/site/b.sh": "for n in cx19 \\\n    cx27; do :; done\n",
        "scripts/gnss/site/c.sh": "URL=${URL:-http://cx43:12048}\n",
        "scripts/gnss/d.sh": "URL=${URL:-http://cx43:12048}\n",
        "scripts/gnss/site/e.py": 'NODES = [\n    "cx19",\n    "cx27",\n]\n',
        "python/scripts/gnss/f.py": 'ap.add_argument("--url", default="http://cx43:12048")\n',
        "config/g.py": 'X = os.environ.get("X", "cx19 cx27").split()\n',
    }
    for path, text in sorted(bad.items()):
        check(bool(violations(path, text)), "fails: %s %r" % (path, text))
    good = {
        "scripts/gnss/site/a.sh": "# for n in cx19 cx27; do\nNODES=${GNSS_NODES:-}  # cx19 cx27\n",
        "scripts/gnss/site/b.py": '"""Usage: x.py --nodes cx19 cx27"""\nimport os  # cx19 cx27\n',
        "scripts/gnss/c.sh": 'echo "curl http://cx19:12048/config"\n',
        "python/scripts/gnss/d.py": 'help="node REST base (e.g. http://cx43:12048)"\n',
    }
    for path, text in sorted(good.items()):
        v = violations(path, text)
        check(not v, "passes: %s %r %s" % (path, text, v))


def test_no_hand_typed_hosts():
    files = scanned_files()
    check(len(files) > 50, "scanned %d files" % len(files))
    found = []
    for path in files:
        with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
            found += violations(path, fh.read())
    for v in found:
        print("    " + v)
    check(
        not found,
        "no hand-typed host lists or default host URLs (%d found); take them from "
        "config/gnss_fleet.py" % len(found),
    )


def run(cmd, cwd=None):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env={"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/")},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    return p.returncode, p.stdout


def test_shell_helper():
    site = os.path.join(ROOT, "scripts", "gnss", "site")
    rc, out = run(
        [
            "bash",
            "-c",
            '. "$1/kotekan_src.sh"; gnss_fleet_hosts "$GNSS_CHECKOUT"',
            "-",
            site,
        ]
    )
    check(
        rc == 0 and out.split() == gnss_fleet.hosts(),
        "gnss_fleet_hosts prints the fleet's hosts (rc %d: %r)" % (rc, out.strip()),
    )
    rc, out = run(
        [
            "bash",
            "-c",
            '. "$1/kotekan_src.sh"; gnss_fleet_rest_url "$GNSS_CHECKOUT"',
            "-",
            site,
        ]
    )
    fl = gnss_fleet.Fleet()
    check(
        rc == 0 and out.strip() == fl.rest_url(fl.hosts[0]),
        "gnss_fleet_rest_url prints the first host's REST URL (rc %d: %r)"
        % (rc, out.strip()),
    )


def test_no_fleet_list_fails():
    """A checkout without the fleet list: the defaults stop and say why. No node is reached:
    the list is read before anything else, and ssh is a stub."""
    d = tempfile.mkdtemp(prefix="fleet_hosts_")
    try:
        site = os.path.join(d, "scripts", "gnss", "site")
        os.makedirs(site)
        for f in ("kotekan_src.sh", "node_wedge_probe.sh", "port_axis_gate.py"):
            shutil.copy(os.path.join(HERE, f), site)
        rc, out = run(
            [
                "bash",
                "-c",
                'ssh() { echo SSH; }; export -f ssh; bash "$1"',
                "-",
                os.path.join(site, "node_wedge_probe.sh"),
            ]
        )
        check(
            rc != 0 and "no fleet host list" in out and "SSH" not in out,
            "node_wedge_probe.sh with no fleet list exits non-zero and says why (rc %d: %r)"
            % (rc, out.strip().splitlines()[-1:]),
        )
        rc, out = run(["/usr/bin/python3", os.path.join(site, "port_axis_gate.py")])
        check(
            rc != 0 and "no fleet host list" in out,
            "port_axis_gate.py with no fleet list exits non-zero and says why (rc %d: %r)"
            % (rc, out.strip().splitlines()[-1:]),
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        try:
            t()
        except Exception as e:
            check(False, "%s raised %s: %s" % (t.__name__, type(e).__name__, e))
    if _fails:
        print("\n%d FAILED" % len(_fails))
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
