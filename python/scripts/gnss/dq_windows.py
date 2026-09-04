#!/usr/bin/env python3
"""Data-quality windows: which capture periods to EXCLUDE from analysis.

Dev happens on the live node, so the archive contains stretches that are not representative
sky data. docs/data_quality_windows.md is the record; this is the loader, so analysis scripts
filter through ONE place instead of each carrying its own folklore about "that bad Tuesday".

    from dq_windows import load, is_bad
    w = load()
    if is_bad(t_unix, "l1_gps", w): continue

Severity: `drop` excludes, `suspect` is usable if you say so, `note` is context only.
`is_bad` treats `drop` as bad by default; pass severities=("drop","suspect") to be strict.
"""
import datetime as _dt
import os

DOC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "..", "..", "docs", "data_quality_windows.md")


def _parse_t(s):
    s = s.strip()
    if s == "-":
        return None
    return _dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=_dt.timezone.utc).timestamp()


def load(path=None):
    """[(t0, t1|None, scope, severity, reason)] from the markdown table."""
    path = path or os.path.normpath(DOC)
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        if line.count("|") < 4 or line.lstrip().startswith("*"):
            continue
        f = [x.strip() for x in line.split("|")]
        if len(f) < 5 or not f[0][:2].isdigit():
            continue
        try:
            out.append((_parse_t(f[0]), _parse_t(f[1]), f[2], f[3], f[4]))
        except ValueError:
            continue
    return out


def is_bad(t, scope="all", windows=None, severities=("drop",)):
    """Is capture time `t` (unix seconds) inside an excluded window for `scope`?"""
    for t0, t1, sc, sev, _ in (windows if windows is not None else load()):
        if sev not in severities:
            continue
        if sc != "all" and scope != "all" and not scope.startswith(sc):
            continue
        if t >= t0 and (t1 is None or t < t1):
            return True
    return False


if __name__ == "__main__":
    w = load()
    print("%d windows" % len(w))
    for t0, t1, sc, sev, why in w:
        a = _dt.datetime.fromtimestamp(t0, _dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ")
        b = ("open" if t1 is None
             else _dt.datetime.fromtimestamp(t1, _dt.timezone.utc).strftime("%H:%MZ"))
        print("  %s -> %-6s %-8s %-8s %s" % (a, b, sc, sev, why[:70]))
