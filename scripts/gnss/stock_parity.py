#!/usr/bin/env python3
"""Is a GNSS node config's stock half production's? Checked block by block, not asserted.

    scripts/gnss/stock_parity.py config/generated/chord_gnss_cx51_multi.yaml
    scripts/gnss/stock_parity.py config/generated/chord_gnss_cx51_multi.yaml --live cx47

The GNSS nodes run production's pipeline and the GNSS branch in one kotekan (DPDK owns the
NICs). Every top-level block not named gnss* feeds the shared receiver (recv1: the N^2
visibilities, the DishInputs subset, the bad-feed mask), so it must be what a stock node
runs. A stale base fails silently: the nodes ship self-consistent, wrong N^2.

The reference is by default the stock render of config/chord/pathfinder.j2 (and its includes),
rendered exactly as kotekan renders it, which is also the generator's base. With --live, a
running stock node's /config, which also carries the values choco pushes at runtime (the EOP
table, bffs's bad inputs): those must then be equal too, so --live checks the live injection
end to end.

Every difference is declared below, with its reason, or the check fails: an unrecorded
deviation is how a base drifts.
"""
import argparse
import fnmatch
import json
import os
import re
import sys
import urllib.request

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.append(os.path.join(ROOT, "python", "scripts", "gnss"))
import kotekan_src  # noqa: E402

# (path pattern, kinds, why). Kinds: + added, - removed, ~ changed. Patterns are fnmatch over
# the dotted path with list indices as plain components (commands[0] matches commands.*). Each
# is ours to upstream or retire. All but two are data-neutral; the excision exemption and the
# N2 projection change what recv1 receives in the GNSS lobes, by design.
DECLARED = [
    (
        "rest_server",
        "+",
        "GNSS viewer: CORS and the REST thread's cores; the same port (12048) stock serves on",
    ),
    (
        "config_tracker.upstream_fetch_retries",
        "~",
        "startup FPGA-config fetch budget 2 -> 5: chive can take over 10 s to answer at startup",
    ),
    (
        "config_tracker.upstream_fetch_timeout_seconds",
        "~",
        "10 -> 30 s, same reason; startup only",
    ),
    (
        "run_rfi_sktilde.gpu_*.commands.*.rfi_first_stage_excision_exempt_freq_ids",
        "+",
        "GNSS lobes exempt from first-stage excision (our cudaRFISKtilde key, default off): their "
        "N2 feeds the satellite projection and is lost to cosmology anyway",
    ),
    (
        "n2_accumulate.accum_*.in_buf",
        "~",
        "live GNSS N2 projection: N2Accumulate reads the projected copy "
        "gnss_n2_proj_buffer{,_1} written by gnss_n2_project",
    ),
]
# Values stock gets by REST at runtime; the generator injects the current ones. Different from
# the bare render by design, and equal to a live stock node's, so --live does not allow them.
LIVE = [
    (
        "earth_rotation_data.earth_orientation_parameter_table",
        "~",
        "choco's EOP table (stock: pushed by choco)",
    ),
    (
        "updatable_config.bad_inputs.*",
        "~",
        "bffs's bad-input list (stock: relayed by choco)",
    ),
]


def is_gnss(key):
    return key.startswith("gnss")


def template():
    """The stock reference by default: config/chord/pathfinder.j2 in the kotekan tree."""
    return kotekan_src.src("config", "chord", "pathfinder.j2")


def render_stock(path=None):
    """The template (default template()) rendered stock, exactly as kotekan does (see
    kotekan/kotekan.cpp)."""
    import jinja2

    d, f = os.path.split(os.path.abspath(path or template()))
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(d), autoescape=jinja2.select_autoescape()
    )
    return yaml.safe_load(env.get_template(f).render({}))


def load(src):
    """A config from a path (.j2 rendered stock, .json, or yaml) or http://host:port/config."""
    if src.startswith("http://"):
        with urllib.request.urlopen(src, timeout=10) as r:
            return json.loads(r.read().decode())
    if src.endswith(".j2"):
        return render_stock(src)
    with open(src) as fh:
        return (
            json.load(fh)
            if src.endswith(".json")
            else yaml.load(fh, Loader=yaml.CSafeLoader)
        )


def differences(ref, cfg):
    """[(kind, path, detail)] over the non-GNSS blocks; lists of dicts are walked by index."""
    out = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b), key=str):
                p = "%s.%s" % (path, k) if path else str(k)
                if k not in a:
                    out.append(("+", p, json.dumps(b[k])[:120]))
                elif k not in b:
                    out.append(("-", p, json.dumps(a[k])[:120]))
                else:
                    walk(a[k], b[k], p)
        elif (
            isinstance(a, list)
            and isinstance(b, list)
            and len(a) == len(b)
            and a
            and all(isinstance(x, dict) for x in a + b)
        ):
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, "%s[%d]" % (path, i))
        elif a != b:
            out.append(
                ("~", path, "%s -> %s" % (json.dumps(a)[:60], json.dumps(b)[:60]))
            )

    walk(
        {k: v for k, v in ref.items() if not is_gnss(k)},
        {k: v for k, v in cfg.items() if not is_gnss(k)},
        "",
    )
    return out


def classify(diffs, live=False):
    """Split differences into (declared, undeclared). `live`: the reference is a running stock
    node, so the runtime-pushed values must match it rather than merely differ from the render."""
    allowed = DECLARED + ([] if live else LIVE)
    declared, undeclared = [], []
    for kind, path, detail in diffs:
        flat = re.sub(r"\[(\d+)\]", r".\1", path)
        why = next(
            (
                w
                for pat, kinds, w in allowed
                if kind in kinds and fnmatch.fnmatchcase(flat, pat)
            ),
            None,
        )
        (declared if why else undeclared).append((kind, path, detail, why))
    return declared, undeclared


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("config", help="a generated node config (yaml)")
    ap.add_argument(
        "--stock",
        help="reference: a .j2 template (rendered stock, as kotekan renders it), a .json or "
        "yaml file, or the http:// URL of a /config endpoint (default: "
        "config/chord/pathfinder.j2 in the kotekan tree; ignored with --live)",
    )
    ap.add_argument(
        "--live",
        metavar="HOST",
        help="use HOST's live /config on port 12048 as the reference instead of --stock. "
        "HOST must run a stock node; the values choco pushes at runtime (EOP table, "
        "bad inputs) must then match too",
    )
    ap.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="print only undeclared differences (and the summary line if any)",
    )
    a = ap.parse_args()

    ref_src = "http://%s:12048/config" % a.live if a.live else a.stock or template()
    ref, cfg = load(ref_src), load(a.config)
    declared, undeclared = classify(differences(ref, cfg), live=bool(a.live))
    n_stock = sum(1 for k in cfg if not is_gnss(k))
    if not a.quiet or undeclared:
        print(
            "%s vs %s: %d stock blocks, %d declared deviation(s), %d UNDECLARED"
            % (a.config, ref_src, n_stock, len(declared), len(undeclared))
        )
    if not a.quiet:
        for kind, path, detail, why in declared:
            print("  ok  %s %s  -- %s" % (kind, path, why))
    for kind, path, detail, _ in undeclared:
        print("  !!  %s %s  %s" % (kind, path, detail))
    sys.exit(1 if undeclared else 0)


if __name__ == "__main__":
    main()
