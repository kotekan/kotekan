#!/usr/bin/env python3
"""Is a GNSS node config's STOCK half production's? Checked block by block, not asserted.

    scripts/gnss/stock_parity.py config/generated/chord_gnss_cx51_multi.yaml
    scripts/gnss/stock_parity.py config/generated/chord_gnss_cx51_multi.yaml --live cx47

Our six nodes run production's pipeline and the GNSS branch in ONE kotekan (DPDK owns the
NICs). Everything that is not GNSS -- every block not named gnss* -- feeds the shared receiver
(recv1: the N^2 visibilities, the DishInputs subset, the bad-feed mask), so it must be what a
stock node runs. Until 2026-10-02 it was not: the generator injected into a capture from
08-31, which went stale without a sound, and our nodes shipped self-consistent, wrong N^2 --
two boards short, an old dish table, no subset, no mask (fixtures/stock_parity_20261002/).

THE REFERENCE. By default, the STOCK RENDER of config/chord_pathfinder.j2 -- no gnss_node,
rendered exactly as kotekan renders it -- which is also the generator's base. With --live, a
running stock node's /config, which also carries the values choco PUSHES at runtime (the EOP
table, bffs's bad inputs): those must then be equal too, so --live checks the live injection
end to end.

THE RULE. Every difference is DECLARED below, with its reason, or the check fails. A
deviation nobody wrote down is how the last base drifted.
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
TEMPLATE = os.path.join(ROOT, "config", "chord_pathfinder.j2")

# (path pattern, kinds, why). Kinds: + added, - removed, ~ changed. Patterns are fnmatch over
# the dotted path with list indices as plain components (commands[0] matches commands.*). Each
# is DATA-NEUTRAL --
# none changes what reaches recv1 -- and each is ours to upstream or retire.
DECLARED = [
    ("rest_server", "+",
     "GNSS viewer: CORS and the REST thread's cores; the same port (12048) stock serves on"),
    ("config_tracker.upstream_fetch_retries", "~",
     "startup FPGA-config fetch budget 2 -> 5: chive measured p90 6.5 s, max 16.2 s (08-08)"),
    ("config_tracker.upstream_fetch_timeout_seconds", "~",
     "10 -> 30 s, same reason; startup only"),
    ("dpdk.resync_max_advances", "+",
     "bounded capture resync instead of a wedge (our dpdk code; 0, the default, is stock)"),
    ("run_recv_rfi_*.gpu_*.commands.*.expect_quantity_name", "+",
     "ring-copy descriptor guard (our cudaCopyFromRingbuffer; startup FATALs 08-31)"),
    ("run_rfi_sktilde.gpu_*.commands.*.rfi_first_stage_excision_exempt_freq_ids", "+",
     "GNSS lobes exempt from first-stage excision (our cudaRFISKtilde key, default off): their "
     "N2 feeds the satellite projection and is lost to cosmology anyway (KV, 10-02)"),
]
# Values stock gets by REST at runtime; the generator injects the current ones. Different from
# the bare render by design, and EQUAL to a live stock node's -- so --live does not allow them.
LIVE = [
    ("earth_rotation_data.earth_orientation_parameter_table", "~",
     "choco's EOP table (stock: pushed by choco)"),
    ("updatable_config.bad_inputs.*", "~",
     "bffs's bad-input list (stock: relayed by choco)"),
]


def is_gnss(key):
    return key.startswith("gnss")


def render_stock(path=TEMPLATE):
    """The template rendered stock, exactly as kotekan does (see kotekan/kotekan.cpp)."""
    import jinja2
    d, f = os.path.split(os.path.abspath(path))
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(d),
                             autoescape=jinja2.select_autoescape())
    return yaml.safe_load(env.get_template(f).render({}))


def load(src):
    """A config from a path (.j2 rendered stock, .json, or yaml) or http://host:port/config."""
    if src.startswith("http://"):
        with urllib.request.urlopen(src, timeout=10) as r:
            return json.loads(r.read().decode())
    if src.endswith(".j2"):
        return render_stock(src)
    with open(src) as fh:
        return json.load(fh) if src.endswith(".json") else yaml.load(fh, Loader=yaml.CSafeLoader)


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
        elif (isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and a
              and all(isinstance(x, dict) for x in a + b)):
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, "%s[%d]" % (path, i))
        elif a != b:
            out.append(("~", path, "%s -> %s" % (json.dumps(a)[:60], json.dumps(b)[:60])))

    walk({k: v for k, v in ref.items() if not is_gnss(k)},
         {k: v for k, v in cfg.items() if not is_gnss(k)}, "")
    return out


def classify(diffs, live=False):
    """Split differences into (declared, undeclared). `live`: the reference is a running stock
    node, so the runtime-pushed values must MATCH rather than merely differ from the render."""
    allowed = DECLARED + ([] if live else LIVE)
    declared, undeclared = [], []
    for kind, path, detail in diffs:
        flat = re.sub(r"\[(\d+)\]", r".\1", path)
        why = next((w for pat, kinds, w in allowed
                    if kind in kinds and fnmatch.fnmatchcase(flat, pat)), None)
        (declared if why else undeclared).append((kind, path, detail, why))
    return declared, undeclared


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="a generated node config (yaml)")
    ap.add_argument("--stock", default=TEMPLATE,
                    help="reference: a .j2 (rendered stock), .json/.yaml, or http URL "
                         "(default: config/chord_pathfinder.j2)")
    ap.add_argument("--live", metavar="HOST",
                    help="reference = HOST's live /config on 12048 (a STOCK node, e.g. cx47); "
                         "the runtime-pushed values must then match too")
    ap.add_argument("-q", "--quiet", action="store_true", help="print only failures")
    a = ap.parse_args()

    ref_src = "http://%s:12048/config" % a.live if a.live else a.stock
    ref, cfg = load(ref_src), load(a.config)
    declared, undeclared = classify(differences(ref, cfg), live=bool(a.live))
    n_stock = sum(1 for k in cfg if not is_gnss(k))
    if not a.quiet or undeclared:
        print("%s vs %s: %d stock blocks, %d declared deviation(s), %d UNDECLARED"
              % (a.config, ref_src, n_stock, len(declared), len(undeclared)))
    if not a.quiet:
        for kind, path, detail, why in declared:
            print("  ok  %s %s  -- %s" % (kind, path, why))
    for kind, path, detail, _ in undeclared:
        print("  !!  %s %s  %s" % (kind, path, detail))
    sys.exit(1 if undeclared else 0)


if __name__ == "__main__":
    main()
