#!/usr/bin/env python3
"""Generate (or verify) every CHORD GNSS node config from one versioned manifest.

    scripts/gnss/gen_fleet.py config/gnss_fleet_chord.yaml            # write them
    scripts/gnss/gen_fleet.py config/gnss_fleet_chord.yaml --check    # verify, write nothing
    scripts/gnss/gen_fleet.py config/gnss_fleet_chord.yaml --node cx19 --print-cmd

WHAT THIS IS FOR. `gen_chord_gnss_config.py` takes ~48 flags and, until 2026-08-09, the set
that produced the running fleet was recorded nowhere -- so six configs the instrument ran on
were unreproducible in practice, and changing a node meant hand-patching a file stamped DO
NOT HAND-EDIT. This driver makes the flags a committed artifact and `--check` makes that
enforceable: it regenerates into memory and compares byte-for-byte, so a hand-edit, a stale
base, or a generator whose output moved all show up as a red gate rather than as a node that
quietly differs from its five siblings.

`--check` is a GATE. It exits non-zero on the first mismatch and prints the diff, and it is
the thing to run before any fleet restart -- a node started from a config nobody can
regenerate is a node whose behaviour has no explanation when it misbehaves.
"""
import argparse
import difflib
import os
import shlex
import subprocess
import sys

import yaml

K = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = os.path.join(K, "config", "gen_chord_gnss_config.py")
OUTDIR = os.path.join(K, "config", "generated")


def flags_from(mapping):
    """Manifest keys -> generator flags, same convention as gnss_chains_chord.yaml:
    true -> a bare flag, false/None -> omitted, anything else -> --key value.

    A LIST repeats the flag: `extra-signal: [A, B]` -> `--extra-signal A --extra-signal B`,
    which is what argparse's action="append" wants. Needed the moment a node carries more
    than one extra chain (E5a *and* B2a on the same 1176.45 MHz tap).
    """
    out = []
    for k, v in mapping.items():
        if v is False or v is None:
            continue
        for item in (v if isinstance(v, (list, tuple)) else [v]):
            out.append("--" + k)
            if item is not True:
                out.append(str(item))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--check", action="store_true",
                    help="regenerate into memory and compare against the committed files; "
                         "write nothing, exit non-zero on any difference")
    ap.add_argument("--node", action="append", default=None,
                    help="restrict to these nodes (repeatable); default is every node")
    ap.add_argument("--print-cmd", action="store_true",
                    help="print the full generator command line instead of running it")
    ap.add_argument("--print-path", action="store_true",
                    help="print the config path this manifest owns for each node and exit. "
                         "Lets a caller ask 'is THIS file one you own?' exactly, rather than "
                         "guessing from the node name -- the suffix lives in the manifest.")
    a = ap.parse_args()

    man_path = a.manifest if os.path.isabs(a.manifest) else os.path.join(K, a.manifest)
    with open(man_path) as f:
        man = yaml.safe_load(f)

    if a.print_path:
        sfx = man.get("suffix", "")
        for n in sorted(man["nodes"]):
            if not a.node or n in a.node:
                print(os.path.join(OUTDIR, "chord_gnss_%s%s.yaml" % (n, sfx)))
        return

    base = man["base"]
    base = base if os.path.isabs(base) else os.path.join(K, "config", base)
    if not os.path.exists(base):
        sys.exit("base config %s does not exist. It is a real INPUT -- the generator injects\n"
                 "into it -- so it must be committed, not re-captured ad hoc." % base)
    suffix = man.get("suffix", "")
    common = flags_from(man.get("common") or {})

    nodes = sorted(man["nodes"])
    if a.node:
        missing = [n for n in a.node if n not in nodes]
        if missing:
            sys.exit("not in the manifest: %s (have %s)" % (", ".join(missing), ", ".join(nodes)))
        nodes = [n for n in nodes if n in a.node]

    bad, written = [], []
    for node in nodes:
        out = os.path.join(OUTDIR, "chord_gnss_%s%s.yaml" % (node, suffix))
        cmd = ([sys.executable, GEN, "--base", base, "--node", node]
               + common + flags_from(man["nodes"][node] or {}))

        if a.print_cmd:
            print(" ".join(shlex.quote(c) for c in cmd + ["--out", out]))
            continue

        # No --out: take the config on stdout so --check never touches the tree. The
        # generator writes its human summary to stderr, so stdout is the file exactly.
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=K)
        if p.returncode != 0:
            # ⚠️ EXIT 2, NOT 1. Exit 1 means "the config differs from the manifest" and
            # nothing else; a generator that failed to RUN has said nothing about the config.
            # Conflating them made node_up.sh's preflight report "does not match
            # config/gnss_fleet_chord.yaml" on cx27/cx42/cx51 on 2026-08-09 while all six were
            # byte-identical -- a confident, specific, wrong claim that sent KV hunting a
            # config problem that did not exist. A checker that cannot run must say so.
            sys.stderr.write(p.stderr[-3000:] + "\n")
            sys.stderr.write("generator failed for %s (exit %d)\n" % (node, p.returncode))
            raise SystemExit(2)
        text = p.stdout

        if not a.check:
            # ATOMIC: a sibling .tmp then os.replace. A plain open(out,"w") leaves a window
            # where a concurrent reader -- node_up.sh's preflight, another --check, or kotekan
            # itself starting -- sees a TRUNCATED config. It reads as a difference, so the
            # preflight reports "no longer matches the manifest" about a file that is merely
            # half-written. Same fix and same reason as gnss_ephemeris._atomic_write_bytes,
            # which exists because readers were catching half-written BRDC gzips.
            tmp = out + ".tmp"
            with open(tmp, "w") as f:
                f.write(text)
            os.replace(tmp, out)
            written.append(os.path.relpath(out, K))
            continue

        if not os.path.exists(out):
            print("MISSING  %s" % os.path.relpath(out, K))
            bad.append(node)
            continue
        have = open(out).read()
        if have == text:
            print("ok       %s" % os.path.relpath(out, K))
        else:
            bad.append(node)
            print("DIFFERS  %s" % os.path.relpath(out, K))
            d = list(difflib.unified_diff(have.splitlines(), text.splitlines(),
                                          "committed", "regenerated", lineterm="", n=1))
            print("\n".join("    " + l for l in d[:40]))
            if len(d) > 40:
                print("    ... %d more diff lines" % (len(d) - 40))

    if a.print_cmd:
        return
    if a.check:
        if bad:
            sys.exit("\n%d of %d node config(s) do not match the manifest: %s\n"
                     "Either the file was hand-edited (regenerate: drop --check), or the\n"
                     "generator's output moved (regenerate and review the diff), or the base\n"
                     "changed underneath. All three are things to decide, not to ignore."
                     % (len(bad), len(nodes), ", ".join(bad)))
        print("\nALL %d NODE CONFIGS MATCH THE MANIFEST." % len(nodes))
    else:
        print("\nwrote %d config(s):\n  %s" % (len(written), "\n  ".join(written)))
        print("\nNothing restarts on its own -- `scripts/gnss/node_up.sh <node> restart`.")


if __name__ == "__main__":
    main()
