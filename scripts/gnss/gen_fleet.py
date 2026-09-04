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
import shutil
import subprocess
import sys
import tempfile

import yaml

K = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = os.path.join(K, "config", "gen_chord_gnss_config.py")
OUTDIR = os.path.join(K, "config", "generated")
# The j2 vars fragments, one per node -- config/gnss/gnss_chain.j2's data half, imported by
# chord_pathfinder.j2 as `gnss_vars_<node>.j2`. Generated from the SAME run as the config, so
# the two can never describe different fleets.
#
# ⚠️ THEY ROTTED FOR TWO WEEKS BECAUSE NOTHING OWNED THEM (found 2026-09-02). All six were
# emitted by hand on 08-19 and then left: 107 diff lines behind the generator by the time
# anyone looked -- the pre-08-31 single-NUMA core pools, no phi_fp16, no despread_max_chips,
# missing the B3I/E6 band_power channels, tiles sized 2981888 instead of 917504. Meanwhile
# config/gnss/README.md said they were "field-for-field what we deploy". An artifact whose
# inputs nobody re-runs is a souvenir, and this driver exists exactly so the fleet's
# artifacts have owners and a way to fail.
VARSDIR = os.path.join(K, "config", "gnss")


def _drop_eop(text):
    """Blank the earth_orientation_parameter_table entries out of a rendered config.

    Line-based on purpose: the block is emitted by yaml.safe_dump as an indented list under
    `earth_rotation_data:`, and this runs on the rendered TEXT (which is what --check
    compares) rather than reparsing. Everything outside the table stays byte-exact, so a real
    difference anywhere else -- including elsewhere in earth_rotation_data -- still fails.
    """
    out, skipping = [], False
    for line in text.splitlines(True):
        if line.startswith("  earth_orientation_parameter_table:"):
            out.append(line)
            skipping = True
            continue
        if skipping:
            # yaml.safe_dump renders the entries as "  - delta_UT1_inst: ..." followed by
            # "    t_inst_ns: ...", i.e. a list item at the SAME two-space indent as the key
            # plus four-space continuations. An earlier version required three spaces and so
            # ended the block on its very first entry -- which passed the happy-path check and
            # failed the falsification, because nothing it was meant to skip was skipped.
            if line.startswith("  - ") or line.startswith("    "):
                continue
            skipping = False
        out.append(line)
    return "".join(out)


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


# ---- THE CONTENT GATE (--check-prns) -------------------------------------------------------
# ⚠️ `--check` compares the GENERATED FILES byte-for-byte and is deliberately offline and
# deterministic. This is a different question and deliberately a different flag: are the PRN
# lists in the manifest still TRUE OF THE SKY? They are hand-written strings, and the comments
# beside them are human snapshots of which slots carry an active satellite. The broker, by
# contrast, reads live BRDC every cycle -- so the two drift apart the moment the constellation
# changes, silently, with no error anywhere.
#
# THAT DRIFT HAS ALREADY COST US (2026-08-26). The Galileo list says "the range 1-36 minus the
# slots with no active satellite. Exactly 32 -- full." Measured: FIVE dead slots (1, 14, 18,
# 22, 24) and E36 ACTIVE BUT EXCLUDED. Nothing was scarce -- we carried five empty slots while
# locking out a live satellite. And because the broker's noise-probe selector picks the DEEPEST
# below-horizon PRN, it kept choosing E36, which the node cannot represent; the probe was
# seeded, never reported, and both Galileo chains silently fell back from the q+p presence gate
# to BRIGHTNESS-ONLY for want of a third probe.
#
# EXIT CODE DISCRIMINATES, because the two findings are not equally bad:
#   EXCLUDED (active, rises here, no slot)  -> a satellite we cannot see. FAILS.
#   DEAD     (slot, no active satellite)    -> wasted capacity. WARNS.
# A dead slot is only a problem when paired with an exclusion, which is exactly today's case.
SYS_OF = {"GPS": "G", "GAL": "E", "BDS": "C"}
LAT, LON, ALT = 49.32075144444, -119.62081125, 545.0
# ⚠️ CAPABILITY, NOT JUST ACTIVITY -- and without this the gate cries wolf, which is worse
# than no gate. BeiDou B2a/B2b are BDS-3 signals: C1-C14 are BDS-2 (B1I/B2I era) and do not
# broadcast them, so their exclusion from the list is CORRECT and must never be reported as a
# fault. The manifest states the rule beside the list it governs; this encodes it so the check
# can apply it. A signal absent from this table is assumed to have no PRN restriction.
MIN_PRN = {"BDS_B2A_P_CS": 19, "BDS_B2B_I": 19}
# A satellite that only grazes the horizon has no claim on a slot: use the tracking mask, not
# el > 0. C56 peaks at 6.1 deg over a whole day -- visible to geometry, useless to the loop.
RISE_DEG = 10.0


def check_prns(man, rise_deg=RISE_DEG, hours=24.0, step_min=10.0):
    """Compare every manifest PRN list against live BRDC + a 24 h visibility sweep."""
    import time
    from datetime import datetime, timezone
    sys.path.insert(0, os.path.join(K, "python", "scripts", "gnss"))
    from gnss_ephemeris import fetch_brdc, parse_rinex_nav, predict_all

    lists = {}
    for ent in (man.get("common", {}).get("extra-signal") or []):
        name, _, prn_s = str(ent).partition(":")
        lists[name] = sorted(int(x) for x in prn_s.split(",") if x.strip())
    if not lists:
        print("no extra-signal PRN lists in the manifest -- nothing to check")
        return 0

    eph = parse_rinex_nav(fetch_brdc(datetime.now(timezone.utc)))
    # Max elevation over a whole day: a satellite that never rises here has no claim on a
    # slot however active it is (the BeiDou list is already built this way, and says so).
    now = time.time()
    peak = {}
    for k in range(int(hours * 60 / step_min)):
        pd = predict_all(eph, LAT, LON, ALT, now + k * step_min * 60.0,
                         mask_deg=-90.0, max_age=86400.0)
        for key, v in pd.items():
            if v["el"] > peak.get(key, -90.0):
                peak[key] = v["el"]

    bad = 0
    for name in sorted(lists):
        sysid = SYS_OF.get(name.split("_")[0])
        if sysid is None:
            print("%-16s UNKNOWN constellation prefix -- skipped" % name)
            continue
        cfg = set(lists[name])
        lo = MIN_PRN.get(name, 0)
        active = {p for (s, p) in peak if s == sysid and p >= lo}
        rises = {p for (s, p) in peak if s == sysid and p >= lo and peak[(s, p)] > rise_deg}
        dead = sorted(cfg - active)
        excluded = sorted((active & rises) - cfg)
        print("%-16s slots %2d | capable+active %2d | rises >%.0f deg %2d%s" %
              (name, len(cfg), len(active), rise_deg, len(rises),
               ("  [capability: PRN >= %d]" % lo) if lo else ""))
        if dead:
            print("      WARN  %d dead slot(s) (configured, no active satellite): %s"
                  % (len(dead), dead))
        if excluded:
            bad += 1
            print("      FAIL  %d ACTIVE, CAPABLE and VISIBLE but EXCLUDED: %s  (peak el %s)"
                  % (len(excluded), excluded,
                     ", ".join("%.0f" % peak[(sysid, p)] for p in excluded)))
            if dead:
                print("            -> %d dead slot(s) available; swap rather than resize."
                      % len(dead))
        if not dead and not excluded:
            print("      ok")
    print("\nPRN CONTENT GATE %s" % ("RED (a visible satellite has no slot)" if bad else "GREEN"))
    return 1 if bad else 0


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
    ap.add_argument("--check-prns", action="store_true",
                    help="check the manifest's PRN lists against LIVE BRDC and a 24 h "
                         "visibility sweep: reports dead slots (warn) and active, visible, "
                         "unslotted satellites (fail). NOT part of --check, which is offline "
                         "and byte-deterministic by design; this one needs the network and "
                         "its answer legitimately changes as the constellation does.")
    a = ap.parse_args()

    man_path = a.manifest if os.path.isabs(a.manifest) else os.path.join(K, a.manifest)
    with open(man_path) as f:
        man = yaml.safe_load(f)

    if a.check_prns:
        return check_prns(man)

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
        vars_out = os.path.join(VARSDIR, "gnss_vars_%s.j2" % node)
        cmd = ([sys.executable, GEN, "--base", base, "--node", node]
               + common + flags_from(man["nodes"][node] or {}))

        if a.print_cmd:
            print(" ".join(shlex.quote(c) for c in cmd + ["--out", out]))
            continue

        # The vars go to a temp in BOTH modes -- --check must never touch the tree -- and the
        # generator excludes --emit-j2-vars from the recipe it stamps into the config, so
        # asking for them does not change the config by one byte. (It used to: see the
        # _SINK_FLAGS note in gen_chord_gnss_config.py, which is the --out trap again.)
        vars_tmp = os.path.join(tempfile.mkdtemp(prefix="gen_fleet_vars_"),
                                "gnss_vars_%s.j2" % node)
        cmd = cmd + ["--emit-j2-vars", vars_tmp]

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
            if os.path.exists(vars_tmp):
                vtmp = vars_out + ".tmp"
                shutil.copyfile(vars_tmp, vtmp)
                os.replace(vtmp, vars_out)
                written.append(os.path.relpath(vars_out, K))
            continue

        # The j2 vars carry no EOP table and no live data, so they are a plain byte compare.
        if os.path.exists(vars_tmp):
            want_v = open(vars_tmp).read()
            if not os.path.exists(vars_out):
                print("MISSING  %s" % os.path.relpath(vars_out, K))
                bad.append(node + ":vars")
            elif open(vars_out).read() != want_v:
                bad.append(node + ":vars")
                print("DIFFERS  %s" % os.path.relpath(vars_out, K))
                dv = list(difflib.unified_diff(open(vars_out).read().splitlines(),
                                               want_v.splitlines(),
                                               "committed", "regenerated", lineterm="", n=1))
                print("\n".join("    " + l for l in dv[:20]))
                if len(dv) > 20:
                    print("    ... %d more diff lines" % (len(dv) - 20))
            else:
                print("ok       %s" % os.path.relpath(vars_out, K))

        if not os.path.exists(out):
            print("MISSING  %s" % os.path.relpath(out, K))
            bad.append(node)
            continue
        have = open(out).read()

        # ⚠️ THE EARTH-ROTATION TABLE IS LIVE DATA, NOT A DECLARED CHOICE (2026-08-19).
        # It is fetched from the running fleet at generation time and is a rolling ~5-day
        # window, so it MOVES ON ITS OWN roughly daily. Comparing it here would report drift
        # on all six nodes every morning for a reason nobody intends to reproduce -- and this
        # file's own docstring is about why that is worse than not checking: a gate that
        # cries wolf is a gate people learn to skip. Normalise it out of BOTH sides and let
        # everything else stay byte-exact.
        have_c, text_c = _drop_eop(have), _drop_eop(text)
        if have_c == text_c and have != text:
            print("ok*      %s  (differs only in the live EOP table)"
                  % os.path.relpath(out, K))
            continue
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
