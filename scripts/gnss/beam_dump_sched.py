#!/usr/bin/env python3
"""Arm the fleet's beam-dump gates around GuoWang passes.

Each GPU of a GNSS instance built with --beam-dump-freq-ids carries a FrameWindowGate
(gnss<g>_beamdump_gate) between cudaGnssBeamDump and its rawFileWrite. This tool plans the
passes with python/scripts/gnss/guowang/guowang_passes.py (TLE group hulianwang, boresight
separation) and arms every gate on the same seq window per pass, as scripts/gnss/viscap.py
does for the visibility capture; the seq clock is measured from the gates.

    beam_dump_sched.py plan [--hours 6] [--max-sep 6] [--length 80]      # the windows
    beam_dump_sched.py run  [--hours 6] [--max-sep 6] [--length 80] [--once]   # arm them in turn
    beam_dump_sched.py status | disarm

A pass window is [tca - length/2, tca + length/2); overlapping windows are merged. --extra-tle adds
the satellites of more TLE files (e.g. the inferred DIGUI-H sets of site/gw_hidden_tle.py), their
windows widened by --extra-pad-s on each side for their larger timing error. Files land in
<record_dir>/beamdump on each node (default /tmp/gnss/beamdump), one per --beam-dump-frames-
per-file frames, read with scripts/gnss/beam_dump_read.py. The nodes' own records are the
chip reader's input; nothing here reads them.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(
    0, os.path.join(HERE, "..", "..", "python", "scripts", "gnss", "guowang")
)
import viscap  # noqa: E402

GATES = ["gnss0_beamdump_gate", "gnss1_beamdump_gate"]


def known(paths, revisit_s=None):
    """Satellite names already identified, from --skip-known files: a guowang_1202_db.json
    (its satellites with a known code; one never seen transmitting stays a target) or a
    gw_reduce db_proposals.jsonl (each line's "sat"). With revisit_s, a satellite last observed
    more than revisit_s ago is a target again: its second pass measures how well the TLE and its
    theta predict the code epoch across days. A file that is missing or unreadable adds nothing."""
    import calendar
    import json

    seen = {}

    def note(name, utc):
        t = calendar.timegm(time.strptime(utc, "%Y-%m-%dT%H:%M:%SZ")) if utc else 0.0
        seen[name] = max(seen.get(name, 0.0), t)

    for path in paths:
        try:
            with open(path) as fh:
                if path.endswith(".jsonl"):
                    for ln in fh:
                        if ln.strip():
                            p = json.loads(ln)
                            note(p["sat"], p.get("measured_utc"))
                else:
                    for n, s in json.load(fh)["satellites"].items():
                        if s.get("g2_phase") is not None:
                            note(n, s.get("last_observed_utc") or s.get("measured_utc"))
        except (OSError, ValueError, KeyError):
            continue
    if revisit_s is None:
        return set(seen)
    now = time.time()
    return {n for n, t in seen.items() if now - t < revisit_s}


def revisit(a):
    return a.revisit_hours * 3600.0 if a.revisit_hours else None


def short(name):
    return name.replace("HULIANWANG ", "")


def windows(a, t0, t1):
    """Merged [start, end, names] windows of the passes closer than max_sep in [t0, t1], less
    the satellites already identified (--skip-known)."""
    import guowang_passes as gp

    class P:
        pass

    args = P()
    args.tle = a.tle
    args.fetch = a.fetch
    args.cache_dir = a.cache_dir
    args.name_filter = a.name_filter
    args.az = gp.BORESIGHT["az"]
    args.el = gp.BORESIGHT["el"]
    args.coarse = 10.0
    args.near = a.max_sep
    _, passes, _, _, _ = gp.compute(args, t0, t1)
    for path in a.extra_tle:
        if not os.path.exists(path):
            print("extra TLE %s: missing, skipped" % path)
            continue
        args.tle = path
        _, more, _, _, _ = gp.compute(args, t0, t1)
        for p in more:
            p["pad"] = a.extra_pad_s
        passes += more
    skip = known(a.skip_known, revisit(a))
    passes = [
        p
        for p in passes
        if p["sep"] <= a.max_sep
        and p["el"] >= a.el_min
        and short(p["name"]) not in skip
    ]
    out = []
    for p in sorted(passes, key=lambda p: p["tca"]):
        half = a.length / 2.0 + p.get("pad", 0.0)
        s, e = p["tca"] - half, p["tca"] + half
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
            out[-1][2].append("%s %.2f deg" % (p["name"], p["sep"]))
        else:
            out.append([s, e, ["%s %.2f deg" % (p["name"], p["sep"])]])
    return out


def fmt(t):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)) + "Z"


def cmd_plan(a, nodes):
    now = time.time()
    for s, e, names in windows(a, now, now + a.hours * 3600.0):
        print("%s .. %s (%.0f s): %s" % (fmt(s), fmt(e), e - s, "; ".join(names)))


def cmd_run(a, nodes):
    now = time.time()
    wins = windows(a, now, now + a.hours * 3600.0)
    if not wins:
        sys.exit("no pass within %.0f deg in the next %.1f h" % (a.max_sep, a.hours))
    for s, e, names in wins:
        if e < time.time():
            continue
        # the library grows while this runs: drop a window whose satellites were identified since
        skip = known(a.skip_known, revisit(a))
        if a.skip_known and all(short(n.rsplit(" ", 2)[0]) in skip for n in names):
            print(
                "skip window %s .. %s: %s identified since"
                % (fmt(s), fmt(e), "; ".join(names))
            )
            continue
        lead = max(0.0, s - time.time())
        print(
            "next window %s .. %s: %s; arming in %.0f s"
            % (fmt(s), fmt(e), "; ".join(names), lead)
        )
        # arm shortly before the window so the seq rate measured now still holds; the window
        # itself is placed by the measured rate (viscap.arm's start_in is seconds from now)
        if lead > a.lead:
            time.sleep(lead - a.lead)
        start_in = max(1.0, s - time.time())
        rec = viscap.arm(nodes, GATES, start_in, e - max(s, time.time()))
        if a.armed_log:
            # which GPUs record this window: the reducer expects exactly these
            rec.update(t_start=s, t_end=e, utc_start=fmt(s), targets=names)
            with open(a.armed_log, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        if a.once:
            return
        time.sleep(max(0.0, e - time.time()) + 2.0)
    print("no more windows")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mode", choices=["plan", "run", "status", "disarm"])
    ap.add_argument(
        "--node", action="append", help="node (repeatable); default the fleet"
    )
    ap.add_argument(
        "--hours", type=float, default=6.0, help="planning horizon from now"
    )
    ap.add_argument(
        "--max-sep",
        type=float,
        default=6.0,
        help="deg from boresight at closest approach",
    )
    ap.add_argument(
        "--el-min", type=float, default=20.0, help="deg elevation at closest approach"
    )
    ap.add_argument(
        "--length",
        type=float,
        default=80.0,
        help="s per pass window, centred on the closest approach",
    )
    ap.add_argument(
        "--lead", type=float, default=20.0, help="s before a window to arm the gates"
    )
    ap.add_argument("--once", action="store_true", help="arm the next window and exit")
    ap.add_argument(
        "--tle", help="TLE file (default: the newest cached hulianwang group)"
    )
    ap.add_argument(
        "--fetch", action="store_true", help="refresh the TLE cache if older than 2 h"
    )
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/guowang"))
    ap.add_argument("--name-filter", default="DIGUI")
    ap.add_argument(
        "--extra-tle",
        action="append",
        default=[],
        help="more satellites from this TLE file (repeatable); a missing file is skipped",
    )
    ap.add_argument(
        "--extra-pad-s",
        type=float,
        default=0.0,
        help="s added to each side of the windows of the --extra-tle satellites",
    )
    ap.add_argument(
        "--skip-known",
        action="append",
        default=[],
        help="record only satellites not in this file (repeatable): a guowang_1202_db.json or a "
        "gw_reduce db_proposals.jsonl; re-read before each window",
    )
    ap.add_argument(
        "--armed-log",
        default="/home/kvand/gnss/builds/gw1202/armed_windows.jsonl",
        help="append each window's armed gates and seq range here (read by gw_reduce.py); '' = off",
    )
    ap.add_argument(
        "--revisit-hours",
        type=float,
        help="with --skip-known, record an identified satellite again once its last observation "
        "is older than this (default: never)",
    )
    a = ap.parse_args()
    nodes = a.node or viscap.fleet_hosts()
    if a.mode == "status":
        viscap.status(nodes, GATES)
    elif a.mode == "disarm":
        viscap.disarm(nodes, GATES)
    elif a.mode == "plan":
        cmd_plan(a, nodes)
    else:
        cmd_run(a, nodes)


if __name__ == "__main__":
    main()
