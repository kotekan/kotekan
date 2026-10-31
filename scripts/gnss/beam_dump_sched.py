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

A pass window is [tca - length/2, tca + length/2); overlapping windows are merged. Files land in
<record_dir>/beamdump on each node (default /tmp/gnss/beamdump), one per --beam-dump-frames-
per-file frames, read with scripts/gnss/beam_dump_read.py. The nodes' own records are the
chip reader's input; nothing here reads them.
"""
import argparse
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


def windows(a, t0, t1):
    """Merged [start, end, names] windows of the passes closer than max_sep in [t0, t1]."""
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
    passes = [p for p in passes if p["sep"] <= a.max_sep and p["el"] >= a.el_min]
    out = []
    for p in sorted(passes, key=lambda p: p["tca"]):
        s, e = p["tca"] - a.length / 2.0, p["tca"] + a.length / 2.0
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
        viscap.arm(nodes, GATES, start_in, e - max(s, time.time()))
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
