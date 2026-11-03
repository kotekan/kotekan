#!/usr/bin/env python3
"""Arm the fleet's lane-dump gates around bright GNSS passes.

Each GPU of a GNSS instance built with --lane-dump-period-s carries a FrameWindowGate
(gnss<g>_lanedump_gate) between GnssN2Project's lane dump and its rawFileWrite. This tool plans
the passes of the broadcast-ephemeris satellites (GPS, Galileo, BeiDou, GLONASS, QZSS) that come
within --max-sep of the dish boresight, where a bright satellite captures the weaker lanes, and
arms every gate on the same seq window per pass: --pre before the closest approach (the lanes
as they were) to --post after it.

    lane_dump_sched.py plan [--hours 12] [--max-sep 3]    # the windows
    lane_dump_sched.py run  [--hours 24] [--max-sep 3] [--once]   # arm them in turn
    lane_dump_sched.py status | disarm

The gate's clock is the frame's F-engine seq. A dump frame comes every --lane-dump-period-s,
too seldom to measure the seq rate from two polls as viscap.py does, so the window is placed
from the gates' last seq with the geometric rate (one seq per 5.12 us sample). The broadcast
ephemeris is read from the broker's cache and never fetched: this tool writes nothing there.
Files land in <record_dir>/lanedump on each node; read them with lane_dump_read.py.
"""
import argparse
import math
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "scripts", "gnss"))
sys.path.insert(
    0, os.path.join(HERE, "..", "..", "python", "scripts", "gnss", "guowang")
)
import viscap  # noqa: E402

GATES = ["gnss0_lanedump_gate", "gnss1_lanedump_gate"]
SEQ_RATE = 1.0 / 5.12e-6  # F-engine samples per second


def ephemeris():
    """The broker's cached broadcast ephemeris for today, read only."""
    import gnss_brdc_supply as bs
    import gnss_ephemeris as ge

    when = datetime.now(timezone.utc)
    files = bs._cached_sources(when, bs._offline_cache_dir(when, bs.CACHE))
    if not files:
        sys.exit("no cached BRDC under %s" % bs.CACHE)
    return ge.parse_rinex_nav(files)


def passes(eph, t0, t1, max_sep, step=30.0):
    """[(t_closest, name, sep_deg)] of every satellite whose boresight separation dips below
    max_sep in [t0, t1], refined to 1 s around the coarse minimum."""
    import gnss_ephemeris as ge
    import guowang_passes as gp

    import numpy as np

    bore = gp.enu_unit(gp.BORESIGHT["az"], gp.BORESIGHT["el"])
    site = gp.SITE

    def seps(t):
        out = {}
        sky = ge.predict_all(
            eph,
            site["lat"],
            site["lon"],
            site["h"],
            t,
            max_age=86400.0,
            health_gate=False,
        )
        for (sys_, prn), p in sky.items():
            u = gp.enu_unit(p["az"], p["el"])
            c = float(np.clip(np.dot(u, bore), -1.0, 1.0))
            out["%s%02d" % (sys_, prn)] = math.degrees(math.acos(c))
        return out

    coarse = [(t, seps(t)) for t in np.arange(t0, t1, step)]
    found = []
    for i in range(1, len(coarse) - 1):
        for name, s in coarse[i][1].items():
            a = coarse[i - 1][1].get(name, 180.0)
            b = coarse[i + 1][1].get(name, 180.0)
            if s <= a and s < b and s < max_sep + 1.0:
                fine = [
                    (seps(t).get(name, 180.0), t)
                    for t in np.arange(coarse[i][0] - step, coarse[i][0] + step, 1.0)
                ]
                sep, t = min(fine)
                if sep < max_sep:
                    found.append((float(t), name, sep))
    return sorted(found)


def windows(a, t0, t1):
    """Merged [start, end, labels] windows around the passes closer than max_sep."""
    out = []
    for t, name, sep in passes(ephemeris(), t0 - a.post, t1, a.max_sep):
        s, e = t - a.pre, t + a.post
        if e < t0:
            continue
        label = "%s %.2f deg %s" % (name, sep, fmt(t)[11:19])
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
            out[-1][2].append(label)
        else:
            out.append([s, e, [label]])
    return out


def fmt(t):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)) + "Z"


def arm(nodes, s, e):
    """Arm every gate on [s, e) (unix), placed from the fleet's last seq."""
    st = viscap.status(nodes, GATES, quiet=True)
    last = max((v["last_seq"] for v in st.values() if "error" not in v), default=-1)
    if last < 0:
        sys.exit("no lane-dump gate reports a seq -- built with --lane-dump-period-s?")
    now = time.time()
    start_seq = last + int(round((s - now) * SEQ_RATE))
    end_seq = last + int(round((e - now) * SEQ_RATE))
    print("window [%d, %d), last_seq %d" % (start_seq, end_seq, last))
    ok = 0
    for n, g, base in viscap.each_gate(nodes, GATES):
        try:
            viscap._post(base + "/arm", {"start_seq": start_seq, "end_seq": end_seq})
            ok += 1
        except Exception as ex:
            print("  %s/%s: FAILED (%s)" % (n, g, ex), file=sys.stderr)
    print("%d/%d gates armed" % (ok, len(nodes) * len(GATES)))


def cmd_plan(a, nodes):
    now = time.time()
    for s, e, labels in windows(a, now, now + a.hours * 3600.0):
        print("%s .. %s (%.0f s): %s" % (fmt(s), fmt(e), e - s, "; ".join(labels)))


def cmd_run(a, nodes):
    now = time.time()
    wins = windows(a, now, now + a.hours * 3600.0)
    if not wins:
        sys.exit("no pass within %.1f deg in the next %.1f h" % (a.max_sep, a.hours))
    for s, e, labels in wins:
        if e < time.time():
            continue
        lead = max(0.0, s - time.time())
        print(
            "%s next window %s .. %s: %s; arming in %.0f s"
            % (fmt(time.time()), fmt(s), fmt(e), "; ".join(labels), lead),
            flush=True,
        )
        if lead > a.lead:
            time.sleep(lead - a.lead)
        arm(nodes, max(s, time.time() + 1.0), e)
        sys.stdout.flush()
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
        "--hours", type=float, default=12.0, help="planning horizon from now"
    )
    ap.add_argument(
        "--max-sep",
        type=float,
        default=3.0,
        help="deg from boresight at closest approach (default %(default)s)",
    )
    ap.add_argument(
        "--pre", type=float, default=900.0, help="s before the closest approach"
    )
    ap.add_argument(
        "--post", type=float, default=600.0, help="s after the closest approach"
    )
    ap.add_argument(
        "--lead", type=float, default=20.0, help="s before a window to arm the gates"
    )
    ap.add_argument("--once", action="store_true", help="arm the next window and exit")
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
