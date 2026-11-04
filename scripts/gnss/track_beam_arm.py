#!/usr/bin/env python3
"""Keep the fleet's track-beam gates armed.

The continuous GuoWang stream (docs/CHORD_GUOWANG_TRACK_BEAM.md) passes gnss<g>_trackbeam_gate
only while a window is armed. Each run extends every gate's window to --duration seconds past
the fleet's last seq, with start_seq 0 so a re-arm never drops a frame. Run it from cron more
often than --duration: if the cron stops, every gate closes within --duration and no disk
fills.

With --heartbeat FILE it also requires the cf05 decode loop (builds/gw1202/track_beam_loop.py)
to be alive: the loop writes FILE on the NFS home every pull, with the number of files waiting
on each node. A heartbeat older than --heartbeat-max-age or a node backlog over --max-backlog
disarms every gate, so a dead or stuck loop stops the recording instead of filling the nodes.

    track_beam_arm.py [--duration 720] [--heartbeat FILE] [--node cx27 ...]
    track_beam_arm.py --disarm
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viscap  # noqa: E402

GATES = ["gnss0_trackbeam_gate", "gnss1_trackbeam_gate"]


def heartbeat_problem(path, max_age, max_backlog):
    """Why the loop's heartbeat says not to arm, or None."""
    try:
        with open(path) as fh:
            hb = json.load(fh)
    except (OSError, ValueError) as e:
        return "heartbeat %s unreadable (%s)" % (path, e)
    age = time.time() - hb.get("t", 0)
    if age > max_age:
        return "heartbeat %s is %.0f s old (max %.0f)" % (path, age, max_age)
    for node, n in sorted(hb.get("backlog", {}).items()):
        if n is not None and n > max_backlog:
            return "%d files waiting on %s (max %d)" % (n, node, max_backlog)
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--node", action="append", help="node (repeatable); default the fleet"
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=720.0,
        help="seconds of window past the fleet's last seq (default 720)",
    )
    ap.add_argument("--disarm", action="store_true", help="close every gate now")
    ap.add_argument(
        "--heartbeat",
        help="the decode loop's heartbeat file; arm only while it is fresh",
    )
    ap.add_argument(
        "--heartbeat-max-age",
        type=float,
        default=900.0,
        help="seconds before the heartbeat counts as stale (default 900)",
    )
    ap.add_argument(
        "--max-backlog",
        type=int,
        default=600,
        help="files waiting on any node before disarming (default 600, ~50 min)",
    )
    a = ap.parse_args()
    nodes = a.node or viscap.fleet_hosts()
    if a.disarm:
        viscap.disarm(nodes, GATES)
        return
    if a.heartbeat:
        why = heartbeat_problem(a.heartbeat, a.heartbeat_max_age, a.max_backlog)
        if why:
            print(
                "%s %s: disarming"
                % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), why),
                file=sys.stderr,
            )
            viscap.disarm(nodes, GATES)
            sys.exit(1)
    rate, step, last = viscap.measure_rate(nodes, GATES)
    end = last + max(1, int(round(a.duration * rate / step))) * step
    armed, total = 0, 0
    for n, g, base in viscap.each_gate(nodes, GATES):
        total += 1
        try:
            viscap._post(base + "/arm", {"start_seq": 0, "end_seq": int(end)})
            armed += 1
        except Exception as e:
            print("  %s/%s: FAILED (%s)" % (n, g, e), file=sys.stderr)
    print(
        "%s armed %d/%d gates to seq %d (+%.0f s)"
        % (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            armed,
            total,
            end,
            a.duration,
        )
    )
    if armed == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
