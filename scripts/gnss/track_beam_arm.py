#!/usr/bin/env python3
"""Keep the fleet's track-beam gates armed.

The continuous GuoWang stream (docs/CHORD_GUOWANG_TRACK_BEAM.md) passes gnss<g>_trackbeam_gate
only while a window is armed. Each run extends every gate's window to --duration seconds past
the fleet's last seq, with start_seq 0 so a re-arm never drops a frame. Run it from cron more
often than --duration: if the cron stops, every gate closes within --duration and no disk
fills.

    track_beam_arm.py [--duration 720] [--node cx27 ...]
    track_beam_arm.py --disarm
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viscap  # noqa: E402

GATES = ["gnss0_trackbeam_gate", "gnss1_trackbeam_gate"]


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
    a = ap.parse_args()
    nodes = a.node or viscap.fleet_hosts()
    if a.disarm:
        viscap.disarm(nodes, GATES)
        return
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
