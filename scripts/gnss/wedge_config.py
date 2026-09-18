#!/usr/bin/env python3
"""Derive a WEDGE-PROBE config from a generated node config.

    wedge_config.py <in.yaml> <out.yaml> [--lock device|stream] [--keep-bases]

WHY THIS EXISTS. The half-node wedge (chord-gpu-command-mutex-wedge) was retired on
2026-08-31 by locking per CUDA stream instead of per device and giving each chain private
streams, but WHY the holding thread never returned from the driver was never established --
so the fix is a blast-radius reduction on an unexplained fault, and it may have traded one
failure class for another. This script rebuilds the conditions that wedged, so the fault can
be re-excited on purpose and caught by the instrumentation rather than by a backtrace taken
hours later.

WHAT IT CHANGES, and nothing else:
  * gpu_command_lock: device  -- ONE queuing mutex per GPU across every pipeline's whole
    command loop, the pre-2026-08-31 behaviour.
  * every cudaProcess back to cuda_stream_base 0 with num_cuda_streams 3, so all of them
    share streams 0/1/2 as they did then. Without this the device-wide lock is only half the
    original condition. --keep-bases leaves the private streams in place, which is the other
    cell of the experiment: does the lock alone reproduce it?
  * the watchdog knobs. These are read at the root, and config lookup walks up, so one key
    covers every stage.

The output is a SEPARATE file: the input is the armed config the node runs and is never
touched.
"""
import argparse
import sys

import yaml

BANNER = """# WEDGE PROBE -- DERIVED, DIAGNOSTIC, NOT A PRODUCTION CONFIG.
# Rebuilt by scripts/gnss/wedge_config.py from {src}
# gpu_command_lock={lock}  bases={bases}  watchdog={wd}s  stuck={stuck}s
#
# This config deliberately restores the conditions of the half-node wedge. Expect degraded
# throughput: with shared bases every pipeline on a GPU queues through one mutex onto three
# streams. Run it only on a node set aside for the hunt.
"""


def walk(node, fn):
    if isinstance(node, dict):
        if node.get("kotekan_stage") == "cudaProcess":
            fn(node)
            return
        for v in node.values():
            walk(v, fn)
    elif isinstance(node, list):
        for v in node:
            walk(v, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--lock", choices=["device", "stream"], default="device")
    ap.add_argument("--keep-bases", action="store_true",
                    help="leave private cuda_stream_base values alone")
    ap.add_argument("--watchdog-s", type=float, default=2.0)
    ap.add_argument("--stuck-s", type=float, default=5.0)
    ap.add_argument("--slow-command-s", type=float, default=0.5)
    ap.add_argument("--eop", metavar="STATE_JSON",
                    help="stage a fresh earth_rotation_data table from choco's state.json "
                         "(scp choco:/var/lib/choco/eop/state.json). A RESTART REVERTS THE "
                         "LIVE TABLE to whatever the config file holds, so a probe config "
                         "derived from a config whose window is closing will silently take "
                         "the node down at that boundary.")
    args = ap.parse_args()

    with open(args.src) as fh:
        cfg = yaml.safe_load(fh)

    # Root keys: every stage inherits these, because config lookup walks up.
    if args.eop:
        import datetime
        import json
        with open(args.eop) as fh:
            table = json.load(fh)["earth_orientation_parameter_table"]
        when = lambda e: datetime.datetime.fromtimestamp(  # noqa: E731
            e["t_inst_ns"] / 1e9, datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        lo, hi = when(table[0]), when(table[-1])
        # Refuse rather than ship a window that does not contain now: the node has
        # fatal_eop_out_of_range set and exits at the edge, and a probe run that dies of its
        # own config teaches nothing.
        if not (lo <= now <= hi):
            sys.exit("REFUSING: %s spans %s..%s, which does not contain now (%s)"
                     % (args.eop, lo.date(), hi.date(), now.date()))
        old = cfg.get("earth_rotation_data", {}).get("earth_orientation_parameter_table")
        if old:
            prev = {e["t_inst_ns"]: e for e in old}
            clash = [when(e).date() for e in table if e["t_inst_ns"] in prev
                     and any(abs(prev[e["t_inst_ns"]][k] - e[k]) > 1e-9
                             for k in ("delta_UT1_inst", "xp_as", "yp_as"))]
            if clash:
                sys.exit("REFUSING: the new table disagrees with the old one on %s -- one of "
                         "them is wrong and this is not the place to find out which" % clash)
        cfg.setdefault("earth_rotation_data", {})["earth_orientation_parameter_table"] = table
        print("EOP: %s .. %s (%.1f h past now)"
              % (lo.date(), hi.date(), (hi - now).total_seconds() / 3600.0))

    cfg["gpu_command_lock"] = args.lock
    cfg["wedge_watchdog_s"] = args.watchdog_s
    cfg["wedge_watchdog_stuck_s"] = args.stuck_s
    cfg["slow_command_warn_s"] = args.slow_command_s

    n = [0]
    moved = [0]

    def fix(stage):
        n[0] += 1
        if args.keep_bases:
            return
        if stage.get("cuda_stream_base", 0) != 0:
            moved[0] += 1
        stage["cuda_stream_base"] = 0
        stage["num_cuda_streams"] = 3

    walk(cfg, fix)
    if n[0] == 0:
        sys.exit("no cudaProcess stages found in %s" % args.src)

    with open(args.dst, "w") as fh:
        fh.write(BANNER.format(src=args.src, lock=args.lock,
                               bases="private (kept)" if args.keep_bases else "collapsed to 0",
                               wd=args.watchdog_s, stuck=args.stuck_s))
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=False, width=200)
    print("%s: %d cudaProcess stages, %d moved back to base 0, lock=%s -> %s"
          % (args.src, n[0], moved[0], args.lock, args.dst))


if __name__ == "__main__":
    main()
