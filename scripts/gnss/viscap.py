#!/usr/bin/env python3
"""Arm, watch and disarm the fleet's (N+M)^2 visibility capture.

Each GPU of a GNSS instance built with --vis-capture carries a FrameWindowGate
(gnss<g>_viscap_gate) between its dual correlator and the rawFileWrites: one for the capture
tiles and one ctl writer per captured chain. The gate passes frames only inside an armed
[start_seq, end_seq) window, where seq is the absolute F-engine sample of the frame's first
hop -- one clock for the whole fleet, so arming every gate on the same numbers makes every
node's files start on the same hop.

    viscap.py status                       # last seq, armed windows, counters, per gate
    viscap.py arm --start-in 30 --duration 300     # fleet-wide window, 30 s from now, 5 min
    viscap.py arm --start-seq S --end-seq E # explicit window (frame-aligned by the gate)
    viscap.py disarm                        # close every window now
    viscap.py watch [--every 5]             # status every 5 s until no gate is armed

The seq rate is measured (two status polls) rather than assumed, and cross-checked against
the frame geometry (8192 hops x 5.12 us). Files land in <record_dir>/viscap on each node's
own nvme (default /tmp/gnss/viscap); see scripts/gnss/viscap_read.py for the layout and
docs/CHORD_VIS_CAPTURE.md for the whole procedure.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

CONF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config")
PORT = 12048
# One gate per GPU gates every captured chain on that GPU.
GATES = ["gnss0_viscap_gate", "gnss1_viscap_gate"]
FRAME_S = 8192 * 5120e-9  # hops per frame x hop length -- the geometry cross-check only


def fleet_hosts():
    """The fleet's hosts in slot order: config/gnss_fleet_chord.yaml nodes:, read by
    config/gnss_fleet.py."""
    sys.path.insert(0, CONF)
    try:
        import gnss_fleet

        return gnss_fleet.hosts()
    except Exception as e:
        sys.exit(
            "no fleet host list from %s/gnss_fleet.py (%s: %s); name the nodes explicitly"
            % (CONF, type(e).__name__, e)
        )


def _get(url, timeout=6):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def _post(url, payload, timeout=6):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def each_gate(nodes, gates):
    for n in nodes:
        for g in gates:
            host = n if ":" in n else "%s:%d" % (n, PORT)
            yield n, g, "http://%s/%s" % (host, g)


def status(nodes, gates, quiet=False):
    out = {}
    for n, g, base in each_gate(nodes, gates):
        try:
            out[(n, g)] = _get(base + "/status")
        except Exception as e:
            out[(n, g)] = {"error": str(e)}
    if not quiet:
        print(
            "%-6s %-18s %6s %16s %16s %16s %8s %8s %8s"
            % (
                "node",
                "gate",
                "armed",
                "last_seq",
                "start_seq",
                "end_seq",
                "passed",
                "dropped",
                "in-win-drop",
            )
        )
        for (n, g), s in sorted(out.items()):
            if "error" in s:
                print("%-6s %-18s  UNREACHABLE: %s" % (n, g, s["error"]))
                continue
            print(
                "%-6s %-18s %6s %16d %16d %16d %8d %8d %8d"
                % (
                    n,
                    g,
                    "yes" if s["armed"] else "no",
                    s["last_seq"],
                    s["start_seq"],
                    s["end_seq"],
                    s["passed"],
                    s["dropped"],
                    s["dropped_in_window"],
                )
            )
    return out


def measure_rate(nodes, gates, dt=2.0):
    """seq per second, from two polls of the reachable gates; also the frame step in seq."""
    a = status(nodes, gates, quiet=True)
    t0 = time.time()
    time.sleep(dt)
    b = status(nodes, gates, quiet=True)
    t1 = time.time()
    rates, steps, last = [], [], -1
    for k, sb in b.items():
        sa = a.get(k, {})
        if "error" in sb or "error" in sa or sb["seq_per_frame"] <= 0:
            continue
        if sb["last_seq"] > sa["last_seq"]:
            rates.append((sb["last_seq"] - sa["last_seq"]) / (t1 - t0))
        steps.append(sb["seq_per_frame"])
        last = max(last, sb["last_seq"])
    if not rates:
        sys.exit("no gate is advancing -- is the fleet up and the gate stage built in?")
    if len(set(steps)) != 1:
        sys.exit("gates disagree on seq_per_frame: %s" % sorted(set(steps)))
    rate = sorted(rates)[len(rates) // 2]
    step = steps[0]
    expect = step / FRAME_S
    if abs(rate / expect - 1.0) > 0.05:
        print(
            "  WARNING: measured seq rate %.4g/s is %.1f%% off the geometric %.4g/s "
            "(8192 hops x 5.12 us per frame) -- check the clock source before trusting "
            "the window length" % (rate, 100 * (rate / expect - 1), expect),
            file=sys.stderr,
        )
    return rate, step, last


def arm(nodes, gates, start_in, duration, start_seq=None, end_seq=None):
    rate, step, last = measure_rate(nodes, gates)
    if start_seq is None:
        # Frame-aligned to the grid the gates are actually seeing: last + k frames.
        k = max(1, int(round(start_in * rate / step)))
        start_seq = last + k * step
    if end_seq is None:
        end_seq = start_seq + max(1, int(round(duration * rate / step))) * step
    n_frames = (end_seq - start_seq) // step
    print(
        "seq rate %.6g /s, %d seq/frame (%.2f ms), fleet last_seq %d"
        % (rate, step, 1e3 * step / rate, last)
    )
    print(
        "window [%d, %d): %d frames = %.1f s, opening in ~%.1f s"
        % (
            start_seq,
            end_seq,
            n_frames,
            n_frames * step / rate,
            (start_seq - last) / rate,
        )
    )
    ok = 0
    for n, g, base in each_gate(nodes, gates):
        try:
            r = _post(
                base + "/arm", {"start_seq": int(start_seq), "end_seq": int(end_seq)}
            )
            print("  %s/%s: armed (last_seq %d)" % (n, g, r.get("last_seq", -1)))
            ok += 1
        except Exception as e:
            print("  %s/%s: FAILED (%s)" % (n, g, e), file=sys.stderr)
    print("%d/%d gates armed" % (ok, len(nodes) * len(gates)))
    return ok


def disarm(nodes, gates):
    for n, g, base in each_gate(nodes, gates):
        try:
            r = _post(base + "/disarm", {})
            print(
                "  %s/%s: disarmed, %d frames passed in total"
                % (n, g, r.get("passed", -1))
            )
        except Exception as e:
            print("  %s/%s: FAILED (%s)" % (n, g, e), file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cmd", choices=["status", "arm", "disarm", "watch"])
    ap.add_argument(
        "--nodes",
        default="",
        help="comma list; host or host:port (default port %d); default the fleet "
        "(config/gnss_fleet.py hosts)" % PORT,
    )
    ap.add_argument(
        "--gates",
        default=",".join(GATES),
        help="gate stage names (default %s)" % ",".join(GATES),
    )
    ap.add_argument(
        "--tags",
        default=None,
        metavar="TAGS",
        help="chain tags, comma list: address gnss<g><tag>_viscap_gate on both GPUs "
        "for each tag ('' = gnss<g>_viscap_gate); overrides --gates. The generator "
        "builds one gate per GPU (tag ''), which gates every captured chain, so the "
        "default --gates already covers them all.",
    )
    ap.add_argument(
        "--start-in",
        type=float,
        default=30.0,
        metavar="S",
        help="arm: open the window this many seconds from now, frame-aligned "
        "(default %(default)s; ignored with --start-seq)",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=300.0,
        metavar="S",
        help="arm: window length, seconds (default %(default)s; ignored with --end-seq)",
    )
    ap.add_argument("--start-seq", type=int, default=None)
    ap.add_argument("--end-seq", type=int, default=None)
    ap.add_argument(
        "--every",
        type=float,
        default=5.0,
        help="watch: poll period, seconds (default %(default)s)",
    )
    a = ap.parse_args()
    nodes = [n for n in a.nodes.split(",") if n] or fleet_hosts()
    gates = [g for g in a.gates.split(",") if g]
    if a.tags is not None:
        gates = [
            "gnss%d%s_viscap_gate" % (g, t) for g in (0, 1) for t in a.tags.split(",")
        ]
    if a.cmd == "status":
        status(nodes, gates)
    elif a.cmd == "arm":
        arm(nodes, gates, a.start_in, a.duration, a.start_seq, a.end_seq)
    elif a.cmd == "disarm":
        disarm(nodes, gates)
    elif a.cmd == "watch":
        while True:
            print(time.strftime("%H:%M:%S"))
            s = status(nodes, gates)
            if not any(v.get("armed") for v in s.values()):
                print("no gate armed")
                break
            time.sleep(a.every)


if __name__ == "__main__":
    main()
