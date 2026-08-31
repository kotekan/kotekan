#!/usr/bin/env python3
"""THE TWO PORTS OF A NODE MUST AGREE ON THE FPGA SEQ AXIS.

WHY THIS EXISTS (2026-08-31, cx19 port 1). After a node restart, an F-engine
link can come back from the switch-port bounce DESYNCED AND DARK: valid CRS
packets at the full rate, ZERO payloads, and a seq axis that resumed ~25 s
behind the fleet (the board's counter paused while the link was down). The
capture worker then tracks that stale axis faithfully -- frames advance at
exactly real time, all-zero, stamped hundreds of frames old. EVERY health
instrument stays green: packet counters full rate, zero drops, all threads
accruing CPU, GPU 75% busy, records arriving at the gather with age ~20 ms.
The ONLY tells are (a) the gather's last_win for that instance pinned a
constant N windows behind its siblings, and (b) this check: the two ports'
newest FPGA_SEQ_NUM differ by N frames instead of <=2.

Measured that day: cx19 port1-port0 = -608.0 frames (-25.50 s); the five
healthy nodes all read +1.0 frame (ring-position jitter).

usage:  port_axis_gate.py [node ...]     (default: the whole fleet)
exit 0 = every node's ports agree within TOL; exit 1 = a stale axis, named.
"""
import json
import sys
import urllib.request

NODES_DEFAULT = ["cx19", "cx27", "cx42", "cx43", "cx44", "cx51"]
TOL_FRAMES = 4  # ring-position jitter is +-1; leave headroom, catch 600+
SAMPLES_PER_FRAME = 8192
FRAME_S = 0.04194304


def newest_seq(node: str, buf: str) -> int:
    url = f"http://{node}:12048/buffer_frame?name={buf}"
    with urllib.request.urlopen(url, timeout=15) as r:
        d = json.load(r)
    return int(d["metadata"]["FPGA_SEQ_NUM"])


def main() -> int:
    nodes = sys.argv[1:] or NODES_DEFAULT
    bad = 0
    for n in nodes:
        try:
            s0 = newest_seq(n, "host_voltage_buffer_0")
            s1 = newest_seq(n, "host_voltage_buffer_1")
        except Exception as e:  # a dead node is its own, different alarm
            print(f"  {n}: UNREACHABLE ({e})")
            bad += 1
            continue
        df = (s1 - s0) / SAMPLES_PER_FRAME
        verdict = "ok" if abs(df) <= TOL_FRAMES else "STALE AXIS -- an F-engine link is desynced"
        print(f"  {n}: port1-port0 {df:+9.1f} frames ({df * FRAME_S:+8.2f} s)  {verdict}")
        if abs(df) > TOL_FRAMES:
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
