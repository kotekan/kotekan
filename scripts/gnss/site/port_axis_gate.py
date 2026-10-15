#!/usr/bin/env python3
"""Check that the two ports of each node agree on the FPGA seq axis.

After a node restart an F-engine link can come back from the switch-port bounce desynced and
dark: valid CRS packets at the full rate, all-zero payloads, and a seq axis that resumed behind
the fleet because the board's counter paused while the link was down. The capture worker tracks
that stale axis faithfully, so frames advance at real time, all zero, stamped old, and every
health counter (packet rate, drops, thread CPU, GPU load, record age at the gather) stays
normal. Two things show it: the gather's last_win for that instance sits a constant N windows
behind its siblings, and the two ports' newest FPGA_SEQ_NUM differ by N frames. Healthy ports
differ by about one frame (ring-position jitter).

usage:  port_axis_gate.py [node ...]     (default: the whole fleet)
exit 0 = every node's ports agree within TOL_FRAMES; exit 1 = a stale axis or an unreachable
node, named.
"""
import json
import sys
import urllib.request

NODES_DEFAULT = ["cx19", "cx27", "cx42", "cx43", "cx44", "cx51"]
TOL_FRAMES = 4  # ring-position jitter is +-1 frame; 4 leaves headroom
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
        verdict = (
            "ok"
            if abs(df) <= TOL_FRAMES
            else "STALE AXIS -- an F-engine link is desynced"
        )
        print(
            f"  {n}: port1-port0 {df:+9.1f} frames ({df * FRAME_S:+8.2f} s)  {verdict}"
        )
        if abs(df) > TOL_FRAMES:
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
