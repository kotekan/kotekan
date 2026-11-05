#!/usr/bin/env python3
"""Rolling-fit canary: residual / raw power per channel of each GPU from short dump windows, every --every s,
with each GPU's rolling-fit state. Compare the armed GPU's channels with the other GPU's (same node, same sky).

    dtv_canary.py --node cx51 [--seconds 6] [--rounds 10] [--every 60]

Run on cf06 (dtv_live.py's WORK); the window files are deleted on the node and here after each round.
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dtv_live  # noqa: E402
import dtv_state  # noqa: E402  (dtv_live put its directory on the path)


def ratios(a, tag, g):
    fr = list(dtv_state.frames(dtv_live.tag_files(a, tag, g)))
    out = {}
    if fr:
        for c, fid in enumerate(fr[0]["freq"]):
            use = [f for f in fr if (f["act"][c] >= 2).any()]
            if use:
                out[int(fid)] = sum(f["pw_res"][c].sum() for f in use) / max(sum(f["pw_raw"][c].sum() for f in use), 1e-9)
    return out, len(fr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--node", required=True)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--every", type=float, default=60.0)
    a = ap.parse_args()
    a.beam = False
    for _ in range(a.rounds):
        t0 = time.time()
        a.tag = "can" + time.strftime("%H%M%S", time.gmtime())
        dtv_live.cmd_window(a)
        d = os.path.join(dtv_live.WORK, a.node, a.tag)
        names = sorted(os.listdir(d)) if os.path.isdir(d) else []
        if names:
            subprocess.run(["ssh", a.node, "cd /tmp/gnss/dtv && rm -f " + " ".join(names)], check=False)
        parts = []
        for g in (0, 1):
            r, nf = ratios(a, a.tag, g)
            s = dtv_live.get(a, dtv_live.cmd_path(g) + "/status").get("rolling", {})
            parts.append("g%d %s refits %s (%d fr) %s" % (g, "ROLL" if s.get("enable") else "loop", s.get("refits"), nf,
                         " ".join("%d:%.4f" % kv for kv in sorted(r.items()))))
        print("%s %s res/raw | %s" % (time.strftime("%H:%M:%SZ", time.gmtime()), a.node, " | ".join(parts)), flush=True)
        dtv_live.drop_window(a, a.tag)
        time.sleep(max(0.0, a.every - (time.time() - t0)))


if __name__ == "__main__":
    main()
