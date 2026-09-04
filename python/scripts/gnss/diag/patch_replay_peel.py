#!/usr/bin/env python3
"""Toggle the P9 L5 pilot peel in config/replay_l5_peel.yaml -- the A/B lever.

    patch_replay_peel.py on|off

ON  = exactly the reverted 6ba3aaae recipe: epl frames 15360 -> 23040 (6 output rows/spec:
      rows 4/5 carry the peel residual), tracker peel params on all three chains, combiner
      peel_depth. OFF = revert to the plain replay.
Idempotent either way; prints what state it left the file in.
"""
import re
import sys

P = "config/replay_l5_peel.yaml"
TRK = """          peel: true
          peel_gain_alpha: 0.01
          peel_warmup_records: 100
          peel_fll_gain: 0.05
"""

def main(mode):
    s = open(P).read()
    on = "peel: true" in s
    if mode == "on" and not on:
        s = s.replace("+ 15360 * ", "+ 23040 * ")
        # insert tracker peel params right after each seed_endpoint line
        s = re.sub(r'(^ +seed_endpoint: "/(?:gps|gal|bds)_track/set_seeds"\n)',
                   lambda m: m.group(1) + TRK, s, flags=re.M)
        # combiner: peel_depth beside the existing bit_export
        s = s.replace("bit_export: true", "bit_export: true, peel_depth: true")
    elif mode == "off" and on:
        s = s.replace("+ 23040 * ", "+ 15360 * ")
        s = s.replace(TRK, "")
        s = s.replace("bit_export: true, peel_depth: true", "bit_export: true")
    open(P, "w").write(s)
    n_peel = s.count("peel: true")
    n_frame = s.count("+ 23040 * ")
    n_depth = s.count("peel_depth: true")
    print("state: peel:true x%d, 23040-frames x%d, peel_depth x%d  (want 3/3/3 on, 0/0/0 off)"
          % (n_peel, n_frame, n_depth))
    ok = (n_peel, n_frame, n_depth) == ((3, 3, 3) if mode == "on" else (0, 0, 0))
    print("PATCH %s" % ("OK" if ok else "INCONSISTENT -- do not run the bench"))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
