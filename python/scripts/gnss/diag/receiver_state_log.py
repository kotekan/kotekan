#!/usr/bin/env python3
"""Long-baseline accumulator for the receiver-state exports.

    nohup nice -n 15 taskset -c 14-19 diag/receiver_state_log.py \
        --out /tmp/rxstate/state.jsonl --interval 10 &

Read-only. Downsamples the 1 Hz exports to one record per --interval and appends compact
JSONL, so multi-hour questions have data without filling the cache directory (8 chains at
1 Hz is ~415 MB/day raw; at 10 s it is ~41 MB/day).

THE QUESTION THIS EXISTS FOR
----------------------------
Each dongle's LO error is dominated by a per-tuning frac-N synthesis CONSTANT -- measured
-0.0958 ppm (L1) / -0.0159 (L2C) / +0.0286 (L5), which is why bands must not be averaged.
But all three airspys are disciplined by ONE GPSDO, so the model should be

    ppm_band(t) = C_band  +  delta(t)          C per dongle, delta COMMON

If delta is real and measurable, the L1 dongle -- 4 chains, the most satellites, the
quietest residuals -- measures it for the whole receiver, and the L5/L2C chains inherit
their drift instead of each solving it from 2-3 satellites. That is the cross-band assist
S5 needs, and it cannot be settled on a 15 minute window: over 10 min the per-band drifts
(-0.0005 to +0.005 ppm) are comparable to the noise. Hours will separate them.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import receiver_state  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.expanduser("~/.cache/kotekan_gps/state"))
    ap.add_argument("--out", default="/tmp/rxstate/state.jsonl")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--max-age-s", type=float, default=30.0)
    ap.add_argument("--hours", type=float, default=0.0, help="0 = run until killed")
    args = ap.parse_args()

    d = os.path.dirname(args.out)
    if d:
        os.makedirs(d, exist_ok=True)
    end = time.time() + args.hours * 3600.0 if args.hours > 0 else None
    n = 0
    with open(args.out, "a") as f:
        while end is None or time.time() < end:
            t = time.time()
            try:
                names = sorted(os.listdir(args.dir))
            except Exception:
                time.sleep(args.interval)
                continue
            for name in names:
                if not name.endswith(".json"):
                    continue
                rec = receiver_state.read_state(os.path.join(args.dir, name),
                                                max_age_s=args.max_age_s, t_now=t)
                if rec is None:
                    continue
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                n += 1
            f.flush()
            time.sleep(args.interval)
    print("wrote %d records" % n)


if __name__ == "__main__":
    main()
