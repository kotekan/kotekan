"""VETO the epochs where a satellite sits close enough to boresight to rail the quantiser.

    gnss_beam_veto.py --obs p2_*.jsonl --el0 82.7 --az0 176.4 --deg 5 --suffix v

⚠️ THE ARRAY IS SHARED, SO THE VETO MUST BE CROSS-CHAIN. One satellite near boresight rails
the 4+4b quantiser for EVERYTHING: L5, E5a, E5b, B2a and B2b all ride the same nibbles
([[chord-nothing-is-per-node]] in the frequency domain). A per-chain veto would clean the
chain carrying the bright satellite and leave the other four contaminated, which is worse
than not filtering at all -- the four would then look like clean controls.

MEASURED ON 2026-08-25/26 (51,607 samples, 7,480 epochs), with the beam removed first by
subtracting each sample's own 2-degree radius-bin median:

    closest sat to boresight   residual of FAR (>40 deg) sats   tracked sats/epoch
        0- 3 deg                      +2.70 dB                        4.0
        3- 5 deg                      +2.04 dB                        6.0
        5- 8 deg                      +0.68 dB                        6.0
        8-12 deg                      +0.11 dB                        6.0
       12-20 deg                      -0.16 dB                        7.0
       20-90 deg                      -0.30 dB                        5.0

So the contamination is TWO effects at once, and they push opposite ways:
  * the survivors read HIGH by 2-3 dB (railing is not a clean attenuation -- it redistributes
    power, and the 5-minute probe pedestal cannot follow a transit that lasts minutes, so the
    debias under-subtracts exactly when it matters);
  * and the POPULATION shrinks, 7 satellites per epoch down to 4 -- the losses the archive
    can never show, because it only records satellites that were present. Judging the bias on
    the survivors alone is the [[chord-a-gate-that-cannot-fail]] mistake in map form.

⚠️ CONSEQUENCE FOR THE MAP: vetoing removes the samples nearest boresight, i.e. THE MAIN LOBE
ITSELF. That is the honest outcome, not a bug -- the main lobe cannot be measured with this
data because the main lobe is what breaks the measurement. Render the vetoed map for the
wide-angle pattern and quote the core from the unvetoed one as a LOWER BOUND with the bias
named.

@author Keith Vanderlinde
"""
import argparse
import json
import math
from collections import defaultdict


def uvec(el_deg, az_deg):
    e, a = math.radians(el_deg), math.radians(az_deg)
    return (math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", nargs="+", required=True)
    ap.add_argument("--el0", type=float, required=True)
    ap.add_argument("--az0", type=float, required=True)
    ap.add_argument("--deg", type=float, default=5.0)
    ap.add_argument("--suffix", default="v")
    args = ap.parse_args()

    b = uvec(args.el0, args.az0)

    def sep(el, az):
        v = uvec(el, az)
        return math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(v, b))))))

    # PASS 1: the closest approach at each epoch, pooled over EVERY chain.
    closest = defaultdict(lambda: 1e9)
    for p in args.obs:
        for line in open(p):
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = round(d["t"])
            s = sep(d["el"], d["az"])
            if s < closest[t]:
                closest[t] = s
    vetoed = {t for t, s in closest.items() if s < args.deg}
    print("epochs %d, vetoed %d (%.1f%%) at < %.1f deg from (el %.1f, az %.1f)"
          % (len(closest), len(vetoed), 100.0 * len(vetoed) / max(len(closest), 1),
             args.deg, args.el0, args.az0))

    # PASS 2: drop every row in a vetoed epoch -- not just the bright satellite.
    for p in args.obs:
        out = p.replace(".jsonl", "_%s.jsonl" % args.suffix)
        n = k = 0
        with open(p) as fh, open(out, "w") as oh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                n += 1
                if round(d["t"]) in vetoed:
                    continue
                oh.write(line)
                k += 1
        print("   %-28s %6d -> %6d rows" % (p.rsplit("/", 1)[-1], n, k))


if __name__ == "__main__":
    main()
