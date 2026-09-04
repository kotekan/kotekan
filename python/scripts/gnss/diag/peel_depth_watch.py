#!/usr/bin/env python3
"""Sample peel depth at 1 Hz from the combiner -- an instrument that cannot alias.

WHY THIS EXISTS. The peel health line in the kotekan log is emitted every 1000 frames =
9.984 s, and a GPS frame is 30 s = 3.005 x 9.984 s. Consecutive lines therefore sample almost
exactly the SAME phase of the frame and take ~30 min to sweep it. On 2026-07-25 that made a
constructed-bit run look like a total collapse -- ten consecutive lines reading 0 peeling --
when the viewer, updating continuously, showed it oscillating between 20-30 dB and 0 dB. The
log said "dead"; the truth was "alive 60% of the time". Any instrument whose sampling period
is commensurate with the signal's period will lie to you like this.

Constructed bits only cover subframes 1-3, so the expected pattern is ~18 s of depth followed
by ~12 s of nothing (sf4+sf5), CONSTELLATION-SYNCHRONISED so every GPS satellite goes dark
together. This prints depth per PRN once a second so that pattern is visible directly.

    diag/peel_depth_watch.py [seconds] [--url http://localhost:12048] [--combiner l1_gps_combiner]
"""
import argparse
import json
import math
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("secs", nargs="?", type=float, default=90.0)
    ap.add_argument("--url", default="http://localhost:12048")
    ap.add_argument("--combiner", default="l1_gps_combiner")
    a = ap.parse_args()
    end = time.time() + a.secs
    t0 = time.time()
    print("t(s)  n_deep  depths per PRN (dB, from deep/peel_deep)")
    while time.time() < end:
        try:
            rows = json.loads(urllib.request.urlopen(
                "%s/%s/get_status" % (a.url, a.combiner), timeout=3).read().decode())
        except Exception as e:
            print("  poll failed: %s" % e)
            time.sleep(1.0)
            continue
        out = []
        for r in rows:
            # NB the status field is deep_amplitude, not "deep" (the viewer renames it).
            d, pd = r.get("deep_amplitude"), r.get("peel_deep")
            dsnr, dfl = r.get("deep_snr") or 0.0, r.get("deep_floor") or 0.0
            psnr = r.get("peel_deep_snr") or 0.0
            if not (d and pd and d > 0 and pd > 0 and dsnr > 1.1 * dfl):
                continue
            # At floor the residual is not a measurement -- report the detection limit as a
            # BOUND, exactly as the viewer does (2026-07-24: unbounded division read >=61 dB).
            if psnr <= 1.1 * dfl:
                out.append((r["prn"], 20.0 * math.log10(dsnr / max(dfl, 1e-9)), True))
            else:
                out.append((r["prn"], 20.0 * math.log10(d / pd), False))
        out.sort(key=lambda x: -x[1])
        print("%5.1f %6d  " % (time.time() - t0, len(out))
              + " ".join("%d:%s%.0f" % (p, ">" if b else "", v) for p, v, b in out[:10]))
        time.sleep(1.0)


if __name__ == "__main__":
    main()
