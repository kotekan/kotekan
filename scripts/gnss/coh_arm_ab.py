#!/usr/bin/env python3
"""THE TWO FLEET-COH ARMS, run side by side in ONE process, with their internals printed.

FLEET-COH reports A and B every cycle and they disagree by 6x on a held satellite. They call
the SAME estimator against the SAME combiner endpoints; the only difference is the feed:

  A  source=telemetry  (the assembler's record header, frame-synced)
  B  no source         (polls /get_records, the combiner's export)

Everything cheap has already been excluded by measurement:
  * NOT the combine -- the gap is 6x at the SINGLE INSTANCE level (best inst 14 vs 86).
  * NOT N -- arm A had MORE records available (128 vs 100) and still lost.
  * NOT phase -- coh_source_ab.py intersected the feeds hop by hop: phases identical to
    2e-21 rad/hop drift. Amplitudes differ by a CONSTANT 6.34e9 (a units convention), which
    an SNR cannot see.
  * NOT the rate fit -- hop_rate_hz only produces rate_hz/rate_sigma_hz, not deep_snr.

So stop guessing and print what the estimator itself did. Same moment, same PRN, both arms.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "scripts", "gnss"))
from gnss_broker import telem  # noqa: E402
from gnss_broker.fleet import fleet_coherent  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gather", default="127.0.0.1:11061")
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--prn", type=int, default=23)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--telem-windows", type=int, default=32)
    ap.add_argument("--min-instances", type=int, default=2)
    ap.add_argument("--min-records", type=int, default=16)
    ap.add_argument("--all-prns", action="store_true",
                    help="pass prns=None, as the broker does with its full seed set --\n"
                         "the fold's per-PRN answer must not depend on who else is in\n"
                         "the request, and this is how you find out that it does")
    ap.add_argument("--nodes", default="cx19,cx27,cx42,cx43,cx44,cx51")
    a = ap.parse_args()

    eps = []
    for n in a.nodes.split(","):
        for g in (0, 1):
            eps.append("http://%s:12049/gnss%d_n2combine" % (n, g))

    host, port = telem.parse_endpoint(a.gather)
    cl = telem.TelemClient(host=host, port=port, depth=4096, chains={a.chain})
    cl.start()
    time.sleep(a.seconds)
    src = cl.coherent_source(a.chain, prns=None if a.all_prns else {a.prn},
                             n_win=a.telem_windows, lag=1)
    cl.stop()

    print("telemetry source: %d instances, %d hops for PRN %d"
          % (len(src[0]), len(src[0].get(sorted(src[0])[0], {}).get(a.prn, {})) if src[0] else 0,
             a.prn))

    _p = None if a.all_prns else {a.prn}
    A = fleet_coherent(eps, a.min_instances, a.min_records, prns=_p, log=None,
                       floor_margin=1.0, seed=1, source=src)
    B = fleet_coherent(eps, a.min_instances, a.min_records, prns=_p, log=None,
                       floor_margin=1.0, seed=1)

    for name, R in (("A  telemetry", A), ("B  /get_records", B)):
        v = R.get(a.prn)
        print("\n=== %s ===" % name)
        if not v:
            print("  (no row -- PRN dropped before the fold)")
            continue
        for k in ("deep_snr", "best_inst_snr", "n_src", "records", "deep_records",
                  "coherence_s", "floor", "align", "present", "amplitude", "deep_amplitude",
                  "split_imbalance", "rate_hz"):
            if k in v:
                print("  %-16s %s" % (k, v[k]))
        if v.get("dropped"):
            print("  dropped         %s" % v["dropped"])
        pi = v.get("per_inst") or {}
        if pi:
            print("  per_inst (deep_snr each):")
            for u in sorted(pi, key=lambda x: -pi[x])[:6]:
                print("     %-52s %8.2f" % (str(u)[-46:], pi[u]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
