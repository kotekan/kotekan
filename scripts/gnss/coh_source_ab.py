#!/usr/bin/env python3
"""WHY THE SAME ESTIMATOR SCORES 6x APART ON TWO FEEDS OF THE SAME RECORDS (task #47/#59).

FLEET-COH prints two arms every cycle. They poll the SAME combiner endpoints with the SAME
estimator; the only difference is where the records come from:

  arm A  telemetry (#59): the ASSEMBLER's record header, P_re/P_im/P_energy, frame-synced
  arm B  /get_records:    the COMBINER's exported records

Measured on sky 2026-08-15 with G23 held on peak (fleet q 3.3): A 20 (best inst 14),
B 123 (best inst 86) -- and arm A integrates MORE records (telem-windows 32 x 4 = 128 against
B's ~100). More data, six times worse, at the SINGLE-INSTANCE level, so it is not the combine
and it is not N. That leaves the numbers themselves.

⚠️ THE RULE THIS FOLLOWS (gnss-estimator-input-intersection): per-record is healthy and
cross-record is broken, so the fault is in a SHARED INPUT, not in the estimator. Do not compare
the two OUTPUTS -- intersect their INPUTS, hop by hop, and look at what differs.

    ./coh_source_ab.py --prn 23 --inst cx27.0

Prints, for hops both feeds carry: |A|, the phase, and -- the point -- the phase DIFFERENCE
between feeds and whether it is constant (a convention offset, harmless to a fold that
derotates) or rotating (a lost derotation, which is exactly what destroys a coherent sum).
"""
import argparse
import cmath
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "scripts", "gnss"))
from gnss_broker import telem  # noqa: E402


def combiner_records(url, prn, timeout=10.0):
    """{hop: (A, energy)} from /get_records -- arm B's feed, formed as fleet_coherent forms it."""
    with urllib.request.urlopen("%s/get_records" % url, timeout=timeout) as h:
        recs = json.loads(h.read().decode())
    out = {}
    for r in recs or []:
        if int(r.get("prn", 0)) != prn:
            continue
        for x in r.get("records") or []:
            hop, re_, im_, en = int(x[0]), float(x[1]), float(x[2]), float(x[3])
            if en > 0.0:
                out[hop] = (complex(re_, im_) / en, en)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gather", default="127.0.0.1:11061")
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--prn", type=int, default=23)
    ap.add_argument("--inst", default="cx27.0", help="instance tag, e.g. cx27.0")
    ap.add_argument("--combiner", default="http://cx27:12049/gnss0_n2combine")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--show", type=int, default=25)
    a = ap.parse_args()

    host, port = telem.parse_endpoint(a.gather)
    cl = telem.TelemClient(host=host, port=port, depth=4096, chains={a.chain})
    cl.start()
    time.sleep(a.seconds)
    tsrc, _now = cl.coherent_source(a.chain, prns={a.prn}, n_win=64, lag=1)
    cl.stop()
    tele = tsrc.get(a.inst, {}).get(a.prn, {})
    comb = combiner_records(a.combiner, a.prn)
    if not tele:
        raise SystemExit("no telemetry for %s PRN %d (instances: %s)"
                         % (a.inst, a.prn, ", ".join(sorted(tsrc))))
    if not comb:
        raise SystemExit("no /get_records rows for PRN %d at %s" % (a.prn, a.combiner))

    shared = sorted(set(tele) & set(comb))
    print("telemetry hops %d (%d..%d) | combiner hops %d (%d..%d) | SHARED %d"
          % (len(tele), min(tele), max(tele), len(comb), min(comb), max(comb), len(shared)))
    if not shared:
        print("\n⚠️ NO SHARED HOPS -- the two feeds do not even describe the same records.")
        print("   That alone would explain the 6x: the arms are integrating different data.")
        return 1

    print("\n   hop           |A| tele    |A| comb   ratio |  phase tele  phase comb   DIFF")
    prev = None
    diffs = []
    for hop in shared[-a.show:]:
        At, et = tele[hop]
        Ac, ec = comb[hop]
        d = cmath.phase(At * Ac.conjugate())
        diffs.append((hop, d))
        print("  %12d  %10.3e %10.3e %6.2f | %+10.3f %+10.3f  %+7.3f%s"
              % (hop, abs(At), abs(Ac), (abs(At) / abs(Ac)) if abs(Ac) else 0,
                 cmath.phase(At), cmath.phase(Ac), d,
                 "" if prev is None else "  (d/drec %+.3f)" % (d - prev)))
        prev = d

    # THE VERDICT. A constant offset is a convention and a coherent fold does not care. A
    # phase that WALKS is a lost derotation, and it destroys the sum in exactly this way.
    if len(diffs) >= 4:
        st = [d for _h, d in diffs]
        unwrapped = [st[0]]
        for x in st[1:]:
            k = round((unwrapped[-1] - x) / (2 * cmath.pi))
            unwrapped.append(x + 2 * cmath.pi * k)
        span = max(unwrapped) - min(unwrapped)
        drift = (unwrapped[-1] - unwrapped[0]) / (diffs[-1][0] - diffs[0][0] or 1)
        print("\n  phase difference: span %.3f rad over %d records, drift %+.3e rad/hop"
              % (span, len(unwrapped), drift))
        if span < 0.5:
            print("  -> CONSTANT. The feeds agree up to a convention; the 6x is NOT here.")
        else:
            print("  -> ROTATING. One feed carries a derotation the other does not. A coherent")
            print("     sum over %d records with %.2f rad of uncorrected walk loses the"
                  % (len(unwrapped), span))
            print("     majority of its aperture -- which is the 6x.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
