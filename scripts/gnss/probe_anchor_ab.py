#!/usr/bin/env python3
"""WHY THE TWO NOISE ANCHORS DISAGREE BY 1.4 dB (task #57 pair leg / #62).

cn0_prompt_gate measured, on gps_l5 2026-08-15: the two feeds' SIGNAL powers agree to
0.02-0.25 dB while their PROBE (noise) powers sit 1.43 dB apart -- comb-derived below
header-derived -- which shifts every strong satellite's served C/N0 by the same +1.4 dB.

Both paths ride in the SAME telemetry frame, so the comparison can be made per record,
per instance, on identical samples -- no EMA, no poll, nothing else in the loop:

  header  row()[REC_P_RE/IM]/REC_P_ENERGY : the assembler's slot, element-combined with
          the PREVIOUS record's weights (causal)
  comb    comb_epl() channel columns      : element-combined with weights ALREADY UPDATED
          by this very record (#62, non-causal), then channel-meaned here exactly as
          combdll.prompt_cn0 forms its per-record power

If the per-record ratio on PROBES is ~-1.4 dB while on SIGNAL rows it is ~0, the anchor
gap IS #62's non-causal weighting acting differently on noise than on signal, and the fix
is #62 itself. If the per-record paths AGREE on probes, the gap lives in the combiner's
EMA instead (stale content, empty-record dilution) and #62 is exonerated HERE.

    ./probe_anchor_ab.py --chain gps_l5 --prns 3,17,19 --sig-prns 10,20,27
"""
import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "scripts", "gnss"))
from gnss_broker import telem  # noqa: E402
from gnss_broker.telem import REC_P_RE, REC_P_IM, REC_P_ENERGY  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gather", default="127.0.0.1:11061")
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--prns", default="3,17,19", help="probe PRNs")
    ap.add_argument("--sig-prns", default="", help="signal PRNs for the control arm")
    ap.add_argument("--seconds", type=float, default=15.0)
    a = ap.parse_args()
    probes = [int(x) for x in a.prns.split(",") if x]
    sigs = [int(x) for x in a.sig_prns.split(",") if x]

    host, port = telem.parse_endpoint(a.gather)
    cl = telem.TelemClient(host=host, port=port, depth=4096, chains={a.chain})
    cl.start()
    time.sleep(a.seconds)
    wins = cl.windows(a.chain, lag=1)
    frames = {}
    for w in wins:
        for inst, f in cl.frame_set(a.chain, w).items():
            frames[(w, inst)] = f
    cl.stop()
    if not frames:
        raise SystemExit("no frames for %s" % a.chain)

    print("%d frames over %d windows" % (len(frames), len(wins)))
    print("\n            n_rec   header p    comb p     comb/header")
    for label, prns in (("PROBE", probes), ("SIGNAL", sigs)):
        for prn in prns:
            hd, cb, n = 0.0, 0.0, 0
            for (_w, _inst), f in frames.items():
                for r in range(f.n_rec):
                    if not f.has_record(r):
                        continue
                    row = f.row(r, prn)
                    cmb = f.comb_epl(r, prn)
                    if row is None or not cmb:
                        continue
                    en = row[REC_P_ENERGY]
                    if en <= 0.0:
                        continue
                    # both as |A|^2 with A energy-normalised, the shared convention
                    h = (row[REC_P_RE] ** 2 + row[REC_P_IM] ** 2) / (en * en)
                    gP = 0j
                    eP = 0.0
                    for _fid, _E, P, _L, (_wE, wP, _wL) in cmb:
                        gP += P * wP
                        eP += wP
                    if eP <= 0.0:
                        continue
                    c = (abs(gP) / eP) ** 2
                    hd += h
                    cb += c
                    n += 1
            if n == 0:
                print("  %-6s %-4d  none" % (label, prn))
                continue
            print("  %-6s %-4d %6d  %10.3e %10.3e   %+6.2f dB"
                  % (label, prn, n, hd / n, cb / n,
                     10.0 * math.log10((cb / n) / (hd / n))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
