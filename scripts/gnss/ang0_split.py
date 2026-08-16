#!/usr/bin/env python3
"""DO THE INSTANCES BUILD THE SAME REPLICA CARRIER PHASE? -- #72's decisive test.

THE QUESTION. The across-band phase carries an INTERCEPT that differs per (instance, PRN):
all twelve instances agree on the delay SLOPE to ~3 ns (|R| 0.997-1.000 against noise probes
at 0.09-0.19), but their constants do not agree across satellites (mean |R| 0.220 over 10
sats, null E 0.280, 95% 0.536). That is a carrier-phase constant built PER SATELLITE, i.e.
inside the replica -- and by the lockstep rule it is a BUG, because an instance is only
`freq_id mod 8` routing ([[chord-nothing-is-per-node]]).

WHY IT NEEDED NEW TELEMETRY. Every input to ang0 = 2*pi*frac(f*n0/fs) checks out fleet-common
from outside: f_offset identical in every generated config, doppler/ctrim from the shared
broker seed, record UTC spread EXACTLY 0 at double precision. None of that clears ang0,
because REC_DOPPLER is a float32 whose ulp at 10 kHz is ~1e-3 Hz, while the absolute-sample
lever (n0 ~ 1.9e15, 6.8 days) turns 5e-7 Hz into a FULL RADIAN. The telemetry was four orders
too coarse to see the effect size -- an experiment that cannot succeed
([[experiment-that-cannot-succeed]]). So the despread now records what it actually handed the
kernel and ships it in REC_ANG0 / REC_PHI_DDOP.

READING THE RESULT, for the same (PRN, absolute window, record slot):

  ang0 EQUAL across instances     -> ang0 is EXONERATED. The intercept is built somewhere else
                                     in the synthesis (Phi, the per-channel filter) -- look there.
  ang0 DIFFERS across instances   -> THE BUG, LOCATED. Every input is fleet-common, so a
                                     difference in the output means the inputs are not actually
                                     identical at double precision, and the ang0 spread says by
                                     how much: df = d(ang0)/(2*pi*n0/fs).

  phi_ddop spread ~ refresh_hz    -> instances hold Phi built at DIFFERENT Dopplers (ensure_phi
                                     only rebuilds past refresh_hz), the other per-(instance,PRN)
                                     state. Small spread exonerates it.

⚠️ COMPARE ONLY WITHIN ONE (window, record slot). ang0 moves every record by design -- the
Doppler is re-propagated -- so pooling records manufactures a spread that is not a disagreement.

    ./ang0_split.py --chain gps_l5          # ON cf06
"""
import argparse
import json
import math
import statistics
import sys
import os
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "scripts", "gnss"))
from gnss_broker import telem  # noqa: E402

FS = 3.2e9
REC_ANG0 = telem.REC_ANG0
REC_PHI_DDOP = telem.REC_PHI_DDOP


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gather", default="127.0.0.1:11061")
    ap.add_argument("--broker", default="http://127.0.0.1:12060")
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--min-inst", type=int, default=10)
    a = ap.parse_args()

    try:
        with urllib.request.urlopen("%s/%s/get_status" % (a.broker.rstrip("/"), a.chain),
                                    timeout=10) as r:
            rows = json.loads(r.read().decode())
        probes = {int(x["prn"]) for x in rows if x.get("noise_probe")}
        cn0 = {int(x["prn"]): x.get("cn0_prompt_db") for x in rows}
    except Exception:                                                  # noqa: BLE001
        probes, cn0 = set(), {}

    host, port = telem.parse_endpoint(a.gather)
    cl = telem.TelemClient(host=host, port=port, depth=2048, chains={a.chain})
    cl.start()
    time.sleep(a.seconds)
    wins = cl.windows(a.chain, lag=1)[-a.windows:]

    # {prn: [ (spread_rad, spread_ddop_hz, n_inst, per_inst_dict) ]} -- one entry per record
    acc, sample = {}, {}
    for w in wins:
        fs_ = cl.frame_set(a.chain, w)
        if len(fs_) < a.min_inst:
            continue
        n_rec = max((f.n_rec for f in fs_.values()), default=0)
        for r in range(n_rec):
            per = {}
            for inst, f in fs_.items():
                if r >= f.n_rec or not f.has_record(r):
                    continue
                for prn in f.prns():
                    row = f.row(r, prn)
                    if row is None:
                        continue
                    g, d = row[REC_ANG0], row[REC_PHI_DDOP]
                    if g != g:          # NaN = this producer built no job
                        continue
                    per.setdefault(prn, {})[inst] = (g, d)
            for prn, d in per.items():
                if len(d) < a.min_inst:
                    continue
                gs = [v[0] for v in d.values()]
                ds = [v[1] for v in d.values() if v[1] == v[1]]
                # ang0 is an angle: compare it as one, so 0.01 vs 6.27 is 0.02 rad apart.
                mu = math.atan2(sum(math.sin(g) for g in gs), sum(math.cos(g) for g in gs))
                dev = [abs((g - mu + math.pi) % (2 * math.pi) - math.pi) for g in gs]
                acc.setdefault(prn, []).append(
                    (max(dev), (max(ds) - min(ds)) if len(ds) > 1 else 0.0, len(d)))
                sample.setdefault(prn, (w, r, dict(d)))
    cl.stop()
    if not acc:
        raise SystemExit("no records with >= %d instances -- run this ON cf06, and check the "
                         "nodes are on the v4 telemetry (RECORD_FLOATS 28)" % a.min_inst)

    print("\nang0 SPREAD ACROSS INSTANCES, within one (window, record slot)\n")
    print("  %-5s %-7s %-7s %-8s %-12s %-12s %-12s %s"
          % ("prn", "cn0", "n_rec", "n_inst", "med|dev| rad", "max|dev| rad",
             "ddop spread", ""))
    verdict_bug = []
    for prn in sorted(acc):
        v = acc[prn]
        med = statistics.median(x[0] for x in v)
        mx = max(x[0] for x in v)
        dd = statistics.median(x[1] for x in v)
        print("  %-5d %-7s %-7d %-8d %-12.4g %-12.4g %-12.4g %s"
              % (prn, ("%.1f" % cn0[prn]) if cn0.get(prn) is not None else "--",
                 len(v), max(x[2] for x in v), med, mx, dd,
                 "probe" if prn in probes else ""))
        if med > 1e-3:
            verdict_bug.append(prn)

    # float32 carries ~1e-7 rad here, so anything above ~1e-6 is real and not quantisation.
    print("\n  float32 resolution at this magnitude is ~1e-7 rad, so a spread above ~1e-6 is "
          "REAL.")
    if verdict_bug:
        allmed = statistics.median(statistics.median(x[0] for x in acc[p])
                                   for p in verdict_bug)
        # invert the lever: what Doppler difference would explain that spread?
        for days in (3.0, 6.8):
            n0 = days * 86400.0 * FS
            print("    ang0 spread %.3g rad  =>  %.3g Hz of Doppler disagreement at %.1f days "
                  "of uptime" % (allmed, allmed / (2 * math.pi * n0 / FS), days))
        print("\n  => ang0 DIFFERS between instances on %d PRN(s): %s"
              % (len(verdict_bug), verdict_bug))
        print("     THE BUG IS LOCATED. Every ang0 input is fleet-common by construction, so a")
        print("     difference in the output means they are not identical at double precision.")
        w, r, d = sample[verdict_bug[0]]
        print("\n     sample -- PRN %d, window %d slot %d:" % (verdict_bug[0], w, r))
        for inst in sorted(d):
            print("       %-8s ang0 %+12.8f rad   phi_ddop %+10.4g Hz" % (inst, d[inst][0],
                                                                          d[inst][1]))
    else:
        print("\n  => ang0 AGREES across instances (median spread below 1e-3 rad).")
        print("     ang0 IS EXONERATED. The per-(instance, PRN) intercept is built elsewhere in")
        print("     the synthesis -- the per-channel filter Phi is the remaining candidate, and")
        print("     phi_ddop above says whether the instances even built it at the same Doppler.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
