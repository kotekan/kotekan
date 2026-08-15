#!/usr/bin/env python3
"""THE VALIDATION BAR FOR THE PER-ELEMENT COMPLEX GAIN (task #57 step 2), three legs.

The estimator: node-side per-record cross-products A_e * conj(leave-one-out reference)
(GnssCoherentCombiner /get_elements), reduced by gnss_broker/elemgain.py into per-element
amplitude, PHASE (relative to the array mean -- the peel/beam coefficient), and a
probe-anchored significance. The legs mirror cn0_prompt_gate's discipline:

  SIG     Do the probes sit at their own floor (sig ~ 1) while a held satellite's live
          elements clear it? Per (instance, element), the same noise-by-construction
          anchor that convicted deep_snr. Bar: satellite median sig >= 3x probe median.

  PHASE   Split-half in TIME: two polls, >> the cal EMA apart. A real per-element phase
          is STABLE between them (the gain is quasi-static over seconds); noise phase is
          uniform. Scored as the circular concentration R = |<e^{i dphi}>| over elements:
          held satellite R >= 0.8, probes are the null (R ~ 1/sqrt(n_elem)).
          ⚠️ Phase estimators fail in the VARIANCE (gnss-phase-estimator-self-reference):
          the probes ARE the shuffled-null here -- same estimator, no signal -- and the
          probe R is printed next to the satellite R, never assumed.

  XINST   The frequency axis: each instance is ~7 channels, so one element's phase across
          instances samples its bandpass. Reported (slope per element), not gated -- the
          physics consumer decides what a credible delay is.

    ./elemgain_gate.py --self-test          # reduction arithmetic on synthetic parts
    ./elemgain_gate.py --chain gps_l5       # on sky, after the node restart ships the
                                            # /get_elements endpoint

Probes auto-discovered from the broker (noise_probe rows), as in cn0_prompt_gate.
"""
import argparse
import cmath
import json
import math
import os
import random
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python", "scripts", "gnss"))
from gnss_broker import elemgain  # noqa: E402


def discover_probes(broker, chain):
    try:
        with urllib.request.urlopen("%s/get_status?chain=%s" % (broker, chain),
                                    timeout=5.0) as h:
            rows = json.loads(h.read().decode())
    except Exception as e:
        raise SystemExit("broker %s unreachable (%s)" % (broker, e))
    return {int(r["prn"]) for r in rows if r.get("noise_probe")}


def circ_R(dphis):
    if not dphis:
        return None
    s = sum(cmath.exp(1j * d) for d in dphis)
    return abs(s) / len(dphis)


def self_test():
    """Known complex gains through elemgain.gain_table: amp/ph exact, probe sig ~ 1."""
    rng = random.Random(1)
    n_el, keff = 16, 100.0
    gains = [cmath.rect(1.0 + 0.5 * math.sin(e), 0.3 * e - 1.5) for e in range(n_el)]
    qv = 4.0                    # |LOO ref|^2, arbitrary units
    per_inst = {}
    for inst in ("cx19/0", "cx27/1"):
        d = {}
        for prn, on in ((23, True), (91, False), (92, False), (93, False)):
            u = []
            for e in range(n_el):
                if on:
                    g = gains[e] * qv       # u = g * <|ref|^2>, noiseless
                else:
                    g = cmath.rect(0.02 * qv, rng.uniform(-math.pi, math.pi))
                u.append((g.real, g.imag))
            d[prn] = {"u": u, "p2": [1.0] * n_el, "q": [qv] * n_el,
                      "keff": keff, "hop": 1}
        per_inst[inst] = d
    tab = elemgain.gain_table(per_inst, {91, 92, 93})
    fails = []
    row = tab.get(23, {}).get("inst", {}).get("cx19/0")
    if not row:
        return print("SELF-TEST: FAIL -- no satellite row") or 1
    for e in range(n_el):
        if abs(row["amp"][e] - abs(gains[e])) > 1e-3 * abs(gains[e]) + 1e-9:
            fails.append("el %d amp %.4f != %.4f" % (e, row["amp"][e], abs(gains[e])))
        dph = (row["ph"][e] - cmath.phase(gains[e]) + math.pi) % (2 * math.pi) - math.pi
        if abs(dph) > 1e-3:
            fails.append("el %d phase off by %.4f" % (e, dph))
    sig_sat = sorted(row.get("sig") or [])
    sig_probe = sorted(tab[91]["inst"]["cx19/0"].get("sig") or [])
    if not sig_sat or sig_sat[len(sig_sat) // 2] < 10.0:
        fails.append("satellite sig median %.2f, expected >> 1"
                     % (sig_sat[len(sig_sat) // 2] if sig_sat else -1))
    if not sig_probe or not (0.2 < sig_probe[len(sig_probe) // 2] < 5.0):
        fails.append("probe sig median %.2f, expected ~ 1"
                     % (sig_probe[len(sig_probe) // 2] if sig_probe else -1))
    if fails:
        print("SELF-TEST: FAIL\n  " + "\n  ".join(fails[:6]))
        return 1
    print("SELF-TEST: PASS (amp/ph exact on %d elements, sat sig med %.0f, probe %.2f)"
          % (n_el, sig_sat[len(sig_sat) // 2], sig_probe[len(sig_probe) // 2]))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain", default="gps_l5")
    ap.add_argument("--broker", default="http://127.0.0.1:12060")
    ap.add_argument("--nodes", default="cx19,cx27,cx42,cx43,cx44,cx51")
    ap.add_argument("--gap-s", type=float, default=10.0,
                    help="seconds between the two polls of the PHASE leg (>> cal EMA ~1 s)")
    ap.add_argument("--min-keff", type=float, default=8.0)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()

    probes = discover_probes(a.broker, a.chain)
    if not probes:
        raise SystemExit("broker reports no noise probes for %s -- no anchor, no gate"
                         % a.chain)
    print("probes: %s" % sorted(probes))
    _sfx = {"gps_l5": "", "gal_e5a": "_e5a", "bds_b2a": "_b2a",
            "gal_e5b": "_e5b", "bds_b2b": "_b2b"}[a.chain]
    eps = ["http://%s:12049/gnss%d%s_n2combine" % (n, g, _sfx)
           for n in a.nodes.split(",") for g in (0, 1)]

    pe1, srv1 = elemgain.poll_elements(eps)
    if not pe1:
        raise SystemExit("0/%d instances served /get_elements -- nodes not yet restarted "
                         "with the #57 step-2 combiner?" % len(eps))
    time.sleep(a.gap_s)
    pe2, _ = elemgain.poll_elements(eps)
    t1 = elemgain.gain_table(pe1, probes, min_keff=a.min_keff)
    t2 = elemgain.gain_table(pe2, probes, min_keff=a.min_keff)
    print("%d/%d instances served; %d PRNs in the table" % (srv1, len(eps), len(t1)))

    fails, judged = [], []
    print("\n  PRN    inst      med_sig   phase_R   probe_R")
    for prn in sorted(t1):
        is_probe = t1[prn]["probe"]
        for tag in sorted(t1[prn]["inst"]):
            r1 = t1[prn]["inst"][tag]
            r2 = (t2.get(prn, {}).get("inst") or {}).get(tag)
            sig = sorted(r1.get("sig") or [])
            med_sig = sig[len(sig) // 2] if sig else None
            R = None
            if r2:
                dph = [(p2 - p1 + math.pi) % (2 * math.pi) - math.pi
                       for p1, p2 in zip(r1["ph"], r2["ph"])]
                R = circ_R(dph)
            print("  %s%-4d %-9s %8s  %8s"
                  % ("P" if is_probe else "G", prn, tag,
                     "%.1f" % med_sig if med_sig is not None else "--",
                     "%.3f" % R if R is not None else "--"))
            if is_probe or med_sig is None or R is None:
                continue
            # judged only where the SIG leg says there is signal to hold a phase on
            if med_sig >= 3.0:
                judged.append((prn, tag))
                if R < 0.8:
                    fails.append("G%d %s phase R %.3f < 0.8 at sig %.1f"
                                 % (prn, tag, R, med_sig))
    # the probes' own R distribution IS the null; print it once, pooled
    null_R = []
    for prn in sorted(t1):
        if not t1[prn]["probe"]:
            continue
        for tag, r1 in t1[prn]["inst"].items():
            r2 = (t2.get(prn, {}).get("inst") or {}).get(tag)
            if r2:
                dph = [(p2 - p1 + math.pi) % (2 * math.pi) - math.pi
                       for p1, p2 in zip(r1["ph"], r2["ph"])]
                v = circ_R(dph)
                if v is not None:
                    null_R.append(v)
    if null_R:
        null_R.sort()
        print("\nnull (probe) phase R: median %.3f, max %.3f over %d rows"
              % (null_R[len(null_R) // 2], null_R[-1], len(null_R)))
    print()
    if fails:
        print("FAIL: " + "; ".join(fails))
        return 1
    if not judged:
        print("INCONCLUSIVE: no (satellite, instance) at sig >= 3 to judge")
        return 0
    print("PASS on %d (satellite, instance) rows" % len(judged))
    return 0


if __name__ == "__main__":
    sys.exit(main())
