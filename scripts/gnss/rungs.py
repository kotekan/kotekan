#!/usr/bin/env python3
"""Show the coherence ladder RUNG BY RUNG, across the fleet -- the 8.18.3 item-1 measurement.

WHY. `deep_snr` is a MAX over the ladder, so a small value is ambiguous in exactly the way that
matters. Two failures look identical in the winner alone and want opposite fixes:

  SNR-LIMITED   every rung sits near the floor, and the LONGEST rung is the best of them.
                Coherence is fine; there simply is not enough signal. Fix = more SNR per record
                (cross-node coherent combining, 8.18.3 item 2).

  PHASE-LIMITED short rungs rise well above the floor and long rungs collapse back toward it,
                so the ladder peaks at a knee partway down. Coherence is the binding constraint
                and integrating longer actively hurts. Fix = a better carrier / shorter span.

The discriminator is the SHAPE: snr/floor as a function of rung span. An ideal SNR-limited
detector gains sqrt(T) with span, so snr/sqrt(span) is FLAT; a phase-limited one falls off. That
normalized column is printed alongside, because a raw snr column rising with length is what
"working" and "not cohering at all" both look like.

usage:  rungs.py [--prn 3,18,26] [--top 8] [--watch 20]
"""
import argparse
import json
import math
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/kvand/gnss/kotekan/python/scripts/gnss")

DEFAULT_ENDPOINTS = [
    "http://cx19:12049/gnss0_combine", "http://cx51:12049/gnss0_combine",
] + ["http://%s:12049/gnss%d_combine" % (n, g)
     for n in ("cx27", "cx42", "cx43", "cx44", "cx47", "cx52") for g in (0, 1)]

ap = argparse.ArgumentParser()
ap.add_argument("--endpoints", default=",".join(DEFAULT_ENDPOINTS))
ap.add_argument("--prn", default="", help="comma list; default = the strongest few")
ap.add_argument("--top", type=int, default=6)
ap.add_argument("--watch", type=float, default=0.0, help="repeat every N s (0 = once)")
a = ap.parse_args()

EPS = [e.strip() for e in a.endpoints.split(",") if e.strip()]
WANT = set(int(x) for x in a.prn.replace(" ", "").split(",") if x)


def get(url):
    try:
        return json.load(urllib.request.urlopen(url + "/get_status", timeout=6))
    except Exception:
        return None


def once():
    with ThreadPoolExecutor(max_workers=len(EPS)) as ex:
        rows = list(ex.map(get, EPS))
    live = [(EPS[i], r) for i, r in enumerate(rows) if r]
    if not live:
        print("no combiner answered")
        return

    # Collect every instance's ladder per PRN. Rungs are keyed by RECORD COUNT, which is the
    # ladder's own axis and identical across instances (same integration_length, same emit).
    per = {}   # prn -> {records -> [(snr, floor, span_s), ...]}
    strength = {}
    rho2 = {}  # prn -> per-record POWER SNR each instance implies, from the incoherent moments
    for _, r in live:
        for e in r:
            p = int(e.get("prn", 0))
            if p <= 0:
                continue
            strength[p] = strength.get(p, 0.0) + float(e.get("p_pow") or 0.0)
            flr = float(e.get("deep_floor") or 0.0)
            # PER-RECORD POWER SNR from the INCOHERENT statistic, which is what makes the ladder
            # readable. amp_snr = s^2 * K^0.25 / N and is normalized to ~1 under H0 REGARDLESS of
            # K, so the signal part is (amp_snr - 1) and s^2/N = (amp_snr - 1) / K^0.25. Subtract
            # the 1: using amp_snr raw credits pure noise with a per-record signal, and then every
            # coherent number looks like a deficit.
            K = float(e.get("deep_records") or 0) or 100.0
            A = float(e.get("amp_snr") or 0.0)
            if A > 0:
                rho2.setdefault(p, []).append(max(0.0, A - 1.0) / (K ** 0.25))
            for rung in (e.get("rungs") or []):
                n, snr, span = int(rung[0]), float(rung[1]), float(rung[2])
                per.setdefault(p, {}).setdefault(n, []).append((snr, flr, span))

    if not per:
        print("no rungs published -- is the fleet running the 2026-08-04 combiner?")
        return
    prns = sorted(WANT) if WANT else [p for p in sorted(strength, key=strength.get, reverse=True)
                                      if p in per][:a.top]
    for p in prns:
        lad = per.get(p)
        if not lad:
            continue
        rs = sorted(rho2.get(p, [0.0]))
        rr = rs[len(rs) // 2]   # median instance's per-record power SNR
        print("PRN %-3d  (%d instances reporting; per-record power SNR %.4f -> voltage %.3f)"
              % (p, len(live), rr, math.sqrt(rr)))
        print("   %8s %8s %9s %9s %9s %8s   %s" % ("records", "span_s", "snr_med", "snr_max",
                                                   "predicted", "obs/pred", "vs bar"))
        for n in sorted(lad, reverse=True):
            v = lad[n]
            snrs = sorted(x[0] for x in v)
            med = snrs[len(snrs) // 2]
            mx = snrs[-1]
            span = v[0][2]
            flr = max(x[1] for x in v)
            # WHAT A PERFECTLY COHERENT SUM WOULD READ. The stage's statistic is
            # |sum x| / sqrt(sum|x|^2), which is ~1 under H0 and sqrt(1 + L*rho^2) with signal.
            # This column is the whole point: comparing the ladder against its OWN prediction
            # separates "not enough signal" (obs ~= pred, both small) from "not cohering"
            # (obs << pred, and the deficit grows with span).
            pred = math.sqrt(1.0 + n * rr)
            bar = 2.0 * (flr if flr > 0 else (math.sqrt(2.0 * math.log(2.0)) + 1.0))
            mark = "CLEARS" if mx >= bar else ("%.2fx" % (mx / bar) if bar > 0 else "?")
            print("   %8d %8.3f %9.2f %9.2f %9.2f %8s   %s"
                  % (n, span, med, mx, pred,
                     "%.2f" % (med / pred) if pred > 0 else "-", mark))
        # THE FIRST QUESTION IS NOT WHICH SHAPE -- IT IS WHETHER THERE IS A SHAPE. A coherent sum
        # of L random-phase terms has |sum|/sqrt(sum|x|^2) ~ 1 by construction: snr ~ 1 is what
        # PURE NOISE reads, on every rung, at every length. Fitting "phase-limited" vs
        # "SNR-limited" to scatter around 1 is exactly the failure 8.18.5 exists to prevent, so
        # refuse the verdict until some rung is meaningfully off the Rayleigh value.
        meds = []
        for n in sorted(lad, reverse=True):
            v = lad[n]
            meds.append((n, sorted(x[0] for x in v)[len(v) // 2], v[0][2]))
        if max(m for _, m, _ in meds) < 2.0:
            print("   -> ALL RUNGS AT THE RAYLEIGH VALUE (max median %.2f, noise reads ~1.0):"
                  " no coherent signal at ANY span, so the ladder shape says nothing yet.\n"
                  % max(m for _, m, _ in meds))
        else:
            # Now the shape means something. Flat in snr/sqrt(T) = SNR-limited (gain tracks the
            # span, coherence is not the wall). Rising as the rung SHORTENS = phase-limited: the
            # long windows are losing what they integrate.
            norm = [(n, m / math.sqrt(s)) for n, m, s in meds if s > 0]
            r = norm[-1][1] / norm[0][1] if norm and norm[0][1] > 0 else float("nan")
            print("   -> %s\n" % ("PHASE-limited (short rungs %.2fx better per sqrt(s))" % r
                                  if r > 1.3 else
                                  "SNR-limited (ladder gains with span; coherence is not the wall)"
                                  if r < 1.15 else "mixed / inconclusive (%.2fx)" % r))


once()
if a.watch > 0:
    import time
    while True:
        time.sleep(a.watch)
        print("=" * 72)
        once()
