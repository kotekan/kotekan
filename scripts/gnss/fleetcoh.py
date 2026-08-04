#!/usr/bin/env python3
"""Cross-node COHERENT combining of the per-record prompts -- 8.18.3 item 2, measured first.

One combiner instance correlates 7 of the fleet's 106 channels: 8.8 dB below what the array
actually collects, and per-record SNR is what caps the coherent span (8.18.2). Summing the
per-record COMPLEX prompts across all 14 instances recovers it. The window means cannot -- the
phase noise has already eaten those -- which is why the stage now exports per record.

THE PART THAT MAKES IT TRACTABLE. Each instance carries an unknown constant phase (cable, LO,
whatever the F-engine's per-node history is). Solving for it looks circular -- you need coherence
to measure the offset that buys you coherence -- but it is not, because the SIGNAL's phase noise
CANCELS in the cross product:

    Sum_k A_i[k] conj(A_ref[k])   ->   |S|^2 K exp(j(phi_i - phi_ref))

Both instances see the same sky and the same wandering signal phase, so their DIFFERENCE is the
constant instrumental term and the sum accumulates coherently over every record regardless of how
badly the signal itself is cohering. The calibration is immune to the very problem it fixes.

WHAT THIS PRINTS, and why it is built this way (8.18.5: measure on signal AND on noise, and
measure whether it is stable):

  * per-instance vs combined per-record SNR, so the claimed 8.8 dB is checked, not assumed;
  * the coherence ladder on the combined stream next to the best single instance's;
  * a NULL, on the same data: each instance's records rolled by a different random shift. That
    destroys the cross-instance signal alignment while leaving every noise statistic and every
    array length untouched, so whatever the null reports is what this pipeline manufactures out
    of nothing. A gain that survives against the null is real; one that does not is the fourth
    statistic that measured its own construction.
  * the fitted offsets across repeated polls -- a per-node constant that will not sit still is
    not a per-node constant.

usage:  fleetcoh.py [--prn 3] [--polls 3] [--interval 20]
"""
import argparse
import cmath
import json
import math
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_ENDPOINTS = [
    "http://cx19:12049/gnss0_combine", "http://cx51:12049/gnss0_combine",
] + ["http://%s:12049/gnss%d_combine" % (n, g)
     for n in ("cx27", "cx42", "cx43", "cx44", "cx47", "cx52") for g in (0, 1)]

ap = argparse.ArgumentParser()
ap.add_argument("--endpoints", default=",".join(DEFAULT_ENDPOINTS))
ap.add_argument("--prn", type=int, default=0, help="0 = the strongest PRN the fleet reports")
ap.add_argument("--polls", type=int, default=3)
ap.add_argument("--interval", type=float, default=20.0)
ap.add_argument("--seed", type=int, default=1234, help="null-control shuffle (reproducible)")
a = ap.parse_args()
EPS = [e.strip() for e in a.endpoints.split(",") if e.strip()]


def fetch(url, what):
    try:
        return json.load(urllib.request.urlopen(url + what, timeout=10))
    except Exception:
        return None


def poll(what):
    with ThreadPoolExecutor(max_workers=len(EPS)) as ex:
        out = list(ex.map(lambda u: fetch(u, what), EPS))
    return {EPS[i]: r for i, r in enumerate(out) if r is not None}


def pick_prn(status):
    """Strongest by summed prompt power -- the same currency the fleet DLL ranks on."""
    tot = {}
    for r in status.values():
        for e in r:
            p = int(e.get("prn", 0))
            if p > 0:
                tot[p] = tot.get(p, 0.0) + float(e.get("p_pow") or 0.0)
    return max(tot, key=tot.get) if tot else 0


def streams(records, prn):
    """{endpoint -> {hop -> (complex amplitude, energy)}} for one PRN."""
    out = {}
    for ep, rows in records.items():
        for e in rows:
            if int(e.get("prn", 0)) != prn:
                continue
            d = {}
            for hop, re_, im_, en in (e.get("records") or []):
                if en > 0:
                    d[int(hop)] = (complex(re_, im_), float(en))
            if d:
                out[ep] = d
    return out


def solve_offsets(st, hops, ref):
    """Instrumental phase per instance, from the phase-noise-immune cross product."""
    off, wt = {}, {}
    for ep, d in st.items():
        acc = 0j
        n = 0
        for h in hops:
            if h in d and h in st[ref]:
                acc += d[h][0] * st[ref][h][0].conjugate()
                n += 1
        off[ep] = cmath.phase(acc) if n else 0.0
        # |acc| relative to the incoherent product bounds how well the offset is determined:
        # 1.0 = perfectly coherent, ~1/sqrt(n) = consistent with no relationship at all.
        inc = sum(abs(d[h][0]) * abs(st[ref][h][0]) for h in hops if h in d and h in st[ref])
        wt[ep] = (abs(acc) / inc) if inc > 0 else 0.0
    return off, wt


def combine(st, hops, off, roll=None):
    """ML combine per record: sum(energy * A * e^-j.phi) / sum(energy).

    `roll` shifts each instance's records by a per-instance offset IN HOP INDEX -- the null. It
    keeps every amplitude, every energy and every array length, and changes only WHICH record
    lines up with which, so it isolates the one thing the combine claims: alignment.
    """
    out = []
    for h in hops:
        num, den = 0j, 0.0
        for ep, d in st.items():
            k = h if roll is None else h + roll.get(ep, 0)
            if k not in d:
                continue
            A, en = d[k]
            num += en * A * cmath.exp(-1j * off.get(ep, 0.0))
            den += en
        if den > 0:
            out.append(num / den)
    return out


def rec_snr(xs):
    """Per-record voltage SNR: |mean| over the scatter about it, x sqrt(n) removed.

    Deliberately NOT a coherent statistic -- this is the thing the coherent span depends on, so
    measuring it coherently would beg the question. Uses the median absolute amplitude as the
    signal scale and the quadrature spread as noise.
    """
    if len(xs) < 8:
        return float("nan")
    m = sum(xs) / len(xs)
    if abs(m) == 0:
        return float("nan")
    u = m / abs(m)
    q = [(x / u).imag for x in xs]           # quadrature: signal-free by construction
    mq = sum(q) / len(q)
    var = sum((v - mq) ** 2 for v in q) / max(1, len(q) - 1)
    return abs(m) / math.sqrt(var) if var > 0 else float("nan")


def ladder(xs, rec_dt):
    """The stage's plain-coherent ladder, in Python: full window then octave-shorter suffixes.

    Floor is the stage's own: sqrt(2 ln 2) + 1, no alignment search in a straight sum.
    """
    flr = math.sqrt(2.0 * math.log(2.0)) + 1.0
    n = len(xs)
    lens, l = [n], n // 2
    while l >= 7 and len(lens) < 5:
        lens.append(l)
        l //= 2
    rows = []
    for L in lens:
        w = xs[-L:]
        s = sum(w)
        # Same construction as gnss::coherent_sum: coherent sum over the incoherent scale.
        inc = math.sqrt(sum(abs(x) ** 2 for x in w) * L)
        rows.append((L, L * rec_dt, (abs(s) / inc * math.sqrt(L)) if inc > 0 else 0.0, flr))
    return rows


print("polling %d combiner endpoints" % len(EPS))
hist = []
for it in range(a.polls):
    if it:
        time.sleep(a.interval)
    status = poll("/get_status")
    recs = poll("/get_records")
    if not recs:
        print("no /get_records -- is record_export set and the 2026-08-04 binary deployed?")
        sys.exit(1)
    prn = a.prn or pick_prn(status)
    st = streams(recs, prn)
    if len(st) < 2:
        print("PRN %d: only %d instance(s) with records; nothing to combine" % (prn, len(st)))
        continue

    # Hops present in the MOST instances -- the combine is only meaningful where they overlap.
    seen = {}
    for d in st.values():
        for h in d:
            seen[h] = seen.get(h, 0) + 1
    nmax = max(seen.values())
    hops = sorted(h for h, c in seen.items() if c >= max(2, int(0.8 * nmax)))
    ref = max(st, key=lambda e: sum(abs(v[0]) ** 2 * v[1] for v in st[e].values()))

    off, wt = solve_offsets(st, hops, ref)
    comb = combine(st, hops, off)
    # Null: a fixed pseudo-random hop roll per instance. Same seed every poll so the control is
    # the same control, and >= 8 records so no roll accidentally re-aligns.
    rnd = random.Random(a.seed)
    roll = {ep: rnd.choice([r for r in range(-97, 98) if abs(r) >= 8]) for ep in st}
    null = combine(st, hops, off, roll=roll)

    singles = sorted((rec_snr([st[ep][h][0] for h in hops if h in st[ep]]), ep) for ep in st)
    best_s, best_ep = singles[-1]
    c_snr, n_snr = rec_snr(comb), rec_snr(null)

    # Record period from the hop spacing: hops are exact integers off one F-engine counter.
    dh = min((hops[i + 1] - hops[i]) for i in range(len(hops) - 1)) if len(hops) > 1 else 1
    rec_dt = dh * 16384 / 1.5625e9 * 8 if False else None   # not assumed; see below
    # Prefer the truth from the status block: coherence_s / deep_records is the stage's own
    # record period, measured rather than derived from a chain of constants.
    for r in status.values():
        for e in r:
            if int(e.get("prn", 0)) == prn and (e.get("deep_records") or 0) > 1 \
                    and (e.get("coherence_s") or 0) > 0:
                rec_dt = e["coherence_s"] / e["deep_records"]
                break
        if rec_dt:
            break
    rec_dt = rec_dt or 0.010486

    print("\n=== PRN %d   %d instances, %d overlapping hops   (ref %s)"
          % (prn, len(st), len(hops), ref.split("/")[2]))
    print("  per-record SNR:  best single %.3f (%s)   combined %.3f   NULL %.3f"
          % (best_s, best_ep.split("/")[2], c_snr, n_snr))
    if best_s > 0 and c_snr == c_snr:
        print("  gain over best single: %+.2f dB   (ideal for %d instances: %+.2f dB)"
              % (20 * math.log10(c_snr / best_s), len(st), 10 * math.log10(len(st))))
    if n_snr == n_snr and c_snr == c_snr:
        print("  combined / null: %.2fx  %s" % (c_snr / n_snr if n_snr else float("inf"),
              "REAL" if n_snr and c_snr > 2 * n_snr else "NOT DISTINGUISHABLE FROM ALIGNMENT LUCK"))
    print("  offset coherence (1.0 = fully determined, ~%.2f = no relationship):"
          % (1.0 / math.sqrt(max(1, len(hops)))))
    for ep in sorted(wt, key=wt.get, reverse=True):
        print("    %-28s phi %+7.1f deg   coh %.3f" % (ep.split("/")[2] + "/" + ep.split("/")[-1],
                                                       math.degrees(off[ep]), wt[ep]))
    print("  ladder (records, span_s, snr, floor x2 = bar %.2f):"
          % (2 * (math.sqrt(2 * math.log(2)) + 1)))
    bar = 2 * (math.sqrt(2 * math.log(2)) + 1)
    for tag, xs in (("combined", comb), ("best single", [st[best_ep][h][0] for h in hops
                                                         if h in st[best_ep]]), ("null", null)):
        cells = " ".join("%d:%.2f" % (L, s) for L, _, s, _ in ladder(xs, rec_dt))
        top = max((s for _, _, s, _ in ladder(xs, rec_dt)), default=0.0)
        print("    %-12s %s   %s" % (tag, cells, "CLEARS" if top >= bar else "below bar"))
    hist.append((prn, {ep: math.degrees(off[ep]) for ep in off}))

if len(hist) > 1:
    print("\n=== offset stability across %d polls (a per-node constant that moves is not one)"
          % len(hist))
    eps = set().union(*[set(h[1]) for h in hist])
    for ep in sorted(eps):
        v = [h[1][ep] for h in hist if ep in h[1]]
        if len(v) < 2:
            continue
        # Circular spread, so +179 and -179 do not read as 358 degrees of drift.
        m = cmath.phase(sum(cmath.exp(1j * math.radians(x)) for x in v) / len(v))
        dev = [((x - math.degrees(m) + 180) % 360) - 180 for x in v]
        print("    %-28s mean %+7.1f  spread %5.1f deg"
              % (ep.split("/")[2] + "/" + ep.split("/")[-1], math.degrees(m),
                 max(dev) - min(dev)))
