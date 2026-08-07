#!/usr/bin/env python3
"""Why is path B's deep_rate_q 3-7x below path A's when amp_snr differs by only 17-29%?

    qgap.py                                  # cx19, path A vs both path-B combiners
    qgap.py --a cx19:gnss0_combine --b cx19:gnss0_n2combine
    qgap.py --json a.json b.json             # from saved /get_records dumps

WHY THIS EXISTS. 2026-08-07: path B's deep fold reads coh_frac 0.02-0.10 against path A's
0.93-0.97, and the cause is that its rate search fails the q >= 10 gate (4.5-11.4 vs 29-38).
Per-record despread is fine -- on-sky A/B agrees to |corr| 0.989 and amp_snr is 71-83% of
path A. So a 3-7x q deficit has to come from somewhere other than per-record amplitude, and
guessing which term costs a config change and a fleet restart to disprove.

q is peak/median of the record-rate spectrum, so it factors exactly:

    q = |X(f_pk)| / median|X|

and each factor has its own failure mode. A DIFFERENT NUMBER OF RECORDS moves both (peak
~ L, noise median ~ sqrt(L), so q ~ sqrt(L)). ZERO OR DROPPED RECORDS cut the peak without
cutting the median. A NON-FLAT NOISE FLOOR (records correlated in time, or a handful of
outliers) raises the median alone. This prints all of them side by side against the same
PRNs, so the gap is attributed rather than assumed.

Reads /get_records, which exports [hop, re, im, energy] per record -- the same endpoint used
to diagnose the defect in 11.14.1, and the same arithmetic as rate_search in
GnssCoherentCombiner.cpp (4x oversampled integer-indexed DFT on the uniform hop grid).
"""

import argparse
import json
import sys
import urllib.request

import numpy as np


def fetch(spec):
    """spec = 'host:stage' -> the /get_records list."""
    host, stage = spec.split(":", 1)
    url = "http://%s:12049/%s/get_records" % (host, stage)
    with urllib.request.urlopen(url, timeout=25) as r:
        return json.load(r)


def rate_spectrum(rec, nmax=None):
    """Reproduce GnssCoherentCombiner's rate_search on one PRN's record list.

    Returns (q, peak, median, f_hz, L, n_zero, mags). dt is the MINIMUM consecutive spacing,
    not the mean: with gaps the mean is not the record period, and a wrong dt rescales every
    frequency. Records land on a uniform grid with holes; a hole contributes exactly zero.
    """
    a = np.asarray(rec, dtype=np.float64)
    if nmax:
        a = a[-nmax:]
    hop = a[:, 0]
    x = a[:, 1] + 1j * a[:, 2]
    L = len(x)
    if L < 8:
        return None
    d = np.diff(hop)
    d = d[d > 1e-9]
    if not len(d):
        return None
    dt = d.min()
    m = np.rint((hop - hop[0]) / dt).astype(np.int64)
    span = int(m.max())
    nb = min(4 * (span + 1), 4096)
    if nb < 8:
        return None
    # integer-indexed twiddle, exactly as the C++ does
    i = np.arange(nb)
    idx = (np.outer(i, m) % nb).astype(np.int64)
    tw = np.exp(-2j * np.pi * np.arange(nb) / nb)
    mags = np.abs((tw[idx] * x[None, :]).sum(axis=1))
    ip = int(np.argmax(mags))
    med = float(np.median(mags))
    q = mags[ip] / med if med > 0 else 0.0
    frac = ip / nb
    # dt here is in HOPS; convert with the 5.12 us CHORD hop so f is Hz
    dt_s = dt * 5.12e-6
    f = (frac if frac < 0.5 else frac - 1.0) / dt_s
    n_zero = int((np.abs(x) == 0).sum())
    return dict(q=q, peak=float(mags[ip]), med=med, f=f, L=L, nb=nb, dt=dt,
                n_zero=n_zero, mags=mags, x=x, m=m)


def summarize(tag, s):
    x = s["x"]
    absx = np.abs(x)
    # coherent gain actually achieved at the winning bin, vs the ideal L
    gain = s["peak"] / absx.sum() if absx.sum() > 0 else 0.0
    # lag-1 phase step: the ramp the search is supposed to find
    st = np.angle(x[1:] * np.conj(x[:-1]))
    print("  %-6s L=%-4d dt=%-6d nb=%-5d zeros=%-3d  |v| med %.3e  sum|v| %.3e" %
          (tag, s["L"], s["dt"], s["nb"], s["n_zero"], np.median(absx), absx.sum()))
    print("         peak %.4e  median %.4e  q %7.2f   f %+8.3f Hz  cohgain %.3f" %
          (s["peak"], s["med"], s["q"], s["f"], gain))
    print("         phase step %+.3f +- %.3f rad   |v| spread(p90/p10) %.2f" %
          (np.mean(st), np.std(st), np.percentile(absx, 90) / max(np.percentile(absx, 10), 1e-30)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", default="cx19:gnss0_combine", help="path A combiner host:stage")
    ap.add_argument("--b", default="cx19:gnss0_n2combine", help="path B combiner host:stage")
    ap.add_argument("--json", nargs=2, metavar=("A", "B"), default=None,
                    help="read saved /get_records dumps instead of polling")
    ap.add_argument("--match-length", action="store_true",
                    help="trim both to the same RECORD COUNT before transforming, which removes"
                         " the sqrt(L) term and shows what is left")
    ap.add_argument("--match-span", action="store_true",
                    help="trim both to the same TIME SPAN in hops (the physically fair"
                         " comparison when record lengths differ)")
    a = ap.parse_args()

    if a.json:
        A = json.load(open(a.json[0]))
        B = json.load(open(a.json[1]))
    else:
        A, B = fetch(a.a), fetch(a.b)

    Ad = {r["prn"]: r["records"] for r in A}
    Bd = {r["prn"]: r["records"] for r in B}
    common = sorted(set(Ad) & set(Bd))
    if not common:
        print("no PRN in both exports", file=sys.stderr)
        return 1

    print("q = peak/median of the record-rate spectrum. gate deep_rate_min_q = 10.0")
    print("A = %s   B = %s   PRNs in both: %s\n" % (a.a, a.b, common))

    rows = []
    for p in common:
        ra, rb = Ad[p], Bd[p]
        if a.match_length:
            n = min(len(ra), len(rb))
            ra, rb = ra[-n:], rb[-n:]
        elif a.match_span:
            # equal span in HOPS, so a 4x longer record simply contributes 4x fewer samples
            h0 = max(ra[0][0], rb[0][0])
            h1 = min(ra[-1][0], rb[-1][0])
            ra = [r for r in ra if h0 <= r[0] <= h1]
            rb = [r for r in rb if h0 <= r[0] <= h1]
        sa, sb = rate_spectrum(ra), rate_spectrum(rb)
        if not sa or not sb:
            continue
        print("PRN %d" % p)
        summarize("A", sa)
        summarize("B", sb)
        print("         RATIO A/B: q %.2f | peak %.2f | median %.2f | sqrt(L) %.2f | |v| %.2f"
              % (sa["q"] / sb["q"] if sb["q"] else float("nan"),
                 sa["peak"] / sb["peak"] if sb["peak"] else float("nan"),
                 sa["med"] / sb["med"] if sb["med"] else float("nan"),
                 (sa["L"] / sb["L"]) ** 0.5,
                 np.median(np.abs(sa["x"])) / max(np.median(np.abs(sb["x"])), 1e-30)))
        print()
        rows.append((p, sa, sb))

    if rows:
        qa = np.array([r[1]["q"] for r in rows])
        qb = np.array([r[2]["q"] for r in rows])
        pk = np.array([r[1]["peak"] / r[2]["peak"] for r in rows])
        md = np.array([r[1]["med"] / r[2]["med"] for r in rows])
        print("ATTRIBUTION over %d PRNs (median ratio A/B):" % len(rows))
        print("  q       %.2f   <- what has to be explained" % np.median(qa / qb))
        print("  peak    %.2f   <- coherent signal recovered at the winning bin" % np.median(pk))
        print("  median  %.2f   <- the noise floor the peak is measured against" % np.median(md))
        print("\n  A over gate: %d/%d      B over gate: %d/%d"
              % ((qa >= 10).sum(), len(qa), (qb >= 10).sum(), len(qb)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
