"""Direction-dependent code-delay map fitted jointly with a constant per satellite.

    python3 gdv_fit.py OUTDIR      (on cf06: reads the obs archive; the slim map the broker reads
    is then `python3 -m gnss_broker.gdvmap export OUTDIR/gdv_fit_taua.json config/gdvmap_<p>.json`)

Model per band: x(sat, t) = M(cell(az, el)) + B(sat) + noise, where x is the code excess
delay (m, + = late) relative to the epoch's median satellite. Alternating medians (robust);
gauge: median of M over cells with el >= 45 is 0. Cells: az 5 x el 2.5 deg below el 20,
az 10 x el 5 above.

Rows gated as the shadow feed is: |node trim| < 0.75 x clamp, and |excess| <= 30 m (gross
errors, a replica off its signal).

Checks: (1) odd vs even PRN fitted separately, M compared cell by cell; (2) held out: M and B
from TRAIN, applied to TEST (10-26 00-14Z, the night C41 failed b_sat): per-satellite medians
before and after.
"""
import collections
import json
import math
import os
import statistics as st
import sys

OUT = sys.argv[1]
OBS = "/home/kvand/gnss/fixtures/obs/%s_%s.jsonl"
CHAINS = {  # chain: chip length (m)
    "gal_e5a": 299792458.0 / 10.23e6,
    "gal_e5b": 299792458.0 / 10.23e6,
    "gal_e6": 299792458.0 / 5.115e6,
    "gps_l5": 299792458.0 / 10.23e6,
    "bds_b2a": 299792458.0 / 10.23e6,
    "bds_b3i": 299792458.0 / 10.23e6,
}
RAIL, GROSS_M = 0.75, 30.0
CLAMP = {
    "gal_e5a": 1.473,
    "gal_e5b": 1.473,
    "gal_e6": 0.737,
    "gps_l5": 1.473,
    "bds_b2a": 1.473,
    "bds_b3i": 1.473,
}
TRAIN = (("20261024", 14.0, 24.0), ("20261025", 15.0, 24.0), ("20261026", 14.0, 19.0))
TEST = (("20261026", 0.0, 14.0),)


def cell(az, el):
    if el < 20:
        return (int(az // 5) * 5, round(int(el / 2.5) * 2.5, 1))
    return (int(az // 10) * 10, int(el // 5) * 5)


def load(chain, windows):
    out = []
    for day, h0, h1 in windows:
        p = OBS % (chain, day)
        if not os.path.exists(p):
            continue
        ep = collections.defaultdict(dict)
        with open(p) as f:
            for line in f:
                if '"code_resid_src":"trk"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                h = r["t"] % 86400 / 3600.0
                if not (h0 <= h < h1) or not r.get("fleet_present"):
                    continue
                if (
                    r.get("code_resid_m") is None
                    or r.get("el") is None
                    or r.get("az") is None
                ):
                    continue
                # the shadow feed's rail gate (jtrk): a trim near its clamp has the replica pinned
                if abs(r.get("dll_trim_cpp") or 0.0) >= RAIL * CLAMP[chain]:
                    continue
                # the residual without the map the broker applied (gnss_broker/gdvmap), so a
                # refit measures the array, not the previous map's error
                x = r["code_resid_m"] - (r.get("gdv_m") or 0.0)
                ep[int(r["t"] // 10)][r["prn"]] = (x, r["az"], r["el"], r["t"])
        for k, d in ep.items():
            if len(d) < 5:
                continue
            med = st.median(v[0] for v in d.values())
            for prn, (v, az, el, t) in d.items():
                # gross errors: a replica off its signal (hundreds of m on gps_l5 while present)
                if el > 0 and abs(v - med) <= GROSS_M:
                    out.append((prn, cell(az, el), -(v - med), az, el, t))
    return out


def fit(rows, iters=12, min_n=15):
    B = collections.defaultdict(float)
    M = {}
    for _ in range(iters):
        bc = collections.defaultdict(list)
        for prn, c, x, *_ in rows:
            bc[c].append(x - B[prn])
        M = {c: st.median(v) for c, v in bc.items() if len(v) >= min_n}
        hi = [m for c, m in M.items() if c[1] >= 45]
        g = st.median(hi) if hi else 0.0
        M = {c: m - g for c, m in M.items()}
        bs = collections.defaultdict(list)
        for prn, c, x, *_ in rows:
            if c in M:
                bs[prn].append(x - M[c])
        B = collections.defaultdict(float, {p: st.median(v) for p, v in bs.items()})
    sig = collections.defaultdict(list)
    for prn, c, x, *_ in rows:
        if c in M:
            sig[c].append(x - M[c] - B[prn])
    S = {
        c: 1.4826 * st.median(abs(e) for e in v)
        for c, v in sig.items()
        if len(v) >= min_n
    }
    return M, dict(B), S


def corr(a, b):
    ks = [k for k in a if k in b]
    if len(ks) < 5:
        return len(ks), float("nan"), float("nan")
    x, y = [a[k] for k in ks], [b[k] for k in ks]
    mx, my = st.mean(x), st.mean(y)
    num = sum((p - mx) * (q - my) for p, q in zip(x, y))
    den = (
        math.sqrt(sum((p - mx) ** 2 for p in x) * sum((q - my) ** 2 for q in y)) or 1.0
    )
    return len(ks), num / den, math.sqrt(st.mean((p - q) ** 2 for p, q in zip(x, y)))


res = {}
for ch, chip_m in CHAINS.items():
    tr = load(ch, TRAIN)
    te = load(ch, TEST)
    M, B, S = fit(tr)
    Mo, _, _ = fit([r for r in tr if r[0] % 2 == 1])
    Me, _, _ = fit([r for r in tr if r[0] % 2 == 0])
    n, r, rms = corr(Mo, Me)
    # held-out: per-satellite median of the test night before / after M (and after M + B)
    per = collections.defaultdict(lambda: [[], [], []])
    for prn, c, x, az, el, t in te:
        if c in M:
            per[prn][0].append(x)
            per[prn][1].append(x - M[c])
            per[prn][2].append(x - M[c] - B.get(prn, 0.0))
    before = [abs(st.median(v[0])) for v in per.values() if len(v[0]) >= 30]
    after = [abs(st.median(v[1])) for v in per.values() if len(v[1]) >= 30]
    afterb = [abs(st.median(v[2])) for v in per.values() if len(v[2]) >= 30]
    # held-out residual scatter in the north-low sector, before / after
    nl = [
        (x, x - M[c])
        for prn, c, x, az, el, t in te
        if c in M and el < 15 and (az >= 315 or az < 45)
    ]
    print(
        "%-8s train %6d test %6d | cells %3d | odd/even: %3d cells r %.2f rms %.2f m | test per-sat |median|: "
        "%.2f -> %.2f (map) -> %.2f (map+B) m | test N-low median %+.2f -> %+.2f m"
        % (
            ch,
            len(tr),
            len(te),
            len(M),
            n,
            r,
            rms,
            st.median(before) if before else float("nan"),
            st.median(after) if after else float("nan"),
            st.median(afterb) if afterb else float("nan"),
            st.median(a for a, b in nl) if nl else float("nan"),
            st.median(b for a, b in nl) if nl else float("nan"),
        )
    )
    if ch.startswith("bds"):
        for prn in (41, 9, 27):
            if prn in per and len(per[prn][0]) >= 10:
                v = per[prn]
                print(
                    "   C%02d test night: excess %+.2f m (%+.3f chip) -> map %+.2f m (%+.3f chip) -> map+B %+.2f m; "
                    "B(train) %s"
                    % (
                        prn,
                        st.median(v[0]),
                        st.median(v[0]) / chip_m,
                        st.median(v[1]),
                        st.median(v[1]) / chip_m,
                        st.median(v[2]),
                        "%+.2f m" % B[prn] if prn in B else "none (not in train)",
                    )
                )
    res[ch] = {
        "M": {"%d,%s" % c: m for c, m in M.items()},
        "sigma": {"%d,%s" % c: s for c, s in S.items()},
        "B": {str(p): b for p, b in B.items()},
        "split": {"cells": n, "r": r, "rms_m": rms},
    }
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "gdv_fit_taua.json"), "w") as f:
    json.dump(
        {
            "what": "code excess-delay map M(cell) (m, + = late) fitted jointly with per-satellite "
            "constants B; relative to each epoch's median satellite; gauge median M(el>=45) = 0",
            "pointing": "p5_taua_20261024",
            "train": TRAIN,
            "test": TEST,
            "bands": res,
        },
        f,
    )
print("wrote", os.path.join(OUT, "gdv_fit_taua.json"))
