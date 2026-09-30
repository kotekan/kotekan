#!/usr/bin/env python3
"""Do the dTEC arcs survive boresight transits? A before/after census around a change (--split),
with matched quiet windows as the control.

TRANSITS come from the broadcast ephemeris for EVERY G/E/C satellite in it, healthy or not and
tracked or not: an untracked emitter (E14, the BDS-3 PRNs below 19, G19's L2 P(Y)) leaks into
our chains exactly as a tracked one does. A transit is the span within --bore-deg of boresight
(az 180, el 81.41); 5 deg is the #142 freeze's own bound.

TEC is gnss_tec_chord.py's product, rebuilt with its rules: fadr_g_dop_cycles of two bands
paired at the same fleet grid hop, arcs keyed (fadr_arc, fadr_hop0) in both, coherent C/N0 >=
25 dB-Hz on both, a hole > 30 s ends an arc, a 1-s change > 5 cm splits it, >= 300 s and 30
hops, local-scatter noise <= 0.5 TECU. (One deliberate difference: hops are ordered by time
and a hop that goes backwards ends an arc, so a window that crosses an F-engine re-base,
which restarts the hop count, cannot pair across it.)

Per transit and pair, every satellite with clean TEC on both bands (C/N0 >= 25, el >= 15) in
the 2 min before the window's entry (--pad-s before it) is classified by what is left at the
exit (--pad-s after it):
  SURVIVED     the same product arc on both sides
  BRIDGEABLE   the product arc was cut (C/N0 gate, a hole, a step that came back) but both
               bands' fleet-ADR arcs held and the TEC level across the window agrees within
               --jump-tecu (a line fitted to 10 min of clean data each side, compared at
               mid-window): the carrier phase held, only the product's gate cut it
  SLIPPED      both fleet-ADR arcs held but the level jumps by more than --jump-tecu
  HELD-UNSEEN  both fleet-ADR arcs held, too little clean data after to test the level
  LOCK LOST    a band's fleet-ADR arc restarted inside the window
  GONE         no rows on a band after the window
  NO ARC       clean data going in, but no product arc spanning the entry (too short, or
               dropped by the noise gate) -- counted so nothing leaves the census silently
The transiting satellite itself is counted apart as SELF.
Per chain, independent of TEC: of the PRNs present (fleet_present, el >= 25) going in, how many
kept their fleet-ADR arc (the tracking lock) through the window.
CONTROL windows have the same lengths, sit 45-90 min from a transit with no transit within 30 min
and no --events entry within 10 min, and go through the same census. Galileo's triple-frequency
closure TEC(E5b x E6) - TEC(E5a x E6) is ionosphere-free, so its jump across a window measures the
instrument alone. VTEC (the figures' right column) is vtec_plot's thin-shell mapping at 350 km with
each arc's offset fitted; an arc whose mapping factor swings < 0.4 has that offset ill-determined
and is drawn faint.

usage (on the analysis host, never on the VM that runs the stack: it parses ~8 GB of JSON; needs
numpy and matplotlib; ~2 min):
  python gnss_transit_arcs.py --t-lo YYYY-MM-DDTHH:MM:SS --split YYYY-MM-DDTHH:MM:SS \
      --out OUT_PREFIX [--t-hi ...] [--events FILE] [--broker-lines FILE]
--events: one "YYYY-MM-DDTHH:MM:SS label" per line (restarts, aborts): drawn on the plots and kept
out of the control windows. --broker-lines: the broker's FROZEN/CLEAR lines, printed beside the
ephemeris windows as a cross-check.
"""
import argparse
import calendar
import collections
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

GNSS = "/home/kvand/gnss/kotekan/python/scripts/gnss"
sys.path.insert(0, GNSS)
import gnss_tec_chord as gtc  # noqa: E402  the product's constants
import gnss_ephemeris as ge  # noqa: E402

OBS = "/home/kvand/gnss/fixtures/obs"
NAV = os.path.expanduser("~/.cache/kotekan_gps")
LAT, LON, ALT = 49.32075144444, -119.62081125, 545.0  # passwatch.py's station
BORE_AZ, BORE_EL = 180.0, 81.41
PAIRS = [("gal_e5a", "gal_e6"), ("bds_b2a", "bds_b3i"), ("gps_l5", "gps_l2c")]
CHAINS = [
    "gps_l5",
    "gps_l2c",
    "gal_e5a",
    "gal_e5b",
    "gal_e6",
    "bds_b2a",
    "bds_b2b",
    "bds_b3i",
]
SYS_OF = {c: gtc.SYSOF[c.split("_")[0]] for c in CHAINS}
CN0_MIN, MIN_ARC_S, MAX_GAP_S, STEP_M, MAX_NOISE = 25.0, 300.0, 30.0, 0.05, 0.5
MAX_GAP_HOPS = int(MAX_GAP_S * gtc.GRID_HOPS / gtc.GRID_SECONDS)
CLASSES = [
    "SURVIVED",
    "BRIDGEABLE",
    "SLIPPED",
    "HELD-UNSEEN",
    "LOCK LOST",
    "GONE",
    "NO ARC",
]
# Things that break arcs that are not transits (restarts, aborts): --events, drawn on the plots
# and kept out of the control windows.
EVENTS = []


HOP_S = gtc.GRID_SECONDS / gtc.GRID_HOPS  # 5.12 us, one hop


def epoch_of(t, hop):
    """The F-engine epoch a hop belongs to: the frame0 that t - hop implies, to the hour (frame0s
    are hours to days apart; a row's t is within seconds of its hop)."""
    return int(round((t - hop * HOP_S) / 3600.0))


def order(d):
    """Keys in time order. Every sample is keyed (epoch, hop): the hop count restarts at 0 at a
    re-base, so two epochs in one span reuse the same hop values, and a dict keyed by hop alone
    overwrites one day's samples with another's. Ordering by a row's `t` instead swaps
    neighbours: `t` is the POLL instant, up to ~2 s after the hop it reports, and fadr_g_hist
    hops take their time from it."""
    return sorted(d)


def ts(s):
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%S"))


def hm(t):
    return time.strftime("%m-%d %H:%M:%S", time.gmtime(t))


def bore_sep(az, el):
    a1, e1, a2, e2 = map(math.radians, (az, el, BORE_AZ, BORE_EL))
    c = math.sin(e1) * math.sin(e2) + math.cos(e1) * math.cos(e2) * math.cos(a1 - a2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


# ---------------------------------------------------------------------------------- transits
def nav_paths(t_lo, t_hi):
    """Every cached source for each day, merged per PRN by parse_rinex_nav: a day's R file can be
    thin (a third of the S file's records, with whole satellites missing), and a transit of a
    satellite no source carries is simply not listed."""
    out, t = [], t_lo - 86400
    while t <= t_hi + 86400:
        tag = time.strftime("%Y%j", time.gmtime(t))
        for kind in ("R", "S"):
            p = os.path.join(NAV, "BRDC00WRD_%s_%s0000_01D_MN.rnx.gz" % (kind, tag))
            if os.path.exists(p):
                out.append(p)
        t += 86400
    hourly = os.path.join(NAV, "hourly_MN.rnx.gz")
    if os.path.exists(hourly):
        out.append(hourly)
    return out


def sat_azel(e, gpst, rx):
    """predict_all's geometry WITHOUT its health filter (an unhealthy satellite still emits)."""
    tau = 0.075
    for _ in range(2):
        pos, _vel, _clk = ge.sat_pos_clk(e, gpst - tau)
        th = ge.OMEGA_E[e["sys"]] * tau
        px = pos[0] * math.cos(th) + pos[1] * math.sin(th)
        py = -pos[0] * math.sin(th) + pos[1] * math.cos(th)
        az, el, rng = ge._azel(rx, (px, py, pos[2]), LAT, LON)
        tau = rng / ge.C_LIGHT
    return az, el


def transits(t_lo, t_hi, bore_deg):
    paths = nav_paths(t_lo, t_hi)
    eph = ge.parse_rinex_nav(paths)
    rx = ge._ecef_of_llh(LAT, LON, ALT)
    wins = []
    for key, recs in sorted(eph.items()):
        if key[0] not in ge.OMEGA_E:
            continue

        def sep_at(t):
            g = ge.gpst_of_utc(float(t))
            e = ge.best_eph(recs, g, 4 * 3600.0)
            if e is None:
                return 99.0, None
            az, el = sat_azel(e, g, rx)
            return bore_sep(az, el), e

        tc = np.arange(t_lo, t_hi, 60.0)
        coarse = np.array([sep_at(t)[0] for t in tc])
        cand = np.where(coarse < bore_deg + 3.0)[0]  # a MEO moves < 1 deg/min overhead
        if not len(cand):
            continue
        runs = np.split(cand, np.where(np.diff(cand) > 1)[0] + 1)
        for run in runs:
            ft = np.arange(tc[run[0]] - 120.0, tc[run[-1]] + 120.0, 5.0)
            fs, health = [], None
            for t in ft:
                s, e = sep_at(t)
                fs.append(s)
                if e is not None:
                    health = e.get("health", 0)
            fs = np.array(fs)
            inside = fs < bore_deg
            if not inside.any():
                continue
            idx = np.where(inside)[0]
            for sub in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
                k = sub[np.argmin(fs[sub])]
                wins.append(
                    dict(
                        sat="%s%02d" % (key[0], key[1]),
                        sys=key[0],
                        prn=int(key[1]),
                        t0=float(ft[sub[0]]),
                        t1=float(ft[sub[-1]]),
                        t_ca=float(ft[k]),
                        min_sep=round(float(fs[k]), 2),
                        healthy=health in (0, 0.0, None) or key[0] == "C",
                    )
                )
    return sorted(wins, key=lambda w: w["t0"]), paths


# ------------------------------------------------------------------------------------ loading
def load_band(band, days, t_lo, t_hi):
    """can: gnss_tec_chord.load() exactly (C/N0 gate first); raw: every row of a real satellite
    (no noise probe, el >= 10), ungated. Both {prn: {(epoch, g_hop): tuple}}."""
    can = collections.defaultdict(dict)
    raw = collections.defaultdict(dict)
    for day in days:
        path = os.path.join(OBS, "%s_%s.jsonl" % (band, day))
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                if len(line) < 50:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                t = d["t"]
                if t < t_lo or t > t_hi:
                    continue
                gh, gc = d.get("fadr_g_hop"), d.get("fadr_g_dop_cycles")
                if not gh or gc is None:
                    continue
                prn = d["prn"]
                ep = epoch_of(t, gh)
                arc = (d.get("fadr_arc"), d.get("fadr_hop0"))
                cn, el, az = d.get("cn0_kcoh_dbhz"), d.get("el"), d.get("az")
                hist = d.get("fadr_g_hist") or []
                if cn is not None and cn >= CN0_MIN:
                    dst = can[prn]
                    dst[(ep, gh)] = (gc, arc, t, az, el)
                    for e in hist:
                        h = e[0]
                        if h == gh or (ep, h) in dst:
                            continue
                        dst[(ep, h)] = (
                            e[1],
                            arc,
                            t - (gh - h) / gtc.GRID_HOPS * gtc.GRID_SECONDS,
                            az,
                            el,
                        )
                if d.get("noise_probe") or el is None or el < 10:
                    continue
                fp = bool(d.get("fleet_present"))
                cq = -99.0 if cn is None else cn
                dst = raw[prn]
                dst[(ep, gh)] = (gc, arc, t, el, cq, fp)
                for e in hist:
                    h = e[0]
                    if h == gh or (ep, h) in dst:
                        continue
                    dst[(ep, h)] = (
                        e[1],
                        arc,
                        t - (gh - h) / gtc.GRID_HOPS * gtc.GRID_SECONDS,
                        el,
                        cq,
                        fp,
                    )
    return can, raw


# --------------------------------------------------------------------------- product arcs
def product_arcs(A, B, prn, la, lb, mpt):
    """gnss_tec_chord.main's per-PRN body: joint arcs, step split, length and noise gates.
    Returns (kept arcs, count dropped by the noise gate)."""
    common = {h: A[prn][h] for h in set(A[prn]) & set(B[prn])}
    hops = order(common)
    segs, why, cur = [], [], []
    w = "start"
    for h in hops:
        if cur:
            p = cur[-1]
            r = (
                "arc_a"
                if A[prn][h][1] != A[prn][p][1]
                else "arc_b"
                if B[prn][h][1] != B[prn][p][1]
                else "gap"
                if (h[0] != p[0] or h[1] - p[1] > MAX_GAP_HOPS or h[1] <= p[1])
                else None
            )
            if r is not None:
                segs.append(cur)
                why.append(w)
                cur, w = [], r
        cur.append(h)
    if cur:
        segs.append(cur)
        why.append(w)
    cut, cwhy = [], []
    for seg, w in zip(segs, why):
        cur = [seg[0]]
        cwhy.append(w)
        for h0, h in zip(seg, seg[1:]):
            d = (la * A[prn][h][0] - lb * B[prn][h][0]) - (
                la * A[prn][h0][0] - lb * B[prn][h0][0]
            )
            if (
                abs(d) > STEP_M
                and (A[prn][h][2] - A[prn][h0][2]) < 3.0 * gtc.GRID_SECONDS
            ):
                cut.append(cur)
                cur = []
                cwhy.append("step")
            cur.append(h)
        cut.append(cur)
    arcs, noisy = [], 0
    for seg, w in zip(cut, cwhy):
        tt = np.array([A[prn][h][2] for h in seg])
        el = np.array(
            [np.nan if A[prn][h][4] is None else A[prn][h][4] for h in seg], dtype=float
        )
        if tt[-1] - tt[0] < MIN_ARC_S or len(seg) < 30:
            continue
        gf = np.array([(la * A[prn][h][0] - lb * B[prn][h][0]) / mpt for h in seg])
        gf -= gf.mean()
        k = np.ones(31)
        # scatter about a 31-point local mean (edge windows shrink, as in the product)
        num = np.convolve(gf, k, "same")
        den = np.convolve(np.ones_like(gf), k, "same")
        noise = float(np.sqrt(np.mean((gf - num / den) ** 2)))
        if noise > MAX_NOISE:
            noisy += 1
            continue
        vt, faint = vertical(gf, el)
        arcs.append(
            dict(
                t0=float(tt[0]),
                t1=float(tt[-1]),
                starts_at=w,
                noise=noise,
                t=tt,
                tec=gf,
                vt=vt,
                faint=faint,
            )
        )
    return arcs, noisy


RE_KM, SHELL_KM, MIN_DM = 6371.0, 350.0, 0.4


def vertical(st, el):
    """fixtures/tec_wander/vtec_plot.py's mapping: STEC = M(el) * V + B per arc (thin shell), with B
    the arc's unknown constant. Dividing an arc-mean-removed slant by M would bend a flat VTEC into
    (M - Mbar)/M, so B is fitted with a constant V and (STEC - B)/M is returned, mean removed. B is
    poorly determined when M barely changes: faint = the mapping swings less than MIN_DM."""
    ok = np.isfinite(el)
    out = np.full(len(st), np.nan)
    if ok.sum() < 30:
        return out, True
    m = 1.0 / np.sqrt(
        1.0 - (RE_KM / (RE_KM + SHELL_KM) * np.cos(np.radians(el[ok]))) ** 2
    )
    (_v, b), *_ = np.linalg.lstsq(np.stack([m, np.ones_like(m)], 1), st[ok], rcond=None)
    v = (st[ok] - b) / m
    out[ok] = v - v.mean()
    return out, bool(m.max() - m.min() < MIN_DM)


def raw_joint(Ar, Br, prn, la, lb, mpt):
    if prn not in Ar or prn not in Br:
        return None
    common = {h: Ar[prn][h] for h in set(Ar[prn]) & set(Br[prn])}
    hops = order(common)
    if len(hops) < 30:
        return None
    ka_id, kb_id = {}, {}
    rows = []
    for h in hops:
        ra, rb = Ar[prn][h], Br[prn][h]
        rows.append(
            (
                ra[2],
                (la * ra[0] - lb * rb[0]) / mpt,
                ka_id.setdefault(ra[1], len(ka_id)),
                kb_id.setdefault(rb[1], len(kb_id)),
                ra[4],
                rb[4],
                ra[3],
            )
        )
    a = np.array(rows)
    return dict(
        t=a[:, 0],
        g=a[:, 1],
        ka=a[:, 2].astype(int),
        kb=a[:, 3].astype(int),
        ca=a[:, 4],
        cb=a[:, 5],
        el=a[:, 6],
    )


def lock_segments(rj, gap_s=60.0):
    """Runs over which both bands' fleet-ADR arcs held (no C/N0 gate): the tracking lock."""
    t, ka, kb = rj["t"], rj["ka"], rj["kb"]
    brk = (
        np.where((np.diff(ka) != 0) | (np.diff(kb) != 0) | (np.diff(t) > gap_s))[0] + 1
    )
    return [
        (float(t[s[0]]), float(t[s[-1]]))
        for s in np.split(np.arange(len(t)), brk)
        if len(s) > 1
    ]


def classify(w, arcs, rj, pad, jump_tecu):
    te, tx = w["t0"] - pad, w["t1"] + pad
    t = rj["t"]
    clean = (rj["ca"] >= CN0_MIN) & (rj["cb"] >= CN0_MIN)
    ib = np.where((t >= te - 120) & (t <= te) & clean & (rj["el"] >= 15))[0]
    if len(ib) < 20:
        return None
    j = ib[-1]
    ka0, kb0 = rj["ka"][j], rj["kb"][j]
    info = dict(ka0=int(ka0), kb0=int(kb0))
    arc = next((a for a in arcs if a["t0"] <= te <= a["t1"]), None)
    if arc is not None and arc["t1"] >= tx:
        return dict(info, cls="SURVIVED")
    ia = np.where((t >= tx) & (t <= tx + 300))[0]
    if not len(ia):
        last = np.where((rj["ka"] == ka0) & (rj["kb"] == kb0) & (t >= te))[0]
        return dict(
            info,
            cls="GONE" if arc is not None else "NO ARC",
            t_end=float(t[last[-1]]) if len(last) else te,
        )
    k = ia[0]
    if rj["ka"][k] != ka0 or rj["kb"][k] != kb0:
        mid = np.where((t >= t[j]) & ((rj["ka"] != ka0) | (rj["kb"] != kb0)))[0]
        tl = float(t[mid[0]]) if len(mid) else float(t[k])
        band = "a" if rj["ka"][mid[0] if len(mid) else k] != ka0 else "b"
        return dict(
            info, cls="LOCK LOST" if arc is not None else "NO ARC", t_lost=tl, band=band
        )
    same = (rj["ka"] == ka0) & (rj["kb"] == kb0) & clean
    pre = same & (t >= te - 600) & (t <= te)
    post = same & (t >= tx) & (t <= tx + 600)
    if arc is None:
        return dict(info, cls="NO ARC")
    if pre.sum() < 60 or post.sum() < 60:
        return dict(info, cls="HELD-UNSEEN")
    tc = 0.5 * (w["t0"] + w["t1"])
    p1 = np.polyfit(t[pre] - tc, rj["g"][pre], 1)
    p2 = np.polyfit(t[post] - tc, rj["g"][post], 1)
    jump = float(p2[1] - p1[1])
    return dict(
        info, cls="BRIDGEABLE" if abs(jump) <= jump_tecu else "SLIPPED", jump=jump
    )


def zoom_series(w, rj, c, span_s=2400.0, pad=60.0):
    """Detrended TEC of the lock arc that ENTERED the window: TEC minus the line fitted to the
    10 clean minutes before it. Continuity reads as a flat line through the shaded span."""
    t = rj["t"]
    te = w["t0"] - pad
    clean = (rj["ca"] >= CN0_MIN) & (rj["cb"] >= CN0_MIN)
    same = (rj["ka"] == c["ka0"]) & (rj["kb"] == c["kb0"])
    pre = same & clean & (t >= te - 600) & (t <= te)
    if pre.sum() < 60:
        return None
    p1 = np.polyfit(t[pre] - w["t_ca"], rj["g"][pre], 1)
    sel = np.where(same & (t >= w["t_ca"] - span_s) & (t <= w["t_ca"] + span_s))[0][::5]
    x = t[sel] - w["t_ca"]
    return dict(
        x=(x / 60.0).astype(np.float32),
        y=(rj["g"][sel] - np.polyval(p1, x)).astype(np.float32),
        ok=clean[sel],
    )


# ----------------------------------------------------------------------------------- jobs
def lock_census(raw, wins, pad):
    series = {}
    for prn, dct in raw.items():
        items = [dct[h] for h in order(dct)]
        ids = {}
        t = np.array([r[2] for r in items])
        k = np.array([ids.setdefault(r[1], len(ids)) for r in items])
        fp = np.array([r[5] for r in items], dtype=bool)
        el = np.array([r[3] for r in items])
        series[prn] = (t, k, fp, el)
    out = []
    for w in wins:
        te, tx = w["t0"] - pad, w["t1"] + pad
        res = {}
        for prn, (t, k, fp, el) in series.items():
            ib = np.where((t >= te - 120) & (t <= te) & fp & (el >= 25))[0]
            if not len(ib):
                continue
            ia = np.where((t >= tx) & (t <= tx + 300))[0]
            res[prn] = (
                "GONE"
                if not len(ia)
                else ("HELD" if k[ia[0]] == k[ib[-1]] else "RESTARTED")
            )
        out.append(res)
    return out


def pair_job(args):
    (a, b), days, t_lo, t_hi, wins, pad, jump_tecu = args
    t_start = time.time()
    Ac, Ar = load_band(a, days, t_lo, t_hi)
    Bc, Br = load_band(b, days, t_lo, t_hi)
    fa, fb = gtc.FREQ[a], gtc.FREQ[b]
    la, lb = gtc.C / fa, gtc.C / fb
    mpt = gtc.K * (1.0 / fa ** 2 - 1.0 / fb ** 2)
    sysid = SYS_OF[a]
    res = dict(
        pair=(a, b),
        sys=sysid,
        mpt=mpt,
        arcspan={},
        arcs_ds={},
        lockseg={},
        cls=[],
        zoom=[],
        noisy={},
        lock={a: lock_census(Ar, wins, pad), b: lock_census(Br, wins, pad)},
    )
    arcs_by, rj_by = {}, {}
    for prn in sorted(set(Ar) | set(Br)):
        arcs, noisy = (
            product_arcs(Ac, Bc, prn, la, lb, mpt)
            if (prn in Ac and prn in Bc)
            else ([], 0)
        )
        rj = raw_joint(Ar, Br, prn, la, lb, mpt)
        arcs_by[prn], rj_by[prn] = arcs, rj
        res["noisy"][prn] = noisy
        res["arcspan"][prn] = [(x["t0"], x["t1"]) for x in arcs]
        res["arcs_ds"][prn] = [
            (
                x["t"][::30].astype(np.float64),
                x["tec"][::30].astype(np.float32),
                x["vt"][::30].astype(np.float32),
                x["faint"],
            )
            for x in arcs
        ]
        res["lockseg"][prn] = lock_segments(rj) if rj is not None else []
    for w in wins:
        cw, zw = {}, {}
        for prn, rj in rj_by.items():
            if rj is None:
                continue
            c = classify(w, arcs_by[prn], rj, pad, jump_tecu)
            if c is None:
                continue
            if c["cls"] == "NO ARC":
                te = w["t0"] - pad
                prev = [a_ for a_ in arcs_by[prn] if a_["t1"] < te]
                nxt = [a_ for a_ in arcs_by[prn] if a_["t0"] > te]
                c["why"] = (
                    "last arc ended %.0f min before entry"
                    % ((te - prev[-1]["t1"]) / 60)
                    if prev
                    else "no product arc before entry in the whole span"
                )
                if nxt:
                    c["why"] += ", next begins %+.0f min (%s)" % (
                        (nxt[0]["t0"] - te) / 60,
                        nxt[0]["starts_at"],
                    )
                c["why"] += (
                    ", %d arc(s) of this PRN dropped by the noise gate"
                    % res["noisy"][prn]
                )
            if w["sys"] == sysid and w["prn"] == prn:
                c["self"] = True
            cw[prn] = c
            z = zoom_series(w, rj, c, pad=pad)
            if z is not None:
                zw[prn] = z
        res["cls"].append(cw)
        res["zoom"].append(zw)
    res["secs"] = round(time.time() - t_start, 1)
    return res


def closure_job(args):
    """Galileo's triple-frequency closure TEC(E5b x E6) - TEC(E5a x E6): zero for any ionosphere,
    so a jump in it across a window is the instrument. Per window and satellite clean on all
    three bands going in, with all three fleet-ADR arcs unchanged after: the jump of the closure
    and of TEC(E5a x E6) (lines fitted to 10 clean minutes each side, compared at mid-window)."""
    days, t_lo, t_hi, wins, pad = args
    bands = ("gal_e5a", "gal_e5b", "gal_e6")
    raw = [load_band(b, days, t_lo, t_hi)[1] for b in bands]
    lam = [gtc.C / gtc.FREQ[b] for b in bands]
    f = [gtc.FREQ[b] for b in bands]
    m_ac = gtc.K * (1 / f[0] ** 2 - 1 / f[2] ** 2)
    m_bc = gtc.K * (1 / f[1] ** 2 - 1 / f[2] ** 2)
    out = [dict() for _ in wins]
    for prn in sorted(set(raw[0]) & set(raw[1]) & set(raw[2])):
        common = {
            h: raw[0][prn][h]
            for h in set(raw[0][prn]) & set(raw[1][prn]) & set(raw[2][prn])
        }
        hops = order(common)
        if len(hops) < 60:
            continue
        ids = [dict(), dict(), dict()]
        rows = []
        for h in hops:
            r = [raw[i][prn][h] for i in range(3)]
            rows.append(
                (
                    r[0][2],
                    (lam[0] * r[0][0] - lam[2] * r[2][0]) / m_ac,
                    (lam[1] * r[1][0] - lam[2] * r[2][0]) / m_bc,
                    ids[0].setdefault(r[0][1], len(ids[0])),
                    ids[1].setdefault(r[1][1], len(ids[1])),
                    ids[2].setdefault(r[2][1], len(ids[2])),
                    min(r[0][4], r[1][4], r[2][4]),
                )
            )
        a = np.array(rows)
        t, tec, clo = a[:, 0], a[:, 1], a[:, 2] - a[:, 1]
        k = a[:, 3:6].astype(int)
        clean = a[:, 6] >= CN0_MIN
        for wi, w in enumerate(wins):
            te, tx = w["t0"] - pad, w["t1"] + pad
            ib = np.where((t >= te - 120) & (t <= te) & clean)[0]
            if len(ib) < 20:
                continue
            same = (k == k[ib[-1]]).all(axis=1) & clean
            pre = same & (t >= te - 600) & (t <= te)
            post = same & (t >= tx) & (t <= tx + 600)
            if pre.sum() < 60 or post.sum() < 60:
                continue
            tc = 0.5 * (w["t0"] + w["t1"])
            j = {}
            for name, y in (("tec", tec), ("clo", clo)):
                p1 = np.polyfit(t[pre] - tc, y[pre], 1)
                p2 = np.polyfit(t[post] - tc, y[post], 1)
                j[name] = float(p2[1] - p1[1])
            out[wi][prn] = j
    return out


def lock_job(args):
    band, days, t_lo, t_hi, wins, pad = args
    _c, raw = load_band(band, days, t_lo, t_hi)
    return band, lock_census(raw, wins, pad)


# ---------------------------------------------------------------------------------- figures
# outcome colours: the four ordered outcomes pass the dataviz validator (CVD, normal-vision,
# contrast) on white; the neutrals are named in every legend, never colour alone
COL = {
    "SURVIVED": "#2b7bba",
    "BRIDGEABLE": "#1a9e77",
    "SLIPPED": "#d9730d",
    "HELD-UNSEEN": "#9e9e9e",
    "LOCK LOST": "#b0226f",
    "GONE": "#6d6d6d",
    "NO ARC": "#bdbdbd",
    "SELF": "#542788",
}
MRK = {
    "SURVIVED": "o",
    "BRIDGEABLE": "s",
    "SLIPPED": "^",
    "HELD-UNSEEN": "D",
    "LOCK LOST": "X",
    "GONE": "v",
    "NO ARC": ".",
    "SELF": "*",
}


def dt64(t):
    return (np.asarray(t, dtype=np.float64) * 1e3).astype("datetime64[ms]")


def figures(period, pw, pair_res, out, pad):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D

    # A window with no classified arc anywhere (the F-engine was down, say) has nothing to
    # show, and keeping it would stretch the time axis across the gap.
    pw = [(i, w) for i, w in pw if any(pr["cls"][i] for pr in pair_res)]
    if not pw:
        return []
    idx = [i for i, _ in pw]
    wins = [w for _, w in pw]
    lo, hi = wins[0]["t0"] - 3600, wins[-1]["t1"] + 3600
    files = []
    # 1. survival map
    heights = []
    rows = []
    for pr in pair_res:
        prns = [
            p
            for p in sorted(pr["lockseg"])
            if any(s1 > lo and s0 < hi for s0, s1 in pr["lockseg"][p])
            or any(s1 > lo and s0 < hi for s0, s1 in pr["arcspan"][p])
        ]
        rows.append(prns)
        heights.append(max(len(prns), 3))
    fig, axes = plt.subplots(
        len(pair_res),
        1,
        figsize=(20, 0.26 * sum(heights) + 3.5),
        sharex=True,
        gridspec_kw=dict(height_ratios=heights, hspace=0.12),
    )
    for ax, pr, prns in zip(axes, pair_res, rows):
        for i, p in enumerate(prns):
            for s0, s1 in pr["lockseg"][p]:
                if s1 > lo and s0 < hi:
                    ax.plot(
                        dt64([s0, s1]),
                        [i, i],
                        color="#d9d9d9",
                        lw=7,
                        solid_capstyle="butt",
                    )
            for s0, s1 in pr["arcspan"][p]:
                if s1 > lo and s0 < hi:
                    ax.plot(
                        dt64([s0, s1]),
                        [i, i],
                        color="#3f3f3f",
                        lw=2.2,
                        solid_capstyle="butt",
                    )
        for k, w in zip(idx, wins):
            ax.axvspan(dt64(w["t0"]), dt64(w["t1"]), color="#fdb863", alpha=0.45, lw=0)
            for p, c in pr["cls"][k].items():
                if p in prns:
                    key = "SELF" if c.get("self") else c["cls"]
                    ax.plot(
                        dt64(w["t1"] + pad),
                        prns.index(p),
                        MRK[key],
                        color=COL[key],
                        ms=7,
                        mec="white",
                        mew=0.6,
                        zorder=5,
                    )
        for s, lab in EVENTS:
            if lo < ts(s) < hi:
                ax.axvline(dt64(ts(s)), color="#7f7f7f", ls=":", lw=0.9)
        ax.set_yticks(range(len(prns)))
        ax.set_yticklabels(["%s%02d" % (pr["sys"], p) for p in prns], fontsize=7)
        ax.set_ylim(-0.8, len(prns) - 0.2)
        ax.set_title("%s x %s" % pr["pair"], loc="left", fontsize=10)
        ax.grid(axis="x", color="#eeeeee", lw=0.6)
    for w in wins:
        axes[0].annotate(
            "%s %.1f°%s"
            % (w["sat"], w["min_sep"], "" if w["healthy"] else " (unhealthy)"),
            (dt64(w["t_ca"]), 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            rotation=90,
            va="bottom",
        )
    for s, lab in EVENTS:
        if lo < ts(s) < hi:
            axes[-1].annotate(
                lab,
                (dt64(ts(s)), 0.0),
                xycoords=("data", "axes fraction"),
                xytext=(2, -26),
                textcoords="offset points",
                fontsize=6.5,
                rotation=90,
                va="top",
                color="#555555",
            )
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    hand = [
        Line2D(
            [],
            [],
            color="#d9d9d9",
            lw=7,
            label="lock held (both bands' fleet-ADR arcs, no C/N0 gate)",
        ),
        Line2D(
            [],
            [],
            color="#3f3f3f",
            lw=2.2,
            label="dTEC product arc (gnss_tec_chord rules)",
        ),
        Line2D(
            [], [], color="#fdb863", lw=7, alpha=0.6, label="transit, < 5° of boresight"
        ),
    ]
    hand += [
        Line2D([], [], ls="", marker=MRK[k], color=COL[k], ms=7, label=k)
        for k in [
            "SURVIVED",
            "BRIDGEABLE",
            "SLIPPED",
            "HELD-UNSEEN",
            "LOCK LOST",
            "GONE",
            "SELF",
        ]
    ]
    H = fig.get_size_inches()[1]
    fig.subplots_adjust(top=1.0 - 1.25 / H, bottom=0.9 / H)
    fig.legend(
        handles=hand,
        loc="upper center",
        ncol=5,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0 + 0.55 / H),
    )
    fig.suptitle(
        "dTEC arcs through boresight transits: %s  (outcome drawn at each window's exit + %d s)"
        % (period, pad),
        y=1.0 + 0.95 / H,
        fontsize=12,
    )
    f = "%s_%s_survival.png" % (out, period)
    fig.savefig(f, dpi=110, bbox_inches="tight")
    plt.close(fig)
    files.append(f)
    # 2. per-transit zooms: detrended TEC of the lock arc that entered
    n = len(wins)
    fig, axes = plt.subplots(
        n,
        len(pair_res),
        figsize=(5.2 * len(pair_res), 2.1 * n + 1),
        sharex=True,
        squeeze=False,
    )
    for r, (k, w) in enumerate(zip(idx, wins)):
        for cidx, pr in enumerate(pair_res):
            ax = axes[r][cidx]
            ax.axvspan(
                (w["t0"] - w["t_ca"]) / 60,
                (w["t1"] - w["t_ca"]) / 60,
                color="#fdb863",
                alpha=0.45,
                lw=0,
            )
            ax.axhline(0, color="#bbbbbb", lw=0.6)
            cnt = collections.Counter()
            for p, z in pr["zoom"][k].items():
                c = pr["cls"][k][p]
                key = "SELF" if c.get("self") else c["cls"]
                cnt[key] += 1
                col = COL[key]
                y = np.clip(z["y"], -7.5, 7.5)
                yo = np.where(z["ok"], y, np.nan)
                yg = np.where(z["ok"], np.nan, y)
                ax.plot(z["x"], yo, color=col, lw=0.8, alpha=0.9)
                ax.plot(z["x"], yg, color=col, lw=0.5, alpha=0.35, ls=":")
                if key == "LOCK LOST" and len(z["x"]):
                    ax.plot(z["x"][-1], y[-1], "X", color=col, ms=6)
            ax.set_ylim(-8, 8)
            ax.set_xlim(-40, 40)
            ax.tick_params(labelsize=7)
            ax.text(
                0.01,
                0.97,
                "  ".join("%s %d" % (kk, vv) for kk, vv in sorted(cnt.items())),
                transform=ax.transAxes,
                fontsize=6.5,
                va="top",
            )
            if r == 0:
                ax.set_title("%s x %s" % pr["pair"], fontsize=9)
            if cidx == 0:
                ax.set_ylabel(
                    "%s CA %s\n%.1f°  TECU"
                    % (
                        w["sat"],
                        time.strftime("%H:%M", time.gmtime(w["t_ca"])),
                        w["min_sep"],
                    ),
                    fontsize=8,
                )
    for ax in axes[-1]:
        ax.set_xlabel("minutes from closest approach", fontsize=8)
    fig.suptitle(
        "%s: TEC of each arc that ENTERED the transit, minus the line fitted to its last 10 clean "
        "minutes (solid: C/N0 >= 25 on both bands; dotted: below)" % period,
        fontsize=10,
        y=1.0,
    )
    fig.tight_layout()
    f = "%s_%s_zoom.png" % (out, period)
    fig.savefig(f, dpi=100, bbox_inches="tight")
    plt.close(fig)
    files.append(f)
    # 3. the dTEC arcs, slant (left) and vertical (right); arcs that entered a transit coloured by outcome
    fig, axes = plt.subplots(
        len(pair_res), 2, figsize=(22, 11.5), sharex=True, squeeze=False
    )
    for r, pr in enumerate(pair_res):
        fate = {}
        for k in idx:
            for p, c in pr["cls"][k].items():
                if not c.get("self"):
                    fate.setdefault(p, []).append((wins[idx.index(k)], c["cls"]))
        for col in (0, 1):
            ax = axes[r][col]
            for w in wins:
                ax.axvspan(
                    dt64(w["t0"]), dt64(w["t1"]), color="#fdb863", alpha=0.45, lw=0
                )
            for p, arcs in pr["arcs_ds"].items():
                for tt, yy, vv, faint in arcs:
                    if tt[-1] < lo or tt[0] > hi:
                        continue
                    colr, lw = "#9e9e9e", 0.6
                    for w, cl in fate.get(p, []):
                        if tt[0] <= w["t0"] - pad <= tt[-1]:
                            colr, lw = COL[cl], 1.0
                    y = yy if col == 0 else vv
                    ax.plot(
                        dt64(tt),
                        y,
                        color=colr,
                        lw=lw if not (col and faint) else 0.5,
                        alpha=1.0 if not (col and faint) else 0.3,
                    )
            ax.set_ylim(-12, 12) if col == 0 else ax.set_ylim(-8, 8)
            ax.grid(color="#eeeeee", lw=0.6)
            ax.set_title(
                (
                    "%s x %s: slant TEC, arc mean removed"
                    if col == 0
                    else "%s x %s: vertical TEC (thin shell 350 km, per-arc offset fitted; faint = offset "
                    "ill-determined)"
                )
                % pr["pair"],
                loc="left",
                fontsize=9.5,
            )
            if col == 0:
                ax.set_ylabel("TECU")
    for ax in axes[-1]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.set_xlim(dt64(lo), dt64(hi))
    hand = [
        Line2D([], [], color=COL[k], lw=1.2, label="entered a transit: " + k)
        for k in ["SURVIVED", "BRIDGEABLE", "SLIPPED", "LOCK LOST", "GONE"]
    ]
    hand.append(Line2D([], [], color="#9e9e9e", lw=0.8, label="entered no transit"))
    fig.subplots_adjust(top=0.92, hspace=0.2, wspace=0.08)
    fig.legend(
        handles=hand,
        loc="upper center",
        ncol=6,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle("dTEC arcs, %s" % period, y=0.995)
    f = "%s_%s_arcs.png" % (out, period)
    fig.savefig(f, dpi=90, bbox_inches="tight")
    plt.close(fig)
    files.append(f)
    return files


# ------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-lo", required=True, help="UTC, YYYY-MM-DDTHH:MM:SS")
    ap.add_argument("--t-hi", default=None, help="default: now - 5 min")
    ap.add_argument(
        "--split",
        required=True,
        help="the change under test goes live here: windows before it are the control",
    )
    ap.add_argument("--events", default=None)
    ap.add_argument("--bore-deg", type=float, default=5.0)
    ap.add_argument("--pad-s", type=float, default=60.0)
    ap.add_argument(
        "--jump-tecu",
        type=float,
        default=1.5,
        help="a quarter of an E5a cycle in E5a x E6 is 1.43 TECU",
    )
    ap.add_argument("--broker-lines", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.events:
        for line in open(args.events):
            if line.strip() and not line.startswith("#"):
                when, _, label = line.strip().partition(" ")
                EVENTS.append((when, label.strip()))
    t_lo = ts(args.t_lo)
    t_hi = ts(args.t_hi) if args.t_hi else time.time() - 300
    split = ts(args.split)
    days = sorted(
        {time.strftime("%Y%m%d", time.gmtime(t)) for t in np.arange(t_lo, t_hi, 3600.0)}
        | {time.strftime("%Y%m%d", time.gmtime(t_hi))}
    )
    t0 = time.time()
    wins, paths = transits(t_lo, t_hi, args.bore_deg)
    print(
        "ephemeris: %s -> %d windows < %.1f deg (%.0f s)"
        % (
            [os.path.basename(p)[10:19] for p in paths],
            len(wins),
            args.bore_deg,
            time.time() - t0,
        )
    )
    # usable: 10 min of context each side inside the data, not straddling the split
    keep = []
    for w in wins:
        if w["t0"] - args.pad_s - 720 < t_lo or w["t1"] + args.pad_s + 720 > t_hi:
            continue
        if w["t1"] < split:
            w["period"] = "before"
        elif w["t0"] > split + 900:
            w["period"] = "after"
        else:
            continue
        keep.append(w)

    def period_of(a, b):
        if b < split:
            return "before"
        if a > split + 900:
            return "after"
        return None

    ev = [ts(x) for x, _ in EVENTS]
    ctrl = []
    for w in keep:
        got = 0
        for dt in (2700, -2700, 3600, -3600, 5400, -5400):
            c0, c1 = w["t0"] + dt, w["t1"] + dt
            if (
                c0 - args.pad_s - 720 < t_lo
                or c1 + args.pad_s + 720 > t_hi
                or period_of(c0, c1) != w["period"]
                or any(x["t0"] < c1 + 1800 and x["t1"] > c0 - 1800 for x in wins)
                or any(c0 - 600 < e < c1 + 600 for e in ev)
                or any(x["t0"] < c1 + 300 and x["t1"] > c0 - 300 for x in ctrl)
            ):
                continue
            ctrl.append(
                dict(
                    w,
                    t0=c0,
                    t1=c1,
                    t_ca=w["t_ca"] + dt,
                    sat="ctrl %s%+dm" % (w["sat"], dt // 60),
                    sys="-",
                    prn=-1,
                    ctrl=True,
                )
            )
            got += 1
            if got == 2:
                break
    print(
        "control windows (same lengths, no transit within 30 min, no event within 10 min): %s"
        % {q: sum(1 for c in ctrl if c["period"] == q) for q in ("before", "after")}
    )
    keep = keep + ctrl
    for w in keep:
        if w.get("ctrl"):
            continue
        print(
            "  %-6s %s  %s .. %s  CA %s  %.2f deg%s"
            % (
                w["period"],
                w["sat"],
                hm(w["t0"]),
                hm(w["t1"])[6:],
                hm(w["t_ca"])[6:],
                w["min_sep"],
                "" if w["healthy"] else "  UNHEALTHY",
            )
        )
    if args.broker_lines:
        print("broker #142 freeze windows (FROZEN .. CLEAR, gps_l5):")
        for line in open(args.broker_lines):
            if "FROZEN for a boresight" in line or "transit CLEAR" in line:
                print(
                    "   ",
                    line.split("] ")[0].split()[-1],
                    line.split("dead-reckon: ")[-1][:70],
                )
    with Pool(len(PAIRS) + 3) as pool:
        pr_async = pool.map_async(
            pair_job,
            [(p, days, t_lo, t_hi, keep, args.pad_s, args.jump_tecu) for p in PAIRS],
        )
        lk_async = pool.map_async(
            lock_job,
            [(c, days, t_lo, t_hi, keep, args.pad_s) for c in ("gal_e5b", "bds_b2b")],
        )
        cl_async = pool.apply_async(
            closure_job, ((days, t_lo, t_hi, keep, args.pad_s),)
        )
        pair_res = pr_async.get()
        lock_extra = dict(lk_async.get())
        closure = cl_async.get()
    print("pair jobs: %s s" % [r["secs"] for r in pair_res])
    lock = {}
    for pr in pair_res:
        lock.update(pr["lock"])
    lock.update(lock_extra)
    # ---- census
    KINDS = ("before", "before-ctrl", "after", "after-ctrl")
    table, totals = (
        [],
        {
            p: {q: collections.Counter() for q in KINDS}
            for p in ["pairs", "lock", "closure"]
        },
    )
    jumps = {q: dict(tec=[], clo=[]) for q in KINDS}
    for k, w in enumerate(keep):
        kind = w["period"] + ("-ctrl" if w.get("ctrl") else "")
        row = dict(
            period=w["period"],
            kind=kind,
            sat=w["sat"],
            t_ca=hm(w["t_ca"]),
            min_sep=w["min_sep"],
            healthy=w["healthy"],
            t0=hm(w["t0"]),
            t1=hm(w["t1"]),
            pairs={},
            lock={},
        )
        for pr in pair_res:
            cnt = collections.Counter()
            detail = []
            for p, c in sorted(pr["cls"][k].items()):
                if pr["sys"] == "E" and p in closure[k]:
                    c["clo"] = closure[k][p]["clo"]
                    c["tec3"] = closure[k][p]["tec"]
                key = "SELF" if c.get("self") else c["cls"]
                if (
                    pr["sys"] == "E"
                    and key in ("SURVIVED", "BRIDGEABLE")
                    and "clo" in c
                ):
                    totals["closure"][kind][
                        "%s %s"
                        % (
                            key,
                            "clo-ok" if abs(c["clo"]) <= args.jump_tecu else "clo-JUMP",
                        )
                    ] += 1
                    if abs(c["clo"]) > args.jump_tecu:
                        detail.append(
                            "%s%02d %s but closure jumps %+.2f TECU (TEC %+.2f)"
                            % (pr["sys"], p, key, c["clo"], c["tec3"])
                        )
                cnt[key] += 1
                if key != "SELF":
                    totals["pairs"][kind][key] += 1
                    if "jump" in c:
                        jumps[kind]["tec"].append(abs(c["jump"]))
                    if "clo" in c:
                        jumps[kind]["clo"].append(abs(c["clo"]))
                if key not in ("SURVIVED", "SELF"):
                    detail.append(
                        "%s%02d %s%s%s%s"
                        % (
                            pr["sys"],
                            p,
                            key,
                            (" jump %+.2f" % c["jump"]) if "jump" in c else "",
                            (
                                " at %s (%s)"
                                % (
                                    hm(c["t_lost"])[6:],
                                    pr["pair"][0 if c["band"] == "a" else 1],
                                )
                            )
                            if "t_lost" in c
                            else "",
                            (": " + c["why"]) if "why" in c else "",
                        )
                    )
            row["pairs"]["%s x %s" % pr["pair"]] = dict(counts=dict(cnt), detail=detail)
        for ch in CHAINS:
            st = collections.Counter(
                v
                for p, v in lock[ch][k].items()
                if not (SYS_OF[ch] == w["sys"] and p == w["prn"])
            )
            row["lock"][ch] = dict(st)
            totals["lock"][kind].update({"%s %s" % (ch, s): n for s, n in st.items()})
        table.append(row)
    json.dump(
        dict(
            windows=keep,
            table=table,
            totals={k: {q: dict(v) for q, v in d.items()} for k, d in totals.items()},
            jumps={
                q: {kk: [round(x, 3) for x in v] for kk, v in d.items()}
                for q, d in jumps.items()
            },
            args=vars(args),
            days=days,
        ),
        open(args.out + "_census.json", "w"),
        indent=1,
    )
    print()
    for row in table:
        if row["kind"].endswith("-ctrl"):
            continue
        print(
            "%-6s %-4s CA %s  %.1f deg%s"
            % (
                row["period"],
                row["sat"],
                row["t_ca"],
                row["min_sep"],
                "" if row["healthy"] else "  UNHEALTHY",
            )
        )
        for pn, v in row["pairs"].items():
            print(
                "    %-18s %s"
                % (pn, "  ".join("%s %d" % kv for kv in sorted(v["counts"].items())))
            )
            for dline in v["detail"]:
                print("        %s" % dline)
        print(
            "    lock held/entering: %s"
            % "  ".join(
                "%s %d/%d" % (ch.split("_")[1], v.get("HELD", 0), sum(v.values()))
                for ch, v in row["lock"].items()
            )
        )
    print()
    for q in KINDS:
        c = totals["pairs"][q]
        n = sum(c.values())
        ent = n - c["NO ARC"]
        tj, cj = np.array(jumps[q]["tec"]), np.array(jumps[q]["clo"])
        print(
            "%-11s entered (not NO ARC) %d: %s"
            % (
                q,
                ent,
                "  ".join(
                    "%s %.0f%%" % (kk, 100.0 * c[kk] / ent)
                    for kk in CLASSES
                    if c[kk] and kk != "NO ARC"
                )
                if ent
                else "-",
            )
        )
        if len(tj):
            print(
                "            |TEC jump| where both arcs held: n %d, median %.2f, > 1.5 TECU %.0f%%, > 10 TECU %.0f%%"
                % (
                    len(tj),
                    np.median(tj),
                    100.0 * np.mean(tj > 1.5),
                    100.0 * np.mean(tj > 10),
                )
            )
        if len(cj):
            print(
                "            |closure jump| (Galileo, all three arcs held): n %d, median %.2f, > 1.5 TECU %.0f%%"
                % (len(cj), np.median(cj), 100.0 * np.mean(cj > 1.5))
            )
        print(
            "%-11s victims (all pairs, all transits): %d  %s"
            % (
                q,
                n,
                "  ".join(
                    "%s %d (%.0f%%)" % (k, c[k], 100.0 * c[k] / n)
                    for k in CLASSES
                    if c[k]
                )
                if n
                else "-",
            )
        )
        lk = collections.Counter()
        for kk, v in totals["lock"][q].items():
            lk[kk.split()[-1]] += v
        m = sum(lk.values())
        print(
            "       Galileo survivors vs the triple-frequency closure: %s"
            % "  ".join("%s %d" % kv for kv in sorted(totals["closure"][q].items()))
        )
        print(
            "       tracking lock through the window (all chains): %s"
            % (
                "  ".join(
                    "%s %d (%.0f%%)" % (k, lk[k], 100.0 * lk[k] / m)
                    for k in ("HELD", "RESTARTED", "GONE")
                )
                if m
                else "-"
            )
        )
    files = []
    for q in ("before", "after"):
        pw = [
            (i, w) for i, w in enumerate(keep) if w["period"] == q and not w.get("ctrl")
        ]
        files += figures(q, pw, pair_res, args.out, args.pad_s)
    print("\nwrote %s_census.json and %s" % (args.out, " ".join(files)))
    print("total %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
