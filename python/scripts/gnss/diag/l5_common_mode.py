#!/usr/bin/env python3
"""Discriminate the L5-band ADR systematic: PER-SAT (code-Doppler/timing/multipath)
vs COMMON-MODE (front-end/LO-at-tuning/RFI).

Method: global 180 s windows; per sat with >=80% coverage, quadratic-detrend ADR (m)
on the window -> residual on an integer-second grid; then
  (1) pairwise Pearson r between simultaneous sats (same + cross constellation),
  (2) per-window common-mode fraction (var of mean-across-sats / mean var),
  (3) per-arc scatter vs |doppler| / |dopp-rate| correlation,
  (4) residual autocorrelation timescale + dominant periods for the strongest sat.

  python3 l5_common_mode.py <band_label> <obs1.jsonl> [obs2.jsonl ...]
band_label in {l5,l2c,l1} (sets wavelength).
"""
import sys, json, math
import numpy as np

C_L = 299792458.0
FREQ = {"l5": 1176.45e6, "l2c": 1227.60e6, "l1": 1575.42e6}
WIN = 180.0          # window length s
COVER = 0.8          # min coverage in window
MIN_SIG_LOAD = 8.0
MIN_SIG_WIN = 15.0
ORDER = 2

def load(paths, lam):
    per = {}   # key "G32" -> list (t, adr_m, doppler)
    for path in paths:
        for line in open(path):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("adr_cycles") is None:
                continue
            if (d.get("sig") or 0.0) < MIN_SIG_LOAD:
                continue
            key = "%s%02d" % (d.get("sys", "?"), d["prn"])
            per.setdefault(key, []).append(
                (d["t"], d["adr_cycles"] * lam, d.get("doppler_hz") or 0.0,
                 d.get("adr_arc"), d.get("sig") or 0.0))
    for v in per.values():
        v.sort()
    return per

def main():
    band = sys.argv[1].lower()
    lam = C_L / FREQ[band]
    per = load(sys.argv[2:], lam)
    if not per:
        print("no data"); return
    t0 = min(v[0][0] for v in per.values())
    t1 = max(v[-1][0] for v in per.values())
    print("band %s  sats: %s  span %.0f s" % (band, " ".join(sorted(per)), t1 - t0))

    # residual grids per window: win_idx -> {sat: np.array over grid}
    nwin = int((t1 - t0) // WIN)
    grids = {}
    arc_stats = []   # (sat, scatter, med|dop|, |doprate|)
    for sat, rows in per.items():
        t = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
        dop = np.array([r[2] for r in rows]); arc = np.array([r[3] for r in rows])
        sig = np.array([r[4] for r in rows])
        cad = np.median(np.diff(t)) if len(t) > 3 else 2.0
        for w in range(nwin):
            ws, we = t0 + w * WIN, t0 + (w + 1) * WIN
            m = (t >= ws) & (t < we)
            if m.sum() < COVER * WIN / max(cad, 1.0):  # coverage vs actual cadence
                continue
            if np.median(sig[m]) < MIN_SIG_WIN:
                continue
            tw, yw = t[m], y[m]
            if len(set(arc[m])) != 1:            # arc change inside window -> skip
                continue
            if np.max(np.diff(tw)) > 10.0:       # internal gap
                continue
            co = np.polyfit(tw - tw[0], yw, ORDER)
            res = yw - np.polyval(co, tw - tw[0])
            # integer-second grid via interpolation
            gs = np.arange(math.ceil(ws), math.floor(we))
            gm = (gs >= tw[0]) & (gs <= tw[-1])
            gv = np.full(len(gs), np.nan)
            gv[gm] = np.interp(gs[gm], tw, res)
            grids.setdefault(w, {})[sat] = gv
            sc = 1.4826 * np.median(np.abs(res - np.median(res)))
            dr = abs(np.polyfit(tw - tw[0], dop[m], 1)[0])
            arc_stats.append((sat, sc, np.median(np.abs(dop[m])), dr))

    # (1) pairwise correlations
    same, cross, allr = [], [], []
    pair_detail = {}
    for w, sats in grids.items():
        keys = sorted(sats)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = sats[keys[i]], sats[keys[j]]
                m = ~(np.isnan(a) | np.isnan(b))
                if m.sum() < 100:
                    continue
                ra, rb = a[m], b[m]
                if np.std(ra) == 0 or np.std(rb) == 0:
                    continue
                r = float(np.corrcoef(ra, rb)[0, 1])
                allr.append(r)
                (same if keys[i][0] == keys[j][0] else cross).append(r)
                pk = (keys[i], keys[j])
                pair_detail.setdefault(pk, []).append(r)
    def summ(name, arr):
        if not arr:
            print("  %-18s (none)" % name); return
        arr = np.array(arr)
        print("  %-18s n=%3d  median r=%+.3f   mean r=%+.3f   frac|r|>0.5: %.0f%%"
              % (name, len(arr), np.median(arr), np.mean(arr),
                 100 * np.mean(np.abs(arr) > 0.5)))
    print("\n(1) PAIRWISE residual correlation (window %.0fs, quad detrend):" % WIN)
    summ("same-constellation", same)
    summ("cross-constellation", cross)
    summ("ALL pairs", allr)

    # (2) common-mode fraction per window
    cmfs = []
    for w, sats in grids.items():
        if len(sats) < 3:
            continue
        M = np.vstack([sats[k] for k in sorted(sats)])
        m = ~np.any(np.isnan(M), axis=0)
        if m.sum() < 100:
            continue
        M = M[:, m]
        M = M - M.mean(axis=1, keepdims=True)
        cm = M.mean(axis=0)
        cmf = np.var(cm) / np.mean(np.var(M, axis=1))
        cmfs.append((cmf, M.shape[0]))
    if cmfs:
        med = np.median([c for c, n in cmfs])
        print("\n(2) COMMON-MODE fraction (var(mean)/mean(var), %d windows w/ >=3 sats):"
              % len(cmfs))
        print("    median %.3f   (1.0 = fully common, 1/nsat = independent; nsat med %d -> indep floor %.3f)"
              % (med, np.median([n for c, n in cmfs]),
                 1.0 / np.median([n for c, n in cmfs])))

    # (3) scatter vs doppler
    if len(arc_stats) > 5:
        sc = np.array([a[1] for a in arc_stats])
        dp = np.array([a[2] for a in arc_stats])
        dr = np.array([a[3] for a in arc_stats])
        good = sc > 0
        r_dp = np.corrcoef(dp[good], sc[good])[0, 1]
        r_dr = np.corrcoef(dr[good], sc[good])[0, 1]
        print("\n(3) window scatter vs Doppler (n=%d): r(|dop|)=%+.2f  r(|doprate|)=%+.2f"
              % (good.sum(), r_dp, r_dr))
        print("    scatter med %.3f m  p10 %.3f  p90 %.3f" %
              (np.median(sc), np.percentile(sc, 10), np.percentile(sc, 90)))

    # (4) time structure: strongest sat = most windows
    from collections import Counter
    cnt = Counter()
    for w, sats in grids.items():
        for k in sats:
            cnt[k] += 1
    if cnt:
        best = cnt.most_common(1)[0][0]
        # stitch that sat's windows for autocorr (per-window, then average)
        acs, periods = [], Counter()
        for w, sats in grids.items():
            if best not in sats:
                continue
            v = sats[best]
            m = ~np.isnan(v)
            if m.sum() < 150:
                continue
            x = v[m] - v[m].mean()
            ac = np.correlate(x, x, "full")[len(x) - 1:]
            ac /= ac[0]
            acs.append(ac[:90])
            # dominant period via FFT
            ps = np.abs(np.fft.rfft(x)) ** 2
            fr = np.fft.rfftfreq(len(x), 1.0)
            k = np.argmax(ps[1:]) + 1
            periods[round(1.0 / fr[k])] += 1
        if acs:
            L = min(len(a) for a in acs)
            ac = np.mean([a[:L] for a in acs], axis=0)
            first0 = next((i for i, v in enumerate(ac) if v < 0), L)
            print("\n(4) time structure of %s residual (%d windows):" % (best, len(acs)))
            print("    autocorr: r(1s)=%.2f r(5s)=%.2f r(15s)=%.2f r(30s)=%.2f  first zero-cross ~%ds"
                  % (ac[1], ac[5], ac[15], ac[30] if L > 30 else float('nan'), first0))
            print("    dominant FFT periods (s, count): %s" % periods.most_common(5))

if __name__ == "__main__":
    main()
