#!/usr/bin/env python3
"""Single-band ADR carrier-phase scatter (meters), reusing gnss_dtec_live's validated
arc-split + quadratic detrend + trim high-pass. This is the A/B metric for the overlay-wipe
fix: L5 (overlay pilot, was squared) vs L2C (navwipe, high-SNR reference).

  python3 l5_adr_scatter.py <obs.jsonl> <band>   band in {l5,l2c,e5a,b2a,l1,e1,b1c}
"""
import sys, os, json, statistics

sys.path.insert(0, "/home/lwlab/airspy_gps/kotekan/python/scripts/gnss")
import gnss_dtec_live as dt

# carrier wavelength per band (m)
FREQ = {"l5": 1176.45e6, "l2c": 1227.60e6, "e5a": 1176.45e6, "b2a": 1176.45e6,
        "l1": 1575.42e6, "e1": 1575.42e6, "b1c": 1575.42e6}


def main():
    path, band = sys.argv[1], sys.argv[2].lower()
    lam = dt.C_L / FREQ[band]
    min_sig = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    max_gap, min_arc, order = 10.0, 60.0, 2
    obs = dt.load_obs(path, 0.0, min_sig)          # {prn: [(t,adr,trim,arc,dop)...]}
    for sub_label, sub in (("raw", 0), ("trim-sub", 1)):
        tot, npts = [], 0
        for prn, rows in obs.items():
            # split into pieces on combiner arc change (adr/trim reset)
            pieces, cur, prev = [], [], None
            for r in rows:
                if prev is not None and r[3] != prev and cur:
                    pieces.append(cur); cur = []
                cur.append(r); prev = r[3]
            if cur:
                pieces.append(cur)
            for piece in pieces:
                if len(piece) < 10:
                    continue
                common = [r[0] for r in piece]
                adr = [r[1] for r in piece]
                if sub and all(r[2] is not None for r in piece):
                    tr = [r[2] for r in piece]
                    # high-pass the trim (same as dtec_live): remove its per-piece line
                    n = len(common); mt = sum(common)/n; mv = sum(tr)/n
                    den = sum((t-mt)**2 for t in common) or 1.0
                    sl = sum((t-mt)*(v-mv) for t, v in zip(common, tr))/den
                    tr = [v-(mv+sl*(t-mt)) for t, v in zip(common, tr)]
                    y = [(a-c)*lam for a, c in zip(adr, tr)]
                elif sub:
                    continue  # trim not exported for this piece
                else:
                    y = [a*lam for a in adr]
                sc = dt.arc_scatter(common, y, max_gap, min_arc, order)
                tot += [r for r, nn, s in sc]
                npts += sum(nn for r, nn, s in sc)
        if tot:
            print("  %-9s %d arcs  scatter med %.4f m  p90 %.4f m  (mean %.4f, %d pts)"
                  % (sub_label, len(tot), statistics.median(tot),
                     sorted(tot)[int(len(tot)*0.9)] if len(tot) > 1 else tot[0],
                     statistics.mean(tot), npts))
        else:
            print("  %-9s no arcs" % sub_label)


if __name__ == "__main__":
    main()
