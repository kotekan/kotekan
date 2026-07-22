#!/usr/bin/env python3
"""The channel-level ADR-wander decomposition (the G26 instrument, reusable).
  python3 chan_decompose.py <chan_dump> <obs.jsonl> <sys> <prn>
Reports: live combiner ADR scatter vs smoothed sum-phase wander (the fold-walk
discriminator), intercept-vs-delay decomposition, PCA mode0 linearity, tau wander.
"""
import sys, json
import numpy as np

def main():
    dump, obspath, want_sys, want_prn = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    # ---- parse dump (preallocated float32; tolerant of torn lines) ----
    import os
    est = os.path.getsize(dump) // 45 + 1000
    A = np.empty((est, 5), np.float64)
    n = 0
    for line in open(dump):
        p = line.split()
        if len(p) != 5:
            continue
        try:
            A[n, 0] = float(p[0]); A[n, 1] = float(p[1]); A[n, 2] = float(p[2])
            A[n, 3] = float(p[3]); A[n, 4] = float(p[4])
        except (ValueError, IndexError):
            continue
        n += 1
    A = A[:n]
    t, ch, re_, im_, en = A[:, 0], A[:, 1].astype(int), A[:, 2], A[:, 3], A[:, 4]
    ut, inv = np.unique(t, return_inverse=True)
    M = np.zeros((len(ut), 10), complex); W = np.zeros((len(ut), 10))
    M[inv, ch] = re_ + 1j * im_; W[inv, ch] = en
    full = np.all(W > 0, axis=1)
    ut = ut[full]; M = M[full]
    print(f"{want_sys}{want_prn}: {len(ut)} full epochs, span {ut[-1]-ut[0]:.0f} s")
    Vsum = M.sum(axis=1)

    K = 31; ker = np.ones(K) / K
    s2 = np.convolve(Vsum**2, ker, mode='same')
    phi_sum = np.unwrap(np.angle(s2)) / 2.0
    psi = np.zeros((len(ut), 10))
    for c in range(10):
        psi[:, c] = np.angle(np.convolve(M[:, c] * np.conj(Vsum), ker, mode='same'))
    sl = slice(K, -K)
    ut2 = ut[sl]; psi = psi[sl]; phi_sum = phi_sum[sl]; Msl = np.abs(M[sl])

    WIN = 180.0
    def wander(series, tvec):
        out = []
        t0 = tvec[0]; nw = int((tvec[-1] - t0) // WIN)
        for w in range(nw):
            m = (tvec >= t0 + w * WIN) & (tvec < t0 + (w + 1) * WIN)
            if m.sum() < 100:
                continue
            x = tvec[m] - tvec[m][0]
            r = series[m] - np.polyval(np.polyfit(x, series[m], 2), x)
            out.append(np.std(r))
        return (np.median(out), len(out)) if out else (np.nan, 0)

    f = np.arange(10) - 5.0
    s_t = np.full(len(ut2), np.nan); a_t = np.full(len(ut2), np.nan)
    for k in range(len(ut2)):
        co = np.polyfit(f, psi[k], 1, w=Msl[k])
        s_t[k] = co[0]; a_t[k] = co[1]
    fw = np.sum(Msl * f, axis=1) / np.sum(Msl, axis=1)
    lam = 299792458.0 / 1176.45e6
    W1, n1 = wander(phi_sum, ut2)
    W2, _ = wander(phi_sum + a_t, ut2)
    W3, _ = wander(s_t * fw, ut2)
    tau = s_t / (2 * np.pi * 1e6) * 1e9
    Wt, _ = wander(tau, ut2)
    print(f"  sum-phase wander:      {W1:8.3f} rad = {W1/(2*np.pi)*lam:6.3f} m /180s  ({n1} wins)")
    print(f"  intercept (carrier):   {W2:8.3f} rad = {W2/(2*np.pi)*lam:6.3f} m")
    print(f"  delay-leak (s*fbar):   {W3:8.3f} rad = {W3/(2*np.pi)*lam:6.3f} m")
    print(f"  tau wander:            {Wt:8.3f} ns  (0.1 chip = 9.8 ns)")

    # PCA of residuals
    def resid(series):
        out = np.full(len(ut2), np.nan)
        t0 = ut2[0]; nw = int((ut2[-1] - t0) // WIN)
        for w in range(nw):
            m = (ut2 >= t0 + w * WIN) & (ut2 < t0 + (w + 1) * WIN)
            if m.sum() < 100:
                continue
            x = ut2[m] - ut2[m][0]
            out[m] = series[m] - np.polyval(np.polyfit(x, series[m], 2), x)
        return out
    R = np.array([resid(psi[:, c]) for c in range(10)])
    good = ~np.any(np.isnan(R), axis=0)
    U, S, _ = np.linalg.svd(R[:, good] - R[:, good].mean(axis=1, keepdims=True),
                            full_matrices=False)
    lin = (f - f.mean()); lin /= np.max(np.abs(lin))
    print(f"  PCA var fracs {np.round(S**2/np.sum(S**2),3)[:3]}  mode0-vs-linear r={np.corrcoef(U[:,0],lin)[0,1]:+.3f}")

    # ---- live obs scatter, same soak ----
    rows = []
    for line in open(obspath):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get('sys') != want_sys or d.get('prn') != want_prn:
            continue
        if d.get('adr_cycles') is None or (d.get('sig') or 0) < 10:
            continue
        rows.append((d['t'], d['adr_cycles'] * lam, d.get('adr_arc')))
    rows.sort()
    B = np.array(rows, float)
    if len(B) > 100:
        tb, yb, arc = B.T
        wins = []
        t0 = tb[0]; nw = int((tb[-1] - t0) // WIN); cad = np.median(np.diff(tb))
        for w in range(nw):
            m = (tb >= t0 + w * WIN) & (tb < t0 + (w + 1) * WIN)
            if m.sum() < 0.8 * WIN / max(cad, 1) or len(set(arc[m])) != 1:
                continue
            x = tb[m] - tb[m][0]
            r = yb[m] - np.polyval(np.polyfit(x, yb[m], 2), x)
            wins.append(1.4826 * np.median(np.abs(r - np.median(r))))
        print(f"  LIVE combiner ADR scatter: {np.median(wins):6.3f} m ({len(wins)} wins)")
        print(f"  >>> RATIO live/underlying: {np.median(wins)/(W1/(2*np.pi)*lam):5.1f}x")

if __name__ == "__main__":
    main()
