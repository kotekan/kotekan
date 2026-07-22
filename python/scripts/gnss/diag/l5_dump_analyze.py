#!/usr/bin/env python3
"""Per-record (1 kHz) forensics of the L5-band ADR wander from the combiner phase dump.

Dump line: utc prn ar ai e2 l2 cphase doppler cp ahr ahi
  ar/ai   = full-band despread prompt A (overlay STILL ON for pilots)
  cphase  = dcmd, commanded carrier-phase INCREMENT (cycles) since previous record
  ahr/ahi = head segment of A (chip k side of the code-period boundary)

Reconstruction: the pilot overlay flips A's sign record-to-record; we flatten it
data-driven (choose s_k=+-1 maximizing Re[s_k A_k conj(s_{k-1} A_{k-1})]) -- exact
whenever per-record SNR isn't deep in the noise, and errors show up as 0.5-cycle
outliers we count separately. Then
  dres_k = arg(s_k A_k conj(s_{k-1} A_{k-1}))/2pi,   ADR_k = sum(dcmd - dres)
Outputs per PRN:
  (1) sanity: reconstructed-ADR wander (detrended, m) at obs cadence ~ REST scatter
  (2) micro-null census at 1 kHz: |A| fade depth/duration histogram (invisible at 1 Hz)
  (3) wrap census: |dres| > 0.35 cyc events, their |A| context
  (4) attribution: wander increments split into (a) records at retunes (dcmd rate
      change > eps), (b) records in fades (|A| < half its median), (c) clean records
      -- who carries the variance?
  python3 l5_dump_analyze.py <dump.txt> <band_freq_hz> [prn]
"""
import sys
import numpy as np

def main():
    path = sys.argv[1]
    fc = float(sys.argv[2])
    want = int(sys.argv[3]) if len(sys.argv) > 3 else None
    lam = 299792458.0 / fc
    data = {}
    for line in open(path):
        p = line.split()
        if len(p) != 11:
            continue
        prn = int(p[1])
        if want is not None and prn != want:
            continue
        data.setdefault(prn, []).append([float(x) for x in
                                         (p[0], p[2], p[3], p[6], p[7], p[8])])
    for prn in sorted(data):
        A = np.array(data[prn])
        if len(A) < 5000:
            print("PRN %d: only %d records, skip" % (prn, len(A)))
            continue
        t, ar, ai, dcmd, dop, cp = A.T
        o = np.argsort(t, kind="stable")
        t, ar, ai, dcmd, dop, cp = t[o], ar[o], ai[o], dcmd[o], dop[o], cp[o]
        v = ar + 1j * ai
        amp = np.abs(v)
        dt = np.diff(t)
        trec = np.median(dt)
        # contiguous segments (gap > 3 records breaks continuity like the combiner's gate)
        brk = np.where(dt > 3.5 * trec)[0]
        segs = np.split(np.arange(len(t)), brk + 1)
        segs = [s for s in segs if len(s) > 4000]
        print("\nPRN %d: %d records, t_rec %.4f s, %d segments >4ks, amp med %.3g" %
              (prn, len(t), trec, len(segs), np.median(amp)))
        for si, s in enumerate(segs[:3]):
            ts, vs, dc = t[s], v[s], dcmd[s]
            am = np.abs(vs)
            # overlay sign-flatten
            prod = vs[1:] * np.conj(vs[:-1])
            sflip = np.where(prod.real < 0, -1.0, 1.0)
            sgn = np.concatenate([[1.0], np.cumprod(sflip)])
            w = vs * sgn
            dres = np.angle(w[1:] * np.conj(w[:-1])) / (2 * np.pi)
            dadr = dc[1:] - dres
            adr = np.concatenate([[0.0], np.cumsum(dadr)])
            x = ts - ts[0]
            res_m = (adr - np.polyval(np.polyfit(x, adr, 3), x)) * lam
            # (1) wander at ~2 s cadence like the obs metric
            step = max(1, int(round(2.0 / trec)))
            sc = 1.4826 * np.median(np.abs(res_m - np.median(res_m)))
            print("  seg%d: %.0f s  reconstructed-ADR wander (cubic detrend) %.3f m" %
                  (si, x[-1], sc))
            # (2) micro-null census
            medamp = np.median(am)
            fade = am < 0.5 * medamp
            deep = am < 0.25 * medamp
            # fade runs
            runs = []
            i = 0
            while i < len(fade):
                if fade[i]:
                    j = i
                    while j < len(fade) and fade[j]:
                        j += 1
                    runs.append(j - i)
                    i = j
                else:
                    i += 1
            runs = np.array(runs) if runs else np.array([0])
            print("    micro-nulls: frac(|A|<0.5med)=%.3f  frac(<0.25med)=%.3f  "
                  "fade-runs n=%d med %.0f rec  p95 %.0f rec (%.0f ms)" %
                  (fade.mean(), deep.mean(), len(runs), np.median(runs),
                   np.percentile(runs, 95), np.percentile(runs, 95) * trec * 1e3))
            # (3) wrap census
            big = np.abs(dres) > 0.35
            amp_pair = np.minimum(am[1:], am[:-1])
            print("    wrap-candidates |dres|>0.35cyc: %d (%.4f/rec)  their min|A|/med: %.2f "
                  "(all-rec %.2f)" %
                  (big.sum(), big.mean(), np.median(amp_pair[big]) / medamp if big.any() else -1,
                   np.median(amp_pair) / medamp))
            # (4) attribution of wander variance
            wander_inc = np.diff(res_m)   # per-record increment of the DETRENDED wander (m)
            ddc = np.abs(np.diff(dc))     # dcmd rate change marks retunes
            eps = np.percentile(ddc, 99.5)
            retune = np.concatenate([[False], ddc > max(eps, 1e-6)])[: len(wander_inc)]
            fad = fade[1:][: len(wander_inc)]
            cln = ~(retune | fad)
            tot = np.sum(wander_inc ** 2)
            def part(name, msk):
                if msk.sum() == 0:
                    print("    %-22s 0 recs" % name); return
                print("    %-22s %6.2f%% of recs, %6.2f%% of wander variance  (rms %.4f m/rec)"
                      % (name, 100 * msk.mean(), 100 * np.sum(wander_inc[msk] ** 2) / tot,
                         np.sqrt(np.mean(wander_inc[msk] ** 2))))
            print("    variance attribution:")
            part("retune records", retune)
            part("fade records", fad & ~retune)
            part("clean records", cln)
            # commanded-half structure: dcmd rate (Hz) stats
            f_cmd = dc / trec
            print("    dcmd rate: med %.1f Hz  std %.2f Hz  retunes/s %.2f" %
                  (np.median(f_cmd), np.std(f_cmd), retune.sum() / x[-1]))

if __name__ == "__main__":
    main()
