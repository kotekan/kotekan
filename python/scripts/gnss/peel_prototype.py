#!/usr/bin/env python3
"""VOLTAGE PEEL prototype (single dish, one PRN) on the L1 C/A capture -- proves the method and
measures the achievable PEEL DEPTH before building the kotekan GnssVoltagePeel stage.

Reconstruct PRN's transmitted C/A waveform from the tracked model and subtract it from the voltage:
    g[n] = Re( 2 * a_smooth(t) * e^{i wc (off+n)} ) * code(cp(t))[n] * navbit(t)      (real passband)
    residual = x - g
'a_smooth' is a SLOWLY-varying complex gain (the dish gain toward the sat is ~constant over seconds),
estimated by low-passing the per-record despread with the nav bit removed. The nav bit lives in the
replica (decoded per 20 ms), NOT in a -- so a stays smooth and the +-1 flips are reconstructed exactly.

Peel depth = |A_orig| / |A_resid|, the drop in the sat's coherent despread after subtraction, vs the
gain-estimation window W (longer W -> better a -> deeper peel, until the true gain varies). Uses the
CLOCK-CORRECTED on-peak code track (l1_code_drift): an off-peak replica would peel poorly -- this is
why the l-a work was the prerequisite.
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l1_code_drift import ca_code, FILE, FS, FOFF, CHIP, FCARR, NMS, _load

def peel(prn=32, span_s=9.0):
    x, off, dur = _load(FILE)
    x = x[:int(span_s*FS)]; nrec = len(x)//NMS
    c = ca_code(prn); Crep = np.conj(np.fft.fft(c[(np.arange(NMS)*CHIP/FS % 1023).astype(np.int64)]))
    # coarse+fine Doppler
    def acc(d): return sum(np.abs(np.fft.ifft(np.fft.fft(
        x[b*NMS:(b+1)*NMS]*np.exp(-2j*np.pi*(FOFF+d)*np.arange(off+b*NMS, off+b*NMS+NMS)/FS))*Crep))**2
        for b in range(20))
    d0 = max(np.arange(-5000, 5001, 250), key=lambda d: acc(d).max())
    dop = max(np.arange(d0-200, d0+201, 20), key=lambda d: acc(d).max())
    wc = 2*np.pi*(FOFF+dop)/FS
    # per-record correlation over all lags -> PEAK-TRACK the on-peak code phase (clock-corrected drift)
    corr = np.empty((nrec, NMS), complex)
    for r in range(nrec):
        m = r*NMS+np.arange(NMS)
        corr[r] = np.fft.ifft(np.fft.fft(x[m]*np.exp(-1j*wc*(off+m)))*Crep)
    pk = np.argmax(np.abs(corr), 1)
    lag = np.polyval(np.polyfit(np.arange(nrec), np.unwrap(pk/NMS*2*np.pi)/(2*np.pi)*NMS, 1), np.arange(nrec))
    i0 = np.floor(lag).astype(int) % NMS; f = lag-np.floor(lag)
    A = corr[np.arange(nrec), i0]*(1-f)+corr[np.arange(nrec), (i0+1) % NMS]*f   # per-1ms complex despread (no bit)
    A /= (NMS/2.0)                                                              # -> complex gain a (= A_mean*2)
    # decode nav bit (20 ms epochs): sign of the drift-removed coherent sum
    t = np.arange(nrec)*1e-3
    ph = np.polyval(np.polyfit(t, np.unwrap(np.angle(A**2)), 8), t)/2           # smooth carrier-phase track
    Ad = A*np.exp(-1j*ph)
    nb = (nrec+19)//20; bit = np.ones(nrec)
    for b in range(nb):
        sl = slice(b*20, b*20+20); s = Ad[sl].sum()
        bit[sl] = np.sign(np.real(s*np.exp(-1j*np.angle((Ad[sl]**2).sum())/2))) or 1.0
    # on-peak sampled code + carrier phasor per sample (absolute-anchored), and the nav bit per sample
    def reconstruct(a_gain):
        g = np.empty(nrec*NMS)
        for r in range(nrec):
            m = r*NMS+np.arange(NMS); n = off+m
            # code sampled at the tracked peak: chip advances CHIP/FS per sample, offset by the sub-sample lag
            ph_code = ((np.arange(NMS) - lag[r]) * (CHIP/FS)) % 1023.0
            cd = c[ph_code.astype(np.int64)]
            g[m] = np.real(a_gain[r]*np.exp(1j*wc*n)) * cd * bit[r]      # real passband replica (unit scale)
        gg = g.dot(g)
        return g * (x[:len(g)].dot(g)/gg if gg > 0 else 0.0)            # LS scale fit -> exact amplitude
    # despread helper on any real stream -> integrated |A| (coherent over the whole span, bit+carrier removed)
    def despread_power(sig):
        a = np.empty(nrec, complex)
        for r in range(nrec):
            m = r*NMS+np.arange(NMS)
            cc = np.fft.ifft(np.fft.fft(sig[m]*np.exp(-1j*wc*(off+m)))*Crep)
            a[r] = cc[i0[r]]*(1-f[r])+cc[(i0[r]+1) % NMS]*f[r]
        a = a*np.exp(-1j*ph)*bit                                    # align carrier + bit -> coherent
        return np.abs(a.mean())                                     # integrated coherent amplitude
    A_orig = despread_power(x)
    print("PRN %d dop %+.0f, %d records (%.1fs). original coherent |A| = %.4f" % (prn, dop, nrec, span_s, A_orig))
    print("  W(ms)  peel_depth(dB)   (a smoothed over W records; longer W = better gain est)")
    for W in (1, 5, 20, 100, 500, nrec):
        # DEROTATE by the tracked carrier phase + remove the nav bit -> a slowly-varying gain that can
        # be averaged; smooth over W records (better gain estimate); put the phase + bit back into R.
        g_deph = A*bit*np.exp(-1j*ph)
        if W > 1:
            k = np.ones(W)/W
            g_sm = np.convolve(g_deph.real, k, 'same')+1j*np.convolve(g_deph.imag, k, 'same')
        else:
            g_sm = g_deph
        a_eff = g_sm*np.exp(1j*ph)                                  # carrier phase back (bit re-applied in reconstruct)
        resid = x - reconstruct(a_eff)
        A_res = despread_power(resid)
        print("  %5d   %6.1f dB       (|A_resid| = %.5f)" % (W, 20*np.log10(A_orig/max(A_res, 1e-9)), A_res))

if __name__ == "__main__":
    peel()
