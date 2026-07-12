#!/usr/bin/env python3
"""GPS L1 C/A coherence ladder on a raw 20 MSPS grab (grab_l1.sh -a 0 style) -- the
offline twin of the live auto-coherence ladder, with IDEAL machinery: polynomial code
model, bit-robust smooth carrier fit, exhaustive bit-phase search, exact per-bit wipe.
If deep SNR builds ~sqrt(T) to 1-2 s here, the signal + LO cohere and any live coherence
retreat is pipeline machinery, not physics. Companion to gnss_direct_cn0.py (which gives
the PRN + Doppler to pass in).

2026-07-12 ground truth (PRN 6, the l1_20msps grab): ladder = 124sigma@0.125s ->
350@1s -> 494@2s, ratios exactly sqrt(T); carrier residual 3.2 deg rms over 26 s after a
freq+rate fit. The live GPS retreat to 0.125 s was the carrier-trim LEAK equilibrium
bias (standing resid = trim*leak/gain, ~1 Hz at the GPS 100 Hz fence) -- see
gps_distributed_broker --carrier-leak.

Usage: python3 gnss_coherence_ladder.py [raw.bin [prn dop_hz]]
"""
import numpy as np
import sys

FS = 20e6
IF = 5e6
FC = 1575.42e6
RAW = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/lwlab/airspy_gps/captures/l1_20msps_2026-07-12.bin"
PRN = int(sys.argv[2]) if len(sys.argv) > 2 else 6
DOP0 = float(sys.argv[3]) if len(sys.argv) > 3 else 2062.5   # from gnss_direct_cn0.py
T0, DUR = 2.0, 26.0            # analysis span (s)
NREC = int(DUR * 1000)         # 1 ms records

# --- C/A code (same generator as gnss_direct_cn0) ---
def ca_code(prn):
    taps = {1:(2,6),2:(3,7),3:(4,8),4:(5,9),5:(1,9),6:(2,10),7:(1,8),8:(2,9),9:(3,10),
            10:(2,3),11:(3,4),12:(5,6),13:(6,7),14:(7,8),15:(8,9),16:(9,10),17:(1,4),
            18:(2,5),19:(3,6),20:(4,7),21:(5,8),22:(6,9),23:(1,3),24:(4,6),25:(5,7),
            26:(6,8),27:(7,9),28:(8,10),29:(1,6),30:(2,7),31:(3,8),32:(4,9)}[prn]
    g1 = np.ones(10, int); g2 = np.ones(10, int)
    out = np.empty(1023, int)
    for i in range(1023):
        out[i] = g1[9] ^ (g2[taps[0]-1] ^ g2[taps[1]-1])
        f1 = g1[2] ^ g1[9]
        f2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = np.roll(g1, 1); g1[0] = f1
        g2 = np.roll(g2, 1); g2[0] = f2
    return 1 - 2 * out

raw = np.memmap(RAW, dtype=np.int16, mode="r")
code = ca_code(PRN)
n1ms = int(FS * 1e-3)                      # 20000 samples / record
rep = code[(np.arange(n1ms) * 1.023e6 / FS).astype(np.int64) % 1023].astype(np.float32)
REPF = np.conj(np.fft.fft(rep))

# --- 1. code-phase anchors: FFT-correlation peak every 2 s, poly2 model cp(t) ---
tt1 = np.arange(n1ms) / FS
anchors = []
for ta in np.arange(T0, T0 + DUR + 0.1, 2.0):
    o = int(ta * FS)
    x = raw[o:o+n1ms].astype(np.float32)
    xb = x * np.exp(-2j*np.pi*(IF+DOP0)*(tt1 + o/FS)).astype(np.complex64)
    c = np.abs(np.fft.ifft(np.fft.fft(xb) * REPF))
    pk = int(np.argmax(c))
    # parabolic sub-sample refine
    l, r = c[(pk-1) % n1ms], c[(pk+1) % n1ms]
    d = 0.5*(l - r)/(l + r - 2*c[pk]) if (l + r - 2*c[pk]) != 0 else 0.0
    anchors.append((ta, (pk + d)))
ta = np.array([a[0] for a in anchors])
pkpos = np.unwrap(np.array([a[1] for a in anchors]) * 2*np.pi/n1ms) * n1ms/(2*np.pi)
cp_poly = np.polyfit(ta, pkpos, 2)          # peak sample offset within a record vs t
resid = pkpos - np.polyval(cp_poly, ta)
print("code model: quad fit over %d anchors, resid rms %.2f samples (%.3f chips)"
      % (len(ta), resid.std(), resid.std()/(FS/1.023e6)))

# --- 2. despread every 1 ms record on the code model, fixed carrier DOP0 ---
A = np.empty(NREC, np.complex64)
for k in range(NREC):
    t = T0 + k*1e-3
    o = int(round(t * FS))
    x = raw[o:o+n1ms].astype(np.float32)
    xb = x * np.exp(-2j*np.pi*(IF+DOP0)*(tt1 + o/FS)).astype(np.complex64)
    lag = np.polyval(cp_poly, t)             # replica shifted to the peak
    ilag = int(np.floor(lag)) % n1ms
    fr = lag - np.floor(lag)
    r0 = np.roll(rep, ilag); r1 = np.roll(rep, ilag+1)
    A[k] = np.dot(xb, ((1-fr)*r0 + fr*r1).astype(np.float32))
# noise scale: same despread with a wrong (quiet) PRN's code -- PRN17 is below horizon? use
# a code orthogonal at this lag: shift the true replica by 200 chips (off-peak = noise).
N = np.empty(NREC, np.complex64)
roff = np.roll(rep, int(200*(FS/1.023e6)))
for k in range(0, NREC, 7):                  # subsample for speed
    t = T0 + k*1e-3
    o = int(round(t * FS))
    x = raw[o:o+n1ms].astype(np.float32)
    xb = x * np.exp(-2j*np.pi*(IF+DOP0)*(tt1 + o/FS)).astype(np.complex64)
    N[k] = np.dot(xb, roff)
sig1 = np.std(N[::7].real)                   # per-record per-quadrature noise sigma
print("per-record |A| ~ %.0f, noise sigma %.0f -> 1 ms amp SNR ~ %.1f"
      % (np.abs(A).mean(), sig1, np.abs(A).mean()/sig1))

# --- 3. bit-robust smooth carrier: unwrap arg(A^2)/2 smoothed, poly fit ---
z = A.astype(np.complex128)**2
# smooth over 10 ms to beat the phase noise before unwrapping
ker = np.ones(10)/10
zs = np.convolve(z, ker, mode="same")
ph2 = np.unwrap(np.angle(zs))
tk = T0 + np.arange(NREC)*1e-3
for deg in (1, 2, 3):
    cpoly = np.polyfit(tk, ph2, deg)
    r = ph2 - np.polyval(cpoly, tk)
    print("carrier fit deg %d: residual phase rms %.3f rad (%.1f deg), slope@mid %+.2f Hz, "
          "rate %+.3f Hz/s"
          % (deg, r.std()/2, np.degrees(r.std()/2),
             np.polyval(np.polyder(cpoly), tk.mean())/(4*np.pi),
             (cpoly[deg-2]*2/(4*np.pi) if deg >= 2 else 0.0)))
cpoly = np.polyfit(tk, ph2, 3)
Ac = A * np.exp(-1j*np.polyval(cpoly, tk)/2)

# --- 4. bit sync: 20 ms blocks, best of 20 alignments; wipe the signs ---
best = (-1, 0)
for ph in range(20):
    nblk = (NREC - ph)//20
    b = Ac[ph:ph+nblk*20].reshape(nblk, 20).sum(axis=1)
    e = np.sum(np.abs(b)**2)
    if e > best[0]:
        best = (e, ph)
ph = best[1]
nblk = (NREC - ph)//20
blocks = Ac[ph:ph+nblk*20].reshape(nblk, 20)
bsum = blocks.sum(axis=1)
rot = bsum.sum();  rot /= abs(rot)
signs = np.sign((bsum*np.conj(rot)).real)
flips = int(np.sum(signs[1:] != signs[:-1]))
print("bit sync: phase %d/20, %d blocks, %d bit flips (%.0f%%)"
      % (ph, nblk, flips, 100*flips/max(1, nblk-1)))
wiped = (blocks * signs[:, None]).ravel()    # bit-wiped 1 ms records, coherent

# --- 5. the ladder: deep amplitude SNR vs coherent window ---
print("\ncoherence ladder (ideal wipe + smooth carrier), per-window amp / (sigma*sqrt(n)):")
print("  T(s)    n_win   median SNR   xSNR(0.125s)   sqrt-T pred")
base = None
for T in (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0):
    n = int(T*1000)
    nw = len(wiped)//n
    w = wiped[:nw*n].reshape(nw, n).sum(axis=1)
    snr = np.abs(w)/(sig1*np.sqrt(n))
    med = np.median(snr)
    if T == 0.125: base = med
    print("  %-7g %-7d %-12.1f %-14s %s"
          % (T, nw, med,
             "%.2f" % (med/base) if base else "--",
             "%.2f" % np.sqrt(T/0.125)))
# phase-residual structure across 1 s windows (what would break 1 s coherence)
ph1 = np.angle(np.convolve(wiped, np.ones(50)/50, mode="same"))
print("\nphase drift within 1 s windows (50 ms smoothed, deg rms per window):")
nw = len(wiped)//1000
drift = [np.degrees(np.unwrap(ph1[i*1000:(i+1)*1000:50]).ptp()) for i in range(nw)]
print("  median peak-to-peak %.0f deg, worst %.0f deg" % (np.median(drift), np.max(drift)))
