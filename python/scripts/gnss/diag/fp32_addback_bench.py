#!/usr/bin/env python3
"""At what gain error does the fp32 residual destroy the analytic add-back?

The add-back identity  V_full = V_resid + a*<R_P,R_t>  is algebraically EXACT for
any a. But the pipeline is not algebra:

    peel     : resid = v - a*r_P        <-- stored as float2  (FP32)
    despread : V_resid = <resid, r_t>   <-- accumulated in double
    add-back : V_full = V_resid + a*<R_P,R_t>

If |a*r_P| >> |v|, the subtraction is catastrophic cancellation IN FP32: the
small true signal is rounded off against a huge subtracted term, and the exact
double-precision add-back cannot restore bits that no longer exist.

Observed on the L5 first light (2026-07-26): health-line ratio |resid|/|full|
reached ~30000x, i.e. the peel ADDED enormous power. deep_snr collapsed 158-296
-> 2-8 on exactly the sats that engaged. This bench asks whether that ratio is
sufficient to explain the collapse via fp32 alone -- and where the safe limit is,
which is the number a defensive gate should use.

Model: one channel, N hops. v = signal + noise (complex). r_P = unit replica.
True gain a_true = <v,r_P>/<r_P,r_P>. We sweep an ERRONEOUS gain a = k*a_true.
"""
import numpy as np

rng = np.random.default_rng(7)
N = 1000            # hops per record (L1 and L5 are both 1000 at 20 MSPS)
TRIALS = 40

def run(k, snr_amp=0.05):
    """k = gain error factor (1.0 = perfect). Returns recovered/true correlation."""
    errs = []
    for _ in range(TRIALS):
        # unit-ish replica and a weak embedded signal, as in a real record
        r = (rng.standard_normal(N) + 1j*rng.standard_normal(N)).astype(np.complex128)
        r /= np.sqrt(np.mean(np.abs(r)**2))
        noise = (rng.standard_normal(N) + 1j*rng.standard_normal(N)).astype(np.complex128)
        v = noise + snr_amp * r                       # signal buried in noise
        eP = np.vdot(r, r).real                       # <r,r>
        V_full_true = np.vdot(r, v)                   # <v, r>  (reference, float64)
        a_true = V_full_true / eP
        a = k * a_true                                # the (possibly wrong) gain

        # --- pipeline, with the residual forced through FP32 as in the kernel ---
        resid32 = (v - a * r).astype(np.complex64)    # <-- THE LOSSY STEP
        V_resid = np.vdot(r.astype(np.complex64), resid32).astype(np.complex128)
        V_recovered = V_resid + a * eP                # analytic add-back (double)

        errs.append(abs(V_recovered - V_full_true) / abs(V_full_true))
    return float(np.median(errs))

print("gain error k   |a*r|/|v| approx   add-back relative error   verdict")
print("-" * 74)
for k in (1.0, 2.0, 10.0, 100.0, 1e3, 1e4, 3e4, 1e5, 1e6):
    e = run(k)
    # |a*r|/|v| ~ k * |a_true| * |r| / |v| ; with snr_amp=0.05 and unit r/noise
    ratio = k * 0.05
    if   e < 1e-3: verd = "fine"
    elif e < 0.1:  verd = "degraded"
    elif e < 1.0:  verd = "BADLY degraded"
    else:          verd = "DESTROYED"
    print("%12.0f   %14.1f   %22.3e   %s" % (k, ratio, e, verd))

print()
print("Same sweep with a float64 residual (the proposed fix), for contrast:")
def run64(k, snr_amp=0.05):
    errs = []
    for _ in range(TRIALS):
        r = (rng.standard_normal(N) + 1j*rng.standard_normal(N))
        r /= np.sqrt(np.mean(np.abs(r)**2))
        v = (rng.standard_normal(N) + 1j*rng.standard_normal(N)) + snr_amp*r
        eP = np.vdot(r, r).real
        Vt = np.vdot(r, v); a = k * Vt/eP
        resid = v - a*r                                # stays float64
        errs.append(abs((np.vdot(r, resid) + a*eP) - Vt)/abs(Vt))
    return float(np.median(errs))
for k in (1.0, 1e3, 1e4, 3e4, 1e6):
    print("   k=%-9.0f fp64 residual error = %.3e" % (k, run64(k)))
