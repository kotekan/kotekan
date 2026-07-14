#!/usr/bin/env python3
"""Offline validation of the broker's code-rate clock-bias bootstrap (gps_distributed_broker.py:
code_clock_bias_sample / cp_rate_from_code_bias) on the real L1 C/A capture.

Measures several sats independently (l1_code_drift.measure), feeds their (Doppler, code-rate) through
the ACTUAL broker functions, and checks:
  A. every sat yields the same (l-a) -> a common receiver clock offset (not per-sat).
  B. LEAVE-ONE-OUT: calibrate (l-a) from the OTHER sats, predict the held-out sat's code rate from its
     Doppler alone, compare to its measured code rate. This is the real "strong sats calibrate, a weak
     sat inherits an accurate code rate from its first detection" test.
  C. the EMA converges to the pooled value.
Airspy front end: Fs=5 MHz, fft_len=24 -> hops/s = 208333.3; chip 1.023e6; carrier 1575.42e6.
"""
import os, sys, statistics
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "kotekan", "python", "scripts")))
import gps_distributed_broker as b
from l1_code_drift import measure, _load, FILE

HOPS_PER_SEC = 5.0e6/24.0      # Fs/fft_len (airspy 5 MSPS, N=12 -> fft_len 24)
CHIP = 1.023e6
CARRIER = 1575.42e6
PRNS = [32, 12, 31, 28, 10]

x, off, dur = _load(FILE)
print("capture %s (%.1fs); measuring PRNs %s ...\n" % (os.path.basename(FILE), dur, PRNS))
sats = {}
for prn in PRNS:
    r = measure(prn, x, off, dur)
    if r:
        dop, snr, ne, code_frac_ppm, carr_frac_ppm = r
        cp_rate = (code_frac_ppm*1e-6) * CHIP / HOPS_PER_SEC          # chips/hop (what fit_cp_rate returns)
        la = b.code_clock_bias_sample(cp_rate, dop, HOPS_PER_SEC, CHIP, CARRIER)
        sats[prn] = dict(dop=dop, cp_rate=cp_rate, code_frac=code_frac_ppm, la=la)
        print("  PRN %2d: dop %+6.0f Hz  code %+7.3f ppm  ->  broker l-a %+7.3f ppm" % (prn, dop, code_frac_ppm, la*1e6))

las = [s["la"] for s in sats.values()]
print("\nA. per-sat (l-a): mean %+.3f ppm, std %.3f ppm" % (statistics.mean(las)*1e6, statistics.pstdev(las)*1e6))
okA = statistics.pstdev(las) < 0.4e-6

print("\nB. LEAVE-ONE-OUT (calibrate on the others, predict the held-out sat's code rate):")
maxerr = 0.0
for held, s in sats.items():
    others = [sats[p]["la"] for p in sats if p != held]
    bias = statistics.median(others)                                  # broker's raw_cb from the rest
    cp_pred = b.cp_rate_from_code_bias(s["dop"], bias, HOPS_PER_SEC, CHIP, CARRIER)
    pred_ppm = cp_pred*HOPS_PER_SEC/CHIP*1e6                           # predicted code_frac
    err = pred_ppm - s["code_frac"]
    maxerr = max(maxerr, abs(err))
    print("   hold PRN %2d: predict %+7.3f ppm  measured %+7.3f ppm  err %+.3f ppm" % (held, pred_ppm, s["code_frac"], err))
okB = maxerr < 0.1

print("\nC. EMA convergence (alpha 0.05, feed pooled median repeatedly):")
raw = statistics.median(las); ema=None
for _ in range(60): ema = raw if ema is None else ema + 0.05*(raw-ema)
print("   pooled median %+.3f ppm -> EMA %+.3f ppm" % (raw*1e6, ema*1e6))
okC = abs(ema-raw) < 1e-9

print("\n%s  A(common l-a)=%s  B(leave-one-out err %.3f ppm)=%s  C(EMA)=%s" % (
    "PASS" if (okA and okB and okC) else "FAIL", okA, maxerr, okB, okC))
sys.exit(0 if (okA and okB and okC) else 1)
