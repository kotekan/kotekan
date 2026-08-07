#!/usr/bin/env python3
"""Is fleet_coherent partition-invariant? Same total aperture, split across different node counts.

KV's claim (2026-08-07): each frequency is processed independently, so whether the broker
gathers B bands from 2 nodes or B bands from B nodes, the answer should depend on the number of
bands and the noise per band -- not on how many nodes they are spread over.

Driven through the REAL fleet_coherent by monkeypatching its HTTP fetch, so this tests the
shipped estimator and not a model of it.

Record model, matching what GnssCoherentCombiner actually emits: the header correlation slots
are NORMALIZED by replica energy (rec[4] = acc_ar * inv), so an instance holding m bands has
    A_i(t) = a * exp(i*(phi(t) + theta_i)) + CN(0, sigma^2 / m)      energy_i = m
-- amplitude independent of m, noise falling as 1/sqrt(m), MRC weight = energy = m. Total fleet
SNR is then a*sqrt(sum m_i)/sigma, identical for every partition of the same B.
"""
import importlib.util
import math
import random
import sys

spec = importlib.util.spec_from_file_location(
    "brk", "/home/kvand/gnss/kotekan/python/scripts/gnss/gps_distributed_broker.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

N_REC = 128
HOP0 = 1000000
SKY_RMS = 0.75          # rad, the measured common per-record sky phase
PRN = 7


def make_fleet(n_inst, bands_each, s1, rng):
    """s1 = per-BAND per-record SNR (a/sigma). Returns {url: records}."""
    phi = [rng.gauss(0.0, SKY_RMS) for _ in range(N_REC)]
    theta = [rng.uniform(-math.pi, math.pi) for _ in range(n_inst)]
    out = {}
    for i in range(n_inst):
        mb = bands_each
        sig_n = 1.0 / math.sqrt(mb)          # sigma/sqrt(m) with a = 1, sigma = 1/s1
        recs = []
        for t in range(N_REC):
            a = complex(math.cos(phi[t] + theta[i]), math.sin(phi[t] + theta[i]))
            nz = complex(rng.gauss(0.0, sig_n / s1 / math.sqrt(2)),
                         rng.gauss(0.0, sig_n / s1 / math.sqrt(2)))
            v = a + nz
            recs.append([HOP0 + t * 2048, v.real, v.imag, float(mb)])
        out["http://fake%d" % i] = [{"prn": PRN, "records": recs}]
    return out


def run(n_inst, bands_each, s1, seed):
    rng = random.Random(seed)
    store = make_fleet(n_inst, bands_each, s1, rng)
    m._get = lambda url: store[url.rsplit("/", 1)[0]]
    r = m.fleet_coherent(list(store), min_instances=2, min_records=32,
                         prns={PRN}, log=None, floor_margin=3.0, seed=seed)
    return r.get(PRN)


print("Same total aperture B = 12 bands, split three ways. 128 records, sky phase 0.75 rad rms.")
print("Per-BAND per-record SNR s1 held fixed, so every row has the same total information.\n")
for s1 in (3.0, 1.0, 0.5, 0.25):
    print("s1 = %.2f per band per record   (fleet total SNR/record = %.2f)"
          % (s1, s1 * math.sqrt(12)))
    print("  %-22s %10s %10s %9s %9s" % ("partition", "fleet deep", "best inst", "coh_frac", "align"))
    for n_inst, be in ((2, 6), (4, 3), (6, 2), (12, 1)):
        # average a few seeds so we compare estimators, not one draw of the noise
        ds, bi, cf, al = [], [], [], []
        for k in range(12):
            v = run(n_inst, be, s1, 1234 + 97 * k)
            if not v:
                continue
            ds.append(v["deep_snr"]); bi.append(v["best_inst_snr"])
            cf.append(v["coh_frac"]); al.append(v["align"])
        if not ds:
            print("  %-22s  (no result)" % ("%d inst x %d band" % (n_inst, be)))
            continue
        f = lambda z: sum(z) / len(z)
        print("  %-22s %10.1f %10.1f %9.3f %9.3f"
              % ("%d inst x %d band" % (n_inst, be), f(ds), f(bi), f(cf), f(al)))
    print()
