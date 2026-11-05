#!/usr/bin/env python3
"""Synthetic all-subtracting DTV state (random w, small random g) for GPU timing of a lag grid / source count.
    dtv_synth_state.py --freq 2512,2720,2912 --nsrc 3 --lag-min -16 --n-lag 64 --out st.bin"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/gnss/dtv-wt/scripts/gnss"))
import dtv_state as ds

ap = argparse.ArgumentParser()
ap.add_argument("--freq", required=True)
ap.add_argument("--nsrc", type=int, required=True)
ap.add_argument("--lag-min", type=int, required=True)
ap.add_argument("--n-lag", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rng = np.random.default_rng(1)
NI = ds.NI
recs = {}
for f in (int(v) for v in a.freq.split(",")):
    w = (
        rng.standard_normal((a.nsrc, NI)) + 1j * rng.standard_normal((a.nsrc, NI))
    ).astype(np.complex64) / np.sqrt(2 * NI)
    g = (
        1e-3
        * (
            rng.standard_normal((a.nsrc, a.n_lag, NI))
            + 1j * rng.standard_normal((a.nsrc, a.n_lag, NI))
        )
    ).astype(np.complex64)
    recs[f] = dict(n_set=a.nsrc, subtract=1, w=w, g=g)
ds.write_state(a.out, dict(n_lag=a.n_lag, lag_min=a.lag_min, n_src=a.nsrc, recs=recs))
print("wrote", a.out)
