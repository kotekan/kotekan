#!/usr/bin/env python3
"""Does chip_gather's float boundary reformulation stay exact at ks=1 (L5)?

cudaGnssDespreadKernel.cu::chip_gather replaces the exact double boundary
    khi(d) = floor(base + d*inv_cps),  base = phi*inv_cps
with an integer/fraction split (comment (c)):
    inv_cps = ks + kf   (ks = (int)inv_cps, kf = frac)
    khi(d) = d*ks + floor_f32(f),  f accumulated in FLOAT as f += kf
The kernel comment claims verification "over 16e6 (phi,d) pairs across 5
geometries". L5 is the FIRST chain with inv_cps < 2, i.e. ks = 1, kf = 0.955 --
the largest kf of any band, and the most float accumulation steps per hop
(n_chips ~ Lf/inv_cps is LARGEST when inv_cps is smallest).

A boundary disagreement moves one prototype tap between adjacent chips, which
perturbs the replica. That alone is harmless-ish for tracking -- BUT the peel and
the despread must produce the BIT-IDENTICAL replica for the analytic add-back
    V_full = V_resid + a*<R_P,R_t>
to hold. They call the same function with the same args, so a deterministic
disagreement is shared and cancels... unless the two paths differ in n_chips or
Lf, or the error is large enough that the *gain* estimated on one grid does not
match the replica subtracted on the other.

This bench answers ONE question exactly: how often, and by how much, does the
float form disagree with the double form, as a function of inv_cps?
"""
import numpy as np

def khi_exact(base, d, inv_cps):
    """Reference: float64 throughout."""
    return np.floor(base + d * inv_cps).astype(np.int64)

def khi_fast(base, d_max, inv_cps):
    """Mirror of the kernel: ks integer part, kf accumulated in float32."""
    ks = int(inv_cps)
    kf = np.float32(inv_cps - ks)
    f = np.float32(base)                       # (c) f = base + d*kf, float32
    khi_base = 0
    out = np.empty(d_max, dtype=np.int64)
    for d in range(d_max):
        out[d] = khi_base + int(np.floor(f))
        f = np.float32(f + kf)                 # one float add per chip
        khi_base += ks
    return out

GEOMETRIES = [
    ("L1 C/A  @20MSPS", 20e6 / 1.023e6),
    ("L2C CM  @5MSPS",  5e6 / 1.023e6),
    ("E1C BOC @20MSPS", 20e6 / 2.046e6),
    ("L5 Q5   @20MSPS", 20e6 / 10.23e6),   # <-- ks = 1
]
Lf = 20
N_PHI = 200000
rng = np.random.default_rng(12345)

print("%-18s %8s %5s %7s  %7s  %10s  %10s" %
      ("geometry", "inv_cps", "ks", "kf", "n_chips", "disagree", "rate"))
print("-" * 82)
worst = None
for name, inv_cps in GEOMETRIES:
    ks = int(inv_cps)
    kf = inv_cps - ks
    n_chips = int(np.ceil(Lf / inv_cps)) + 2       # as the host sizes it, +guard
    phis = rng.random(N_PHI)                        # phi in [0,1)
    bad = 0
    total = 0
    maxdiff = 0
    for phi in phis:
        base = phi * inv_cps
        ex = khi_exact(base, np.arange(n_chips), inv_cps)
        fa = khi_fast(base, n_chips, inv_cps)
        d = np.abs(ex - fa)
        nb = int((d != 0).sum())
        bad += nb
        total += n_chips
        if nb:
            maxdiff = max(maxdiff, int(d.max()))
    rate = bad / total
    print("%-18s %8.4f %5d %7.4f  %7d  %10d  %10.2e%s" %
          (name, inv_cps, ks, kf, n_chips, bad, rate,
           "   <-- L5" if "L5" in name else ""))
    if worst is None or rate > worst[1]:
        worst = (name, rate, maxdiff)

print()
print("worst geometry: %s  rate=%.2e  max|khi error|=%d chips" % worst)
print()
print("INTERPRETATION: a nonzero rate moves ONE prototype tap between adjacent")
print("chips. It is only FATAL if the peel and the despread disagree with each")
print("other -- they call chip_gather identically, so a shared error cancels in")
print("the add-back. This bench bounds the error; it does NOT by itself prove")
print("the two paths agree. That needs the n_chips/Lf comparison below.")
