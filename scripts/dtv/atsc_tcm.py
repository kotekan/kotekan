#!/usr/bin/env python3
"""Viterbi decoding of the 8-VSB trellis code (12 interleaved 4-state Ungerboeck encoders, identified from
the data by atsc_tcm_id.py: Z0(n) = Z0(n-2) xor Z1(n-1) per encoder; encoder of data symbol j in data
segment ds = (j + 4 ds) mod 12; levels -7..+7 = Z2Z1Z0 000..111; Z2 uncoded = nearest point of the subset).
Writes the decoded symbol stream (syncs filled with their known values) for regeneration.

    atsc_tcm.py --eq /var/tmp/bb1030/atsc/eq2_30_k00_w8.npz --out /var/tmp/bb1030/atsc/tcm30_k00.npz
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atsc_rx import SEG, FIELD  # noqa: E402
from atsc_eq import pn511  # noqa: E402

LEV = np.arange(8) * 2 - 7  # index Z2Z1Z0 -> level
# state = 2*z0_cur + z0_prev ; input z1 -> subset (z1, z0_cur), next = 2*(z0_prev ^ z1) + z0_cur
NEXT = np.zeros((4, 2), int)
SUB = np.zeros((4, 2), int)  # subset index z1z0 = 2*z1 + z0_cur
for s in range(4):
    z0c, z0p = s >> 1, s & 1
    for z1 in (0, 1):
        NEXT[s, z1] = 2 * (z0p ^ z1) + z0c
        SUB[s, z1] = 2 * z1 + z0c
# incoming pairs for each next state
INC = [
    [(s, z1) for s in range(4) for z1 in (0, 1) if NEXT[s, z1] == ns] for ns in range(4)
]


def viterbi(R):
    """R [n_enc, T] received levels per encoder. Returns decoded level indices [n_enc, T]."""
    ne, T = R.shape
    # branch metrics per subset z1z0 (4): min over z2 of (r - level)^2, and the z2 choice
    lev = LEV.reshape(2, 4)  # [z2, z1z0]
    d = (R[:, :, None, None] - lev[None, None, :, :]) ** 2  # [ne, T, z2, sub]
    bm = d.min(axis=2)
    z2c = d.argmin(axis=2).astype(np.int8)
    M = np.zeros((ne, 4))
    surv = np.zeros((T, ne, 4), np.int8)  # index into INC[ns]
    inc_s = np.array([[p[0] for p in INC[ns]] for ns in range(4)])  # [4, 2]
    inc_z1 = np.array([[p[1] for p in INC[ns]] for ns in range(4)])
    inc_sub = SUB[inc_s, inc_z1]  # [4, 2]
    for t in range(T):
        cand = M[:, inc_s] + bm[:, t][:, inc_sub]  # [ne, 4(ns), 2]
        ch = cand.argmin(axis=2)
        surv[t] = ch
        M = np.take_along_axis(cand, ch[:, :, None], axis=2)[:, :, 0]
        M -= M.min(axis=1, keepdims=True)
    out = np.zeros((ne, T), int)
    st = M.argmin(axis=1)
    e = np.arange(ne)
    for t in range(T - 1, -1, -1):
        ch = surv[t, e, st]
        ps = inc_s[st, ch]
        z1 = inc_z1[st, ch]
        sub = SUB[ps, z1]
        out[:, t] = 4 * z2c[e, t, sub] + sub
        st = ps
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    d = np.load(a.eq)
    ar = d["a"].astype(float)
    fslot = int(d["fslot"])
    nseg = len(ar) // SEG
    A = ar.reshape(nseg, SEG)
    segs = [s for s in range(nseg) if (s - fslot) % FIELD != 0]
    ds = np.array([(s - fslot - 1) % FIELD for s in segs])
    j = np.arange(828)
    # per encoder: positions (segment, column) in transmission order
    pos = [[] for _ in range(12)]
    for s, dsi in zip(segs, ds):
        e = (j + 4 * dsi) % 12
        for q in range(12):
            pos[q].append(np.stack([np.full(69, s), 4 + j[e == q]], axis=1))
    pos = [np.concatenate(p) for p in pos]
    T = min(len(p) for p in pos)
    R = np.array([A[p[:T, 0], p[:T, 1]] for p in pos])
    hard = np.argmin(np.abs(R[:, :, None] - LEV[None, None, :]), axis=2)
    dec = viterbi(R)
    changed = np.mean(dec != hard)
    # the decoded sequence must satisfy the code; the hard one does not
    z0, z1 = dec & 1, (dec >> 1) & 1
    chk = np.mean((z0[:, 2:] ^ z0[:, :-2] ^ z1[:, 1:-1]) == 0)
    hz0, hz1 = hard & 1, (hard >> 1) & 1
    hchk = np.mean((hz0[:, 2:] ^ hz0[:, :-2] ^ hz1[:, 1:-1]) == 0)
    err = R - LEV[dec]
    mer = 10 * np.log10(21.0 / np.mean(err ** 2))
    print(
        "Viterbi over %d symbols per encoder x 12: %.4f of the hard decisions changed; parity satisfied %.4f (hard %.4f); "
        "MER against the decoded symbols %.1f dB (%.0f s)"
        % (T, changed, chk, hchk, mer, time.time() - t0)
    )
    # how often the decoded symbol is not the nearest level (a correction), split by distance
    full = np.zeros((nseg, SEG), int)
    full[:, :4] = np.array([5, -5, -5, 5])
    for q in range(12):
        full[pos[q][:T, 0], pos[q][:T, 1]] = LEV[dec[q]]
    known = np.zeros((nseg, SEG), bool)
    known[:, :4] = True
    for fs in range(fslot % FIELD, nseg, FIELD):
        full[fs, 4:515] = pn511()
        known[fs, 4:515] = True
    filled = np.zeros((nseg, SEG), bool)
    for q in range(12):
        filled[pos[q][:T, 0], pos[q][:T, 1]] = True
    print(
        "decoded stream: %d symbols, %.4f trellis-decoded, %.4f known, %.4f left (field-sync tail, encoder ends)"
        % (full.size, filled.mean(), known.mean(), 1 - filled.mean() - known.mean())
    )
    np.savez(
        a.out,
        sym=full.reshape(-1).astype(np.int8),
        filled=filled.reshape(-1),
        known=known.reshape(-1),
        fslot=fslot,
    )
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
