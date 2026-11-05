#!/usr/bin/env python3
"""Identify the 8-VSB trellis code from sliced symbols: the 12-way encoder interleave and the parity check
Z0(n) + a1 Z0(n-1) + a2 Z0(n-2) + b0 Z1(n) + b1 Z1(n-1) + b2 Z1(n-2) = 0 (per encoder, GF(2)) that the
decisions satisfy far above chance. Levels -7..+7 = Z2Z1Z0 000..111 (A/53 natural mapping).

    atsc_tcm_id.py --eq /var/tmp/bb1030/atsc/eq2_30_k00_w8.npz
"""
import argparse
import itertools
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atsc_rx import SEG, FIELD  # noqa: E402


def data_symbols(dec, fslot, nseg):
    """[(data segment index within the field, symbols[828])] for whole fields after the first field sync."""
    D = dec.reshape(nseg, SEG)
    out = []
    for fs in range(fslot, nseg - FIELD, FIELD):
        for s in range(1, FIELD):
            out.append((s - 1, D[fs + s, 4:]))
    return out


def encoder_streams(segs, rule):
    """Per encoder 0..11: the symbol sequence in transmission order under an interleave rule."""
    streams = [[] for _ in range(12)]
    for ds, sym in segs:
        j = np.arange(828)
        if rule == "j":
            e = j % 12
        elif rule == "j+4s":
            e = (j + 4 * ds) % 12
        elif rule == "j-4s":
            e = (j - 4 * ds) % 12
        elif rule == "j+8s":
            e = (j + 8 * ds) % 12
        for q in range(12):
            streams[q].append(sym[e == q])
    return [np.concatenate(s) for s in streams]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", required=True)
    ap.add_argument("--fields", type=int, default=12)
    a = ap.parse_args()
    d = np.load(a.eq)
    dec = d["dec"].astype(int)
    fslot = int(d["fslot"])
    nseg = len(dec) // SEG
    segs = data_symbols(dec, fslot, nseg)[: a.fields * (FIELD - 1)]
    for rule in ("j", "j+4s", "j-4s", "j+8s"):
        st = encoder_streams(segs, rule)
        res = []
        for a1, a2, b0, b1, b2 in itertools.product((0, 1), repeat=5):
            if (a1, a2, b0, b1, b2) == (0, 0, 0, 0, 0):
                continue
            ok = []
            for s in st:
                idx = (s + 7) // 2
                z0, z1 = idx & 1, (idx >> 1) & 1
                chk = z0[2:] ^ (a1 * z0[1:-1]) ^ (a2 * z0[:-2]) ^ (b0 * z1[2:]) ^ (b1 * z1[1:-1]) ^ (b2 * z1[:-2])
                ok.append(np.mean(chk == 0))
            res.append((np.mean(ok), (a1, a2, b0, b1, b2), min(ok), max(ok)))
        res.sort(key=lambda x: -x[0])
        print("interleave %-5s: best checks (a1 a2 b0 b1 b2: fraction satisfied, min/max over encoders): %s" % (
            rule, "  ".join("%s:%.3f(%.3f/%.3f)" % ("".join(map(str, r[1])), r[0], r[2], r[3]) for r in res[:4])))


if __name__ == "__main__":
    main()
