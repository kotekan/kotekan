#!/usr/bin/env python3
"""Per-frame input covariances of the 2026-10-30 baseband dump for a channel range, from the
per-residue pulls of bb_extract.py (k<kk>_r<rr>.npy uint8 [163840, nkeep, 2, 64] + _freq.npy).

For every channel in --freq (ascending over all residue files) and every 8192-sample frame
(42 ms, the production integration; 20 per chunk) the RAW covariance of the 128 inputs,
C[f] = sum_t x(t) x(t)^H over the frame (exact in float32: terms <= 98, 8192 of them), with
x decoded as (hi nibble - 8) + i (lo nibble - 8). Whitening is applied downstream from the
diagonal. Output <out>: freq int32 [nch], cov complex64 [nch, 20, 128, 128].

    dtv_cov.py --dir /var/tmp/bb1030/dtv --chunk 0 --freq 2407-3072 --out /var/tmp/bb1030/dtv/cov_k00.npz
"""
import argparse
import glob
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import numpy as np

NT, FRAME = 163840, 8192
LUT = np.empty(256, dtype=np.complex64)
for b in range(256):
    LUT[b] = complex((b >> 4) - 8, (b & 0xF) - 8)


def ranges(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True); ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--freq", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    want = set(ranges(a.freq))
    files = sorted(glob.glob(os.path.join(a.dir, "k%02d_r??.npy" % a.chunk)))
    nfr = NT // FRAME
    freqs, covs = [], []
    t0 = time.time()
    for f in files:
        fr = np.load(f.replace(".npy", "_freq.npy"))
        cols = [i for i, q in enumerate(fr) if int(q) in want]
        if not cols:
            continue
        arr = np.load(f, mmap_mode="r")
        cov = np.zeros((len(cols), nfr, 128, 128), np.complex64)
        for j in range(nfr):
            raw = np.asarray(arr[j * FRAME:(j + 1) * FRAME, cols]).reshape(FRAME, len(cols), 128)
            x = LUT[raw].transpose(1, 0, 2)                         # [nc, T, 128]
            cov[:, j] = np.matmul(x.transpose(0, 2, 1).conj(), x)     # C_ik = sum_t conj(x_i) x_k ... see below
        # convention: cov[c, j, i, k] = sum_t x_i(t) conj(x_k(t)) so that the beam w^H C w uses w as
        # "sum_i conj(w_i) x_i" weights; matmul above gives sum_t conj(x_i) x_k = conj of that -> conjugate
        covs.append(np.conj(cov)); freqs.append(fr[cols])
        print("%s %s: %d channels, %.0f s" % (time.strftime("%H:%M:%S", time.gmtime()), os.path.basename(f), len(cols), time.time() - t0), flush=True)
    freq = np.concatenate(freqs); order = np.argsort(freq)
    cov = np.concatenate(covs)[order]
    np.savez(a.out, freq=freq[order].astype(np.int32), cov=cov, chunk=a.chunk, frame=FRAME)
    print("saved %s: %d channels %d..%d, %d frames" % (a.out, len(freq), freq.min(), freq.max(), nfr))


if __name__ == "__main__":
    main()
