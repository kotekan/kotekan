#!/usr/bin/env python3
"""One or more baselines over the whole band from recv1's N2 files, whitened per channel and bin:
R_ab = V_ab / sqrt(V_aa V_bb), so a channel a transmitter dominated and a clean one share a scale. Every 16-channel
chunk of a file is read once (the files are chunked 16 channels x all products x all bins), files in parallel.

    vis_baseline.py --acq DIR --from 1015 --to 1045 [--date 20261106] --pairs 0-11,64-75 --out bl.npz [--procs 10]
"""
import argparse
import glob
import os
import time
from multiprocessing import Pool

import h5py
import numpy as np

PAIRS = None


def one_file(path):
    t0 = time.time()
    with h5py.File(path, "r") as f:
        prod = f["index_map/prod"][:]
        ia, ib = prod["input_a"].astype(int), prod["input_b"].astype(int)
        pid = {(int(x), int(y)): n for n, (x, y) in enumerate(zip(ia, ib))}
        want = sorted({pid[p] for a, b in PAIRS for p in ((a, b), (a, a), (b, b))})
        col = {p: want.index(p) for p in want}
        nf, nb = f["vis"].shape[0], f["vis"].shape[2]
        R = np.full((len(PAIRS), nb, nf), np.nan + 0j, np.complex64)
        for c0 in range(0, nf, 16):
            c1 = min(nf, c0 + 16)
            v = f["vis"][c0:c1, want, :]  # [C, P, nb]
            fl = f["flags"][c0:c1] > 0  # [C, 128, nb]
            for k, (a, b) in enumerate(PAIRS):
                x = v[:, col[pid[(a, b)]], :]
                pa = np.real(v[:, col[pid[(a, a)]], :])
                pb = np.real(v[:, col[pid[(b, b)]], :])
                ok = fl[:, a, :] & fl[:, b, :] & (pa > 0) & (pb > 0)
                r = np.where(ok, x / np.sqrt(np.where(ok, pa * pb, 1.0)), np.nan)
                R[k, :, c0:c1] = r.T
        t = f["bin_t_inst_ns"][:]
        k0 = round(f["index_map/freq"][0]["centre"] / 0.1953125)  # freq_id of index 0
    return dict(path=path, t=t, R=R, k0=k0, secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--acq", required=True)
    ap.add_argument("--from", dest="t0", required=True, help="HHMM of the first file")
    ap.add_argument("--to", dest="t1", required=True, help="HHMM of the last file")
    ap.add_argument(
        "--date", help="YYYYMMDD of the files (an acquisition can span nights)"
    )
    ap.add_argument(
        "--pairs",
        default="0-11,64-75",
        help="input pairs a-b (correlator order pol*64 + dish)",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    global PAIRS
    PAIRS = [tuple(sorted(int(v) for v in p.split("-"))) for p in a.pairs.split(",")]
    files = [
        p
        for p in sorted(glob.glob(os.path.join(a.acq, "vis_*.h5")))
        if a.t0 <= os.path.basename(p).split("_")[3][:4] <= a.t1
        and (a.date is None or os.path.basename(p).split("_")[2].startswith(a.date))
    ]
    print("%d files, pairs %s" % (len(files), PAIRS), flush=True)
    out = []
    with Pool(min(a.procs, len(files))) as pool:
        for r in pool.imap(one_file, files):
            print("%s: %.0f s" % (os.path.basename(r["path"]), r["secs"]), flush=True)
            out.append(r)
    t = np.concatenate([r["t"] for r in out])
    o = np.argsort(t)
    R = np.concatenate([r["R"] for r in out], axis=1)[:, o, :]
    nf = R.shape[2]
    np.savez(
        a.out,
        t_inst_ns=t[o],
        R=R,
        pairs=np.array(PAIRS),
        fid=np.arange(nf) + out[0]["k0"],
        mhz=(np.arange(nf) + out[0]["k0"]) * 0.1953125,
    )
    print("saved %s: %d bins x %d channels" % (a.out, R.shape[1], nf))


if __name__ == "__main__":
    main()
