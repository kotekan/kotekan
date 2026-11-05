#!/usr/bin/env python3
"""Per 10 s bin, for selected 16-channel chunks of the recv1 subset N2 files: the input-whitened
covariance's eigenvalues in noise units, the Tau A template excess, and a few long-baseline visibilities.

Channel index k in the files is 300 MHz + k x 0.1953125 MHz, so freq_id = k + 1536 and the files' 16-channel
chunks start on freq_ids that are multiples of 16. Whitening: R_ik = V_ik / sqrt(V_ii V_kk) over the inputs
the file's flags keep. Noise n = mean of eigenvalues 20..m-8 (m kept inputs). Template (dtv_modes.py
predict_p2): the projector onto the top-2 eigenvectors of the CLEAN chunks' channels, carried to each channel by
the per-input phase slope across adjacent clean channels (the channel itself excluded); excess =
(Tr(Q^H R Q) / 2 - Tr(R) / m) / n, Tau A's power in noise units (the dump's clean value was ~4.7).

    vis_dtv_extract.py --acq DIR --from 0900 --to 1230 --out /var/tmp/kvand-vis/dtv_20261103 [--procs 8]
"""
import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import h5py  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.expanduser("~/gnss/bb1030"))
from dtv_modes import predict_p2, top_vecs  # noqa: E402

CLEAN = [2480, 2544, 2688, 2752, 2848, 2864, 2976]
CLEANED = [2720, 2736, 2896, 2912]  # chunks holding the DTV stage's channels (2714-2744, 2898-2928)
UNCLEANED = [2512, 2944]
CHUNKS = sorted(CLEAN + CLEANED + UNCLEANED)
K0 = 1536
PARTS = None


def file_time(path):
    b = os.path.basename(path).split("_")
    d, t = b[2][:8], b[3]
    return "%s %s:%s:%s" % (d, t[:2], t[2:4], t[4:6])


MIN_KEPT = 64  # a file in which fewer inputs survive every bin's flags (a recv1 restart) is skipped
TEMPLATE = True


def one_file(path):
    T0 = time.time()
    part = os.path.join(PARTS, os.path.basename(path) + ".npz") if PARTS else None
    if part and os.path.exists(part):
        d = np.load(part, allow_pickle=True)
        return {k: d[k].item() if d[k].shape == () else d[k] for k in d.files}
    with h5py.File(path, "r") as f:
        prod = f["index_map/prod"][:]
        a, b = prod["input_a"].astype(int), prod["input_b"].astype(int)
        pos = f["index_map/feed_pos_disp_m"][:]
        pol = f["index_map/pol"][:]
        tins = f["bin_t_inst_ns"][:]
        nb = len(tins)
        fids, vis, flg = [], [], []
        for c0 in CHUNKS:
            k = c0 - K0
            vis.append(f["vis"][k : k + 16])  # [16, 8256, nb]
            flg.append(f["flags"][k : k + 16])  # [16, 128, nb]
            fids += list(range(c0, c0 + 16))
    vis = np.concatenate(vis)  # [C, 8256, nb]
    flg = np.concatenate(flg)
    fids = np.array(fids)
    C = len(fids)
    keep = (flg > 0).all(axis=2).all(axis=0)  # inputs kept in every channel and bin of this file
    m = int(keep.sum())
    if m < MIN_KEPT:
        return dict(path=path, start=file_time(path), kept=m, skipped=True, secs=time.time() - T0)
    clean = np.isin(fids // 16 * 16, CLEAN)
    # three long baselines between kept inputs of one polarisation, longest along the first coordinate
    cand = [(abs(pos[i, 0] - pos[j, 0]), i, j) for i in range(128) for j in range(i + 1, 128)
            if keep[i] and keep[j] and pol[i] == pol[j] and abs(pos[i, 1] - pos[j, 1]) < 1.0]
    cand.sort(reverse=True)
    bl = [(i, j) for _, i, j in cand[:3]]
    pidx = {(int(x), int(y)): n for n, (x, y) in enumerate(zip(a, b))}
    lam = np.zeros((C, 8, nb), np.float32)
    noise = np.zeros((C, nb), np.float32)
    excess = np.full((C, nb), np.nan, np.float32)
    fr = np.zeros((C, len(bl), nb), np.complex64)
    for t in range(nb):
        Rs = np.zeros((C, 128, 128), np.complex128)
        for c in range(C):
            V = np.zeros((128, 128), np.complex128)
            V[a, b] = vis[c, :, t]
            V[b, a] = np.conj(vis[c, :, t])
            d = np.real(np.diag(V)).copy()
            d[~keep] = np.inf
            w = 1.0 / np.sqrt(np.maximum(d, 1e-12))
            w[~keep] = 0.0
            R = V * w[:, None] * w[None, :]
            Rs[c] = R
            for n_, (i, j) in enumerate(bl):
                fr[c, n_, t] = vis[c, pidx[(i, j)], t] * w[i] * w[j]
        ev_all = np.linalg.eigvalsh(Rs)[:, ::-1]  # [C, 128] descending (zeros for dropped inputs at the end)
        n = ev_all[:, 20 : m - 8].mean(axis=1)
        noise[:, t] = n
        lam[:, :, t] = ev_all[:, :8] / n[:, None]
        if not TEMPLATE:
            continue
        # template from the clean channels' top-2 eigenvectors
        P2 = np.zeros((C, 128, 128), np.complex128)
        for c in np.nonzero(clean)[0]:
            _, Vt = top_vecs(Rs[c], 2)
            P2[c] = Vt @ Vt.conj().T
        for c in range(C):
            Pp = predict_p2(P2, fids, clean, fids[c], 96, 1)
            if Pp is None:
                continue
            _, Q = top_vecs(Pp, 2)
            R = Rs[c]
            excess[c, t] = (np.real(np.trace(Q.conj().T @ R @ Q)) / 2 - np.real(np.trace(R)) / m) / n[c]
    r = dict(path=path, start=file_time(path), t_inst_ns=tins, fids=fids, lam=lam, noise=noise, excess=excess,
             fringe=fr, baselines=np.array(bl).reshape(-1, 2), kept=m, skipped=False, secs=time.time() - T0)
    if part:
        np.savez(part, **r)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acq", required=True)
    ap.add_argument("--from", dest="t0", default="0900")
    ap.add_argument("--to", dest="t1", default="1230")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--date", help="YYYYMMDD of the files to take (an acquisition can span nights)")
    ap.add_argument("--no-template", action="store_true", help="eigenvalues only: skip the Tau A template excess")
    a = ap.parse_args()
    global PARTS, TEMPLATE
    PARTS = a.out + "_parts"
    os.makedirs(PARTS, exist_ok=True)
    TEMPLATE = not a.no_template
    files = [p for p in sorted(glob.glob(os.path.join(a.acq, "vis_*.h5")))
             if a.t0 <= os.path.basename(p).split("_")[3][:4] <= a.t1
             and (a.date is None or os.path.basename(p).split("_")[2].startswith(a.date))]
    print("%d files %s .. %s" % (len(files), os.path.basename(files[0]), os.path.basename(files[-1])), flush=True)
    res = []
    with Pool(a.procs) as pool:
        for r in pool.imap(one_file, files):
            if r["skipped"]:
                print("%s: %d kept inputs, SKIPPED (fewer than %d)" % (r["start"], r["kept"], MIN_KEPT), flush=True)
                continue
            print("%s: %d kept inputs, baselines %s, %.0f s" % (r["start"], r["kept"], r["baselines"].tolist(), r["secs"]), flush=True)
            res.append(r)
    nbl = min(r["fringe"].shape[1] for r in res)
    np.savez(a.out + ".npz", starts=np.array([r["start"] for r in res]),
             t_inst_ns=np.concatenate([r["t_inst_ns"] for r in res]), fids=res[0]["fids"],
             lam=np.concatenate([r["lam"] for r in res], axis=2), noise=np.concatenate([r["noise"] for r in res], axis=1),
             excess=np.concatenate([r["excess"] for r in res], axis=1),
             fringe=np.concatenate([r["fringe"][:, :nbl] for r in res], axis=2), baselines=res[0]["baselines"][:nbl],
             kept=np.array([r["kept"] for r in res]), clean=CLEAN, cleaned=CLEANED, uncleaned=UNCLEANED)
    print("saved %s.npz" % a.out)


if __name__ == "__main__":
    main()
