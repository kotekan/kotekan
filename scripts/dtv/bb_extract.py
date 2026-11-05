#!/usr/bin/env python3
"""Pull a channel subset of the 2026-10-30 baseband dump from the per-instance pieces on cs00 to
local disk, reading each 8.07 GB piece SEQUENTIALLY (the only fast path on cs00's array; random
reads crawl) and keeping only the wanted channels.

Pieces: /mnt/cs00/data/baseband/20261030T105920Z/<host>_x000<g>_baseband_subset.<20k>.h5 with
baseband_subset uint8 [163840, 384, 2, 64] (T, F, P, D), coarse_freq attr = the 384 channel ids
(== our freq_id; instance residue = coarse_freq mod 16). Chunk k starts 10:59:19.971Z + 0.838861 k s.

  pull      <out>/k<kk>_r<res>.npy  uint8 [163840, nkeep_inst, 2, 64]  (+ k<kk>_r<res>_freq.npy)
  assemble  <out>/k<kk>.npy         uint8 [163840, nkeep, 2, 64], channels ascending (+ k<kk>_freq.npy)

    bb_extract.py pull --out /var/tmp/bb1030 --freq 5939-6626 --chunks 47-57 [--residues 0,2,4]
    bb_extract.py assemble --out /var/tmp/bb1030 --chunks 47-57

Run on cf06 (or cf05 for a residue subset, then rsync the r-files to cf06) with nice/ionice.
"""
import argparse
import glob
import os
import sys
import time

import h5py
import numpy as np

PIECES = "/mnt/cs00/data/baseband/20261030T105920Z"
NT, NF, BLOCK = 163840, 384, 1700          # BLOCK = 10 HDF5 chunks of 170 samples


def ranges(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return sorted(set(out))


def instance_files():
    """residue -> prefix (host_x000g) from the chunk-0 pieces."""
    res = {}
    for f in sorted(glob.glob(os.path.join(PIECES, "*_baseband_subset.00000000.h5"))):
        with h5py.File(f, "r") as h:
            cf = h["baseband_subset"].attrs["coarse_freq"]
        res[int(cf[0]) % 16] = os.path.basename(f).split("_baseband_subset")[0]
    return res


def pull_piece(path, keep, out_npy, out_freq):
    with h5py.File(path, "r") as h:
        d = h["baseband_subset"]
        cf = np.asarray(d.attrs["coarse_freq"], dtype=np.int64)
        idx = np.nonzero(np.isin(cf, keep))[0]
        if len(idx) == 0:
            return 0, 0.0
        np.save(out_freq, cf[idx].astype(np.int32))
        tmp = out_npy + ".tmp.npy"
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(NT, len(idx), 2, 64))
        t0 = time.time()
        for s in range(0, NT, BLOCK):
            blk = d[s:s + BLOCK]                      # sequential, chunk-aligned, [BLOCK, 384, 2, 64]
            out[s:s + BLOCK] = blk[:, idx]
        out.flush(); del out
        os.rename(tmp, out_npy)
        return len(idx), time.time() - t0


def cmd_pull(a):
    keep = np.array(ranges(a.freq)); chunks = ranges(a.chunks)
    inst = instance_files()
    residues = ranges(a.residues) if a.residues != "all" else sorted(inst)
    os.makedirs(a.out, exist_ok=True)
    for k in chunks:
        for r in residues:
            out_npy = os.path.join(a.out, "k%02d_r%02d.npy" % (k, r))
            if os.path.exists(out_npy):
                continue
            path = os.path.join(PIECES, "%s_baseband_subset.%08d.h5" % (inst[r], 20 * k))
            n, dt = pull_piece(path, keep, out_npy, os.path.join(a.out, "k%02d_r%02d_freq.npy" % (k, r)))
            print("%s chunk %2d residue %2d (%s): %3d ch in %5.0f s (%.0f MB/s of piece)"
                  % (time.strftime("%H:%M:%S", time.gmtime()), k, r, inst[r], n, dt, 8.07e3 / max(dt, 1e-9)), flush=True)


def cmd_assemble(a):
    for k in ranges(a.chunks):
        files = sorted(glob.glob(os.path.join(a.out, "k%02d_r??.npy" % k)))
        if len(files) != 16:
            print("chunk %2d: %d/16 residues present, skipped" % (k, len(files))); continue
        parts = [(np.load(f.replace(".npy", "_freq.npy")), np.load(f, mmap_mode="r")) for f in files]
        freq = np.concatenate([p[0] for p in parts]); order = np.argsort(freq)
        out_path = os.path.join(a.out, "k%02d.npy" % k)
        out = np.lib.format.open_memmap(out_path + ".tmp.npy", mode="w+", dtype=np.uint8, shape=(NT, len(freq), 2, 64))
        col = np.empty(len(freq), dtype=np.int64); col[order] = np.arange(len(freq))   # source column -> output column
        j = 0
        for fr, arr in parts:
            for s in range(0, NT, 8192):
                out[s:s + 8192, col[j:j + len(fr)]] = arr[s:s + 8192]
            j += len(fr)
        out.flush(); del out
        os.rename(out_path + ".tmp.npy", out_path); np.save(os.path.join(a.out, "k%02d_freq.npy" % k), freq[order].astype(np.int32))
        print("chunk %2d assembled: %d channels %d..%d" % (k, len(freq), freq.min(), freq.max()), flush=True)
        if a.rm:
            for f in files:
                os.remove(f); os.remove(f.replace(".npy", "_freq.npy"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull"); p.add_argument("--out", required=True); p.add_argument("--freq", required=True)
    p.add_argument("--chunks", required=True); p.add_argument("--residues", default="all"); p.set_defaults(fn=cmd_pull)
    p = sub.add_parser("assemble"); p.add_argument("--out", required=True); p.add_argument("--chunks", required=True)
    p.add_argument("--rm", action="store_true", help="delete the per-residue files after assembling"); p.set_defaults(fn=cmd_assemble)
    args = ap.parse_args(); args.fn(args)
