#!/usr/bin/env python3
"""Read cudaGnssBeamDump raw files and stitch a pass across GPUs into a channelised npz.

A rawFileWrite file holds frames of [uint32 metadata_size][metadata][frame]; the frame is the
layout of lib/cuda/cudaGnssBeamDumpKernel.hpp: a 64-byte header (int64 seq0, magic, version,
n_chan, n_times, n_inputs, ref_input, conj, eig_iters), then per channel the freq_id, four
floats (lambda1, lambda2, 0, 0), 128 mean powers, 128 complex weights, and n_times half2 samples.

    beam_dump_read.py info  FILE...                      # per file: seq0 range, channels, eigenvalues
    beam_dump_read.py stitch --out pass.npz FILE...      # all GPUs of one pass -> chan [n_chan, n_t],
                                                         # freq_hz, fs_chan, seq0, lam, pw, w (per frame)

The npz is what guowang_search.py and the bb1030 dg_*.py tools read (chan, freq_hz, fs_chan);
the samples are in the F-engine's convention (conj=0), so those tools' --conj applies as for
the baseband dump. Frames missing on some GPU are zero-filled and reported.
"""
import argparse
import glob
import os
import struct
import sys

import numpy as np

HDR = struct.Struct("<qiiiiiiii24x")  # 64 bytes
MAGIC = 0x504D4442
CS = 195312.5


def layout(n_chan, n_times):
    """Section offsets of one frame (mirrors gnss_cuda::beam_dump_layout)."""
    align = lambda off: (off + 15) & ~15
    off_freq = 64
    off_lam = align(off_freq + 4 * n_chan)
    off_pw = align(off_lam + 16 * n_chan)
    off_w = align(off_pw + 4 * 128 * n_chan)
    off_beam = align(off_w + 8 * 128 * n_chan)
    total = align(off_beam + 4 * n_chan * n_times)
    return off_freq, off_lam, off_pw, off_w, off_beam, total


def frames(path):
    """Yield (seq0, freq_ids, lam [n,4], pw [n,128], w [n,128] complex64, beam [n, n_times] complex64)."""
    with open(path, "rb") as fh:
        data = fh.read()
    pos = 0
    while pos + 4 <= len(data):
        (msize,) = struct.unpack_from("<I", data, pos)
        pos += 4 + msize
        if pos + 64 > len(data):
            break
        (
            seq0,
            magic,
            version,
            n_chan,
            n_times,
            n_inputs,
            ref,
            conj,
            iters,
        ) = HDR.unpack_from(data, pos)
        if magic != MAGIC:
            raise SystemExit(
                "%s: bad magic 0x%08x at byte %d" % (path, magic & 0xFFFFFFFF, pos)
            )
        off_freq, off_lam, off_pw, off_w, off_beam, total = layout(n_chan, n_times)
        if pos + total > len(data):
            print("%s: truncated frame at byte %d" % (path, pos), file=sys.stderr)
            break
        f = np.frombuffer(data, np.int32, n_chan, pos + off_freq)
        lam = np.frombuffer(data, np.float32, 4 * n_chan, pos + off_lam).reshape(
            n_chan, 4
        )
        pw = np.frombuffer(data, np.float32, 128 * n_chan, pos + off_pw).reshape(
            n_chan, 128
        )
        w = np.frombuffer(data, np.complex64, 128 * n_chan, pos + off_w).reshape(
            n_chan, 128
        )
        beam = (
            np.frombuffer(data, np.float16, 2 * n_chan * n_times, pos + off_beam)
            .astype(np.float32)
            .view(np.complex64)
            .reshape(n_chan, n_times)
        )
        yield dict(
            seq0=seq0,
            freq_ids=f.copy(),
            lam=lam.copy(),
            pw=pw.copy(),
            w=w.copy(),
            beam=beam,
            ref=ref,
            conj=conj,
            n_times=n_times,
        )
        pos += total


def expand(files):
    out = []
    for f in files:
        out += sorted(glob.glob(f)) if any(ch in f for ch in "*?[") else [f]
    return out


def cmd_info(a):
    for path in expand(a.files):
        seqs = []
        n = 0
        lam1 = []
        for fr in frames(path):
            seqs.append(fr["seq0"])
            lam1.append(fr["lam"][:, 0])
            n = fr["n_times"]
            ids = fr["freq_ids"]
        if not seqs:
            print("%s: no frames" % path)
            continue
        lam1 = np.array(lam1)
        print(
            "%s: %d frames, seq0 %d..%d (step %s), %d channels %d..%d x %d samples; lambda1 median per channel: %s"
            % (
                os.path.basename(path),
                len(seqs),
                seqs[0],
                seqs[-1],
                np.unique(np.diff(seqs)).tolist()[:3],
                len(ids),
                ids.min(),
                ids.max(),
                n,
                np.array2string(
                    np.median(lam1, axis=0), precision=1, max_line_width=200
                ),
            )
        )


def cmd_stitch(a):
    per_seq = {}  # seq0 -> {freq_id: (beam row, lam, pw, w)}
    ids_all = set()
    n_times = None
    ref = None
    conj = None
    for path in expand(a.files):
        for fr in frames(path):
            n_times = fr["n_times"]
            ref = fr["ref"]
            conj = fr["conj"]
            d = per_seq.setdefault(fr["seq0"], {})
            for k, fid in enumerate(fr["freq_ids"]):
                d[int(fid)] = (fr["beam"][k], fr["lam"][k], fr["pw"][k], fr["w"][k])
                ids_all.add(int(fid))
    if not per_seq:
        raise SystemExit("no frames")
    seqs = sorted(per_seq)
    ids = sorted(ids_all)
    step = n_times
    if any((s2 - s1) % step for s1, s2 in zip(seqs[:-1], seqs[1:])):
        print(
            "warning: frame seq0 values are not on a %d-sample grid" % step,
            file=sys.stderr,
        )
    n_frames = (seqs[-1] - seqs[0]) // step + 1
    chan = np.zeros((len(ids), n_frames * n_times), np.complex64)
    lam = np.full((n_frames, len(ids), 4), np.nan, np.float32)
    pw = np.full((n_frames, len(ids), 128), np.nan, np.float32)
    w = np.full((n_frames, len(ids), 128), np.nan, np.complex64)
    have = np.zeros((n_frames, len(ids)), bool)
    col = {fid: i for i, fid in enumerate(ids)}
    for s in seqs:
        j = (s - seqs[0]) // step
        for fid, (row, l, p, ww) in per_seq[s].items():
            i = col[fid]
            chan[i, j * n_times : (j + 1) * n_times] = row
            lam[j, i] = l
            pw[j, i] = p
            w[j, i] = ww
            have[j, i] = True
    missing = (~have).sum()
    np.savez(
        a.out,
        chan=chan,
        freq_hz=np.array(ids, np.float64) * CS,
        fs_chan=CS,
        seq0=seqs[0],
        n_times=n_times,
        lam=lam,
        pw=pw,
        w=w,
        have=have,
        ref_input=ref,
        conj=conj,
    )
    print(
        "saved %s: %d channels (%d..%d) x %d frames (%.3f s) from seq0 %d; %d channel-frames missing (zero-filled)"
        % (
            a.out,
            len(ids),
            ids[0],
            ids[-1],
            n_frames,
            n_frames * n_times * 5.12e-6,
            seqs[0],
            missing,
        )
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info")
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_info)
    p = sub.add_parser("stitch")
    p.add_argument("--out", required=True)
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_stitch)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
