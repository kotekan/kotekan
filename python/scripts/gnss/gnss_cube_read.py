#!/usr/bin/env python3
"""Read the raw beam-cube archive -- the files the cf06 archiver instance writes.

    gnss_cube_read.py ls   <path> [...]        completeness: senders, windows, holes, drops
    gnss_cube_read.py head <file> [--n 1]      one frame's header and a slice of its arrays
    gnss_cube_read.py npz  <path> [...] --out X.npz   the stripped cube, ready to accumulate

⚠️ THE FILE IS SELF-DELIMITING AND SELF-DESCRIBING, AND NOTHING HERE IS CONFIGURED.
Each frame arrives as `uint32 metadata_size | metadata | frame` (rawFileWrite), and the frame's
own header carries the maxima it was sized for -- so this reader never needs the generator's
flags, the node config, or the frame size. That is deliberate: an archive that can only be read
with the config that produced it stops being readable the day the config moves on.

⚠️ WHAT IS AND IS NOT A LOSS. `dropped_windows` is CUMULATIVE per sender, so a rise between
consecutive frames sizes a gap the SENDER made (a full output buffer). A hole in the window
index NOT matched by a rise happened downstream -- bufferSend's drop_frames, the network, the
archiver's own drop_frames -- and the two are worth telling apart, because the first is a
tracker running hot and the second is a transport or a disk. `ls` reports them separately.

⚠️ POWER SUMS, NEVER dB. incoh/coh are SUMS over the records in a window (accumulated in
double, shipped as float32) with `w` the term count -- so a mean is incoh/w and windows ADD.
Decibels do not, which is the whole reason the wire carries linear accumulators.
"""

import argparse
import glob
import os
import struct
import sys

import numpy as np

CUBE_VERSION_MIN = 2
CUBE_CHAIN_CHARS = 48
CUBE_HEADER_BYTES = 96 + CUBE_CHAIN_CHARS


def frame_bytes(max_prn, max_bins, n_elem):
    """The SAME expression as gnss::cube_frame_bytes / gnss_record_layout.cube_frame_bytes."""
    return (CUBE_HEADER_BYTES
            + max_bins * 2 * 4
            + max_prn * 3 * 4
            + max_prn * 8
            + max_prn * max_bins * 2 * 4
            + max_prn * max_bins * n_elem * 3 * 4)


def parse_header(buf):
    (ver, idx, w0, w1, dropped, n_prn, n_bin, n_elem) = struct.unpack_from("<8q", buf, 0)
    win_samples, sample_rate = struct.unpack_from("<2d", buf, 64)
    gpu, bin_width, max_prn, max_bins = struct.unpack_from("<4i", buf, 80)
    chain = buf[96:96 + CUBE_CHAIN_CHARS].split(b"\0")[0].decode("ascii", "replace")
    if ver < CUBE_VERSION_MIN:
        raise SystemExit(
            f"cube frame version {ver}: versions before {CUBE_VERSION_MIN} did not carry the "
            f"maxima, so their frame length is not derivable from the frame. Read them with the "
            f"generator flags that produced them, or (better) re-record -- v1 was never written "
            f"outside a bench.")
    return dict(version=ver, idx=idx, wstart0=w0, wstart1=w1, dropped=dropped, n_prn=n_prn,
                n_bin=n_bin, n_elem=n_elem, win_samples=win_samples, sample_rate=sample_rate,
                gpu=gpu, bin_width=bin_width, max_prn=max_prn, max_bins=max_bins, chain=chain)


def parse_arrays(buf, h):
    """The five payload blocks, sliced to the ACTUAL extents (the pad is dropped here)."""
    mp, mb, ne = h["max_prn"], h["max_bins"], h["n_elem"]
    np_, nb = h["n_prn"], h["n_bin"]
    o = CUBE_HEADER_BYTES

    def take(count, dtype, itemsize):
        nonlocal o
        a = np.frombuffer(buf, dtype=dtype, count=count, offset=o)
        o += count * itemsize
        return a

    phi0 = take(mp, "<f8", 8)[:np_]
    fid_lo = take(mb, "<i4", 4)[:nb]
    fid_hi = take(mb, "<i4", 4)[:nb]
    prn = take(mp, "<i4", 4)[:np_]
    nrec = take(mp, "<i4", 4)[:np_]
    nre = take(mp, "<i4", 4)[:np_]
    w = take(mp * mb, "<f4", 4).reshape(mp, mb)[:np_, :nb]
    en = take(mp * mb, "<f4", 4).reshape(mp, mb)[:np_, :nb]
    coh_re = take(mp * mb * ne, "<f4", 4).reshape(mp, mb, ne)[:np_, :nb]
    coh_im = take(mp * mb * ne, "<f4", 4).reshape(mp, mb, ne)[:np_, :nb]
    incoh = take(mp * mb * ne, "<f4", 4).reshape(mp, mb, ne)[:np_, :nb]
    return dict(phi0=phi0, freq_id_lo=fid_lo, freq_id_hi=fid_hi, prn=prn, n_rec=nrec,
                n_reanchor=nre, w=w, energy=en, coh_re=coh_re, coh_im=coh_im, incoh=incoh)


def iter_frames(path, want_arrays=True):
    """Yield (header, arrays|None) for every frame in one raw file.

    A TRUNCATED TAIL IS NORMAL, NOT AN ERROR: the archiver is writing the newest file while this
    reads it, and a kill leaves a partial frame. Stop cleanly at the first short read -- but
    anything else (a bad version, a length that does not advance) is a real corruption and
    raises, because silently skipping bytes would resynchronise onto garbage.
    """
    with open(path, "rb") as fh:
        blob = fh.read()
    o, n = 0, len(blob)
    while o + 4 <= n:
        (msize,) = struct.unpack_from("<I", blob, o)
        o += 4
        if o + msize > n:
            return
        o += msize
        if o + CUBE_HEADER_BYTES > n:
            return
        h = parse_header(blob[o:o + CUBE_HEADER_BYTES])
        fb = frame_bytes(h["max_prn"], h["max_bins"], h["n_elem"])
        if fb <= 0:
            raise SystemExit(f"{path}: frame at {o} claims a {fb} B length")
        if o + fb > n:
            return
        yield h, (parse_arrays(blob[o:o + fb], h) if want_arrays else None)
        o += fb


def expand(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "**", "*.raw"), recursive=True))
        else:
            out += sorted(glob.glob(p))
    if not out:
        raise SystemExit("no .raw files matched")
    return out


def cmd_ls(args):
    senders = {}
    total = 0
    for path in expand(args.path):
        for h, _ in iter_frames(path, want_arrays=False):
            total += 1
            key = (h["chain"], h["gpu"])
            s = senders.setdefault(key, dict(idx=[], dropped0=h["dropped"], dropped1=h["dropped"],
                                             n_prn=h["n_prn"], n_bin=h["n_bin"],
                                             n_elem=h["n_elem"], win=h["win_samples"],
                                             rate=h["sample_rate"]))
            s["idx"].append(h["idx"])
            s["dropped1"] = h["dropped"]
    print(f"{total} frame(s), {len(senders)} sender(s)")
    print(f"{'sender':<34} {'gpu':>3} {'windows':>8} {'span':>9} {'holes':>7} "
          f"{'dropped':>8} {'shape':>14}")
    grand_holes = grand_drop = 0
    for (chain, gpu), s in sorted(senders.items()):
        idx = np.array(sorted(set(s["idx"])))
        span = int(idx[-1] - idx[0] + 1) if len(idx) else 0
        holes = span - len(idx)
        drop = s["dropped1"] - s["dropped0"]
        grand_holes += holes
        grand_drop += drop
        shape = f"{s['n_prn']}x{s['n_bin']}x{s['n_elem']}"
        print(f"{chain:<34} {gpu:>3} {len(idx):>8} {span:>9} {holes:>7} {drop:>8} {shape:>14}")
    secs = (list(senders.values())[0]["win"] / list(senders.values())[0]["rate"]) if senders else 0
    print(f"\nwindow {secs:.5f} s")
    # ⚠️ THE TWO KINDS OF LOSS ARE NOT THE SAME FAULT, so they are never added together.
    print(f"holes in the window index : {grand_holes}   "
          f"(missing frames NOT claimed by any sender)")
    print(f"sender-side drops         : {grand_drop}   "
          f"(assembler output buffer full -- the tracker outran the archive)")
    if grand_holes and not grand_drop:
        print("⇒ every hole is DOWNSTREAM of the sender: bufferSend drop_frames, the network, "
              "or the archiver. Look there, not at the nodes.")
    if grand_drop and not grand_holes:
        print("⇒ the senders dropped and SAID SO; nothing was lost in transport.")


def cmd_head(args):
    for path in expand([args.file]):
        for i, (h, a) in enumerate(iter_frames(path)):
            if i >= args.n:
                break
            print(f"-- {os.path.basename(path)} frame {i}")
            for k in ("version", "chain", "gpu", "idx", "wstart0", "wstart1", "dropped",
                      "n_prn", "n_bin", "n_elem", "max_prn", "max_bins", "bin_width"):
                print(f"   {k:<12} {h[k]}")
            print(f"   window       {h['win_samples']:.0f} samples "
                  f"({h['win_samples'] / h['sample_rate']:.5f} s)")
            print(f"   freq_id      {a['freq_id_lo'].tolist()} .. {a['freq_id_hi'].tolist()}")
            print(f"   prn          {a['prn'].tolist()}")
            print(f"   n_rec        {a['n_rec'].tolist()}")
            live = a["n_rec"] > 0
            print(f"   live slots   {int(live.sum())} of {h['n_prn']}")
            if live.any():
                p = int(np.argmax(a["n_rec"]))
                inc = a["incoh"][p]
                wgt = np.maximum(a["w"][p], 1e-30)
                mean = inc / wgt[:, None]
                print(f"   PRN {a['prn'][p]} mean |A|^2 per (bin, elem), bin 0: "
                      f"{np.array2string(mean[0][:8], precision=3)}")
        break


def cmd_npz(args):
    """Strip the pad and stack -- the shape an accumulator wants, not a per-frame stream."""
    hdrs, blocks = [], []
    for path in expand(args.path):
        for h, a in iter_frames(path):
            hdrs.append(h)
            blocks.append(a)
    if not hdrs:
        raise SystemExit("no frames")
    np.savez_compressed(
        args.out,
        chain=np.array([h["chain"] for h in hdrs]),
        gpu=np.array([h["gpu"] for h in hdrs]),
        idx=np.array([h["idx"] for h in hdrs], dtype=np.int64),
        wstart0=np.array([h["wstart0"] for h in hdrs], dtype=np.int64),
        win_samples=np.array([h["win_samples"] for h in hdrs]),
        sample_rate=np.array([h["sample_rate"] for h in hdrs]),
        prn=np.array([b["prn"] for b in blocks], dtype=object),
        freq_id_lo=np.array([b["freq_id_lo"] for b in blocks], dtype=object),
        n_rec=np.array([b["n_rec"] for b in blocks], dtype=object),
        w=np.array([b["w"] for b in blocks], dtype=object),
        energy=np.array([b["energy"] for b in blocks], dtype=object),
        incoh=np.array([b["incoh"] for b in blocks], dtype=object),
        coh_re=np.array([b["coh_re"] for b in blocks], dtype=object),
        coh_im=np.array([b["coh_im"] for b in blocks], dtype=object),
        phi0=np.array([b["phi0"] for b in blocks], dtype=object),
        n_reanchor=np.array([b["n_reanchor"] for b in blocks], dtype=object),
        allow_pickle=True)
    print(f"wrote {args.out}: {len(hdrs)} frame(s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ls", help="completeness per sender")
    p.add_argument("path", nargs="+")
    p.set_defaults(fn=cmd_ls)
    p = sub.add_parser("head", help="one frame in full")
    p.add_argument("file")
    p.add_argument("--n", type=int, default=1)
    p.set_defaults(fn=cmd_head)
    p = sub.add_parser("npz", help="stripped arrays for an accumulator")
    p.add_argument("path", nargs="+")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_npz)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
