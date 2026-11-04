#!/usr/bin/env python3
"""Read cudaGnssBeamDump raw files and stitch a pass across GPUs into a channelised npz.

A rawFileWrite file holds frames of [uint32 metadata_size][metadata][frame]; the frame is the
layout of lib/cuda/cudaGnssBeamDumpKernel.hpp: a 64-byte header (int64 seq0, magic, version,
n_chan, n_times, n_inputs, ref_input, conj, eig_iters, n_targets), then per channel the freq_id,
four floats (lambda1, lambda2, 0, 0), 128 mean powers, 128 complex weights, and n_times half2
samples. A version 2 frame with n_targets > 0 (the track beam, docs/CHORD_GUOWANG_TRACK_BEAM.md)
continues with, per target and channel, a 32-byte record (prn, held, coh, power, held_s, age_s),
then the targets' weights [n_targets][n_chan][128] and samples [n_targets][n_chan][n_times].

    beam_dump_read.py info  FILE...                      # per file: seq0 range, channels, eigenvalues,
                                                         # target PRNs and their overlap with the eigen weights
    beam_dump_read.py stitch --out pass.npz FILE...      # all GPUs of one pass -> chan [n_chan, n_t],
                                                         # freq_hz, fs_chan, seq0, lam, pw, w (per frame)
    beam_dump_read.py stitch --target PRN --out x.npz FILE...   # that PRN's target beam instead

The npz is what guowang_search.py and the bb1030 dg_*.py tools read (chan, freq_hz, fs_chan);
the samples are the F-engine's channelised output as decoded (conj=0). Stitch them WITHOUT
conjugating the channels (channels_to_baseband(..., conj=False)): conjugating each channel's
contents while keeping the channel placement mirrors every component about its own channel
centre, which turns a wideband chip-clock line at R into images at 2 N x 195312.5 - R with the
Doppler sign reversed (2026-11-02: the 10.278 / 2.0556 MHz "chip rates" were such images of the
true 10.23 / 2.046 Mcps). Frames missing on some GPU are zero-filled and reported.

The kernel writes each beam in the units and phase of beam_ref_input; if that input carries no
sky signal (2026-11-02: input 100 on the live array) the beams come out ~0 with a random phase
per frame. stitch re-references them to a live input from the saved weights (--ref-input auto),
which `info` reports as "median 128 |q_ref|^2".
"""
import argparse
import glob
import os
import struct
import sys

import numpy as np

HDR = struct.Struct("<qiiiiiiiii20x")  # 64 bytes
MAGIC = 0x504D4442
CS = 195312.5
# gnss_cuda::BeamDumpTarget
TGT = np.dtype(
    [
        ("prn", "<i4"),
        ("held", "<i4"),
        ("coh", "<f4"),
        ("power", "<f4"),
        ("held_s", "<f4"),
        ("age_s", "<f4"),
        ("reserved", "<f4", (2,)),
    ]
)


def layout(n_chan, n_times, n_targets=0):
    """Section offsets of one frame (mirrors gnss_cuda::beam_dump_layout): freq, lam, pw, w, beam,
    total, then the target sections tgt, tw, tbeam (empty, at total, without targets)."""
    align = lambda off: (off + 15) & ~15
    off_freq = 64
    off_lam = align(off_freq + 4 * n_chan)
    off_pw = align(off_lam + 16 * n_chan)
    off_w = align(off_pw + 4 * 128 * n_chan)
    off_beam = align(off_w + 8 * 128 * n_chan)
    off_tgt = align(off_beam + 4 * n_chan * n_times)
    off_tw = align(off_tgt + TGT.itemsize * n_targets * n_chan)
    off_tbeam = align(off_tw + 8 * 128 * n_targets * n_chan)
    total = align(off_tbeam + 4 * n_targets * n_chan * n_times)
    return off_freq, off_lam, off_pw, off_w, off_beam, total, off_tgt, off_tw, off_tbeam


def frames(path):
    """Yield per frame: seq0, freq_ids, lam [n,4], pw [n,128], w [n,128] complex64, beam
    [n, n_times] complex64, ref, conj, n_times, n_targets, and with targets tgt [nt, n] (TGT),
    tw [nt, n, 128] complex64, tbeam [nt, n, n_times] complex64 (else None)."""
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
            n_targets,
        ) = HDR.unpack_from(data, pos)
        if magic != MAGIC:
            raise SystemExit(
                "%s: bad magic 0x%08x at byte %d" % (path, magic & 0xFFFFFFFF, pos)
            )
        if version < 2:
            n_targets = 0
        (
            off_freq,
            off_lam,
            off_pw,
            off_w,
            off_beam,
            total,
            off_tgt,
            off_tw,
            off_tbeam,
        ) = layout(n_chan, n_times, n_targets)
        if pos + total > len(data):
            print("%s: truncated frame at byte %d" % (path, pos), file=sys.stderr)
            break
        tgt = tw = tbeam = None
        if n_targets:
            nt = n_targets
            tgt = np.frombuffer(data, TGT, nt * n_chan, pos + off_tgt).reshape(
                nt, n_chan
            )
            tw = np.frombuffer(
                data, np.complex64, 128 * nt * n_chan, pos + off_tw
            ).reshape(nt, n_chan, 128)
            tbeam = (
                np.frombuffer(
                    data, np.float16, 2 * nt * n_chan * n_times, pos + off_tbeam
                )
                .astype(np.float32)
                .view(np.complex64)
                .reshape(nt, n_chan, n_times)
            )
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
            n_targets=n_targets,
            tgt=tgt.copy() if tgt is not None else None,
            tw=tw.copy() if tw is not None else None,
            tbeam=tbeam,
        )
        pos += total


def target_overlap(w, tw):
    """|<w, tw>|^2 / (|w|^2 |tw|^2): the target weights' overlap with the eigen weights of the
    same channel and frame. Near 1 when the target lane is the channel's dominant source and
    the weights' conjugation convention is right; near 1/128 with the wrong one."""
    d = float(np.sum(np.abs(w) ** 2) * np.sum(np.abs(tw) ** 2))
    return float(abs(np.vdot(w, tw)) ** 2 / d) if d > 0.0 else 0.0


def ref_phased_q(pw, w):
    """The kernel's phase-referenced whitened eigenvector q (|q| = 1) and s = |q_ref| / wt_ref from
    the saved weights w_i = conj(q_i) wt_i s, wt_i = 1 / sqrt(pw_i)."""
    wt = 1.0 / np.sqrt(np.maximum(pw, 1e-3))
    s2 = float(np.sum(np.abs(w) ** 2 * pw))
    if s2 <= 0.0:
        return None, 0.0
    return np.conj(w) / (wt * np.sqrt(s2)), np.sqrt(s2)


def reref_factor(pw, w, j):
    """Scalar that turns the saved beam (reference input's units and phase) into input j's units
    and phase: beam_j = beam * q_j / (wt_j s). Zero when the frame's beam is unrecoverable (the
    kernel zeroes the beam when |q_ref| < 1e-6)."""
    q, s = ref_phased_q(pw, w)
    if q is None:
        return 0.0
    return q[j] * np.sqrt(np.maximum(pw[j], 1e-3)) / s


def expand(files):
    out = []
    for f in files:
        out += sorted(glob.glob(f)) if any(ch in f for ch in "*?[") else [f]
    return out


def first_seq0(path):
    """seq0 of the first frame, from the headers only."""
    with open(path, "rb") as fh:
        (msize,) = struct.unpack("<I", fh.read(4))
        fh.seek(4 + msize)
        return HDR.unpack_from(fh.read(64))[0]


def cmd_info(a):
    for path in expand(a.files):
        seqs = []
        n = 0
        lam1 = []
        qref = []
        qabs = np.zeros(128)
        n_targets = 0
        prns = {}  # (target slot, prn) -> channel-frames
        ovl = {}  # target slot -> overlaps with the eigen weights
        for fr in frames(path):
            seqs.append(fr["seq0"])
            lam1.append(fr["lam"][:, 0])
            n = fr["n_times"]
            ids = fr["freq_ids"]
            ref = fr["ref"]
            for k in range(len(ids)):
                q, _ = ref_phased_q(fr["pw"][k], fr["w"][k])
                if q is not None:
                    qref.append(abs(q[ref]) ** 2)
                    qabs += np.abs(q)
            n_targets = fr["n_targets"]
            for t in range(n_targets):
                for k in range(len(ids)):
                    prn = int(fr["tgt"][t, k]["prn"])
                    prns[(t, prn)] = prns.get((t, prn), 0) + 1
                    if prn > 0:
                        ovl.setdefault(t, []).append(
                            target_overlap(fr["w"][k], fr["tw"][t, k])
                        )
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
        print(
            "   reference input %d: median 128 |q_ref|^2 = %.3f (1 = an average live input; < 0.05 = no sky signal on it,"
            " re-reference with stitch --ref-input); strongest input %d"
            % (ref, 128 * np.median(qref) if qref else 0.0, int(np.argmax(qabs)))
        )
        for t in range(n_targets):
            seen = sorted(
                ((c, p) for (tt, p), c in prns.items() if tt == t), reverse=True
            )
            o = np.array(ovl.get(t, []))
            print(
                "   target %d: PRN channel-frames %s; overlap with the eigen weights median %.3f "
                "(> ~0.5 = the lane owns the channel and the weights' conjugation is right; "
                "~0.01 = conjugated wrong)"
                % (
                    t,
                    ", ".join("%d x%d" % (p, c) for c, p in seen[:6]),
                    float(np.median(o)) if len(o) else float("nan"),
                )
            )


def cmd_stitch(a):
    per_seq = {}  # seq0 -> {freq_id: (beam row, lam, pw, w)}
    ids_all = set()
    n_times = None
    ref = None
    conj = None
    lo, hi = (int(v) for v in a.freq.split("-")) if a.freq else (0, 1 << 30)
    files = expand(a.files)
    seq_lo = a.seq0 if a.seq0 is not None else min(first_seq0(p) for p in files)
    seq_hi = seq_lo + int(a.max_seconds * 195312.5) if a.max_seconds else 1 << 62
    for path in files:
        for fr in frames(path):
            if fr["seq0"] < seq_lo or fr["seq0"] >= seq_hi:
                continue
            n_times = fr["n_times"]
            ref = fr["ref"]
            conj = fr["conj"]
            d = per_seq.setdefault(fr["seq0"], {})
            for k, fid in enumerate(fr["freq_ids"]):
                if not lo <= int(fid) <= hi:
                    continue
                if a.target:
                    # the slot steered at that PRN this frame, if any
                    slots = [
                        t
                        for t in range(fr["n_targets"])
                        if int(fr["tgt"][t, k]["prn"]) == a.target
                    ]
                    if not slots:
                        continue
                    t = slots[0]
                    d[int(fid)] = (
                        fr["tbeam"][t, k],
                        fr["lam"][k],
                        fr["pw"][k],
                        fr["tw"][t, k],
                    )
                else:
                    d[int(fid)] = (fr["beam"][k], fr["lam"][k], fr["pw"][k], fr["w"][k])
                ids_all.add(int(fid))
    if not per_seq:
        raise SystemExit("no frames")
    seqs = sorted(per_seq)
    ids = sorted(ids_all)
    # re-reference the beams to one live input: the kernel scales and phases them by the reference
    # input's eigenvector element, which is ~0 (and its phase random per frame) when that input
    # carries no sky signal; the saved weights hold what is needed to move the reference. A
    # target beam has unit-norm weights with the lane's own phase origin: no re-referencing.
    ref_j = None
    if a.ref_input != "none" and not a.target:
        if a.ref_input == "auto":
            qabs = np.zeros(128)
            for s in seqs:
                for fid, (row, l, p, ww) in per_seq[s].items():
                    q, _ = ref_phased_q(p, ww)
                    if q is not None:
                        qabs += np.abs(q)
            ref_j = int(np.argmax(qabs))
        else:
            ref_j = int(a.ref_input)
        print("beams re-referenced to input %d (units and phase of that input)" % ref_j)
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
            if ref_j is not None:
                row = row * np.complex64(reref_factor(p, ww, ref_j))
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
        ref_input=ref if ref_j is None else ref_j,
        kernel_ref_input=ref,
        conj=conj,
        target=a.target or 0,
    )
    print(
        "saved %s%s: %d channels (%d..%d) x %d frames (%.3f s) from seq0 %d; %d channel-frames missing (zero-filled)"
        % (
            a.out,
            " (target PRN %d)" % a.target if a.target else "",
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
    p.add_argument(
        "--ref-input",
        default="auto",
        help="re-reference the beams to this input: auto (the strongest, default), an index, or none",
    )
    p.add_argument("--freq", help="keep freq_ids LO-HI only (one lobe)")
    p.add_argument(
        "--target",
        type=int,
        help="stitch this PRN's target beam (version 2 frames) instead of the eigen beam; "
        "channel-frames with no slot on that PRN are zero-filled",
    )
    p.add_argument(
        "--seq0", type=int, help="first hop to keep (default: the earliest frame)"
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        help="keep this many seconds from --seq0 (or the start)",
    )
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_stitch)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
