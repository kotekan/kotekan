#!/usr/bin/env python3
"""Read the raw beam-cube archive -- the files the cf06 archiver instance writes.

    gnss_cube_read.py ls   <path> [...]        completeness: senders, windows, holes, drops
    gnss_cube_read.py head <file> [--n 1]      one frame's header and a slice of its arrays

The file is self-delimiting and self-describing, and nothing here is configured. Each frame
is stored as `uint32 metadata_size | metadata | frame` (rawFileWrite, or GnssCubeWrite for the
packed form), and the frame's own header carries the maxima it was sized for, so this reader
needs neither the generator's flags, the node config nor the frame size: the archive stays
readable after the config changes. Packed frames (version 4, GnssCubeWrite.hpp) hold only the
PRN slots with records and the actual bins; they are read back into the wire version's shape,
or kept packed with `expand=False`.

What is and is not a loss. `dropped_windows` is cumulative per sender, so a rise between
consecutive frames sizes a gap the sender made (a full output buffer). A hole in the window
index not matched by a rise happened downstream -- bufferSend's drop_frames, the network, the
archiver's own drop_frames. The first is a tracker running hot, the second a transport or a
disk, so `ls` reports them separately.

Power sums, never dB. incoh/coh are sums over the records in a window (accumulated in double,
shipped as float32) with `w` the term count, so a mean is incoh/w and windows add.
"""

import argparse
import glob
import os
import struct
import sys

import numpy as np

CUBE_VERSION_MIN = 2
CUBE_CHAIN_CHARS = 48
# Fixed header bytes (= offset of the chain string) by version: v2 has the chain at 96; v3 puts
# `double utc0` at 96 and the chain at 104. The frame's version field selects, so archives of
# both versions stay readable.
CUBE_FIXED_BYTES = {2: 96, 3: 104}
CUBE_HEADER_BYTES = {v: n + CUBE_CHAIN_CHARS for v, n in CUBE_FIXED_BYTES.items()}
CUBE_HEADER_MIN = min(CUBE_HEADER_BYTES.values())  # enough to read the version + maxima
# Packed archive versions (gnss::CUBE_PACKED_VERSION) -> the wire version they pack: its header,
# then its arrays cut to max_prn = the slots with records and max_bins = n_bin, with int32
# slot[max_prn] (each slot's index in the sender's frame) after n_reanchor.
CUBE_PACKED = {4: 3}


def wire_version(version):
    return CUBE_PACKED.get(version, version)


def header_bytes(version):
    return CUBE_HEADER_BYTES.get(wire_version(version), CUBE_HEADER_MIN)


def frame_bytes(max_prn, max_bins, n_elem, version=3):
    """Same expression as gnss::cube_frame_bytes and config/gnss_record_layout.py, which size
    the current version (gnss::cube_packed_bytes for a packed one); `version` selects the header
    size."""
    return (
        CUBE_HEADER_BYTES[wire_version(version)]
        + max_bins * 2 * 4
        + max_prn * (4 if version in CUBE_PACKED else 3) * 4
        + max_prn * 8
        + max_prn * max_bins * 2 * 4
        + max_prn * max_bins * n_elem * 3 * 4
    )


def parse_header(buf):
    (ver, idx, w0, w1, dropped, n_prn, n_bin, n_elem) = struct.unpack_from(
        "<8q", buf, 0
    )
    win_samples, sample_rate = struct.unpack_from("<2d", buf, 64)
    gpu, bin_width, max_prn, max_bins = struct.unpack_from("<4i", buf, 80)
    if ver < CUBE_VERSION_MIN:
        raise SystemExit(
            f"cube frame version {ver}: versions before {CUBE_VERSION_MIN} did not carry the "
            f"maxima, so their frame length is not derivable from the frame. Read them with the "
            f"generator flags that produced them, or re-record."
        )
    if wire_version(ver) not in CUBE_HEADER_BYTES:
        raise SystemExit(
            f"cube frame version {ver}: this reader knows {sorted(CUBE_HEADER_BYTES)} and packed "
            f"{sorted(CUBE_PACKED)} -- update CUBE_FIXED_BYTES from gnssRecord.hpp"
        )
    # utc0: UTC of absolute F-engine sample 0 (v3). A v2 frame has none: reported as None,
    # never guessed.
    utc0 = struct.unpack_from("<d", buf, 96)[0] if ver >= 3 else None
    co = CUBE_FIXED_BYTES[wire_version(ver)]
    chain = buf[co : co + CUBE_CHAIN_CHARS].split(b"\0")[0].decode("ascii", "replace")
    return dict(
        version=ver,
        idx=idx,
        wstart0=w0,
        wstart1=w1,
        dropped=dropped,
        n_prn=n_prn,
        n_bin=n_bin,
        n_elem=n_elem,
        win_samples=win_samples,
        sample_rate=sample_rate,
        gpu=gpu,
        bin_width=bin_width,
        max_prn=max_prn,
        max_bins=max_bins,
        chain=chain,
        utc0=utc0,
        header_bytes=header_bytes(ver),
    )


def parse_arrays(buf, h, expand=True):
    """The five payload blocks, sliced to the actual extents (the pad is dropped here). A packed
    frame is read back into the wire shapes, or with expand=False kept to its live slots with
    `slot`, each one's index among the chain's n_prn."""
    if h["version"] in CUBE_PACKED:
        return parse_packed(buf, h, expand)
    mp, mb, ne = h["max_prn"], h["max_bins"], h["n_elem"]
    np_, nb = h["n_prn"], h["n_bin"]
    o = h["header_bytes"]

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
    return dict(
        phi0=phi0,
        freq_id_lo=fid_lo,
        freq_id_hi=fid_hi,
        prn=prn,
        n_rec=nrec,
        n_reanchor=nre,
        w=w,
        energy=en,
        coh_re=coh_re,
        coh_im=coh_im,
        incoh=incoh,
    )


def parse_packed(buf, h, expand=True):
    """A packed frame read back into parse_arrays' shapes: each live slot at its own index among
    the chain's n_prn, zero in the slots that had no records (as the wire frame holds them).
    With expand=False, the live slots only, in the frame's order, and `slot`."""
    nl, nb, ne, np_ = h["max_prn"], h["max_bins"], h["n_elem"], h["n_prn"]
    o = h["header_bytes"]

    def take(count, dtype, itemsize):
        nonlocal o
        a = np.frombuffer(buf, dtype=dtype, count=count, offset=o)
        o += count * itemsize
        return a

    phi0 = take(nl, "<f8", 8)
    fid_lo = take(nb, "<i4", 4)
    fid_hi = take(nb, "<i4", 4)
    prn = take(nl, "<i4", 4)
    nrec = take(nl, "<i4", 4)
    nre = take(nl, "<i4", 4)
    slot = take(nl, "<i4", 4)
    if nl and (slot.min() < 0 or slot.max() >= np_ or len(set(slot.tolist())) != nl):
        raise SystemExit(
            f"packed cube frame idx {h['idx']}: slots {slot.tolist()} for n_prn {np_}"
        )
    out = dict(freq_id_lo=fid_lo, freq_id_hi=fid_hi)
    for name, a, dtype in (
        ("phi0", phi0, "<f8"),
        ("prn", prn, "<i4"),
        ("n_rec", nrec, "<i4"),
        ("n_reanchor", nre, "<i4"),
    ):
        if expand:
            full = np.zeros(np_, dtype)
            full[slot] = a
            a = full
        out[name] = a
    if not expand:
        out["slot"] = slot
    for name, inner in (
        ("w", ()),
        ("energy", ()),
        ("coh_re", (ne,)),
        ("coh_im", (ne,)),
        ("incoh", (ne,)),
    ):
        n = nl * nb * (ne if inner else 1)
        a = take(n, "<f4", 4).reshape((nl, nb) + inner)
        if expand:
            full = np.zeros((np_, nb) + inner, "<f4")
            full[slot] = a
            a = full
        out[name] = a
    return out


def iter_frames(path, want_arrays=True, expand=True):
    """Yield (header, arrays|None) for every frame in one raw file. The arrays are views into the
    file's buffer, which stays in memory while any of them does; `expand` is parse_arrays'.

    A truncated tail is normal, not an error: the archiver may be writing the newest file while
    this reads it, and a kill leaves a partial frame. Stop cleanly at the first short read; but
    anything else (a bad version, a length that does not advance) is a real corruption and
    raises, because silently skipping bytes would resynchronise onto garbage.
    """
    with open(path, "rb") as fh:
        blob = fh.read()
    view = memoryview(blob)
    o, n = 0, len(blob)
    while o + 4 <= n:
        (msize,) = struct.unpack_from("<I", blob, o)
        o += 4
        if o + msize > n:
            return
        o += msize
        if o + CUBE_HEADER_MIN > n:
            return
        # The version sits at byte 0 and the header grows with it: peek the minimum, size the rest.
        (ver,) = struct.unpack_from("<q", blob, o)
        hb = header_bytes(ver)
        if o + hb > n:
            return
        h = parse_header(blob[o : o + hb])
        fb = frame_bytes(h["max_prn"], h["max_bins"], h["n_elem"], h["version"])
        if fb <= 0:
            raise SystemExit(f"{path}: frame at {o} claims a {fb} B length")
        if o + fb > n:
            return
        yield h, (parse_arrays(view[o : o + fb], h, expand) if want_arrays else None)
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
            s = senders.setdefault(
                key,
                dict(
                    idx=[],
                    dropped0=h["dropped"],
                    dropped1=h["dropped"],
                    n_prn=h["n_prn"],
                    n_bin=h["n_bin"],
                    n_elem=h["n_elem"],
                    win=h["win_samples"],
                    rate=h["sample_rate"],
                ),
            )
            s["idx"].append(h["idx"])
            s["dropped1"] = h["dropped"]
    print(f"{total} frame(s), {len(senders)} sender(s)")
    print(
        f"{'sender':<34} {'gpu':>3} {'windows':>8} {'span':>9} {'holes':>7} "
        f"{'dropped':>8} {'shape':>14}"
    )
    grand_holes = grand_drop = grand_unclaimed = 0
    for (chain, gpu), s in sorted(senders.items()):
        idx = np.array(sorted(set(s["idx"])))
        span = int(idx[-1] - idx[0] + 1) if len(idx) else 0
        holes = span - len(idx)
        drop = s["dropped1"] - s["dropped0"]
        grand_holes += holes
        grand_drop += drop
        # A window the sender dropped is a hole in its index too; the rest no sender claims. A
        # counter that fell (the sender restarted, so it counts from 0 again) claims nothing.
        grand_unclaimed += max(0, holes - max(drop, 0))
        shape = f"{s['n_prn']}x{s['n_bin']}x{s['n_elem']}"
        print(
            f"{chain:<34} {gpu:>3} {len(idx):>8} {span:>9} {holes:>7} {drop:>8} {shape:>14}"
        )
    secs = (
        (list(senders.values())[0]["win"] / list(senders.values())[0]["rate"])
        if senders
        else 0
    )
    print(f"\nwindow {secs:.5f} s")
    # The holes split into the two kinds of loss, which are different faults: the windows the
    # senders dropped, and the rest, which no sender claims.
    print(f"holes in the window index : {grand_holes}")
    print(
        f"sender-side drops         : {grand_drop}   "
        f"(assembler output buffer full -- the tracker outran the archive)"
    )
    print(
        f"unclaimed (downstream)    : {grand_unclaimed}   "
        f"(missing frames not claimed by any sender)"
    )
    if grand_unclaimed and not grand_drop:
        print(
            "⇒ every hole is DOWNSTREAM of the sender: bufferSend drop_frames, the network, "
            "or the archiver. Look there, not at the nodes."
        )
    if grand_drop and not grand_unclaimed:
        print("⇒ the senders dropped and said so; nothing was lost in transport.")


def cmd_head(args):
    for path in expand([args.file]):
        for i, (h, a) in enumerate(iter_frames(path)):
            if i >= args.n:
                break
            print(f"-- {os.path.basename(path)} frame {i}")
            for k in (
                "version",
                "chain",
                "gpu",
                "idx",
                "wstart0",
                "wstart1",
                "dropped",
                "n_prn",
                "n_bin",
                "n_elem",
                "max_prn",
                "max_bins",
                "bin_width",
                "utc0",
            ):
                print(f"   {k:<12} {h[k]}")
            if h["utc0"]:
                import datetime

                t = h["utc0"] + h["wstart0"] / h["sample_rate"]
                iso = datetime.datetime.fromtimestamp(t, datetime.UTC).isoformat(
                    timespec="milliseconds"
                )
                print(f"   utc(wstart0) {iso}")
            print(
                f"   window       {h['win_samples']:.0f} samples "
                f"({h['win_samples'] / h['sample_rate']:.5f} s)"
            )
            print(
                f"   freq_id      {a['freq_id_lo'].tolist()} .. {a['freq_id_hi'].tolist()}"
            )
            print(f"   prn          {a['prn'].tolist()}")
            print(f"   n_rec        {a['n_rec'].tolist()}")
            live = a["n_rec"] > 0
            print(f"   live slots   {int(live.sum())} of {h['n_prn']}")
            if live.any():
                p = int(np.argmax(a["n_rec"]))
                inc = a["incoh"][p]
                wgt = np.maximum(a["w"][p], 1e-30)
                mean = inc / wgt[:, None]
                print(
                    f"   PRN {a['prn'][p]} mean |A|^2 per (bin, elem), bin 0: "
                    f"{np.array2string(mean[0][:8], precision=3)}"
                )
        break


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ls", help="completeness per sender")
    p.add_argument("path", nargs="+")
    p.set_defaults(fn=cmd_ls)
    p = sub.add_parser(
        "head", help="the first --n frames: header and a slice of the arrays"
    )
    p.add_argument("file")
    p.add_argument("--n", type=int, default=1)
    p.set_defaults(fn=cmd_head)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
