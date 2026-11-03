#!/usr/bin/env python3
"""Read GnssN2Project lane-dump raw files (lib/stages/gnss/GnssN2Project.hpp, "Lane dump").

A rawFileWrite file holds frames of [uint32 metadata_size][metadata][frame]. A frame is a
64-byte header (int64 seq, double t_utc, int32 version, n, n_ch, max_cand, slot_bytes,
lane_conj, int64 frame_bytes, int32 magic) and n_ch channel slots: freq_id, the candidate
counts, the lane basis size and mean auto; the raw live block's upper triangle; and per lane
candidate, in the order GnssN2Project tested them, its owner, power, coherence, hold age,
track, source (0 board, 1 fleet), Gram-Schmidt verdict (1 added, 0 duplicate, -1 not tested)
and new fraction, and its direction on the stations axis.

    lane_dump_read.py info FILE...     # per frame: time, channels, candidates, verdicts
"""
import argparse
import struct
import sys

import numpy as np

HDR = struct.Struct("<qdiiiiiiqi12x")  # 64 bytes
SLOT = struct.Struct("<iiiii4xd32x")  # 64 bytes
MAGIC = 0x454E414C
VERSION = 1


def cand_dtype(n):
    return np.dtype(
        [
            ("owner", "S64"),
            ("power", "<f8"),
            ("coh", "<f8"),
            ("held_s", "<f8"),
            ("t_utc", "<f8"),
            ("los", "<f8", 3),
            ("los_rate", "<f8", 3),
            ("src", "<i4"),
            ("xfer", "<i4"),
            ("verdict", "<i4"),
            ("new_frac", "<f4"),
            ("v", "<c8", n),
        ]
    )


def hermitian(tri, n):
    """The n x n block from its upper triangle (i <= j, row-major)."""
    V = np.zeros((n, n), np.complex64)
    iu = np.triu_indices(n)
    V[iu] = tri
    V = V + V.conj().T
    V[np.diag_indices(n)] *= 0.5
    return V


def frames(path, with_v=True):
    """Yield one dict per frame: seq, t_utc, n, lane_conj and channels, a list of dicts with
    freq_id, n_cand_total, n_local, k, mean_auto, V (n x n complex64, if with_v) and cand (a
    structured array, cand_dtype(n))."""
    with open(path, "rb") as fh:
        data = fh.read()
    pos = 0
    while pos + 4 <= len(data):
        (msize,) = struct.unpack_from("<I", data, pos)
        pos += 4 + msize
        if pos + HDR.size > len(data):
            break
        (
            seq,
            t_utc,
            ver,
            n,
            n_ch,
            max_cand,
            slot_bytes,
            conj,
            frame_bytes,
            magic,
        ) = HDR.unpack_from(data, pos)
        if magic != MAGIC or ver != VERSION:
            raise SystemExit(
                "%s: not a version-%d lane dump frame at byte %d (magic 0x%08x, version %d)"
                % (path, VERSION, pos, magic & 0xFFFFFFFF, ver)
            )
        if pos + frame_bytes > len(data):
            print("%s: truncated frame at byte %d" % (path, pos), file=sys.stderr)
            break
        dt = cand_dtype(n)
        n_tri = n * (n + 1) // 2
        chans = []
        for c in range(n_ch):
            s = pos + 64 + c * slot_bytes
            fid, n_rec, n_tot, n_local, k, mean_auto = SLOT.unpack_from(data, s)
            ch = dict(
                freq_id=fid,
                n_cand_total=n_tot,
                n_local=n_local,
                k=k,
                mean_auto=mean_auto,
                cand=np.frombuffer(data, dt, n_rec, s + 64 + 8 * n_tri).copy(),
            )
            if with_v:
                ch["V"] = hermitian(np.frombuffer(data, np.complex64, n_tri, s + 64), n)
            chans.append(ch)
        yield dict(seq=seq, t_utc=t_utc, n=n, lane_conj=conj, channels=chans)
        pos += frame_bytes


def cmd_info(a):
    for path in a.files:
        print(path)
        for fr in frames(path, with_v=False):
            ch = fr["channels"]
            cand = [c["cand"] for c in ch]
            tested = sum(int((x["verdict"] >= 0).sum()) for x in cand)
            dup = sum(int((x["verdict"] == 0).sum()) for x in cand)
            held = sum(int((x["held_s"] > 0).sum()) for x in cand)
            fleet = sum(int((x["src"] == 1).sum()) for x in cand)
            print(
                "  seq %d t %.3f: %d channels, candidates %d recorded of %d (fleet %d, held "
                "%d), lanes %d, duplicates %d of %d tested"
                % (
                    fr["seq"],
                    fr["t_utc"],
                    len(ch),
                    sum(len(x) for x in cand),
                    sum(c["n_cand_total"] for c in ch),
                    fleet,
                    held,
                    sum(c["k"] for c in ch),
                    dup,
                    tested,
                )
            )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="per frame: time, channels, candidates, verdicts")
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_info)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
