#!/usr/bin/env python3
"""Self-test for viscap_read: synthetic tiles + ctl frames with a known value in every cell,
written in rawFileWrite framing, decoded through load(), and checked cell by cell for the
mixed, AA and BB blocks (lane/element/tile addressing, Hermitian completion, ctl join).

    viscap_selftest.py            # exits 0 on PASS
"""
import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viscap_read as vr  # noqa: E402

N_REC, N_CHAN, N_PRN, N_LIVE, NUM_SYNTH, HOPS = 4, 2, 3, 32, 128, 2048
FREQ_IDS = [5972, 5988]


def cell(rec, chan, tile, i, j):
    """The value planted in tile cell (i, j) of (rec, chan): unique, int32-safe."""
    v = rec * 100000 + chan * 20000 + tile * 256 + i * 16 + j
    return v, -v


def build_frames(has_aa, has_bb, n_frames=2):
    n_tile, n_mixed, n_aa, _ = vr.tile_counts(N_LIVE, has_aa, has_bb, NUM_SYNTH)
    tiles_frames, ctl_frames = [], []
    for fr in range(n_frames):
        t = np.zeros((N_REC, N_CHAN, n_tile, 16, 16, 2), np.int32)
        for r in range(N_REC):
            for c in range(N_CHAN):
                for k in range(n_tile):
                    i, j = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
                    re, im = cell(r, c, k, i, j)
                    t[r, c, k, :, :, 0] = re
                    t[r, c, k, :, :, 1] = im
        tiles_frames.append(t.tobytes())
        hdr = np.zeros(1, vr.HDR)
        hdr["n_rec"], hdr["n_prn"], hdr["n_chan"], hdr["n_jobs"] = (
            N_REC,
            N_PRN,
            N_CHAN,
            4 * N_PRN,
        )
        hdr["seq0"] = 1000 + fr * 8192 * 16384
        hdr["utc0"] = 1.5e9
        win = np.array(
            [hdr["seq0"][0] + r * HOPS * 16384 for r in range(vr.MAX_REC)], "<i8"
        )
        ctl = np.zeros((vr.MAX_REC, N_PRN), vr.PRNCTL)
        for p in range(N_PRN):
            ctl["run"][:, p] = 1 if p != 1 else 0  # slot 1 idle
            ctl["prn"][:, p] = 10 + p
            ctl["job0"][:, p] = 4 * p if p != 1 else -1
            ctl["fcar_report"][:, p] = 100.0 * p
        energy = np.zeros((4 * N_PRN * vr.MAX_REC, N_CHAN))
        for p in range(N_PRN):
            for row in range(4):
                energy[4 * p + row, :] = (
                    2048.0 * (1 + p) * (1 + row)
                )  # rms = sqrt(E/hops)
        ctl_frames.append(
            hdr.tobytes() + win.tobytes() + ctl.tobytes() + energy.tobytes()
        )
    return tiles_frames, ctl_frames, n_tile, n_mixed, n_aa


def write_series(d, kind, frames):
    with open(os.path.join(d, "cx99_gnss0_e5a_%s_0000000.raw" % kind), "wb") as f:
        for fr in frames:
            f.write(struct.pack("<I", 0) + fr)


def write_configs(d, has_aa, has_bb):
    import yaml

    cfg = {
        "gnss0_e5a_n2dual": {
            "commands": [
                {
                    "name": "cudaGnssInject",
                    "hops_per_record": HOPS,
                    "channel_ids": FREQ_IDS,
                },
                {
                    "name": "cudaCorrelatorDual",
                    "num_live_elements": N_LIVE,
                    "num_synth": NUM_SYNTH,
                    "gnss_gather_aa": has_aa,
                    "gnss_gather_bb": has_bb,
                },
            ]
        }
    }
    node = {"array": {"live_element_ranges": [[0, 15], [64, 79]]}}
    cp, np_ = os.path.join(d, "cfg.yaml"), os.path.join(d, "node.yaml")
    yaml.safe_dump(cfg, open(cp, "w"))
    yaml.safe_dump(node, open(np_, "w"))
    return cp, np_


def check(has_aa, has_bb):
    d = tempfile.mkdtemp(prefix="viscap_selftest_")
    tiles, ctls, n_tile, n_mixed, n_aa = build_frames(has_aa, has_bb)
    write_series(d, "vistiles", tiles)
    write_series(d, "visctl", ctls)
    cp, np_ = write_configs(d, has_aa, has_bb)
    axes, dec = vr.load(d, "cx99", 0, cp, np_, tag="_e5a")
    fails = 0

    def expect(name, got, want):
        nonlocal fails
        if not np.array_equal(got, want):
            fails += 1
            print("  FAIL %s: %d cells differ" % (name, int((got != want).sum())))

    nlive16 = (N_LIVE + 15) // 16
    R = len(dec["winstart"])
    assert (
        R == 2 * N_REC and axes["has_bb"] == has_bb and axes["num_synth"] == NUM_SYNTH
    )
    # mixed: lane L = 4p + row at gi = 128 + L; tile ((gi>>4) - 8) * nlive16 + (e>>4)
    p_, row_, e_ = np.meshgrid(
        np.arange(N_PRN), np.arange(4), np.arange(N_LIVE), indexing="ij"
    )
    gi = 128 + 4 * p_ + row_
    k = ((gi >> 4) - 8) * nlive16 + (e_ >> 4)
    for rr in range(R):
        r, c = rr % N_REC, np.arange(N_CHAN)[:, None, None, None]
        re, im = cell(r, c, k[None], gi[None] & 15, e_[None] & 15)
        expect("vis_mixed rec %d" % rr, dec["vis_mixed"][rr], re + 1j * im)
    if has_aa:
        i_, j_ = np.meshgrid(np.arange(N_LIVE), np.arange(N_LIVE), indexing="ij")
        lo = np.minimum(i_, j_)
        hi = np.maximum(i_, j_)
        k1, k2 = hi >> 4, lo >> 4
        kt = n_mixed + k1 * (k1 + 1) // 2 + k2
        for rr in range(R):
            r, c = rr % N_REC, np.arange(N_CHAN)[:, None, None]
            re, im = cell(r, c, kt[None], hi[None] & 15, lo[None] & 15)
            v = re + 1j * im
            v = np.where(i_[None] >= j_[None], v, np.conj(v))  # upper = conj of lower
            expect("vis_aa rec %d" % rr, dec["vis_aa"][rr], v)
    else:
        assert dec["vis_aa"] is None
    if has_bb:
        a_, b_ = np.meshgrid(np.arange(NUM_SYNTH), np.arange(NUM_SYNTH), indexing="ij")
        lo = np.minimum(a_, b_)
        hi = np.maximum(a_, b_)
        k1, k2 = hi >> 4, lo >> 4
        kt = n_mixed + n_aa + k1 * (k1 + 1) // 2 + k2
        for rr in range(R):
            r, c = rr % N_REC, np.arange(N_CHAN)[:, None, None]
            re, im = cell(r, c, kt[None], hi[None] & 15, lo[None] & 15)
            v = re + 1j * im
            v = np.where(a_[None] >= b_[None], v, np.conj(v))
            expect("vis_bb rec %d" % rr, dec["vis_bb"][rr], v)
    else:
        assert dec["vis_bb"] is None
    # ctl join: PRNs, run flags, energies and the quantizer scale per slot
    expect("prn", dec["prn"][0], np.array([10, 11, 12], np.uint16))
    expect("run", dec["run"][0], np.array([1, 0, 1], np.uint8))
    s = dec["scale"][0]  # [n_prn, 4, n_chan]
    want = np.zeros_like(s)
    for p in (0, 2):
        for row in range(4):
            want[p, row, :] = 7.0 / (3.0 * np.sqrt((1 + p) * (1 + row)))
    if not np.allclose(s, want):
        fails += 1
        print("  FAIL scale")
    print(
        "  aa=%s bb=%s: %d tiles/chan, %d records -> %s"
        % (has_aa, has_bb, n_tile, R, "PASS" if fails == 0 else "FAIL")
    )
    return fails


def main():
    fails = sum(
        check(aa, bb) for aa, bb in ((False, False), (True, False), (True, True))
    )
    print("viscap_selftest: %s" % ("PASS" if fails == 0 else "FAIL (%d)" % fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
