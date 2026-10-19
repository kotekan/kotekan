#!/usr/bin/env python3
"""Self-test for viscap_read: synthetic tiles + ctl frames with a known value in every cell,
written in rawFileWrite framing, decoded through load(), and checked cell by cell for the
mixed, AA and BB blocks (lane/element/tile addressing, Hermitian completion, ctl join). Two
layouts: one chain per correlator (gnss_gather_aa/bb, its own vistiles series), and the merged
capture gather production writes (gnss_gathers with 'cap': one capvistiles series per GPU over
the captured chains' lanes, from a nonzero gather lane base, and the union of their channels,
which need not be the same for every chain). viscap_triangle must refuse chains on different
channels.

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
NLIVE16 = (N_LIVE + 15) // 16

# Merged arms: (tag, local channels, freq_ids, synth_lane_base) per captured chain, 16 lanes
# each (N_PRN slots x 4 rows fit one tile row). The capture gather starts at the lowest
# captured base, above a chain at lane 0 it does not capture.
MERGED = {
    "equal channels": [
        ("", [3, 5], [5972, 5988], 16),
        ("_e5a", [3, 5], [5972, 5988], 32),
    ],
    # the second chain's channels are not in the gather's (ascending) order
    "unequal channels": [
        ("", [3, 5], [5972, 5988], 16),
        ("_e5a", [9, 5, 7], [6020, 5988, 6004], 32),
    ],
    # chains on disjoint channels share a lane range, as the generator lays them out
    "disjoint channels": [
        ("", [3, 5], [5972, 5988], 16),
        ("_e5b", [7, 9], [6132, 6148], 16),
    ],
}
CHAIN_LANES = 16


def cell(rec, chan, tile, i, j):
    """The value planted in tile cell (i, j) of (rec, chan): unique, int32-safe."""
    v = rec * 100000 + chan * 20000 + tile * 256 + i * 16 + j
    return v, -v


def tile_frames(n_chan, n_tile, n_frames=2):
    """Tiles frames [rec][chan][tile][16][16][2], chan the position on the tiles' own axis."""
    out = []
    for fr in range(n_frames):
        t = np.zeros((N_REC, n_chan, n_tile, 16, 16, 2), np.int32)
        for r in range(N_REC):
            for c in range(n_chan):
                for k in range(n_tile):
                    i, j = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
                    re, im = cell(r, c, k, i, j)
                    t[r, c, k, :, :, 0] = re
                    t[r, c, k, :, :, 1] = im
        out.append(t.tobytes())
    return out


def ctl_frames(n_chan, n_frames=2, prn0=10):
    """One chain's ctl frames: slot p holds PRN prn0 + p, slot 1 idle."""
    out = []
    for fr in range(n_frames):
        hdr = np.zeros(1, vr.HDR)
        hdr["n_rec"], hdr["n_prn"], hdr["n_chan"], hdr["n_jobs"] = (
            N_REC,
            N_PRN,
            n_chan,
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
            ctl["prn"][:, p] = prn0 + p
            ctl["job0"][:, p] = 4 * p if p != 1 else -1
            ctl["fcar_report"][:, p] = 100.0 * p
        energy = np.zeros((4 * N_PRN * vr.MAX_REC, n_chan))
        for p in range(N_PRN):
            for row in range(4):
                energy[4 * p + row, :] = (
                    2048.0 * (1 + p) * (1 + row)
                )  # rms = sqrt(E/hops)
        out.append(hdr.tobytes() + win.tobytes() + ctl.tobytes() + energy.tobytes())
    return out


def build_frames(has_aa, has_bb, n_frames=2):
    n_tile, n_mixed, n_aa, _ = vr.tile_counts(N_LIVE, has_aa, has_bb, NUM_SYNTH)
    return (
        tile_frames(N_CHAN, n_tile, n_frames),
        ctl_frames(N_CHAN, n_frames),
        n_tile,
        n_mixed,
        n_aa,
    )


def write_series(d, kind, frames, tag="_e5a"):
    with open(os.path.join(d, "cx99_gnss0%s_%s_0000000.raw" % (tag, kind)), "wb") as f:
        for fr in frames:
            f.write(struct.pack("<I", 0) + fr)


def write_node(d):
    import yaml

    node = {"array": {"live_element_ranges": [[0, 15], [64, 79]]}}
    np_ = os.path.join(d, "node.yaml")
    yaml.safe_dump(node, open(np_, "w"))
    return np_


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
    cp = os.path.join(d, "cfg.yaml")
    yaml.safe_dump(cfg, open(cp, "w"))
    return cp, write_node(d)


def write_merged_config(d, chains, cap_chans, cap_base, cap_lanes, has_bb):
    """The generator's merged gnss0_n2dual: every chain's injector at its lane base, one
    correlator with a gather per chain plus the capture gather 'cap'."""
    import yaml

    injects = [
        {
            "name": "cudaGnssInject",
            "gnss_ctl_name": "gnss0_l2c_n2ctl",
            "hops_per_record": HOPS,
            "gnss_local_channels": [3],
            "channel_ids": [5972],
            "synth_lane_base": 0,
        }
    ]
    for tag, local, fids, base in chains:
        injects.append(
            {
                "name": "cudaGnssInject",
                "gnss_ctl_name": "gnss0%s_n2ctl" % tag,
                "hops_per_record": HOPS,
                "gnss_local_channels": local,
                "channel_ids": fids,
                "synth_lane_base": base,
            }
        )
    corr = {
        "name": "cudaCorrelatorDual",
        "num_live_elements": N_LIVE,
        "num_synth": 384,
        "gnss_local_channels": cap_chans,
        "gnss_gathers": ["gps_l2c"]
        + ["chain%s" % t for t, _, _, _ in chains]
        + ["cap"],
        "gather_cap_channels": cap_chans,
        "gather_cap_lane_base": cap_base,
        "gather_cap_lanes": cap_lanes,
        "gather_cap_aa": True,
        "gather_cap_bb": has_bb,
    }
    cfg = {"gnss0_n2dual": {"commands": injects + [corr]}}
    cp = os.path.join(d, "cfg.yaml")
    yaml.safe_dump(cfg, open(cp, "w"))
    return cp, write_node(d)


def check_decoded(dec, chan_pos, lane_base, num_synth, n_mixed, n_aa, has_aa, has_bb):
    """Cell-by-cell check of one chain's decode. chan_pos[c] is the position of the chain's
    channel c on the tiles' channel axis, lane_base its first lane on the tiles' synth axis."""
    fails = 0

    def expect(name, got, want):
        nonlocal fails
        if got is None or got.shape != want.shape:
            fails += 1
            print(
                "  FAIL %s: shape %s, want %s"
                % (name, None if got is None else got.shape, want.shape)
            )
        elif not np.array_equal(got, want):
            fails += 1
            print("  FAIL %s: %d cells differ" % (name, int((got != want).sum())))

    R = len(dec["winstart"])
    if R != 2 * N_REC:
        print("  FAIL %d records decoded, want %d" % (R, 2 * N_REC))
        return fails + 1
    cpos = np.array(chan_pos)
    # mixed: lane L = lane_base + 4p + row at gi = 128 + L; tile ((gi>>4) - 8) * nlive16 +
    # (e>>4)
    p_, row_, e_ = np.meshgrid(
        np.arange(N_PRN), np.arange(4), np.arange(N_LIVE), indexing="ij"
    )
    gi = 128 + lane_base + 4 * p_ + row_
    k = ((gi >> 4) - 8) * NLIVE16 + (e_ >> 4)
    for rr in range(R):
        r, c = rr % N_REC, cpos[:, None, None, None]
        re, im = cell(r, c, k[None], gi[None] & 15, e_[None] & 15)
        expect("vis_mixed rec %d" % rr, dec["vis_mixed"][rr], re + 1j * im)
    if has_aa:
        i_, j_ = np.meshgrid(np.arange(N_LIVE), np.arange(N_LIVE), indexing="ij")
        lo = np.minimum(i_, j_)
        hi = np.maximum(i_, j_)
        k1, k2 = hi >> 4, lo >> 4
        kt = n_mixed + k1 * (k1 + 1) // 2 + k2
        for rr in range(R):
            r, c = rr % N_REC, cpos[:, None, None]
            re, im = cell(r, c, kt[None], hi[None] & 15, lo[None] & 15)
            v = re + 1j * im
            v = np.where(i_[None] >= j_[None], v, np.conj(v))  # upper = conj of lower
            expect("vis_aa rec %d" % rr, dec["vis_aa"][rr], v)
    elif dec["vis_aa"] is not None:
        fails += 1
        print("  FAIL vis_aa decoded without an AA block")
    if has_bb:
        a_, b_ = np.meshgrid(np.arange(num_synth), np.arange(num_synth), indexing="ij")
        lo = np.minimum(a_, b_)
        hi = np.maximum(a_, b_)
        k1, k2 = hi >> 4, lo >> 4
        kt = n_mixed + n_aa + k1 * (k1 + 1) // 2 + k2
        for rr in range(R):
            r, c = rr % N_REC, cpos[:, None, None]
            re, im = cell(r, c, kt[None], hi[None] & 15, lo[None] & 15)
            v = re + 1j * im
            v = np.where(a_[None] >= b_[None], v, np.conj(v))
            expect("vis_bb rec %d" % rr, dec["vis_bb"][rr], v)
    elif dec["vis_bb"] is not None:
        fails += 1
        print("  FAIL vis_bb decoded without a BB block")
    # ctl join: run flags and the quantizer scale per slot
    expect("run", dec["run"][0], np.array([1, 0, 1], np.uint8))
    s = dec["scale"][0]  # [n_prn, 4, n_chan]
    want = np.zeros((N_PRN, 4, len(chan_pos)))
    for p in (0, 2):
        for row in range(4):
            want[p, row, :] = 7.0 / (3.0 * np.sqrt((1 + p) * (1 + row)))
    if s.shape != want.shape or not np.allclose(s, want):
        fails += 1
        print("  FAIL scale")
    return fails


def check(has_aa, has_bb):
    d = tempfile.mkdtemp(prefix="viscap_selftest_")
    tiles, ctls, n_tile, n_mixed, n_aa = build_frames(has_aa, has_bb)
    write_series(d, "vistiles", tiles)
    write_series(d, "visctl", ctls)
    cp, np_ = write_configs(d, has_aa, has_bb)
    axes, dec = vr.load(d, "cx99", 0, cp, np_, tag="_e5a")
    assert axes["has_bb"] == has_bb and axes["num_synth"] == NUM_SYNTH
    fails = check_decoded(
        dec, range(N_CHAN), 0, NUM_SYNTH, n_mixed, n_aa, has_aa, has_bb
    )
    if not np.array_equal(dec["prn"][0], np.array([10, 11, 12], np.uint16)):
        fails += 1
        print("  FAIL prn")
    print(
        "  one chain, aa=%s bb=%s: %d tiles/chan -> %s"
        % (has_aa, has_bb, n_tile, "PASS" if fails == 0 else "FAIL")
    )
    return fails


def write_merged_capture(d, chains, has_bb):
    """One capvistiles series over the union of the chains' channels and lanes, one ctl
    series per chain, and the merged config -> (config, node yaml, the capture's geometry)."""
    cap_chans = sorted(set().union(*[set(local) for _, local, _, _ in chains]))
    cap_base = min(b for _, _, _, b in chains)
    cap_lanes = max(b for _, _, _, b in chains) + CHAIN_LANES - cap_base
    # The capture gather's tile count per channel, as cudaCorrelatorDual's
    # build_tile_selection lays it out: mixed rows x live columns, the AA triangle, the BB
    # triangle over the gather's rows.
    n_rows = cap_lanes // 16
    n_mixed = n_rows * NLIVE16
    n_aa = NLIVE16 * (NLIVE16 + 1) // 2
    n_tile = n_mixed + n_aa + (n_rows * (n_rows + 1) // 2 if has_bb else 0)
    write_series(d, "capvistiles", tile_frames(len(cap_chans), n_tile), tag="")
    for ci, (tag, local, _, _) in enumerate(chains):
        write_series(d, "visctl", ctl_frames(len(local), prn0=10 + 20 * ci), tag=tag)
    cp, np_ = write_merged_config(d, chains, cap_chans, cap_base, cap_lanes, has_bb)
    geo = dict(chans=cap_chans, base=cap_base, lanes=cap_lanes)
    geo.update(n_tile=n_tile, n_mixed=n_mixed, n_aa=n_aa)
    return cp, np_, geo


def check_merged(name, has_bb):
    """Merged capture: each chain decoded through its own ctl series."""
    chains = MERGED[name]
    d = tempfile.mkdtemp(prefix="viscap_selftest_")
    cp, np_, geo = write_merged_capture(d, chains, has_bb)
    fails = 0
    for ci, (tag, local, fids, base) in enumerate(chains):
        axes, dec = vr.load(d, "cx99", 0, cp, np_, tag=tag)
        if list(axes["freq_id"]) != fids or axes["num_synth"] != geo["lanes"]:
            fails += 1
            print(
                "  FAIL chain '%s' axes: freq_id %s num_synth %d"
                % (tag, list(axes["freq_id"]), axes["num_synth"])
            )
        f = check_decoded(
            dec,
            [geo["chans"].index(c) for c in local],
            base - geo["base"],
            geo["lanes"],
            geo["n_mixed"],
            geo["n_aa"],
            True,
            has_bb,
        )
        want_prn = np.array([10 + 20 * ci + p for p in range(N_PRN)], np.uint16)
        if not np.array_equal(dec["prn"][0], want_prn):
            f += 1
            print("  FAIL chain '%s' prn %s" % (tag, dec["prn"][0]))
        print(
            "  merged, %s, chain '%s' (channels %s of %s, lanes from %d), bb=%s: %d "
            "tiles/chan -> %s"
            % (
                name,
                tag,
                local,
                geo["chans"],
                base - geo["base"],
                has_bb,
                geo["n_tile"],
                "PASS" if f == 0 else "FAIL",
            )
        )
        fails += f
    return fails


def check_triangle_refuses():
    """viscap_triangle has one channel axis for all its --tags chains, so chains on different
    channels must be refused. Equal channel counts: nothing fails on a shape."""
    import viscap_triangle

    d = tempfile.mkdtemp(prefix="viscap_selftest_")
    cp, np_, _ = write_merged_capture(d, MERGED["disjoint channels"], False)
    out = os.path.join(d, "tri.h5")
    argv = sys.argv
    sys.argv = ["viscap_triangle.py", d] + "--node cx99 --gpu 0 --tags ,_e5b".split()
    sys.argv += ["--config", cp, "--node-yaml", np_, "--to-h5", out]
    try:
        viscap_triangle.main()
        code = 0
    except SystemExit as e:
        code = e.code
    finally:
        sys.argv = argv
    ok = code not in (None, 0) and not os.path.exists(out)
    print(
        "  triangle, chains on different channels: %s -> %s"
        % ("refused" if code else "wrote %s" % out, "PASS" if ok else "FAIL")
    )
    return 0 if ok else 1


def main():
    fails = sum(
        check(aa, bb) for aa, bb in ((False, False), (True, False), (True, True))
    )
    fails += sum(
        check_merged(name, bb)
        for name, bb in (
            ("equal channels", True),
            ("unequal channels", False),
            ("unequal channels", True),
            ("disjoint channels", True),
        )
    )
    fails += check_triangle_refuses()
    print("viscap_selftest: %s" % ("PASS" if fails == 0 else "FAIL (%d)" % fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
