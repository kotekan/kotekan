#!/usr/bin/env python3
"""Pack a visibility capture into the (N+M)(N+M+1)/2 triangle per record and channel.

    viscap_triangle.py <dir> --node cx19 --gpu 0 --tags ,_e5a,_b2a --config <node yaml> \
        --node-yaml config/chord_gnss_node.yaml --to-h5 cx19_gnss0_tri.h5 [--p-only] [--raw]

The axis is [the N live elements | the M synth lanes of the capture frame]. V[i, j] = x_i conj(x_j)
with x an element voltage or a replica lane: the N^2 block from the AA tiles, the lanes x
elements block from the mixed tiles (synth on the row side, as the correlator writes it), the
replica x replica block from the BB tiles. Blocks the capture did not gather are NaN.

Each --tags entry names a chain whose ctl series sits beside the tiles (the primary is ''),
and its ctl gives the PRN, run flag and 4-bit quantizer scale of its own lanes per record.
By default every lane and the replica block are divided by those scales (synth lanes then
carry the replica's true pre-quantization amplitude); --raw keeps tile units. Lanes no ctl
covers (idle slots beyond a chain's list) are zero in the tiles and marked chain '' here.
--p-only keeps one lane per slot (the prompt), dropping E, L and P_HEAD.

Output (h5): tri[R, C, K] complex64, K = M(M+1)/2, the UPPER triangle (i <= j) in row-major
order over the M = N + L axis; axis_kind[M] ('E' element / 'L' lane), axis_id[M] (element id
or lane index); lane_chain[L], lane_slot[L], lane_row[L] (0 E, 1 P, 2 L, 3 P_HEAD); per record
lane_prn[R, L], lane_run[R, L]; winstart[R] (absolute ADC sample); freq_id[C]. Processed a
few files at a time, so a whole capture streams through.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viscap_read as vr  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dir")
    ap.add_argument("--node", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument(
        "--tags",
        default="",
        help="chain tags with a ctl series, comma list, '' = "
        "the primary (default: the primary only)",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--node-yaml",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "config",
            "chord_gnss_node.yaml",
        ),
    )
    ap.add_argument("--to-h5", required=True)
    ap.add_argument(
        "--files", default=None, help="file index range lo:hi (default: all)"
    )
    ap.add_argument(
        "--chunk", type=int, default=2, help="files per pass (default 2 = ~48 frames)"
    )
    ap.add_argument(
        "--p-only", action="store_true", help="one lane per slot: the prompt"
    )
    ap.add_argument(
        "--raw", action="store_true", help="keep tile units (no quantizer scale)"
    )
    a = ap.parse_args()
    tags = a.tags.split(",")
    import h5py

    # the lane geometry from the first chain's view of the frame
    freq_ids, elems, hops, has_aa0 = vr.config_axes(
        a.config, a.gpu, tags[0], a.node_yaml
    )
    lanes0 = vr.config_lanes(a.config, a.gpu, tags[0])
    if not lanes0["merged"] and len(tags) > 1:
        sys.exit(
            "one correlator pass per chain in this capture: each chain is its own "
            "(N+M) triangle with its own tiles series -- run once per --tags entry"
        )
    L, N = lanes0["num_synth"], len(elems)
    has_aa, has_bb = has_aa0 or lanes0["has_aa"], lanes0["has_bb"]
    n_files = len(vr.series(a.dir, a.node, a.gpu, tags[0], "visctl"))
    lo, hi = (0, n_files) if a.files is None else [int(x) for x in a.files.split(":")]

    # lane table: chain, slot, row per lane, from each chain's lane base and slot count
    lane_chain = np.array([""] * L, dtype=object)
    lane_slot = np.full(L, -1, np.int32)
    lane_row = np.full(L, -1, np.int32)
    chain_base = {}
    for t in tags:
        ln = vr.config_lanes(a.config, a.gpu, t)
        chain_base[t] = ln["lane_base"]
    keep = np.ones(L, bool)
    axis_lane = np.arange(L)
    if a.p_only:
        keep[:] = False
    M = (
        N + int(keep.sum()) if not a.p_only else None
    )  # set once the slot counts are known

    with h5py.File(a.to_h5, "w") as f:
        f["freq_id"] = np.array(freq_ids, np.int32)
        f["element_id"] = np.array(elems, np.int32)
        f.attrs.update(
            node=a.node,
            gpu=a.gpu,
            hops_per_record=hops,
            sample_rate_hz=3.2e9,
            scaled=not a.raw,
            p_only=a.p_only,
            orientation="V[i, j] = x_i * conj(x_j); lanes rows carry synth * conj(E)",
            packing="upper triangle i <= j, row-major over [elements | lanes]",
        )
        tri = None
        R_total = 0
        for f0 in range(lo, hi, a.chunk):
            sl = slice(f0, min(hi, f0 + a.chunk))
            views = {}
            for t in tags:
                axes_t, dec_t = vr.load(
                    a.dir, a.node, a.gpu, a.config, a.node_yaml, tag=t, files=sl
                )
                views[t] = (axes_t, dec_t)
            ax0, d0 = views[tags[0]]
            R, C = d0["winstart"].shape[0], len(freq_ids)
            if tri is None:
                # lanes each chain owns: its slots x 4 rows from its lane base
                for t in tags:
                    n_prn = views[t][1]["prn"].shape[1]
                    for p in range(n_prn):
                        for r in range(4):
                            ln = chain_base[t] + 4 * p + r
                            lane_chain[ln], lane_slot[ln], lane_row[ln] = (
                                t or "primary",
                                p,
                                r,
                            )
                            keep[ln] = (r == 1) if a.p_only else True
                axis_lane = np.where(keep)[0]
                Lk = len(axis_lane)
                M = N + Lk
                iu = np.triu_indices(M)
                K = len(iu[0])
                f["axis_kind"] = np.array(["E"] * N + ["L"] * Lk, dtype="S1")
                f["axis_id"] = np.concatenate(
                    [np.array(elems, np.int32), axis_lane.astype(np.int32)]
                )
                f["lane_chain"] = np.array(
                    [str(x) for x in lane_chain[axis_lane]], dtype="S16"
                )
                f["lane_slot"] = lane_slot[axis_lane]
                f["lane_row"] = lane_row[axis_lane]
                tri = f.create_dataset(
                    "tri",
                    shape=(0, C, K),
                    maxshape=(None, C, K),
                    dtype=np.complex64,
                    chunks=(4, C, K),
                    compression="lzf",
                )
                lane_prn = f.create_dataset(
                    "lane_prn", shape=(0, Lk), maxshape=(None, Lk), dtype=np.uint16
                )
                lane_run = f.create_dataset(
                    "lane_run", shape=(0, Lk), maxshape=(None, Lk), dtype=np.uint8
                )
                win = f.create_dataset(
                    "winstart", shape=(0,), maxshape=(None,), dtype=np.int64
                )
                print(
                    "axis: %d elements + %d lanes of %d (%s); %d triangle entries per record "
                    "and channel; AA %s, BB %s"
                    % (
                        N,
                        Lk,
                        L,
                        "P only" if a.p_only else "E/P/L/PH",
                        K,
                        has_aa,
                        has_bb,
                    )
                )

            # the full Hermitian matrix for this chunk, then the packed triangle
            V = np.full((R, C, M, M), np.nan, np.complex64)
            if has_aa and d0["vis_aa"] is not None:
                V[:, :, :N, :N] = d0["vis_aa"]
            # lanes x lanes: the whole synth axis, scaled per lane where a ctl knows the scale
            scale = np.ones((R, C, L), np.float32)  # per lane
            prn_r = np.zeros((R, L), np.uint16)
            run_r = np.zeros((R, L), np.uint8)
            for t in tags:
                dt = views[t][1]
                n_prn = dt["prn"].shape[1]
                for p in range(n_prn):
                    for r in range(4):
                        ln = chain_base[t] + 4 * p + r
                        prn_r[:, ln] = dt["prn"][:, p]
                        run_r[:, ln] = dt["run"][:, p]
                        s = dt["scale"][:, p, r, :]  # [R, C]
                        scale[:, :, ln] = np.where(s > 0, s, 1.0)
            if a.raw:
                scale[:] = 1.0
            if has_bb and d0["vis_bb"] is not None:
                bb = d0["vis_bb"][:, :, axis_lane][:, :, :, axis_lane]
                sk = scale[:, :, axis_lane]
                V[:, :, N:, N:] = bb / (sk[:, :, :, None] * sk[:, :, None, :])
            else:
                V[:, :, N:, N:] = np.nan
            # lanes x elements: each chain's mixed block at its lanes; idle lanes stay zero
            mixed = np.zeros((R, C, L, N), np.complex64)
            for t in tags:
                dt = views[t][1]
                n_prn = dt["prn"].shape[1]
                base = chain_base[t]
                mixed[:, :, base : base + 4 * n_prn, :] = dt["vis_mixed"].reshape(
                    R, C, 4 * n_prn, N
                )
            mk = mixed[:, :, axis_lane, :] / scale[:, :, axis_lane][:, :, :, None]
            V[:, :, N:, :N] = mk
            V[:, :, :N, N:] = np.conj(np.swapaxes(mk, 2, 3))
            packed = V[:, :, iu[0], iu[1]]
            tri.resize(R_total + R, axis=0)
            tri[R_total:] = packed
            lane_prn.resize(R_total + R, axis=0)
            lane_prn[R_total:] = prn_r[:, axis_lane]
            lane_run.resize(R_total + R, axis=0)
            lane_run[R_total:] = run_r[:, axis_lane]
            win.resize(R_total + R, axis=0)
            win[R_total:] = d0["winstart"]
            R_total += R
            print("  files %d..%d: %d records" % (sl.start, sl.stop - 1, R_total))
        f.attrs["utc0_sample0"] = float(ax0["utc0"])
    print(
        "wrote %s: %d records x %d channels x %d entries"
        % (a.to_h5, R_total, len(freq_ids), K)
    )


if __name__ == "__main__":
    main()
