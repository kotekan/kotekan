#!/usr/bin/env python3
"""Read cudaDtvSubtract's output and write its state files.

The stage (lib/cuda/cudaDtvSubtract.hpp) writes one frame per 8192 hops: the weights in use, the
per-input powers before and after the subtraction, |s|^2 and the lag correlations
R_xs,i(lag) = sum_t x_i(t) conj(s(t - lag)) and R_ss(d) = sum_t s(t) conj(s(t - d)) of each
source's timestream s (lib/cuda/cudaDtvSubtractKernel.hpp, DtvOutLayout). A state file gives it
the weights w and the impulse responses g per channel and source (DtvStateHdr, DtvStateRec).

    dtv_state.py info FILE...                         # per channel: sources, power removed
    dtv_state.py fit --source K --out st.bin [--state old.bin] [--freq F,F] [--first N] [--count M] FILE...
    dtv_state.py show st.bin

fit sums the correlations of source K over the selected frames (those with the source active and
the same weights), solves the least-squares normal equations per input,
sum_m g(lag_m) R_ss(lag_j - lag_m) = R_xs(lag_j), and writes a state with source K subtracting.
The other sources come from --state (default: as the frames show them, observe-only sources
without g are kept observe-only). The files are rawFileWrite output ([uint32 metadata size]
[metadata][frame]); the stage's are gated by gnss<g>_dtvdump_gate like the beam dump's.
"""
import argparse
import glob
import struct
import sys

import numpy as np

NI = 128
OUT_HDR = struct.Struct("<qiiiiiiii24x")  # 64 bytes
OUT_MAGIC, OUT_VERSION = 0x4F565444, 1
ST_HDR = struct.Struct("<8i")
ST_REC = struct.Struct("<4i")
ST_MAGIC, ST_VERSION = 0x53565444, 1


def align(off):
    return (off + 15) & ~15


def layout(n_chan, n_src, n_lag):
    """Section offsets of one output frame: mirrors gnss_cuda::dtv_out_layout."""
    L = {"freq": 64}
    L["act"] = align(L["freq"] + 4 * n_chan)
    L["pw_raw"] = align(L["act"] + 4 * n_chan * n_src)
    L["pw_res"] = align(L["pw_raw"] + 4 * NI * n_chan)
    L["ps"] = align(L["pw_res"] + 4 * NI * n_chan)
    L["w"] = align(L["ps"] + 4 * n_chan * n_src)
    L["rss"] = align(L["w"] + 8 * NI * n_chan * n_src)
    L["rxs"] = align(L["rss"] + 8 * n_lag * n_chan * n_src)
    L["bytes"] = align(L["rxs"] + 8 * n_lag * NI * n_chan * n_src)
    return L


def frames(paths):
    """Yield one dict per output frame, in file order."""
    for path in paths:
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
                n_lag,
                lag_min,
                n_src,
            ) = OUT_HDR.unpack_from(data, pos)
            if magic != OUT_MAGIC or version != OUT_VERSION or n_inputs != NI:
                raise SystemExit("%s: bad frame header at byte %d" % (path, pos))
            L = layout(n_chan, n_src, n_lag)
            if pos + L["bytes"] > len(data):
                print("%s: truncated frame at byte %d" % (path, pos), file=sys.stderr)
                break

            def arr(key, dtype, count, shape):
                return np.frombuffer(data, dtype, count, pos + L[key]).reshape(shape)

            yield dict(
                seq0=seq0,
                n_times=n_times,
                n_lag=n_lag,
                lag_min=lag_min,
                n_src=n_src,
                freq=arr("freq", np.int32, n_chan, (n_chan,)).copy(),
                act=arr("act", np.int32, n_chan * n_src, (n_chan, n_src)).copy(),
                pw_raw=arr("pw_raw", np.float32, NI * n_chan, (n_chan, NI)).copy(),
                pw_res=arr("pw_res", np.float32, NI * n_chan, (n_chan, NI)).copy(),
                ps=arr("ps", np.float32, n_chan * n_src, (n_chan, n_src)).copy(),
                w=arr(
                    "w", np.complex64, NI * n_chan * n_src, (n_chan, n_src, NI)
                ).copy(),
                rss=arr(
                    "rss", np.complex64, n_lag * n_chan * n_src, (n_chan, n_src, n_lag)
                ).copy(),
                rxs=arr(
                    "rxs",
                    np.complex64,
                    n_lag * NI * n_chan * n_src,
                    (n_chan, n_src, n_lag, NI),
                ).copy(),
            )
            pos += L["bytes"]


def expand(paths):
    out = []
    for p in paths:
        out += sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
    return out


def read_state(path):
    with open(path, "rb") as fh:
        data = fh.read()
    magic, version, n_chan, n_inputs, n_lag, lag_min, n_src, _ = ST_HDR.unpack_from(
        data, 0
    )
    if magic != ST_MAGIC or version != ST_VERSION or n_inputs != NI:
        raise SystemExit("%s: not a DTV state file" % path)
    pos = ST_HDR.size
    recs = {}
    for _ in range(n_chan):
        fid, n_set, subtract, _r = ST_REC.unpack_from(data, pos)
        pos += ST_REC.size
        w = np.frombuffer(data, np.complex64, n_src * NI, pos).reshape(n_src, NI).copy()
        pos += 8 * n_src * NI
        g = (
            np.frombuffer(data, np.complex64, n_src * n_lag * NI, pos)
            .reshape(n_src, n_lag, NI)
            .copy()
        )
        pos += 8 * n_src * n_lag * NI
        recs[fid] = dict(n_set=n_set, subtract=subtract, w=w, g=g)
    return dict(n_lag=n_lag, lag_min=lag_min, n_src=n_src, recs=recs)


def write_state(path, st):
    """st: n_lag, lag_min, n_src, recs {freq_id: n_set, subtract, w [n_src, NI], g [n_src, n_lag, NI]}."""
    with open(path, "wb") as fh:
        fh.write(
            ST_HDR.pack(
                ST_MAGIC,
                ST_VERSION,
                len(st["recs"]),
                NI,
                st["n_lag"],
                st["lag_min"],
                st["n_src"],
                0,
            )
        )
        for fid in sorted(st["recs"]):
            r = st["recs"][fid]
            fh.write(ST_REC.pack(int(fid), int(r["n_set"]), int(r["subtract"]), 0))
            fh.write(np.asarray(r["w"], np.complex64).tobytes())
            fh.write(np.asarray(r["g"], np.complex64).tobytes())


def solve_g(rxs, rss, lag_min):
    """g [n_lag, NI] from summed R_xs [n_lag, NI] and R_ss(d), d = 0..n_lag-1: the normal equations
    sum_m g_m T[m, j] = R_xs(lag_j) with T[m, j] = R_ss(lag_j - lag_m) (Hermitian Toeplitz)."""
    n_lag = rss.shape[0]
    d = np.arange(n_lag)[None, :] - np.arange(n_lag)[:, None]  # lag_j - lag_m
    T = np.where(d >= 0, rss[np.abs(d)], np.conj(rss[np.abs(d)])).astype(np.complex128)
    T += np.eye(n_lag) * 1e-9 * np.real(np.trace(T))
    return np.linalg.solve(T.T, rxs.astype(np.complex128)).astype(np.complex64)


def cmd_info(a):
    fr = list(frames(expand(a.files)))
    if not fr:
        raise SystemExit("no frames")
    f0 = fr[0]
    print(
        "%d frames, seq0 %d..%d, %d channels %s, %d source slots, lags %d..%d"
        % (
            len(fr),
            fr[0]["seq0"],
            fr[-1]["seq0"],
            len(f0["freq"]),
            f0["freq"].tolist(),
            f0["n_src"],
            f0["lag_min"],
            f0["lag_min"] + f0["n_lag"] - 1,
        )
    )
    for c, fid in enumerate(f0["freq"]):
        acts = np.array([f["act"][c] for f in fr])
        sub = [f for f in fr if (f["act"][c] >= 2).any()]
        line = "  freq_id %d (%.3f MHz): act per source (first/last frame) %s / %s" % (
            fid,
            fid * 0.1953125,
            acts[0].tolist(),
            acts[-1].tolist(),
        )
        if sub:
            raw = np.sum([f["pw_raw"][c] for f in sub], axis=0)
            res = np.sum([f["pw_res"][c] for f in sub], axis=0)
            ratio = res / np.maximum(raw, 1e-9)
            line += (
                "; subtracting in %d frames: residual / raw power per input median %.3f (p10 %.3f, p90 %.3f), total %.3f"
                % (
                    len(sub),
                    np.median(ratio),
                    np.percentile(ratio, 10),
                    np.percentile(ratio, 90),
                    res.sum() / raw.sum(),
                )
            )
        obs = [f for f in fr if (f["act"][c] >= 1).any()]
        if obs:
            ps = np.mean([f["ps"][c] for f in obs], axis=0) / f0["n_times"]
            line += "; mean |s|^2 per source %s" % " ".join("%.3g" % v for v in ps)
        print(line)


def cmd_fit(a):
    fr = list(frames(expand(a.files)))
    if not fr:
        raise SystemExit("no frames")
    fr = fr[a.first :] if a.count is None else fr[a.first : a.first + a.count]
    f0 = fr[0]
    n_lag, lag_min, n_src, k = f0["n_lag"], f0["lag_min"], f0["n_src"], a.source
    if not 0 <= k < n_src:
        raise SystemExit("--source %d outside 0..%d" % (k, n_src - 1))
    if a.state:
        st = read_state(a.state)
        if st["n_lag"] != n_lag or st["lag_min"] != lag_min or st["n_src"] != n_src:
            raise SystemExit(
                "%s: lag grid or source count differs from the frames'" % a.state
            )
    else:
        st = dict(n_lag=n_lag, lag_min=lag_min, n_src=n_src, recs={})
    only = {int(v) for v in a.freq.split(",")} if a.freq else None
    for c, fid in enumerate(f0["freq"]):
        if only is not None and int(fid) not in only:
            continue
        use = [f for f in fr if f["act"][c][k] >= 1]
        if not use:
            print(
                "freq_id %d: source %d not active in the selected frames, skipped"
                % (fid, k)
            )
            continue
        # one set of weights: the last frames' (an eigen start or a load changes them)
        wl = use[-1]["w"][c][k]
        use = [f for f in use if np.allclose(f["w"][c][k], wl, rtol=1e-6, atol=1e-9)]
        rxs = np.sum([f["rxs"][c][k] for f in use], axis=0)
        rss = np.sum([f["rss"][c][k] for f in use], axis=0)
        g = solve_g(rxs, rss, lag_min)
        r = st["recs"].get(int(fid))
        if r is None:
            r = dict(
                n_set=0,
                subtract=0,
                w=np.zeros((n_src, NI), np.complex64),
                g=np.zeros((n_src, n_lag, NI), np.complex64),
            )
            # the frames' other active sources, with their weights and no g (observe-only)
            for j in range(n_src):
                if use[-1]["act"][c][j] >= 1:
                    r["w"][j] = use[-1]["w"][c][j]
                    r["n_set"] = max(r["n_set"], j + 1)
            st["recs"][int(fid)] = r
        r["w"][k] = wl
        r["g"][k] = g
        # every source of a subtracting record subtracts: keep those with a fitted g (a source
        # above k with weights but no g would only subtract zero, so drop it)
        fitted = [j for j in range(n_src) if np.any(r["g"][j] != 0)]
        r["n_set"] = max(fitted) + 1
        r["subtract"] = 1
        prof = np.sum(np.abs(g) ** 2, axis=1)
        top = np.argsort(prof)[::-1][:5]
        expl = np.real(np.sum(np.conj(g) * rxs)) / max(
            np.sum([f["pw_raw"][c].sum() for f in use]), 1e-9
        )
        print(
            "freq_id %d: source %d fitted over %d frames; strongest lags (hops: sum |g|^2) %s; fraction of the "
            "input power the model explains %.3f"
            % (
                fid,
                k,
                len(use),
                " ".join("%+d:%.3g" % (lag_min + j, prof[j]) for j in top),
                expl,
            )
        )
    write_state(a.out, st)
    print("wrote %s: %d channels" % (a.out, len(st["recs"])))


def cmd_show(a):
    st = read_state(a.file)
    print(
        "%s: lags %d..%d, %d source slots, %d channels"
        % (
            a.file,
            st["lag_min"],
            st["lag_min"] + st["n_lag"] - 1,
            st["n_src"],
            len(st["recs"]),
        )
    )
    for fid, r in sorted(st["recs"].items()):
        print(
            "  freq_id %d: %d source(s) set, %s; |w| %s; sum |g|^2 %s"
            % (
                fid,
                r["n_set"],
                "subtracting" if r["subtract"] else "observe-only",
                " ".join("%.3g" % np.linalg.norm(r["w"][k]) for k in range(r["n_set"])),
                " ".join(
                    "%.3g" % np.sum(np.abs(r["g"][k]) ** 2) for k in range(r["n_set"])
                ),
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
    p = sub.add_parser("fit")
    p.add_argument(
        "--freq",
        help="freq_ids (comma-separated) to fit source K on; the others keep their --state "
        "record (an eigen start covers every channel of a GPU, a source is wanted only where a "
        "station is)",
    )
    p.add_argument("--source", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--state", help="state to start from (the other sources)")
    p.add_argument(
        "--first", type=int, default=0, help="first frame to use, in file order"
    )
    p.add_argument(
        "--count", type=int, help="frames to use (default: all from --first)"
    )
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_fit)
    p = sub.add_parser("show")
    p.add_argument("file")
    p.set_defaults(fn=cmd_show)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
