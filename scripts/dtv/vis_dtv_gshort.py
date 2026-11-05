#!/usr/bin/env python3
"""How fast the DTV stage's fits go stale, from 0.5 s to hours, from the stage's own refit windows.

Each stage frame (8192 hops, ~42 ms) carries, per channel and source, R_xs(lag, input) and R_ss(lag). For any
impulse response g, the power it leaves on a set of frames is P_in - 2 Re<g, R_xs> + g^H T(R_ss) g, so the
extra residual of a stale g_a on frames b, cost(g_a, b) - cost(g_b, b) with g_b the fit on b itself, is exact
(no white-s assumption) and P_in cancels. It is reported as a fraction of the power the fresh fit removes.

    vis_dtv_gshort.py reduce --dir /var/tmp/kvand-dtvwin/cx19 --out gshort_cx19.npy [--procs 8]

reduce, per refit window (contiguous frames, ~10 s), GPU, channel and source:
  sums of R_xs and R_ss (any g can be costed on the window later), the window's even/odd fit noise,
  block pairs inside the window (blocks of 12, 24, 48 frames: 0.5-8 s apart), and a causal replay of an
  exponentially weighted rolling fit (time constant TAUS frames, re-solved every UPDATE frames) on the last
  part of the window, against a fit frozen at the start of that part.
"""
import argparse
import glob
import os
import struct
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.expanduser("~/gnss/dtv-wt/scripts/gnss"))
import dtv_state  # noqa: E402

FRAME_S = 8192 * 5.12e-6
BLOCKS = (12, 24, 48)
TAUS = (12, 24, 48, 96)
UPDATE = 24
TEST0 = 120  # rolling replay: frames before this one only train


def toeplitz(rss):
    n = rss.shape[0]
    d = np.arange(n)[None, :] - np.arange(n)[:, None]
    return np.where(d >= 0, rss[np.abs(d)], np.conj(rss[np.abs(d)])).astype(
        np.complex128
    )


def solve(sx, st):
    T = toeplitz(st)
    T += np.eye(T.shape[0]) * 1e-9 * np.real(np.trace(T))
    return np.linalg.solve(T.T, sx)


def cost(g, sx, st):
    """Residual power minus input power, summed over inputs: -2 Re<g, R_xs> + g^H T g."""
    y = toeplitz(st).T @ g
    return float(np.real(np.sum(y * np.conj(g))) - 2 * np.real(np.sum(g * np.conj(sx))))


def excess(ga, sx, st):
    """Extra residual of ga on (sx, st) over the fit to (sx, st) itself, as a fraction of the power that fit removes."""
    gb = solve(sx, st)
    cb = cost(gb, sx, st)
    return (cost(ga, sx, st) - cb) / max(-cb, 1e-30)


def index(d):
    """Files per GPU grouped into windows of contiguous frames: [(gpu, [paths])]."""
    out = []
    for gpu in (0, 1):
        files = sorted(glob.glob(os.path.join(d, "*_gnss%d_dtv_*.raw" % gpu)))
        cur, last = [], None
        for p in files:
            with open(p, "rb") as fh:
                head = fh.read(4)
                (msize,) = struct.unpack("<I", head)
                fh.seek(4 + msize)
                hdr = dtv_state.OUT_HDR.unpack(fh.read(64))
            seq0, n_chan, n_lag, n_src = hdr[0], hdr[3], hdr[6], hdr[8]
            rec = 4 + msize + dtv_state.layout(n_chan, n_src, n_lag)["bytes"]
            nf = os.path.getsize(p) // rec
            if last is not None and seq0 != last + 8192:
                out.append((gpu, cur))
                cur = []
            cur.append(p)
            last = seq0 + (nf - 1) * 8192
        if cur:
            out.append((gpu, cur))
    return out


def one_window(args):
    gpu, paths = args
    fr = list(dtv_state.frames(paths))
    nf = len(fr)
    first_n = sum(1 for _ in dtv_state.frames(paths[:1]))
    t0 = os.path.getmtime(paths[0]) - first_n * FRAME_S
    rec = dict(
        gpu=gpu, t0=t0, nf=nf, seq0=fr[0]["seq0"], fids=fr[0]["freq"].copy(), ch={}
    )
    if nf < 2 * max(BLOCKS):
        return rec
    for c, fid in enumerate(fr[0]["freq"]):
        for k in range(fr[0]["n_src"]):
            if any(f["act"][c][k] < 1 for f in fr) or any(
                not np.array_equal(f["w"][c][k], fr[-1]["w"][c][k]) for f in fr
            ):
                continue
            X = np.array([f["rxs"][c][k] for f in fr], np.complex128)  # [nf, n_lag, NI]
            S = np.array([f["rss"][c][k] for f in fr], np.complex128)  # [nf, n_lag]
            r = dict(
                sx=X.sum(0).astype(np.complex64),
                st=S.sum(0),
                eo=excess(
                    solve(X[0::2].sum(0), S[0::2].sum(0)),
                    X[1::2].sum(0),
                    S[1::2].sum(0),
                ),
            )
            for B in BLOCKS:
                nb = nf // B
                Xb = X[: nb * B].reshape(nb, B, *X.shape[1:]).sum(1)
                Sb = S[: nb * B].reshape(nb, B, -1).sum(1)
                gb = [solve(Xb[i], Sb[i]) for i in range(nb)]
                pair = [[] for _ in range(nb)]
                for a in range(nb):
                    for b in range(nb):
                        if a != b:
                            pair[abs(a - b)].append(excess(gb[a], Xb[b], Sb[b]))
                eo = [
                    excess(
                        solve(
                            X[i * B : (i + 1) * B : 2].sum(0),
                            S[i * B : (i + 1) * B : 2].sum(0),
                        ),
                        X[i * B + 1 : (i + 1) * B : 2].sum(0),
                        S[i * B + 1 : (i + 1) * B : 2].sum(0),
                    )
                    for i in range(nb)
                ]
                r["blk%d" % B] = np.array(
                    [np.mean(v) if v else np.nan for v in pair]
                )  # index = separation in blocks
                r["eo%d" % B] = float(np.mean(eo))
            # causal rolling replay on frames TEST0.., against the best fit to those frames
            xt, stt = X[TEST0:].sum(0), S[TEST0:].sum(0)
            gt = solve(xt, stt)
            ct = cost(gt, xt, stt)
            roll = []
            for tau in TAUS:
                lam = np.exp(-1.0 / tau)
                A = np.zeros_like(X[0])
                Z = np.zeros_like(S[0])
                tot, g = 0.0, None
                for j in range(nf):
                    if j >= TEST0:
                        if g is None or (j - TEST0) % UPDATE == 0:
                            g = solve(A, Z)
                        tot += cost(g, X[j], S[j])
                    A = lam * A + X[j]
                    Z = lam * Z + S[j]
                roll.append((tot - ct) / max(-ct, 1e-30))
            r["roll"] = np.array(roll)
            r["frozen"] = (
                cost(solve(X[:TEST0].sum(0), S[:TEST0].sum(0)), xt, stt) - ct
            ) / max(-ct, 1e-30)
            rec["ch"][(int(fid), k)] = r
    return rec


def cmd_reduce(a):
    wins = index(a.dir)
    print(
        "%d windows (%d frames or more: %d)" % (len(wins), 2 * max(BLOCKS), len(wins)),
        flush=True,
    )
    with Pool(a.procs) as pool:
        res = pool.map(one_window, wins, chunksize=1)
    res = [r for r in res if r["ch"]]
    np.save(a.out, np.array(res, dtype=object), allow_pickle=True)
    print("%d windows with fits -> %s" % (len(res), a.out))


def state_index(states, node):
    """Per GPU: [(mtime, path, {k: mtime of the fit that set g_k})] in time order."""
    out = {}
    for gpu in (0, 1):
        rows = []
        for p in glob.glob(os.path.join(states, node, "st_g%d_*.bin" % gpu)):
            b = os.path.basename(p)[:-4].split("_")  # st, gN, tag, sK|rK
            rows.append((os.path.getmtime(p), p, int(b[3][1])))
        rows.sort()
        fitted, idx = {}, []
        for mt, p, k in rows:
            fitted[k] = mt
            idx.append((mt, p, dict(fitted)))
        out[gpu] = idx
    return out


def cmd_cost(a):
    """Per node: the cost of the g the stage actually had loaded during each window, and of every earlier
    window's fit on every later window (same GPU, channel, source), with the fit-noise floors."""
    R = list(np.load(a.reduced, allow_pickle=True))
    sidx = state_index(a.states, a.node)
    cache = {}
    rows = []  # (kind, gpu, fid, k, t_a, t_b, age, value, floor)
    for gpu in (0, 1):
        W = sorted((r for r in R if r["gpu"] == gpu), key=lambda r: r["t0"])
        t0s = np.array([r["t0"] for r in W])
        for r in W:
            prior = [s for s in sidx[gpu] if s[0] < r["t0"] - 0.5]
            if not prior:
                continue
            mt, p, fitted = prior[-1]
            # a state's data time = the start of the window it was fitted on: the last window before it was written
            fitted = {
                k: t0s[max(np.searchsorted(t0s, v) - 1, 0)] for k, v in fitted.items()
            }
            if p not in cache:
                cache[p] = dtv_state.read_state(p)
            st = cache[p]
            for (fid, k), v in r["ch"].items():
                rs = st["recs"].get(fid)
                if rs is None or k not in fitted or not np.any(rs["g"][k]):
                    continue
                ex = excess(
                    rs["g"][k].astype(np.complex128),
                    v["sx"].astype(np.complex128),
                    v["st"],
                )
                rows.append(
                    (
                        "loaded",
                        gpu,
                        fid,
                        k,
                        fitted[k],
                        r["t0"],
                        r["t0"] - fitted[k],
                        ex,
                        v["eo"] / 2,
                    )
                )
        for i, ra in enumerate(W):
            for rb in W[i + 1 :]:
                if rb["t0"] - ra["t0"] > a.max_pair_h * 3600:
                    break
                for key, va in ra["ch"].items():
                    vb = rb["ch"].get(key)
                    if vb is None:
                        continue
                    ga = solve(va["sx"].astype(np.complex128), va["st"])
                    ex = excess(ga, vb["sx"].astype(np.complex128), vb["st"])
                    rows.append(
                        (
                            "pair",
                            gpu,
                            key[0],
                            key[1],
                            ra["t0"],
                            rb["t0"],
                            rb["t0"] - ra["t0"],
                            ex,
                            (va["eo"] + vb["eo"]) / 4,
                        )
                    )
    dt = np.dtype(
        [
            ("kind", "U6"),
            ("gpu", "i4"),
            ("fid", "i4"),
            ("k", "i4"),
            ("ta", "f8"),
            ("tb", "f8"),
            ("age", "f8"),
            ("value", "f8"),
            ("floor", "f8"),
        ]
    )
    out = np.array(rows, dtype=dt)
    np.save(a.out, out)
    print(
        "%s: %d loaded-state rows, %d window pairs -> %s"
        % (
            a.node,
            (out["kind"] == "loaded").sum(),
            (out["kind"] == "pair").sum(),
            a.out,
        )
    )


EDGE = {2714, 2743, 2744, 2898, 2928}
PERIODS = (("02:00-06:30Z", 2.0, 6.5, "-"), ("06:30-09:30Z", 6.5, 9.5, "--"))


def band(fid):
    return "ATSC 24" if fid < 2800 else "ATSC 30"


def cmd_plot(a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vis_dtv_plot import C_A24, C_A30, C_CLEAN

    col = {"ATSC 24": C_A24, "ATSC 30": C_A30}
    W = [r for f in a.reduced for r in np.load(f, allow_pickle=True)]
    C = np.concatenate([np.load(f) for f in a.costs])
    day0 = min(r["t0"] for r in W) // 86400 * 86400
    hr = lambda t: (np.asarray(t) - day0) / 3600.0  # noqa: E731
    C = C[~np.isin(C["fid"], list(EDGE))]
    cb = np.array([band(f) for f in C["fid"]])
    y_all = C["value"] - C["floor"]

    def within(k, b, h0, h1, B=24):
        rows = [
            v["blk%d" % B] - v["eo%d" % B] / 2
            for r in W
            if h0 <= hr(r["t0"]) < h1
            for (fid, kk), v in r["ch"].items()
            if kk == k and fid not in EDGE and band(fid) == b
        ]
        m = min(len(x) for x in rows)
        Y = np.array([x[:m] for x in rows])
        return np.arange(1, m) * B * FRAME_S, np.nanmedian(Y[:, 1:], axis=0)

    edges = np.geomspace(20, 12 * 3600, 22)

    def pairs(k, b, h0, h1):
        s_ = (
            (C["kind"] == "pair")
            & (C["k"] == k)
            & (cb == b)
            & (hr(C["ta"]) >= h0)
            & (hr(C["tb"]) < h1)
        )
        x, y = C["age"][s_], y_all[s_]
        med = [
            np.median(y[(x >= e0) & (x < e1)])
            if ((x >= e0) & (x < e1)).sum() >= 10
            else np.nan
            for e0, e1 in zip(edges[:-1], edges[1:])
        ]
        return np.sqrt(edges[1:] * edges[:-1]), np.array(med)

    def roll(k, b, h0, h1, key="roll"):
        v = [
            r["ch"][(fid, kk)][key]
            for r in W
            if h0 <= hr(r["t0"]) < h1
            for (fid, kk) in r["ch"]
            if kk == k and fid not in EDGE and band(fid) == b
        ]
        return np.median(np.array(v), axis=0)

    fig = plt.figure(figsize=(15, 11.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0, :])
    for b in ("ATSC 24", "ATSC 30"):
        for name, h0, h1, ls in PERIODS:
            xw, yw = within(0, b, h0, h1)
            xp, yp = pairs(0, b, h0, h1)
            x = np.concatenate([xw, xp])
            y = np.concatenate([yw, yp])
            ok = np.isfinite(y) & (y > 0)
            ax.plot(
                x[ok],
                y[ok],
                color=col[b],
                ls=ls,
                lw=2,
                marker="o",
                ms=3.5,
                label="%s, %s" % (b, name),
            )
            s_ = (
                (C["kind"] == "loaded")
                & (C["k"] == 0)
                & (cb == b)
                & (hr(C["tb"]) >= h0)
                & (hr(C["tb"]) < h1)
            )
            ax.scatter(
                C["age"][s_],
                np.maximum(y_all[s_], 1e-5),
                s=6,
                color=col[b],
                alpha=0.25,
                lw=0,
            )
            rr = roll(0, b, h0, h1)
            print(
                "%s %s source 0: within-window 1 s %.2e, 5 s %.2e, 9 s %.2e | window pairs %s | loaded g: median %.2e at "
                "median age %.0f s (r0 windows, age > 240 s: %.2e) | rolling tau 0.5/1/2/4 s %s; frozen 5 s %.2e"
                % (
                    b,
                    name,
                    yw[0],
                    yw[4],
                    yw[-1],
                    " ".join(
                        "%.0fs:%.2e" % (xx, yy)
                        for xx, yy in zip(xp, yp)
                        if np.isfinite(yy)
                    ),
                    np.median(y_all[s_]),
                    np.median(C["age"][s_]),
                    np.median(y_all[s_ & (C["age"] > 240)]),
                    " ".join("%.2e" % v for v in rr),
                    roll(0, b, h0, h1, "frozen"),
                )
            )
            if ls == "-":
                ax.axhline(
                    rr[1],
                    color=col[b],
                    lw=1.2,
                    ls=":",
                    label="%s: rolling fit, tau = 1 s, re-solved every second" % b,
                )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.7, 15 * 3600)
    ax.set_ylim(2e-5, 1.0)
    for xx, lab in ((60, "1 min"), (390, "refit loop\n(6.5 min)"), (3600, "1 h")):
        ax.axvline(xx, color=C_CLEAN, lw=0.6)
        ax.text(xx * 1.05, 0.5, lab, fontsize=8, color="#374151", va="top")
    ax.set_xlabel(
        "age of the fit (s): blocks inside one 10 s window, then window pairs"
    )
    ax.set_ylabel("extra residual / power the fresh fit removes")
    ax.set_title(
        "Source 0: what a fit of a given age costs, median over channels (dots: the g the stage actually had loaded); "
        "fit noise subtracted",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(color="#e5e7eb", lw=0.6, which="both")

    ax = fig.add_subplot(gs[1, 0])
    for b in ("ATSC 24", "ATSC 30"):
        for k, ls in ((1, "-"), (2, "--")):
            xw, yw = within(k, b, 2.0, 9.5)
            xp, yp = pairs(k, b, 2.0, 9.5)
            x, y = np.concatenate([xw, xp]), np.concatenate([yw, yp])
            ok = np.isfinite(y) & (y > 0)
            ax.plot(
                x[ok],
                y[ok],
                color=col[b],
                ls=ls,
                lw=1.8,
                marker="o",
                ms=3,
                label="%s, source %d" % (b, k),
            )
            print(
                "%s source %d 02:00-09:30Z: within-window 1 s %.2e, 9 s %.2e | pairs %s | rolling %s"
                % (
                    b,
                    k,
                    yw[0],
                    yw[-1],
                    " ".join(
                        "%.0fs:%.2e" % (xx, yy)
                        for xx, yy in zip(xp, yp)
                        if np.isfinite(yy)
                    ),
                    " ".join("%.2e" % v for v in roll(k, b, 2.0, 9.5)),
                )
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.7, 15 * 3600)
    ax.set_ylim(1e-4, 3.0)
    ax.set_xlabel("age of the fit (s)")
    ax.set_ylabel("extra residual / power the fresh fit removes")
    ax.set_title("Sources 1 and 2, 02:00-09:30Z", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(color="#e5e7eb", lw=0.6, which="both")

    ax = fig.add_subplot(gs[1, 1])
    taus = np.array(TAUS) * FRAME_S
    for b in ("ATSC 24", "ATSC 30"):
        for k, ls in ((0, "-"), (1, "--"), (2, ":")):
            ax.plot(
                taus,
                roll(k, b, 2.0, 9.5),
                color=col[b],
                ls=ls,
                lw=1.8,
                marker="o",
                ms=4,
                label="%s, source %d" % (b, k),
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(taus)
    ax.set_xticklabels(["%.1f" % t for t in taus])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(3e-5, 1.0)
    ax.set_xlabel(
        "rolling-fit time constant tau (s), re-solved every %.0f s" % (UPDATE * FRAME_S)
    )
    ax.set_ylabel("extra residual / power the best fit removes")
    ax.set_title(
        "Causal replay of a rolling fit on the last ~5 s of each window, 02:00-09:30Z",
        fontsize=10,
    )
    ax.legend(fontsize=8, ncol=2, loc="upper right", frameon=False)
    ax.grid(color="#e5e7eb", lw=0.6, which="both")
    fig.suptitle(
        "How fast the DTV fits go stale: stage refit windows of %s, 11-04"
        % ", ".join(a.nodes),
        fontsize=12,
    )
    fig.savefig(a.out, dpi=110)
    print("saved", a.out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("reduce")
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="first N windows only (test)")
    p = sp.add_parser("cost")
    p.add_argument("reduced")
    p.add_argument("--node", required=True)
    p.add_argument(
        "--states", default=os.path.expanduser("~/gnss/builds/dtv-live/states-rev2")
    )
    p.add_argument("--max-pair-h", type=float, default=12.0)
    p.add_argument("--out", required=True)
    p = sp.add_parser("plot")
    p.add_argument("--reduced", nargs="+", required=True)
    p.add_argument("--costs", nargs="+", required=True)
    p.add_argument("--nodes", nargs="+", default=["cx19", "cx51"])
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "plot":
        cmd_plot(a)
    elif a.cmd == "cost":
        cmd_cost(a)
    elif a.cmd == "reduce":
        if a.limit:
            global index
            full = index
            index = lambda d: full(d)[: a.limit]  # noqa: E731
        cmd_reduce(a)


if __name__ == "__main__":
    main()
