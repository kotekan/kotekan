#!/usr/bin/env python3
"""How the DTV stage's fitted impulse responses g drift over a night, from the saved state files.

A refit keeps the beam weights w (set by the night's eigen starts) and re-solves g_k(lag, input) for
source k: the response of each input to the beam timestream s_k. For a fixed w, g changes only when
the relation between the inputs and the beam changes: per-input gains/phases (analog chain), the
multipath (lag shape), or the beam's DTV-to-noise ratio (common scale).

    vis_dtv_gdrift.py extract --states DIR --date 20261104 --until 1300 --out gdrift_20261104.npz [nodes...]
    vis_dtv_gdrift.py plot gdrift_20261104.npz --out dtv_gdrift_20261104.png

extract, per node / channel / source, over the refits of that source (tags HHMM_sK = setup, HHMMSS_rK):
  t, E = ||g||^2, lag profile sum_i |g(lag, i)|^2, per-input complex gain vs the setup (beta),
  and for every pair of refits (a < b): the change ||g_b - g_a||^2 / ||g_b||^2, and what is left of it
  after the best common complex scale and after the best per-input complex gain (fit of g_b from g_a).
"""
import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.expanduser("~/gnss/dtv-wt/scripts/gnss"))
import dtv_state  # noqa: E402

TAG = re.compile(r"st_g(\d)_(\d{4})_s(\d)\.bin$|st_g(\d)_(\d{6})_r(\d)\.bin$")


def parse(path, day0):
    m = TAG.search(os.path.basename(path))
    if not m:
        return None
    if m.group(1) is not None:
        g, hms, k, setup = int(m.group(1)), m.group(2) + "00", int(m.group(3)), True
    else:
        g, hms, k, setup = int(m.group(4)), m.group(5), int(m.group(6)), False
    t = day0 + int(hms[:2]) * 3600 + int(hms[2:4]) * 60 + int(hms[4:6])
    return g, t, k, setup


def one_node(args):
    states, node, day0, until = args
    files = []
    for p in glob.glob(os.path.join(states, node, "st_g*_*.bin")):
        r = parse(p, day0)
        if r and r[1] <= until:
            files.append((r[1], r[0], r[2], r[3], p))
    files.sort()
    out = []
    for gpu in (0, 1):
        for k in (0, 1, 2):
            sel = [f for f in files if f[1] == gpu and f[2] == k]
            # the setup state of source k first, then its refits in time order
            sel = [f for f in sel if f[3]][-1:] + [f for f in sel if not f[3]]
            if len(sel) < 3:
                continue
            ts, G, W = [], {}, {}
            for t, _, _, _, p in sel:
                st = dtv_state.read_state(p)
                ts.append(t)
                for fid, r in st["recs"].items():
                    G.setdefault(fid, []).append(r["g"][k])
                    W.setdefault(fid, []).append(r["w"][k])
            ts = np.array(ts, np.float64)
            for fid in sorted(G):
                g = np.array(G[fid], np.complex128)  # [T, n_lag, NI]
                w = np.array(W[fid], np.complex128)
                if len(g) != len(ts) or not np.any(g[0]):
                    continue
                T = len(ts)
                flat = g.reshape(T, -1)
                gram = flat.conj() @ flat.T  # <g_a, g_b>
                e = np.real(np.diag(gram))
                d_tot = (e[None, :] + e[:, None] - 2 * np.real(gram)) / e[
                    None, :
                ]  # [a, b] / ||g_b||^2
                d_scal = 1 - np.abs(gram) ** 2 / (e[:, None] * e[None, :])
                gi = np.einsum("aln,bln->nab", g.conj(), g)  # per-input <g_a,i, g_b,i>
                ei = np.real(np.einsum("naa->na", gi))  # [NI, T]
                with np.errstate(invalid="ignore", divide="ignore"):
                    left = np.where(
                        ei[:, :, None] > 0,
                        ei[:, None, :] - np.abs(gi) ** 2 / ei[:, :, None],
                        ei[:, None, :],
                    )
                d_inp = np.sum(left, axis=0) / e[None, :]
                beta = np.where(
                    ei[:, 0:1] > 0, gi[:, 0, :] / np.maximum(ei[:, 0:1], 1e-30), 0
                ).T  # [T, NI] vs the setup
                out.append(
                    dict(
                        node=node,
                        gpu=gpu,
                        fid=int(fid),
                        k=k,
                        t=ts,
                        E=e,
                        lagp=np.sum(np.abs(g) ** 2, axis=2),
                        d_tot=d_tot,
                        d_scal=d_scal,
                        d_inp=d_inp,
                        beta=beta.astype(np.complex64),
                        ein=ei[:, 0],
                        wdev=float(
                            np.max(np.abs(w - w[0])) / max(np.max(np.abs(w[0])), 1e-30)
                        ),
                    )
                )
    return out


def cmd_extract(a):
    day0 = np.datetime64(
        "%s-%s-%sT00:00" % (a.date[:4], a.date[4:6], a.date[6:]), "s"
    ).astype(np.int64)
    until = day0 + int(a.until[:2]) * 3600 + int(a.until[2:]) * 60
    nodes = a.nodes or sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(a.states, "cx*"))
    )
    with Pool(min(len(nodes), a.procs)) as pool:
        res = [
            r
            for rs in pool.map(one_node, [(a.states, n, day0, until) for n in nodes])
            for r in rs
        ]
    np.save(a.out, np.array(res, dtype=object), allow_pickle=True)
    print("%d channel-source series from %s -> %s" % (len(res), " ".join(nodes), a.out))
    for r in res:
        if r["wdev"] > 1e-6:
            print(
                "WARNING %s fid %d source %d: w changed across the series (max rel %.2g)"
                % (r["node"], r["fid"], r["k"], r["wdev"])
            )


def cmd_gfull(a):
    """The full g series of a few channels (one node, one GPU, one source): setup + refits up to --until."""
    day0 = np.datetime64(
        "%s-%s-%sT00:00" % (a.date[:4], a.date[4:6], a.date[6:]), "s"
    ).astype(np.int64)
    until = day0 + int(a.until[:2]) * 3600 + int(a.until[2:]) * 60
    sel = []
    for p in glob.glob(os.path.join(a.states, a.node, "st_g%d_*.bin" % a.gpu)):
        r = parse(p, day0)
        if r and r[2] == a.source and r[1] <= until:
            sel.append((r[1], r[3], p))
    sel.sort()
    sel = [f for f in sel if f[1]][-1:] + [f for f in sel if not f[1]]
    fids = [int(f) for f in a.fids.split(",")]
    ts, g, lag_min = [], {f: [] for f in fids}, None
    for t, _, p in sel:
        st = dtv_state.read_state(p)
        lag_min = st["lag_min"]
        ts.append(t)
        for f in fids:
            g[f].append(st["recs"][f]["g"][a.source])
    np.savez(
        a.out,
        t=np.array(ts),
        fids=np.array(fids),
        lag_min=lag_min,
        node=a.node,
        gpu=a.gpu,
        source=a.source,
        **{"g%d" % f: np.array(g[f]) for f in fids}
    )
    print(
        "%s gpu%d source %d: %d states of %s -> %s"
        % (a.node, a.gpu, a.source, len(ts), fids, a.out)
    )


HOP_US = 5.12
EDGE = {
    2714,
    2743,
    2744,
    2898,
    2928,
}  # band-edge channels: weak DTV, left out of the medians
EPISODE = (
    8 + 5 / 60,
    8 + 50 / 60,
)  # 11-04: both stations' residual ~9.8 despite full refits
TRANSIT = 10 + 38 / 60  # Tau A, 11-04


def band(f):
    return "ATSC 24" if f < 2800 else "ATSC 30"


def marks(ax, horizontal=False):
    if horizontal:  # over a waterfall: lines on top
        for y in EPISODE:
            ax.axhline(y, color="#111827", lw=0.8, ls="--", zorder=5)
        ax.axhline(TRANSIT, color="#111827", lw=0.8, ls=":", zorder=5)
        return
    ax.axvspan(*EPISODE, color="#e5e7eb", alpha=0.8, lw=0, zorder=0)
    ax.axvline(TRANSIT, color="#111827", lw=0.8, ls=":")


def cmd_plot(a):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vis_dtv_plot import C_A24, C_A30, C_CLEAN

    col = {"ATSC 24": C_A24, "ATSC 30": C_A30}
    R = [r for r in np.load(a.summary, allow_pickle=True) if r["fid"] not in EDGE]
    day0 = R[0]["t"][0] // 86400 * 86400
    hr = lambda t: (np.asarray(t) - day0) / 3600.0  # noqa: E731

    # ---- figure 1: the fits themselves (one node/GPU, source 0)
    G = np.load(a.gfull)
    th = hr(G["t"])
    lags = (int(G["lag_min"]) + np.arange(G["g%d" % G["fids"][0]].shape[1])) * HOP_US
    nf = len(G["fids"])
    fig = plt.figure(figsize=(14, 5.2 * nf), constrained_layout=True)
    gs = fig.add_gridspec(nf, 2)
    norm = matplotlib.colors.LogNorm(1e-4, 0.3)
    for row, f in enumerate(G["fids"]):
        g = G["g%d" % f].astype(np.complex128)
        e0 = np.sum(np.abs(g[0]) ** 2)
        for c, (Z, title) in enumerate(
            (
                (
                    np.sum(np.abs(g) ** 2, axis=2) / e0,
                    "the fit: power of g per lag, summed over inputs",
                ),
                (
                    np.sum(np.abs(g - g[0]) ** 2, axis=2) / e0,
                    "its change since the 01:41Z setup, per lag",
                ),
            )
        ):
            ax = fig.add_subplot(gs[row, c])
            yy = np.append(th, th[-1] + np.median(np.diff(th)))
            im = ax.pcolormesh(
                np.append(lags, lags[-1] + HOP_US) - HOP_US / 2,
                yy,
                np.maximum(Z, 1e-6),
                cmap="Blues",
                norm=norm,
                shading="flat",
                rasterized=True,
            )
            marks(ax, horizontal=True)
            ax.set_ylim(th[0], yy[-1])
            ax.set_xlabel("lag (us; 1 hop = %.2f us)" % HOP_US)
            ax.set_ylabel("UTC hour, 11-04")
            ax.set_title(
                "%s gpu%s freq_id %d (%s), source %s: %s"
                % (G["node"], G["gpu"], f, band(f), G["source"], title),
                fontsize=10,
            )
            fig.colorbar(
                im, ax=ax, shrink=0.85, pad=0.01, label="fraction of ||g(setup)||^2"
            )
    fig.suptitle(
        "DTV stage fits through the night: dashed = 08:05-08:50Z high-residual episode, dotted = Tau A transit",
        fontsize=12,
    )
    fig.savefig(a.out_fits, dpi=110)
    print("saved", a.out_fits)

    # ---- figure 2: how fast the fits drift, every node and channel
    fig = plt.figure(figsize=(15, 13), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15])
    ax = fig.add_subplot(gs[0, :])
    for b in ("ATSC 24", "ATSC 30"):
        rs = [r for r in R if r["k"] == 0 and band(r["fid"]) == b]
        t = hr(rs[0]["t"])
        D = np.array([r["d_tot"][0, :] for r in rs])
        ax.fill_between(
            t,
            np.percentile(D, 25, axis=0),
            np.percentile(D, 75, axis=0),
            color=col[b],
            alpha=0.15,
            lw=0,
        )
        ax.plot(
            t,
            np.median(D, axis=0),
            color=col[b],
            lw=2,
            label="%s: median of %d channels (band: quartiles)" % (b, len(rs)),
        )
        Di = np.array([r["d_inp"][0, :] for r in rs])
        ax.plot(
            t,
            np.median(Di, axis=0),
            color=col[b],
            lw=1.2,
            ls="--",
            label="%s: left after a per-input complex gain" % b,
        )
    marks(ax)
    ax.set_xlim(1.5, 13.1)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("||g(t) - g(setup)||^2 / ||g(t)||^2")
    ax.set_title(
        "Source 0: change of the fit since the 01:41Z setup (w fixed since then)",
        fontsize=10,
    )
    ax.legend(fontsize=8, ncol=2, loc="upper left", frameon=False)
    ax.grid(color="#e5e7eb", lw=0.6)

    ax = fig.add_subplot(gs[1, :])
    for b in ("ATSC 24", "ATSC 30"):
        for k, ls in ((0, "-"), (1, "--"), (2, ":")):
            rs = [r for r in R if r["k"] == k and band(r["fid"]) == b]
            t = hr(rs[0]["t"])
            j = np.arange(2, len(t))
            S = np.median([r["d_tot"][j - 1, j] for r in rs], axis=0)
            S = np.where(
                np.diff(t)[j - 1] * 60 > 25, np.nan, S
            )  # no line across the Tau A blackout
            ax.plot(
                t[j],
                S,
                color=col[b],
                ls=ls,
                lw=1.8 if k == 0 else 1.2,
                label="%s, source %d (every %.0f min)"
                % (b, k, np.median(np.diff(t[1:])) * 60),
            )
    marks(ax)
    ax.set_yscale("log")
    ax.set_xlim(1.5, 13.1)
    ax.set_ylim(3e-3, 1.0)
    ax.set_xlabel("UTC hour, 11-04")
    ax.set_ylabel("||g - g(previous refit)||^2 / ||g||^2")
    ax.set_title(
        "Change at each refit, median over channels (sources 1-2 not refitted 09:38-11:53Z)",
        fontsize=10,
    )
    ax.legend(fontsize=8, ncol=2, loc="upper left", frameon=False)
    ax.grid(color="#e5e7eb", lw=0.6, which="both")

    # structure function: quiet first half vs the active hours
    ax = fig.add_subplot(gs[2, 0])
    bins = np.array([4, 7, 10, 15, 25, 40, 60, 100, 160, 270])
    mid = np.sqrt(bins[1:] * bins[:-1])
    for b in ("ATSC 24", "ATSC 30"):
        for (h0, h1), ls in (((2.0, 6.5), "-"), ((6.5, 9.5), "--")):
            acc = [[] for _ in mid]
            for r in (r for r in R if r["k"] == 0 and band(r["fid"]) == b):
                t = hr(r["t"])
                idx = np.where((t >= h0) & (t <= h1))[0]
                A, B = np.meshgrid(idx, idx, indexing="ij")
                m = A < B
                A, B = A[m], B[m]
                bi = np.digitize((r["t"][B] - r["t"][A]) / 60, bins) - 1
                for i in range(len(mid)):
                    acc[i].extend(r["d_tot"][A[bi == i], B[bi == i]])
            y = np.array([np.median(v) if len(v) > 20 else np.nan for v in acc])
            ax.plot(
                mid,
                y,
                color=col[b],
                ls=ls,
                lw=2,
                marker="o",
                ms=4,
                label="%s, pairs within %04.1f-%04.1fZ" % (b, h0, h1),
            )
            print(
                "%s %04.1f-%04.1fZ structure function: %s"
                % (
                    b,
                    h0,
                    h1,
                    " ".join("%.0fmin:%.3f" % (m_, v) for m_, v in zip(mid, y)),
                )
            )
    ax.plot(mid, 0.0025 * mid, color=C_CLEAN, lw=1, label="slope 1 (random walk)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time between the two fits (minutes)")
    ax.set_ylabel("||g_b - g_a||^2 / ||g_b||^2")
    ax.set_title(
        "Source 0: change vs time apart, median over channels and pairs", fontsize=10
    )
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(color="#e5e7eb", lw=0.6, which="both")

    # is the per-input part the analog chain? then both stations see the same per-input phases
    ax = fig.add_subplot(gs[2, 1])
    hh = a.at
    ph = {}
    rs0 = [r for r in R if r["k"] == 0]
    j = int(np.argmin(np.abs(hr(rs0[0]["t"]) - hh)))
    good = np.all([r["ein"] / r["ein"].max() > 0.05 for r in rs0], axis=0)
    for b in ("ATSC 24", "ATSC 30"):
        acc = 0
        for r in (r for r in rs0 if band(r["fid"]) == b):
            z = r["beta"][j].astype(np.complex128)
            z = np.where(np.abs(z) > 0, z / np.abs(z), 0)
            ref = np.sum(r["ein"] * z)
            acc = acc + z * np.conj(ref) / abs(ref)
        ph[b] = np.angle(acc)
    x, y = ph["ATSC 24"][good], ph["ATSC 30"][good]
    rho = np.corrcoef(x, y)[0, 1]
    lim = 1.1 * max(np.abs(x).max(), np.abs(y).max())
    ax.plot(
        [-lim, lim],
        [-lim, lim],
        color=C_CLEAN,
        lw=1,
        label="equal drift on both stations (analog chain or a shared arrival-direction change)",
    )
    ax.scatter(
        x,
        y,
        s=18,
        color="#374151",
        label="one input (%d with DTV response on every channel)" % good.sum(),
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("ATSC 24: per-input phase of g vs setup (rad), channel median")
    ax.set_ylabel("ATSC 30: same")
    ax.set_title(
        "Per-input phase drift at %s UTC, common phase removed: correlation %.2f"
        % ("%02d:%02d" % (int(hh), round(hh % 1 * 60)), rho),
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.grid(color="#e5e7eb", lw=0.6)
    print(
        "per-input phase drift at %.2fh: rms ATSC 24 %.3f, ATSC 30 %.3f rad; correlation %.2f"
        % (hh, x.std(), y.std(), rho)
    )
    fig.suptitle(
        "How fast the DTV fits drift: all 8 nodes, %d channels (band edges left out)"
        % len(rs0),
        fontsize=12,
    )
    fig.savefig(a.out_drift, dpi=110)
    print("saved", a.out_drift)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("extract")
    p.add_argument(
        "--states", default=os.path.expanduser("~/gnss/builds/dtv-live/states-rev2")
    )
    p.add_argument("--date", required=True)
    p.add_argument("--until", default="1300", help="HHMM UTC: last refit to use")
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument("nodes", nargs="*")
    p = sp.add_parser("gfull")
    p.add_argument(
        "--states", default=os.path.expanduser("~/gnss/builds/dtv-live/states-rev2")
    )
    p.add_argument("--date", required=True)
    p.add_argument("--until", default="1300")
    p.add_argument("--node", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--source", type=int, default=0)
    p.add_argument("--fids", required=True, help="comma-separated freq_ids")
    p.add_argument("--out", required=True)
    p = sp.add_parser("plot")
    p.add_argument("summary")
    p.add_argument("gfull")
    p.add_argument(
        "--at",
        type=float,
        default=12.5,
        help="UTC hour of the per-input phase comparison",
    )
    p.add_argument("--out-fits", required=True)
    p.add_argument("--out-drift", required=True)
    a = ap.parse_args()
    if a.cmd == "plot":
        cmd_plot(a)
    elif a.cmd == "extract":
        cmd_extract(a)
    elif a.cmd == "gfull":
        cmd_gfull(a)


if __name__ == "__main__":
    main()
