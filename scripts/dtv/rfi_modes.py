#!/usr/bin/env python3
"""Spatial modes per channel in the 10-30 baseband dump (chunk 0: 0.84 s, 10:59:20Z, Tau A at boresight), from
dtv_cov.py outputs. Each input is whitened by its own power over the chunk; eigenvalues of the chunk covariance
are in NOISE units (noise = mean of eigenvalues 20..m-8). Per channel: the top eigenvalues, how many modes sit
above 2 and above Tau A (the clean reference channels' top eigenvalue), and how steady the top mode is over
the 20 frames (42 ms each): the coefficient of variation of its frame power and |u_frame^H u_chunk|^2.

    rfi_modes.py cov_a.npz [cov_b.npz ...] [--out fig.png]

What the counts mean for the subtraction stage: one mode per independent transmitter (or multipath
cluster the beam resolves); a mode whose frame power swings (traffic) still subtracts if its direction
(u) holds, since the stage fits the template s = w^H x every frame.
"""
import argparse
import os

import numpy as np

CS = 0.1953125
BANDS = (  # (label, lo MHz, hi MHz): North American allocations
    ("MUOS sat downlink 360-380", 360.0, 380.0),
    ("ATSC TV 470-608", 470.0, 608.0),
    ("600 MHz LTE/NR DL (B71/n71) 617-652", 617.0, 652.0),
    ("700 lower unpaired D/E (B29 SDL) 716-728", 716.0, 728.0),
    ("700 lower A/B/C DL (B12/17) 728-746", 728.0, 746.0),
    ("700 upper C DL (B13) 746-756", 746.0, 756.0),
    ("700 public-safety DL (B14) 758-768", 758.0, 768.0),
    ("850 cellular DL (B5) 869-894", 869.0, 894.0),
)


def band_of(mhz):
    for lab, lo, hi in BANDS:
        if lo <= mhz < hi:
            return lab
    return "clean / other"


def modes(cov):
    """cov [20, 128, 128] raw -> (lam [m] noise units desc, cv, overlap, kept inputs)."""
    C = cov.sum(axis=0)
    d = np.real(np.diag(C))
    keep = d > 1e-6 * np.median(d[d > 0]) if (d > 0).any() else d > 0
    w = np.zeros(128)
    w[keep] = 1.0 / np.sqrt(d[keep])
    R = (C * w[:, None] * w[None, :])[np.ix_(keep, keep)]
    m = keep.sum()
    ev, U = np.linalg.eigh(R)
    ev, U = ev[::-1], U[:, ::-1]
    noise = ev[20 : m - 8].mean()
    u1 = U[:, 0]
    nf = cov.shape[0]
    p, ov = [], []
    for j in range(nf):
        Rj = (cov[j] * w[:, None] * w[None, :])[np.ix_(keep, keep)] * nf
        p.append(np.real(u1.conj() @ Rj @ u1))
        e, V = np.linalg.eigh(Rj)
        ov.append(abs(V[:, -1].conj() @ u1) ** 2)
    p = np.array(p)
    return ev / noise, p.std() / p.mean(), float(np.median(ov)), int(m)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("covs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    for path in a.covs:
        z = np.load(path)
        for c, fid in enumerate(z["freq"]):
            lam, cv, ov, m = modes(z["cov"][c].astype(np.complex128))
            rows.append(
                dict(
                    fid=int(fid),
                    mhz=fid * CS,
                    lam=lam,
                    cv=cv,
                    ov=ov,
                    m=m,
                    band=band_of(fid * CS),
                )
            )
    rows.sort(key=lambda r: r["fid"])
    clean = [r for r in rows if r["band"] == "clean / other"]
    # Tau A's level: the clean channels' top eigenvalue, interpolated in frequency
    cf = np.array([r["mhz"] for r in clean])
    cl = np.array([r["lam"][0] for r in clean])

    def taua(mhz):
        return float(np.interp(mhz, cf, cl)) if len(cf) else 5.0

    print(
        "%6s %8s %-40s %4s  %-38s %4s %4s %6s %5s"
        % (
            "fid",
            "MHz",
            "band",
            "m",
            "lam1..6 (noise units)",
            ">2",
            ">TA",
            "cv1",
            "ov1",
        )
    )
    for r in rows:
        r["ta"] = taua(r["mhz"])
        r["n2"] = int((r["lam"] > 2).sum())
        r["nta"] = int((r["lam"] > r["ta"] * 1.2).sum())
        print(
            "%6d %8.3f %-40s %4d  %-38s %4d %4d %6.3f %5.3f"
            % (
                r["fid"],
                r["mhz"],
                r["band"][:40],
                r["m"],
                " ".join("%5.1f" % x for x in r["lam"][:6]),
                r["n2"],
                r["nta"],
                r["cv"],
                r["ov"],
            )
        )
    print(
        "\nper band (channel medians; Tau A level = clean channels' top eigenvalue, interpolated):"
    )
    print(
        "%-42s %4s %8s %8s %6s %6s %6s %6s"
        % ("band", "n", "lam1", "TauA", ">2", ">1.2TA", "cv1", "ov1")
    )
    for lab in [b[0] for b in BANDS] + ["clean / other"]:
        rs = [r for r in rows if r["band"] == lab]
        if not rs:
            continue
        print(
            "%-42s %4d %8.1f %8.1f %6.1f %6.1f %6.3f %6.3f"
            % (
                lab[:42],
                len(rs),
                np.median([r["lam"][0] for r in rs]),
                np.median([r["ta"] for r in rs]),
                np.median([r["n2"] for r in rs]),
                np.median([r["nta"] for r in rs]),
                np.median([r["cv"] for r in rs]),
                np.median([r["ov"] for r in rs]),
            )
        )
    if a.out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        groups = [b for b in BANDS if any(r["band"] == b[0] for r in rows)]
        fig, axs = plt.subplots(
            1,
            len(groups),
            figsize=(3.2 * len(groups) + 1, 4.6),
            sharey=True,
            squeeze=False,
        )
        shades = ["#0b3d91", "#2563eb", "#60a5fa", "#93c5fd", "#bfdbfe", "#dbeafe"]
        for ax, (lab, lo, hi) in zip(axs[0], groups):
            pad = 3.0
            rs = [r for r in rows if lo - pad <= r["mhz"] <= hi + pad]
            x = np.array([r["mhz"] for r in rs])
            for k in range(5, -1, -1):
                ax.plot(
                    x,
                    [r["lam"][k] for r in rs],
                    "o",
                    ms=3,
                    color=shades[k],
                    label="eigenvalue %d" % (k + 1),
                )
            ax.plot(
                x,
                [r["ta"] for r in rs],
                color="#6b7280",
                lw=1,
                ls="--",
                label="Tau A level (clean channels)",
            )
            ax.axvspan(lo, hi, color="#fde68a", alpha=0.3, lw=0)
            ax.set_yscale("log")
            ax.set_title(lab, fontsize=8)
            ax.set_xlabel("MHz", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(color="#e5e7eb", lw=0.6)
        axs[0][0].set_ylabel("whitened eigenvalue (noise units)")
        axs[0][0].legend(fontsize=6, loc="upper left", frameon=False)
        fig.suptitle(
            "Spatial modes per channel\n10-30 dump chunk 0 (0.84 s, Tau A at boresight); shaded: the allocation",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(a.out, dpi=110)
        print("saved", a.out)


if __name__ == "__main__":
    main()
