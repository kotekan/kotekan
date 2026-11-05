#!/usr/bin/env python3
"""Where the DTV is, from an F-engine digital gain table: the table equalises each channel's output power, so
input power ~ 1/|gain|^2 and a transmitter shows as a dip in the gains. Per freq_id (= the table index;
freq = freq_id x 0.1953125 MHz): the median |gain| over live inputs against a running-median baseline;
excess = -20 log10(gain / baseline) dB of input power. Prints the 6 MHz North American TV channels (14-36,
470-608 MHz) with their excess, and every other run of channels above --thresh dB.

    dtv_gain_survey.py [--table NAME] [--thresh 0.5] [--out fig.png]

Single-input power only sees transmitters within ~10 dB of the system noise per input; the array (eigenvalues)
sees far weaker ones, so this is a lower bound on what needs cleaning.
"""
import argparse
import os

import h5py
import numpy as np

D = "/mnt/cs00/data/chord_pathfinder_f_engine_data/data/digital_gains/"
CS = 0.1953125
LIVE = {
    2714,
    2715,
    2716,
    2717,
    2718,
    2719,
    2720,
    2721,
    2722,
    2723,
    2724,
    2725,
    2726,
    2727,
    2728,
    2729,
    2730,
    2731,
    2732,
    2733,
    2734,
    2735,
    2736,
    2737,
    2738,
    2739,
    2740,
    2741,
    2742,
    2743,
    2744,
}
LIVE |= {f + 184 for f in LIVE}  # ATSC 30: 2898-2928
ARRAY = {
    17,
    19,
    24,
    30,
    31,
    33,
    35,
}  # 8-VSB stations the array sees in the 10-30 baseband dump (pilot at edge+0.31)


def tv_channels():
    """(number, lower MHz) for UHF 14-36 (the post-repack band; 37 is radio astronomy)."""
    return [(n, 470.0 + 6.0 * (n - 14)) for n in range(14, 37)]


def run_median(x, half):
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.nanmedian(x[max(0, i - half) : i + half + 1])
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--table",
        default="20261023T210749Z_chord_pathfinder_digitalgain",
        help="flat equalised baseline",
    )
    ap.add_argument(
        "--thresh", type=float, default=0.5, help="dB of input-power excess that counts"
    )
    ap.add_argument(
        "--half",
        type=int,
        default=160,
        help="running-median half width (channels) for the baseline",
    )
    ap.add_argument("--out")
    a = ap.parse_args()
    with h5py.File(os.path.join(D, a.table, "00000000.h5"), "r") as f:
        g = np.abs(f["gain_coeff"][0]).astype(np.float64)  # [8192, 128]
        centre = f["index_map/freq"]["centre"][:]
    live = np.median(g, axis=0) > 0
    live &= (
        np.ptp(g, axis=0) > 0
    )  # an input left at a constant (default) gain was not equalised
    med = np.median(g[:, live], axis=1)
    fid = np.arange(len(med))
    band = (fid * CS >= 300) & (fid * CS <= 1500)
    lg = np.where(band & (med > 0), np.log10(np.maximum(med, 1e-12)), np.nan)
    base = run_median(lg, a.half)
    exc = -20 * (lg - base)  # dB of input power above the local baseline
    # per-input agreement: fraction of live inputs whose own gain dips by > thresh/2 in this channel
    gi = np.log10(np.maximum(g[:, live], 1e-12))
    bi = base[:, None] + (np.nanmedian(gi - lg[:, None], axis=0))[None, :]
    frac = np.nanmean(-20 * (gi - bi) > a.thresh / 2, axis=1)
    print(
        "table %s: %d live inputs; centre(freq_id 2714) = %.3f MHz (freq_id x CS = %.3f)"
        % (a.table, live.sum(), centre[2714], 2714 * CS)
    )

    print(
        "\nUHF TV channels (6 MHz; core = the PFB channels inside lower+0.31 .. upper-0.31 MHz):"
    )
    print(
        "%4s %-14s %-11s %5s %7s %7s %6s  %s"
        % ("ch", "MHz", "freq_ids", "n", "med dB", "max dB", "inputs", "status")
    )
    found = []
    for n, lo in tv_channels():
        f0, f1 = int(np.ceil(lo / CS)), int(np.floor((lo + 6.0) / CS))
        core = (fid >= int(np.ceil((lo + 0.31) / CS))) & (
            fid <= int(np.floor((lo + 5.69) / CS))
        )
        e = exc[core]
        hit = np.nanmedian(e) > a.thresh
        st = "CLEANED" if set(range(f0, f1 + 1)) & LIVE else ""
        if hit:
            found.append((n, lo, f0, f1))
        print(
            "%4d %5.0f-%-8.0f %4d-%-6d %5d %7.2f %7.2f %5.0f%%  %s%s"
            % (
                n,
                lo,
                lo + 6,
                f0,
                f1,
                f1 - f0 + 1,
                np.nanmedian(e),
                np.nanmax(e),
                100 * np.nanmedian(frac[core]),
                "DTV " if hit else "",
                st,
            )
        )
    nch = sum(f1 - f0 + 1 for _, _, f0, f1 in found)
    print(
        "-> %d TV channels above %.1f dB: %s; %d PFB channels (%d already cleaned)"
        % (
            len(found),
            a.thresh,
            " ".join(str(n) for n, *_ in found),
            nch,
            sum(1 for _, _, f0, f1 in found for x in range(f0, f1 + 1) if x in LIVE),
        )
    )

    print("\nother runs above %.1f dB (>= 3 channels), 300-1500 MHz:" % a.thresh)
    on = np.nan_to_num(exc) > a.thresh
    i = 0
    tv = (fid * CS >= 470) & (fid * CS < 608)
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            if j - i + 1 >= 3 and not tv[i : j + 1].all():
                print(
                    "  %4d-%-4d %7.2f-%-7.2f MHz %3d ch  median %.2f dB, max %.2f dB"
                    % (
                        i,
                        j,
                        i * CS,
                        j * CS,
                        j - i + 1,
                        np.median(exc[i : j + 1]),
                        np.max(exc[i : j + 1]),
                    )
                )
            i = j + 1
        else:
            i += 1
    if a.out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(15, 7.5))
        for axx, (x0, x1) in zip(ax, ((300, 1500), (460, 620))):
            s = (fid * CS >= x0) & (fid * CS <= x1)
            axx.plot(fid[s] * CS, exc[s], color="#374151", lw=0.7)
            axx.axhline(a.thresh, color="#9ca3af", lw=0.6, ls="--")
            for n, lo in tv_channels():
                if x0 <= lo <= x1:
                    seen = n in ARRAY
                    cl = any(
                        x in LIVE
                        for x in range(
                            int(np.ceil(lo / CS)), int(np.floor((lo + 6) / CS)) + 1
                        )
                    )
                    axx.axvspan(
                        lo,
                        lo + 6,
                        color="#ea580c" if cl else ("#2563eb" if seen else "#e5e7eb"),
                        alpha=0.25 if (seen or cl) else 0.35,
                        lw=0,
                    )
                    if x1 - x0 < 400:
                        axx.text(
                            lo + 3,
                            0.97,
                            str(n),
                            transform=axx.get_xaxis_transform(),
                            ha="center",
                            va="top",
                            fontsize=8,
                        )
            axx.set_xlim(x0, x1)
            axx.set_ylabel("input power above baseline (dB)\nfrom the equalised gains")
            axx.grid(color="#e5e7eb", lw=0.6)
        ax[0].set_title(
            "F-engine gain table %s: input power above a running-median baseline (orange: ATSC stations "
            "cleaned now; blue: stations the ARRAY sees in the 10-30 dump, not cleaned)"
            % a.table[:16],
            fontsize=10,
        )
        ax[1].set_xlabel("MHz (bottom: the UHF TV band, channel numbers on top)")
        fig.tight_layout()
        fig.savefig(a.out, dpi=110)
        print("saved", a.out)


if __name__ == "__main__":
    main()
