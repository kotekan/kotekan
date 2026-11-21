#!/usr/bin/env python3
"""F-engine benchmark figures: FFT accuracy per implementation, and per-window GPU
time / realtime utilisation across FFT sizes."""

import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e8e8e6"
# Fixed categorical order; color follows the entity in every figure.
COLORS = {
    "cuFFT fp16": "#2a78d6",
    "tensor fp16": "#eb6834",
    "tensor TF32": "#1baf7a",
    "cuFFT fp32": "#eda100",
    "tensor fp16 fused": "#e87ba4",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": TEXT,
        "axes.edgecolor": TEXT2,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT2,
        "ytick.color": TEXT2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    }
)


def style_line(ax, xs, ys, name, direct=True):
    ax.plot(
        xs,
        ys,
        color=COLORS[name],
        linewidth=2.2,
        marker="o",
        markersize=6.5,
        markerfacecolor=COLORS[name],
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        label=name,
        zorder=3,
    )
    if direct:
        # Direct label at the line's right end (relief rule for the low-contrast slot too).
        ax.annotate(
            name,
            (xs[-1], ys[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=TEXT,
            fontsize=10.5,
            va="center",
        )


def legend(ax):
    ax.legend(loc="upper left", frameon=False, fontsize=10.5, labelcolor=TEXT)


def accuracy_figure(path_csv, path_png):
    rows = list(csv.DictReader(open(path_csv)))
    names = ["cuFFT fp32", "tensor TF32", "tensor fp16", "cuFFT fp16"]
    fig, ax = plt.subplots(figsize=(8.6, 5.2), layout="constrained")
    xs = sorted({int(r["n"]) for r in rows})
    for name in names:
        ys = [float(r[name]) for n in xs for r in rows if int(r["n"]) == n]
        style_line(ax, xs, ys, name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(xs, [f"{n}" for n in xs])
    ax.set_xlabel("FFT length n (batched sub-FFT of the four-step split)")
    ax.set_ylabel("relative L2 error vs double-precision DFT")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0e}")
    ax.set_title(
        "Batched FFT accuracy on GB10: tensor-core vs cuFFT, fp16 vs fp32/TF32",
        fontsize=12.5,
        loc="left",
    )
    legend(ax)
    ax.set_xlim(xs[0] * 0.9, xs[-1] * 3.1)
    fig.savefig(path_png, dpi=180)
    print("wrote", path_png)


def utilisation_figure(path_csv, path_png):
    rows = list(csv.DictReader(open(path_csv)))
    label = {
        ("cufft", "fp16"): "cuFFT fp16",
        ("tensor", "fp16"): "tensor fp16",
        ("tensor", "fp32"): "tensor TF32",
        ("cufft", "fp32"): "cuFFT fp32",
        ("tensor-fused", "fp16"): "tensor fp16 fused",
    }
    series = {}
    for r in rows:
        key = label[(r["impl"], r["precision"])]
        series.setdefault(key, []).append(
            (int(r["n"]), float(r["gpu_s_mean"]) * 1e3, float(r["utilisation"]) * 100)
        )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.2), layout="constrained")
    for name in [k for k in COLORS if k in series]:
        pts = sorted(series[name])
        xs = [p[0] for p in pts]
        # Legends carry identity here: the fp16 paths converge, so direct labels collide.
        style_line(ax1, xs, [p[1] for p in pts], name, direct=False)
        style_line(ax2, xs, [p[2] for p in pts], name, direct=False)
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        xs = sorted({p[0] for pts in series.values() for p in pts})
        ax.set_xticks(xs, [f"2^{int(math.log2(n))}" for n in xs])
        ax.set_xlabel("FFT window length (samples)")
        ax.set_xlim(xs[0] * 0.8, xs[-1] * 6)
    ax1.set_yscale("log")
    legend(ax1)
    legend(ax2)
    ax1.set_ylabel("GPU time per window (ms)")
    ax1.set_title("Per-window GPU time, one ADC chain", fontsize=12.5, loc="left")
    ax2.set_ylabel("share of the realtime window period (%)")
    ax2.set_title(
        "GPU utilisation at 3.2 Gsps (one of two ADC chains)", fontsize=12.5, loc="left"
    )
    ax2.axhline(50, color=TEXT2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax2.annotate(
        "budget for two chains on one GPU",
        (0.02, 50),
        xycoords=("axes fraction", "data"),
        xytext=(0, 5),
        textcoords="offset points",
        color=TEXT2,
        fontsize=10,
    )
    ax2.set_ylim(0, None)
    fig.savefig(path_png, dpi=180)
    print("wrote", path_png)


if __name__ == "__main__":
    if sys.argv[1] == "accuracy":
        accuracy_figure(sys.argv[2], sys.argv[3])
    else:
        utilisation_figure(sys.argv[2], sys.argv[3])
