#!/usr/bin/env python3
"""The rolling-g stage's own telemetry over a night, from dtv_roll.py watch's log (one status line per GPU per
minute): per channel the interval residual / raw power and the fraction source 0 explains, plus trips, holds and
the Tau A freeze. Prints station medians per --bin minutes; with --out draws them against UTC.

res/raw is the whole residual over all 128 inputs (sky + noise + what DTV is left), so its floor is not zero and
the band-edge channels (little DTV) sit near 1: judge DTV left from the recv1 N2 eigenvalues (vis_dtv_*.py).

    dtv_roll_night.py --log /var/tmp/kvand-dtvlive2/watch_roll.log --day 20261104 [--bin 10] [--out fig.png]
"""
import argparse
import calendar
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

C_A24, C_A30, C_EDGE = "#2563eb", "#ea580c", "#6b7280"
# channel centre outside the ATSC flat band (lower edge + 0.31 .. upper edge - 0.31 MHz): roll-off, little DTV
EDGE = {2714, 2743, 2744, 2898, 2899, 2928}
STAT = re.compile(r"^(\d\d):(\d\d):(\d\d)Z (cx\d+) g(\d) (ON |off) refits (\d+)(?: \(\+\d+\))? holds (\d+) trips (\d+) "
                  r"failed (\d+) ([\d.]+) ms tau \[[^]]*\] freeze \[([^]]*)\] subtracting (\d+)/(\d+) \| res/raw / expl0: (.*)$")
CH = re.compile(r"(\d+):([\d.-]+)/([\d.-]+)([TW]?)")


def station(f):
    return "A24" if 2714 <= f <= 2744 else ("A30" if 2898 <= f <= 2928 else None)


def parse(path, day):
    t0 = calendar.timegm(time.strptime(day, "%Y%m%d"))
    rows, last, off = [], None, 0.0
    for ln in open(path):
        m = STAT.match(ln)
        if not m:
            continue
        s = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
        if last is not None and s + off < last - 43200:
            off += 86400.0
        last = s + off
        for c in CH.finditer(m[15]):
            rr = float(c[2]) if c[2] != "-" else np.nan
            ex = float(c[3]) if c[3] != "-" else np.nan
            rows.append((t0 + s + off, m[4], int(m[5]), int(c[1]), rr, ex, c[4], int(m[9]), int(m[8]),
                         m[12].replace(" ", ""), int(m[13])))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--day", required=True, help="YYYYMMDD of the log's first line")
    ap.add_argument("--bin", type=float, default=10.0, help="minutes")
    ap.add_argument("--out")
    ap.add_argument("--marks", default="", help="HH:MM=label,... (UTC on the day after --day when < 12:00)")
    a = ap.parse_args()
    rows = parse(a.log, a.day)
    t = np.array([r[0] for r in rows])
    fid = np.array([r[3] for r in rows])
    rr = np.array([r[4] for r in rows])
    ex = np.array([r[5] for r in rows])
    gpu = np.array(["%s g%d" % (r[1], r[2]) for r in rows])
    trips = np.array([r[7] for r in rows])
    freeze = np.array([r[9] for r in rows])
    st = np.array([station(f) for f in fid])
    core = np.array([f not in EDGE for f in fid])
    print("%d channel rows, %s .. %s, %d channels" % (len(rows), time.strftime("%m-%d %H:%MZ", time.gmtime(t.min())),
          time.strftime("%m-%d %H:%MZ", time.gmtime(t.max())), len(set(zip(gpu, fid)))))

    # trip events: per GPU, minutes where the cumulative count rose
    tg = {}
    for g in sorted(set(gpu)):
        sel = (gpu == g)
        tt, k = t[sel], trips[sel]
        _, first = np.unique(tt, return_index=True)
        tt, k = tt[first], k[first]
        for i in np.nonzero(np.diff(k) > 0)[0]:
            tg.setdefault(tt[i + 1], []).append("%s +%d" % (g, k[i + 1] - k[i]))
    print("trip events (minute: gpu +n):")
    for tt in sorted(tg):
        print("  %s %s" % (time.strftime("%H:%MZ", time.gmtime(tt)), ", ".join(tg[tt])))
    fz = sorted(set(time.strftime("%H:%M", time.gmtime(x)) for x in t[freeze != "0,0,0"]))
    if fz:
        print("sources 1-2 frozen %s .. %sZ" % (fz[0], fz[-1]))

    edges = np.arange(np.floor(t.min() / (a.bin * 60)) * a.bin * 60, t.max() + a.bin * 60, a.bin * 60)
    idx = np.digitize(t, edges) - 1
    print("\n%-6s  %-28s  %-28s  edge res/raw" % ("UTC", "A24 core res/raw  expl0 (n)", "A30 core res/raw  expl0 (n)"))
    series = {k: [] for k in ("t", "A24rr", "A24ex", "A30rr", "A30ex", "A24rr10", "A24rr90", "A30rr10", "A30rr90",
                              "A24ex10", "A24ex90", "A30ex10", "A30ex90", "edge")}
    for b in range(len(edges) - 1):
        sel = idx == b
        if not sel.any():
            continue
        series["t"].append(edges[b] + a.bin * 30)
        out = []
        for s in ("A24", "A30"):
            m = sel & (st == s) & core & np.isfinite(rr)
            # per channel median within the bin, then across channels
            per_rr = [np.nanmedian(rr[m & (gpu == g) & (fid == f)]) for g, f in set(zip(gpu[m], fid[m]))]
            per_ex = [np.nanmedian(ex[m & (gpu == g) & (fid == f)]) for g, f in set(zip(gpu[m], fid[m]))]
            series[s + "rr"].append(np.median(per_rr))
            series[s + "ex"].append(np.median(per_ex))
            for q, nm in ((10, "10"), (90, "90")):
                series[s + "rr" + nm].append(np.percentile(per_rr, q))
                series[s + "ex" + nm].append(np.percentile(per_ex, q))
            out.append("%.3f  %.3f (%2d)" % (np.median(per_rr), np.median(per_ex), len(per_rr)))
        e = sel & ~core & np.isfinite(rr)
        series["edge"].append(np.nanmedian(rr[e]) if e.any() else np.nan)
        if int((edges[b] % 3600) / 60) % 30 == 0 or a.bin >= 30:
            print("%s  %-28s  %-28s  %.3f" % (time.strftime("%H:%MZ", time.gmtime(edges[b])), out[0], out[1],
                                              series["edge"][-1]))
    if not a.out:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vis_dtv_plot import hour_angle_min

    day0 = calendar.timegm(time.strptime(a.day, "%Y%m%d"))
    h = lambda x: (np.asarray(x) - day0) / 3600.0  # noqa: E731  hours from the log's first UTC midnight
    S = {k: np.array(v) for k, v in series.items()}
    fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    for s, col, lab in (("A24", C_A24, "ATSC 24 (530-536 MHz)"), ("A30", C_A30, "ATSC 30 (566-572 MHz)")):
        for k, (axx, key) in enumerate(((ax[0], "rr"), (ax[1], "ex"))):
            axx.fill_between(h(S["t"]), S[s + key + "10"], S[s + key + "90"], color=col, alpha=0.15, lw=0)
            axx.plot(h(S["t"]), S[s + key], color=col, lw=2, label=lab + ", median of %s channels" % (
                "core" if True else ""))
    ax[0].plot(h(S["t"]), S["edge"], color=C_EDGE, lw=1.2, ls="--", label="band-edge channels (little DTV), median")
    ax[0].set_ylabel("residual / raw power\n(all inputs, stage interval)")
    ax[1].set_ylabel("fraction source 0 explains")
    ax[0].set_ylim(0, 1)
    ax[1].set_ylim(0, 1)
    # Tau A transit and freeze window
    tt = np.arange(t.min(), t.max(), 60.0)
    ha = hour_angle_min(tt * 1e9)
    i0 = np.argmin(np.abs(ha))
    fzm = np.array([x != "0,0,0" for x in freeze])
    for axx in ax:
        axx.axvline(h(tt[i0]), color="k", lw=0.8, ls=":")
        if fzm.any():
            axx.axvspan(h(t[fzm].min()), h(t[fzm].max()), color="#9ca3af", alpha=0.18, lw=0)
        for tr in tg:
            axx.axvline(h(tr), color="#dc2626", lw=0.6, alpha=0.5, ymin=0.95)
        for mk in filter(None, a.marks.split(",")):
            hm, lab = mk.split("=")
            x = int(hm[:2]) + int(hm[3:]) / 60.0 + (24 if int(hm[:2]) < 12 else 0)
            axx.axvline(x, color="#374151", lw=0.8, ls="--")
            if axx is ax[0]:
                axx.text(x, 0.97, " " + lab, fontsize=8, va="top", color="#374151")
        axx.grid(alpha=0.25)
    ax[0].text(h(tt[i0]), 0.03, " Tau A transit\n (shaded: sources 1-2 frozen)", fontsize=8, color="#374151")
    ax[0].legend(loc="center right", fontsize=8, framealpha=0.9)
    ax[1].set_xlabel("UTC (hours; 24 = 00Z 11-05)")
    ticks = np.arange(np.ceil(h(t.min())), np.floor(h(t.max())) + 1)
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels(["%02d" % (x % 24) for x in ticks])
    ax[0].set_title("Rolling-g stage telemetry, 16 GPUs (bands: 10-90 %% of core channels; red ticks: guard trips)")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
