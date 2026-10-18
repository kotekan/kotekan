#!/usr/bin/env python3
"""Step 0b re-survey (docs/CHORD_JOINT_TRACKING.md 6.3): fit the station position error from the tracker code residual with the troposphere
applied. Model per row: resid = clk(epoch) + dU*sin(el) + dN*cos(el)cos(az) + dE*cos(el)sin(az).
The per-epoch clock is removed as a fixed effect (within-10-s demeaning of y and the regressors).
Sign: resid = measured - model in metres (a delay is negative, verified by the troposphere flattening
the low-elevation bins), i.e. resid = r_model - r_true = -(d . u) with d = model station - true station
(ENU) and u the unit vector to the satellite. The printed dU/dN/dE are therefore d = MODEL - TRUE: the
model station sits that far up/north/east of the true one, and the correction is to SUBTRACT them.
Gate after a position change: dU, dN, dE within +-2 m of 0 on the next day's obs.
usage: gnss_resurvey.py OBS_DIR YYYYMMDD band [band ...]"""
import argparse
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "python",
        "scripts",
        "gnss",
    ),
)
import gnss_atmos as ATM  # noqa: E402

SITE = (
    49.32001414,
    -119.62262691,
    545.0,
)  # config/gnss_chains_chord.yaml common lat/lon/alt
import glob  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument("obs_dir")
ap.add_argument("day", help="YYYYMMDD of the obs files")
ap.add_argument("bands", nargs="+")
ap.add_argument("--lat", type=float, default=SITE[0])
ap.add_argument("--lon", type=float, default=SITE[1])
ap.add_argument("--alt", type=float, default=SITE[2])
ap.add_argument(
    "--no-tropo", action="store_true", help="fit without the troposphere added back"
)
a = ap.parse_args()
LAT, LON, ALT = a.lat, a.lon, a.alt
odir, day = a.obs_dir, a.day
for band in a.bands:
    rows = []
    for f in glob.glob("%s/%s_%s.jsonl" % (odir, band, day)):
        for i, raw in enumerate(open(f, "rb")):
            if i % 3:
                continue
            if b'"code_resid_m"' not in raw or b'"el"' not in raw:
                continue
            try:
                r = json.loads(raw)
            except Exception:
                continue
            if r.get("code_resid_src") != "trk" or not r.get("code_resid_clk"):
                continue
            el, az, cr = r.get("el"), r.get("az"), r.get("code_resid_m")
            if el is None or az is None or cr is None or el < 7.0:
                continue
            rows.append(
                (
                    int(r["t"] // 10),
                    r["prn"],
                    el,
                    az,
                    cr + (0.0 if a.no_tropo else ATM.tropo_delay_m(el, LAT, ALT)),
                )
            )
    # fixed effects: demean within epoch buckets with >= 4 PRNs
    by = {}
    for b, p, el, az, y in rows:
        by.setdefault(b, []).append((el, az, y))
    Y, X = [], []
    for b, v in by.items():
        if len(v) < 4:
            continue
        els = np.radians([q[0] for q in v])
        azs = np.radians([q[1] for q in v])
        ys = np.array([q[2] for q in v])
        reg = np.column_stack(
            [np.sin(els), np.cos(els) * np.cos(azs), np.cos(els) * np.sin(azs)]
        )
        # robust centre: median
        ys = ys - np.median(ys)
        reg = reg - np.median(reg, axis=0)
        Y.append(ys)
        X.append(reg)
    if not Y:
        print("%s: no data" % band)
        continue
    y = np.concatenate(Y)
    x = np.vstack(X)
    m = np.ones(len(y), bool)
    for _ in range(4):
        sol, *_ = np.linalg.lstsq(x[m], y[m], rcond=None)
        res = y - x @ sol
        s = 1.4826 * np.median(np.abs(res[m] - np.median(res[m])))
        m = np.abs(res) < 3 * s
    dU, dN, dE = -sol  # d = model - true
    print(
        "%-8s rows %6d  model-true: dU %+6.2f m  dN %+6.2f m  dE %+6.2f m   (robust rms %.2f m, %d used)"
        % (band, len(y), dU, dN, dE, s, m.sum())
    )
