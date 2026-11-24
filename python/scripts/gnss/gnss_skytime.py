#!/usr/bin/env python3
"""The capture's time anchor measured from the sky: how far the host-labelled time of the samples
is from GPS time, from the CNAV messages the nav decoders read.

    python3 gnss_skytime.py --yaml chains_l5.yaml MARKS.json [MARKS.json ...] [--json OUT]

Each mark (nav_decode.py --timemarks-file) is a CRC-checked message: the host-labelled UTC of its
first symbol's edge, and the TOW count it carries, which is the satellite time at which the next
message starts. That message started TOW * 6 s - its length (6 s on L5, 12 s on L2C) of satellite
time; less the satellite clock and plus the range over c, it arrived at that GPS time. The
anchor error is the host label minus that. It holds whatever the host clock reads, to half a week
(the TOW does not carry the week), and to the 1 ms of the records that place the edges.

Positive means the labels are late. bladeRFInput fixes the anchor once, from the system clock at
the first sample, so a host clock that was wrong when kotekan started shows here for the whole
run. The site comes from the broker yaml; only offsets are printed.
"""

import argparse
import json
import re
import sys
import warnings
from datetime import datetime, timezone

import numpy as np

import gnss_brdc_supply
import gnss_ephemeris as ge
from gnss_broker.sky import HINT_MAX_AGE_S

GPS0_UNIX = 315964800.0
LEAP_S = 18.0
MSG_S = {"GPS_L5_CNAV": 6.0, "GPS_L2C_CNAV": 12.0}
_KEY = r"\n    %s: ([-0-9.eE+]+)"
# A GPS satellite at 45 deg elevation, no clock: the first pass's range.
NOMINAL_ROW = {"range_m": 2.2e7, "sat_clk_s": 0.0}


def site_of(yaml_path):
    s = open(yaml_path).read()
    return tuple(float(re.search(_KEY % k, s).group(1)) for k in ("lat", "lon", "alt"))


def anchor_error_s(mark, row):
    """The host label minus the GPS time the message start arrived, s; row is the satellite's
    predict_all row at about that time (range_m, sat_clk_s)."""
    g_host = mark["t_host"] - GPS0_UNIX + LEAP_S
    t_tx = (g_host // 604800.0) * 604800.0 + mark["tow"] * 6.0 - MSG_S[mark["signal"]]
    t_tx += 604800.0 * round((g_host - t_tx) / 604800.0)
    return g_host - (t_tx - row["sat_clk_s"] + row["range_m"] / ge.C_LIGHT)


def measure(marks, eph, site):
    """[(signal, prn, error s)] for the marks whose satellite the ephemeris predicts. A first
    pass on a nominal range (good to ~10 ms) finds the time the satellite is predicted at, so a
    host clock hours off still finds its satellites; records up to sky.HINT_MAX_AGE_S old do,
    since 1 ms is 300 km of range."""
    out = []
    for m in marks:
        if m.get("signal") not in MSG_S:
            continue
        t0 = m["t_host"] - anchor_error_s(m, NOMINAL_ROW)
        row = ge.predict_all(
            eph,
            *site,
            datetime.fromtimestamp(t0, tz=timezone.utc),
            mask_deg=-90.0,
            max_age=HINT_MAX_AGE_S,
            sysc="G",
        ).get(("G", m["prn"]))
        if row is not None:
            out.append((m["signal"], m["prn"], anchor_error_s(m, row)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("marks", nargs="+", help="nav_decode.py --timemarks-file outputs")
    ap.add_argument("--yaml", required=True, help="the broker yaml holding the site")
    ap.add_argument("--json", help="write the result here")
    a = ap.parse_args(argv)
    marks = []
    for p in a.marks:
        marks += json.load(open(p)).get("marks", [])
    warnings.simplefilter("ignore")
    eph = ge.parse_rinex_nav(gnss_brdc_supply.cached_brdc())
    res = measure(marks, eph, site_of(a.yaml))
    if not res:
        print("no usable marks")
        return 1
    err = np.array([r[2] for r in res]) * 1e3
    doc = {
        "t": datetime.now(timezone.utc).timestamp(),
        "n_marks": len(res),
        "n_sats": len({r[1] for r in res}),
        "anchor_error_ms": round(float(np.median(err)), 3),
        "spread_ms": [round(float(err.min()), 3), round(float(err.max()), 3)],
    }
    print(
        "capture anchor %+.3f ms from the sky (%d marks, %d satellites, %+.3f..%+.3f ms)"
        % (doc["anchor_error_ms"], doc["n_marks"], doc["n_sats"], *doc["spread_ms"])
    )
    if a.json:
        with open(a.json, "w") as f:
            json.dump(doc, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
