#!/usr/bin/env python3
"""The capture's time anchor measured from the sky: how far the host-labelled time of the samples
is from GPS time, from the CNAV messages the nav decoders read.

    python3 gnss_skytime.py MARKS.json ... [--yaml chains_l5.yaml | --site-json FIX.json]
                            [--decoded-eph FILE,...] [--json OUT]

Each mark (nav_decode.py --timemarks-file) is a CRC-checked message: the host-labelled UTC of its
first symbol's edge, and the TOW count it carries, which is the satellite time at which the next
message starts. That message started TOW * 6 s - its length (6 s on L5, 12 s on L2C) of satellite
time; less the satellite clock and plus the range over c, it arrived at that GPS time. The
anchor error is the host label minus that, to the 1 ms of the records that place the edges. The
week is a type 10 message's week number, so it holds whatever date the host clock reads; with no
type 10 yet, the host's own week, to half a week.

Positive means the labels are late. bladeRFInput fixes the anchor once, from the system clock at
the first sample, so a host clock that was wrong when kotekan started shows here for the whole
run. The site comes from the broker yaml or a gnss_fix.py fix, or there is none (nominal light
time); the orbits from the BRDC cache or the decoded ephemerides. Only offsets are printed.
"""

import argparse
import json
import re
import sys
import warnings
from datetime import datetime, timezone

import numpy as np

import gnss_ephemeris as ge
import gnss_fix
import gnss_pvt
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


def anchor_error_s(mark, row, g_ref=None):
    """The host label minus the GPS time the message start arrived, s; row is the satellite's
    predict_all row at about that time (range_m, sat_clk_s). The TOW is placed in the week
    nearest g_ref (GPS seconds), by default the host label's own."""
    g_host = mark["t_host"] - GPS0_UNIX + LEAP_S
    g_ref = g_host if g_ref is None else g_ref
    t_tx = mark["tow"] * 6.0 - MSG_S[mark["signal"]]
    t_tx += 604800.0 * round((g_ref - t_tx) / 604800.0)
    return g_host - (t_tx - row["sat_clk_s"] + row["range_m"] / ge.C_LIGHT)


def week_ref(marks):
    """GPS seconds of the newest type 10 message, from its own week number: the week every mark
    is placed in, so a host date off by more than half a week reads right. None without one.
    """
    tens = [m for m in marks if m.get("wn") is not None and m.get("signal") in MSG_S]
    if not tens:
        return None
    m = max(tens, key=lambda m: m["t_host"])
    return m["wn"] * 604800.0 + m["tow"] * 6.0 - MSG_S[m["signal"]]


def measure(marks, orbits, site=None):
    """[(signal, prn, error s)] for the marks whose satellite the orbits know. A first pass on
    a nominal range (good to ~10 ms) finds the time to evaluate the satellite at, so a host clock
    hours off still finds it. With the site (ECEF) the light time is the range, to 1 ms;
    without one it stays nominal, ~10 ms, still enough for the date and the 20 ms epoch. An
    orbit with no clock (BeiDou's) leaves af0 out, under 1 ms."""
    out = []
    g_ref = week_ref(marks)
    for m in marks:
        if m.get("signal") not in MSG_S:
            continue
        t0 = m["t_host"] - anchor_error_s(m, NOMINAL_ROW, g_ref)
        t = ge.gpst_of_utc(datetime.fromtimestamp(t0, tz=timezone.utc))
        e = orbits.record("G", m["prn"], t)
        if e is None:
            continue
        rng = gnss_fix.geometry(e, site, t)[0] if site is not None else None
        clk = gnss_fix.sat_state(e, t - (rng or NOMINAL_ROW["range_m"]) / ge.C_LIGHT)[2]
        row = {
            "range_m": rng if rng is not None else NOMINAL_ROW["range_m"],
            "sat_clk_s": clk or 0.0,
        }
        out.append((m["signal"], m["prn"], anchor_error_s(m, row, g_ref)))
    return out


def run(marks_paths, yaml=None, site_json=None, decoded=None):
    """measure() on marks files, the site from a broker yaml or a gnss_fix.py --json fix (or
    none), the orbits from decoded ephemeris files or the BRDC cache (records up to
    sky.HINT_MAX_AGE_S old: 1 ms is 300 km of range)."""
    marks = []
    for p in marks_paths:
        marks += json.load(open(p)).get("marks", [])
    warnings.simplefilter("ignore")
    site = None
    if yaml:
        site = np.array(gnss_pvt._llh_to_ecef(*site_of(yaml)))
    elif site_json:
        site = np.array(json.load(open(site_json))["ecef"])
    orbits = gnss_fix.Orbits(decoded, max_age_s=HINT_MAX_AGE_S)
    return measure(marks, orbits, site), site is not None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("marks", nargs="+", help="nav_decode.py --timemarks-file outputs")
    ap.add_argument("--yaml", help="the broker yaml holding the site")
    ap.add_argument("--site-json", help="or a gnss_fix.py --json fix as the site")
    ap.add_argument(
        "--decoded-eph", help="comma-separated decoded ephemeris files, not the BRDC"
    )
    ap.add_argument("--json", help="write the result here")
    a = ap.parse_args(argv)
    res, sited = run(
        a.marks,
        a.yaml,
        a.site_json,
        a.decoded_eph.split(",") if a.decoded_eph else None,
    )
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
        "light_time": "range" if sited else "nominal",
    }
    print(
        "capture anchor %+.3f ms from the sky (%d marks, %d satellites, %+.3f..%+.3f ms, %s "
        "light time)"
        % (
            doc["anchor_error_ms"],
            doc["n_marks"],
            doc["n_sats"],
            *doc["spread_ms"],
            doc["light_time"],
        )
    )
    if a.json:
        with open(a.json, "w") as f:
            json.dump(doc, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
