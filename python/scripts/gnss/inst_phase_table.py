#!/usr/bin/env python3
"""The fleet-wide instrumental phase per input and channel, for carrying a lane from the channel
it was measured on to another (#246, docs/CHORD_FLEET_LANES.md "The instrumental phase table").

A lane carried from f_src to f needs each input's instrumental phase change between the two
channels as well as the geometric one. Measured from a Tau A calibration (fixtures/hi21_feasibility
cal_track.py + cal_solve.py) with the geometry toward Tau A taken off, that phase is one number per
input and channel; one delay per input across the band leaves 6-10 deg rms of it (2026-11-26).
On the 2026-11-25 G03 lane dump, a gps_l2c lane carried 12 MHz with it reads overlap^2 0.96
against the sky's top mode where the geometric carry read 0.91-0.92 and one delay per input
0.93-0.94 (fixtures/deproj_1125/xfer_phase_test.py).

The table holds the calibration's phase at knots (its solved channels, at most one per
KNOT_MHZ) in the frame of the science N^2's eigenvectors, the frame a lane is used in after
lane_conj. Between knots the phase is interpolated linearly per input (the calibration itself is
a straight line through its solved channels inside the projected range). A lane in that frame
is carried by phi(f) - phi(f_src).

The calibration is valid under the F-engine start it was made under (made_frame0). For data
under another start every input's phase turns by 2 pi f n Ts, n its whole-sample steps since
(fengine_steps.steps_between with pol_step: the table combines both pol halves); turned()
refuses while a start in between is unmeasured, and the relay then sends no table.

  build CAL.npz OUT.json [--frame0 FILE] [--lo MHZ] [--hi MHZ]
  models TABLE.json OUT.json BAND:LO-HI [--models DIR] [--passes N] [--frame0 FILE]
        The table with knots at freq_ids LO and HI for BAND (l1ca), outside the calibration's
        range, from the element models the watch keeps (elem_shared_ref.py, DIR/models): per
        input the negative of the model's phase, which is the table's phase plus one constant per
        band inside its range (8-14 deg rms on l3oc, b2b, l2c, l2of, b3i, 2026-11-27). Refused
        when those bands disagree with the table by more than MODEL_RMS_MAX_DEG, or when the
        table was made under another F-engine start than the models.
  show TABLE.json [--steps STEPS.json] [--data-frame0 N]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fengine_steps  # noqa: E402

C = 299792458.0
CH_HZ = 195312.5
KNOT_MHZ = 0.8  # at most one knot per this many MHz
FRAME0 = "/home/kvand/gnss/builds/gw1202/frame0.json"
DIR = "/home/kvand/gnss/fixtures/obs/inst_phase"  # the relay reads DIR/current.json
# Past the projected range (1140.0-1304.9 MHz) by a few MHz: outside its knots a node holds the
# edge knot's phase (gnssFleetLane.hpp InstPhaseTable::phase_at).
LO_MHZ, HI_MHZ = 1135.0, 1310.0
MODELS = "/home/kvand/gnss/fixtures/obs/elem_ref/models"
# The bands whose models check the table: inside its range, by their carriers (MHz).
MODEL_CHECK_MHZ = {
    "l3oc": 1202.025,
    "b2b": 1207.14,
    "l2c": 1227.6,
    "l2of": 1246.0,
    "b3i": 1268.52,
}
MODEL_RMS_MAX_DEG = 25.0


LAT = math.radians(
    49.3199845
)  # as fixtures/hi21_feasibility/hi21_lib.py, which made the calibration


def enu(ha, dec, lat=LAT):
    """Unit vector (east, north, up) of a source at hour angle ha, declination dec (radians)."""
    e = -math.cos(dec) * math.sin(ha)
    n = math.cos(lat) * math.sin(dec) - math.sin(lat) * math.cos(dec) * math.cos(ha)
    u = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha)
    return np.array([e, n, u])


def start_of(frame0_file, t):
    """The frame0_int of the last F-engine start at or before unix time t."""
    with open(frame0_file) as f:
        starts = [s["frame0_int"] for s in json.load(f) if s["frame0_int"] <= t]
    if not starts:
        raise ValueError("no F-engine start before %s" % t)
    return max(starts)


def build(cal, frame0_file=FRAME0, lo_mhz=LO_MHZ, hi_mhz=HI_MHZ, source=""):
    """The table (a dict) from a calibration npz loaded as `cal`."""
    fid = np.asarray(cal["k"]) + 1536
    f = fid * CH_HZ
    n0 = enu(float(cal["ha_on"]), float(cal["dec"]))
    u = np.asarray(cal["G"]).astype(complex) * np.exp(
        -2j * np.pi * np.outer(f / C, np.asarray(cal["pos"]) @ n0)
    )
    ok = np.asarray(cal["ok_g"]) & (np.abs(u) > 0)
    if "kept" in cal:
        ok &= np.repeat(np.asarray(cal["kept"]), ok.shape[1] // 2, axis=1)
    rows = np.flatnonzero((f >= lo_mhz * 1e6) & (f <= hi_mhz * 1e6) & ok.any(axis=1))
    knots, last = [], -1e30
    for r in rows:
        if f[r] - last >= KNOT_MHZ * 1e6:
            knots.append(r)
            last = f[r]
    phase = [
        [
            round(math.degrees(math.atan2(x.imag, x.real)), 2) if g else None
            for x, g in zip(u[r], ok[r])
        ]
        for r in knots
    ]
    t_tr = float(cal["t_tr"])
    return {
        "what": "instrumental phase per input (deg), science-N^2 frame, geometry toward the "
        "calibrator off; carry a lane by phi(f) - phi(f_src) (inst_phase_table.py)",
        "source": source,
        "t_tr": t_tr,
        "t_tr_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_tr)),
        "made_frame0": start_of(frame0_file, t_tr),
        "adc_sample_ns": 0.3125,
        "n_inputs": int(u.shape[1]),
        "fids": [int(fid[r]) for r in knots],
        "phase_deg": phase,
    }


def turned(table, steps_table, data_frame0):
    """The table's knots turned to data under `data_frame0` by the steps since its start: a copy
    with made_frame0 replaced by data_frame0. An input a row in between did not read becomes
    None. Raises ValueError when the steps (or a pol_step) between are not measured."""
    steps = fengine_steps.steps_between(
        steps_table, table["made_frame0"], data_frame0, pol_step=True
    )
    ts = table["adc_sample_ns"] * 1e-9
    out = dict(table, made_frame0=data_frame0, turned_from=table["made_frame0"])
    out["phase_deg"] = []
    for fid, row in zip(table["fids"], table["phase_deg"]):
        f = fid * CH_HZ
        out["phase_deg"].append(
            [
                None
                if p is None or s is None
                else round((p + 360.0 * f * s * ts + 180.0) % 360.0 - 180.0, 2)
                for p, s in zip(row, steps)
            ]
        )
    return out


def phase_at(table, fid):
    """Per input, the table's phase (radians) at fid, linear between the knots on either side
    that hold the input (the nearer knot's value outside them); NaN for an input no knot holds."""
    fids = np.asarray(table["fids"], dtype=float)
    P = np.array(
        [
            [np.nan if p is None else math.radians(p) for p in row]
            for row in table["phase_deg"]
        ]
    )
    out = np.full(P.shape[1], np.nan)
    for i in range(P.shape[1]):
        ok = np.isfinite(P[:, i])
        if not ok.any():
            continue
        x, y = fids[ok], np.unwrap(P[ok, i])
        out[i] = np.interp(fid, x, y)
    return out


def carry(table, v, fid_src, fid):
    """A lane v (science-N^2 frame, one entry per input) carried in instrumental phase from fid_src
    to fid; inputs the table does not hold are left as they are."""
    d = phase_at(table, fid) - phase_at(table, fid_src)
    return np.asarray(v) * np.where(np.isfinite(d), np.exp(1j * np.nan_to_num(d)), 1.0)


def model_phase(passes, band):
    """Per input, the band's element model over the watch's passes (each turned onto the
    first in global phase), as a complex array; NaN where an input has none."""
    acc = None
    for m in passes:
        z = np.array(
            [complex(*x) if x else np.nan for x in m["bands"][band]], dtype=complex
        )
        if acc is None:
            acc = z
            continue
        ok = np.isfinite(z) & np.isfinite(acc)
        acc = acc + z * np.exp(-1j * np.angle(np.sum(z[ok] * np.conj(acc[ok]))))
    return acc


def from_models(table, passes, band, lo_fid, hi_fid):
    """The table with knots lo_fid and hi_fid for band from the models in passes, and the rms
    (deg) of each MODEL_CHECK_MHZ band against the table."""
    rms = {}
    for b, mhz in MODEL_CHECK_MHZ.items():
        d = -np.angle(model_phase(passes, b)) - phase_at(table, mhz * 1e6 / CH_HZ)
        ok = np.isfinite(d)
        r = np.angle(np.exp(1j * (d[ok] - np.angle(np.sum(np.exp(1j * d[ok]))))))
        rms[b] = round(math.degrees(math.sqrt(np.mean(r ** 2))), 1)
    held = np.isfinite(phase_at(table, table["fids"][0]))
    ph = -np.angle(model_phase(passes, band))
    row = [
        round(math.degrees(x), 2) if h and np.isfinite(x) else None
        for x, h in zip(ph, held)
    ]
    out = dict(table)
    knots = sorted(
        list(zip(table["fids"], table["phase_deg"])) + [(lo_fid, row), (hi_fid, row)]
    )
    out["fids"] = [f for f, _ in knots]
    out["phase_deg"] = [r for _, r in knots]
    return out, rms


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("cal")
    b.add_argument("out")
    b.add_argument("--frame0", default=FRAME0)
    b.add_argument("--lo", type=float, default=LO_MHZ)
    b.add_argument("--hi", type=float, default=HI_MHZ)
    m = sub.add_parser("models")
    m.add_argument("table")
    m.add_argument("out")
    m.add_argument("band", help="BAND:LO-HI, the knots' freq_ids (l1ca:8061-8071)")
    m.add_argument("--models", default=MODELS)
    m.add_argument("--passes", type=int, default=12)
    m.add_argument("--frame0", default=FRAME0)
    s = sub.add_parser("show")
    s.add_argument("table")
    s.add_argument("--steps", default=fengine_steps.TABLE)
    s.add_argument("--data-frame0", type=int)
    a = ap.parse_args()
    if a.cmd == "build":
        tab = build(np.load(a.cal), a.frame0, a.lo, a.hi, source=os.path.abspath(a.cal))
        with open(a.out + ".tmp", "w") as f:
            json.dump(tab, f)
        os.replace(a.out + ".tmp", a.out)
        print(
            "%s: %d knots %.1f-%.1f MHz, made under start %d (Tau A %s)"
            % (
                a.out,
                len(tab["fids"]),
                tab["fids"][0] * CH_HZ / 1e6,
                tab["fids"][-1] * CH_HZ / 1e6,
                tab["made_frame0"],
                tab["t_tr_utc"],
            )
        )
        return
    with open(a.table) as f:
        tab = json.load(f)
    if a.cmd == "models":
        band, _, rng = a.band.partition(":")
        lo, _, hi = rng.partition("-")
        files = sorted(os.listdir(a.models))[-a.passes :]
        passes = [json.load(open(os.path.join(a.models, f))) for f in files]
        t_last = float(files[-1].split(".")[0])
        if start_of(a.frame0, t_last) != tab["made_frame0"]:
            raise SystemExit(
                "the table was made under start %d, the models under %d: build it again"
                % (tab["made_frame0"], start_of(a.frame0, t_last))
            )
        out, rms = from_models(tab, passes, band, int(lo), int(hi))
        print("models %s .. %s: rms against the table %s" % (files[0], files[-1], rms))
        if max(rms.values()) > MODEL_RMS_MAX_DEG:
            raise SystemExit("the models do not follow the table: nothing written")
        out["models"] = {
            "band": band,
            "from": files[0],
            "to": files[-1],
            "rms_deg": rms,
        }
        with open(a.out + ".tmp", "w") as f:
            json.dump(out, f)
        os.replace(a.out + ".tmp", a.out)
        print("%s: %d knots, %s at %s-%s" % (a.out, len(out["fids"]), band, lo, hi))
        return
    n_none = sum(p is None for row in tab["phase_deg"] for p in row)
    print(
        "%d knots, made under start %d (Tau A %s), %d of %d entries empty"
        % (
            len(tab["fids"]),
            tab["made_frame0"],
            tab["t_tr_utc"],
            n_none,
            len(tab["fids"]) * tab["n_inputs"],
        )
    )
    if a.data_frame0:
        try:
            turned(tab, fengine_steps.load(a.steps), a.data_frame0)
            print("turns to start %d" % a.data_frame0)
        except ValueError as e:
            print("does not turn to start %d: %s" % (a.data_frame0, e))


if __name__ == "__main__":
    main()
