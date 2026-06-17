#!/usr/bin/env python3
"""Post-process GpsReplicaCorrelator records into per-satellite beam tracks.

The kotekan ``GpsReplicaCorrelator`` stage writes, per integration, one record
per PRN to a raw buffer (via ``rawFileWrite``). Each record is::

    9 x float32: prn, doppler_hz, code_phase_chips, peak_amp, peak_re,
                 peak_im, snr, nav_sign, phase_cont
    1 x float64: UTC seconds (trailing two float32 slots)

This script reads those records and, for each detection, attaches the
satellite's apparent altitude/azimuth at the record's UTC for a given observer,
using skyfield + GPS TLEs (Celestrak). The result is the beam-response sample
set: as a satellite transits, its ``peak_amp`` vs ``(alt, az)`` traces the beam.

alt/az is intentionally done here in post-processing rather than in the C++ DSP
stage: it is a slow, pure function of (PRN, time, observer) and has no place in
the per-millisecond correlation path.

Examples
--------
    # dump records (no skyfield needed) as CSV
    python gps_beamtrack.py ./gps_records --no-altaz > tracks.csv

    # attach alt/az for an observer, locked detections only
    python gps_beamtrack.py ./gps_records \
        --lat 43.66 --lon -79.40 --alt 100 --locked-only > tracks.csv
"""

import argparse
import glob
import os
import struct
import sys
from datetime import datetime, timezone

import numpy as np

RECORD_SLOTS = 11          # float32 slots per PRN record
RECORD_BYTES = RECORD_SLOTS * 4
UTC_SLOT = 9               # trailing float64 occupies slots 9-10
FIELDS = ["prn", "doppler_hz", "code_phase_chips", "peak_amp", "peak_re",
          "peak_im", "snr", "nav_sign", "phase_cont"]

# Celestrak GPS operational TLEs.
DEFAULT_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle"


def _infer_n_prn(path):
    """Infer PRNs-per-frame from a single-frame file size: 4-byte meta header
    plus n_prn * RECORD_BYTES of payload."""
    size = os.path.getsize(path)
    payload = size - 4
    if payload <= 0 or payload % RECORD_BYTES != 0:
        raise ValueError(
            "%s: size %d not a single frame of %d-byte records (pass --n-prn for "
            "multi-frame files)" % (path, size, RECORD_BYTES))
    return payload // RECORD_BYTES


def read_records(paths, n_prn=None):
    """Read all records from the given file paths into a structured array with
    columns FIELDS + 'utc'. Each file is a sequence of frames, each frame a
    4-byte metadata length (expected 0) followed by n_prn records."""
    if n_prn is None:
        n_prn = _infer_n_prn(paths[0])
    frame_stride = 4 + n_prn * RECORD_BYTES

    rows = []
    for path in paths:
        buf = open(path, "rb").read()
        for off in range(0, len(buf) - frame_stride + 1, frame_stride):
            meta = struct.unpack_from("<I", buf, off)[0]
            base = off + 4 + meta
            for p in range(n_prn):
                ro = base + p * RECORD_BYTES
                vals = np.frombuffer(buf, dtype="<f4", count=9, offset=ro)
                utc = struct.unpack_from("<d", buf, ro + UTC_SLOT * 4)[0]
                rows.append(tuple(vals) + (utc,))

    dtype = [(f, "<f4") for f in FIELDS] + [("utc", "<f8")]
    return np.array(rows, dtype=dtype)


def _load_gps_satellites(tle_source):
    """Return {prn: EarthSatellite} from a Celestrak TLE file/URL. Names carry
    the PRN as '... (PRN NN)'."""
    import re
    from skyfield.api import load

    sats = load.tle_file(tle_source)
    by_prn = {}
    for s in sats:
        m = re.search(r"PRN\s*(\d+)", s.name or "")
        if m:
            by_prn[int(m.group(1))] = s
    return by_prn


def attach_altaz(records, lat, lon, alt_m, tle_source=DEFAULT_TLE_URL):
    """Return (alt_deg, az_deg) arrays aligned with `records`, NaN where the PRN
    has no TLE. Requires skyfield."""
    from skyfield.api import load, wgs84

    ts = load.timescale()
    observer = wgs84.latlon(lat, lon, elevation_m=alt_m)
    by_prn = _load_gps_satellites(tle_source)

    alt = np.full(len(records), np.nan, dtype=float)
    az = np.full(len(records), np.nan, dtype=float)
    # Group by PRN to reuse each satellite object; skyfield vectorizes over time.
    for prn in np.unique(records["prn"].astype(int)):
        sat = by_prn.get(int(prn))
        if sat is None:
            continue
        sel = records["prn"].astype(int) == prn
        times = ts.from_datetimes(
            [datetime.fromtimestamp(u, tz=timezone.utc) for u in records["utc"][sel]])
        topo = (sat - observer).at(times).altaz()
        alt[sel] = topo[0].degrees
        az[sel] = topo[1].degrees
    return alt, az


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="record file, directory, or glob")
    ap.add_argument("--n-prn", type=int, default=None,
                    help="PRNs per frame (default: infer from a single-frame file)")
    ap.add_argument("--lat", type=float, help="observer latitude, deg")
    ap.add_argument("--lon", type=float, help="observer longitude, deg")
    ap.add_argument("--alt", type=float, default=0.0, help="observer height, m")
    ap.add_argument("--tle", default=DEFAULT_TLE_URL, help="TLE file path or URL")
    ap.add_argument("--no-altaz", action="store_true", help="skip alt/az (just dump records)")
    ap.add_argument("--locked-only", action="store_true",
                    help="keep only phase-locked detections (nav_sign != 0)")
    ap.add_argument("--min-snr", type=float, default=0.0, help="drop records below this SNR")
    ap.add_argument("--out", default="-", help="output CSV path ('-' = stdout)")
    args = ap.parse_args(argv)

    if os.path.isdir(args.path):
        paths = sorted(glob.glob(os.path.join(args.path, "*.raw")))
    else:
        paths = sorted(glob.glob(args.path))
    if not paths:
        ap.error("no record files matched: %s" % args.path)

    recs = read_records(paths, args.n_prn)
    if args.locked_only:
        recs = recs[recs["nav_sign"] != 0.0]
    if args.min_snr > 0.0:
        recs = recs[recs["snr"] >= args.min_snr]

    cols = list(FIELDS) + ["utc"]
    alt = az = None
    if not args.no_altaz:
        if args.lat is None or args.lon is None:
            ap.error("--lat/--lon required for alt/az (or pass --no-altaz)")
        try:
            alt, az = attach_altaz(recs, args.lat, args.lon, args.alt, args.tle)
            cols += ["alt_deg", "az_deg"]
        except ImportError:
            print("warning: skyfield not installed; emitting records without alt/az",
                  file=sys.stderr)

    out = sys.stdout if args.out == "-" else open(args.out, "w")
    out.write(",".join(cols) + "\n")
    for i, r in enumerate(recs):
        vals = ["%d" % int(r["prn"])] + ["%.6g" % r[c] for c in cols[1:len(FIELDS)]]
        vals.append("%.6f" % r["utc"])
        if alt is not None:
            vals += ["%.4f" % alt[i], "%.4f" % az[i]]
        out.write(",".join(vals) + "\n")
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
