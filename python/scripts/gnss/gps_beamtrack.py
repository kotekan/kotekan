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
from datetime import datetime, timedelta, timezone

import numpy as np

RECORD_SLOTS = 11          # float32 slots per PRN record
RECORD_BYTES = RECORD_SLOTS * 4
UTC_SLOT = 9               # trailing float64 occupies slots 9-10
FIELDS = ["prn", "doppler_hz", "code_phase_chips", "peak_amp", "peak_re",
          "peak_im", "snr", "nav_sign", "phase_cont"]

# Celestrak GPS operational TLEs.
DEFAULT_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle"
# Celestrak GLONASS operational TLEs. Names carry the Uragan number, with GLONASS-K marked by a
# trailing K -- the capability marker glonass_block_by_prn() reads. Slot numbers come from the
# IGS SINEX by catalog number (these names have no PRN field), which load_gps_satellites()
# already does as its unnamed-satellite fallback.
GLONASS_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=glo-ops&FORMAT=tle"


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


# Galileo TLE names ('GSAT0207 (GALILEO 14)') carry the GSAT platform number, NOT the E-PRN
# (SVID) the spreading codes are keyed by. The mapping is operator metadata: transcribed from
# the ESA GSC constellation-information page (gsc-europa.eu/system-service-status/
# constellation-information, fetched 2026-07-11; machine-extracted, 32 entries). It changes
# only at launches/decommissionings -- a TLE GSAT missing here is skipped with a warning
# (decommissioned or not yet mapped; refresh from the GSC page). GSAT0104 (E20, failed 2014)
# and GSAT0205 (E24) absent from the page are intentionally unmapped.
GSAT_TO_PRN = {
    "GSAT0101": 11,
    "GSAT0102": 12,
    "GSAT0103": 19,
    "GSAT0201": 18,
    "GSAT0202": 14,
    "GSAT0203": 26,
    "GSAT0204": 22,
    "GSAT0206": 30,
    "GSAT0207": 7,
    "GSAT0208": 8,
    "GSAT0209": 9,
    "GSAT0210": 1,
    "GSAT0211": 2,
    "GSAT0212": 3,
    "GSAT0213": 4,
    "GSAT0214": 5,
    "GSAT0215": 21,
    "GSAT0216": 25,
    "GSAT0217": 27,
    "GSAT0218": 31,
    "GSAT0219": 36,
    "GSAT0220": 13,
    "GSAT0221": 15,
    "GSAT0222": 33,
    "GSAT0223": 34,
    "GSAT0224": 10,
    "GSAT0225": 29,
    "GSAT0226": 23,
    "GSAT0227": 6,
    "GSAT0232": 16,
    "GSAT0233": 28,
    "GSAT0234": 32,
}


def _igs_prn_by_catnum():
    """{NORAD catalog number: (constellation letter, prn int)} from the IGS satellite
    metadata SINEX (files.igs.org/pub/station/general/igs_satellite_metadata.snx) -- the
    authoritative multi-GNSS SVN/COSPAR/SatCat/PRN registry. Constellations whose TLE names
    carry no PRN (BeiDou 'BEIDOU-3 M23', and a backstop for Galileo) resolve through this:
    TLEs carry the catalog number, the SINEX maps it to the CURRENTLY-VALID PRN assignment.
    Cached beside the TLEs for 30 days; empty dict if unavailable (offline -> those sats
    are skipped with a warning)."""
    import re, time, urllib.request
    cache = os.path.join(os.path.expanduser("~"), ".cache", "kotekan_gps")
    os.makedirs(cache, exist_ok=True)
    snx = os.path.join(cache, "igs_satellite_metadata.snx")
    if not os.path.exists(snx) or time.time() - os.path.getmtime(snx) > 30 * 86400:
        try:
            urllib.request.urlretrieve(
                "https://files.igs.org/pub/station/general/igs_satellite_metadata.snx", snx)
        except Exception as e:
            if not os.path.exists(snx):
                print("[gps_beamtrack] IGS satellite metadata unavailable (%s)" % e,
                      file=sys.stderr)
                return {}
    svn_cat, svn_prn = {}, {}
    section = None
    now = time.strftime("%Y:%j:00000", time.gmtime())
    for line in open(snx, errors="replace"):
        if line.startswith("+SATELLITE/"):
            section = line.strip().lstrip("+")
            continue
        if line.startswith("-SATELLITE/"):
            section = None
            continue
        if line.startswith("*") or not line.strip():
            continue
        f = line.split()
        if section == "SATELLITE/IDENTIFIER" and len(f) >= 3 and f[2].isdigit():
            svn_cat[f[0]] = int(f[2])
        elif section == "SATELLITE/PRN" and len(f) >= 4:
            svn, t_from, t_to, prn = f[0], f[1], f[2], f[3]
            current = (t_to == "0000:000:00000") or (t_to >= now)
            if current and re.fullmatch(r"[A-Z]\d{2}", prn):
                svn_prn[svn] = prn
    out = {}
    for svn, cat in svn_cat.items():
        prn = svn_prn.get(svn)
        if prn:
            out[cat] = (prn[0], int(prn[1:]))
    return out


def load_gps_satellites(tle_source):
    """Return {prn: EarthSatellite} from a Celestrak TLE file/URL.

    GPS names carry the PRN as '... (PRN NN)'; Galileo names carry 'GSAT####' which is
    mapped to the E-PRN via GSAT_TO_PRN -- so the same loader (and everything downstream:
    visibility, predicted Doppler, az/el) serves both constellations; the caller picks the
    constellation by the TLE GROUP in the url (gps-ops / galileo).

    A URL is downloaded into a per-user cache dir (~/.cache/kotekan_gps), not the
    current directory: skyfield names the cache after the URL basename, so the
    default loader otherwise drops a stray 'gp.php' in the repo root on every run.
    The cache is shared across scripts and reused until stale, so this also avoids
    re-hitting Celestrak each broker cycle. Local file paths load in place."""
    import re
    from skyfield.api import Loader, load

    if isinstance(tle_source, str) and tle_source.startswith(("http://", "https://")):
        cache = os.path.join(os.path.expanduser("~"), ".cache", "kotekan_gps")
        os.makedirs(cache, exist_ok=True)
        # Cache PER GROUP: every Celestrak gp.php URL shares the basename "gp.php", and
        # skyfield keys its cache on the basename -- so gps-ops/galileo/beidou would all
        # silently REUSE whichever group was fetched first (observed: the viewer plotted
        # Galileo and BeiDou "positions" exactly on top of the GPS sats, and the E/C brokers
        # hinted GPS Doppler at E/C PRNs -- Galileo detected anyway only because the wide
        # pre-solve margins covered the error). Name the file after the GROUP query param.
        m = re.search(r"GROUP=([A-Za-z0-9_-]+)", tle_source)
        fname = "tle_%s.txt" % (m.group(1) if m else "default")
        sats = Loader(cache, verbose=False).tle_file(tle_source, filename=fname)
    else:
        sats = load.tle_file(tle_source)
    by_prn = {}
    for s in sats:
        name = s.name or ""
        m = re.search(r"PRN\s*(\d+)", name)
        if m:
            by_prn[int(m.group(1))] = s
            continue
        # BeiDou names carry the PRN as '... (C46)' -- and it DISAGREES with the IGS SINEX
        # mapping for several sats (C46/C48 mapped to stale assignments there), so the name
        # is authoritative when present; the SINEX below is the fallback for unnamed sats.
        m = re.search(r"\(C(\d+)\)", name)
        if m:
            by_prn[int(m.group(1))] = s
            continue
        g = re.search(r"GSAT0\d{3}", name)
        if g:
            prn = GSAT_TO_PRN.get(g.group(0))
            if prn is not None:
                by_prn[prn] = s
            else:
                print("[gps_beamtrack] unmapped Galileo %s -- refresh GSAT_TO_PRN"
                      % g.group(0), file=sys.stderr)
            continue
        # No PRN in the name (BeiDou et al.): NORAD catalog number -> IGS SINEX registry.
        global _IGS_CAT_MAP
        if _IGS_CAT_MAP is None:
            _IGS_CAT_MAP = _igs_prn_by_catnum()
        hit = _IGS_CAT_MAP.get(getattr(s.model, "satnum", -1))
        if hit:
            by_prn[hit[1]] = s
    return by_prn


_IGS_CAT_MAP = None


def gps_block_by_prn(tle_source=DEFAULT_TLE_URL, _sats=None):
    """Return {prn: block} (e.g. 'IIF', 'IIR', 'IIRM', 'III') read from the Celestrak TLE
    NAMES ('GPS BIII-10 (PRN 13)'). Satellite BLOCK/signal capability is NOT carried by any
    almanac or the TLE elements themselves -- it is metadata -- but Celestrak encodes it in
    the sat name, so we read it from the SAME live feed the visibility uses instead of
    hardcoding a list that goes stale as satellites launch / commission / retire."""
    import re
    sats = _sats if _sats is not None else load_gps_satellites(tle_source)
    blocks = {}
    for prn, s in sats.items():
        m = re.search(r"GPS\s+B([A-Z]+)-?\d", s.name or "")
        blocks[prn] = m.group(1) if m else "?"
    return blocks


def glonass_block_by_prn(tle_source=GLONASS_TLE_URL, _sats=None):
    """Return {slot: block} for GLONASS, where block is 'K' for a GLONASS-K satellite and 'M'
    otherwise. Celestrak names carry the Uragan number in parentheses and mark the K-series with
    a trailing K -- 'COSMOS 2501 (702K)' is a K, 'COSMOS 2456 (730)' is not. Exactly the trick
    gps_block_by_prn() uses on 'GPS BIII-10 (PRN 13)': read the capability from the SAME live
    feed the visibility comes from, so it tracks launches and retirements by itself.

    ★ This is what gates L3OC: only the K satellites carry GLONASS's CDMA signals (7 of 28
    operational as of 2026-08). 'M' lumps together GLONASS-M and anything unrecognised -- the
    distinction that matters here is K vs not-K, and calling the remainder 'M' is accurate for
    every currently-operational non-K satellite."""
    import re
    sats = _sats if _sats is not None else load_gps_satellites(tle_source)
    blocks = {}
    for slot, s in sats.items():
        m = re.search(r"\((\d+)(K?)\)", s.name or "")
        blocks[slot] = "K" if (m and m.group(2)) else "M"
    return blocks


def _signal_block_filter(signal):
    """Predicate over a GPS block name ('IIR','IIRM','IIF','III',...) selecting the blocks that
    broadcast `signal`. Returns None when EVERY GPS satellite carries the signal (no filter).
    Block name is as read from the Celestrak TLE by gps_block_by_prn(). Signal is the config's
    `signal:` token (e.g. GPS_L1CA, GPS_L2C_CM, GPS_L5_Q, GPS_L1C_P). Which block first carried
    each civil signal: L1 C/A -> all; L2C -> Block IIR-M+; L5 -> Block IIF+; L1C -> Block III."""
    s = (signal or "").upper()
    if "L1CA" in s:                         # L1 C/A: every GPS satellite (check before 'L1C')
        return None
    if "L5" in s:                           # L5: Block IIF and newer
        return lambda b: b == "IIF" or b.startswith("III")
    if "L2C" in s or "L2_CM" in s:          # L2C: Block IIR-M and newer
        return lambda b: b == "IIRM" or b == "IIF" or b.startswith("III")
    if "L1C" in s:                          # L1C: Block III only
        return lambda b: b.startswith("III")
    return None                             # unknown -> don't filter


def _glonass_block_filter(signal):
    """The GLONASS counterpart of _signal_block_filter. The CDMA signals (L3OC, and later L2OC /
    L1OC) are GLONASS-K only; the FDMA signals (L1OF/L2OF) are on every satellite."""
    s = (signal or "").upper()
    if "OC" in s:                           # L3OC / L2OC / L1OC -- CDMA, K satellites only
        return lambda b: b == "K"
    return None                             # L1OF/L2OF (and unknown) -- every satellite


def glonass_fdma_trackable(_cache={}):
    """Slots we can actually track on GLONASS FDMA = those with a known frequency-channel number.

    ★ FDMA capability is not about the satellite's generation, it is about whether we KNOW ITS
    CARRIER. Every satellite transmits L1OF/L2OF, but they are separated by frequency, so a slot
    whose k we cannot read is one we would have to search at the wrong frequency -- it has no
    business in a seed or hint list. That makes "has a k" the natural capability test, filling the
    same role the GPS block name and the GLONASS 'K' marker play for the CDMA signals.

    Cached for an hour; empty set on failure => caller disables the filter rather than darking
    the chain (the standing fail-open-is-worse-than-fail-loud tradeoff for capability lookups)."""
    import time
    now = time.monotonic()
    if _cache.get("t") and now - _cache["t"] < 3600.0:
        return _cache["v"]
    try:
        import gnss_ephemeris as ge
        v = set(ge.glonass_freq_channels())
    except Exception as e:
        print("[gps_beamtrack] GLONASS frequency plan unavailable (%s); FDMA capability filter "
              "DISABLED" % e, file=sys.stderr)
        v = set()
    _cache["t"], _cache["v"] = now, v
    return v


def signal_capable_prns(signal, tle_source=DEFAULT_TLE_URL, _sats=None):
    """PRNs whose satellite block actually broadcasts `signal`, from the live Celestrak names, so
    it tracks launches/commissioning automatically. Returns ALL PRNs when the signal is on every
    satellite (GPS L1 C/A, GLONASS L1OF/L2OF) or is unknown. Non-transmitting sats belong nowhere
    near a search list -- correlating their (absent) code only wastes channels and manufactures
    false detections.

    Dispatches on the signal token's constellation prefix: GLO_* reads the GLONASS K marker (and
    defaults to the glo-ops TLE group), everything else reads the GPS block name."""
    sig_u = (signal or "").upper()
    if sig_u.startswith("GLO_") and "OF" in sig_u:
        # FDMA: capability = "we know this satellite's carrier" (see glonass_fdma_trackable).
        cap = glonass_fdma_trackable()
        if cap:
            return cap
        # fall through to the block path (which returns everything) if the plan is unavailable
    if (signal or "").upper().startswith("GLO_"):
        # Guard the easy mistake: a GLONASS signal asked for against the GPS default TLE group
        # would find no 'K' marker on anything and return an EMPTY set, which callers treat as
        # "lookup failed -> filter disabled". Substitute the right group instead of silently
        # producing a filter that does nothing.
        blocks = glonass_block_by_prn(
            GLONASS_TLE_URL if tle_source == DEFAULT_TLE_URL else tle_source, _sats)
        pred = _glonass_block_filter(signal)
    else:
        blocks = gps_block_by_prn(tle_source, _sats)
        pred = _signal_block_filter(signal)
    if pred is None:
        return set(blocks)
    return {prn for prn, b in blocks.items() if pred(b)}


def l1c_capable_prns(tle_source=DEFAULT_TLE_URL, _sats=None):
    """PRNs that broadcast L1C = GPS Block III onward (TLE block name starts 'III'). Derived
    from the live Celestrak names, so it tracks launches/commissioning automatically. (Older
    IIR/IIR-M/IIF have no L1C.)"""
    return signal_capable_prns("GPS_L1C_P", tle_source, _sats)


def predict_dopplers(lat, lon, alt_m, prns=None, t_utc=None, f_carrier_hz=1575.42e6,
                     tle_source=DEFAULT_TLE_URL, _sats=None):
    """Predict {prn: (doppler_hz, doppler_rate_hz_per_s, elevation_deg, range_m)} for GPS sats
    at a receiver (lat, lon, alt_m WGS84) and time t_utc (UTC datetime, default now).
    range_m is the slant range (for propagation delay, ~64-89 ms; TLE accuracy ~km -> ~us,
    far better than the ~10 ms the CL time-assist needs). Access entries by INDEX so the
    tuple can grow.

    This is the GEOMETRIC (line-of-sight range-rate) Doppler only. It does NOT
    include the receiver clock-frequency error, which adds a *common* offset across
    all PRNs (airspy TCXO ~1 ppm -> ~+-1.5 kHz at L1); solve that one bias from the
    measured Doppler of an acquired sat, then it applies to every prediction.

    Sign convention: +Doppler = satellite approaching (received carrier shifted up),
    matching the GNSS pipeline's reported Doppler. Requires skyfield. Pass _sats
    (a {prn: EarthSatellite} from load_gps_satellites) to avoid reloading TLEs.
    """
    from skyfield.api import load, wgs84
    C = 299792458.0
    ts = load.timescale()
    observer = wgs84.latlon(lat, lon, elevation_m=alt_m)
    by_prn = _sats if _sats is not None else load_gps_satellites(tle_source)
    if t_utc is None:
        t_utc = datetime.now(tz=timezone.utc)
    dt = 0.5
    t0 = ts.from_datetime(t_utc)
    t1 = ts.from_datetime(t_utc + timedelta(seconds=dt))

    def _doppler(sat, t):
        g = (sat - observer).at(t)
        r = g.position.km * 1e3          # m  (sat relative to observer)
        v = g.velocity.km_per_s * 1e3    # m/s
        range_rate = float(np.dot(r, v) / np.linalg.norm(r))  # +ve = receding
        return -f_carrier_hz * range_rate / C                 # +ve = approaching

    out = {}
    for prn in (prns if prns is not None else list(by_prn)):
        sat = by_prn.get(int(prn))
        if sat is None:
            continue
        d0 = _doppler(sat, t0)
        rate = (_doppler(sat, t1) - d0) / dt
        g0 = (sat - observer).at(t0)
        elev = g0.altaz()[0].degrees
        rng = float(np.linalg.norm(g0.position.km)) * 1e3  # slant range, m
        out[int(prn)] = (d0, rate, float(elev), rng)
    return out


def predict_skypos(lat, lon, alt_m, prns=None, t_utc=None,
                   tle_source=DEFAULT_TLE_URL, _sats=None):
    """Predict {prn: (az_deg, el_deg)} for GPS sats at a receiver (lat, lon,
    alt_m WGS84) and time t_utc (UTC datetime, default now).

    Azimuth is degrees clockwise from true North (N=0, E=90); elevation is
    degrees above the horizon. Companion to predict_dopplers (which returns the
    line-of-sight Doppler + elevation but drops azimuth); this is what the live
    viewer's sky plot needs. Pass _sats (a {prn: EarthSatellite} from
    load_gps_satellites) to avoid reloading TLEs. Requires skyfield.
    """
    from skyfield.api import load, wgs84
    ts = load.timescale()
    observer = wgs84.latlon(lat, lon, elevation_m=alt_m)
    by_prn = _sats if _sats is not None else load_gps_satellites(tle_source)
    if t_utc is None:
        t_utc = datetime.now(tz=timezone.utc)
    t0 = ts.from_datetime(t_utc)

    out = {}
    for prn in (prns if prns is not None else list(by_prn)):
        sat = by_prn.get(int(prn))
        if sat is None:
            continue
        alt_, az_, _ = (sat - observer).at(t0).altaz()
        out[int(prn)] = (float(az_.degrees), float(alt_.degrees))
    return out


def attach_altaz(records, lat, lon, alt_m, tle_source=DEFAULT_TLE_URL):
    """Return (alt_deg, az_deg) arrays aligned with `records`, NaN where the PRN
    has no TLE. Requires skyfield."""
    from skyfield.api import load, wgs84

    ts = load.timescale()
    observer = wgs84.latlon(lat, lon, elevation_m=alt_m)
    by_prn = load_gps_satellites(tle_source)

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
