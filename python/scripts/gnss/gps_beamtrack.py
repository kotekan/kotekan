#!/usr/bin/env python3
"""Celestrak TLE helpers (satellites by PRN, sky position, geometric Doppler, signal
capability by block)."""

import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

# Celestrak GPS operational TLEs.
DEFAULT_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle"
# Celestrak GLONASS operational TLEs. Names carry the Uragan number, with GLONASS-K marked by a
# trailing K, the capability marker glonass_block_by_prn() reads. These names have no PRN field,
# so load_gps_satellites() takes the slot from the IGS SINEX by catalog number.
GLONASS_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=glo-ops&FORMAT=tle"


# Galileo TLE names ('GSAT0207 (GALILEO 14)') carry the GSAT platform number, not the E-PRN
# (SVID) the spreading codes are keyed by. The mapping is operator metadata, transcribed from
# the ESA GSC constellation-information page (gsc-europa.eu/system-service-status/
# constellation-information). It changes only at launches and decommissionings: a TLE GSAT
# missing here is skipped with a warning (refresh from the GSC page). GSAT0104 (E20) and
# GSAT0205 (E24) are not on that page and stay unmapped.
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
    metadata SINEX (files.igs.org/pub/station/general/igs_satellite_metadata.snx), the
    multi-GNSS SVN/COSPAR/SatCat/PRN registry. TLE names that carry no PRN (GLONASS, BeiDou
    'BEIDOU-3 M23') resolve through it: the TLE carries the catalog number, the SINEX maps it
    to the PRN assignment in force today (started, and open or ending in the future), so a
    pre-announced future assignment does not displace the current one. Re-downloaded
    beside the TLEs when older than 30 days (a stale copy is kept if the download fails); empty
    dict, with one warning, if there is no copy at all."""
    import re, time, urllib.request

    cache = os.path.join(os.path.expanduser("~"), ".cache", "kotekan_gps")
    os.makedirs(cache, exist_ok=True)
    snx = os.path.join(cache, "igs_satellite_metadata.snx")
    if not os.path.exists(snx) or time.time() - os.path.getmtime(snx) > 30 * 86400:
        try:
            # Beside the file, then renamed in: a concurrent reader (the broker's prnmap)
            # never sees a partial copy.
            tmp = snx + ".tmp.%d" % os.getpid()
            urllib.request.urlretrieve(
                "https://files.igs.org/pub/station/general/igs_satellite_metadata.snx",
                tmp,
            )
            os.replace(tmp, snx)
        except Exception as e:
            if not os.path.exists(snx):
                print(
                    "[gps_beamtrack] IGS satellite metadata unavailable (%s)" % e,
                    file=sys.stderr,
                )
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
            current = t_from <= now and (t_to == "0000:000:00000" or t_to >= now)
            if current and re.fullmatch(r"[A-Z]\d{2}", prn):
                svn_prn[svn] = prn
    out = {}
    for svn, cat in svn_cat.items():
        prn = svn_prn.get(svn)
        if prn:
            out[cat] = (prn[0], int(prn[1:]))
    return out


def load_gps_satellites(tle_source, max_age_s=None):
    """Return {prn: EarthSatellite} from a Celestrak TLE file or URL.

    The PRN comes from the name where it carries one: GPS '... (PRN NN)', BeiDou '... (Cnn)',
    Galileo 'GSAT####' via GSAT_TO_PRN. Any other name (GLONASS, unlabelled BeiDou) resolves
    by catalog number through the IGS SINEX. The caller picks the constellation by the TLE
    GROUP in the URL (gps-ops, galileo, beidou, glo-ops).

    A URL is downloaded into ~/.cache/kotekan_gps rather than the working directory, where
    skyfield's default loader would write 'gp.php'. The file is shared across scripts and
    reused for as long as it exists, unless max_age_s is given: a copy older than that is
    re-downloaded, and kept if the download fails. Local paths load in place."""
    import re, time
    from skyfield.api import Loader, load

    if isinstance(tle_source, str) and tle_source.startswith(("http://", "https://")):
        cache = os.path.join(os.path.expanduser("~"), ".cache", "kotekan_gps")
        os.makedirs(cache, exist_ok=True)
        # One cache file per group: every Celestrak URL has the basename "gp.php", and
        # skyfield keys its cache on the basename, so without the GROUP in the file name every
        # group would reuse whichever one was fetched first.
        m = re.search(r"GROUP=([A-Za-z0-9_-]+)", tle_source)
        fname = "tle_%s.txt" % (m.group(1) if m else "default")
        loader = Loader(cache, verbose=False)
        path = os.path.join(cache, fname)
        sats = None
        if (
            max_age_s is not None
            and os.path.exists(path)
            and time.time() - os.path.getmtime(path) > max_age_s
        ):
            try:
                sats = loader.tle_file(tle_source, filename=fname, reload=True)
            except Exception as e:
                print(
                    "[gps_beamtrack] TLE refresh failed (%s); using the cached %s"
                    % (e, fname),
                    file=sys.stderr,
                )
        if sats is None:
            sats = loader.tle_file(tle_source, filename=fname)
    else:
        sats = load.tle_file(tle_source)
    by_prn = {}
    for s in sats:
        name = s.name or ""
        m = re.search(r"PRN\s*(\d+)", name)
        if m:
            by_prn[int(m.group(1))] = s
            continue
        # BeiDou names carry the PRN as '... (C46)'. The name wins over the SINEX, which can
        # hold a stale assignment (C46, C48); the SINEX below serves the names without one.
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
                print(
                    "[gps_beamtrack] unmapped Galileo %s -- refresh GSAT_TO_PRN"
                    % g.group(0),
                    file=sys.stderr,
                )
            continue
        # No PRN in the name (GLONASS, unlabelled BeiDou): NORAD catalog number -> IGS SINEX.
        global _IGS_CAT_MAP
        if _IGS_CAT_MAP is None:
            _IGS_CAT_MAP = _igs_prn_by_catnum()
        hit = _IGS_CAT_MAP.get(getattr(s.model, "satnum", -1))
        if hit:
            by_prn[hit[1]] = s
    return by_prn


_IGS_CAT_MAP = None


def gps_block_by_prn(tle_source=DEFAULT_TLE_URL, _sats=None):
    """Return {prn: block} (e.g. 'IIF', 'IIR', 'IIRM', 'III'; '?' if none) read from the
    Celestrak TLE names ('GPS BIII-10 (PRN 13)'). The block, and so the signal capability, is
    not in any almanac or in the elements, but Celestrak puts it in the name, so it comes from
    the same feed as the visibility rather than from a hard-coded list."""
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
    a trailing K: 'COSMOS 2501 (702K)' is a K, 'COSMOS 2456 (730)' is not. Like
    gps_block_by_prn(), it reads the capability from the feed the visibility comes from.

    This gates L3OC: only the K satellites carry GLONASS's CDMA signals. 'M' covers GLONASS-M
    and anything unrecognised; the distinction that matters is K vs not-K."""
    import re

    sats = _sats if _sats is not None else load_gps_satellites(tle_source)
    blocks = {}
    for slot, s in sats.items():
        m = re.search(r"\((\d+)(K?)\)", s.name or "")
        blocks[slot] = "K" if (m and m.group(2)) else "M"
    return blocks


def _signal_block_filter(signal):
    """Predicate over a GPS block name ('IIR','IIRM','IIF','III',...) selecting the blocks that
    broadcast `signal`. Returns None when every GPS satellite carries the signal, or the signal
    is not recognised (no filter). Block name is as read by gps_block_by_prn(). `signal` is the
    broker's --signal-capability value, a chain name (gps_l5, gps_l2c) or a signal token
    (GPS_L5_Q, GPS_L1C_P), matched case-insensitively by substring. Which blocks carry each
    civil signal: L1 C/A -> all; L2C -> Block IIR-M+; L5 -> Block IIF+; L1C -> Block III."""
    s = (signal or "").upper()
    if "L1CA" in s:  # L1 C/A: every GPS satellite (check before 'L1C')
        return None
    if "L5" in s:  # L5: Block IIF and newer
        return lambda b: b == "IIF" or b.startswith("III")
    if "L2C" in s or "L2_CM" in s:  # L2C: Block IIR-M and newer
        return lambda b: b == "IIRM" or b == "IIF" or b.startswith("III")
    if "L1C" in s:  # L1C: Block III only
        return lambda b: b.startswith("III")
    return None  # unknown -> don't filter


def _glonass_block_filter(signal):
    """The GLONASS counterpart of _signal_block_filter. The CDMA signals (L3OC, and later L2OC /
    L1OC) are GLONASS-K only; the FDMA signals (L1OF/L2OF) are on every satellite."""
    s = (signal or "").upper()
    if "OC" in s:  # L3OC / L2OC / L1OC -- CDMA, K satellites only
        return lambda b: b == "K"
    return None  # L1OF/L2OF (and unknown) -- every satellite


def glonass_fdma_trackable(_cache={}):
    """GLONASS slots trackable on FDMA: those with a known frequency-channel number.

    FDMA capability is not the satellite's generation but whether its carrier is known. Every
    satellite transmits L1OF/L2OF, separated by frequency, so a slot whose k cannot be read
    would be searched at the wrong frequency and has no place in a seed or hint list. "Has a
    k" plays the role the GPS block name and the GLONASS 'K' marker play for CDMA signals.

    Cached for an hour, a failure included. On failure it returns an empty set, with a stderr
    line, and the caller disables the filter rather than darkening the chain."""
    import time

    now = time.monotonic()
    if _cache.get("t") and now - _cache["t"] < 3600.0:
        return _cache["v"]
    try:
        import gnss_ephemeris as ge

        v = set(ge.glonass_freq_channels())
    except Exception as e:
        print(
            "[gps_beamtrack] GLONASS frequency plan unavailable (%s); FDMA capability filter "
            "DISABLED" % e,
            file=sys.stderr,
        )
        v = set()
    _cache["t"], _cache["v"] = now, v
    return v


def signal_capable_prns(signal, tle_source=DEFAULT_TLE_URL, _sats=None):
    """PRNs whose satellite block broadcasts `signal`, from the Celestrak names. Returns every
    PRN in the TLE group when the signal is on every satellite (GPS L1 C/A) or is unknown.
    GLONASS FDMA returns the slots with a known frequency channel (glonass_fdma_trackable),
    or every slot if that lookup fails. A satellite that does not transmit the signal has no
    place in a search list: correlating its absent code wastes channels and makes false
    detections.

    Dispatches on the signal's constellation prefix: GLO_* reads the GLONASS K marker (and
    swaps the GPS default TLE group for glo-ops), everything else reads the GPS block name."""
    sig_u = (signal or "").upper()
    if sig_u.startswith("GLO_") and "OF" in sig_u:
        # FDMA: capability = "we know this satellite's carrier" (see glonass_fdma_trackable).
        cap = glonass_fdma_trackable()
        if cap:
            return cap
        # fall through to the block path (which returns everything) if the plan is unavailable
    if (signal or "").upper().startswith("GLO_"):
        # A GLONASS signal asked for against the GPS default TLE group would find no 'K' marker
        # and return an empty set, which callers treat as "lookup failed, filter disabled".
        # Substitute the GLONASS group instead.
        blocks = glonass_block_by_prn(
            GLONASS_TLE_URL if tle_source == DEFAULT_TLE_URL else tle_source, _sats
        )
        pred = _glonass_block_filter(signal)
    else:
        blocks = gps_block_by_prn(tle_source, _sats)
        pred = _signal_block_filter(signal)
    if pred is None:
        return set(blocks)
    return {prn for prn, b in blocks.items() if pred(b)}


def predict_dopplers(
    lat,
    lon,
    alt_m,
    prns=None,
    t_utc=None,
    f_carrier_hz=1575.42e6,
    tle_source=DEFAULT_TLE_URL,
    _sats=None,
):
    """Predict {prn: (doppler_hz, doppler_rate_hz_per_s, elevation_deg, range_m)} for the TLE's
    satellites at a receiver (lat, lon, alt_m WGS84) and time t_utc (UTC datetime, default
    now). range_m is the slant range, for the propagation delay (~64-89 ms; a TLE's ~km error
    is ~us). Access entries by index so the tuple can grow.

    This is the geometric (line-of-sight range-rate) Doppler only. It does not include the
    receiver clock-frequency error, which adds a common offset across all PRNs (a 1 ppm TCXO
    gives ~1.5 kHz at L1); solve that one bias from the measured Doppler of an acquired
    satellite and it applies to every prediction.

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
        r = g.position.km * 1e3  # m  (sat relative to observer)
        v = g.velocity.km_per_s * 1e3  # m/s
        range_rate = float(np.dot(r, v) / np.linalg.norm(r))  # +ve = receding
        return -f_carrier_hz * range_rate / C  # +ve = approaching

    out = {}
    for prn in prns if prns is not None else list(by_prn):
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


def predict_skypos(
    lat, lon, alt_m, prns=None, t_utc=None, tle_source=DEFAULT_TLE_URL, _sats=None
):
    """Predict {prn: (az_deg, el_deg)} for the TLE's satellites at a receiver (lat, lon,
    alt_m WGS84) and time t_utc (UTC datetime, default now).

    Azimuth is degrees clockwise from true North (N=0, E=90); elevation is
    degrees above the horizon. Companion to predict_dopplers, which drops azimuth;
    the live viewer's TLE sky positions come from here. Pass _sats (a {prn: EarthSatellite}
    from load_gps_satellites) to avoid reloading TLEs. Requires skyfield.
    """
    from skyfield.api import load, wgs84

    ts = load.timescale()
    observer = wgs84.latlon(lat, lon, elevation_m=alt_m)
    by_prn = _sats if _sats is not None else load_gps_satellites(tle_source)
    if t_utc is None:
        t_utc = datetime.now(tz=timezone.utc)
    t0 = ts.from_datetime(t_utc)

    out = {}
    for prn in prns if prns is not None else list(by_prn):
        sat = by_prn.get(int(prn))
        if sat is None:
            continue
        alt_, az_, _ = (sat - observer).at(t0).altaz()
        out[int(prn)] = (float(az_.degrees), float(alt_.degrees))
    return out
