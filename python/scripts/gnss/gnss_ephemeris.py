#!/usr/bin/env python3
"""Broadcast-ephemeris dead-reckoning: fetch + parse the IGS multi-GNSS BRDC RINEX-3 nav
file and predict per-satellite position / range / range-rate / SATELLITE CLOCK -- the two
quantities the TLE/almanac class fundamentally lacks at useful precision (TLE: km orbits,
no usable clock; BRDC: ~1-2 m orbits, ~5 ns clocks, refreshed ~2-hourly).

This is the precision layer under dead-reckoned operations (2026-07-13): with it +
receiver time to <~300 ns, code phase is predictable to <0.3 chips for every visible
satellite -- the search demotes to bootstrap (one-time receiver-clock solve), fallback,
and integrity monitor. The receiver-clock solve itself: measured-vs-predicted code phase
over ALL tracked sats, robust median (each sat is an independent clock measurement;
outliers flag stale ephemerides / maneuvers / false locks).

Supports GPS (G), Galileo (E), BeiDou-3 (C; BDT = GPST - 14 s handled; GEOs excluded
upstream by the BEIDOU-3 name filter). Pure stdlib + math -- no georinex dependency.

Usage:
    from gnss_ephemeris import fetch_brdc, parse_rinex_nav, predict_all
    path = fetch_brdc()                          # cached daily file
    eph = parse_rinex_nav(path)                  # {(sys, prn): [records]}
    pred = predict_all(eph, lat, lon, alt, t_utc)  # {(sys,prn): dict(az, el, range_m,
                                                   #   range_rate_mps, sat_clk_s, ...)}
Self-test: python3 gnss_ephemeris.py            (fetch + predict, prints visible sats)
"""
import json
import gzip
import math
import os
import re
import threading
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta

C_LIGHT = 299792458.0
GPS_UTC_LEAP = 18.0          # GPST - UTC (2026; bump at the next leap second)
BDT_GPST = -14.0             # BDT = GPST - 14 s
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)

# Gravitational parameter / earth rotation per constellation ICD.
MU = {"G": 3.986005e14, "E": 3.986004418e14, "C": 3.986004418e14}
OMEGA_E = {"G": 7.2921151467e-5, "E": 7.2921151467e-5, "C": 7.292115e-5}
F_REL = -4.442807633e-10     # relativistic clock coefficient (s/sqrt(m))

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "kotekan_gps")


def _atomic_write_bytes(path, data):
    """Write `data` to `path` atomically: a sibling .tmp then os.replace (atomic on the same
    filesystem). The cache is SHARED across the per-band obs loggers, so a plain open().write()
    lets another band's reader hit a half-written gzip mid-download -> parse exception ->
    'geometry omitted' -> a dropped az/el row (the L2C/L5 beam-map coverage gap). os.replace
    makes readers see only the old or the new file, never a torn one."""
    # ⚠️ THE TEMP NAME MUST BE UNIQUE PER WRITER, and this used a fixed `path + ".tmp"`.
    # The docstring above already says the cache is SHARED across writers -- five chain
    # threads, and a search instance beside them -- so two of them refreshing the same day's
    # file both opened the SAME temp path, interleaved their bytes into it, and then both
    # renamed. os.replace makes the RENAME atomic; it does nothing about two writers sharing
    # the file being renamed. Usually the two downloads are identical and the corruption is
    # invisible; the case that bites is one of them truncated (a timeout, a short read), which
    # publishes a torn gzip that every reader then fails to parse.
    #
    # Not a free-threading bug -- file writes release the GIL, so this races today. Found by
    # the concurrency audit for the free-threaded move (docs/CHORD_FREE_THREADING.md).
    tmp = "%s.tmp.%d.%d" % (path, os.getpid(), threading.get_ident())
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        # Leaving a stray .tmp.<pid>.<tid> behind would accumulate one file per failed
        # refresh per thread, forever, in a cache directory nobody prunes.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _earthdata_token():
    """Earthdata bearer token for the CDDIS BRDC mirror, from $EARTHDATA_TOKEN or
    <cache>/.earthdata_token (chmod 600, never committed). None -> CDDIS is skipped."""
    t = os.environ.get("EARTHDATA_TOKEN")
    if t and t.strip():
        return t.strip()
    try:
        with open(os.path.join(CACHE, ".earthdata_token")) as f:
            return f.read().strip() or None
    except Exception:
        return None


# CDDIS's merged daily mixed-nav products, by the `kind` slot the caller is filling. These are
# NOT the BKG filename -- see _brdc_sources.
_CDDIS_DAILY = {"R": "BRDC00IGS_R_%04d%03d0000_01D_MN.rnx.gz",    # IGS merged broadcast
                "S": "BRDM00DLR_S_%04d%03d0000_01D_MN.rnx.gz"}    # DLR multi-GNSS


def _brdc_sources(kind, year, doy):
    """Ordered (url, headers) mirrors for one daily mixed-nav file of variant `kind` (R or S):
      1. BKG (igs.bkg.bund.de) -- the historical no-auth source, product BRDC00WRD.
      2. CDDIS (cddis.nasa.gov) with an Earthdata bearer token -- for when BKG is unreachable
         (2026-07-21: igs.bkg.bund.de went down and the node lost all ephemeris).
    CDDIS accepts `Authorization: Bearer <token>` directly on the archive URL (a valid token
    returns the file, an invalid one 401s), so no cookie/redirect handling is needed.

    ⚠️⚠️ THE CDDIS FALLBACK ASKED FOR A FILE CDDIS HAS NEVER HELD, AND SO COULD ONLY 404
    (found 2026-08-27, while BKG was unreachable for the second time). It reused BKG's product
    name -- BRDC00WRD_{R,S}_..._01D_MN.rnx.gz -- against /gnss/data/daily/YYYY/brdc/, a
    directory that holds ONLY the legacy short-name GPS/GLONASS files (brdcDDD0.YYn/.YYg). No
    mixed-nav file has ever lived there under any name. So the mirror added specifically to
    survive a BKG outage returned 404 on its first, second and every subsequent call, in
    complete silence: `_fetch_brdc_merged` treats a failed source as "try the next one", and
    when the next one is the end of the list the whole day is skipped. That is a fallback that
    CANNOT SUCCEED -- the same defect class as a search that cannot return a positive
    ([[chord-broker-refactor]]).
    CDDIS does carry merged dailies, under /gnss/data/daily/YYYY/DDD/YYp/, as BRDC00IGS_R
    (IGS) and BRDM00DLR_S (DLR). Measured on DOY 238: G32/E30/C39 and G32/E30/C36 -- as wide
    as BKG's, or wider.

    ⚠️ BUT CDDIS PUBLISHES ONLY CLOSED DAYS. The YYp directory for the CURRENT day does not
    exist (404), so this mirror can rescue `back=1` and any replay, and CANNOT rescue today.
    When BKG is down, the CURRENT day has exactly one source: the station-hourly merge
    (_fetch_station_hourly). That is why its coverage test is load-bearing, not cosmetic.
    """
    name = "BRDC00WRD_%s_%04d%03d0000_01D_MN.rnx.gz" % (kind, year, doy)
    srcs = [("https://igs.bkg.bund.de/root_ftp/IGS/BRDC/%04d/%03d/%s" % (year, doy, name), {})]
    tok = _earthdata_token()
    cd = _CDDIS_DAILY.get(kind)
    if tok and cd:
        srcs.append(("https://cddis.nasa.gov/archive/gnss/data/daily/%04d/%03d/%02dp/%s"
                     % (year, doy, year % 100, cd % (year, doy)),
                     {"Authorization": "Bearer " + tok}))
    return srcs


# Well-run IGS stations to borrow FRESH ephemeris from (per-station hourly mixed-nav) when the
# merged daily BRDC is stale -- e.g. during a BKG/IGS outage the merged product's own toe's can
# freeze hours back while individual stations keep decoding the live broadcast. Broadcast
# ephemeris is receiver-independent, so any station that tracked a sat carries that sat's current
# ephemeris; a handful of geographically-spread stations covers the visible set.
#
# ⚠️⚠️ GEOGRAPHIC DIVERSITY IS THE WHOLE POINT, AND THE LIST DID NOT HAVE ANY (2026-08-27).
# The first four entries were all Canadian, the merge stopped at the first four stations that
# ANSWERED, and two of those carry no BeiDou at all -- so BRUX, sixth and worth 15 in-slot BDS
# on its own, was never reached. A satellite that is below the horizon at every station in the
# list has no ephemeris ANYWHERE in the merge, and BDS-3's MEO plane spends its thin time over
# eastern North America. Measured at 10 UTC that day, with BKG down and the hourly carrying the
# whole sky:
#     the 4 stations that answered : 15 BDS,  11 in slots 19-42
#     with these stations added    : 37 BDS,  23 in slots 19-42
# The BeiDou chains were left with 2 probe-eligible satellites against the 3 the presence gate
# needs, so bds_b2a ran UNANCHORED -- nothing admitted, nothing trimmed.
#
# Ordered best-coverage-first (BRUX/KIRU are strong on all three constellations) so the
# coverage test below can stop early on a good hour instead of walking the whole list.
_HOURLY_STATIONS = ["BRUX00BEL", "KIRU00SWE", "ALGO00CAN", "AL2H00CAN", "DARW00AUS",
                    "IISC00IND", "MGUE00ARG", "STJO00CAN", "USN700USA", "JFNG00CHN",
                    "CUT000AUS", "NRC100CAN"]

# What "enough coverage" means, per constellation: (min_prn, distinct PRNs >= min_prn) across
# the merged bodies.
# ⚠️ THIS IS THE FIX FOR COUNTING STATIONS. `len(bodies) >= 4` counted sources, and a source
# that carries zero BeiDou is not a source of BeiDou -- NRC1 and STJO both answered with 0 BDS
# and both consumed a slot in the quota. Count the THING WE WANT.
# ⚠️ AND THE THING WE WANT FOR C IS BDS-3, NOT BDS. C01-C18 are BDS-2: they broadcast B1I, not
# B2a/B2b, and every BeiDou chain here runs alm_min_prn = 19 for exactly that reason. A bare
# "21 BeiDou satellites" looks healthy while only 16 of them are satellites this observatory
# can track -- a count that includes what you cannot use is the same mistake as a probe with no
# slot. Measured 2026-08-27 12:21: BRUX alone carries C21/16-BDS-3 and would have satisfied a
# target of 18-counting-BDS-2 on its own, stopping the merge one station in.
# The targets sit below a full constellation (G ~32, E ~28, BDS-3 ~24 reachable) because one
# hour never sees all of it: they mean "this merge is not obviously starved", not "complete".
_HOURLY_TARGET = {"G": (1, 20), "E": (1, 18), "C": (19, 20)}
_HOURLY_MIN_STATIONS = 2  # never trust ONE file, however wide: a truncated or mid-write hourly
                          # is indistinguishable from a thin sky, and a second source is ~5 s
_HOURLY_MAX_FETCH = 12    # bound the cost: ~1.5 MB and a few seconds. It is a budget over the
                          # WHOLE walk now, not per hour -- the walk unions hours until the
                          # coverage target is met, so a per-hour cap would let one thin hour
                          # spend the entire allowance and still come back short.
_HOURLY_TTL_S = 1800.0      # reuse a merge that MET its target for half an hour
_HOURLY_THIN_TTL_S = 300.0  # ...but retry a BELOW-TARGET one in five minutes. The usual cause
                            # is merging early in the hour, before the slow stations have
                            # uploaded the hour that just closed, and they are typically there
                            # a few minutes later -- so the retry is cheap and usually wins.
_HOURLY_COV = "hourly_MN.cov.json"


def _hourly_cov_write(cache_dir, thin, got):
    """Record whether the merge beside it met target. The cache gate cannot re-derive this
    cheaply (it would have to parse and re-count every call), and a gate that cannot tell a
    thin file from a full one is exactly the defect this pair of functions exists to close."""
    try:
        _atomic_write_bytes(os.path.join(cache_dir, _HOURLY_COV),
                            json.dumps({"thin": bool(thin), "got": got,
                                        "t": time.time()}).encode("ascii"))
    except Exception:
        pass


def _hourly_cov_read(cache_dir):
    """(thin, got). ⚠️ UNKNOWN COUNTS AS THIN. A merge written before this bookkeeping existed,
    or a sidecar lost to a partial write, must get the SHORT ttl and be re-merged -- assuming
    a file is complete because we cannot tell is how the thin merge survived 30 minutes in the
    first place."""
    try:
        with open(os.path.join(cache_dir, _HOURLY_COV)) as f:
            d = json.load(f)
        return bool(d.get("thin", True)), str(d.get("got", "?"))
    except Exception:
        return True, "no coverage record"


# How old a broadcast record may be and still COUNT toward the merge's coverage. Deliberately
# tighter than predict_all's 4 h validity: a record at 3h59m satisfies the predictor for one
# more minute and then stops, so counting it as coverage buys a merge that is starved before
# the next refresh. Half the window leaves the sky usable across a full refresh interval.
def _rec_is_fresh(line, when, window_s=None):
    """Is this RINEX 3 nav record's epoch recent enough to COUNT toward coverage?

    ⚠️ THE WINDOW IS THE VALIDITY WINDOW, NOT A TIGHTER "FRESHNESS". It was 2 h for the few
    hours between introducing this test and introducing the rolling store, on the reasoning
    that a nearly-expired record should not satisfy the exit test. With a store that rebuilt
    itself every call that was right. With one that ACCUMULATES it is wrong, and measurably so:
    BDS-3 updates hourly on the hour and IGS hourlies publish ~15-25 min late, so BeiDou's
    freshest record is legitimately up to ~2 h 25 min old -- and a 2 h cut therefore counted
    **C0** on a store holding **23 predictable BeiDou satellites**, declared BELOW TARGET, and
    would have walked every station and every hour once every 300 s to fix a sky that was
    already complete.

    The reason a tight cut is no longer needed is structural, not a relaxation: the walk now
    ALWAYS attempts the two newest hours, so new records arrive every call regardless of what
    the exit test says. The exit test's only remaining job is "do I have enough of the sky
    across the window", and the honest measure of that is the window itself.

    The record's first line carries sys+PRN then the toc as `YYYY MM DD HH MM SS`. Anything
    unparseable counts as FRESH: this test exists to reject records we can prove are stale,
    not to silently shrink coverage whenever a format surprises us -- refusing on doubt here
    would make the merge walk every station and every hour for nothing.
    """
    try:
        f = line[4:23].split()
        if len(f) < 6:
            return True
        toc = datetime(int(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4]),
                       int(float(f[5])), tzinfo=timezone.utc)
        return abs((when - toc).total_seconds()) <= (window_s or _HOURLY_KEEP_S)
    except Exception:
        return True


def _hourly_target_met(seen, fetched):
    return (fetched >= _HOURLY_MIN_STATIONS
            and all(len(seen[k2]) >= v[1] for k2, v in _HOURLY_TARGET.items()))

# This module is a LIBRARY -- imported by the broker, the gates and half a dozen one-shot
# scripts -- so it owns no logger. LOG_HOOK lets a host install its own tagged one (the broker
# sets it to the per-chain `_log` on the first predict); until then, stderr, which
# broker_restart.sh already folds into the broker log (`> "$LOG" 2>&1`). Silent is not an
# option here: a thin merge is invisible downstream and only shows up as a collapsed
# prediction, minutes later, on a different chain.
LOG_HOOK = None


def _log_hourly(msg):
    try:
        if LOG_HOOK is not None:
            LOG_HOOK(msg)
            return
    except Exception:
        pass
    sys.stderr.write("%s\n" % msg)
    sys.stderr.flush()


def _rinex_newest_epoch(path):
    """Newest broadcast epoch (UTC datetime) in a gzipped RINEX-3 nav file, or None. RINEX-3
    nav epoch lines are 'SYS PRN YYYY MM DD HH MM SS ...' (SYS in G/E/C)."""
    import gzip
    newest = None
    try:
        with gzip.open(path, "rt", errors="replace") as f:
            for line in f:
                if len(line) > 23 and line[0] in "GEC" and line[1:3].isdigit() \
                        and line[4:8].strip().isdigit():
                    try:
                        e = datetime(int(line[4:8]), int(line[9:11]), int(line[12:14]),
                                     int(line[15:17]), int(line[18:20]), tzinfo=timezone.utc)
                    except Exception:
                        continue
                    if newest is None or e > newest:
                        newest = e
    except Exception:
        return None
    return newest


def _hourly_sources(station, when, token):
    """Ordered (url, headers) mirrors for ONE station's hourly mixed-nav file.

    ⚠️⚠️ THE CURRENT DAY HANGS ENTIRELY OFF THIS FUNCTION, SO IT MUST NOT BE SINGLE-HOMED.
    Measured 2026-08-27: BKG unreachable (second outage), and CDDIS publishes merged dailies
    only for CLOSED days -- so on any day BKG is down, these per-station hourlies are the ONLY
    source of a current-day ephemeris for E and C. They were fetched from CDDIS alone, behind
    an Earthdata bearer token: one expired credential or one CDDIS outage and every chain goes
    dark on the live sky, with nothing else to fall back to.

    ESA's GSSC carries the same IGS per-station hourly products, anonymously. It is not merely
    a spare: measured the same afternoon, GSSC had hour 11 populated (230 stations) while
    CDDIS's hour-11 directory was still empty, so it is sometimes an hour FRESHER.
    """
    srcs = []
    doy = when.timetuple().tm_yday
    name = "%s_R_%04d%03d%02d00_01H_MN.rnx.gz" % (station, when.year, doy, when.hour)
    if token:
        srcs.append(("https://cddis.nasa.gov/archive/gnss/data/hourly/%04d/%03d/%02d/%s"
                     % (when.year, doy, when.hour, name),
                     {"Authorization": "Bearer " + token}))
    srcs.append(("ftp://gssc.esa.int/gnss/data/hourly/%04d/%03d/%02d/%s"
                 % (when.year, doy, when.hour, name), {}))
    return srcs


# How much history the rolling store keeps. predict_all's validity window is 4 h from toe, so a
# record older than this cannot contribute to any prediction and is pure weight.
# ⚠️ FILTERED ON toc, NOT toe -- the epoch on the record's first line. They coincide for GPS and
# Galileo and sit within minutes for BeiDou, and the margin below absorbs that; parsing toe would
# mean parsing the whole record, which is parse_rinex_nav's job, not the supply layer's.
_HOURLY_KEEP_S = 14400.0


def _split_records(body):
    """Split a RINEX 3 nav body into (sysc, prn, toc_key, text) records.

    A record is a line starting `Xnn ` followed by its continuation lines, which are indented.
    Returned in file order; anything before the first record header is dropped.
    """
    out, cur, key = [], [], None
    for ln in body.splitlines(True):
        if len(ln) > 3 and ln[0] in "GECRJIS" and ln[1:3].isdigit() and ln[3] == " ":
            if cur and key:
                out.append(key + ("".join(cur),))
            f = ln[4:23].split()
            key = (ln[0], int(ln[1:3]), " ".join(f[:6]) if len(f) >= 6 else ln[4:23])
            cur = [ln]
        elif cur:
            cur.append(ln)
    if cur and key:
        out.append(key + ("".join(cur),))
    return out


def _toc_of(key, default=None):
    """datetime of a record key's toc field, or `default` if it will not parse."""
    try:
        f = key[2].split()
        return datetime(int(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4]),
                        int(float(f[5])), tzinfo=timezone.utc)
    except Exception:
        return default


def _store_load(cache_dir, when):
    """The rolling store: {(sys, prn, toc): text}, pruned to _HOURLY_KEEP_S, plus its header.

    ⚠️ UNPARSEABLE toc IS KEPT, not dropped -- the same rule as _rec_is_fresh. This layer
    discards what it can prove is expired; it never discards on doubt.
    """
    import gzip
    path = os.path.join(cache_dir, "hourly_MN.rnx.gz")
    if not os.path.exists(path):
        return {}, None
    try:
        txt = gzip.decompress(open(path, "rb").read()).decode("ascii", "replace")
    except Exception:
        return {}, None
    k = txt.find("END OF HEADER")
    if k < 0:
        return {}, None
    eoh = txt.find("\n", k) + 1
    keep = {}
    for sysc, prn, toc, text in _split_records(txt[eoh:]):
        t = _toc_of((sysc, prn, toc))
        if t is not None and abs((when - t).total_seconds()) > _HOURLY_KEEP_S:
            continue
        keep[(sysc, prn, toc)] = text
    return keep, txt[:eoh]


def _fetch_station_hourly(when, cache_dir, token):
    """MERGE several IGS stations' hourly mixed-nav into one fresh nav file -- a fallback for a
    stale merged daily, and THE only current-day source when the daily mirrors are down. One
    station is unreliable (a given hour it may carry Galileo/BeiDou but no fresh GPS), so union
    a handful for full-constellation coverage: keep one header, concatenate every station's
    record block. UNIONS ACROSS HOURS until coverage is met.
    Returns cache_dir/hourly_MN.rnx.gz or None.

    ⚠️⚠️ COVERAGE DECIDES, NOT RECENCY -- AND THE FIRST VERSION LET RECENCY DECIDE SILENTLY.
    An IGS station publishes the hour that just CLOSED, and the uploads trickle in over the
    following ~40 min. This used to take the first hour that returned ANY station and stop.
    Measured 2026-08-27, two runs 22 min apart on the SAME hour (17 UTC):
        18:17 -- 404 records, C>=19 = 12   (half the stations had not uploaded yet)
        18:39 -- 823 records, C>=19 = 21
    The thin one won because it came first, the 30-minute cache gate then pinned it, and a
    thin merge is indistinguishable downstream from a healthy one -- it just starves a
    constellation somewhere else. It left BeiDou with 2 probe-eligible satellites against the
    3 the presence gate needs, so both b2a and b2b ran UNANCHORED: nothing admitted, nothing
    trimmed, for half an hour, with no line anywhere naming the cause.

    An hour older is ~1 h more toe age against a 4 h validity window -- nothing. Missing half
    the constellation is not nothing. So the walk now UNIONS hours and stops on COVERAGE:
    same statistic the merge already computed, now used as the exit test instead of "somebody
    answered". Same lesson as the four other fallbacks purged today -- a degraded path that
    reproduces the OUTPUT (a file) while dropping the SAFEGUARD (that the file is complete).
    """
    import gzip
    # Cache-gate: hourly station files update ~hourly, and fetch_brdc now includes this in EVERY
    # merge (not just during a daily outage), so reuse a recent local rather than re-pulling a
    # handful of station files each call. 30 min keeps it fresh without hammering CDDIS.
    #
    # ⚠️ BUT A THIN MERGE IS CACHED FOR MINUTES, NOT THE FULL PERIOD. Pinning a starved file
    # for 30 min is what turned "we merged early in the hour" into "BeiDou is dark until the
    # next refresh"; the stations that were missing are usually there a few minutes later, so
    # the retry is nearly free and almost always succeeds. The coverage is recorded next to
    # the file because it cannot be recovered from the file cheaply, and a cache gate that
    # cannot tell a thin file from a full one is the bug restated.
    local = os.path.join(cache_dir, "hourly_MN.rnx.gz")
    if os.path.exists(local):
        age = time.time() - os.path.getmtime(local)
        thin, cov = _hourly_cov_read(cache_dir)
        if age < (_HOURLY_THIN_TTL_S if thin else _HOURLY_TTL_S):
            if thin:
                _log_hourly("BRDC hourly merge: serving a BELOW-TARGET merge from cache "
                            "(%s, %.0f s old, retry in %.0f s). Prediction is thin -- probes, "
                            "drop gates and search hints all ride this."
                            % (cov, age, _HOURLY_THIN_TTL_S - age))
            return local
    # ---- THE ROLLING STORE ---------------------------------------------------------------
    # ⚠️⚠️ ACCUMULATE, DO NOT REBUILD. Every earlier version built the file from ONLY the hours
    # it walked that call, so the file held exactly enough history to satisfy the count at the
    # instant it was written -- and nothing for the hours after. That is fine for a
    # constellation whose records arrive continuously and fatal for one that updates on a
    # schedule. BDS-3 broadcasts hourly ON THE HOUR and IGS hourlies publish ~15-25 min after
    # the hour closes, so BeiDou's freshest available record ages 25 min -> 2 h 25 min and then
    # steps back. Measured 2026-08-27 20:57 UTC, identical at all twelve stations (it is a
    # property of the CONSTELLATION, not of any mirror): G freshest 55 min, E 75, C 115.
    # The merge then held C 28 records at toc age 115/175/295 min against Galileo's 284 -- the
    # median BeiDou record was 73% expired the moment the file was written, so an hour later it
    # fell out of predict_all's 4 h window and predictable C went 23 -> 8. Both BeiDou chains
    # drifted back toward PRESENCE UNANCHORED, on a healthy sky, with every source working.
    #
    # And there is NO daily backstop to cover the gap: today's daily has never been fetched
    # (BKG down, CDDIS publishes only CLOSED days) and yesterday's is ~25 h old, every record
    # far past 4 h. The whole sky rests on this file, so this file has to hold the window.
    #
    # Keeping a rolling union over _HOURLY_KEEP_S makes coverage a property of the WINDOW rather
    # than of one fetch. Records are keyed (sys, prn, toc), so re-fetching an hour is idempotent
    # and two stations carrying the same broadcast record store it once.
    store, header = _store_load(cache_dir, when)
    n_start = len(store)
    fetched = 0
    seen = {"G": set(), "E": set(), "C": set()}
    for (sysc, prn, toc) in store:
        if sysc in seen and prn >= _HOURLY_TARGET[sysc][0]:
            t = _toc_of((sysc, prn, toc))
            if t is None or abs((when - t).total_seconds()) <= _HOURLY_KEEP_S:
                seen[sysc].add(prn)
    done = set()                      # (station, hour) already fetched -- never pay twice
    for dh in (0, 1, 2, 3):
        h = when - timedelta(hours=dh)
        # ⚠️ THE EXIT TEST GOVERNS HOW FAR BACK TO WALK, NOT WHETHER TO FETCH AT ALL. With a
        # store that already satisfies coverage, an unconditional `if met: break` would stop
        # before pulling the newest hour and the store would never refresh -- it would coast to
        # expiry and then collapse, which is the bug this block exists to end, wearing a
        # different hat. So the two newest hours are ALWAYS attempted; older ones are top-up.
        if dh >= 2 and _hourly_target_met(seen, max(fetched, _HOURLY_MIN_STATIONS)):
            break
        for st in _HOURLY_STATIONS:
            if fetched >= _HOURLY_MAX_FETCH:
                break
            if dh >= 2 and _hourly_target_met(seen, max(fetched, _HOURLY_MIN_STATIONS)):
                break
            if (st, h.hour) in done:
                continue
            done.add((st, h.hour))
            # ⚠️ MIRROR ORDER IS PER STATION, NOT PER HOUR. A mirror that has not published
            # this hour yet 404s while the other already has it, and the point is to get the
            # FRESHEST hour available anywhere -- not to pick a host and then walk backwards
            # in time on it alone.
            txt = None
            for url, hdrs in _hourly_sources(st, h, token):
                # A dead host must be paid for ONCE, not once per station: twelve stations x a
                # 20 s timeout is four minutes of a broker cycle.
                if _src_skip(url):
                    continue
                try:
                    req = urllib.request.Request(url, headers=hdrs)
                    with urllib.request.urlopen(req, timeout=20) as r:
                        raw = r.read()
                    if raw[:2] != b"\x1f\x8b":
                        continue          # a 200 that is not gzip: this hour, not this host
                    _src_ok(url)
                    txt = gzip.decompress(raw).decode("ascii", "replace")
                    break
                except Exception as e:
                    _src_failed(url, exc=e)   # 404 = this hour is not published yet, not a
                    continue                  # dead host -- _src_failed knows the difference
            if txt is None:
                continue
            k = txt.find("END OF HEADER")
            if k < 0:
                continue
            eoh = txt.find("\n", k) + 1
            if header is None:
                header = txt[:eoh]
            fetched += 1
            # INTO THE STORE, keyed (sys, prn, toc): re-fetching an hour is idempotent, and two
            # stations carrying the same broadcast record store it once.
            for sysc, prn, toc, text in _split_records(txt[eoh:]):
                t = _toc_of((sysc, prn, toc))
                if t is not None and abs((when - t).total_seconds()) > _HOURLY_KEEP_S:
                    continue
                store[(sysc, prn, toc)] = text
            for ln in txt[eoh:].splitlines():
                if len(ln) > 3 and ln[0] in seen and ln[1:3].isdigit():
                    prn = int(ln[1:3])
                    # ⚠️⚠️ COUNT WHAT IS USABLE, NOT WHAT IS PRESENT. The exit test asks
                    # "have I got enough satellites"; a record whose toe is already past the
                    # predictor's validity window answers yes and predicts nothing. Measured
                    # 2026-08-27 20:18, on a merge that reported `target met` with C21:
                    # only SEVEN of those 21 BeiDou records were predictable, and the probe
                    # band had 2 candidates against the 3 the presence gate needs -- so both
                    # BeiDou chains sat UNANCHORED behind a merge that called itself full.
                    # This is the same defect as counting SOURCES instead of coverage, and
                    # as stopping on recency instead of coverage: an exit test on a PROXY
                    # for the thing wanted rather than on the thing itself.
                    if prn >= _HOURLY_TARGET[ln[0]][0] and _rec_is_fresh(ln, when):
                        seen[ln[0]].add(prn)
    # ---- one file: the whole rolling window, not just this call's fetches -----------------
    if not (header and store):
        return None
    got = "/".join("%s%d" % (k2, len(seen[k2])) for k2 in "GEC")
    thin = [k2 for k2, v in _HOURLY_TARGET.items() if len(seen[k2]) < v[1]]
    hours = sorted({hh for _, hh in done})
    # Oldest first, so best_eph's scan and any human reading the file both see time order.
    order = sorted(store, key=lambda k: (_toc_of(k, datetime(1980, 1, 6, tzinfo=timezone.utc)), k))
    bodies = [store[k] for k in order]
    # ⚠️ A NEGATIVE AGE IS NORMAL, NOT A BUG. GPS/Galileo centre toe in the fit interval, so a
    # freshly broadcast record is routinely stamped up to ~1 h AHEAD of now. Say "ahead" rather
    # than printing a minus sign that reads like a clock fault at 3am.
    span = [t for t in (_toc_of(k) for k in order) if t is not None]
    if span:
        _n = (when - max(span)).total_seconds() / 60.0
        _o = (when - min(span)).total_seconds() / 60.0
        age = "newest %s, oldest %.0f min old" % (
            ("%.0f min ahead" % -_n) if _n < 0 else ("%.0f min old" % _n), _o)
    else:
        age = "?"
    if thin:
        # ⚠️ SAY SO. A merge that never reached its target still returns a file, and a thin
        # file is indistinguishable from a healthy one downstream -- it just makes a
        # constellation's prediction collapse an hour later, somewhere else.
        _log_hourly("BRDC hourly merge: %d station-file(s) over hour(s) %s -> %s; BELOW TARGET "
                    "for %s (want %s). Prediction for %s will be thin -- probes, drop gates "
                    "and search hints all ride this. Retrying in %.0f s."
                    % (fetched, ",".join("%02d" % x for x in hours), got, "+".join(thin),
                       "/".join("%s%d" % (k2, _HOURLY_TARGET[k2][1]) for k2 in "GEC"),
                       "+".join(thin), _HOURLY_THIN_TTL_S))
    else:
        _log_hourly("BRDC hourly merge: %d station-file(s) over hour(s) %s -> %s, target met. "
                    "Store %d -> %d record(s); %s."
                    % (fetched, ",".join("%02d" % x for x in hours), got,
                       n_start, len(store), age))
    _atomic_write_bytes(local, gzip.compress(
        (header + "".join(bodies)).encode("ascii", "replace")))
    _hourly_cov_write(cache_dir, bool(thin), got)
    return local


def _try_refresh_daily(kind, year, doy, cache_dir, tok):
    """Best-effort fetch of ONE daily-BRDC variant for the per-PRN merge -- S (CDDIS super-sled,
    widest coverage) or R (BKG). Reuses a <2 h cache, else re-downloads via the same mirror list
    as the primary (_brdc_sources routes each variant to whoever serves it). Returns its cached
    path or None; never raises. This is what guarantees the FRESH variant reaches the merge even
    when the primary fetch returned the frozen one -- the BKG-froze-at-08:00 / CDDIS-fresh case."""
    name = "BRDC00WRD_%s_%04d%03d0000_01D_MN.rnx.gz" % (kind, year, doy)
    local = os.path.join(cache_dir, name)
    if os.path.exists(local) and time.time() - os.path.getmtime(local) < 7200:
        return local
    tried = 0
    for url, headers in _brdc_sources(kind, year, doy):
        if _src_skip(url):
            continue                      # known dead, still cooling down -- do not pay for it
        tried += 1
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            if data[:2] != b"\x1f\x8b":
                _src_failed(url)          # login/error/404 page: this mirror is not serving
                continue
            _src_ok(url)
            _atomic_write_bytes(local, data)
            return local
        except Exception as e:
            _src_failed(url, exc=e)
            continue
    # EVERY source skipped means every source is cooling down. Clear the marks and let the next
    # call try again rather than starving forever on our own bookkeeping.
    if tried == 0:
        _src_dead.clear()
    return local if os.path.exists(local) else None



# ── NEGATIVE CACHE FOR DEAD SOURCES (2026-08-27) ───────────────────────────────────────────
# ⚠️ A DEAD MIRROR MUST BE PAID FOR ONCE, NOT ONCE PER CHAIN PER REFRESH. Every fetch here is
# SYNCHRONOUS inside the broker's cycle, and the sources are tried in order, so while
# igs.bkg.bund.de is unreachable each of the five chain threads independently opens a socket to
# it and blocks for the full 30 s connect timeout -- before falling through to CDDIS, which
# works. Observed 2026-08-27: five sockets in SYN-SENT to 141.74.52.58:443 on every cycle, the
# broker crawling, and startup taking minutes instead of seconds. The data was never at risk
# (the fallback delivers); the LATENCY was, and it is the kind that reads as "the broker hung".
#
# So: remember which source URL prefix just failed and skip it for _SRC_COOLDOWN_S. The cost of
# being wrong is bounded and self-correcting -- a source that recovers is retried after the
# cooldown, and the merge is per-PRN freshest-wins, so skipping a live-but-slow mirror for a
# few minutes costs nothing a later merge cannot repair.
_SRC_COOLDOWN_S = 300.0
_src_dead = {}          # url prefix -> time.time() when it may be retried


def _src_key(url):
    """Scheme+host, so one dead host is skipped for every path under it."""
    try:
        pr = urllib.parse.urlsplit(url)
        return pr.scheme + "://" + pr.netloc
    except Exception:
        return url


def _src_skip(url, now=None):
    return (_src_dead.get(_src_key(url), 0.0) > (now or time.time()))


# Statuses that are a statement about the PATH, not about the host. Everything else -- a
# timeout, a refused connection, a 5xx, a 200 that is not gzip -- says the mirror is not
# serving, and blacklisting it is right.
_PATH_STATUS = (404, 410)
# The FTP dialect of the same thing. 550 is "no such file or directory", 450 is "temporarily
# unavailable" -- both about the path. urllib wraps them in a URLError with no .code, so the
# status only survives in the message ("ftp error: 550 CWD command failed: ...").
_FTP_PATH_RE = re.compile(r"ftp error:\s*(\d{3})")
_FTP_PATH_STATUS = ("550", "450", "553")


def _is_path_failure(exc):
    """Is this exception about the PATH (absent file) rather than the HOST (not serving)?"""
    if exc is None:
        return False
    if getattr(exc, "code", None) in _PATH_STATUS:
        return True
    m = _FTP_PATH_RE.search(str(exc))
    return bool(m and m.group(1) in _FTP_PATH_STATUS)


def _src_failed(url, now=None, exc=None):
    """Blacklist a source host for the cooldown -- unless the failure was about the PATH.

    ⚠️ A 404 IS NOT A DEAD HOST, AND TREATING IT AS ONE POISONS THE MIRROR THAT WORKS
    (2026-08-27). The key is scheme+host, deliberately, so one dead server is skipped for
    every path under it. But the daily fetch asks CDDIS for the CURRENT day, and CDDIS only
    publishes CLOSED days -- so that request 404s every time, by design, and it was marking
    the whole of cddis.nasa.gov dead for 300 s. The very next call is the one that fetches
    YESTERDAY's daily from CDDIS, which does exist and is exactly the fallback that has to
    work while BKG is unreachable. A guaranteed-absent path was disabling a live server.

    ⚠️ AND IT HAS AN FTP DIALECT, found the same afternoon by actually exercising the new
    GSSC mirror: the hour-walk starts at the CURRENT hour, whose directory does not exist
    yet, and GSSC answers `URLError('ftp error: 550 CWD command failed')` -- no `.code` at
    all, so the HTTP-only test passed it straight through to the blacklist. ONE not-yet-
    published hour disabled the entire mirror for 300 s, on its first call: the whole
    12-station x 4-hour walk collapsed to a single attempt. Fixing only the HTTP half would
    have shipped a second mirror that never fetched anything.
    """
    if _is_path_failure(exc):
        return
    _src_dead[_src_key(url)] = (now or time.time()) + _SRC_COOLDOWN_S


def _src_ok(url):
    _src_dead.pop(_src_key(url), None)


def fetch_brdc(when=None, cache_dir=CACHE):
    """Ordered list of currently-available BRDC nav-source files (best-first). parse_rinex_nav
    merges them PER-PRN (freshest toe per PRN wins, union of records), so no single frozen
    product can starve a PRN another source carries fresh.

    ⚠️ GNSS_BRDC_DIR PINS THE SKY (task #29). When set, return exactly the nav files in that
    directory -- no cache, no network, no date logic. This is what makes the broker
    equivalence gate hermetic: a transcript replay is a claim about CODE, but the ephemeris
    is an input too, refreshed several times a day, and it moved the on-sky digests TWICE in
    two days with byte-identical code (docs/gnss_gpu_search.md 11.18.2b -- the second time
    was midnight UTC rolling the day-of-year during a review of unrelated work). Replays run
    against a committed-by-hash snapshot; live brokers never set this and keep the fresh sky.

    Historically this returned one path chosen by a fragile single-file fallback: prefer the
    merged daily, and only if IT was globally stale (>2.5 h) borrow the station-hourly instead.
    That failed on a PARTIAL freeze -- 2026-07-22 BKG's daily froze at 08:00 UTC, so the code
    dropped to the station-hourly, which carries only a thin ~18-PRN BeiDou set and has no C39 at
    all, while the CDDIS daily (fresh C39 to 16:00) sat unused in cache -> C31/C39 showed 'el --'
    despite strong tracking. Now every source contributes and the merge picks each PRN's freshest
    record. Callers all do parse_rinex_nav(fetch_brdc()), which accepts a list transparently."""
    pin = os.environ.get("GNSS_BRDC_DIR")
    if pin:
        pinned = sorted(os.path.join(pin, f) for f in os.listdir(pin)
                        if f.endswith((".rnx", ".rnx.gz")))
        if not pinned:
            raise RuntimeError("GNSS_BRDC_DIR=%s contains no .rnx/.rnx.gz nav files -- a "
                               "pinned replay must not fall through to the live sky" % pin)
        return pinned
    when = when or datetime.now(timezone.utc)
    primary = _fetch_brdc_merged(when, cache_dir)   # authoritative daily (keeps prev-day safety)
    srcs = [primary]
    tok = _earthdata_token()
    doy = when.timetuple().tm_yday
    # Both daily variants (S first = widest), then the station-hourly -- each best-effort and
    # cache-gated, so a failure just leaves `primary` and never blocks. A stale/frozen variant is
    # harmless: its older toe's simply lose the per-PRN freshest-toe selection.
    for kind in ("S", "R"):
        p = _try_refresh_daily(kind, when.year, doy, cache_dir, tok)
        if p and p not in srcs:
            srcs.append(p)
    hourly = _fetch_station_hourly(when, cache_dir, tok)
    if hourly and hourly not in srcs:
        srcs.append(hourly)
    return srcs


def _fetch_brdc_merged(when=None, cache_dir=CACHE):
    """The merged daily BRDC file for `when` (BKG first, CDDIS-with-token fallback)."""
    os.makedirs(cache_dir, exist_ok=True)
    when = when or datetime.now(timezone.utc)
    # Process each DAY fully -- fresh cache, then (re-)download, then a STALE same-day cache --
    # BEFORE falling back to the previous day. This ordering is load-bearing: a current-day
    # file predicts even hours stale (its toe's still bracket "now"), but YESTERDAY's file
    # predicts NOTHING today (all toe's > best_eph's 4 h window -> predict_all returns 0 ->
    # the broker seeds/hints nothing -> the require_hint search goes dark on every band).
    # The old code returned a cached yesterday file (back==1) UNCONDITIONALLY the moment a
    # current-day re-fetch failed, handing the broker a dead-on-arrival ephemeris during a
    # BRDC-server outage (2026-07-21: whole node stopped acquiring while today's valid file
    # sat one line up in the cache). Fix: only fall to the previous day when THIS day yields
    # nothing at all -- not merely because its re-download failed.
    for back in (0, 1):
        d = when - timedelta(days=back)
        doy = d.timetuple().tm_yday
        # (kind, local path) PAIRS, not bare paths: the mirrors serve DIFFERENT product names
        # for the same variant (BKG BRDC00WRD vs CDDIS BRDC00IGS/BRDM00DLR), so the download
        # loop below needs to know which variant slot it is filling. A bare path list left
        # `kind` leaking out of the loop that built it -- always "R" by the time it was read.
        locals_ = [(kind, os.path.join(
            cache_dir, "BRDC00WRD_%s_%04d%03d0000_01D_MN.rnx.gz" % (kind, d.year, doy)))
            for kind in ("S", "R")]
        # 1) a FRESH current-day cache (< 2 h) or ANY cached previous-day file (final/immutable)
        #    is usable as-is, no network needed.
        for _kind, local in locals_:
            if os.path.exists(local) and (back == 1
                                          or time.time() - os.path.getmtime(local) < 7200):
                return local
        # 2) (re-)download this day (current-day files GROW through the day, so refresh a stale
        #    cache to pull the newest toe's). Try each mirror (BKG, then CDDIS if a token exists);
        #    a valid BRDC is gzip, so reject anything without the gzip magic (an auth/error page
        #    served as 200 must never be cached as if it were ephemeris).
        # ⚠️ THE SAME NEGATIVE CACHE AS _fetch_brdc_variant, AND IT MUST BE. This is a second
        # copy of that mirror loop, and patching only the other one left BKG being dialled on
        # every cycle anyway -- five sockets still in SYN-SENT, from HERE (2026-08-27). Two
        # copies of a retry policy is two places to fix and one to forget.
        for kind, local in locals_:
            for url, headers in _brdc_sources(kind, d.year, doy):
                if _src_skip(url):
                    continue
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=30) as r:
                        data = r.read()
                    if data[:2] != b"\x1f\x8b":
                        _src_failed(url)
                        continue  # not gzip (login/error page) -> try the next mirror
                    _src_ok(url)
                    _atomic_write_bytes(local, data)
                    return local
                except Exception as e:
                    _src_failed(url, exc=e)
                    continue
        # 3) download failed but a STALE same-day cache exists -> use it (still predicts) rather
        #    than falling to the previous day's dead-for-today ephemeris.
        for _kind, local in locals_:
            if os.path.exists(local):
                return local
        # nothing for this day at all -> fall back to the previous day (early-UTC-morning case).
    raise RuntimeError("no BRDC file reachable (network?) and no cache")


def _f(s):
    """RINEX float field ('D' exponents, blanks)."""
    s = s.strip().replace("D", "E").replace("d", "E")
    return float(s) if s else 0.0


def parse_rinex_nav(paths):
    """RINEX 3 mixed nav -> {(sys, prn): [eph dicts sorted by toe_gpst]}. G/E/C only.

    `paths` is a single path OR a list of paths (fetch_brdc returns a list). A list is merged
    PER-PRN: the union of every source's records, deduped by toe, so the freshest valid
    ephemeris for each PRN is available no matter which source carries it. This is what lets a
    frozen daily product (BKG stuck at 08:00 UTC) be transparently backfilled by a sibling
    source (the CDDIS daily, or the station-hourly merge) that still has the sat -- the root of
    the C31/C39 'el --' gap: C39 was absent from the sparse hourly the old single-file fallback
    picked, but present fresh in the CDDIS daily sitting unused in cache. sources are passed
    best-first, so a daily's record wins an identical-toe tie over the same record elsewhere."""
    if isinstance(paths, (list, tuple)):
        merged = {}
        for p in paths:
            try:
                one = _parse_rinex_nav_file(p)
            except Exception:
                continue
            for key, recs in one.items():
                b = merged.setdefault(key, {})
                for e in recs:
                    b.setdefault(round(e["toe_gpst"], 3), e)  # first (best) source wins same toe
        return {k: sorted(v.values(), key=lambda e: e["toe_gpst"]) for k, v in merged.items()}
    return _parse_rinex_nav_file(paths)


def glonass_freq_channels(paths=None):
    """{slot: frequency-channel number k} for GLONASS, read from the SAME BRDC files everything
    else uses. `paths` defaults to fetch_brdc().

    ★ WHY THIS EXISTS AS ITS OWN PARSER. GLONASS is FDMA: every satellite transmits the SAME
    511-chip code and is separated by CARRIER, satellite k sitting at 1246.0 + k*0.4375 MHz on
    L2. So a GLONASS receiver needs k per satellite before it can even search -- and k is not in
    the TLE, not in the satellite name, and NOT a constant (it is reassigned as satellites are
    replaced, so a hardcoded table silently goes stale and points the search at the wrong
    carrier).

    ★ AND WHY IT IS NOT parse_rinex_nav(). That parser filters `sysc in "GEC"` and skips GLONASS
    entirely, correctly: GLONASS broadcasts a position/velocity/acceleration state vector to be
    integrated by Runge-Kutta, not a Keplerian element set, so none of the orbit code applies.
    But k is just an INTEGER sitting in those same records -- no orbit maths needed to read it.
    This function takes only that one field and ignores everything else, so we get the frequency
    plan from an authoritative live source without owning a GLONASS propagator.

    RINEX 3 GLONASS record = 4 lines: epoch/clock, then X/X'/X''/health, Y/Y'/Y''/**freq num**,
    Z/Z'/Z''/age. k is the 4th field of the SECOND orbit line. Returns {} if nothing parses.
    ⚠️ k is NOT unique across the constellation: antipodal satellites share a channel (they are
    never simultaneously visible), so expect ~14 distinct values over ~24 satellites.
    """
    if paths is None:
        paths = fetch_brdc()
    if isinstance(paths, str):
        paths = [paths]
    out = {}
    for path in paths:
        try:
            op = gzip.open if str(path).endswith(".gz") else open
            with op(path, "rt", errors="replace") as f:
                lines = f.readlines()
        except Exception:
            continue
        i = 0
        while i < len(lines) and "END OF HEADER" not in lines[i]:
            i += 1
        i += 1
        while i + 3 < len(lines):
            line = lines[i]
            if not line.startswith("R"):
                i += 1
                continue
            try:
                slot = int(line[1:3])
                k = int(round(_f(lines[i + 2][61:80])))
            except Exception:
                i += 1
                continue
            i += 4
            if -7 <= k <= 6 and 1 <= slot <= 32:
                out.setdefault(slot, k)  # first (freshest source) wins
    return out


def _parse_rinex_nav_file(path):
    """Parse ONE RINEX 3 mixed-nav file -> {(sys, prn): [eph dicts sorted by toe_gpst]}."""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines) and "END OF HEADER" not in lines[i]:
        i += 1
    i += 1
    out = {}
    # broadcast orbit line counts per system (RINEX 3): G/E/C all use 7 orbit lines.
    while i < len(lines):
        line = lines[i]
        sysc = line[0] if line else ""
        if sysc not in "GEC":
            i += 1
            continue
        try:
            prn = int(line[1:3])
            yy, mo, dd, hh, mi, ss = (int(line[4:8]), int(line[9:11]), int(line[12:14]),
                                      int(line[15:17]), int(line[18:20]), int(line[21:23]))
            af = [_f(line[23:42]), _f(line[42:61]), _f(line[61:80])]
            orb = []
            for k in range(1, 8):
                l2 = lines[i + k]
                orb += [_f(l2[4:23]), _f(l2[23:42]), _f(l2[42:61]), _f(l2[61:80])]
        except Exception:
            i += 1
            continue
        i += 8
        toc = datetime(yy, mo, dd, hh, mi, ss, tzinfo=timezone.utc)  # in the SYSTEM time
        toc_gpst = (toc - GPS_EPOCH).total_seconds()
        if sysc == "C":
            toc_gpst -= BDT_GPST  # BDT epoch -> GPST seconds-since-GPS-epoch
        e = dict(sys=sysc, prn=prn, af0=af[0], af1=af[1], af2=af[2],
                 crs=orb[1], dn=orb[2], m0=orb[3],
                 cuc=orb[4], ecc=orb[5], cus=orb[6], sqrta=orb[7],
                 toe_sow=orb[8], cic=orb[9], omega0=orb[10], cis=orb[11],
                 i0=orb[12], crc=orb[13], omega=orb[14], omegadot=orb[15],
                 # health = BROADCAST ORBIT 6 field 2 (orb[21]): SV health (G/E) / SatH1 (C).
                 # It was orb[25] -- field 2 of orbit line 7 -- which is a DIFFERENT quantity in
                 # every system, and the mistake was invisible in two of the three:
                 #   G: Fit interval  -> reads 4.0 on every healthy sat -> the strict G/E gate in
                 #      predict_all rejected THE ENTIRE GPS CONSTELLATION (measured 2026-07-16:
                 #      predict_all returned 0 GPS of 32 parsed, so every GPS observable row was
                 #      logged with NO el/az/range -> no CMC, no IPP, and TEC from L1+L2C (both
                 #      GPS!) was impossible). This is the real cause of the "el 1% populated"
                 #      symptom that the BRDC staleness fix (6976642e) only partly addressed.
                 #   E: blank -> 0.0 -> Galileo passed the gate BY ACCIDENT.
                 #   C: AODC -> ~1 -> BeiDou was let through by the SatH1 exemption below, which
                 #      was written for orb[21] all along but never actually read it.
                 idot=orb[16], week=orb[18], health=orb[21], toc_gpst=toc_gpst,
                 # Fields the ORBIT model does not need but the LNAV BIT ENCODER does
                 # (gps_lnav_encode.py): subframe 1 carries URA/IODC/TGD/L2 flags and
                 # subframes 2/3 carry IODE, none of which were kept before 2026-07-25.
                 # RINEX has them all; dropping them silently is what made "encode the
                 # ephemeris back into subframe bits" look impossible from BRDC alone.
                 # ORBIT 1 = [IODE, Crs, dn, M0] (0-3); ORBIT 5 = [IDOT, L2 codes, week,
                 # L2P flag] (16-19); ORBIT 6 = [accuracy, health, TGD, IODC] (20-23);
                 # ORBIT 7 = [transmission time, fit interval] (24-25). health=orb[21]
                 # above is the anchor that fixes this indexing.
                 iode=orb[0], l2_codes=orb[17], l2p_flag=orb[19], accuracy=orb[20],
                 tgd=orb[22], iodc=orb[23], fit=orb[25] if len(orb) > 25 else 0.0)
        # toe as absolute GPST: the RINEX week field is the system's own week count --
        # GPS/GAL: GPS-aligned continuous week; BDS: BDT week (epoch 2006-01-01) + 14 s.
        if sysc in "GE":
            e["toe_gpst"] = e["week"] * 604800.0 + e["toe_sow"]
        else:
            e["toe_gpst"] = (e["week"] + 1356.0) * 604800.0 + e["toe_sow"] - BDT_GPST
        if e["sqrta"] > 1000.0:  # sanity: reject empty/garbled records
            out.setdefault((sysc, prn), []).append(e)
        # (GLONASS and SBAS records have different layouts; skipped by the sysc filter.)
    for k in out:
        out[k].sort(key=lambda e: e["toe_gpst"])
    return out


def sat_pos_clk(e, t_gpst):
    """ICD Kepler propagation -> ECEF position (m), velocity (m/s), clock offset (s).
    t_gpst = seconds since the GPS epoch (continuous, no week rollover)."""
    mu = MU[e["sys"]]
    om_e = OMEGA_E[e["sys"]]
    a = e["sqrta"] ** 2
    n = math.sqrt(mu / a ** 3) + e["dn"]
    tk = t_gpst - e["toe_gpst"]
    m = e["m0"] + n * tk
    ek = m
    for _ in range(12):
        ek = m + e["ecc"] * math.sin(ek)
    sv = math.sqrt(1 - e["ecc"] ** 2)
    vk = math.atan2(sv * math.sin(ek), math.cos(ek) - e["ecc"])
    phi = vk + e["omega"]
    s2p, c2p = math.sin(2 * phi), math.cos(2 * phi)
    du = e["cus"] * s2p + e["cuc"] * c2p
    dr = e["crs"] * s2p + e["crc"] * c2p
    di = e["cis"] * s2p + e["cic"] * c2p
    u = phi + du
    r = a * (1 - e["ecc"] * math.cos(ek)) + dr
    ik = e["i0"] + di + e["idot"] * tk
    om = e["omega0"] + (e["omegadot"] - om_e) * tk - om_e * e["toe_sow"]
    x, y = r * math.cos(u), r * math.sin(u)
    so, co = math.sin(om), math.cos(om)
    si, ci = math.sin(ik), math.cos(ik)
    X = x * co - y * ci * so
    Y = x * so + y * ci * co
    Z = y * si
    # VELOCITY BY CENTRAL DIFFERENCE. This was a FORWARD difference over 0.5 s, commented
    # "adequate: range-rate to ~mm/s". It is not, and the error is not noise -- a forward
    # difference does not estimate v(t), it estimates v(t + h/2). That is a 0.25 s TIME-TAG
    # OFFSET, and a time offset in Doppler is dop_rate * 0.25.
    #
    # Along the line of sight a 0.37 Hz/s Doppler rate is ~0.094 m/s^2 of range acceleration,
    # so the bias is 0.094 * 0.25 = 0.024 m/s -> 0.09 Hz at 1176.45 MHz. The cross-window phase
    # budget (task #52) needs the carrier to ~15 mHz for 0.1 rad over one 1.049 s window, so
    # this ONE first-order scheme was 6x the entire budget.
    #
    # Central differencing costs one extra position evaluation and is second-order: the leading
    # error becomes (h^2/6)*jerk, and for a GPS-altitude orbit (omega^3 r ~ 8e-5 m/s^3) that is
    # ~3e-6 m/s -- 13 microHz, four orders below the budget. Do NOT shrink h to compensate for
    # the old scheme: the bias is first-order in h, so halving h only halves it, while the
    # central form removes it outright.
    dt = 0.5
    e2 = dict(e)
    pp = _pos_only(e2, t_gpst + dt)
    pm = _pos_only(e2, t_gpst - dt)
    V = tuple((pp[k] - pm[k]) / (2.0 * dt) for k in range(3))
    # clock: polynomial + relativistic
    tc = t_gpst - e["toc_gpst"]
    clk = e["af0"] + e["af1"] * tc + e["af2"] * tc * tc \
        + F_REL * e["ecc"] * e["sqrta"] * math.sin(ek)
    return (X, Y, Z), V, clk


# -- BROADCAST GROUP DELAY (task A0b) -----------------------------------------------------
# Carrier frequencies (Hz) and the gamma = (f_ref/f_signal)^2 scalings the ICDs use.
F_E1 = F_L1 = 1575.42e6
F_E5A = F_L5 = 1176.45e6
F_E5B = 1207.14e6
GAMMA_L1L5 = (F_L1 / F_L5) ** 2          # 1.79329
GAMMA_E1E5A = (F_E1 / F_E5A) ** 2        # 1.79329
GAMMA_E1E5B = (F_E1 / F_E5B) ** 2        # 1.70325

# RINEX 3 GAL dataSources bits (BROADCAST ORBIT 5 field 2, parsed into e["l2_codes"]):
# bit 8 = clock referenced to (E5a, E1)  [F/NAV];  bit 9 = referenced to (E5b, E1)  [I/NAV].
_GAL_CLK_E5A = 0x100
_GAL_CLK_E5B = 0x200


def group_delay_s(e, signal, dcb=None):
    """Seconds to ADD to the broadcast clock so it refers to `signal`'s own code phase.

    THE BROADCAST CLOCK IS NOT REFERENCED TO OUR SIGNAL. Every constellation fits its clock
    to some other combination -- GPS LNAV to L1/L2, Galileo F/NAV to E1/E5a and I/NAV to
    E1/E5b, BeiDou legacy to B3I -- and a single-frequency user of anything else owes the
    corresponding differential. Until 2026-08-23 nothing here applied it: the decoders
    parsed TGD/BGD and the correction path ignored them (buglist A0b).

    SIZE, MEASURED on the 2026-08-23 BRDC before writing this: the CONSTELLATION-COMMON
    part is small -- GPS TGD median -8.4 ns -> +0.15 chips at L5, Galileo BGD -0.9 ns ->
    +0.02 chips -- and a common term is degenerate with the receiver clock anyway, so a
    chain that solves its own clock never sees it. What this actually buys is the PER-SAT
    spread: GPS TGD ranges -17.7..+6.5 ns and BDS TGD +-45 ns, i.e. +-0.3..0.5 chips
    landing straight in b_sat. Do NOT expect it to move a constellation offset.

    OUR GALILEO BRDC IS MIXED, and best_eph picks by freshness: 19 sats carried F/NAV
    (clock ref E5a,E1) and 11 carried I/NAV (clock ref E5b,E1) on 2026-08-23. So the
    reference a satellite gets is arbitrary AND CAN FLIP at a refresh. Both cross-type
    conversions are implemented from the ICD identity t_E1 = t_IF_a - BGD_a = t_IF_b -
    BGD_b; without them a record-type flip would step that satellite's clock by a few ns.

    Returns 0.0 when the term is unavailable rather than guessing -- notably BeiDou B2a,
    whose TGD_B2ap lives in B-CNAV2 and is absent from RINEX 3 (only TGD1 B1I/B3I and
    TGD2 B2I/B3I are broadcast there).
    """
    if not signal:
        return 0.0
    sig = str(signal).lower()
    sysc = e.get("sys")
    # MEASURED BIAS BEATS BROADCAST, when we have it (2026-08-23). The MGEX DCB product
    # covers every signal we track -- including BeiDou B2a, whose broadcast term is not in
    # RINEX 3 at all -- and is a global-network estimate rather than the satellite's own.
    # ⚠️ It is ZERO-MEAN PER CONSTELLATION by construction, so it supplies the PER-SAT
    # spread and never a constellation-common level; the broadcast term is the only source
    # of the latter, and for B2a there is none. Falls through to broadcast when a pair is
    # missing, so a partial product degrades per-satellite rather than all-or-nothing.
    if dcb:
        try:
            import gnss_dcb as _dcbmod
            _inav = bool(int(e.get("l2_codes") or 0) & _GAL_CLK_E5B) if sysc == "E" else False
            _v = _dcbmod.signal_bias_s(dcb, sysc, int(e.get("prn") or 0), sig,
                                       gal_inav=_inav)
            if _v is not None and math.isfinite(_v):
                return _v
        except Exception:
            pass
    tgd = float(e.get("tgd") or 0.0)        # G: TGD | E: BGD(E1,E5a) | C: TGD1(B1I,B3I)
    tgd2 = float(e.get("iodc") or 0.0)      # E: BGD(E1,E5b) | C: TGD2(B2I,B3I) | G: IODC(!)
    if sysc == "G":
        # IS-GPS-705: an L5 user owes gamma_15 * TGD (plus ISC_L5I5, which is CNAV-only and
        # not in RINEX 3 -- so this is the larger half of the term, not all of it).
        if "l5" in sig:
            return -GAMMA_L1L5 * tgd
        if "l1" in sig:
            return -tgd
        return 0.0
    if sysc == "E":
        ds = int(e.get("l2_codes") or 0)
        ref_a = bool(ds & _GAL_CLK_E5A)     # F/NAV: clock is the E1/E5a iono-free combination
        ref_b = bool(ds & _GAL_CLK_E5B)     # I/NAV: clock is the E1/E5b iono-free combination
        if not (ref_a or ref_b):
            ref_a = True                    # unflagged records: treat as F/NAV, the ICD default
        if "e5a" in sig:
            if ref_a:
                return -GAMMA_E1E5A * tgd
            # I/NAV record used for E5a: t_IF_a = t_IF_b + BGD_a - BGD_b, then apply E5a's own.
            return -(GAMMA_E1E5A - 1.0) * tgd - tgd2
        if "e5b" in sig:
            if ref_b:
                return -GAMMA_E1E5B * tgd2
            # F/NAV record used for E5b, the mirror of the above.
            return -tgd - (GAMMA_E1E5B - 1.0) * tgd2
        if "e1" in sig:
            return -(tgd if ref_a else tgd2)
        return 0.0
    if sysc == "C":
        # BDS legacy D1/D2 clock is B3I-referenced. B2b shares B2I's carrier (1207.14 MHz),
        # so TGD2 is the right band term for it; B2a (1176.45) needs TGD_B2ap, which this
        # message does not carry -- return 0 and stay honest rather than borrow TGD2.
        if "b2b" in sig:
            return -tgd2
        if "b1" in sig:
            return -tgd
        return 0.0
    return 0.0


def _pos_only(e, t_gpst):
    mu = MU[e["sys"]]
    om_e = OMEGA_E[e["sys"]]
    a = e["sqrta"] ** 2
    n = math.sqrt(mu / a ** 3) + e["dn"]
    tk = t_gpst - e["toe_gpst"]
    m = e["m0"] + n * tk
    ek = m
    for _ in range(12):
        ek = m + e["ecc"] * math.sin(ek)
    sv = math.sqrt(1 - e["ecc"] ** 2)
    vk = math.atan2(sv * math.sin(ek), math.cos(ek) - e["ecc"])
    phi = vk + e["omega"]
    s2p, c2p = math.sin(2 * phi), math.cos(2 * phi)
    u = phi + e["cus"] * s2p + e["cuc"] * c2p
    r = a * (1 - e["ecc"] * math.cos(ek)) + e["crs"] * s2p + e["crc"] * c2p
    ik = e["i0"] + e["cis"] * s2p + e["cic"] * c2p + e["idot"] * tk
    om = e["omega0"] + (e["omegadot"] - om_e) * tk - om_e * e["toe_sow"]
    x, y = r * math.cos(u), r * math.sin(u)
    so, co = math.sin(om), math.cos(om)
    si, ci = math.sin(ik), math.cos(ik)
    return (x * co - y * ci * so, x * so + y * ci * co, y * si)


def _ecef_of_llh(lat, lon, alt):
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    sl, cl = math.sin(math.radians(lat)), math.cos(math.radians(lat))
    so, co = math.sin(math.radians(lon)), math.cos(math.radians(lon))
    N = a / math.sqrt(1 - e2 * sl * sl)
    return ((N + alt) * cl * co, (N + alt) * cl * so, (N * (1 - e2) + alt) * sl)


def _azel(rx, sat, lat, lon):
    dx = [s - r for s, r in zip(sat, rx)]
    sl, cl = math.sin(math.radians(lat)), math.cos(math.radians(lat))
    so, co = math.sin(math.radians(lon)), math.cos(math.radians(lon))
    e = -so * dx[0] + co * dx[1]
    n = -sl * co * dx[0] - sl * so * dx[1] + cl * dx[2]
    u = cl * co * dx[0] + cl * so * dx[1] + sl * dx[2]
    rng = math.sqrt(sum(d * d for d in dx))
    return math.degrees(math.atan2(e, n)) % 360.0, math.degrees(math.asin(u / rng)), rng


def gpst_of_utc(t_utc):
    """datetime UTC (or unix seconds) -> continuous GPST seconds since the GPS epoch."""
    if isinstance(t_utc, (int, float)):
        t_utc = datetime.fromtimestamp(t_utc, tz=timezone.utc)
    return (t_utc - GPS_EPOCH).total_seconds() + GPS_UTC_LEAP


def best_eph(records, t_gpst, max_age=14400.0):
    """Freshest record whose toe is within max_age (4 h) of t; None if all stale."""
    cand = [e for e in records if abs(t_gpst - e["toe_gpst"]) < max_age]
    return min(cand, key=lambda e: abs(t_gpst - e["toe_gpst"])) if cand else None


def predict_all(eph, lat, lon, alt, t_utc, mask_deg=0.0, max_age=14400.0, signal=None,
                dcb=None):
    """Per visible sat: az/el, geometric range (m) with Earth-rotation (Sagnac)
    correction, range-rate (m/s), sat clock (s). Receiver clock NOT included --
    solving it from measured-vs-predicted IS the time bootstrap.

    max_age (s) is the toe validity window (default 4 h). GEOMETRY-only callers
    (observables az/el/range for the sky map / mapping factor) can widen it so a
    stale-in-memory or offline-cached ephemeris still yields az/el through a network
    gap -- Keplerian propagation is sub-degree/few-km for many hours past toe. The
    precise-clock callers (dead-reckon) keep the tight default; toe_age_s is per-sat
    so a consumer can still gate on freshness."""
    t = gpst_of_utc(t_utc)
    rx = _ecef_of_llh(lat, lon, alt)
    out = {}
    for key, recs in eph.items():
        e = best_eph(recs, t, max_age)
        if e is None:
            continue
        if e.get("health", 0) not in (0, 0.0) and key[0] != "C":
            # BDS exemption -- KEPT, but its original rationale was an artifact and is void:
            # it was written when health was misparsed from orb[25], which for BDS is AODC
            # (~1 on everything). The 2026-07-13 "healthy C21 reads health=1" evidence was
            # really "C21's AODC is 1" and proves nothing about SatH1. With the field fixed
            # (orb[21]) BDS SatH1 reads 0.0 on 36 of 37 sats, i.e. it looks as trustworthy
            # as G/E. Left permissive for now because a BDS-3 sat CAN legitimately carry
            # SatH1=1 (the word is B1I-referenced) while its B1C is fine -- dropping it would
            # silently lose that sat. The measured-vs-predicted integrity residuals referee.
            # TODO: re-test strict-for-C on sky now that the word is real; if no healthy sat
            # is lost, delete this exemption.
            continue
        # light-time + Sagnac: two iterations are plenty
        tau = 0.075
        for _ in range(2):
            pos, vel, clk = sat_pos_clk(e, t - tau)
            # rotate the sat position into the ECEF frame at reception time
            th = OMEGA_E[e["sys"]] * tau
            px = pos[0] * math.cos(th) + pos[1] * math.sin(th)
            py = -pos[0] * math.sin(th) + pos[1] * math.cos(th)
            pos_rx = (px, py, pos[2])
            az, el, rng = _azel(rx, pos_rx, lat, lon)
            tau = rng / C_LIGHT
        if el < mask_deg:
            continue
        rr = sum((p - r) * v for p, r, v in zip(pos_rx, rx, vel)) / rng
        # A0b: the broadcast clock refers to some OTHER signal combination; make it refer
        # to ours. `signal=None` (every non-broker caller) keeps the old, uncorrected value.
        _gd = group_delay_s(e, signal, dcb)
        out[key] = dict(az=az, el=el, range_m=rng, range_rate_mps=rr,
                        sat_clk_s=clk + _gd, tgd_s=_gd,
                        toe_age_s=t - e["toe_gpst"])
    return out


if __name__ == "__main__":
    lat, lon, alt = 43.968697, -79.252106, 260.0
    path = fetch_brdc()
    print("BRDC:", path)
    eph = parse_rinex_nav(path)
    print("parsed: %d sats (%s)" % (len(eph), "".join(sorted(set(k[0] for k in eph)))))
    pred = predict_all(eph, lat, lon, alt, datetime.now(timezone.utc), mask_deg=5.0)
    print("visible now: %d" % len(pred))
    for (s, p), v in sorted(pred.items(), key=lambda kv: -kv[1]["el"])[:12]:
        dop = -v["range_rate_mps"] / C_LIGHT * 1575.42e6
        print("  %s%02d el %4.1f az %5.1f  range %8.1f km  dop %+7.1f Hz  clk %+9.3f us  toe_age %4.0f min"
              % (s, p, v["el"], v["az"], v["range_m"] / 1e3, dop,
                 v["sat_clk_s"] * 1e6, v["toe_age_s"] / 60))
