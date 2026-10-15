#!/usr/bin/env python3
"""Broadcast-ephemeris supply: where the nav files come from and whether they cover the sky.

The sources are the merged daily BRDC (BKG, then CDDIS) and a rolling store of IGS
station-hourly files; parse_rinex_nav merges them per PRN. The broker owns the cache; other
processes should read it through cached_brdc().

broker_equiv replays pin the sky through GNSS_BRDC_DIR, so the fetch and merge code never runs
in a replay digest. Its gates are gnss_broker/test_skyscope.py (the policy decisions) and
gnss_broker/test_brdc_golden.py (the merged file, byte for byte, against captured station
files and a pinned clock). Run both after any change here.

The dependency is one-way: gnss_ephemeris imports this module, never the reverse, and nothing
here knows about orbits. Code that needs best_eph or predict_all belongs in gnss_ephemeris.

Mutable module state lives here (LOG_HOOK, _src_dead, _HOURLY_STATIONS): read and write it
through this module by name.
"""
import json
import os
import re
import threading
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "kotekan_gps")


def _atomic_write_bytes(path, data):
    """Write `data` to `path` atomically: a per-writer temp file beside it, then os.replace
    (atomic on one filesystem). The cache is shared across processes and threads, so a reader
    sees the old file or the new one, never a half-written gzip."""
    # The temp name must be unique per writer. Several threads and processes refresh the same
    # file; os.replace makes the rename atomic but does nothing for two writers sharing one
    # temp file, which interleaves their bytes and can publish a torn gzip no reader can parse.
    # File writes release the GIL, so threads race here with or without free threading.
    tmp = "%s.tmp.%d.%d" % (path, os.getpid(), threading.get_ident())
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        # Remove the temp file: nothing prunes this cache directory.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _earthdata_token():
    """Earthdata bearer token for the CDDIS BRDC mirror, from <cache>/.earthdata_token
    (chmod 600, never committed). None -> CDDIS is skipped."""
    try:
        with open(os.path.join(CACHE, ".earthdata_token")) as f:
            return f.read().strip() or None
    except Exception:
        return None


# CDDIS's merged daily mixed-nav products, by the `kind` slot the caller is filling. These are
# not the BKG product name; see _brdc_sources.
_CDDIS_DAILY = {
    "R": "BRDC00IGS_R_%04d%03d0000_01D_MN.rnx.gz",  # IGS merged broadcast
    "S": "BRDM00DLR_S_%04d%03d0000_01D_MN.rnx.gz",  # DLR multi-GNSS
}


def _brdc_sources(kind, year, doy):
    """Ordered (url, headers) mirrors for one daily mixed-nav file of variant `kind` (R or S):
      1. BKG (igs.bkg.bund.de), no auth, product BRDC00WRD.
      2. CDDIS (cddis.nasa.gov) with an Earthdata bearer token, when one exists: BRDC00IGS_R
         (IGS) and BRDM00DLR_S (DLR) under /gnss/data/daily/YYYY/DDD/YYp/. The BKG product
         name does not exist there; CDDIS's daily/YYYY/brdc/ holds only the legacy short-name
         GPS/GLONASS files.
    CDDIS accepts `Authorization: Bearer <token>` directly on the archive URL (a valid token
    returns the file, an invalid one 401s), so no cookie/redirect handling is needed.

    CDDIS publishes only closed days: the current day's YYp directory 404s, so this mirror
    can serve the previous day and replays, never today. While BKG is down, today's only
    source is the station-hourly merge (_fetch_station_hourly), which is why its coverage
    test matters.
    """
    name = "BRDC00WRD_%s_%04d%03d0000_01D_MN.rnx.gz" % (kind, year, doy)
    srcs = [
        (
            "https://igs.bkg.bund.de/root_ftp/IGS/BRDC/%04d/%03d/%s"
            % (year, doy, name),
            {},
        )
    ]
    tok = _earthdata_token()
    cd = _CDDIS_DAILY.get(kind)
    if tok and cd:
        srcs.append(
            (
                "https://cddis.nasa.gov/archive/gnss/data/daily/%04d/%03d/%02dp/%s"
                % (year, doy, year % 100, cd % (year, doy)),
                {"Authorization": "Bearer " + tok},
            )
        )
    return srcs


# IGS stations whose hourly mixed-nav files feed the station-hourly merge. Broadcast ephemeris is
# receiver-independent, so any station that tracked a satellite carries its current ephemeris,
# and the stations keep decoding the live broadcast while the merged daily's newest records lag
# hours behind (a BKG or IGS outage).
#
# Geographic diversity is the point: a satellite below the horizon at every listed station has
# no ephemeris anywhere in the merge, so the list spans continents (test_skyscope checks the
# country mix). Ordered best-coverage-first (BRUX and KIRU are strong on all three
# constellations) so the fetch budget, _HOURLY_MAX_FETCH, goes to the widest stations first and
# a top-up hour meets the target in fewer fetches.
_HOURLY_STATIONS = [
    "BRUX00BEL",
    "KIRU00SWE",
    "ALGO00CAN",
    "AL2H00CAN",
    "DARW00AUS",
    "IISC00IND",
    "MGUE00ARG",
    "STJO00CAN",
    "USN700USA",
    "JFNG00CHN",
    "CUT000AUS",
    "NRC100CAN",
]

# What "enough coverage" means, per constellation: (min_prn, required count of distinct PRNs
# >= min_prn with a usable record). Count satellites, not stations that answered: a station
# that carries no BeiDou is not a source of BeiDou. For C the floor is 19, the broker's BDS-3
# PRN floor (min_prn in gnss_broker/signals.py, alm_min_prn in gps_distributed_broker.py):
# BDS-2 satellites do not broadcast B1C/B2a/B2b, and counting them makes a thin BDS-3 sky look
# healthy.
# The targets sit below a full constellation (G ~32, E ~28, BDS-3 ~24 reachable): they mean
# "this merge is not starved", not "complete".
_HOURLY_TARGET = {"G": (1, 20), "E": (1, 18), "C": (19, 20)}
# Station files _hourly_target_met requires. The walk's exit test passes it
# max(fetched, _HOURLY_MIN_STATIONS), so this does not bind there.
_HOURLY_MIN_STATIONS = (
    2  # one file, however wide, may be a truncated hourly that looks like a thin sky
)
# Successful station-file fetches per walk, across all hours: the walk unions hours until the
# target is met, so a per-hour cap would let one thin hour spend the allowance and still come
# back short.
_HOURLY_MAX_FETCH = (
    12  # a budget of ~1.5 MB and a few seconds, spent on the widest stations first
)
_HOURLY_TTL_S = 1800.0  # reuse a merge that met its target for half an hour
# A below-target merge is retried sooner: the usual cause is merging early in the hour, before
# the slow stations upload the hour that just closed, and they are usually there minutes later.
_HOURLY_THIN_TTL_S = (
    300.0  # but retry a below-target merge after five minutes (see above)
)
_HOURLY_COV = "hourly_MN.cov.json"


def _hourly_cov_write(cache_dir, thin, got):
    """Record beside the merged file whether it met its target, so the cache gate can pick its
    TTL without parsing and recounting the file."""
    try:
        _atomic_write_bytes(
            os.path.join(cache_dir, _HOURLY_COV),
            json.dumps({"thin": bool(thin), "got": got, "t": time.time()}).encode(
                "ascii"
            ),
        )
    except Exception:
        pass


def _hourly_cov_read(cache_dir):
    """(thin, got): got is the coverage summary, or why it is unknown. Unknown counts as thin:
    a missing or unreadable record gets the short TTL and a re-merge, never the assumption that
    the file is complete.

    A record more than 1 s older than its file means another writer produced the file. The
    cache is shared, and a long-running process runs whatever version of this code it imported
    at start, so its merge may be wrong. That case reads as thin with a FOREIGN WRITE reason,
    so our own logic re-merges within the short TTL instead of keeping the other writer's file.
    """
    try:
        cov = os.path.join(cache_dir, _HOURLY_COV)
        with open(cov) as f:
            d = json.load(f)
        merged = os.path.join(cache_dir, "hourly_MN.rnx.gz")
        if (
            os.path.exists(merged)
            and os.path.getmtime(merged) > os.path.getmtime(cov) + 1.0
        ):
            return (
                True,
                (
                    "FOREIGN WRITE: hourly_MN.rnx.gz is %.0f s newer than its coverage "
                    "record, so another process (or another version of this code) wrote "
                    "it -- treating as thin and re-merging"
                    % (os.path.getmtime(merged) - os.path.getmtime(cov))
                ),
            )
        return bool(d.get("thin", True)), str(d.get("got", "?"))
    except Exception:
        return True, "no coverage record"


def _rec_is_fresh(line, when):
    """Is this RINEX 3 nav record's epoch within _HOURLY_KEEP_S of `when`, so it counts toward
    coverage?

    The window is predict_all's validity window, not a tighter freshness cut. BDS-3 updates
    hourly on the hour and IGS hourlies publish ~15-25 min late, so BeiDou's freshest record
    is legitimately up to ~2 h 25 min old; a cut under that age would count a full BeiDou sky
    as missing and send the walk over every station and hour. Fresh records still arrive every
    call because the walk always fetches the two newest hours; the exit test only asks whether
    the window holds enough of the sky.

    The record's first line carries sys+PRN, then the toc as `YYYY MM DD HH MM SS`. Anything
    unparseable counts as fresh: the test rejects only records it can prove stale, so a format
    surprise does not shrink coverage.
    """
    try:
        f = line[4:23].split()
        if len(f) < 6:
            return True
        toc = datetime(
            int(f[0]),
            int(f[1]),
            int(f[2]),
            int(f[3]),
            int(f[4]),
            int(float(f[5])),
            tzinfo=timezone.utc,
        )
        return abs((when - toc).total_seconds()) <= _HOURLY_KEEP_S
    except Exception:
        return True


def _hourly_target_met(seen, fetched):
    return fetched >= _HOURLY_MIN_STATIONS and all(
        len(seen[k2]) >= v[1] for k2, v in _HOURLY_TARGET.items()
    )


# This module is a library and owns no logger. A host installs its own through LOG_HOOK (the
# broker's sky.py sets its `_log`); until then messages go to stderr, which the broker unit
# appends to the broker log (StandardError=inherit). A thin merge must be logged: downstream it
# is invisible until a constellation's prediction collapses.
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


def _hourly_sources(station, when, token):
    """Ordered (url, headers) mirrors for one station's hourly mixed-nav file: CDDIS (when an
    Earthdata token exists), then ESA's GSSC (anonymous FTP).

    While BKG is down these hourlies are the only current-day ephemeris (CDDIS publishes merged
    dailies only for closed days), so they must not depend on one host or one credential. GSSC
    carries the same IGS hourly products and sometimes publishes an hour before CDDIS.
    """
    srcs = []
    doy = when.timetuple().tm_yday
    name = "%s_R_%04d%03d%02d00_01H_MN.rnx.gz" % (station, when.year, doy, when.hour)
    if token:
        srcs.append(
            (
                "https://cddis.nasa.gov/archive/gnss/data/hourly/%04d/%03d/%02d/%s"
                % (when.year, doy, when.hour, name),
                {"Authorization": "Bearer " + token},
            )
        )
    srcs.append(
        (
            "ftp://gssc.esa.int/gnss/data/hourly/%04d/%03d/%02d/%s"
            % (when.year, doy, when.hour, name),
            {},
        )
    )
    return srcs


# How much history the rolling store keeps: predict_all's 4 h validity window, so an older
# record cannot contribute to any prediction. Filtered on toc (the epoch on the record's first
# line), not toe: they coincide for GPS and Galileo and differ by minutes for BeiDou, and
# parsing toe means parsing the whole record, which is parse_rinex_nav's job.
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
        return datetime(
            int(f[0]),
            int(f[1]),
            int(f[2]),
            int(f[3]),
            int(f[4]),
            int(float(f[5])),
            tzinfo=timezone.utc,
        )
    except Exception:
        return default


def _store_load(cache_dir, when):
    """The rolling store: {(sys, prn, toc): text}, pruned to _HOURLY_KEEP_S, plus its header.

    An unparseable toc is kept, as in _rec_is_fresh: this layer drops what it can prove
    expired, never on doubt.
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
    """Merge IGS stations' hourly mixed-nav files into the rolling store,
    cache_dir/hourly_MN.rnx.gz, and return its path (None if there is nothing to write).

    It backs up a stale merged daily and is the only current-day source while the daily
    mirrors are down. One station's hour may lack a constellation, so the walk unions stations
    over the two newest hours, then tops up from older hours until every constellation meets
    _HOURLY_TARGET.

    Coverage decides, not recency. A station publishes the hour that just closed and uploads
    trickle in over the next ~40 min, so the first hour that answers is often half-populated,
    and a thin merge looks healthy downstream while it starves a constellation. An hour older
    costs ~1 h of toe age against a 4 h window; missing half a constellation costs far more.
    """
    import gzip

    # Cache gate: _fetch_brdc_now calls this on every refresh, so reuse the merged file for
    # _HOURLY_TTL_S when it met its target, but only for _HOURLY_THIN_TTL_S when it did not:
    # the missing stations usually upload within minutes. Whether it met its target is recorded
    # beside the file (_hourly_cov_write), since recounting means parsing it.
    local = os.path.join(cache_dir, "hourly_MN.rnx.gz")
    if os.path.exists(local):
        age = time.time() - os.path.getmtime(local)
        thin, cov = _hourly_cov_read(cache_dir)
        if age < (_HOURLY_THIN_TTL_S if thin else _HOURLY_TTL_S):
            if thin:
                _log_hourly(
                    "BRDC hourly merge: serving a BELOW-TARGET merge from cache "
                    "(%s, %.0f s old, retry in %.0f s). Prediction is thin -- probes, "
                    "drop gates and search hints all ride this."
                    % (cov, age, _HOURLY_THIN_TTL_S - age)
                )
            return local
    # ---- the rolling store ---------------------------------------------------------------
    # Accumulate, do not rebuild. A file built from only this call's hours holds just enough
    # history for the count at the moment it is written. BDS-3 updates hourly on the hour and
    # IGS hourlies publish ~15-25 min after the hour closes, so BeiDou's freshest available
    # record is up to ~2 h 25 min old, and a rebuilt file's BeiDou records can expire before the
    # next refresh. While BKG is down no daily covers that gap (CDDIS publishes only closed
    # days, and yesterday's records are past 4 h), so this file must hold the whole window.
    # Records are keyed (sys, prn, toc), so re-fetching an hour is idempotent and a record
    # carried by two stations is stored once.
    store, header = _store_load(cache_dir, when)
    n_start = len(store)
    fetched = 0
    seen = {"G": set(), "E": set(), "C": set()}
    for (sysc, prn, toc) in store:
        if sysc in seen and prn >= _HOURLY_TARGET[sysc][0]:
            t = _toc_of((sysc, prn, toc))
            if t is None or abs((when - t).total_seconds()) <= _HOURLY_KEEP_S:
                seen[sysc].add(prn)
    done = set()  # (station, hour) already fetched -- never pay twice
    for dh in (0, 1, 2, 3):
        h = when - timedelta(hours=dh)
        # The exit test decides how far back to walk, not whether to fetch: the two newest
        # hours are always attempted, older ones only to top up coverage. Stopping as soon as
        # the store met its target would stop it refreshing, and it would coast to expiry.
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
            # Mirror order is per station, not per hour: a mirror that has not published this
            # hour yet 404s while the other may have it, and the aim is the freshest hour
            # available anywhere, not one host walked backwards in time.
            txt = None
            for url, hdrs in _hourly_sources(st, h, token):
                # A dead host is paid for once, not once per station: twelve stations x a 20 s
                # timeout is four minutes.
                if _src_skip(url):
                    continue
                try:
                    req = urllib.request.Request(url, headers=hdrs)
                    with urllib.request.urlopen(req, timeout=20) as r:
                        raw = r.read()
                    if raw[:2] != b"\x1f\x8b":
                        continue  # a 200 that is not gzip: this hour, not this host
                    _src_ok(url)
                    txt = gzip.decompress(raw).decode("ascii", "replace")
                    break
                except Exception as e:
                    _src_failed(
                        url, exc=e
                    )  # a 404 or FTP 550 is an hour not yet published, not a
                    continue  # dead host; _src_failed blacklists only the latter
            if txt is None:
                continue
            k = txt.find("END OF HEADER")
            if k < 0:
                continue
            eoh = txt.find("\n", k) + 1
            if header is None:
                header = txt[:eoh]
            fetched += 1
            for sysc, prn, toc, text in _split_records(txt[eoh:]):
                t = _toc_of((sysc, prn, toc))
                if t is not None and abs((when - t).total_seconds()) > _HOURLY_KEEP_S:
                    continue
                store[(sysc, prn, toc)] = text
            for ln in txt[eoh:].splitlines():
                if len(ln) > 3 and ln[0] in seen and ln[1:3].isdigit():
                    prn = int(ln[1:3])
                    # Count usable records, not present ones: a record past the predictor's
                    # validity window satisfies a count and predicts nothing.
                    if prn >= _HOURLY_TARGET[ln[0]][0] and _rec_is_fresh(ln, when):
                        seen[ln[0]].add(prn)
    # ---- one file: the whole rolling window, not just this call's fetches -----------------
    if not (header and store):
        return None
    got = "/".join("%s%d" % (k2, len(seen[k2])) for k2 in "GEC")
    thin = [k2 for k2, v in _HOURLY_TARGET.items() if len(seen[k2]) < v[1]]
    hours = sorted({hh for _, hh in done})
    # Oldest first, then by key: a deterministic order (test_brdc_golden hashes the file) that
    # reads in time order.
    order = sorted(
        store, key=lambda k: (_toc_of(k, datetime(1980, 1, 6, tzinfo=timezone.utc)), k)
    )
    bodies = [store[k] for k in order]
    # A negative age is normal: GPS and Galileo centre toe in the fit interval, so a fresh record
    # is often stamped up to ~1 h ahead of now. The log says "ahead" rather than printing a
    # minus sign that reads as a clock fault.
    span = [t for t in (_toc_of(k) for k in order) if t is not None]
    if span:
        _n = (when - max(span)).total_seconds() / 60.0
        _o = (when - min(span)).total_seconds() / 60.0
        age = "newest %s, oldest %.0f min old" % (
            ("%.0f min ahead" % -_n) if _n < 0 else ("%.0f min old" % _n),
            _o,
        )
    else:
        age = "?"
    if thin:
        # A below-target merge still returns a file, so log it: downstream a thin file looks
        # healthy until a constellation's prediction collapses.
        _log_hourly(
            "BRDC hourly merge: %d station-file(s) over hour(s) %s -> %s; BELOW TARGET "
            "for %s (want %s). Prediction for %s will be thin -- probes, drop gates "
            "and search hints all ride this. Retrying in %.0f s."
            % (
                fetched,
                ",".join("%02d" % x for x in hours),
                got,
                "+".join(thin),
                "/".join("%s%d" % (k2, _HOURLY_TARGET[k2][1]) for k2 in "GEC"),
                "+".join(thin),
                _HOURLY_THIN_TTL_S,
            )
        )
    else:
        _log_hourly(
            "BRDC hourly merge: %d station-file(s) over hour(s) %s -> %s, target met. "
            "Store %d -> %d record(s); %s."
            % (
                fetched,
                ",".join("%02d" % x for x in hours),
                got,
                n_start,
                len(store),
                age,
            )
        )
    _atomic_write_bytes(
        local, gzip.compress((header + "".join(bodies)).encode("ascii", "replace"))
    )
    _hourly_cov_write(cache_dir, bool(thin), got)
    return local


def _try_refresh_daily(kind, year, doy, cache_dir, tok):
    """Best-effort fetch of one daily-BRDC variant (S or R) for the per-PRN merge. Reuses a
    cache younger than 2 h, else downloads through _brdc_sources. Returns the cached path or
    None; never raises. Fetching each variant on its own lets a fresh one reach the merge when
    _fetch_brdc_merged returned a frozen one (one variant can stop updating mid-day while the
    other does not)."""
    name = "BRDC00WRD_%s_%04d%03d0000_01D_MN.rnx.gz" % (kind, year, doy)
    local = os.path.join(cache_dir, name)
    if os.path.exists(local) and time.time() - os.path.getmtime(local) < 7200:
        return local
    tried = 0
    for url, headers in _brdc_sources(kind, year, doy):
        if _src_skip(url):
            continue  # known dead, still cooling down -- do not pay for it
        tried += 1
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            if data[:2] != b"\x1f\x8b":
                _src_failed(url)  # login/error/404 page: this mirror is not serving
                continue
            _src_ok(url)
            _atomic_write_bytes(local, data)
            return local
        except Exception as e:
            _src_failed(url, exc=e)
            continue
    # Every source was skipped, so every one is cooling down: clear all the marks so the next
    # call retries at once instead of waiting out the cooldown.
    if tried == 0:
        _src_dead.clear()
    return local if os.path.exists(local) else None


# ── Negative cache for dead sources ────────────────────────────────────────────────────────
# A dead mirror is paid for once, not on every fetch. The sources are tried in order and an
# unreachable host blocks each attempt for the full timeout (30 s per daily, 20 s per station
# file) before the next mirror is tried. A host that fails is skipped for _SRC_COOLDOWN_S.
# Being wrong is bounded: a recovered host is retried after the cooldown, and the merge is
# per-PRN freshest-wins, so a later merge repairs anything a skipped mirror would have supplied.
_SRC_COOLDOWN_S = 300.0
_src_dead = {}  # scheme://host -> time.time() when it may be retried


def _src_key(url):
    """Scheme+host, so one dead host is skipped for every path under it."""
    try:
        pr = urllib.parse.urlsplit(url)
        return pr.scheme + "://" + pr.netloc
    except Exception:
        return url


def _src_skip(url, now=None):
    return _src_dead.get(_src_key(url), 0.0) > (now or time.time())


# Statuses that are a statement about the path, not the host. Any other failure (a timeout, a
# refused connection, a 5xx, a daily mirror's 200 that is not gzip) says the host is not
# serving, and blacklisting it is right.
_PATH_STATUS = (404, 410)
# The FTP equivalents: 550 "no such file or directory", 450 "file unavailable" and 553 "file
# name not allowed". urllib wraps them in a URLError with no .code, so the status survives only
# in the message ("ftp error: 550 CWD command failed: ...").
_FTP_PATH_RE = re.compile(r"ftp error:\s*(\d{3})")
_FTP_PATH_STATUS = ("550", "450", "553")


def _is_path_failure(exc):
    """Is this exception about the path (absent file) rather than the host (not serving)?"""
    if exc is None:
        return False
    if getattr(exc, "code", None) in _PATH_STATUS:
        return True
    m = _FTP_PATH_RE.search(str(exc))
    return bool(m and m.group(1) in _FTP_PATH_STATUS)


def _src_failed(url, now=None, exc=None):
    """Blacklist a source host for _SRC_COOLDOWN_S, unless the failure was about the path.

    The key is scheme+host, so one dead server is skipped for every path under it. An absent
    path is not a dead host: CDDIS never has the current day's daily, and the hour walk starts
    at the current hour, which no mirror has published yet. Those requests fail by design, as
    an HTTP 404 or an FTP 550 (a URLError with no .code), and blacklisting on them would
    disable the host that serves the previous day or the previous hour.
    """
    if _is_path_failure(exc):
        return
    _src_dead[_src_key(url)] = (now or time.time()) + _SRC_COOLDOWN_S


def _src_ok(url):
    _src_dead.pop(_src_key(url), None)


# A daily last written longer ago than this cannot hold a record inside predict_all's 4 h toe
# window, so cached_brdc skips it. Generous but bounded: a daily is rewritten only while its
# day is current.
_CACHED_DAILY_MAX_AGE_S = 172800.0  # 2 days


def cached_brdc(cache_dir=CACHE):
    """Read-only: the nav files already in the cache, freshest first by mtime. Never fetches,
    never writes.

    For every consumer that is not the broker. The cache is shared, and a process that writes
    it with its own copy of the merge logic (a long-running one runs whatever version it
    imported) undoes the broker's rolling store. The broker owns the cache because its
    correctness depends on it; everyone else reads what is there. A consumer that only wants
    geometry could instead ask the broker, which knows every satellite's az/el.

    Returns [] when the cache is empty, so callers can fall back rather than raise.
    """
    # Not the whole directory: the cache keeps every daily ever downloaded, and parsing them all
    # costs tens of megabytes for records best_eph then discards. Take the station-hourly merge
    # plus dailies written within _CACHED_DAILY_MAX_AGE_S.
    try:
        names = [f for f in os.listdir(cache_dir) if f.endswith((".rnx", ".rnx.gz"))]
    except Exception:
        return []
    now = time.time()
    out = []
    for f in names:
        full = os.path.join(cache_dir, f)
        try:
            age = now - os.path.getmtime(full)
        except OSError:
            continue
        if f.startswith("hourly_") or age <= _CACHED_DAILY_MAX_AGE_S:
            out.append((age, full))
    out.sort()  # freshest first
    return [f for _, f in out]


# ── The fetch runs off the control loop ────────────────────────────────────────────────────
# A network fetch inside the broker's control pass stalls it (20 s per station file, 30 s per
# daily mirror) past the telemetry ring's ~10 s of windows. Lost windows make
# fleetadr.fold_record break every fleet-ADR arc on that chain, so one slow mirror would throw
# away the carrier level of every satellite on every chain. The loop is handed what is already
# on disk, and one daemon thread at a time refreshes it; the next call after it lands sees the
# new files. The synchronous path remains for when there is nothing to hand out: a pinned
# replay, an empty cache, or a caller with no ephemeris at all (block=True).
_REFRESH = {"thread": None, "lock": threading.Lock()}


def _cached_sources(when, cache_dir):
    """The cached files fetch_brdc hands out without touching the network: the daily variants
    on disk for `when`'s day (the previous day's only when this day has none, as in
    _fetch_brdc_merged), then the station-hourly merge. [] when nothing is cached.
    parse_rinex_nav merges per PRN by toe, so the order only breaks identical-toe ties."""
    out = []
    for back in (0, 1):
        d = when - timedelta(days=back)
        doy = d.timetuple().tm_yday
        for kind in ("S", "R"):
            p = os.path.join(
                cache_dir,
                "BRDC00WRD_%s_%04d%03d0000_01D_MN.rnx.gz" % (kind, d.year, doy),
            )
            if os.path.exists(p):
                out.append(p)
        if out:
            break
    hourly = os.path.join(cache_dir, "hourly_MN.rnx.gz")
    if os.path.exists(hourly):
        out.append(hourly)
    return out


def _refresh_in_background(when, cache_dir):
    """Start _fetch_brdc_now on a daemon thread unless one is still running. Returns True when
    a thread was started. A failure is logged and the cached files stay in service."""
    with _REFRESH["lock"]:
        t = _REFRESH["thread"]
        if t is not None and t.is_alive():
            return False

        def run():
            try:
                _t0 = time.time()
                _got = _fetch_brdc_now(when, cache_dir)
                _new = [
                    os.path.basename(f)
                    for f in _got
                    if os.path.exists(f) and os.path.getmtime(f) >= _t0 - 1.0
                ]
                if _new:
                    _log_hourly(
                        "BRDC background refresh: %d source(s), %d rewritten in %.1f s: %s"
                        % (len(_got), len(_new), time.time() - _t0, " ".join(_new))
                    )
            except Exception as e:
                _log_hourly(
                    "BRDC background refresh failed (%s); the cached sources stay "
                    "in service until the next refresh" % e
                )

        t = threading.Thread(target=run, name="brdc-refresh", daemon=True)
        _REFRESH["thread"] = t
        t.start()
        return True


_OFFLINE_WHEN_S = 3600.0


def _offline_cache_dir(when, cache_dir):
    """A `when` more than an hour from the wall clock gets its own cache, cache_dir/offline/<day>.

    The rolling store in cache_dir belongs to the process that follows the clock (the broker).
    An offline caller asking for another day's sky (the beam cube) would otherwise merge the
    live store against that day, prune it to that window and rewrite it, and the broker's next
    reload would load the wrong day's records and lose the sky. The guard lives here, not in
    the callers, so no caller can bypass it."""
    if when is None or os.path.basename(os.path.dirname(cache_dir)) == "offline":
        return cache_dir
    if abs((datetime.now(timezone.utc) - when).total_seconds()) < _OFFLINE_WHEN_S:
        return cache_dir
    d = os.path.join(cache_dir, "offline", when.strftime("%Y%m%d"))
    os.makedirs(d, exist_ok=True)
    return d


def fetch_brdc(when=None, cache_dir=CACHE, block=False):
    """Ordered list of BRDC nav-source files for `when`, without blocking the caller on the
    network: the files already on disk are returned and a refresh starts on a daemon thread
    (see the note above _REFRESH). Synchronous (the full _fetch_brdc_now) only when
    GNSS_BRDC_DIR pins the sky, when nothing is cached, or when block=True: a caller with no
    ephemeris yet must wait, one with an ephemeris must not.

    The ordering and merging contract is documented on _fetch_brdc_now."""
    if os.environ.get("GNSS_BRDC_DIR") or block:
        return _fetch_brdc_now(when, cache_dir)
    when = when or datetime.now(timezone.utc)
    cache_dir = _offline_cache_dir(when, cache_dir)
    have = _cached_sources(when, cache_dir)
    if not have:
        return _fetch_brdc_now(when, cache_dir)
    _refresh_in_background(when, cache_dir)
    return have


def _fetch_brdc_now(when=None, cache_dir=CACHE):
    """Synchronous: fetch, then return the ordered list of BRDC nav-source files, best-first.
    parse_rinex_nav merges them per PRN (union of records, freshest toe per PRN wins), so one
    frozen product cannot starve a PRN another source carries fresh. Every source contributes;
    callers do parse_rinex_nav(fetch_brdc(...)).

    GNSS_BRDC_DIR pins the sky: when set, return exactly the nav files in that directory (no
    cache, no network, no date logic), and raise if it holds none. That makes broker_equiv's
    replays hermetic: the ephemeris is an input too, refreshed several times a day, and it
    moves the on-sky digests with identical code. Live brokers never set it.

    This function fetches and writes, so it belongs to the process that owns the cache, the
    broker. Anything else that wants geometry should use cached_brdc() or ask the broker."""
    pin = os.environ.get("GNSS_BRDC_DIR")
    if pin:
        pinned = sorted(
            os.path.join(pin, f)
            for f in os.listdir(pin)
            if f.endswith((".rnx", ".rnx.gz"))
        )
        if not pinned:
            raise RuntimeError(
                "GNSS_BRDC_DIR=%s contains no .rnx/.rnx.gz nav files -- a "
                "pinned replay must not fall through to the live sky" % pin
            )
        return pinned
    when = when or datetime.now(timezone.utc)
    cache_dir = _offline_cache_dir(when, cache_dir)
    # The daily is preferred, not required. _fetch_brdc_merged raises when no daily is
    # reachable or cached, as can happen after the UTC day roll while BKG is down
    # (CDDIS publishes only closed days). The station-hourly merge alone still carries a usable
    # sky, so log the failure and go on rather than discard it.
    srcs = []
    try:
        srcs.append(_fetch_brdc_merged(when, cache_dir))  # authoritative daily
    except Exception as _e:
        _log_hourly(
            "BRDC daily unavailable (%s) -- continuing on the station-hourly merge "
            "alone. Expected while BKG is down and CDDIS has not yet closed the day; "
            "the hourly IS the current-day source in that regime. The sky is thinner "
            "than a daily+hourly union, NOT absent." % _e
        )
    tok = _earthdata_token()
    doy = when.timetuple().tm_yday
    # Both daily variants (S first), then the station-hourly merge, each best-effort and
    # cache-gated: a failure only leaves that source out. A stale variant is harmless; its
    # older toes lose the per-PRN freshest-toe selection.
    for kind in ("S", "R"):
        p = _try_refresh_daily(kind, when.year, doy, cache_dir, tok)
        if p and p not in srcs:
            srcs.append(p)
    hourly = _fetch_station_hourly(when, cache_dir, tok)
    if hourly and hourly not in srcs:
        srcs.append(hourly)
    if not srcs:
        # The daily and the hourly both failed: fatal, and the message names both.
        raise RuntimeError(
            "no BRDC source at all: the daily is unreachable/uncached AND the "
            "station-hourly merge returned nothing. Check BKG/CDDIS/GSSC "
            "reachability and %s" % cache_dir
        )
    return srcs


def _fetch_brdc_merged(when=None, cache_dir=CACHE):
    """The merged daily BRDC file for `when` (BKG first, CDDIS-with-token fallback)."""
    os.makedirs(cache_dir, exist_ok=True)
    when = when or datetime.now(timezone.utc)
    # Finish each day (fresh cache, then download, then a stale same-day cache) before falling
    # back to the previous day. A current-day file predicts even when hours stale, because its
    # toes still bracket now; yesterday's stops predicting a few hours into the UTC day
    # (best_eph's 4 h window), and then the broker seeds and hints nothing. So the previous
    # day is used only when this day yields nothing at all, not merely when its download fails.
    for back in (0, 1):
        d = when - timedelta(days=back)
        doy = d.timetuple().tm_yday
        # (kind, local path) pairs: the mirrors serve different product names for the same
        # variant (BKG BRDC00WRD vs CDDIS BRDC00IGS/BRDM00DLR), so the download loop below
        # needs to know which variant slot it is filling.
        locals_ = [
            (
                kind,
                os.path.join(
                    cache_dir,
                    "BRDC00WRD_%s_%04d%03d0000_01D_MN.rnx.gz" % (kind, d.year, doy),
                ),
            )
            for kind in ("S", "R")
        ]
        # 1) a fresh current-day cache (< 2 h) or any cached previous-day file (final) is
        #    usable as-is, no network needed.
        for _kind, local in locals_:
            if os.path.exists(local) and (
                back == 1 or time.time() - os.path.getmtime(local) < 7200
            ):
                return local
        # 2) (re-)download this day (current-day files grow through the day, so refresh a stale
        #    cache to pull the newest toes). Try each mirror (BKG, then CDDIS if a token exists);
        #    a valid BRDC is gzip, so reject anything without the gzip magic (an auth/error page
        #    served as 200 must never be cached as if it were ephemeris). This loop repeats
        #    _try_refresh_daily's and must keep the same negative cache.
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
        # 3) download failed but a stale same-day cache exists -> use it (still predicts) rather
        #    than falling to the previous day's dead-for-today ephemeris.
        for _kind, local in locals_:
            if os.path.exists(local):
                return local
        # nothing for this day at all -> fall back to the previous day (early-UTC-morning case).
    raise RuntimeError("no BRDC file reachable (network?) and no cache")
