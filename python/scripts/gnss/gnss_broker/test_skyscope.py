"""The BRDC sky supply: per-constellation prediction state, coverage-driven merging, mirrors.

    python3 -m gnss_broker.test_skyscope

Replays pin the sky through GNSS_BRDC_DIR, so they never exercise the fetch path or the
collapse bridge; these tests do, against stubs:

  1. brdc_predict's collapse bookkeeping is per (constellation, min_prn). receiver.brdc() hands
     every chain one shared store so one parse serves G/E/C, but `peak_n` and `last_good`
     describe one constellation's sky: at store scope, peak_n would be a max over
     constellations (BeiDou's smaller sky would read as a permanent collapse against GPS's
     peak) and the bridge would serve another constellation's satellites.
  2. gnss_brdc_supply's station-hourly merge stops on coverage of usable records per
     constellation, not on how many stations answered: a station that carries no BDS-3 must
     not fill the quota.
  3. The mirrors: CDDIS is asked for a file it publishes, a 404 or an FTP 550 (an absent
     path) does not blacklist the host, every station hourly has an anonymous mirror, and
     fetch_brdc serves the cache and refreshes on a thread rather than stalling the control
     loop.

Each is a fallback that would otherwise reproduce the primary's output while dropping one of
its safeguards, running exactly when things are already degraded (the fallback audit in
docs/CHORD_PEER_COMPARISON_PURGE.md, appendix).

@author Keith Vanderlinde
"""

import gzip
import os
import shutil
import sys
import time
import tempfile


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


# ---------------------------------------------------------------------------------------
# 1. brdc_predict: per-constellation collapse bookkeeping
# ---------------------------------------------------------------------------------------


class _FakeEphMod(object):
    """Stands in for gnss_ephemeris. `visible` is what predict_all returns, set per call."""

    def __init__(self):
        self.visible = {}
        self.LOG_HOOK = None
        self.fetches = 0

    def predict_all(self, eph, lat, lon, alt, t, mask_deg=0.0, **kw):
        return dict(self.visible)

    # A collapse forces eph_t back so the next call re-fetches; the stub has to answer that
    # or the test measures its own AttributeError instead of the bridge.
    def fetch_brdc(self, when=None, block=False):
        self.fetches += 1
        return "stub"

    def parse_rinex_nav(self, path):
        return {"x": 1}


class _CountingEphMod(_FakeEphMod):
    """_FakeEphMod that counts parse attempts and fails them while `fail` is set."""

    def __init__(self):
        _FakeEphMod.__init__(self)
        self.parses = 0
        self.fail = False

    def parse_rinex_nav(self, path):
        self.parses += 1
        if self.fail:
            raise IOError("stub parse failure")
        return {"x": 1}


def _sats(sysc, prns, el=10.0):
    return {
        (sysc, p): dict(
            az=0.0,
            el=el,
            range_m=2.0e7,
            range_rate_mps=0.0,
            sat_clk_s=0.0,
            toe_age_s=0.0,
        )
        for p in prns
    }


def _predict(state, ge, sysc, min_prn, prns, el=10.0):
    from gnss_broker.sky import brdc_predict
    from datetime import datetime, timezone

    ge.visible = _sats(sysc, prns, el)
    return brdc_predict(
        state,
        49.32,
        -119.62,
        545.0,
        sysc,
        min_prn,
        datetime.now(timezone.utc),
        1176.45e6,
    )


def test_predict_scope():
    from gnss_broker import sky

    print("brdc_predict: the collapse bookkeeping is PER CONSTELLATION")
    ge = _FakeEphMod()
    # eph_t fresh so no refetch is attempted; `mod` is the stub.
    state = {"mod": ge, "eph": {"x": 1}, "eph_t": sky._now()}

    # GPS predicts first on the shared store and sets a wide peak.
    g = _predict(state, ge, "G", 1, range(1, 25))
    check(len(g) == 24, "GPS predicts its 24 satellites")

    # BeiDou's first prediction, 8 satellites. Against GPS's peak this would be a collapse;
    # against its own (none yet) it is simply the sky.
    c = _predict(state, ge, "C", 19, range(19, 27))
    check(len(c) == 8, "BeiDou's first prediction returns its own 8 satellites")
    check(
        all(p >= 19 for p in c),
        "and not one PRN below its min_prn -- the constellation identity survives",
    )
    check(
        not any(p < 19 for p in c),
        "specifically: no PRN below 19, where GPS and Galileo PRNs would leak in",
    )

    # The two peaks are independent.
    sc = state.get("scope", {})
    check(
        sc.get(("G", 1), {}).get("peak_n") == 24
        and sc.get(("C", 19), {}).get("peak_n") == 8,
        "each constellation carries its own peak (G 24, C 8), not one shared max",
    )

    # A real BeiDou collapse: 24 satellites, then 8.
    ge2 = _FakeEphMod()
    st2 = {"mod": ge2, "eph": {"x": 1}, "eph_t": sky._now()}
    _predict(st2, ge2, "C", 19, range(19, 43))  # 24 BDS-3, sets the peak
    _predict(st2, ge2, "G", 1, range(1, 31))  # GPS runs in between on the same store
    c2 = _predict(st2, ge2, "C", 19, range(19, 27))  # 8 -- a genuine collapse
    check(
        len(c2) == 24,
        "a REAL collapse still bridges (24 sats returned for 8 predicted)",
    )
    check(
        all(p >= 19 for p in c2),
        "and it bridges on BEIDOU's last good set, never on the GPS set that ran between",
    )

    # Recovery clears the bridge.
    c3 = _predict(st2, ge2, "C", 19, range(19, 43))
    check(
        len(c3) == 24 and "bridge_since" not in st2.get("scope", {}).get(("C", 19), {}),
        "recovery comes off the bridge and forgets it",
    )


def test_bridge_speaks():
    from gnss_broker import sky

    print(
        "brdc_predict: a bridged sky says so every time it is served, not once per 600 s"
    )
    ge = _FakeEphMod()
    state = {"mod": ge, "eph": {"x": 1}, "eph_t": sky._now()}
    said = []
    orig_rl, orig_log = sky._log_rl, sky._log
    sky._log_rl = lambda key, msg, every_s=10.0: said.append(msg)
    sky._log = lambda msg: said.append(msg)
    try:
        _predict(state, ge, "C", 19, range(19, 43))
        said[:] = []
        for _ in range(3):
            _predict(state, ge, "C", 19, range(19, 27))
        # The announcement sits outside the 600 s refetch cooldown, on its own rate-limit
        # key (stubbed here), because every bridged cycle serves frozen elevations.
        bridged = [m for m in said if "COLLAPSE" in m]
        check(
            len(bridged) == 3,
            "every bridged cycle announces (3 of 3), not just the first",
        )
        check(
            all("BRIDGING" in m for m in bridged),
            "and it says BRIDGING -- it names the frozen sky, not the refetch",
        )
        check(
            all("STALE" in m for m in bridged),
            "and warns the elevations are stale -- probes and drop gates ride them",
        )
        said[:] = []
        _predict(state, ge, "C", 19, range(19, 43))
        check(any("RECOVERED" in m for m in said), "recovery is announced too")
    finally:
        sky._log_rl, sky._log = orig_rl, orig_log


def test_failed_refresh_backs_off():
    from gnss_broker import sky

    print("brdc_predict: a failed refresh is retried in 10 min, not on the next pass")
    ge = _CountingEphMod()
    t = [1.0e6]
    saved = (sky._now, sky._log, sky._log_rl)
    sky._now = lambda: t[0]
    sky._log = lambda msg: None
    sky._log_rl = lambda key, msg, every_s=10.0: None
    try:
        # The last good parse was 1000 s ago, past the 900 s gate, and the refresh fails.
        state = {"mod": ge, "eph": {"x": 1}, "eph_t": t[0] - 1000.0}
        ge.fail = True
        _predict(state, ge, "G", 1, range(1, 25))
        t[0] += 2.0
        g = _predict(state, ge, "G", 1, range(1, 25))
        check(
            ge.parses == 1,
            "two passes 2 s apart make one attempt, not one each (%d)" % ge.parses,
        )
        check(len(g) == 24, "and the pass predicts on the set in hand")
        t[0] += 588.0
        _predict(state, ge, "G", 1, range(1, 25))
        check(ge.parses == 1, "590 s after the failure it still waits (%d)" % ge.parses)
        t[0] += 12.0
        ge.fail = False
        _predict(state, ge, "G", 1, range(1, 25))
        check(ge.parses == 2, "602 s after it, it tries again (%d)" % ge.parses)
        check(state["eph_t"] == t[0], "and the good parse restarts the gate")

        # A bridged BeiDou scope re-parses on the 60 s gate, and the collapse forces eph_t
        # back; neither may cut the back-off short.
        ge = _CountingEphMod()
        ge.fail = True
        lg = {19: (0.0,) * 9}
        state = {
            "mod": ge,
            "eph": {"x": 1},
            "eph_t": t[0] - 1000.0,
            "scope": {("C", 19): {"peak_n": 24, "last_good": lg, "bridge_since": t[0]}},
        }
        _predict(state, ge, "C", 19, range(19, 27))
        t[0] += 61.0
        _predict(state, ge, "C", 19, range(19, 27))
        check(
            ge.parses == 1,
            "on the bridge's 60 s gate a failed refresh still waits (%d)" % ge.parses,
        )
    finally:
        sky._now, sky._log, sky._log_rl = saved


def test_bridge_rereads_soon():
    from gnss_broker import sky

    print("brdc_predict: a bridged constellation re-parses within a minute")
    ge = _CountingEphMod()
    t = [1.0e6]
    saved = (sky._now, sky._log, sky._log_rl)
    sky._now = lambda: t[0]
    sky._log = lambda msg: None
    sky._log_rl = lambda key, msg, every_s=10.0: None
    try:
        state = {"mod": ge, "eph": {"x": 1}, "eph_t": t[0]}
        _predict(state, ge, "C", 19, range(19, 43))  # 24 BDS-3 set the peak
        t[0] += 2.0
        _predict(state, ge, "C", 19, range(19, 27))  # 8: collapse, bridge
        t[0] += 2.0
        _predict(state, ge, "C", 19, range(19, 27))
        check(
            ge.parses == 1,
            "the collapse forces a re-parse on the next pass (%d)" % ge.parses,
        )
        # That re-parse read the cached files; fetch_brdc only started the download, and
        # what it lands must be read while the bridge lasts.
        t[0] += 61.0
        _predict(state, ge, "C", 19, range(19, 27))
        check(
            ge.parses == 2,
            "61 s later, still on the bridge, it re-parses again (%d)" % ge.parses,
        )
        t[0] += 61.0
        _predict(state, ge, "G", 1, range(1, 31))
        check(
            ge.parses == 2,
            "a constellation off the bridge keeps the 900 s gate (%d)" % ge.parses,
        )
        _predict(state, ge, "C", 19, range(19, 43))  # recovered
        n = ge.parses
        t[0] += 61.0
        _predict(state, ge, "C", 19, range(19, 43))
        check(
            ge.parses == n,
            "and once BeiDou recovers it is back on the 900 s gate (%d)" % ge.parses,
        )
    finally:
        sky._now, sky._log, sky._log_rl = saved


def test_one_refresh_per_store():
    import threading
    from gnss_broker import sky

    print("brdc_predict: chains sharing the store do not parse the same files together")
    entered, release = threading.Event(), threading.Event()

    class _SlowEphMod(_CountingEphMod):
        def parse_rinex_nav(self, path):
            self.parses += 1
            if self.parses == 1:
                entered.set()
                release.wait(3.0)
            return {"x": 1}

    ge = _SlowEphMod()
    saved = sky._log
    sky._log = lambda msg: None
    state = {"mod": ge, "eph": {"x": 1}, "eph_t": sky._now() - 1000.0}
    th = threading.Thread(target=_predict, args=(state, ge, "G", 1, range(1, 31)))
    th.start()
    try:
        check(entered.wait(3.0), "the first chain is inside its parse")
        t0 = time.monotonic()
        c = _predict(state, ge, "C", 19, range(19, 43))
        waited = time.monotonic() - t0
        check(
            ge.parses == 1,
            "a second chain arriving meanwhile does not parse too (%d)" % ge.parses,
        )
        check(len(c) == 24, "it predicts on the set in hand instead")
        check(
            waited < 1.0, "without waiting for the first one's parse (%.2f s)" % waited
        )
    finally:
        release.set()
        th.join(3.0)
        sky._log = saved


# ---------------------------------------------------------------------------------------
# 2. _fetch_station_hourly: stop on coverage, not on a station count
# ---------------------------------------------------------------------------------------

_HDR = (
    "     3.05           NAVIGATION DATA     M                   RINEX VERSION / TYPE\n"
    "                                                            END OF HEADER\n"
)


def _rinex(prns, age_s=0.0):
    """A minimal RINEX-3 nav body carrying one record per (sys, prn).

    The epoch is the wall clock minus `age_s`, never a fixed date: coverage counts only
    records fresh enough to predict from, so a dated fixture would go stale with the calendar.
    Pass `age_s` when a test wants staleness.
    """
    from datetime import datetime, timezone, timedelta

    toc = (datetime.now(timezone.utc) - timedelta(seconds=age_s)).strftime(
        "%Y %m %d %H %M %S"
    )
    out = []
    for sysc, ps in prns.items():
        for p in ps:
            out.append("%s%02d %s 0.0e+00 0.0e+00 0.0e+00\n" % (sysc, p, toc))
            out.append("     0.0e+00 0.0e+00 0.0e+00 0.0e+00\n")
    return "".join(out)


def test_hourly_coverage():
    print("_fetch_station_hourly: coverage is the quota, not the number of sources")
    from datetime import datetime, timezone
    import urllib.request

    # Station catalogue for this test. NOBDS* answer wide on G/E but carry only BDS-2
    # (C01-C18), which the BeiDou target does not count.
    catalogue = {
        "NOBDS1": {"G": range(1, 31), "E": range(1, 26), "C": range(1, 19)},
        "NOBDS2": {"G": range(1, 31), "E": range(1, 26), "C": range(1, 19)},
        "NOBDS3": {"G": range(1, 31), "E": range(1, 26), "C": range(1, 19)},
        "NOBDS4": {"G": range(1, 31), "E": range(1, 26), "C": range(1, 19)},
        "WIDEAA": {"G": range(1, 31), "E": range(1, 26), "C": range(19, 41)},
        "WIDEBB": {"G": range(1, 31), "E": range(1, 26), "C": range(19, 41)},
    }
    asked = []

    class _Resp(object):
        def __init__(self, data):
            self.data = data

        def read(self):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    _urls = []

    def fake_urlopen(req, **kw):
        name = req.full_url.rsplit("/", 1)[-1]
        st = name.split("_")[0]
        asked.append(st)
        _urls.append(name)
        if st not in catalogue:
            raise IOError("404")
        return _Resp(gzip.compress((_HDR + _rinex(catalogue[st])).encode("ascii")))

    tmp = tempfile.mkdtemp(prefix="hourlytest-")
    import gnss_brdc_supply as ge_sup

    saved = (ge_sup._HOURLY_STATIONS, urllib.request.urlopen, ge_sup.LOG_HOOK)
    logs = []
    try:
        ge_sup.LOG_HOOK = logs.append
        urllib.request.urlopen = fake_urlopen

        # (a) Stations that carry no usable BeiDou must not satisfy the quota. Four
        # BeiDou-blind stations come first, so a stop on station count would fill its quota
        # before reaching any BDS-3.
        ge_sup._HOURLY_STATIONS = [
            "NOBDS1",
            "NOBDS2",
            "NOBDS3",
            "NOBDS4",
            "WIDEAA",
            "WIDEBB",
        ]
        asked[:] = []
        p = ge_sup._fetch_station_hourly(datetime.now(timezone.utc), tmp, "tok")
        body = gzip.open(p, "rt").read()
        check(
            "C19" in body,
            "the merge reaches a station that actually carries BDS-3 (a stop at "
            "len(bodies)>=4 ends one station short of it)",
        )
        check(
            asked[:5] == ["NOBDS1", "NOBDS2", "NOBDS3", "NOBDS4", "WIDEAA"],
            "four BDS-2-only stations do not end the merge -- they answered, they did not "
            "cover",
        )
        # The exit test governs how far back to walk, not whether to fetch at all. The two
        # newest hours are always attempted, because the store accumulates and an
        # unconditional "coverage met -> stop" would never pull new records into it: the store
        # would coast to expiry and then collapse. So WIDEBB may be asked for the newest
        # hours; what must not happen is walking further back once coverage holds.
        # BRUX00BEL_R_20262392000_01H_MN.rnx.gz -> field 2 is YYYYDDDHHMM; the hour is [7:9].
        _hours = {u.split("_")[2][7:9] for u in _urls if len(u.split("_")) > 2}
        check(
            len(_hours) >= 1 and all(h.isdigit() for h in _hours),
            "...and the assertion below is reading real hour fields (%s), not a vacuous set"
            % sorted(_hours),
        )
        check(
            len(_hours) <= 2,
            "coverage being met stops the walk going further back -- at most the two newest "
            "hours were touched (%s)" % sorted(_hours),
        )

        # (b) BDS-2 does not count toward the BeiDou target.
        # A fresh store per case: the store accumulates across calls by design, so a case
        # that reuses the previous case's directory tests the union, not its own premise.
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        ge_sup._HOURLY_STATIONS = ["NOBDS1", "NOBDS2"]
        asked[:] = []
        logs[:] = []
        ge_sup._fetch_station_hourly(datetime.now(timezone.utc), tmp, "tok")
        check(
            any("BELOW TARGET" in m for m in logs),
            "a merge with 18 BDS-2 sats and no BDS-3 reports BELOW TARGET, not silence",
        )
        check(
            any(
                "C" in m.split("BELOW TARGET for ")[1].split(" ")[0]
                for m in logs
                if "BELOW TARGET" in m
            ),
            "and it names C -- C01-C18 broadcast B1I, not B2a; they are not the population",
        )

        # (c) One wide station is still not enough on its own.
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        ge_sup._HOURLY_STATIONS = ["WIDEAA", "WIDEBB"]
        asked[:] = []
        ge_sup._fetch_station_hourly(datetime.now(timezone.utc), tmp, "tok")
        check(
            len(asked) >= 2,
            "even a station that alone meets every target gets a second source "
            "(a truncated hourly reads exactly like a thin sky)",
        )
    finally:
        ge_sup._HOURLY_STATIONS, urllib.request.urlopen, ge_sup.LOG_HOOK = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_station_diversity():
    import gnss_brdc_supply as sup

    print("the station list: geography is the point")
    tail = [s[-3:] for s in sup._HOURLY_STATIONS]
    check(
        len(set(tail)) >= 4, "at least four countries in the list",
    )
    head = [s[-3:] for s in sup._HOURLY_STATIONS[:4]]
    check(
        len(set(head)) >= 3,
        "and the first four -- the ones a good hour actually reaches -- are not one country",
    )


# ---------------------------------------------------------------------------------------
# 3. _brdc_sources: a mirror that can only 404 is not a mirror
# ---------------------------------------------------------------------------------------


def test_brdc_sources():
    import gnss_brdc_supply as sup

    print("_brdc_sources: the CDDIS fallback must ask for a file CDDIS actually holds")
    saved = sup._earthdata_token
    sup._earthdata_token = lambda: "TESTTOKEN"
    try:
        for kind in ("S", "R"):
            srcs = sup._brdc_sources(kind, 2026, 239)
            urls = [u for u, _ in srcs]
            check(
                len(urls) == 2 and "bkg.bund.de" in urls[0],
                "kind %s: BKG is still first and unauthenticated" % kind,
            )
            cd = urls[1]
            check("cddis.nasa.gov" in cd, "kind %s: CDDIS is the fallback" % kind)
            # CDDIS's /daily/YYYY/brdc/ holds only legacy short-name GPS/GLONASS files; its
            # merged dailies live under /daily/YYYY/DDD/YYp/ as BRDC00IGS and BRDM00DLR. Asking
            # CDDIS for BKG's BRDC00WRD product can only 404.
            check(
                "BRDC00WRD" not in cd,
                "kind %s: it does not ask CDDIS for BKG's BRDC00WRD product" % kind,
            )
            check(
                "/2026/239/26p/" in cd,
                "kind %s: and uses the YYYY/DDD/YYp path where merged dailies live"
                % kind,
            )
            check(
                not cd.endswith("/daily/2026/brdc/" + cd.rsplit("/", 1)[-1]),
                "kind %s: never /daily/YYYY/brdc/ -- legacy .YYn/.YYg only" % kind,
            )
        s_url = sup._brdc_sources("S", 2026, 239)[1][0]
        r_url = sup._brdc_sources("R", 2026, 239)[1][0]
        check(
            s_url != r_url,
            "S and R map to different CDDIS products (BRDM00DLR vs BRDC00IGS), not one file",
        )
    finally:
        sup._earthdata_token = saved


def test_404_is_not_a_dead_host():
    import urllib.error
    import gnss_brdc_supply as sup

    print("_src_failed: a 404 is a statement about the PATH, not about the host")
    saved = dict(sup._src_dead)
    try:
        sup._src_dead.clear()
        # The daily fetch asks CDDIS for the current day, which CDDIS never publishes: a
        # guaranteed 404. Blacklisting the host for it would also block the previous day's
        # fetch, which exists and is the fallback while BKG is down.
        url = "https://cddis.nasa.gov/archive/gnss/data/daily/2026/239/26p/x.rnx.gz"
        sup._src_failed(
            url, exc=urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        )
        check(
            not sup._src_skip(
                "https://cddis.nasa.gov/archive/gnss/data/daily/2026/238/"
                "26p/y.rnx.gz"
            ),
            "a 404 on one path leaves every other path on that host reachable",
        )

        sup._src_dead.clear()
        sup._src_failed(url, exc=urllib.error.URLError("timed out"))
        check(
            sup._src_skip(url),
            "a timeout still blacklists the host -- that is the case the "
            "negative cache exists for",
        )

        sup._src_dead.clear()
        sup._src_failed(
            url, exc=urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)
        )
        check(
            sup._src_skip(url),
            "and so does a 5xx -- the server answered, but not with data",
        )

        sup._src_dead.clear()
        sup._src_failed(url)  # the not-gzip case: a 200 serving a login/error page
        check(
            sup._src_skip(url),
            "a 200 that is not gzip blacklists too (login/error page)",
        )

        # The FTP dialect: urllib reports an FTP 550 as a bare URLError with no .code, so the
        # status survives only in the message. The hour walk starts at the current hour, whose
        # directory does not exist yet, so a 550 treated as a host failure would blacklist
        # GSSC on its first call.
        ftp = "ftp://gssc.esa.int/gnss/data/hourly/2026/239/12/BRUX00BEL_R_x_01H_MN.rnx.gz"
        sup._src_dead.clear()
        sup._src_failed(
            ftp,
            exc=urllib.error.URLError(
                "ftp error: 550 CWD command failed: directory not found."
            ),
        )
        check(
            not sup._src_skip(ftp),
            "an FTP 550 (no such directory) is a PATH failure -- the hour is not published "
            "yet, the mirror is fine",
        )

        sup._src_dead.clear()
        sup._src_failed(
            ftp, exc=urllib.error.URLError("ftp error: 421 service not available")
        )
        check(sup._src_skip(ftp), "but an FTP 421 (service unavailable) does blacklist")

        sup._src_dead.clear()
        sup._src_failed(
            ftp, exc=urllib.error.URLError("[Errno 111] Connection refused")
        )
        check(sup._src_skip(ftp), "and so does a refused connection")
    finally:
        sup._src_dead.clear()
        sup._src_dead.update(saved)


def test_skipped_sources_keep_their_marks():
    import urllib.error
    import urllib.request
    import gnss_brdc_supply as sup

    print("_try_refresh_daily: sources that are all cooling down are not dialled")
    dials = []

    def fake_urlopen(req, **kw):
        dials.append(req.full_url)
        raise urllib.error.URLError("timed out")

    saved = (dict(sup._src_dead), sup._earthdata_token, urllib.request.urlopen)
    try:
        # No Earthdata token: BKG is the only daily source. _fetch_brdc_merged has just marked
        # it dead, and the hour walk's GSSC mark is still cooling down.
        sup._earthdata_token = lambda: None
        urllib.request.urlopen = fake_urlopen
        sup._src_dead.clear()
        bkg = sup._brdc_sources("S", 2026, 239)[0][0]
        gssc = "ftp://gssc.esa.int/gnss/data/hourly/2026/239/12/x.rnx.gz"
        sup._src_failed(bkg, exc=urllib.error.URLError("timed out"))
        sup._src_failed(gssc, exc=urllib.error.URLError("timed out"))
        with tempfile.TemporaryDirectory() as d:
            for kind in ("S", "R"):
                sup._try_refresh_daily(kind, 2026, 239, d, None)
        check(
            not dials,
            "neither variant dials the dead BKG host again (%d dial(s))" % len(dials),
        )
        check(sup._src_skip(bkg), "and BKG's mark is still there")
        check(
            sup._src_skip(gssc),
            "and the hourly host's mark survives for the walk that runs next",
        )
    finally:
        sup._src_dead.clear()
        sup._src_dead.update(saved[0])
        sup._earthdata_token, urllib.request.urlopen = saved[1], saved[2]


def test_hourly_mirrors():
    import datetime
    import gnss_brdc_supply as sup

    print("_hourly_sources: the CURRENT day must not be single-homed")
    when = datetime.datetime(2026, 8, 27, 11, tzinfo=datetime.timezone.utc)

    with_tok = sup._hourly_sources("BRUX00BEL", when, "TESTTOKEN")
    hosts = [u.split("/")[2] for u, _ in with_tok]
    check(
        len(set(hosts)) >= 2, "two INDEPENDENT hosts serve each station file, not one"
    )
    check(
        any("Authorization" in h for _, h in with_tok),
        "CDDIS is still there, with its bearer token",
    )

    # No token at all (an expired Earthdata credential), or CDDIS down. While BKG is also
    # down these hourlies are the only source of a current-day ephemeris for E and C, so a
    # token-only path would be a single point of failure for the whole live sky.
    no_tok = sup._hourly_sources("BRUX00BEL", when, None)
    check(
        len(no_tok) >= 1,
        "with no token there is still a source -- the sky does not depend on a credential",
    )
    check(
        all(not h for _, h in no_tok), "and that source needs no authentication at all"
    )
    check(
        all("_01H_MN.rnx.gz" in u for u, _ in with_tok + no_tok),
        "every mirror asks for the same per-station hourly mixed-nav product",
    )
    check(
        all("2026/239/11/" in u for u, _ in with_tok + no_tok),
        "and for the same hour -- mirrors are tried per station, so the freshest hour any "
        "of them has is the hour we get",
    )


def test_hourly_coverage_beats_recency():
    """The hour walk stops on coverage, not on the first hour that answers.

    An IGS station publishes the hour that just closed and the uploads trickle in over
    ~40 min, so a merge run early in the hour sees only part of the network. A thin merge
    must not count as done, and is cached for less time than a full one; a BeiDou merge short
    of target can leave the chains too few probe-eligible satellites to anchor presence.
    """
    import tempfile
    from gnss_brdc_supply import (
        _hourly_target_met,
        _hourly_cov_read,
        _hourly_cov_write,
        _HOURLY_TARGET,
        _HOURLY_TTL_S,
        _HOURLY_THIN_TTL_S,
    )

    print("\nthe hour walk stops on COVERAGE, not on 'somebody answered'")

    full = {k: set(range(v[0], v[0] + v[1])) for k, v in _HOURLY_TARGET.items()}
    check(
        _hourly_target_met(full),
        "a merge meeting every constellation's target is DONE",
    )

    # GPS and Galileo at target, BeiDou short (12 BDS-3 against 20): not done, however many
    # stations answered.
    thin_c = {"G": set(range(1, 26)), "E": set(range(1, 23)), "C": set(range(19, 31))}
    check(
        len(thin_c["C"]) == 12 and not _hourly_target_met(thin_c),
        "a G25/E22/C12 merge is not done -- BeiDou is still short",
    )

    check(
        _HOURLY_THIN_TTL_S < _HOURLY_TTL_S,
        "a below-target merge is cached for less time than a full one (%.0f s vs %.0f s) -- "
        "the missing stations are usually there minutes later"
        % (_HOURLY_THIN_TTL_S, _HOURLY_TTL_S),
    )

    with tempfile.TemporaryDirectory() as d:
        check(
            _hourly_cov_read(d)[0] is True,
            "no coverage record counts as thin -- a file we cannot judge must be "
            "re-merged, not assumed complete",
        )
        _hourly_cov_write(d, False, "G29/E29/C21")
        thin, got = _hourly_cov_read(d)
        check(
            thin is False and got == "G29/E29/C21",
            "a merge that met target records that it did, and what it got",
        )
        _hourly_cov_write(d, True, "G25/E22/C12")
        check(
            _hourly_cov_read(d)[0] is True,
            "and a thin one records that too, so the cache gate can tell them apart -- a "
            "gate that cannot keeps a thin merge as long as a full one",
        )
        # A foreign write must not be inherited. The cache is shared by every process that
        # imports gnss_brdc_supply, and a long-running one holds whatever version of the code
        # it started with; a file newer than its coverage sidecar was written by someone else.
        import time as _t

        _mf = os.path.join(d, "hourly_MN.rnx.gz")
        open(_mf, "wb").close()
        _hourly_cov_write(d, False, "G30/E30/C23")  # our record: file is complete
        check(
            _hourly_cov_read(d)[0] is False, "sidecar written WITH its file is believed"
        )
        os.utime(
            _mf, (_t.time() + 60, _t.time() + 60)
        )  # someone else rewrites the file
        _thin, _why = _hourly_cov_read(d)
        check(
            _thin is True and "FOREIGN WRITE" in _why,
            "a file NEWER than its coverage record is refused and re-merged, not inherited",
        )


def test_coverage_counts_only_usable_records():
    """Coverage counts records fresh enough to predict from, not every PRN present.

    A merge can carry enough distinct PRNs and still predict few satellites if their records
    are past the validity window; counting PRNs counts what is present, and the exit test
    wants what is usable.
    """
    from datetime import datetime, timezone, timedelta
    from gnss_brdc_supply import _rec_is_fresh, _HOURLY_KEEP_S

    print("\ncoverage counts USABLE records, not merely present ones")
    now = datetime(2026, 8, 27, 20, 0, 0, tzinfo=timezone.utc)
    mk = lambda dt: "C19 %s  0.0 0.0 0.0" % (now - dt).strftime("%Y %m %d %H %M %S")
    check(_rec_is_fresh(mk(timedelta(minutes=30)), now), "a 30 min old record counts")
    check(
        not _rec_is_fresh(mk(timedelta(hours=5)), now),
        "a 5 h old record does not -- it is already past predict_all's 4 h window",
    )
    # The cut is the validity window; a tighter one is a bug, not caution. BDS-3 updates
    # hourly on the hour and IGS hourlies publish ~15-25 min late, so BeiDou's freshest record
    # is legitimately up to ~2 h 25 min old, and a 2 h cut would count a complete BeiDou sky
    # as below target.
    check(
        _rec_is_fresh(mk(timedelta(minutes=145)), now),
        "a 2 h 25 min old record COUNTS -- that is BeiDou at the bottom of its hourly "
        "publish cycle, not a stale record",
    )
    check(
        abs(_HOURLY_KEEP_S - 14400.0) < 1.0,
        "the count window is the validity window (%.0f s)" % _HOURLY_KEEP_S,
    )
    # A record dated ahead of now is normal (GPS/Galileo centre toe in the fit interval).
    check(
        _rec_is_fresh(mk(timedelta(minutes=-45)), now),
        "a record stamped 45 min in the FUTURE counts -- toe is centred in the fit interval",
    )
    # Unparseable counts as fresh: the test rejects records that are provably stale, and
    # refusing on doubt would make the merge walk every station and every hour for nothing.
    check(
        _rec_is_fresh("C19 garbage", now) and _rec_is_fresh("", now),
        "an unparseable epoch counts as FRESH -- this rejects the provably stale, it does "
        "not shrink coverage whenever a format surprises us",
    )


def test_cached_brdc_never_writes():
    """One writer: cached_brdc is read-only, and returns a bounded list.

    The nav cache is a shared path any importer can write. A consumer that only wants
    geometry (gnss_observables) must not refetch and rebuild the merged file on its own
    schedule with whatever version of the code it started with. Consumers read; the broker
    writes.
    """
    import tempfile
    import gnss_brdc_supply as sup

    print("\ncached_brdc: read-only, and a bounded list")
    with tempfile.TemporaryDirectory() as d:
        for n, age in (
            ("hourly_MN.rnx.gz", 60.0),
            ("BRDC00WRD_R_20262380000_01D_MN.rnx.gz", 3600.0),
            ("BRDC00WRD_R_20262100000_01D_MN.rnx.gz", 40 * 86400.0),
        ):
            f = os.path.join(d, n)
            open(f, "wb").close()
            t = time.time() - age
            os.utime(f, (t, t))
        before = {n: os.stat(os.path.join(d, n)) for n in os.listdir(d)}
        got = [os.path.basename(x) for x in sup.cached_brdc(d)]
        after = {n: os.stat(os.path.join(d, n)) for n in os.listdir(d)}
        check(
            set(before) == set(after)
            and all(before[n].st_mtime == after[n].st_mtime for n in before),
            "it creates nothing and touches nothing -- the whole point",
        )
        check("hourly_MN.rnx.gz" in got, "the station-hourly merge is always included")
        check(
            "BRDC00WRD_R_20262380000_01D_MN.rnx.gz" in got, "a recent daily is included"
        )
        check(
            "BRDC00WRD_R_20262100000_01D_MN.rnx.gz" not in got,
            "a 40-day-old daily is not -- it cannot carry a record inside the 4 h toe "
            "window",
        )
        check(got and got[0] == "hourly_MN.rnx.gz", "freshest first")
    check(
        sup.cached_brdc("/tmp/definitely-not-a-cache-dir-xyz") == [],
        "a missing cache returns [] so the caller can fall back, not raise",
    )


def test_refresh_cadence_matches_the_product():
    """brdc_predict re-parses at least as often as the station-hourly product updates.

    A broadcast record is valid for 4 h from its toe, so between refreshes the toes age and
    satellites drop out of the prediction one at a time, while the survivors are predicted
    from ever-older records whose model error grows into the trim.
    """
    from gnss_broker import sky

    print("\nthe sky is refreshed as often as its source updates")
    iv = sky._SKY_REFRESH_S
    check(
        iv <= 3600.0,
        "the refresh gate is at most the station-hourly update period (%.0f s <= 3600) -- "
        "asking less often than the product changes guarantees a stale sky between refreshes"
        % iv,
    )
    check(
        iv >= 300.0,
        "...but not so often it re-parses for nothing (%.0f s >= 300; fetch_brdc's own "
        "caches make the call cheap, not free)" % iv,
    )
    # The 4 h validity window is what makes the cadence matter -- if it ever widens, this
    # test's premise changes and someone should notice here.
    from gnss_ephemeris import predict_all
    import inspect

    d = inspect.signature(predict_all).parameters["max_age"].default
    check(
        abs(d - 14400.0) < 1.0,
        "predict_all still assumes a 4 h toe window (%.0f s) -- the refresh cadence is "
        "chosen against it" % d,
    )


def test_current_day_daily_ttl():
    """The current-day daily is re-fetched once it is an hour old.

    With the station-hourly merge down, GLONASS rests on BKG's current-day daily, which
    carries R ~25 min behind, and an R record is cut at GLONASS_MAX_AGE_S (2.5 h). The file's
    age before a re-fetch, plus that lag, plus one sky refresh to notice the age and one to
    read the new file, must stay under the cap, or R satellites drop out before each re-fetch.
    """
    import urllib.request
    import gnss_brdc_supply as sup
    from datetime import datetime, timezone
    from gnss_broker import sky
    from gnss_ephemeris import GLONASS_MAX_AGE_S

    print("\nthe current-day daily is re-fetched hourly (GLONASS's fallback supply)")
    body = gzip.compress(b"a fresh daily")
    dials = []

    class _Resp(object):
        def read(self):
            return body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, **kw):
        dials.append(req.full_url)
        return _Resp()

    now = datetime.now(timezone.utc)
    doy = now.timetuple().tm_yday
    name = "BRDC00WRD_S_%04d%03d0000_01D_MN.rnx.gz" % (now.year, doy)

    def via_variant(d):
        return sup._try_refresh_daily("S", now.year, doy, d, None)

    def via_merged(d):
        return sup._fetch_brdc_merged(now, d)

    saved = (dict(sup._src_dead), sup._earthdata_token, urllib.request.urlopen)
    try:
        sup._earthdata_token = lambda: None
        urllib.request.urlopen = fake_urlopen
        sup._src_dead.clear()
        for age_min, want in ((50, False), (70, True)):
            for who, call in (
                ("_try_refresh_daily", via_variant),
                ("_fetch_brdc_merged", via_merged),
            ):
                with tempfile.TemporaryDirectory() as d:
                    p = os.path.join(d, name)
                    with open(p, "wb") as f:
                        f.write(gzip.compress(b"an old daily"))
                    t = time.time() - age_min * 60.0
                    os.utime(p, (t, t))
                    del dials[:]
                    got = call(d)
                    with open(p, "rb") as f:
                        now_body = f.read()
                    check(
                        got == p and bool(dials) == want and (now_body == body) == want,
                        "%s: a %d min old current-day daily is %s"
                        % (who, age_min, "re-fetched" if want else "reused, no dial"),
                    )
    finally:
        sup._src_dead.clear()
        sup._src_dead.update(saved[0])
        sup._earthdata_token, urllib.request.urlopen = saved[1], saved[2]

    lag = 25 * 60.0
    check(
        sup._DAILY_TTL_S + lag + 2 * sky._SKY_REFRESH_S <= GLONASS_MAX_AGE_S,
        "TTL %.0f s + BKG's ~%.0f s R lag + two %.0f s sky refreshes fits the %.0f s R cap"
        % (sup._DAILY_TTL_S, lag, sky._SKY_REFRESH_S, GLONASS_MAX_AGE_S),
    )


def test_fetch_brdc_off_the_loop():
    """fetch_brdc hands the control loop the cached files and refreshes on a thread.

    The station-hourly merge pulls up to _HOURLY_MAX_FETCH files with a 20 s timeout each.
    The telemetry ring holds ~10 s of windows, so a fetch inside the control pass that takes
    longer loses windows, and fleetadr breaks the carrier arc of every satellite on every
    chain at once, losing the arc level a geometry-free TEC cannot recover. The synchronous
    path remains only for a pinned replay, an empty cache, and block=True.
    """
    import threading
    import gnss_brdc_supply as sup
    from datetime import datetime, timezone

    print("\nfetch_brdc: the loop gets the disk, the network gets a thread")
    now = datetime.now(timezone.utc)
    daily = "BRDC00WRD_S_%04d%03d0000_01D_MN.rnx.gz" % (
        now.year,
        now.timetuple().tm_yday,
    )
    calls = []
    gate = threading.Event()
    # Thread.start() can return before the thread's body runs, so `calls` is empty until the
    # refresh thread reaches slow(); `entered` is what "in flight" means to this test.
    entered = threading.Event()
    real = sup._fetch_brdc_now

    def slow(when=None, cache_dir=sup.CACHE):
        calls.append((threading.current_thread().name, cache_dir))
        entered.set()
        gate.wait(3.0)
        return ["fresh"]

    sup._fetch_brdc_now = slow
    pin = os.environ.pop("GNSS_BRDC_DIR", None)
    try:
        with tempfile.TemporaryDirectory() as d:
            for n in (daily, "hourly_MN.rnx.gz"):
                open(os.path.join(d, n), "wb").close()
            t0 = time.time()
            got = sup.fetch_brdc(cache_dir=d)
            dt = time.time() - t0
            check(dt < 0.5, "returns in %.3f s while the fetch itself is blocked" % dt)
            check(
                sorted(os.path.basename(x) for x in got)
                == sorted([daily, "hourly_MN.rnx.gz"]),
                "...with the cached daily and hourly merge",
            )
            check(entered.wait(3.0), "the refresh thread is in flight")
            sup.fetch_brdc(cache_dir=d)
            check(
                len(calls) == 1,
                "a second call while the refresh is in flight starts no second one (%d)"
                % len(calls),
            )
            gate.set()
            sup._REFRESH["thread"].join(3.0)
            check(
                calls and calls[0][0] == "brdc-refresh" and calls[0][1] == d,
                "the refresh ran on the daemon thread, against the same cache",
            )
            calls.clear()
            got = sup.fetch_brdc(cache_dir=d, block=True)
            check(
                got == ["fresh"]
                and calls
                and calls[0][0] == threading.main_thread().name,
                "block=True (no ephemeris yet) fetches synchronously",
            )
        calls.clear()
        with tempfile.TemporaryDirectory() as e:
            got = sup.fetch_brdc(cache_dir=e)
            check(
                got == ["fresh"]
                and calls
                and calls[0][0] == threading.main_thread().name,
                "an EMPTY cache blocks: there is nothing to hand out",
            )
        calls.clear()
        with tempfile.TemporaryDirectory() as f:
            open(os.path.join(f, "pinned.rnx"), "wb").close()
            os.environ["GNSS_BRDC_DIR"] = f
            got = sup.fetch_brdc(cache_dir=f)
            check(
                got == ["fresh"]
                and calls
                and calls[0][0] == threading.main_thread().name,
                "a pinned replay stays synchronous and hermetic",
            )
            del os.environ["GNSS_BRDC_DIR"]
    finally:
        sup._fetch_brdc_now = real
        if pin is not None:
            os.environ["GNSS_BRDC_DIR"] = pin


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    test_predict_scope()
    test_bridge_speaks()
    test_failed_refresh_backs_off()
    test_bridge_rereads_soon()
    test_one_refresh_per_store()
    test_hourly_coverage()
    test_station_diversity()
    test_brdc_sources()
    test_404_is_not_a_dead_host()
    test_skipped_sources_keep_their_marks()
    test_hourly_mirrors()
    test_hourly_coverage_beats_recency()
    test_coverage_counts_only_usable_records()
    test_cached_brdc_never_writes()
    test_refresh_cadence_matches_the_product()
    test_current_day_daily_ttl()
    test_fetch_brdc_off_the_loop()
    print("\n%s (%d check(s) failed)" % ("FAIL" if _fails else "PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
