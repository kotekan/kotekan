"""The sky's two silent starvations: a shared prediction state, and a merge that counted sources.

    python3 -m gnss_broker.test_skyscope

WHY THIS EXISTS. On 2026-08-27 KV asked why BeiDou had zero noise probes. The answer was a
four-link chain, and the digest gate could not see ANY of it -- replays pin the sky through
GNSS_BRDC_DIR, so neither the fetch path nor the collapse bridge is ever exercised. Both
defects therefore need a test that can actually go red:

  1. THE PREDICTION STATE IS SHARED AND THE COLLAPSE BOOKKEEPING WAS NOT SCOPED.
     receiver.brdc() hands all five chains ONE dict so a single parse serves G/E/C -- right,
     and deliberate. But brdc_predict stored `peak_n` and `last_good` in it at top level:
       * peak_n became a MAX OVER CONSTELLATIONS. BeiDou's ~13 in-window sats were judged
         against GPS's ~24, so `len(out) < 0.5*peak` was permanently true -- bds_b2a fired
         PREDICTION COLLAPSE 9 ms after its own almanac init, against a peak it never set.
       * last_good became WHICHEVER CHAIN RAN LAST, so BeiDou's "bridge" served Galileo's and
         GPS's satellites, with their elevations. In the live log: `bds_b2a: noise probe PRN 1
         seeded (elev -88)` and `bds_b2b: noise probe PRN 3 seeded (elev -32)` on chains whose
         min_prn is 19, 1.5 s after the Galileo chains seeded the same PRN 3.
     The probe selector then chose foreign PRNs; the two that happened to land in slots 19-42
     survived --probe-require-slot, and bds_b2a sat at 2 probes -> UNANCHORED -> nothing
     admitted, nothing trimmed.

  2. THE HOURLY MERGE COUNTED STATIONS, NOT COVERAGE. `len(bodies) >= 4` stopped at the first
     four sources that ANSWERED. Two of the four (NRC1, STJO) carry no BeiDou at all and still
     consumed quota, so BRUX -- sixth in a list whose first four were all Canadian, and worth
     15 in-slot BDS on its own -- was never reached.

Both are the audited fallback shape (docs/CHORD_PEER_RELATIVE_AUDIT.md): a path that
reproduces the primary's OUTPUT while dropping one of its invariants, running exactly when
things are already degraded.

@author Keith Vanderlinde
"""

import gzip
import os
import shutil
import sys
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

    # A collapse forces eph_t back so the NEXT call re-fetches; the stub has to answer that
    # or the test measures its own AttributeError instead of the bridge.
    def fetch_brdc(self, when=None):
        self.fetches += 1
        return "stub"

    def parse_rinex_nav(self, path):
        return {"x": 1}


def _sats(sysc, prns, el=10.0):
    return {(sysc, p): dict(az=0.0, el=el, range_m=2.0e7, range_rate_mps=0.0,
                            sat_clk_s=0.0, toe_age_s=0.0) for p in prns}


def _predict(state, ge, sysc, min_prn, prns, el=10.0):
    from gnss_broker.sky import brdc_predict
    from datetime import datetime, timezone
    ge.visible = _sats(sysc, prns, el)
    return brdc_predict(state, 49.32, -119.62, 545.0, sysc, min_prn,
                        datetime.now(timezone.utc), 1176.45e6)


def test_predict_scope():
    from gnss_broker import sky
    print("brdc_predict: the collapse bookkeeping is PER CONSTELLATION")
    ge = _FakeEphMod()
    # eph_t fresh so no refetch is attempted; `mod` is the stub.
    state = {"mod": ge, "eph": {"x": 1}, "eph_t": sky._now()}

    # GPS goes first and sets a wide peak -- this is the running order in the live broker
    # (gps_l5 initialises ~1 s before bds_b2a).
    g = _predict(state, ge, "G", 1, range(1, 25))
    check(len(g) == 24, "GPS predicts its 24 satellites")

    # BeiDou's FIRST prediction, 8 satellites. Against GPS's peak this is a "collapse";
    # against its own (nothing yet) it is simply the sky.
    c = _predict(state, ge, "C", 19, range(19, 27))
    check(len(c) == 8, "BeiDou's first prediction returns ITS OWN 8 satellites")
    check(all(p >= 19 for p in c),
          "and NOT ONE PRN below its min_prn -- the constellation identity survives")
    check(not any(p < 19 for p in c),
          "specifically: no PRN 1/2/5/14, the GPS and Galileo probes seen leaking on sky")

    # The two peaks are independent.
    sc = state.get("scope", {})
    check(sc.get(("G", 1), {}).get("peak_n") == 24
          and sc.get(("C", 19), {}).get("peak_n") == 8,
          "each constellation carries its OWN peak (G 24, C 8), not one shared max")

    # A REAL BeiDou collapse: it had 24, now has 8.
    ge2 = _FakeEphMod()
    st2 = {"mod": ge2, "eph": {"x": 1}, "eph_t": sky._now()}
    _predict(st2, ge2, "C", 19, range(19, 43))          # 24 BDS-3, sets the peak
    _predict(st2, ge2, "G", 1, range(1, 31))            # GPS runs in between, as it does
    c2 = _predict(st2, ge2, "C", 19, range(19, 27))     # 8 -- a genuine collapse
    check(len(c2) == 24, "a REAL collapse still bridges (24 sats returned for 8 predicted)")
    check(all(p >= 19 for p in c2),
          "and it bridges on BEIDOU's last good set, never on the GPS set that ran between")

    # Recovery clears the bridge.
    c3 = _predict(st2, ge2, "C", 19, range(19, 43))
    check(len(c3) == 24
          and "bridge_since" not in st2.get("scope", {}).get(("C", 19), {}),
          "recovery comes off the bridge and forgets it")


def test_bridge_speaks():
    from gnss_broker import sky
    print("brdc_predict: a bridged sky says so EVERY time it is served, not once per 600 s")
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
        # ⚠️ The OLD code logged inside the 600 s refetch guard, so cycles 2..N were silent
        # while every one of them returned a frozen elevation set. bds_b2a bridged for 25
        # minutes on one line of log.
        bridged = [m for m in said if "COLLAPSE" in m]
        check(len(bridged) == 3, "every bridged cycle announces (3 of 3), not just the first")
        check(all("BRIDGING" in m for m in bridged),
              "and it says BRIDGING -- the old line named the refetch, not the frozen sky")
        check(all("STALE" in m for m in bridged),
              "and warns the elevations are stale -- probes and drop gates ride them")
        said[:] = []
        _predict(state, ge, "C", 19, range(19, 43))
        check(any("RECOVERED" in m for m in said), "recovery is announced too")
    finally:
        sky._log_rl, sky._log = orig_rl, orig_log


# ---------------------------------------------------------------------------------------
# 2. _fetch_station_hourly: stop on COVERAGE, not on a station count
# ---------------------------------------------------------------------------------------

_HDR = ("     3.05           NAVIGATION DATA     M                   RINEX VERSION / TYPE\n"
        "                                                            END OF HEADER\n")


def _rinex(prns):
    """A minimal RINEX-3 nav body carrying one record per (sys, prn)."""
    out = []
    for sysc, ps in prns.items():
        for p in ps:
            out.append("%s%02d 2026 08 27 10 00 00 0.0e+00 0.0e+00 0.0e+00\n" % (sysc, p))
            out.append("     0.0e+00 0.0e+00 0.0e+00 0.0e+00\n")
    return "".join(out)


def test_hourly_coverage():
    import gnss_ephemeris as ge
    print("_fetch_station_hourly: coverage is the quota, not the number of sources")
    from datetime import datetime, timezone
    import urllib.request

    # Station catalogue for this test. NOBDS answers wide on G/E and carries only BDS-2 --
    # exactly NRC1/STJO's live behaviour, and exactly what used to consume quota.
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

    def fake_urlopen(req, **kw):
        name = req.full_url.rsplit("/", 1)[-1]
        st = name.split("_")[0]
        asked.append(st)
        if st not in catalogue:
            raise IOError("404")
        return _Resp(gzip.compress((_HDR + _rinex(catalogue[st])).encode("ascii")))

    tmp = tempfile.mkdtemp(prefix="hourlytest-")
    saved = (ge._HOURLY_STATIONS, urllib.request.urlopen, ge.LOG_HOOK)
    logs = []
    try:
        ge.LOG_HOOK = logs.append
        urllib.request.urlopen = fake_urlopen

        # (a) The stations that carry no usable BeiDou must NOT satisfy the quota.
        # FOUR BeiDou-blind stations first: that is the live arrangement, and it is exactly
        # what `len(bodies) >= 4` got wrong -- it filled its quota before reaching any BDS-3.
        ge._HOURLY_STATIONS = ["NOBDS1", "NOBDS2", "NOBDS3", "NOBDS4", "WIDEAA", "WIDEBB"]
        asked[:] = []
        p = ge._fetch_station_hourly(datetime.now(timezone.utc), tmp, "tok")
        body = gzip.open(p, "rt").read()
        check("C19" in body,
              "the merge reaches a station that actually carries BDS-3 (len(bodies)>=4 "
              "stopped one station short of it, every hour, for as long as it existed)")
        check(asked[:5] == ["NOBDS1", "NOBDS2", "NOBDS3", "NOBDS4", "WIDEAA"],
              "four BDS-2-only stations do NOT end the merge -- they answered, they did not "
              "cover")
        check("WIDEBB" not in asked,
              "and it stops as soon as coverage IS met -- no pointless sixth fetch")

        # (b) BDS-2 does not count toward the BeiDou target.
        os.utime(p, (0, 0))
        ge._HOURLY_STATIONS = ["NOBDS1", "NOBDS2"]
        asked[:] = []
        logs[:] = []
        ge._fetch_station_hourly(datetime.now(timezone.utc), tmp, "tok")
        check(any("BELOW TARGET" in m for m in logs),
              "a merge with 18 BDS-2 sats and no BDS-3 reports BELOW TARGET, not silence")
        check(any("C" in m.split("BELOW TARGET for ")[1].split(" ")[0]
                  for m in logs if "BELOW TARGET" in m),
              "and it names C -- C01-C18 broadcast B1I, not B2a; they are not the population")

        # (c) One wide station is still not enough on its own.
        os.utime(p, (0, 0))
        ge._HOURLY_STATIONS = ["WIDEAA", "WIDEBB"]
        asked[:] = []
        ge._fetch_station_hourly(datetime.now(timezone.utc), tmp, "tok")
        check(len(asked) >= 2,
              "even a station that alone meets every target gets a second source "
              "(a truncated hourly reads exactly like a thin sky)")
    finally:
        ge._HOURLY_STATIONS, urllib.request.urlopen, ge.LOG_HOOK = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_station_diversity():
    import gnss_ephemeris as ge
    print("the station list: geography is the point")
    tail = [s[-3:] for s in ge._HOURLY_STATIONS]
    check(len(set(tail)) >= 4,
          "at least four countries in the list (it was CAN,CAN,CAN,CAN,USA,BEL)")
    head = [s[-3:] for s in ge._HOURLY_STATIONS[:4]]
    check(len(set(head)) >= 3,
          "and the first four -- the ones a good hour actually reaches -- are not one country")


# ---------------------------------------------------------------------------------------
# 3. _brdc_sources: a mirror that can only 404 is not a mirror
# ---------------------------------------------------------------------------------------

def test_brdc_sources():
    import gnss_ephemeris as ge
    print("_brdc_sources: the CDDIS fallback must ask for a file CDDIS actually holds")
    saved = ge._earthdata_token
    ge._earthdata_token = lambda: "TESTTOKEN"
    try:
        for kind in ("S", "R"):
            srcs = ge._brdc_sources(kind, 2026, 239)
            urls = [u for u, _ in srcs]
            check(len(urls) == 2 and "bkg.bund.de" in urls[0],
                  "kind %s: BKG is still first and unauthenticated" % kind)
            cd = urls[1]
            check("cddis.nasa.gov" in cd, "kind %s: CDDIS is the fallback" % kind)
            # ⚠️ THE BUG: it asked CDDIS for BKG's product under /daily/YYYY/brdc/, a directory
            # that holds ONLY legacy short-name GPS/GLONASS files. Every call 404'd, silently,
            # from the day it was added (2026-07-21) to the day BKG next went down.
            check("BRDC00WRD" not in cd,
                  "kind %s: it does NOT ask CDDIS for BKG's BRDC00WRD product" % kind)
            check("/2026/239/26p/" in cd,
                  "kind %s: and uses the YYYY/DDD/YYp path where merged dailies live" % kind)
            check(not cd.endswith("/daily/2026/brdc/" + cd.rsplit("/", 1)[-1]),
                  "kind %s: never /daily/YYYY/brdc/ -- legacy .YYn/.YYg only" % kind)
        s_url = ge._brdc_sources("S", 2026, 239)[1][0]
        r_url = ge._brdc_sources("R", 2026, 239)[1][0]
        check(s_url != r_url,
              "S and R map to DIFFERENT CDDIS products (BRDM00DLR vs BRDC00IGS), not one file")
    finally:
        ge._earthdata_token = saved


def test_404_is_not_a_dead_host():
    import urllib.error
    import gnss_ephemeris as ge
    print("_src_failed: a 404 is a statement about the PATH, not about the host")
    saved = dict(ge._src_dead)
    try:
        ge._src_dead.clear()
        # The daily fetch asks CDDIS for the CURRENT day, which CDDIS never publishes -- a
        # guaranteed 404. Under the old rule that disabled cddis.nasa.gov for 300 s, taking
        # the YESTERDAY fetch (which exists, and is the whole fallback while BKG is down)
        # down with it.
        url = "https://cddis.nasa.gov/archive/gnss/data/daily/2026/239/26p/x.rnx.gz"
        ge._src_failed(url, exc=urllib.error.HTTPError(url, 404, "Not Found", {}, None))
        check(not ge._src_skip("https://cddis.nasa.gov/archive/gnss/data/daily/2026/238/"
                               "26p/y.rnx.gz"),
              "a 404 on one path leaves every OTHER path on that host reachable")

        ge._src_dead.clear()
        ge._src_failed(url, exc=urllib.error.URLError("timed out"))
        check(ge._src_skip(url), "a TIMEOUT still blacklists the host -- that is the case the "
                                "negative cache exists for (BKG, twice now)")

        ge._src_dead.clear()
        ge._src_failed(url, exc=urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None))
        check(ge._src_skip(url), "and so does a 5xx -- the server answered, but not with data")

        ge._src_dead.clear()
        ge._src_failed(url)   # the not-gzip case: a 200 serving a login/error page
        check(ge._src_skip(url), "a 200 that is not gzip blacklists too (login/error page)")

        # ⚠️ THE FTP DIALECT. urllib reports an FTP 550 as a bare URLError with NO .code, so
        # the HTTP-only test waved it through to the blacklist. The hour-walk starts at the
        # CURRENT hour, whose directory does not exist yet -- so GSSC blacklisted itself on
        # its very first call and the whole 12-station x 4-hour walk collapsed to one attempt.
        ftp = "ftp://gssc.esa.int/gnss/data/hourly/2026/239/12/BRUX00BEL_R_x_01H_MN.rnx.gz"
        ge._src_dead.clear()
        ge._src_failed(ftp, exc=urllib.error.URLError(
            "ftp error: 550 CWD command failed: directory not found."))
        check(not ge._src_skip(ftp),
              "an FTP 550 (no such directory) is a PATH failure -- the hour is not published "
              "yet, the mirror is fine")

        ge._src_dead.clear()
        ge._src_failed(ftp, exc=urllib.error.URLError("ftp error: 421 service not available"))
        check(ge._src_skip(ftp), "but an FTP 421 (service unavailable) does blacklist")

        ge._src_dead.clear()
        ge._src_failed(ftp, exc=urllib.error.URLError("[Errno 111] Connection refused"))
        check(ge._src_skip(ftp), "and so does a refused connection")
    finally:
        ge._src_dead.clear()
        ge._src_dead.update(saved)


def test_hourly_mirrors():
    import datetime
    import gnss_ephemeris as ge
    print("_hourly_sources: the CURRENT day must not be single-homed")
    when = datetime.datetime(2026, 8, 27, 11, tzinfo=datetime.timezone.utc)

    with_tok = ge._hourly_sources("BRUX00BEL", when, "TESTTOKEN")
    hosts = [u.split("/")[2] for u, _ in with_tok]
    check(len(set(hosts)) >= 2,
          "two INDEPENDENT hosts serve each station file, not one")
    check(any("Authorization" in h for _, h in with_tok),
          "CDDIS is still there, with its bearer token")

    # ⚠️ THE CASE THIS EXISTS FOR: no token at all (expired Earthdata credential), or CDDIS
    # down. On any day BKG is also down -- as on 2026-08-27 -- these hourlies are the ONLY
    # source of a current-day ephemeris for E and C, so a token-only path is a single point
    # of failure for the whole live sky.
    no_tok = ge._hourly_sources("BRUX00BEL", when, None)
    check(len(no_tok) >= 1,
          "with NO token there is still a source -- the sky does not depend on a credential")
    check(all(not h for _, h in no_tok),
          "and that source needs no authentication at all")
    check(all("_01H_MN.rnx.gz" in u for u, _ in with_tok + no_tok),
          "every mirror asks for the same per-station hourly mixed-nav product")
    check(all("2026/239/11/" in u for u, _ in with_tok + no_tok),
          "and for the SAME hour -- mirrors are tried per station, so the freshest hour any "
          "of them has is the hour we get")


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    test_predict_scope()
    test_bridge_speaks()
    test_hourly_coverage()
    test_station_diversity()
    test_brdc_sources()
    test_404_is_not_a_dead_host()
    test_hourly_mirrors()
    print("\n%s (%d check(s) failed)" % ("FAIL" if _fails else "PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
