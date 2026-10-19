#!/usr/bin/env python3
"""tle_refresh.py's own check: a stale TLE cache is replaced by a good fetch and kept on any
failure.

    python3 scripts/gnss/site/test_tle_refresh.py

Nothing reaches Celestrak: fetch is replaced and the cache and log point at a throwaway
directory.
"""
import os
import re
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tle_refresh  # noqa: E402

_fails = []

# Two GPS sets as Celestrak serves them (CRLF line ends, padded names).
GOOD = (
    "GPS BIIR-5  (PRN 22)    \r\n"
    "1 26407U 00040A   26292.18277200 -.00000020  00000+0  00000+0 0  9992\r\n"
    "2 26407  54.8306 210.2424 0116366 304.5830 228.2183  2.00558714192435\r\n"
    "GPS BIIR-8  (PRN 16)    \r\n"
    "1 27663U 03005A   26291.77121942 -.00000019  00000+0  00000+0 0  9994\r\n"
    "2 27663  54.8601 210.0640 0152075  54.0691 317.1644  2.00553929173796\r\n"
).encode("ascii")
OLD = b"the stale copy\n"
DAY = 86400.0


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


class Run(object):
    """One main() against a cache holding `files` ({name: age in days}) with fetch replaced
    by `answer` (bytes, or an exception to raise)."""

    def __init__(self, files, answer):
        self.d = tempfile.mkdtemp(prefix="tle_refresh_")
        self.urls = []
        for name, age_d in files.items():
            p = os.path.join(self.d, name)
            with open(p, "wb") as fh:
                fh.write(OLD)
            t = time.time() - age_d * DAY
            os.utime(p, (t, t))
        tle_refresh.CACHE = self.d
        tle_refresh.LOG = os.path.join(self.d, "log")
        tle_refresh.MAX_AGE_S = 3 * DAY
        real = tle_refresh.fetch

        def fetch(url):
            self.urls.append(url)
            if isinstance(answer, Exception):
                raise answer
            return answer

        tle_refresh.fetch = fetch
        try:
            self.rc = tle_refresh.main()
        finally:
            tle_refresh.fetch = real
        with open(tle_refresh.LOG) as fh:
            self.log = fh.read().splitlines()

    def body(self, name):
        with open(os.path.join(self.d, name), "rb") as fh:
            return fh.read()

    def age_d(self, name):
        return (time.time() - os.path.getmtime(os.path.join(self.d, name))) / DAY

    def leftovers(self):
        return [n for n in os.listdir(self.d) if ".tmp." in n]

    def close(self):
        shutil.rmtree(self.d, ignore_errors=True)


def test_good_fetch_replaces_a_stale_copy():
    r = Run({"tle_gps-ops.txt": 5.0, "tle_galileo.txt": 1.0}, GOOD)
    try:
        check(
            r.urls == [tle_refresh.URL % "gps-ops"],
            "only the copy past 3 days is fetched, from its group's URL (%r)" % r.urls,
        )
        check(
            r.body("tle_gps-ops.txt") == GOOD and r.age_d("tle_gps-ops.txt") < 0.01,
            "the stale copy is replaced and its age restarts",
        )
        check(
            r.body("tle_galileo.txt") == OLD and r.age_d("tle_galileo.txt") > 0.99,
            "the 1-day copy is left alone",
        )
        check(
            r.rc == 0 and len(r.log) == 1, "exit 0 and one log line (%r)" % r.log,
        )
        check(
            "gps-ops refreshed (2 sats" in r.log[0] and "galileo fresh" in r.log[0],
            "the line names each group and what happened to it",
        )
        check(not r.leftovers(), "no temp file is left beside the cache")
    finally:
        r.close()


def test_failures_keep_the_stale_copy():
    cut = GOOD[: GOOD.index(b"2 27663")]
    torn = GOOD.replace(b"0  9994", b"0  9995")
    sets = GOOD.split(b"\r\n")
    crossed = b"\r\n".join([sets[i] for i in (0, 1, 5, 3, 4, 2)]) + b"\r\n"
    for answer, what in (
        (OSError("HTTP Error 403: Forbidden"), "a failed fetch"),
        (
            b"<html><body>No GP data found</body></html>\n",
            "an error page served as 200",
        ),
        (b"", "an empty body"),
        (cut, "a body cut off inside a set"),
        (torn, "a set whose checksum does not match"),
        (crossed, "a set whose line 2 is another satellite's"),
    ):
        r = Run({"tle_glo-ops.txt": 5.0}, answer)
        try:
            check(
                r.urls == [tle_refresh.URL % "glo-ops"]
                and r.body("tle_glo-ops.txt") == OLD
                and r.age_d("tle_glo-ops.txt") > 4.99,
                "%s keeps the stale copy, unchanged and still stale" % what,
            )
            check(
                r.rc == 1
                and len(r.log) == 1
                and "glo-ops FAILED" in r.log[0]
                and "kept the 5.0 d old copy" in r.log[0],
                "and says so in its one line, with exit 1 (%r)" % r.log,
            )
            check(not r.leftovers(), "and leaves no temp file")
        finally:
            r.close()


def test_failed_replace_keeps_the_stale_copy():
    real = tle_refresh.os.replace

    def refuse(src, dst):
        raise OSError("disk full")

    tle_refresh.os.replace = refuse
    try:
        r = Run({"tle_beidou.txt": 5.0}, GOOD)
    finally:
        tle_refresh.os.replace = real
    try:
        check(
            r.body("tle_beidou.txt") == OLD and r.rc == 1 and "FAILED" in r.log[0],
            "a write that cannot land keeps the stale copy (%r)" % r.log,
        )
        check(not r.leftovers(), "and removes its temp file")
    finally:
        r.close()


def test_only_rebuildable_cache_names():
    r = Run(
        {
            "tle_default.txt": 9.0,
            "tle_gps-ops.txt.download": 9.0,
            "tle_gps-ops.txt.tmp.123": 9.0,
            "igs_satellite_metadata.snx": 90.0,
        },
        GOOD,
    )
    try:
        check(
            not r.urls,
            "a GROUP-less copy, a download in progress, a temp file and the SINEX are "
            "not fetched (%r)" % r.urls,
        )
        check(
            r.rc == 0 and len(r.log) == 1 and "no tle_<GROUP>.txt" in r.log[0],
            "and the run still logs its one line (%r)" % r.log,
        )
    finally:
        r.close()


def test_urls_match_the_readers():
    """Every Celestrak URL the readers load is cached under a name this job maps back to the
    same URL (gps_beamtrack.load_gps_satellites names the file after GROUP=)."""
    scripts = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "python", "scripts"
    )
    urls = set()
    for rel in ("gnss/gps_beamtrack.py", "js_viewer/livebeam_server.py"):
        with open(os.path.join(scripts, rel)) as fh:
            urls.update(re.findall(r'"(https://celestrak\.org/[^"]+)"', fh.read()))
    check(len(urls) >= 4, "the readers name at least four groups (%d)" % len(urls))
    for url in sorted(urls):
        group = re.search(r"GROUP=([A-Za-z0-9_-]+)", url).group(1)
        m = tle_refresh.NAME_RE.match("tle_%s.txt" % group)
        check(
            m is not None and tle_refresh.URL % m.group(1) == url,
            "%s is refreshed from the URL it was loaded from" % group,
        )


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        try:
            t()
        except Exception as e:
            check(False, "%s raised %s: %s" % (t.__name__, type(e).__name__, e))
    if _fails:
        print("\n%d FAILED" % len(_fails))
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
