"""The ephemeris SUPPLY has an output gate now.

    python3 gnss_broker/test_brdc_golden.py                  # check against the golden digest
    python3 gnss_broker/test_brdc_golden.py --print-digest    # after a DELIBERATE change

WHY THIS EXISTS. broker_equiv replays pin the sky through GNSS_BRDC_DIR, so the fetch/merge path
NEVER EXECUTES in any of the seven digest fixtures -- all seven stayed EQUIVALENT across six
supply defects fixed in one evening (2026-08-27: sources-not-coverage, recency-not-coverage,
presence-not-usability, a CDDIS mirror that could only 404, a 404 blacklisting a live host, and
a merge that rebuilt instead of accumulating). Every one changed the CONTENT of the merged file
and no gate could see it.

test_skyscope.py covers the POLICY decisions -- does the walk stop on the right thing, does a
thin merge get the short TTL. This covers the OUTPUT: real captured station files, a pinned
clock, the whole merge, one hash.

⚠️ THE GOLDEN LIVES HERE, IN GIT, UNDER REVIEW -- not in the fixture manifest. A golden
regenerated alongside its fixture can never go red, which is the defect this file exists to
catch, not to reproduce.

⚠️ HASH THE DECOMPRESSED BYTES. gzip embeds an mtime, so the .gz is not reproducible even when
its content is.

FIXTURE: $GNSS_FIXTURES/brdc_merge_golden, from scripts/gnss/brdc_fixture_capture.py.
A MISSING FIXTURE IS A FAILURE, NOT A SKIP -- "the gate did not run" and "the gate passed" must
never look alike (gate.sh's own rule, and the fleetdll lesson behind it).
"""
import gzip
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

FIXTURE = os.path.join(os.environ.get("GNSS_FIXTURES", "/home/kvand/gnss/fixtures"),
                       "brdc_merge_golden")

# The merged file's decompressed SHA-256, for the pinned clock in the fixture manifest.
GOLDEN = "35222025baf6ae15b3721f5019ab56c72a4e6c49a259c551db66f9b8daf8cfdd"

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def _serve(names):
    """A urlopen that serves exactly `names` from the fixture and 404s everything else."""
    class _R(object):
        def __init__(self, d):
            self.d = d

        def read(self):
            return self.d

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(req, **kw):
        n = req.full_url.rsplit("/", 1)[-1]
        p = os.path.join(FIXTURE, n)
        if n in names and os.path.exists(p):
            with open(p, "rb") as f:
                return _R(f.read())
        # ⚠️ 404, NOT a bare exception. A 404 is a statement about the PATH and must not
        # blacklist the host -- otherwise the second hour of the walk is served by a mirror
        # this test itself killed, and the golden would encode that accident.
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)
    return fake


def _merge(ge, when, names, cache_dir):
    saved = urllib.request.urlopen
    try:
        urllib.request.urlopen = _serve(names)
        return ge._fetch_station_hourly(when, cache_dir, "tok")
    finally:
        urllib.request.urlopen = saved


def _digest(path):
    return hashlib.sha256(gzip.decompress(open(path, "rb").read())).hexdigest()


def main():
    import gnss_ephemeris as ge
    ge.LOG_HOOK = lambda m: None

    man = os.path.join(FIXTURE, "manifest.json")
    if not os.path.exists(man):
        print("FAIL: no fixture at %s -- run scripts/gnss/brdc_fixture_capture.py.\n"
              "      This is a FAILURE, not a skip: a gate that did not run must never look "
              "like a gate that passed." % FIXTURE)
        return 1
    with open(man) as f:
        m = json.load(f)
    when = datetime.strptime(m["when"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    allf = [x["name"] for x in m["files"]]
    hours = sorted({x["hour"] for x in m["files"]})
    newest = [x["name"] for x in m["files"] if x["hour"] == hours[-1]]
    print("brdc merge: %d fixture file(s), hours %s, when %s\n" % (len(allf), hours, m["when"]))

    d = tempfile.mkdtemp(prefix="brdcgold-")
    try:
        p = _merge(ge, when, allf, d)
        if p is None:
            print("FAIL: the merge returned None on a fixture of %d files" % len(allf))
            return 1
        got = _digest(p)
        if "--print-digest" in sys.argv:
            print('GOLDEN = "%s"' % got)
            return 0
        check(got == GOLDEN,
              "the merged file is byte-identical to the golden\n"
              "          got    %s\n          golden %s" % (got, GOLDEN))

        # IDEMPOTENT. Re-merging the same inputs into a store that already holds them must not
        # change the file -- records are keyed (sys, prn, toc) precisely so a re-fetch is free.
        os.utime(os.path.join(d, "hourly_MN.rnx.gz"), (0, 0))   # expire the cache gate only
        p2 = _merge(ge, when, allf, d)
        check(_digest(p2) == got, "re-merging the same inputs changes nothing (idempotent)")

        # ORDER INDEPENDENT. Accumulating hour-by-hour must land on the same file as taking
        # both at once. Otherwise the store's content depends on fetch HISTORY, and a
        # constellation's coverage would silently depend on when the broker happened to start.
        d2 = tempfile.mkdtemp(prefix="brdcgold2-")
        try:
            _merge(ge, when, newest, d2)
            os.utime(os.path.join(d2, "hourly_MN.rnx.gz"), (0, 0))
            p3 = _merge(ge, when, allf, d2)
            check(_digest(p3) == got,
                  "accumulating newest-hour-then-all equals all-at-once (order independent)")
        finally:
            shutil.rmtree(d2, ignore_errors=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)

    print("\n%s (%d check(s) failed)" % ("FAIL" if _fails else "PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
