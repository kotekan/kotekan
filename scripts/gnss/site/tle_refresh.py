#!/usr/bin/env python3
"""Keep the Celestrak TLE caches fresh. Cron, on the gnss VM.

    41 */6 * * *  /home/kvand/gnss/kotekan/scripts/gnss/site/tle_refresh.py

gps_beamtrack.load_gps_satellites caches each Celestrak group as
~/.cache/kotekan_gps/tle_<GROUP>.txt and downloads it again only when it is missing or the
caller passes max_age_s. The viewer does; the broker's TLE fallbacks (the almanac without
BRDC, the signal-capability gate without the IGS registry) and the offline tools do not, and
the broker must not fetch at start, or a replay would depend on the day it ran. So this job
re-fetches every tle_<GROUP>.txt in the cache older than TLE_MAX_AGE_S. A fetch that fails,
or returns anything that does not parse as whole TLE sets, keeps the stale copy; a good one
replaces it atomically, so a reader sees the old file or the new one.

One line per run to TLE_LOG. Exit 1 when any group failed.

env: TLE_CACHE (default ~/.cache/kotekan_gps), TLE_LOG (default
     /var/tmp/gnss-logs/tle_refresh.log), TLE_MAX_AGE_S (default 259200, 3 days).
"""
import os
import re
import sys
import time
import urllib.request

CACHE = os.environ.get(
    "TLE_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "kotekan_gps")
)
LOG = os.environ.get("TLE_LOG", "/var/tmp/gnss-logs/tle_refresh.log")
MAX_AGE_S = float(os.environ.get("TLE_MAX_AGE_S", "259200"))
# The cache name gps_beamtrack gives a Celestrak URL (GROUP=<group> -> tle_<group>.txt), and the
# URL it was fetched from. tle_default.txt is a URL without a GROUP, which cannot be rebuilt.
NAME_RE = re.compile(r"^tle_([A-Za-z0-9_-]+)\.txt$")
URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=%s&FORMAT=tle"


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def _checksum_ok(line):
    """A TLE line's column 69 is the sum of its digits, with '-' counting 1, mod 10."""
    s = 0
    for c in line[:68]:
        if c.isdigit():
            s += int(c)
        elif c == "-":
            s += 1
    return line[68].isdigit() and s % 10 == int(line[68])


def count_tles(data):
    """Number of satellites in a Celestrak FORMAT=tle body (name, line 1, line 2 per set), or 0
    when it is not whole, checksummed sets: an error page, a rate-limit notice or a cut-off
    body must not replace a good copy."""
    try:
        lines = [ln.rstrip() for ln in data.decode("ascii").splitlines() if ln.strip()]
    except UnicodeDecodeError:
        return 0
    if not lines or len(lines) % 3:
        return 0
    for i in range(0, len(lines), 3):
        l1, l2 = lines[i + 1], lines[i + 2]
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            return 0
        if len(l1) < 69 or len(l2) < 69 or l1[2:7] != l2[2:7]:
            return 0
        if not (_checksum_ok(l1) and _checksum_ok(l2)):
            return 0
    return len(lines) // 3


def refresh(path, group, now):
    """Re-fetch one cache file if it is older than MAX_AGE_S. Returns (ok, note)."""
    age_s = now - os.path.getmtime(path)
    age_d = age_s / 86400.0
    if age_s <= MAX_AGE_S:
        return True, "%s fresh (%.1f d)" % (group, age_d)
    try:
        data = fetch(URL % group)
    except Exception as e:
        return False, "%s FAILED (%s), kept the %.1f d old copy" % (group, e, age_d)
    n = count_tles(data)
    if not n:
        why = "%d bytes, not TLE sets" % len(data)
        return False, "%s FAILED (%s), kept the %.1f d old copy" % (group, why, age_d)
    tmp = "%s.tmp.%d" % (path, os.getpid())
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except OSError as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False, "%s FAILED (%s), kept the %.1f d old copy" % (group, e, age_d)
    return True, "%s refreshed (%d sats, was %.1f d old)" % (group, n, age_d)


def main():
    now = time.time()
    notes, ok = [], True
    try:
        names = sorted(os.listdir(CACHE))
    except OSError as e:
        names, ok = [], False
        notes.append("cache unreadable (%s)" % e)
    for name in names:
        m = NAME_RE.match(name)
        if not m or m.group(1) == "default":
            continue
        good, note = refresh(os.path.join(CACHE, name), m.group(1), now)
        ok = ok and good
        notes.append(note)
    line = "%s tle_refresh: %s" % (
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "; ".join(notes) or "no tle_<GROUP>.txt in %s" % CACHE,
    )
    with open(LOG, "a") as fh:
        fh.write(line + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
