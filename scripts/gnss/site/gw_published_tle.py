#!/usr/bin/env python3
"""TLEs of the published HULIANWANG objects that the TLE group leaves out, for the GuoWang tools.

CelesTrak's TLE format cannot carry the six-digit catalogue numbers given since 2026-07-11, so the
group fetched as TLEs (guowang_passes.fetch_tle, transit_windows.py) holds 177 of the 240 DIGUI
satellites and none catalogued after that date; the tools also read it for DIGUI names only. The
same query in CSV (OMM) format has every object. This job fetches it, at most once per --min-age-s
(CelesTrak serves a query once per 2 h per client), and writes the objects the DIGUI TLE files
lack, the six-digit ones and every non-DIGUI name (HULIANWANG JISHU SHIYAN), as TLEs with Alpha-5
catalogue numbers (100203 -> A0203), which sgp4 and skyfield read. Names are the catalogue's with
spaces made hyphens: 'HULIANWANG DIGUI 233' -> 'HULIANWANG DIGUI-233', 'HULIANWANG JISHU SHIYAN 8A'
-> 'HULIANWANG JISHU-SHIYAN-8A'.

    gw_published_tle.py [--csv FILE] [--cache-dir DIR] [--min-age-s 7200]      gnss venv (sgp4)

Writes, in --cache-dir: gp_hulianwang_<stamp>.csv (the fetch), published_hulianwang_<stamp>.tle (the
TLEs of that fetch, kept for the tools that pick the element set nearest an observation) and
published_hulianwang.tle (a copy of the newest, for the ones that read one file). Never
hulianwang_*.tle: newest_cache would take it for the group. A failed fetch keeps the newest CSV.
Run every 2 h.
"""
import argparse
import glob
import os
import sys
import time
import urllib.request

URL = "https://celestrak.org/NORAD/elements/gp.php?NAME=HULIANWANG&FORMAT=CSV"
CACHE_DIR = os.path.expanduser("~/.cache/guowang")


def write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def fetch(cache_dir, min_age_s):
    """Path of the newest gp_hulianwang_*.csv, fetching a new one when it is older than min_age_s."""
    files = sorted(glob.glob(os.path.join(cache_dir, "gp_hulianwang_*.csv")))
    last = files[-1] if files else None
    if last and time.time() - os.path.getmtime(last) < min_age_s:
        return last
    try:
        with urllib.request.urlopen(URL, timeout=60) as r:
            txt = r.read().decode("utf-8", "replace")
        if not txt.startswith("OBJECT_NAME,") or txt.count("\n") < 100:
            raise RuntimeError("not a GP CSV: %r" % txt[:200])
    except Exception as e:
        if last is None:
            raise
        print("# fetch failed (%s); using %s" % (e, last))
        return last
    path = os.path.join(
        cache_dir, time.strftime("gp_hulianwang_%Y%m%dT%H%M%SZ.csv", time.gmtime())
    )
    write_atomic(path, txt)
    print("# fetched %s" % path)
    return path


def tles(csv_path):
    """[(name, line 1, line 2)] of the objects the DIGUI TLE files lack."""
    from sgp4 import exporter, omm
    from sgp4.api import Satrec

    out = []
    with open(csv_path) as fh:
        for f in omm.parse_csv(fh):
            name = f["OBJECT_NAME"].strip()
            if "DIGUI" in name and int(f["NORAD_CAT_ID"]) < 100000:
                continue
            s = Satrec()
            omm.initialize(s, f)
            l1, l2 = exporter.export_tle(s)
            out.append(
                (
                    "HULIANWANG " + name.replace("HULIANWANG ", "").replace(" ", "-"),
                    l1,
                    l2,
                )
            )
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", help="a GP CSV to use instead of fetching")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--min-age-s", type=float, default=7200.0)
    a = ap.parse_args()
    csv_path = a.csv or fetch(a.cache_dir, a.min_age_s)
    sets = tles(csv_path)
    text = "".join("%-24s\n%s\n%s\n" % s for s in sets)
    stamp = os.path.basename(csv_path).replace("gp_hulianwang_", "").replace(".csv", "")
    hist = os.path.join(a.cache_dir, "published_hulianwang_%s.tle" % stamp)
    write_atomic(hist, text)
    write_atomic(os.path.join(a.cache_dir, "published_hulianwang.tle"), text)
    print(
        "# %s: %d objects (%d DIGUI, %d others) -> %s and published_hulianwang.tle"
        % (
            csv_path,
            len(sets),
            sum("DIGUI" in s[0] for s in sets),
            sum("DIGUI" not in s[0] for s in sets),
            hist,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
