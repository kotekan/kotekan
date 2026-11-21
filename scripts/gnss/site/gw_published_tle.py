#!/usr/bin/env python3
"""Fetch CelesTrak's OMM elements of every HULIANWANG object into the cache the GuoWang tools read.

CelesTrak's TLE format cannot carry the six-digit catalogue numbers given since 2026-07-11, so the
GuoWang tools read OMM (GP) records instead: guowang_passes.OMM_URL, every DIGUI and JISHU SHIYAN
object as CSV. CelesTrak serves a query once per 2 h per client, so this job is the one fetch:
gw_ephem_table.py and beam_dump_sched.py read the newest cache (guowang_passes.newest_cache) and
fetch only when run with --fetch. The name is the job's older one, kept for its cron line; it
writes no TLEs now.

    gw_published_tle.py [--cache-dir DIR] [--min-age-s 7200]      gnss venv (sgp4)

Writes --cache-dir/gp_hulianwang_<stamp>.csv when the newest is older than --min-age-s. A failed
fetch keeps the newest. Run every 2 h.
"""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(
    0, os.path.join(HERE, "..", "..", "..", "python", "scripts", "gnss", "guowang")
)

import guowang_passes as gp  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache-dir", default=gp.CACHE_DIR)
    ap.add_argument("--min-age-s", type=float, default=7200.0)
    a = ap.parse_args()
    path = gp.fetch_omm(a.cache_dir, a.min_age_s)
    from sgp4 import omm

    with open(path) as fh:
        names = [gp.omm_name(f["OBJECT_NAME"]) for f in omm.parse_csv(fh)]
    kinds = collections.Counter(
        "DIGUI" if "DIGUI" in n else "JISHU-SHIYAN" if "JISHU" in n else "other"
        for n in names
    )
    print(
        "# %s: %d objects (%s)"
        % (path, len(names), ", ".join("%d %s" % (v, k) for k, v in sorted(kinds.items())))
    )


if __name__ == "__main__":
    sys.exit(main())
