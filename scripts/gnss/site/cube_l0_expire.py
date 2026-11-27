#!/usr/bin/env python3
"""Expire the beam cube's L0 days: delete <ROOT>/l0/<pointing>/<sender>/<YYYYMMDD>[_*].h5 older than
--days, where rung12 and rung60 of the same file exist. A day without both rungs stays and is listed.

    cube_l0_expire.py ROOT [--days 30] [--delete]        (a dry run without --delete)

Run daily from cron on the host that runs the staging loop (KV 2026-11-27: L0 kept 30 days, rungs kept).
The cube tree on cs00 is snapshotted, so the space returns only when the snapshots holding a file expire.
"""
import argparse
import datetime
import os
import re

DAY = re.compile(r"^(\d{8})(_.*)?\.h5$")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--delete", action="store_true")
    a = ap.parse_args()
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = (now - datetime.timedelta(days=a.days)).strftime("%Y%m%d")
    l0 = os.path.join(a.root, "l0")
    gone, gb_gone, norung = 0, 0.0, []
    for dirpath, _, files in os.walk(l0):
        for f in files:
            m = DAY.match(f)
            if not m or m.group(1) >= cutoff:
                continue
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, l0)
            if all(os.path.exists(os.path.join(a.root, r, rel)) for r in ("rung12", "rung60")):
                gone += 1
                gb_gone += os.path.getsize(path) / 1e9
                if a.delete:
                    os.remove(path)
            else:
                norung.append(rel)
    verb = "deleted" if a.delete else "would delete"
    print(
        "%s %s L0 older than %s: %d file(s), %.0f GB; %d without both rungs kept"
        % (now.strftime("%Y-%m-%dT%H:%M:%SZ"), verb, cutoff, gone, gb_gone, len(norung))
    )
    for rel in norung[:5]:
        print("  no rung:", rel)


if __name__ == "__main__":
    main()
