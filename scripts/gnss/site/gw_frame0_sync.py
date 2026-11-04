#!/usr/bin/env python3
"""Add a new CHORD F-engine start to the GuoWang reducer's frame0 list.

Every F-engine start resets the fpga seq to 0. The GuoWang beam-dump manifest and gw_reduce.py place
the dump files in time from builds/gw1202/frame0.json, and hold out windows whose start is missing.
The broker logs the start it runs on ("time anchor: CHORD F-engine frame0 = <unix> s") once, when it
starts; this reads the newest such line of the broker log, or, when the log has none (it rotates
daily at 00:23Z), of the newest rotated log (<log>.1.gz, read once per rotation), and appends it to
frame0.json when no entry has the same integer second. Run from guowang_db_sync.sh (cron, gnss VM).

    gw_frame0_sync.py [--log /var/tmp/gnss-logs/broker.log] [--frame0 .../frame0.json]
"""
import argparse
import gzip
import json
import os
import re
import time

ANCHOR = re.compile(rb"time anchor: CHORD F-engine frame0 = (\d+)\.(\d+) s")
CHUNK = 16 << 20  # the log is read backwards in chunks: memory stays at one chunk


def newest_anchor(path):
    """(int, frac) of the last anchor line, searching back from the end of the log."""
    with open(path, "rb") as fh:
        end = fh.seek(0, os.SEEK_END)
        while end > 0:
            start = max(0, end - CHUNK)
            fh.seek(start)
            found = ANCHOR.findall(
                fh.read(end - start + 200)
            )  # overlap: a line split at a boundary
            if found:
                i, f = found[-1]
                return int(i), float("0." + f.decode())
            end = start
    return None


def newest_anchor_gz(path):
    """(int, frac) of the last anchor line of a gzipped log, streamed front to back."""
    last, tail = None, b""
    with gzip.open(path, "rb") as fh:
        while True:
            buf = fh.read(CHUNK)
            if not buf:
                return last
            found = ANCHOR.findall(tail + buf)
            if found:
                i, f = found[-1]
                last = int(i), float("0." + f.decode())
            tail = buf[-200:]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--log", default="/var/tmp/gnss-logs/broker.log")
    ap.add_argument("--frame0", default="/home/kvand/gnss/builds/gw1202/frame0.json")
    a = ap.parse_args()
    anchor = newest_anchor(a.log)
    rotated = a.log + ".1.gz"
    seen = os.path.join(os.path.dirname(a.frame0), ".frame0_sync_rotated")
    if anchor is None and os.path.exists(rotated):
        stamp = str(os.path.getmtime(rotated))
        if not os.path.exists(seen) or open(seen).read() != stamp:
            anchor = newest_anchor_gz(rotated)
            with open(seen, "w") as fh:
                fh.write(stamp)
    if anchor is None:
        return
    with open(a.frame0) as fh:
        starts = json.load(fh)
    if any(e["frame0_int"] == anchor[0] for e in starts):
        return
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(anchor[0]))
    starts.append(
        dict(
            frame0_int=anchor[0],
            frame0_frac=anchor[1],
            start_utc=utc,
            source="broker.log time anchor (gw_frame0_sync.py, %s)"
            % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
    )
    tmp = a.frame0 + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(starts, fh, indent=1)
    os.replace(tmp, a.frame0)
    print(
        "added F-engine start %s (frame0 %d%s)"
        % (utc, anchor[0], ("%.9f" % anchor[1])[1:])
    )


if __name__ == "__main__":
    main()
