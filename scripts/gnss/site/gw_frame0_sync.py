#!/usr/bin/env python3
"""Add a new CHORD F-engine start to the GuoWang reducer's frame0 list.

Every F-engine start resets the fpga seq to 0. The GuoWang beam-dump manifest, gw_reduce.py and
track_beam_loop.py place the files in time from builds/gw1202/frame0.json, and hold out or mistime
files whose start is missing. Two sources, each appended when no entry has the same integer second:

  - the F-engine controller (GET http://crs-control:54321/get-frame0-time), which serves a new start
    within seconds of it; its frame0_nano is one GPS week rollover early (+1024 weeks), and it
    answers {} while it initializes. Its start is appended only once a node runs on it
    (<node>:12048/telescope/time0_ns, the endpoint the broker reads; nodes relaunch onto a new start
    within ~15 s), so a start the controller serves and then abandons never gets in. The entry takes
    the fraction of the newest entry, the broker's convention (the controller's differs by ~75 ns);
  - the broker's "time anchor: CHORD F-engine frame0 = <unix> s" line, which comes once the broker
    has confirmed the start (three 60 s checks, ~4 min). The newest such line of the broker log is
    read, or, when the log has none (it rotates daily at 00:23Z), of the newest rotated log
    (<log>.1.gz, read once per rotation).

The broker is the cross-check: a broker start that disagrees with the controller's after CONFIRM_S,
or a fraction that disagrees with the stored entry, is reported as FRAME0 MISMATCH and nothing is
overwritten. Run from cron on the gnss VM (guowang_db_sync.sh, and on its own for a fast pick-up).

    gw_frame0_sync.py [--log /var/tmp/gnss-logs/broker.log] [--frame0 .../frame0.json]
                      [--controller URL | --no-controller] [--nodes cx19,cx43,...]
"""
import argparse
import gzip
import json
import os
import re
import sys
import time
import urllib.request

ANCHOR = re.compile(rb"time anchor: CHORD F-engine frame0 = (\d+)\.(\d+) s")
CHUNK = 16 << 20  # the log is read backwards in chunks: memory stays at one chunk
CONTROLLER = "http://crs-control:54321/get-frame0-time"
GPS_ROLLOVER_S = 1024 * 7 * 86400  # the controller's frame0 is one GPS week rollover early
NODES = "cx19,cx27,cx43,cx44,cx51,cx52"  # any one confirms; cx42/cx47 are lent out at times
CONFIRM_S = 600  # the broker logs a new start ~4 min after it; later, a different start is a mismatch


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


def controller_frame0(url):
    """(integer second, start_ctime) of the controller's current F-engine start, or None while it
    initializes or cannot be reached."""
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            d = json.load(r)
    except (OSError, ValueError) as e:
        print("controller %s: %s" % (url, e), file=sys.stderr)
        return None
    if "frame0_nano" not in d:
        return None
    return d["frame0_nano"] // 10**9 + GPS_ROLLOVER_S, d.get("start_ctime")


def node_runs_on(nodes, frame0_int):
    """The first node whose telescope/time0_ns is on this start, or None."""
    for n in nodes:
        try:
            with urllib.request.urlopen("http://%s:12048/telescope/time0_ns" % n, timeout=5) as r:
                if int(json.load(r)["time0_ns"]) // 10**9 == frame0_int:
                    return n
        except (OSError, ValueError, KeyError, TypeError):
            continue  # a node being down or restarting is the normal case here
    return None


def broker_frame0(log, frame0):
    """(int, frac) of the broker's newest anchor, or None."""
    anchor = newest_anchor(log)
    rotated = log + ".1.gz"
    seen = os.path.join(os.path.dirname(frame0), ".frame0_sync_rotated")
    if anchor is None and os.path.exists(rotated):
        stamp = str(os.path.getmtime(rotated))
        if not os.path.exists(seen) or open(seen).read() != stamp:
            anchor = newest_anchor_gz(rotated)
            with open(seen, "w") as fh:
                fh.write(stamp)
    return anchor


def utc(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--log", default="/var/tmp/gnss-logs/broker.log")
    ap.add_argument("--frame0", default="/home/kvand/gnss/builds/gw1202/frame0.json")
    ap.add_argument("--controller", default=CONTROLLER)
    ap.add_argument("--no-controller", action="store_true", help="the broker log only")
    ap.add_argument("--nodes", default=NODES, help="nodes that confirm a controller start")
    a = ap.parse_args()
    ctl = None if a.no_controller else controller_frame0(a.controller)
    broker = broker_frame0(a.log, a.frame0)
    with open(a.frame0) as fh:
        starts = json.load(fh)
    by_int = {e["frame0_int"]: e for e in starts}
    now = utc(time.time())
    added = []
    node = None
    if ctl and ctl[0] not in by_int:
        node = node_runs_on(a.nodes.split(","), ctl[0])
    if node:
        e = dict(
            frame0_int=ctl[0],
            frame0_frac=starts[-1]["frame0_frac"],
            start_utc=utc(ctl[0]),
            source="F-engine controller get-frame0-time + 1024 GPS weeks, %s running on it,"
            " fraction as the broker's (gw_frame0_sync.py, %s)" % (node, now),
        )
        added.append(e)
        by_int[ctl[0]] = e
    if broker and broker[0] not in by_int:
        e = dict(
            frame0_int=broker[0],
            frame0_frac=broker[1],
            start_utc=utc(broker[0]),
            source="broker.log time anchor (gw_frame0_sync.py, %s)" % now,
        )
        added.append(e)
        by_int[broker[0]] = e
    if ctl and broker:
        if broker[0] != ctl[0] and (ctl[1] is None or time.time() - ctl[1] > CONFIRM_S):
            print(
                "FRAME0 MISMATCH: controller start %d (%s), broker anchor %d (%s)"
                % (ctl[0], utc(ctl[0]), broker[0], utc(broker[0])),
                file=sys.stderr,
            )
        elif broker[0] == ctl[0] and abs(by_int[ctl[0]]["frame0_frac"] - broker[1]) > 1e-9:
            print(
                "FRAME0 MISMATCH: start %d stored with fraction %.9f, broker anchor %.9f"
                % (ctl[0], by_int[ctl[0]]["frame0_frac"], broker[1]),
                file=sys.stderr,
            )
    if not added:
        return
    starts.extend(added)
    tmp = a.frame0 + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(starts, fh, indent=1)
    os.replace(tmp, a.frame0)
    for e in added:
        print(
            "added F-engine start %s (frame0 %d%s) from %s"
            % (e["start_utc"], e["frame0_int"], ("%.9f" % e["frame0_frac"])[1:],
               e["source"].split(" (")[0].split(" get-")[0])
        )

if __name__ == "__main__":
    main()
