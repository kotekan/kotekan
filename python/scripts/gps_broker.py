#!/usr/bin/env python3
"""Broker control loop: visibility -> assignment -> per-worker REST masks.

The control plane of the distributed-band pipeline (hosting locally for now).
Periodically computes which GNSS PRNs are visible and distributes them across the
correlator *workers*, POSTing each worker only the PRNs it should measure to its
``/set_mask`` endpoint. Unlike gps_almanac_rest.py -- which sends the same full
visible list to every instance -- this load-balances the visible set across
workers so each does a share of the work.

Division of labour: which worker owns which PFB channels/band is STATIC (set at
config time, since it follows the hardware band-split) and is captured here by
each worker's signal. What is DYNAMIC -- which PRNs are up -- is what this loop
distributes. For GPS all PRNs of a signal share one carrier (CDMA, same
channels), so carrier-aware placement reduces to matching the signal and then
load-balancing PRNs; the channel-range straddle case (a band split across nodes)
is the C++ gnssBroker assign_targets reference, enforced at config time.

assign_targets() mirrors lib/stages/gnssBroker.cpp (least-loaded, deterministic)
and is unit-tested in tests/test_gps_broker.py.

Dependencies: skyfield (via gps_beamtrack); REST via stdlib urllib.

Examples
--------
    # 4 L1 workers, load-balance visible PRNs every 60 s
    python gps_broker.py --lat 43.66 --lon -79.40 \
        --workers gps_a,gps_b,gps_c,gps_d

    # mixed signals + a PRN-range cap, one-shot
    python gps_broker.py --lat 43.66 --lon -79.40 \
        --workers "l1a:GPS_L1CA,l1b:GPS_L1CA,l2:GPS_L2C_CL:1-32" --once
"""

import argparse
import json
import sys
import time
import urllib.request
from collections import namedtuple
from datetime import datetime, timedelta, timezone

Worker = namedtuple("Worker", ["name", "signal", "prn_lo", "prn_hi"])


def assign_targets(workers, targets):
    """Place each (signal, prn) target on the least-loaded worker that correlates
    its signal and accepts its PRN. Returns (worker_prns, unplaced), where
    worker_prns maps every worker name -> sorted PRN list. Deterministic:
    workers are considered in name order. Mirrors gnssBroker.cpp."""
    order = sorted(workers, key=lambda w: w.name)
    worker_prns = {w.name: [] for w in order}
    load = {w.name: 0 for w in order}
    unplaced = []

    for signal, prn in targets:
        best = None
        for w in order:
            if w.signal != signal or prn < w.prn_lo or prn > w.prn_hi:
                continue
            if best is None or load[w.name] < load[best.name]:
                best = w
        if best is None:
            unplaced.append((signal, prn))
        else:
            worker_prns[best.name].append(prn)
            load[best.name] += 1

    for prns in worker_prns.values():
        prns.sort()
    return worker_prns, unplaced


def parse_workers(spec):
    """Parse "name[:signal[:lo-hi]],..." into a list of Worker."""
    workers = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        name = parts[0]
        signal = parts[1] if len(parts) > 1 and parts[1] else "GPS_L1CA"
        lo, hi = 1, 32
        if len(parts) > 2 and parts[2]:
            lo, hi = (int(x) for x in parts[2].split("-"))
        workers.append(Worker(name, signal, lo, hi))
    return workers


def _log(msg):
    print("%s %s" % (datetime.now(timezone.utc).isoformat(timespec="seconds"), msg),
          file=sys.stderr, flush=True)


def visible_prns(sats, lat, lon, alt_m, mask_deg, look_ahead_s):
    """PRNs above `mask_deg` now or within `look_ahead_s` (a satellite rising
    before the next cycle is enabled slightly early rather than clipped)."""
    import numpy as np
    from gps_visibility import altaz_over
    now = datetime.now(timezone.utc)
    times = [now, now + timedelta(seconds=look_ahead_s)]
    aa = altaz_over(sats, lat, lon, alt_m, times)
    return sorted(p for p, (a, _z) in aa.items() if float(np.max(a)) >= mask_deg)


def post_mask(rest_url, stage, prns, timeout=10.0):
    """POST {"active_prns": prns} to <rest_url>/<stage>/set_mask."""
    url = "%s/%s/set_mask" % (rest_url.rstrip("/"), stage)
    body = json.dumps({"active_prns": prns}).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, required=True, help="observer latitude, deg")
    ap.add_argument("--lon", type=float, required=True, help="observer longitude, deg")
    ap.add_argument("--alt", type=float, default=0.0, help="observer height, m")
    ap.add_argument("--mask", type=float, default=5.0, help="horizon mask elevation, deg")
    ap.add_argument("--workers", default="gps_a,gps_b,gps_c,gps_d",
                    help='workers as "name[:signal[:lo-hi]],..."')
    ap.add_argument("--rest-url", default="http://localhost:12048",
                    help="kotekan REST base URL")
    ap.add_argument("--interval", type=float, default=60.0, help="update period, s")
    ap.add_argument("--once", action="store_true", help="update once and exit")
    args = ap.parse_args(argv)

    from gps_beamtrack import DEFAULT_TLE_URL, load_gps_satellites
    workers = parse_workers(args.workers)
    signals = sorted({w.signal for w in workers})
    sats = load_gps_satellites(DEFAULT_TLE_URL)
    _log("loaded %d satellites; workers=%s signals=%s" %
         (len(sats), [w.name for w in workers], signals))

    while True:
        try:
            look = args.interval if not args.once else 0.0
            vis = visible_prns(sats, args.lat, args.lon, args.alt, args.mask, look)
            targets = [(sig, prn) for sig in signals for prn in vis]
            worker_prns, unplaced = assign_targets(workers, targets)
            ok = 0
            for w in workers:
                try:
                    post_mask(args.rest_url, w.name, worker_prns[w.name])
                    ok += 1
                except Exception as e:  # noqa: BLE001 -- keep the loop alive
                    _log("POST to %s failed: %s" % (w.name, e))
            _log("visible=%s assigned=%s%s [%d/%d ok]" %
                 (vis, {k: v for k, v in worker_prns.items() if v},
                  (" unplaced=%s" % unplaced) if unplaced else "", ok, len(workers)))
        except Exception as e:  # noqa: BLE001 -- unattended: never die on a transient
            _log("update cycle failed: %s" % e)

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
