#!/usr/bin/env python3
"""Almanac -> REST mask updater for a running GPS capture.

Periodically computes which GPS PRNs are above the observer's horizon and POSTs
that set to each GpsReplicaCorrelator instance's ``/set_mask`` endpoint, so
below-horizon satellites stop being correlated (compute saving) without
restarting kotekan. Intended to run alongside a multi-day capture
(config/airspy_gps_allsky.yaml).

Note: for 32-PRN GPS this is optional -- listing all 32 statically and letting
the correlator self-acquire works fine and is simplest. The mask matters more
when you go multi-constellation (100+ codes) and can't afford them all.

Location and time: observer position is passed on the command line (the antenna
is fixed); time is the host system clock (NTP is plenty for alt/az). Neither is
derived from the GPS signal -- we don't decode the nav message. (A site with a
timing GPS could feed position/time from gpsd instead; not wired here.)

Dependencies: skyfield (via gps_beamtrack); REST via stdlib urllib (no requests).

Examples
--------
    # update the 4 allsky instances every 60 s, PRNs above 5 deg
    python gps_almanac_rest.py --lat 43.66 --lon -79.40 \
        --stages gps_a,gps_b,gps_c,gps_d

    # one-shot (compute + post once, then exit)
    python gps_almanac_rest.py --lat 43.66 --lon -79.40 --stages c --once
"""

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

from gps_beamtrack import DEFAULT_TLE_URL, load_gps_satellites
from gps_visibility import altaz_over

TLE_REFRESH_S = 86400.0  # re-pull TLEs daily (they go stale over a multi-day run)


def _log(msg):
    print("%s %s" % (datetime.now(timezone.utc).isoformat(timespec="seconds"), msg),
          file=sys.stderr, flush=True)


def visible_prns(sats, lat, lon, alt_m, mask_deg, look_ahead_s):
    """PRNs above `mask_deg` now or within `look_ahead_s` (so a satellite rising
    before the next update is enabled slightly early rather than clipped)."""
    now = datetime.now(timezone.utc)
    times = [now, now + timedelta(seconds=look_ahead_s)]
    aa = altaz_over(sats, lat, lon, alt_m, times)
    import numpy as np
    return sorted(p for p, (a, _z) in aa.items() if float(np.max(a)) >= mask_deg)


def post_mask(rest_url, stage, prns, timeout=10.0):
    """POST {"active_prns": prns} to <rest_url>/<stage>/set_mask. The stage
    enables its own configured PRNs that appear in the list and disables the
    rest, so the same full list can go to every instance."""
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
    ap.add_argument("--stages", default="gps_a,gps_b,gps_c,gps_d",
                    help="comma-separated stage unique_names to update")
    ap.add_argument("--rest-url", default="http://localhost:12048",
                    help="kotekan REST base URL")
    ap.add_argument("--interval", type=float, default=60.0, help="update period, s")
    ap.add_argument("--tle", default=DEFAULT_TLE_URL, help="TLE file path or URL")
    ap.add_argument("--once", action="store_true", help="update once and exit")
    args = ap.parse_args(argv)

    stages = [s for s in args.stages.split(",") if s]
    sats = load_gps_satellites(args.tle)
    tle_loaded = time.time()
    _log("loaded %d GPS satellites; updating %s every %gs" %
         (len(sats), stages, args.interval))

    while True:
        try:
            if time.time() - tle_loaded > TLE_REFRESH_S:
                sats = load_gps_satellites(args.tle)
                tle_loaded = time.time()
                _log("refreshed TLEs (%d sats)" % len(sats))

            look = args.interval if not args.once else 0.0
            prns = visible_prns(sats, args.lat, args.lon, args.alt, args.mask, look)
            ok = 0
            for stage in stages:
                try:
                    post_mask(args.rest_url, stage, prns)
                    ok += 1
                except Exception as e:  # noqa: BLE001 -- keep the loop alive
                    _log("POST to %s failed: %s" % (stage, e))
            _log("visible (>=%.0f deg): %s  [%d/%d stages updated]" %
                 (args.mask, prns, ok, len(stages)))
        except Exception as e:  # noqa: BLE001 -- unattended: never die on a transient
            _log("update cycle failed: %s" % e)

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
