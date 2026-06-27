#!/usr/bin/env python3
"""Currently-visible GPS PRNs for a site -- print them, or patch a config's prns/n_prn.

The GPS constellation rotates ~half an orbit in ~8 h, so a hardcoded search PRN list
goes stale (the broker gates by almanac, but the SEARCH only looks for the config
list). run_live.sh calls this with --patch when LAT/LON are set, so each run searches
what's actually up. Standalone:  python3 gps_visible_prns.py --lat .. --lon .. --alt ..
"""
import argparse
import re
import sys
from datetime import datetime, timezone


def visible_prns(lat, lon, alt, mask_deg, tle=None):
    """PRNs above the elevation mask now, highest elevation first."""
    from gps_beamtrack import predict_dopplers
    kw = {"tle_source": tle} if tle else {}
    pred = predict_dopplers(lat, lon, alt, prns=list(range(1, 33)),
                            t_utc=datetime.now(timezone.utc), **kw)
    up = sorted(((p, d[2]) for p, d in pred.items() if d[2] >= mask_deg),
                key=lambda x: -x[1])
    return [p for p, _ in up]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--alt", type=float, default=0.0)
    ap.add_argument("--mask", type=float, default=10.0, help="elevation mask, deg")
    ap.add_argument("--max", type=int, default=12, help="cap the list (highest elev first)")
    ap.add_argument("--tle", default=None, help="TLE file/URL (default Celestrak gps-ops)")
    ap.add_argument("--patch", help="config to rewrite prns:/n_prn: from the visible set")
    ap.add_argument("--out", help="write the patched config here (default: in place)")
    args = ap.parse_args(argv)

    prns = visible_prns(args.lat, args.lon, args.alt, args.mask, args.tle)[:args.max]
    if not prns:
        sys.exit("no GPS sats above %g deg -- check time/location/network" % args.mask)

    if args.patch:
        txt = open(args.patch).read()
        lst = "[" + ", ".join(str(p) for p in prns) + "]"
        txt, n1 = re.subn(r"^prns:.*$", "prns: " + lst, txt, flags=re.M)
        txt, n2 = re.subn(r"^n_prn:.*$", "n_prn: %d" % len(prns), txt, flags=re.M)
        if n1 != 1 or n2 != 1:
            sys.exit("patch failed: matched prns:=%d n_prn:=%d (expected 1 each)" % (n1, n2))
        open(args.out or args.patch, "w").write(txt)
        sys.stderr.write("[visible_prns] %d up: %s\n" % (len(prns), lst))
    else:
        print(" ".join(str(p) for p in prns))


if __name__ == "__main__":
    main()
