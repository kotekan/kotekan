#!/usr/bin/env python3
"""Move the receiver's site by an east/north/up offset in metres, such as a self-survey result.

    python3 gnss_set_site.py --enu dE dN dU --yaml chains_l5.yaml [--broker URL] [--restart U ...]
    python3 gnss_set_site.py --fix-json FIX.json --yaml chains_l5.yaml [--restart U ...]

Writes the broker yaml's site (its lat/lon/alt lines, at their own precision) moved by the
offset, so a restart keeps it, and moves the running broker live through POST /set_site when the
move is at most publish.SITE_LIVE_MAX_M. Anything that reads the site only at start (the
observables writers, the viewers) needs a restart to follow, and so does the broker after a
larger move: --restart names systemd user units to restart. --fix-json moves the site to a
gnss_fix.py --json fix (its ECEF), however far. Only offsets are printed, never a position.
"""

import argparse
import json
import re
import subprocess
import sys
import urllib.request

import numpy as np

import gnss_pvt
from gnss_broker.publish import SITE_LIVE_MAX_M

_KEY = r"\n    %s: ([-0-9.eE+]+)"


def move_yaml(path, d_enu):
    """Rewrite the yaml's site moved by d_enu (m). Returns the ENU offset actually written, which
    differs from d_enu only by the lines' rounding."""
    s = open(path).read()
    old, places = {}, {}
    for k in ("lat", "lon", "alt"):
        m = re.findall(_KEY % k, s)
        if len(m) != 1:
            raise SystemExit(
                "%s: want one '    %s:' line, found %d" % (path, k, len(m))
            )
        old[k] = float(m[0])
        places[k] = len(m[0].split(".")[1]) if "." in m[0] else 0
    apr = gnss_pvt._llh_to_ecef(old["lat"], old["lon"], old["alt"])
    axes = gnss_pvt._enu_axes(old["lat"], old["lon"])
    new = dict(
        zip(
            ("lat", "lon", "alt"), gnss_pvt._ecef_to_llh(apr + axes.T @ np.array(d_enu))
        )
    )
    for k in ("lat", "lon", "alt"):
        m = re.search(_KEY % k, s)
        s = s[: m.start(1)] + "%.*f" % (places[k], new[k]) + s[m.end(1) :]
    with open(path, "w") as f:
        f.write(s)
    got = {k: float(re.search(_KEY % k, s).group(1)) for k in ("lat", "lon", "alt")}
    return axes @ (gnss_pvt._llh_to_ecef(got["lat"], got["lon"], got["alt"]) - apr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--enu", nargs=3, type=float, metavar=("E", "N", "U"))
    ap.add_argument("--fix-json", help="move to this gnss_fix.py --json fix instead")
    ap.add_argument("--yaml", required=True, help="the broker yaml holding the site")
    ap.add_argument("--broker", default="http://127.0.0.1:12060", help="its publisher")
    ap.add_argument(
        "--restart", nargs="*", default=[], help="systemd user units to restart"
    )
    a = ap.parse_args(argv)
    if (a.enu is None) == (a.fix_json is None):
        ap.error("give one of --enu and --fix-json")
    d = a.enu
    if a.fix_json:
        txt = open(a.yaml).read()
        lat, lon, alt = (
            float(re.search(_KEY % k, txt).group(1)) for k in ("lat", "lon", "alt")
        )
        fix = np.array(json.load(open(a.fix_json))["ecef"])
        d = list(
            gnss_pvt._enu_axes(lat, lon) @ (fix - gnss_pvt._llh_to_ecef(lat, lon, alt))
        )
    written = move_yaml(a.yaml, d)
    print("%s: site moved E %+.2f N %+.2f U %+.2f m" % (a.yaml, *written))
    restart = list(a.restart)
    if float(np.linalg.norm(d)) <= SITE_LIVE_MAX_M:
        req = urllib.request.Request(
            a.broker.rstrip("/") + "/set_site",
            data=json.dumps({"d_e": d[0], "d_n": d[1], "d_u": d[2]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                print("broker: moved live (request #%d)" % json.load(r)["seq"])
        except Exception as e:
            print("broker: not moved live (%s); restart it to take the yaml" % e)
    else:
        print(
            "broker: %.1f m is more than a live move takes; restart it"
            % np.linalg.norm(d)
        )
    for u in restart:
        subprocess.run(["systemctl", "--user", "restart", u], check=False)
        print("restarted %s" % u)
    return 0


if __name__ == "__main__":
    sys.exit(main())
