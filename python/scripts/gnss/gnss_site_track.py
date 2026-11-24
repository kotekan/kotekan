#!/usr/bin/env python3
"""Keep the receiver's site on its own position fix: refine a static site, or follow a moving one.

    python3 gnss_site_track.py --mode refine|mobile --yaml chains_l5.yaml [--every-s S]
                               [--move-m 10] [--broker URL] [--obs-glob G]
                               [--until-settled N [--max-s S]]

The solve is the Position menu's (its 15 deg mask, and the Klobuchar delay the observables
apply): gnss_pvt over the observables rows (each chain's code
residual against the broker's model at its site), every chain's own clock. The broker moves the
site live (POST /set_site, at most publish.SITE_LIVE_MAX_M a request), so a correction goes as
steps of that size a second apart; the yaml follows each (gnss_set_site.move_yaml), so a restart
keeps it. A correction over --restart-over-m writes the yaml and restarts the broker and the
site's readers instead, as gnss_set_site does.

refine (default every 15 s): a solve over the last 90 s of carrier-smoothed rows each time; the
site moves when the last --confirm solves agree, their median offset over --move-m and over
five times their own scatter (the solves wander by metres within minutes, more than their
formal sigma says). After a move nothing is judged until a whole smoothing window has passed,
so no solve mixes rows from both sites; nor at the start, since the site may have just been
moved. Nor while any chain's trackers are not on their signals (DISC_MAX_CHIPS). With
--until-settled it stops once N judgments in a
row (each over the last --confirm solves) have needed no move, or after --max-s, so a site
found from the sky is then fixed and measurements stay comparable.

mobile (default every 5 s): an unsmoothed solve of the newest rows, and the site follows it.
Only the site moves: the model carries no receiver velocity, so a receiver moving faster than
a walk will put its Doppler off by v/lambda (4 Hz per m/s at L5).

Only offsets are logged.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request

import numpy as np

import gnss_pvt
from gnss_broker.publish import SITE_LIVE_MAX_M
from gnss_fix import site_of
from gnss_set_site import move_yaml

SMOOTH_S = 90.0
# A chain whose satellites' median code discriminator is over this is not on its signals yet:
# for minutes after a broker start (its trims and joint clock settling) L5, E5a and B2a sat at
# 0.11-0.20 chips and the solve read a 20 m height offset as 3 m (11-24); settled, 0.02-0.05.
DISC_MAX_CHIPS = 0.08


def log(msg):
    print("[site %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def solve(obs_glob, yaml, smooth_s):
    """(d_enu, sigma_enu, n_sats) of the combined fix against the yaml's site, or None."""
    meas = gnss_pvt.measurements_from_obs([obs_glob], 30.0, time.time(), smooth_s)
    if not meas:
        return None
    discs = {}
    for m in meas:
        if "disc_chips" in m and m["sys"] != "W":  # GuoWang's TLE model leaves no fix
            discs.setdefault(m["group"], []).append(m["disc_chips"])
    off = {g: np.median(d) for g, d in discs.items() if len(d) >= 3}
    off = {g: v for g, v in off.items() if v > DISC_MAX_CHIPS}
    if off:
        log(
            "not judged: trackers not on their signals yet (median |dll_disc| %s chips)"
            % ", ".join("%s %.2f" % (g, v) for g, v in sorted(off.items()))
        )
        return None
    sol = gnss_pvt.solve(meas, *site_of(yaml), min_el_deg=15.0).get("combined")
    if not sol or sol.get("n_sats", 0) < 6:
        return None
    d = np.array([sol["d_e"], sol["d_n"], sol["d_u"]])
    sig = np.array([sol["sigma_e"], sol["sigma_n"], sol["sigma_u"]])
    return d, sig, sol["n_sats"]


def move(a, d):
    """Move the site by d (ENU m): live in steps, or by the yaml and restarts if large."""
    mag = float(np.linalg.norm(d))
    if a.dry_run:
        log("would move E %+.1f N %+.1f U %+.1f m" % tuple(d))
        return
    if mag > a.restart_over_m:
        move_yaml(a.yaml, list(d))
        for u in ("gnss-broker", "gnss-obs", "gnss-sky", "gnss-viewer"):
            subprocess.run(["systemctl", "--user", "restart", u], check=False)
        log("moved E %+.1f N %+.1f U %+.1f m (yaml and restart)" % tuple(d))
        return
    n = int(np.ceil(mag / (0.9 * SITE_LIVE_MAX_M))) or 1
    for k in range(n):
        step = d / n
        req = urllib.request.Request(
            a.broker.rstrip("/") + "/set_site",
            data=json.dumps({"d_e": step[0], "d_n": step[1], "d_u": step[2]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=5).read()
        except Exception as e:
            log("live move refused (%s); stopping this correction" % e)
            return
        move_yaml(a.yaml, list(step))
        if k + 1 < n:
            time.sleep(1.0)
    log("moved E %+.1f N %+.1f U %+.1f m live (%d steps)" % (*d, n))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=("refine", "mobile"), default="refine")
    ap.add_argument("--yaml", required=True, help="the broker yaml holding the site")
    ap.add_argument("--broker", default="http://127.0.0.1:12060")
    ap.add_argument("--obs-glob", default="/home/lwlab/gnss/bladerf/obs/*.jsonl")
    ap.add_argument("--every-s", type=float, help="solve period (60 refine, 5 mobile)")
    ap.add_argument("--move-m", type=float, default=10.0, help="refine: move when over")
    ap.add_argument("--confirm", type=int, default=4, help="refine: solves that agree")
    ap.add_argument(
        "--dry-run", action="store_true", help="log the decisions, move nothing"
    )
    ap.add_argument("--restart-over-m", type=float, default=50.0)
    ap.add_argument(
        "--until-settled",
        type=int,
        default=0,
        help="refine: stop once this many judgments in a row need no move",
    )
    ap.add_argument(
        "--max-s", type=float, default=0.0, help="stop unsettled after this long"
    )
    a = ap.parse_args(argv)
    every = a.every_s or (15.0 if a.mode == "refine" else 5.0)
    t_start = time.time()
    hist, hold_until, quiet = [], t_start + SMOOTH_S + 30.0, 0
    log("%s mode, every %.0f s" % (a.mode, every))
    while True:
        if a.until_settled and quiet >= a.until_settled:
            log("settled: %d judgments in a row needed no move" % quiet)
            return 0
        if a.max_s and time.time() - t_start > a.max_s:
            log("not settled after %.0f s" % a.max_s)
            return 1
        t0 = time.time()
        try:
            r = solve(a.obs_glob, a.yaml, SMOOTH_S if a.mode == "refine" else 0.0)
        except Exception as e:
            log("solve failed: %s" % e)
            r = None
        if r is not None and t0 >= hold_until:
            d, sig, n = r
            if a.mode == "mobile":
                if np.linalg.norm(d) > 3.0 * np.linalg.norm(sig):
                    move(a, d)
            else:
                hist = (hist + [d])[-a.confirm :]
                med = np.median(hist, axis=0)
                spread = np.median(np.abs(np.array(hist) - med), axis=0) * 1.4826
                off = float(np.linalg.norm(med))
                log(
                    "fix E %+.1f N %+.1f U %+.1f m, %d sats; median of %d: %.1f m (scatter "
                    "%.1f m)" % (*d, n, len(hist), off, float(np.linalg.norm(spread)))
                )
                if (
                    len(hist) == a.confirm
                    and off > a.move_m
                    and off > 5.0 * float(np.linalg.norm(spread))
                ):
                    move(a, med)
                    hist, hold_until, quiet = [], time.time() + SMOOTH_S + 30.0, 0
                elif len(hist) == a.confirm:
                    quiet += 1
        time.sleep(max(0.0, every - (time.time() - t0)))


if __name__ == "__main__":
    sys.exit(main())
