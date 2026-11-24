#!/usr/bin/env python3
"""Bootstrap the bladeRF receiver from the sky alone: no cached orbits, no time, no position.

    python3 gnss_bootstrap.py [--apply] [--max-wait-s 900] [--truth-yaml Y]

The broker's model needs all three, so it is held off (the gnss-broker unit's
ConditionPathExists on --hold) while this runs:

1. Acquire with no date: every L5 slot hinted at 0 +- 5 kHz (the GPSDO leaves only the
   satellites' own Doppler), each detection seeded, and the tracks held on the combiner's
   discriminators (code by dll_disc, Doppler by carrier_hz_resid), as bladerf/track_live.py
   does. Each tracked satellite's record is logged each second (--rows), with its measured
   Doppler: the record's less the carrier residual.
2. A CNAV decoder of its own (nav_decode.py in place of gnss-nav-decode@misc, into files that
   start empty, with no freshness gate against the host clock and no BRDC) reads those tracks:
   time marks, and orbits with clocks. Wait for a type 10 (the week) and clocks from four
   satellites.
3. Time: gnss_skytime on the marks and the decoded orbits, no site: the anchor's error to ~10 ms.
4. Position: gnss_fix on the logged rows and the decoded orbits, no prior, that error taken off.
5. Time again, the fix as the site: to ~1 ms.
6. With --apply: the clock offset (gnss_set_clock) and, if the fix is over --site-tol-m from the
   configured site, the site (gnss_set_site --fix-json); restart kotekan and release the broker.
   Without it, report only and release the broker as it was.
7. After a site move, once the broker has relocked what it can (--refine-after-s), fix again
   from its own rows, every chain, and move again if needed: a bootstrap fix 163 m off left
   the model-seeded chains at half strength; the chains together land within tens of metres.

Only offsets are printed; the fix itself goes to --fix-json.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request
from decimal import Decimal

import numpy as np

import gnss_fix
import gnss_pvt
import gnss_set_clock
import gnss_skytime

KOTEKAN = "http://localhost:12048"
B = "/home/lwlab/gnss/bladerf"
YAML = B + "/broker/chains_l5.yaml"
FS = Decimal("122.88e6")
NFFT = 64
CARRIER_HZ = 1176.45e6
CHIP_HZ = Decimal("10.23e6")
CODE_CHIPS = 204600  # GPS_L5_Q_NH: the NH20 tiled into one 20 ms code
HOPS_PER_S = 1920000
DECODER_STALL_S = 240.0


def get(path):
    return json.load(urllib.request.urlopen(KOTEKAN + path, timeout=5))


def post(path, body):
    r = urllib.request.Request(
        KOTEKAN + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(r, timeout=5).status


def chips_per_hop(dop):
    return CHIP_HZ / FS * NFFT * (1 + Decimal(dop) / Decimal(CARRIER_HZ))


def log(msg):
    print("[bootstrap %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


class Tracker:
    """L5 tracking with no model: blind hints, seeds from detections, held on the combiner."""

    def __init__(self, rows_path, snr=20.0, lock=5.0):
        self.snr, self.lock = snr, lock
        self.st = {}
        self.rows = open(rows_path, "w")
        prns = get("/config")["l5_search"]["prns"]
        post(
            "/l5_search/set_doppler_hints",
            [{"prn": p, "doppler_hz": 0.0, "margin_hz": 5000.0} for p in prns],
        )

    def step(self, frame0):
        """One pass: seed new detections, nudge and log the tracked, drop the lost."""
        for d in get("/l5_search/get_detections"):
            if d["snr"] < self.snr or d["prn"] in self.st:
                continue
            # the detection's argument is referred to sample 0 with its own Doppler; lift it to
            # the phase at its reference hop, at the hop's last sample as the bank does
            n = Decimal(d["ref_hop"]) * NFFT + NFFT - 1
            ph = (
                Decimal(d["code_phase_chips"])
                + n * chips_per_hop(d["doppler_hz"]) / NFFT
            ) % CODE_CHIPS
            self.st[d["prn"]] = dict(
                dop=d["doppler_hz"], ph=ph, ref=d["ref_hop"], miss=0
            )
            post(
                "/l5_search/set_doppler_hints",
                [{"prn": d["prn"], "doppler_hz": 0.0, "margin_hz": -1.0}],
            )
            log(
                "PRN %d seeded: snr %.0f, %+.0f Hz"
                % (d["prn"], d["snr"], d["doppler_hz"])
            )
        rows = get("/l5_combine/get_status")
        rows = rows if isinstance(rows, list) else rows.get("prns", [])
        status = {r["prn"]: r for r in rows if isinstance(r, dict) and r.get("prn")}
        seeds = []
        for prn, s in sorted(self.st.items()):
            cs = status.get(prn, {})
            hop = cs.get("pow_hop", 0)
            if cs.get("deep_snr", 0.0) >= self.lock and hop > s["ref"]:
                # carry the seed to the combiner's newest hop, then correct it there
                s["ph"] = (
                    s["ph"] + chips_per_hop(s["dop"]) * (hop - s["ref"])
                ) % CODE_CHIPS
                s["ref"] = hop
                s["ph"] = (s["ph"] - Decimal(0.16 * cs["dll_disc"])) % CODE_CHIPS
                s["dop"] -= 0.5 * cs["carrier_hz_resid"]
                s["miss"] = 0
                row = {
                    "t": frame0 + hop / HOPS_PER_S,
                    "band": "gps_l5",
                    "sys": "G",
                    "prn": prn,
                    "carrier_hz": CARRIER_HZ,
                    "rec_hop": hop,
                    "cp_rec_chips": cs["code_phase_chips"],
                    "dop_rec_hz": cs["doppler_hz"],
                    "dop_meas_hz": cs["doppler_hz"] - cs["carrier_hz_resid"],
                    "fleet_present": True,
                }
                self.rows.write(json.dumps(row) + "\n")
            else:
                s["miss"] += 1
            seeds.append(
                {
                    "prn": prn,
                    "doppler_hz": s["dop"],
                    "code_phase_chips": 0.0,
                    "ref_hop": s["ref"],
                    "code_phase_at_ref_chips": float(s["ph"]),
                }
            )
        self.rows.flush()
        for prn in [p for p, s in self.st.items() if s["miss"] > 15]:
            post(
                "/l5_search/set_doppler_hints",
                [{"prn": prn, "doppler_hz": self.st[prn]["dop"], "margin_hz": 500.0}],
            )
            del self.st[prn]
            log("PRN %d lost" % prn)
        if seeds:
            post("/l5_track/set_seeds", seeds)
        return sum(1 for s in self.st.values() if s["miss"] == 0)


def start_decoder(here, run_dir, eph_path, marks_path):
    """nav_decode.py on L5 CNAV into files that start empty, with no freshness gate against the
    host clock and no BRDC, reporting every 15 s."""
    for p in (eph_path, marks_path):
        if os.path.exists(p):
            os.remove(p)
    return subprocess.Popen(
        [sys.executable, "-u", "nav_decode.py"]
        + ["--source", "GPS_L5_CNAV=%s/l5i_symbols" % KOTEKAN]
        + ["--eph-out", eph_path, "--timemarks-file", marks_path]
        + ["--eph-fresh-s", "0", "--no-brdc", "--report-every", "15"],
        cwd=here,
        stdout=open(run_dir + "/bootstrap_nav_decode.log", "a"),
        stderr=subprocess.STDOUT,
    )


def decoded_ready(eph_path, marks_path, t_start):
    """(clocked satellites, marks' satellites, has a type 10), from files written since t_start."""
    n_clk = n_marks = 0
    week = False
    try:
        if os.path.getmtime(eph_path) >= t_start:
            n_clk = sum("af0" in r["eph"] for r in json.load(open(eph_path))["eph"])
        if os.path.getmtime(marks_path) >= t_start:
            marks = json.load(open(marks_path))["marks"]
            n_marks = len({m["prn"] for m in marks})
            week = any("wn" in m for m in marks)
    except (OSError, ValueError, KeyError):
        pass
    return n_clk, n_marks, week


def systemctl(*args):
    subprocess.run(["systemctl", "--user", *args], check=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--apply", action="store_true", help="correct clock and site, restart"
    )
    ap.add_argument("--max-wait-s", type=float, default=900.0)
    ap.add_argument("--hold", default=B + "/broker/HOLD", help="the broker's hold file")
    ap.add_argument("--rows", default=B + "/runs/live/bootstrap_rows.jsonl")
    ap.add_argument("--fix-json", default=B + "/runs/live/bootstrap_fix.json")
    ap.add_argument("--yaml", default=YAML, help="the broker yaml holding the site")
    ap.add_argument("--site-tol-m", type=float, default=50.0)
    ap.add_argument(
        "--min-track-s",
        type=float,
        default=180.0,
        help="track at least this long before solving: the Doppler fix wants the geometry to "
        "move (90 s of five satellites left it 31 km off, and the code fix 163 m)",
    )
    ap.add_argument(
        "--refine-after-s",
        type=float,
        default=180.0,
        help="after a site move, let the broker relock this long, then fix again from its "
        "own rows (every chain) and move again if over --site-tol-m; 0 skips",
    )
    ap.add_argument(
        "--truth-yaml", help="print the fix's offsets from this yaml's site"
    )
    a = ap.parse_args(argv)
    run_dir = os.path.dirname(a.rows)
    eph_path = run_dir + "/bootstrap_eph.json"
    marks_path = run_dir + "/bootstrap_marks.json"
    for p in (eph_path, marks_path):
        if os.path.exists(p):
            os.remove(p)
    frame0 = gnss_fix.FRAME0_UTC

    open(a.hold, "w").close()
    systemctl("stop", "gnss-broker")
    systemctl("stop", "gnss-nav-decode@misc")
    here = os.path.dirname(os.path.abspath(__file__))
    t_start = time.time()
    log("broker held; acquiring L5 with no date (0 +- 5 kHz on every slot)")
    trk = Tracker(a.rows)
    ready = False
    t_log = t_start
    decoder, t_dec, settled, progress = None, 0.0, None, (0, 0)
    while time.time() - t_start < a.max_wait_s:
        time.sleep(1.0)
        n_trk = trk.step(frame0)
        # The decoder starts on settled tracks: symbols from seeds still converging leave it an
        # island of bad symbols whose failed cold decodes back off for minutes (11-24: the two
        # strongest satellites never decoded). It restarts when nothing new decodes for
        # DECODER_STALL_S.
        settled = (settled or time.time()) if n_trk >= 3 else None
        n_clk, n_marks, week = decoded_ready(eph_path, marks_path, t_start)
        if decoder is None and settled and time.time() - settled >= 20.0:
            decoder, t_dec = (
                start_decoder(here, run_dir, eph_path, marks_path),
                time.time(),
            )
            log("CNAV decoder started on %d tracks" % n_trk)
        if decoder is not None:
            if (n_clk, n_marks) != progress:
                progress, t_dec = (n_clk, n_marks), time.time()
            elif time.time() - t_dec > DECODER_STALL_S:
                decoder.terminate()
                decoder, t_dec = (
                    start_decoder(here, run_dir, eph_path, marks_path),
                    time.time(),
                )
                log("CNAV decoder restarted: nothing new for %.0f s" % DECODER_STALL_S)
        if time.time() - t_log >= 30.0:
            t_log = time.time()
            log(
                "%d tracked; CNAV: %d satellites with clocks, marks from %d, week %s"
                % (n_trk, n_clk, n_marks, "yes" if week else "not yet")
            )
        long_enough = time.time() - t_start >= a.min_track_s
        if n_clk >= 4 and n_marks >= 2 and week and long_enough:
            ready = True
            break
    ok = False
    if ready:
        res, _ = gnss_skytime.run([marks_path], decoded=[eph_path])
        e0 = float(np.median([r[2] for r in res]))
        log("time: the anchor is %+.6f s off (nominal light time)" % e0)
        argv_fix = [a.rows, "--decoded-eph", eph_path, "--time-offset-s", str(e0)]
        argv_fix += ["--doppler-span-s", str(time.time() - t_start)]
        argv_fix += ["--json", a.fix_json]
        if a.truth_yaml:
            argv_fix += ["--truth-yaml", a.truth_yaml]
        if gnss_fix.main(argv_fix) == 0 and os.path.exists(a.fix_json):
            res, _ = gnss_skytime.run(
                [marks_path], site_json=a.fix_json, decoded=[eph_path]
            )
            e1 = float(np.median([r[2] for r in res]))
            log("time with the fix as the site: the anchor is %+.6f s off" % e1)
            ok = True
    else:
        log("no time and position within %.0f s" % a.max_wait_s)
    moved = 0.0
    if ok and a.apply:
        old = gnss_set_clock.read_offset(gnss_set_clock.FILE)
        gnss_set_clock.write_offset(gnss_set_clock.FILE, old - (e1 - 0.9e-3))
        log("clock offset %+.6f s (was %+.6f)" % (old - (e1 - 0.9e-3), old))
        fix = np.array(json.load(open(a.fix_json))["ecef"])
        site = np.array(gnss_pvt._llh_to_ecef(*gnss_fix.site_of(a.yaml)))
        moved = float(np.linalg.norm(fix - site))
        if moved > a.site_tol_m:
            subprocess.run(
                [sys.executable, "gnss_set_site.py", "--fix-json", a.fix_json]
                + ["--yaml", a.yaml, "--broker", "http://127.0.0.1:1"],
                check=False,
                cwd=here,
            )
            for u in ("gnss-obs", "gnss-sky", "gnss-viewer"):
                systemctl("restart", u)
        else:
            log("the configured site is %.0f m from the fix: kept" % moved)
        systemctl("restart", "gnss-kotekan")
    if decoder is not None:
        decoder.terminate()
    systemctl("start", "gnss-nav-decode@misc")
    os.remove(a.hold)
    systemctl("start", "gnss-broker")
    log("broker released")
    if ok and a.apply and moved > a.site_tol_m and a.refine_after_s > 0:
        refine(a, here)
    return 0 if ok else 1


def refine(a, here):
    """Fix again from the broker's observables rows, once the chains the moved site let it
    lock have run --refine-after-s: every chain's rows, the first fix as the prior; move the
    site again if it is over --site-tol-m off."""
    log("refining in %.0f s from the broker's own rows" % a.refine_after_s)
    time.sleep(a.refine_after_s)
    newest = {}
    for p in glob.glob(B + "/obs/*_*.jsonl"):
        chain = os.path.basename(p).rsplit("_", 1)[0]
        if chain != "gw_1202" and (
            chain not in newest or os.path.getmtime(p) > os.path.getmtime(newest[chain])
        ):
            newest[chain] = p
    lat, lon, alt = gnss_pvt._ecef_to_llh(np.array(json.load(open(a.fix_json))["ecef"]))
    out = a.fix_json.replace(".json", "_refined.json")
    argv_fix = sorted(newest.values()) + ["--prior", "%.9f,%.9f,%.3f" % (lat, lon, alt)]
    argv_fix += ["--prior-km", "1", "--json", out]
    if a.truth_yaml:
        argv_fix += ["--truth-yaml", a.truth_yaml]
    if gnss_fix.main(argv_fix) != 0 or not os.path.exists(out):
        log("no refined fix")
        return
    fix = np.array(json.load(open(out))["ecef"])
    site = np.array(gnss_pvt._llh_to_ecef(*gnss_fix.site_of(a.yaml)))
    moved = float(np.linalg.norm(fix - site))
    if moved <= a.site_tol_m:
        log("refined fix %.0f m from the site: kept" % moved)
        return
    subprocess.run(
        [sys.executable, "gnss_set_site.py", "--fix-json", out, "--yaml", a.yaml]
        + ["--broker", "http://127.0.0.1:1"],
        check=False,
        cwd=here,
    )
    for u in ("gnss-broker", "gnss-obs", "gnss-sky", "gnss-viewer"):
        systemctl("restart", u)
    log("site moved %.0f m to the refined fix" % moved)


if __name__ == "__main__":
    sys.exit(main())
