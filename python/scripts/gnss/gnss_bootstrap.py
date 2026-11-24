#!/usr/bin/env python3
"""Bootstrap the bladeRF receiver from the sky alone: no cached orbits, no time, no position.

    python3 gnss_bootstrap.py [--apply] [--bands gps_l5,gal_e5a,bds_b2a] [--max-wait-s 900]
                              [--truth-yaml Y]

The broker's model needs all three, so it is held off (the gnss-broker unit's
ConditionPathExists on --hold) while this runs:

1. Acquire with no date, on each of --bands (GPS L5, Galileo E5a, BeiDou B2a; one is enough):
   every search slot hinted at 0 +- 5 kHz (the GPSDO leaves only the satellites' own Doppler),
   each detection seeded, and the tracks held on the combiner's discriminators (code by
   dll_disc, Doppler by carrier_hz_resid), as bladerf/track_live.py does. E5a and B2a
   detections carry their 100 ms phase from the search's nh_sync. Each tracked satellite's
   record is logged each second (--rows), with its measured Doppler: the record's less the
   carrier residual.
2. A decoder of its own (nav_decode.py: L5 CNAV, E5a F/NAV, B2a B-CNAV2, in place of the
   gnss-nav-decode units for those bands, into files that start empty, with no freshness gate
   against the host clock and no BRDC) reads those tracks: time marks, and orbits with clocks.
   Wait for a week number and clocks from four satellites.
3. Time: gnss_skytime on the marks and the decoded orbits, no site: the anchor's error to ~10 ms.
4. Position: gnss_fix on the logged rows and the decoded orbits, no prior, that error taken off.
5. Time again, the fix as the site: to ~1 ms.
6. With --apply: the clock offset (gnss_set_clock) and, if the fix is over --site-tol-m from the
   configured site, the site (gnss_set_site --fix-json); restart kotekan and release the broker.
   Without it, report only and release the broker as it was.
7. After a site move, once the broker has relocked what it can (--refine-after-s), refine the
   site from its own rows (gnss_site_track refine, every chain) until it settles, and leave it
   there: a bootstrap fix 163 m off left the model-seeded chains at half strength, and the
   chains together land within metres.

Only offsets are printed; the fix itself goes to --fix-json.
"""

import argparse
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
import gnss_site_track
import gnss_skytime

KOTEKAN = "http://localhost:12048"
B = "/home/lwlab/gnss/bladerf"
YAML = B + "/broker/chains_l5.yaml"
FS = Decimal("122.88e6")
NFFT = 64
# every band here (L5, E5a, B2a) shares the carrier and the chip rate
CARRIER_HZ = 1176.45e6
CHIP_HZ = Decimal("10.23e6")
HOPS_PER_S = 1920000
DECODER_STALL_S = 240.0
# band -> (search, tracker, combiner, tracker code chips, constellation, decoder, its symbols,
# the gnss-nav-decode unit it stands in for, a detection's least SNR to seed). The SNRs are on
# each search's own surface: L5's 3 x 20 ms windows, E5a's and B2a's 64 x 1 ms.
BANDS = {
    "gps_l5": (
        "l5_search",
        "l5_track",
        "l5_combine",
        204600,
        "G",
        "GPS_L5_CNAV",
        "l5i_symbols",
        "misc",
        20.0,
    ),
    "gal_e5a": (
        "e5a_search",
        "e5a_track",
        "e5a_combine",
        1023000,
        "E",
        "GAL_E5AI_FNAV",
        "e5ai_symbols",
        "fnav",
        8.0,
    ),
    "bds_b2a": (
        "b2a_search",
        "b2a_track",
        "b2a_combine",
        1023000,
        "C",
        "BDS_B2A_BCNAV2",
        "b2ad_symbols",
        "bcnav2",
        8.0,
    ),
}


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
    """One band's tracking with no model: blind hints, seeds from detections, held on the
    combiner."""

    def __init__(self, band, rows, lock=5.0):
        self.band = band
        (
            self.search,
            self.track,
            self.combine,
            self.code_chips,
            self.sys,
            _,
            _,
            _,
            self.snr,
        ) = BANDS[band]
        self.lock = lock
        self.st = {}
        self.rows = rows
        self.prns = get("/config")[self.search]["prns"]
        self.t_hint = 0.0

    def step(self, frame0):
        """One pass: seed new detections, nudge and log the tracked, drop the lost."""
        # Blind hints for every untracked slot, renewed inside the search's hint TTL (30 s): a
        # blind L5 pass takes ~60 s, and hints posted once lapsed after the first (11-24).
        if time.time() - self.t_hint > 10.0:
            self.t_hint = time.time()
            post(
                "/%s/set_doppler_hints" % self.search,
                [
                    {"prn": p, "doppler_hz": 0.0, "margin_hz": 5000.0}
                    for p in self.prns
                    if p not in self.st
                ],
            )
        for d in get("/%s/get_detections" % self.search):
            if d["snr"] < self.snr or d["prn"] in self.st:
                continue
            if d.get("long_code_chips", self.code_chips) != self.code_chips:
                continue  # a primary-only phase: the long code's period is unknown
            if d.get("code_phase_at_ref_chips", -1.0) >= 0.0:
                ph = Decimal(d["code_phase_at_ref_chips"])
            else:
                # the argument is referred to sample 0 with its own Doppler; lift it to the
                # phase at its reference hop, at the hop's last sample as the bank does
                n = Decimal(d["ref_hop"]) * NFFT + NFFT - 1
                ph = (
                    Decimal(d["code_phase_chips"])
                    + n * chips_per_hop(d["doppler_hz"]) / NFFT
                ) % self.code_chips
            self.st[d["prn"]] = dict(
                dop=d["doppler_hz"], ph=ph, ref=d["ref_hop"], miss=0
            )
            post(
                "/%s/set_doppler_hints" % self.search,
                [{"prn": d["prn"], "doppler_hz": 0.0, "margin_hz": -1.0}],
            )
            log(
                "%s PRN %d seeded: snr %.0f, %+.0f Hz"
                % (self.band, d["prn"], d["snr"], d["doppler_hz"])
            )
        rows = get("/%s/get_status" % self.combine)
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
                ) % self.code_chips
                s["ref"] = hop
                s["ph"] = (s["ph"] - Decimal(0.16 * cs["dll_disc"])) % self.code_chips
                s["dop"] -= 0.5 * cs["carrier_hz_resid"]
                s["miss"] = 0
                row = {
                    "t": frame0 + hop / HOPS_PER_S,
                    "band": self.band,
                    "sys": self.sys,
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
                "/%s/set_doppler_hints" % self.search,
                [{"prn": prn, "doppler_hz": self.st[prn]["dop"], "margin_hz": 500.0}],
            )
            del self.st[prn]
            log("%s PRN %d lost" % (self.band, prn))
        if seeds:
            post("/%s/set_seeds" % self.track, seeds)
        return sum(1 for s in self.st.values() if s["miss"] == 0)


def start_decoder(here, run_dir, eph_path, marks_path, bands):
    """nav_decode.py on the bands' messages into files that start empty, with no freshness gate
    against the host clock and no BRDC, reporting every 15 s."""
    for p in (eph_path, marks_path):
        if os.path.exists(p):
            os.remove(p)
    sources = []
    for band in bands:
        sources += ["--source", "%s=%s/%s" % (BANDS[band][5], KOTEKAN, BANDS[band][6])]
    return subprocess.Popen(
        [sys.executable, "-u", "nav_decode.py"]
        + sources
        + ["--eph-out", eph_path, "--timemarks-file", marks_path]
        + ["--eph-fresh-s", "0", "--no-brdc", "--report-every", "15"],
        cwd=here,
        stdout=open(run_dir + "/bootstrap_nav_decode.log", "a"),
        stderr=subprocess.STDOUT,
    )


def decoded_ready(eph_path, marks_path, t_start):
    """(clocked satellites, marks' satellites, has a week number), from files written since
    t_start. A clock is af0 (CNAV, F/NAV) or a0 (B-CNAV2)."""
    n_clk = n_marks = 0
    week = False
    try:
        if os.path.getmtime(eph_path) >= t_start:
            n_clk = sum(
                "af0" in r["eph"] or "a0" in r["eph"]
                for r in json.load(open(eph_path))["eph"]
            )
        if os.path.getmtime(marks_path) >= t_start:
            marks = json.load(open(marks_path))["marks"]
            n_marks = len({(m["signal"], m["prn"]) for m in marks})
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
    ap.add_argument(
        "--bands",
        default="gps_l5,gal_e5a,bds_b2a",
        help="comma-separated bands to acquire and decode (%s)" % ", ".join(BANDS),
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
        help="after a site move, let the broker relock this long, then refine the site from "
        "its own rows until it settles (gnss_site_track); 0 skips",
    )
    ap.add_argument(
        "--truth-yaml", help="print the fix's offsets from this yaml's site"
    )
    a = ap.parse_args(argv)
    bands = a.bands.split(",")
    for band in bands:
        if band not in BANDS:
            ap.error("unknown band %s" % band)
    run_dir = os.path.dirname(a.rows)
    eph_path = run_dir + "/bootstrap_eph.json"
    marks_path = run_dir + "/bootstrap_marks.json"
    for p in (eph_path, marks_path):
        if os.path.exists(p):
            os.remove(p)
    frame0 = gnss_fix.FRAME0_UTC
    units = ["gnss-nav-decode@%s" % BANDS[band][7] for band in bands]

    open(a.hold, "w").close()
    systemctl("stop", "gnss-broker")
    systemctl("stop", *units)
    here = os.path.dirname(os.path.abspath(__file__))
    t_start = time.time()
    log("broker held; acquiring %s with no date (0 +- 5 kHz on every slot)" % a.bands)
    rows = open(a.rows, "w")
    trackers = [Tracker(band, rows) for band in bands]
    ready = False
    t_log = t_start
    decoder, t_dec, settled, progress = None, 0.0, None, (0, 0)
    while time.time() - t_start < a.max_wait_s:
        time.sleep(1.0)
        n_trk = sum(trk.step(frame0) for trk in trackers)
        # The decoder starts on settled tracks: symbols from seeds still converging leave it an
        # island of bad symbols whose failed cold decodes back off for minutes (11-24: the two
        # strongest satellites never decoded). It restarts when nothing new decodes for
        # DECODER_STALL_S.
        settled = (settled or time.time()) if n_trk >= 3 else None
        n_clk, n_marks, week = decoded_ready(eph_path, marks_path, t_start)
        if decoder is None and settled and time.time() - settled >= 20.0:
            decoder, t_dec = (
                start_decoder(here, run_dir, eph_path, marks_path, bands),
                time.time(),
            )
            log("decoder started on %d tracks" % n_trk)
        if decoder is not None:
            if (n_clk, n_marks) != progress:
                progress, t_dec = (n_clk, n_marks), time.time()
            elif time.time() - t_dec > DECODER_STALL_S:
                decoder.terminate()
                decoder, t_dec = (
                    start_decoder(here, run_dir, eph_path, marks_path, bands),
                    time.time(),
                )
                log("decoder restarted: nothing new for %.0f s" % DECODER_STALL_S)
        if time.time() - t_log >= 30.0:
            t_log = time.time()
            log(
                "%d tracked; decoded: %d satellites with clocks, marks from %d, week %s"
                % (n_trk, n_clk, n_marks, "yes" if week else "not yet")
            )
        long_enough = time.time() - t_start >= a.min_track_s
        if n_clk >= 4 and n_marks >= 2 and week and long_enough:
            ready = True
            break
    rows.close()
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
    systemctl("start", *units)
    os.remove(a.hold)
    systemctl("start", "gnss-broker")
    log("broker released")
    if ok and a.apply and moved > a.site_tol_m and a.refine_after_s > 0:
        log("refining in %.0f s from the broker's own rows" % a.refine_after_s)
        time.sleep(a.refine_after_s)
        argv_ref = ["--mode", "refine", "--yaml", a.yaml]
        # 3 m, not the 10 m a standing refine waits for: this is the one refine the site gets
        argv_ref += ["--move-m", "3", "--until-settled", "4", "--max-s", "1200"]
        gnss_site_track.main(argv_ref)
        if a.truth_yaml:
            truth = np.array(gnss_pvt._llh_to_ecef(*gnss_fix.site_of(a.truth_yaml)))
            site = np.array(gnss_pvt._llh_to_ecef(*gnss_fix.site_of(a.yaml)))
            log("site now %.1f m from the truth" % np.linalg.norm(site - truth))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
