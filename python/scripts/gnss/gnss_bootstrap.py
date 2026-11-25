#!/usr/bin/env python3
"""Bootstrap the bladeRF receiver from the sky alone: no cached orbits, no time, no position.

    python3 gnss_bootstrap.py [--apply] [--bands gps_l5,gal_e5a,bds_b2a,gw_1202] [--max-wait-s 900]
                              [--truth-yaml Y]       # or --bands glo_l2of, GLONASS alone

The broker's model needs all three, so it is held off (the gnss-broker unit's
ConditionPathExists on --hold) while this runs:

1. Acquire with no date, on each of --bands (GPS L5, Galileo E5a, BeiDou B2a, GuoWang, GLONASS
   L2OF; one is enough):
   every search slot hinted at 0 +- 5 kHz (the GPSDO leaves only the satellites' own Doppler),
   each detection seeded, and the tracks held on the combiner's discriminators (code by
   dll_disc, Doppler by carrier_hz_resid), as bladerf/track_live.py does. E5a and B2a
   detections carry their 100 ms phase from the search's nh_sync. Each tracked satellite's
   record is logged each second (--rows), with its measured Doppler: the record's less the
   carrier residual.
2. A decoder of its own (nav_decode.py: L5 CNAV, E5a F/NAV, B2a B-CNAV2, GLONASS strings, in
   place of the gnss-nav-decode units for those bands, into files that start empty, with no freshness gate
   against the host clock and no BRDC) reads those tracks: time marks, and orbits with clocks.
   Wait for a week number and three tracked satellites with orbits and clocks: a LEO pass's
   Doppler fixes the position from as few.
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
import signal
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
HOPS_PER_S = 1920000
DECODER_STALL_S = 240.0
# Each band: its search, tracker and combiner stages; the tracker code's length (chips), chip
# rate and carrier; the constellation; its decoder and the symbols it reads, and the
# gnss-nav-decode unit it stands in for (GuoWang's decode service writes only the broker's
# table, so it runs on); the blind hint's half-width (Hz); a detection's least SNR to seed, on
# the search's own surface (L5: 3 x 20 ms windows, E5a and B2a: 64 x 1 ms, GuoWang: 10 x 2 ms);
# and the combiner's deep SNR that counts as locked. A blind pass searches blind_chunk untracked
# slots, ~20 s of the CPU search; a detection is confirmed by a search rate_max x its age wide.
# GuoWang's deep SNR integrates 200 ms coherently, so a seed off its 250 Hz search grid never
# reaches it: there the per-record amp_snr (1 on noise) over amp_lock lets the loop pull the
# Doppler in first. GLONASS L2OF is FDMA: each slot's search, track and rows are on its own
# carrier, the band's plus the slot's prn_freq_offset_hz in the search's config.
BANDS = {
    "gps_l5": dict(
        search="l5_search",
        track="l5_track",
        combine="l5_combine",
        code_chips=204600,
        chip_hz="10.23e6",
        carrier_hz=1176.45e6,
        sys="G",
        decoder="GPS_L5_CNAV",
        symbols="l5i_symbols",
        unit="misc",
        blind_hz=5000.0,
        blind_chunk=16,
        rate_max=1.0,
        seed_snr=20.0,
        lock=5.0,
        amp_lock=float("inf"),
    ),
    "gal_e5a": dict(
        search="e5a_search",
        track="e5a_track",
        combine="e5a_combine",
        code_chips=1023000,
        chip_hz="10.23e6",
        carrier_hz=1176.45e6,
        sys="E",
        decoder="GAL_E5AI_FNAV",
        symbols="e5ai_symbols",
        unit="fnav",
        blind_hz=5000.0,
        blind_chunk=36,
        rate_max=1.0,
        seed_snr=8.0,
        lock=5.0,
        amp_lock=float("inf"),
    ),
    "bds_b2a": dict(
        search="b2a_search",
        track="b2a_track",
        combine="b2a_combine",
        code_chips=1023000,
        chip_hz="10.23e6",
        carrier_hz=1176.45e6,
        sys="C",
        decoder="BDS_B2A_BCNAV2",
        symbols="b2ad_symbols",
        unit="bcnav2",
        blind_hz=5000.0,
        blind_chunk=31,
        rate_max=1.0,
        seed_snr=8.0,
        lock=5.0,
        amp_lock=float("inf"),
    ),
    # LEO at ~1150 km: Doppler to +-25 kHz, moving up to 140 Hz/s near closest approach. The
    # noise slots' deep SNR reads 6-9, a tracked satellite's 40-170.
    "gw_1202": dict(
        search="gw_search",
        track="gw_track",
        combine="gw_combine",
        code_chips=4092,
        chip_hz="2.046e6",
        carrier_hz=1202.025e6,
        sys="W",
        decoder="GW_1202_NAV",
        symbols="gw_symbols",
        unit=None,
        blind_hz=25000.0,
        blind_chunk=12,
        rate_max=150.0,
        seed_snr=20.0,
        lock=15.0,
        amp_lock=4.0,
    ),
    # MEO: Doppler to +-3.5 kHz at L2. The 20 ms search (meander-tiled code, 3 windows) reads
    # noise to ~10 over this window; the combiner's deep SNR is ~10 on a track, under 2.5 on
    # noise (11-24).
    "glo_l2of": dict(
        search="glo_search",
        track="glo_track",
        combine="glo_combine",
        code_chips=10220,
        chip_hz="0.511e6",
        carrier_hz=1246.0e6,
        sys="R",
        decoder="GLO_L2OF",
        symbols="glo_bits",
        unit="glo",
        blind_hz=4000.0,
        blind_chunk=8,
        rate_max=1.0,
        seed_snr=12.0,
        lock=5.0,
        amp_lock=float("inf"),
    ),
}
# The measured Doppler's slope over this long is the seed's rate (Hz/s); none until it spans
# RATE_MIN_S.
RATE_SPAN_S = 10.0
RATE_MIN_S = 3.0
# A detection is seeded only this fresh. A search pass works on one snapshot, so a detection from
# a long blind pass is up to minutes old, and a seed carried from it at constant Doppler lands
# tens of chips off a LEO satellite (11-24: no GuoWang track held). An older one is searched for
# again, narrowly, in a pass of its own.
FRESH_S = 5.0
ROWS_MIN = 20  # rows a satellite needs before the solve counts it
# A row is logged only with the carrier this close: a GuoWang track kept by its amplitude while
# its carrier slips (setting, or not yet pulled in) measured its Doppler 0.7-1.4 m/s off (11-24).
ROW_RESID_HZ = 3.0
PENDING_S = 60.0  # an unconfirmed detection is dropped after this long
CONFIRM_HZ = 500.0


def get(path):
    return json.load(urllib.request.urlopen(KOTEKAN + path, timeout=5))


def post(path, body):
    r = urllib.request.Request(
        KOTEKAN + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(r, timeout=5).status


def chips_per_hop(dop, chip_hz, carrier_hz):
    return Decimal(chip_hz) / FS * NFFT * (1 + Decimal(dop) / Decimal(carrier_hz))


def log(msg):
    print("[bootstrap %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


class Tracker:
    """One band's tracking with no model: blind hints, seeds from detections, held on the
    combiner."""

    def __init__(self, band, rows, offset_s):
        self.band = band
        self.c = BANDS[band]
        self.st = {}
        # prn -> (Doppler Hz, ref_hop, wall time) of a detection waiting for a fresh one
        self.pending = {}
        self.rows = rows
        # the capture's labels run this far from the host clock (clock_offset.json)
        self.offset_s = offset_s
        cfg = get("/config")[self.c["search"]]
        self.prns = cfg["prns"]
        # each slot's own carrier (FDMA), for its rows' Doppler
        self.carrier = {
            p: self.c["carrier_hz"] + df
            for p, df in zip(self.prns, cfg.get("prn_freq_offset_hz", []))
        }
        self.t_hint = 0.0
        self.chunk = 0
        self.hinted = set()
        self.logged = {}  # (sys, prn) -> rows written

    def post_hints(self, frame0, now):
        """The next pass's hints, which are all a pass searches: narrow ones on the detections
        waiting for a fresh one, else the next blind_chunk untracked slots blind. Every other
        slot is cleared, so a pass stays short."""
        c = self.c
        want = {}
        if self.pending:
            for prn, (dop, ref, _) in self.pending.items():
                age = now - (frame0 + ref / HOPS_PER_S)
                margin = min(c["blind_hz"], CONFIRM_HZ + c["rate_max"] * age)
                want[prn] = (dop, margin)
        else:
            free = [p for p in self.prns if p not in self.st]
            for i in range(min(c["blind_chunk"], len(free))):
                want[free[(self.chunk + i) % len(free)]] = (0.0, c["blind_hz"])
            self.chunk += c["blind_chunk"]
        hints = [
            {"prn": p, "doppler_hz": d, "margin_hz": m} for p, (d, m) in want.items()
        ]
        hints += [
            {"prn": p, "doppler_hz": 0.0, "margin_hz": -1.0}
            for p in self.hinted - set(want)
        ]
        if hints:
            post("/%s/set_doppler_hints" % c["search"], hints)
        self.hinted = set(want)

    def cph(self, dop):
        return chips_per_hop(dop, self.c["chip_hz"], self.c["carrier_hz"])

    def step(self, frame0):
        """One pass: seed new detections, nudge and log the tracked, drop the lost."""
        c, L = self.c, self.c["code_chips"]
        now = time.time() + self.offset_s  # the capture's own label for now
        changed = False
        for d in get("/%s/get_detections" % c["search"]):
            prn = d["prn"]
            if d["snr"] < c["seed_snr"] or prn in self.st:
                continue
            if d.get("long_code_chips", L) != L:
                continue  # a primary-only phase: the long code's period is unknown
            if now - (frame0 + d["ref_hop"] / HOPS_PER_S) > FRESH_S:
                p = self.pending.get(prn)
                if p is None or d["ref_hop"] > p[1]:
                    changed = changed or p is None
                    self.pending[prn] = (d["doppler_hz"], d["ref_hop"], time.time())
                continue
            # A first rate from the detection it confirms, when they are far enough apart for
            # the Doppler grid's error (a quarter step at most) to stay small.
            rate, p = 0.0, self.pending.pop(prn, None)
            if p is not None and (d["ref_hop"] - p[1]) / HOPS_PER_S >= 10.0:
                rate = (d["doppler_hz"] - p[0]) / ((d["ref_hop"] - p[1]) / HOPS_PER_S)
                rate = max(-c["rate_max"], min(c["rate_max"], rate))
            changed = True
            if d.get("code_phase_at_ref_chips", -1.0) >= 0.0:
                ph = Decimal(d["code_phase_at_ref_chips"])
            else:
                # the argument is referred to sample 0 with its own Doppler; lift it to the
                # phase at its reference hop, at the hop's last sample as the bank does
                n = Decimal(d["ref_hop"]) * NFFT + NFFT - 1
                ph = (
                    Decimal(d["code_phase_chips"])
                    + n * self.cph(d["doppler_hz"]) / NFFT
                ) % L
            self.st[prn] = dict(
                dop=d["doppler_hz"], rate=rate, hist=[], ph=ph, ref=d["ref_hop"], miss=0
            )
            log(
                "%s PRN %d seeded: snr %.0f, %+.0f Hz, %+.1f Hz/s"
                % (self.band, prn, d["snr"], d["doppler_hz"], rate)
            )
        rows = get("/%s/get_status" % c["combine"])
        rows = rows if isinstance(rows, list) else rows.get("prns", [])
        status = {r["prn"]: r for r in rows if isinstance(r, dict) and r.get("prn")}
        seeds = []
        for prn, s in sorted(self.st.items()):
            cs = status.get(prn, {})
            hop = cs.get("pow_hop", 0)
            # The Doppler and the code come from each record, not the deep sum, so a GuoWang
            # track whose deep lock comes and goes is held and measured all the while.
            present = (
                cs.get("deep_snr", 0.0) >= c["lock"]
                or cs.get("amp_snr", 0.0) >= c["amp_lock"]
            )
            if present and hop > s["ref"]:
                # carry the seed to the combiner's newest hop at its mean Doppler over the step,
                # then correct it there
                dt = (hop - s["ref"]) / HOPS_PER_S
                s["ph"] = (
                    s["ph"]
                    + self.cph(s["dop"] + 0.5 * s["rate"] * dt) * (hop - s["ref"])
                ) % L
                s["dop"] += s["rate"] * dt
                s["ref"] = hop
                s["ph"] = (s["ph"] - Decimal(0.16 * cs["dll_disc"])) % L
                s["dop"] -= 0.5 * cs["carrier_hz_resid"]
                s["miss"] = 0
                dop_meas = cs["doppler_hz"] - cs["carrier_hz_resid"]
                # The rate is the measured Doppler's slope over the last RATE_SPAN_S: GuoWang's
                # moves up to 140 Hz/s, and the tracker holds a seed's Doppler between posts.
                h = [x for x in s["hist"] if hop - x[0] <= RATE_SPAN_S * HOPS_PER_S]
                s["hist"] = h + [(hop, dop_meas)]
                if (hop - s["hist"][0][0]) / HOPS_PER_S >= RATE_MIN_S:
                    t = np.array([(x[0] - hop) / HOPS_PER_S for x in s["hist"]])
                    s["rate"] = float(np.polyfit(t, [x[1] for x in s["hist"]], 1)[0])
                if abs(cs["carrier_hz_resid"]) <= ROW_RESID_HZ:
                    row = {
                        "t": frame0 + hop / HOPS_PER_S,
                        "band": self.band,
                        "sys": c["sys"],
                        "prn": prn,
                        "carrier_hz": self.carrier.get(prn, c["carrier_hz"]),
                        "rec_hop": hop,
                        "cp_rec_chips": cs["code_phase_chips"],
                        "dop_rec_hz": cs["doppler_hz"],
                        "dop_meas_hz": dop_meas,
                        "fleet_present": True,
                    }
                    self.rows.write(json.dumps(row) + "\n")
                    key = (c["sys"], prn)
                    self.logged[key] = self.logged.get(key, 0) + 1
            else:
                s["miss"] += 1
            seeds.append(
                {
                    "prn": prn,
                    "doppler_hz": s["dop"],
                    "doppler_rate_hz_s": s["rate"],
                    "code_phase_chips": 0.0,
                    "ref_hop": s["ref"],
                    "code_phase_at_ref_chips": float(s["ph"]),
                }
            )
        self.rows.flush()
        for prn in [p for p, s in self.st.items() if s["miss"] > 15]:
            # searched for again near where it was lost
            self.pending[prn] = (self.st[prn]["dop"], self.st[prn]["ref"], time.time())
            del self.st[prn]
            changed = True
            log("%s PRN %d lost" % (self.band, prn))
        for prn in [
            p for p, v in self.pending.items() if time.time() - v[2] > PENDING_S
        ]:
            del self.pending[prn]
            changed = True
        # New hints when the waiting set changes, and every 20 s otherwise: within the hints'
        # 30 s TTL, and about one blind pass, so each pass takes the next chunk.
        if changed or time.time() - self.t_hint > 20.0:
            self.t_hint = time.time()
            self.post_hints(frame0, now)
        if seeds:
            post("/%s/set_seeds" % c["track"], seeds)
        return sum(1 for s in self.st.values() if s["miss"] == 0)


def start_decoder(here, run_dir, eph_path, marks_path, bands):
    """nav_decode.py on the bands' messages into files that start empty, with no freshness gate
    against the host clock and no BRDC, reporting every 15 s."""
    for p in (eph_path, marks_path):
        if os.path.exists(p):
            os.remove(p)
    sources = []
    for band in bands:
        c = BANDS[band]
        sources += ["--source", "%s=%s/%s" % (c["decoder"], KOTEKAN, c["symbols"])]
    return subprocess.Popen(
        [sys.executable, "-u", "nav_decode.py"]
        + sources
        + ["--eph-out", eph_path, "--timemarks-file", marks_path]
        # and the receiver's cache, so the broker has these orbits when it is released
        + ["--eph-cache", "--eph-cache-name", "bootstrap"]
        + ["--eph-fresh-s", "0", "--no-brdc", "--report-every", "15"],
        cwd=here,
        stdout=open(run_dir + "/bootstrap_nav_decode.log", "a"),
        stderr=subprocess.STDOUT,
    )


def decoded_ready(eph_path, marks_path, t_start, logged):
    """(satellites with a clocked ephemeris and rows logged, marks' satellites, has a week
    number), from files written since t_start; logged is {(sys, prn): rows}. A clock is af0
    (CNAV, F/NAV, GLONASS, GuoWang) or a0 (B-CNAV2). A satellite decoded but not tracked long enough
    to log rows does not count: in a GuoWang run, two of four did not (11-24)."""
    n_clk = n_marks = 0
    week = False
    try:
        if os.path.getmtime(eph_path) >= t_start:
            n_clk = sum(
                ("af0" in r["eph"] or "a0" in r["eph"])
                and logged.get((r["sys"], r["prn"]), 0) >= ROWS_MIN
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


def _sigterm(signum, frame):
    """SIGTERM (gnss-bootstrap-watch stopped mid-run) unwinds like an error, so the broker is
    released."""
    raise SystemExit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--apply", action="store_true", help="correct clock and site, restart"
    )
    ap.add_argument(
        "--bands",
        default="gps_l5,gal_e5a,bds_b2a,gw_1202",
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
    units = ["gnss-nav-decode@%s" % BANDS[b]["unit"] for b in bands if BANDS[b]["unit"]]

    signal.signal(signal.SIGTERM, _sigterm)
    open(a.hold, "w").close()
    systemctl("stop", "gnss-broker")
    if units:
        systemctl("stop", *units)
    # Whatever happens below, the decoder stops and the broker and decoders come back.
    decoder, ok, moved = None, False, 0.0
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        t_start = time.time()
        log("broker held; acquiring %s with no date (blind on every slot)" % a.bands)
        rows = open(a.rows, "w")
        offset_s = gnss_set_clock.read_offset(gnss_set_clock.FILE)
        trackers = [Tracker(band, rows, offset_s) for band in bands]
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
            settled = (settled or time.time()) if n_trk >= 2 else None
            logged = {k: v for trk in trackers for k, v in trk.logged.items()}
            n_clk, n_marks, week = decoded_ready(eph_path, marks_path, t_start, logged)
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
            if n_clk >= 3 and n_marks >= 2 and week and long_enough:
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
    finally:
        if decoder is not None:
            decoder.terminate()
        if units:
            systemctl("start", *units)
        if os.path.exists(a.hold):
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
