#!/usr/bin/env python3
"""GNSS observables record: the instrument's primary data product (RINEX in spirit).

One JSONL row per satellite per band per epoch, carrying the two ranging observables and
the geometry needed to interpret them:

    CODE   code_resid_m: the tracker's model-removed code range (src trk), else the
           search's (dr_integ); plus the record's raw (cp_rec_chips, dop_rec_hz, rec_hop)
           triple and dll_disc
    PHASE  adr_cycles and the fleet ADR (fadr_*), each with its arc id
    POWER  cn0_coh_dbhz, cn0_inc_dbhz, cn0_prompt_dbhz, cn0_kcoh_dbhz, sig
    GEOM   az, el, range_m, range_rate_mps, sat_clk_s  (BRDC, evaluated at the epoch)

Deliberately raw: no ionosphere modelled, no arcs levelled, and the tracker's code residual
(code_resid_src trk) keeps the receiver clock; only the search's (dr_integ) has it removed.
Every derived quantity (receiver clock, TEC, DCBs, scintillation indices) is a function of
these rows and is computed offline, where it can be re-derived when the model improves.

Why both observables: code is unambiguous but noisy and multipath-ridden (metres); carrier is
millimetre-precise but ambiguous by a whole number of cycles, and that ambiguity resets at
every arc break. Levelling each carrier arc onto the code within that arc buys carrier
precision with code absoluteness, which is why the arc id matters as much as the phase: a
levelling that spans a slip is a wrong answer with a small formal error.

Run one instance per chain. The geometry-free combination of two bands' rows for the same
satellite is the ionosphere.

Usage: the obs component of scripts/gnss/site/stack_components.sh holds the argv, one writer
per chain.
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_atmos
from gnss_ephemeris import parse_rinex_nav, predict_all, gpst_of_utc, C_LIGHT
from gnss_brdc_supply import cached_brdc

# The F-engine hop: an emit's epoch is frame0 + hop * SAMPLES_PER_HOP / SAMPLE_RATE_HZ.
SAMPLES_PER_HOP = 16384
SAMPLE_RATE_HZ = 3.2e9
# Status field holding the emit's absolute F-engine hop. The epoch is built from this integer,
# never from a UTC difference: two 1.79e9 floats subtract to 2.4e-7 s of resolution, which is
# 2.4 chips (71 m) at 10.23 Mchip/s.
HOP_KEY = "fleet_hop"
# Rows from a live broker are a few seconds old; one further than this from wall clock is
# mis-stamped (row_epoch() covers the re-base case).
MAX_EPOCH_SKEW_S = 60.0
# Re-read /telescope/time0_ns this often and exit 3 when it has moved (see the poll loop).
FRAME0_RECHECK_S = 60.0
# How old the broker's trk or dr_integ residual (trk_resid_age_s, dr_integ_age_s) may be and
# still be used for code_resid_m.
INTEG_MAX_AGE_S = 90.0
# toe validity window for geometry (az/el/range). Wider than predict_all's 4 h default, which
# dead reckoning keeps, so a stale or offline ephemeris still yields az/el through an overnight
# network gap (Keplerian propagation stays sub-degree for many hours). eph_age_s rides every
# row with geometry so precision consumers can still gate on freshness.
EPH_GEOM_WINDOW_S = 21600.0
# Poll period. Rows are written once per combiner emit (deduped on the arc/record counters),
# so polling faster than the emit costs nothing.
POLL_S = 1.0


def _get(url, timeout=3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return None


def cn0_dbhz(row, deep_snr, coherence_s):
    """Coherent C/N0 (dB-Hz): the broker's `cn0_coh_db` whenever the row carries the key, None
    included; only a row without it (an older broker) gets 20log10(deep_snr) - 10log10(T).

    The fallback is not the same estimator. The row's deep_snr is overridden in publish.py by
    the fleet-coherent value or the quadrature fallback (coh_src fleet:N / quad:N), so it mixes
    populations with different normalisations and reads up to ~10 dB hot on quad rows;
    cn0_coh_db is built from the best single instance's deep_snr over that instance's own
    coherent span.

    Coherence is still required: a floored deep is rectification noise. Whether the prompt was
    on the signal at all is a separate question this number cannot answer; see prompt_lock."""
    if "cn0_coh_db" in row:
        return row["cn0_coh_db"]
    if deep_snr and deep_snr > 0 and coherence_s and coherence_s > 0:
        return 20.0 * math.log10(deep_snr) - 10.0 * math.log10(coherence_s)
    return None


def code_resid(r, chip_rate_hz):
    """(code_resid_m, src, clk_in) for a broker row: the tracker's range-form residual
    (trk_range_s, receiver clock left in) while it is at most INTEG_MAX_AGE_S old, else the
    search's (dr_integ_chips, clock removed) under the same age bar, else (None, None, False).
    clk_in is True only for the tracker's form."""
    ts = r.get("trk_range_s")
    ta = r.get("trk_resid_age_s")
    if ts is not None and (ta is None or ta <= INTEG_MAX_AGE_S):
        return float(ts) * C_LIGHT, "trk", True
    ic = r.get("dr_integ_chips")
    ia = r.get("dr_integ_age_s")
    if ic is not None and (ia is None or ia <= INTEG_MAX_AGE_S):
        return float(ic) * C_LIGHT / chip_rate_hz, "dr_integ", False
    return None, None, False


def cn0_inc_dbhz(amp, unbiased, t_rec):
    """Incoherent C/N0 (dB-Hz) from the single-record power ratio x = u^2/(a^2 - u^2). Needs
    no coherence, so it survives where the coherent estimator drops out, but its noise split is
    the 4th-moment debias, which a BOC pilot's amplitude modulation fools (reads low by
    ~10log10(t_rec / 1 ms): E1C ~-7, B1C ~-13 dB). Kept as the phase-blind estimator and for
    comparison with commercial receivers; cross-check it against cn0_coh."""
    if amp and unbiased and amp > unbiased > 0 and t_rec > 0:
        x = (unbiased * unbiased) / (amp * amp - unbiased * unbiased)
        return 10.0 * math.log10(x / t_rec)
    return None


def row_epoch(r, frame0, now):
    """(t_epoch, hop, None) for a row this writer may record, else (None, hop, why).

    The epoch is frame0 + hop * SAMPLES_PER_HOP / SAMPLE_RATE_HZ, from an integer on the
    F-engine counter. A missing or non-positive hop (the comb's -1 = "no window") means no row:
    an unanchored row is junk, the same rule as t_epoch < 1e9 below.

    The wall-clock gate (MAX_EPOCH_SKEW_S): this writer polls a live broker, so a row's epoch
    is seconds old, and one that is not is mis-stamped rather than late. Right after an
    F-engine re-base the broker's fleet hop can still come from the previous session's
    windows, and new frame0 + old hop lands hours or days in the future; since files roll on
    the row's epoch, those rows would create future days' files.
    """
    hop = r.get(HOP_KEY) or r.get("pow_hop") or 0
    if not hop or hop < 0:
        return None, hop, "no-hop"
    t_epoch = frame0 + hop * SAMPLES_PER_HOP / SAMPLE_RATE_HZ
    if not (t_epoch > 1.0e9):
        return None, hop, "no-anchor"
    if abs(t_epoch - now) > MAX_EPOCH_SKEW_S:
        return None, hop, "skew"
    return t_epoch, hop, None


def make_obs_writer(path_tmpl, label="", log=None):
    """Append-only JSONL row writer whose file rolls on the row's own UTC.

    Returns f(t_utc, row) -> the path written, reopening whenever the strftime-expanded
    path changes, so one long-lived writer produces one file per UTC day.

    The key is the row's epoch, not wall clock at poll. The two differ either side of
    midnight, and the epoch (the emit's capture time) is what every consumer joins on, so
    filing by it is what makes "the rows in <day>.jsonl are that day's measurements" true.
    A row is always written: an epoch outside today means a differently named file, never a
    dropped row.
    """
    state = {"path": None, "fh": None}

    def _write(t_utc, row):
        path = time.strftime(path_tmpl, time.gmtime(t_utc))
        if path != state["path"]:
            if state["fh"] is not None:
                state["fh"].close()
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            # Line-buffered: a reader tailing the record sees whole rows, and a killed
            # logger loses at most the row it was mid-write on.
            state["fh"] = open(path, "a", buffering=1)
            state["path"] = path
            (log or (lambda m: print(m, file=sys.stderr)))(
                "gnss_observables: %s -> %s" % (label, path)
            )
        state["fh"].write(json.dumps(row, separators=(",", ":")) + "\n")
        return path

    def _close():
        """Only a test or a replay ever ends; the logger itself runs until killed."""
        if state["fh"] is not None:
            state["fh"].close()
            state["fh"], state["path"] = None, None

    _write.close = _close
    return _write


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", required=True)
    ap.add_argument("--combiner", required=True)
    ap.add_argument("--search", required=True)
    ap.add_argument(
        "--sys",
        default="G",
        choices=("G", "E", "C", "R"),
        help="RINEX constellation letter of the chain's satellites: selects their BRDC "
        "records for the geometry and is written as each row's sys",
    )
    ap.add_argument(
        "--band",
        required=True,
        help="the chain's band name (e.g. gps_l5, gal_e6): written as each row's band, "
        "which readers group rows by, and used in log labels; the frequency itself is "
        "--carrier-hz",
    )
    ap.add_argument("--carrier-hz", type=float, required=True)
    ap.add_argument("--chip-rate-hz", type=float, required=True)
    ap.add_argument("--code-length", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--alt", type=float, required=True)
    ap.add_argument(
        "--out",
        required=True,
        help="JSONL obs-log path, appended to. strftime fields (%%Y%%m%%d ...) are "
        "expanded per row against that row's own epoch in UTC, so the file rolls at "
        "UTC midnight without a restart and every row lands in the file of the day "
        "it was measured.",
    )
    # The anchor: the F-engine's sample-0 epoch, served exactly (nanoseconds, identical on
    # every node) at /telescope/time0_ns.
    ap.add_argument(
        "--frame0-url",
        required=True,
        help="node REST base (e.g. http://cx43:12048) serving /telescope/time0_ns, "
        "the UTC of F-engine sample 0: the absolute-time anchor of every row. "
        "Waited for at startup and re-read every 60 s; a changed value exits 3 so "
        "the supervisor restarts the writer on the new epoch.",
    )
    args = ap.parse_args()

    # Fatal rather than 0.0: a missing anchor is not a degraded mode, it is every code
    # residual silently null. A node that does not answer (the F-engine off) is waited for,
    # not exited on: the unit's Restart=on-failure would turn the exit into a 10 s re-import
    # loop from NFS.
    _t0, _wait_warn_t = None, 0.0
    while not _t0:
        try:
            with urllib.request.urlopen(
                args.frame0_url.rstrip("/") + "/telescope/time0_ns", timeout=10
            ) as _r:
                _t0 = json.loads(_r.read().decode()).get("time0_ns")
        except OSError as _e:
            if time.time() - _wait_warn_t > 600.0:
                _wait_warn_t = time.time()
                print(
                    "gnss_observables: time0_ns unreadable at %s (%s); waiting for "
                    "the anchor" % (args.frame0_url, _e),
                    file=sys.stderr,
                )
            time.sleep(60.0)
            continue
        if not _t0:
            raise SystemExit(
                "no time0_ns at %s -- the code residual needs the F-engine "
                "sample-0 epoch; refusing to write null residuals" % args.frame0_url
            )
    frame0 = float(_t0) * 1e-9
    print(
        "gnss_observables: F-engine sample 0 at %.9f (%s)" % (frame0, args.frame0_url),
        file=sys.stderr,
    )
    t_rec = args.code_length / args.chip_rate_hz
    lam = C_LIGHT / args.carrier_hz  # carrier wavelength (m/cycle)

    eph, eph_t, eph_probe_t = None, 0.0, 0.0
    _iono = None  # Klobuchar coefficients beside eph (gnss_atmos.iono_params)
    frame0_check_t, frame0_warn_t = time.time(), 0.0
    last = {}  # prn -> (adr_arc, adr_records) of the last row written (emit dedup)
    dropped, dropped_t = (
        {},
        time.time(),
    )  # rows refused by row_epoch(), reported once a minute
    write_row = make_obs_writer(
        args.out, "%s [%s/%s]" % (args.band, args.sys, args.combiner)
    )
    # The first file is not opened until the first row, so announce the pattern here: a
    # chain that never emits would otherwise leave an empty log, with no way to tell
    # "wrong flags" from "nothing to record".
    print(
        "gnss_observables: %s [%s/%s] -> %s (rolls at UTC midnight, on the row's epoch)"
        % (args.band, args.sys, args.combiner, args.out),
        file=sys.stderr,
    )

    while True:
        t0 = time.time()
        if dropped and t0 - dropped_t >= 60.0:
            print(
                "gnss_observables: %s [%s/%s] dropped in the last %.0f s: %s%s"
                % (
                    args.band,
                    args.sys,
                    args.combiner,
                    t0 - dropped_t,
                    ", ".join(
                        "%s %d" % (k, v)
                        for k, v in sorted(dropped.items())
                        if not k.startswith("_")
                    ),
                    (
                        " (largest skew %+.0f s: a stale hop, e.g. the previous F-engine "
                        "session's right after a re-base)" % dropped["_skew_max"]
                    )
                    if "_skew_max" in dropped
                    else "",
                ),
                file=sys.stderr,
            )
            dropped, dropped_t = {}, t0
        status = _get("%s/%s/get_status" % (args.url, args.combiner))
        dets = _get("%s/%s/get_detections" % (args.url, args.search)) or []
        det_snr = {int(d["prn"]): d.get("snr") for d in dets if "prn" in d}
        now = time.time()
        # The anchor is re-read, and a change is fatal on purpose. frame0 is latched once per
        # process, and an F-engine re-base restarts the hop counter: t_epoch = frame0 + hop
        # would then stamp the new session onto the old epoch, filing rows under a past day
        # with az/el/range evaluated at that past instant. The broker never re-anchors either
        # (gnss_broker/receiver.time_anchor) and exits on a confirmed change. Exit 3 is a
        # failure to systemd, which restarts the unit on the new epoch. An unreadable endpoint
        # keeps the latched anchor and is logged; only a different value is fatal.
        if now - frame0_check_t >= FRAME0_RECHECK_S:
            frame0_check_t = now
            try:
                with urllib.request.urlopen(
                    args.frame0_url.rstrip("/") + "/telescope/time0_ns", timeout=3
                ) as _r:
                    _t0 = json.loads(_r.read().decode()).get("time0_ns")
                if _t0 and abs(float(_t0) * 1e-9 - frame0) > 1e-6:
                    print(
                        "gnss_observables: F-engine sample-0 epoch moved %.9f -> %.9f "
                        "(%+.3f h). Every further row would be stamped on the old epoch; "
                        "exiting 3 for the supervisor to restart this writer on the new one."
                        % (
                            frame0,
                            float(_t0) * 1e-9,
                            (float(_t0) * 1e-9 - frame0) / 3600.0,
                        ),
                        file=sys.stderr,
                    )
                    sys.stderr.flush()
                    sys.exit(3)
            except SystemExit:
                raise
            except Exception as _e:
                if now - frame0_warn_t > 600.0:
                    frame0_warn_t = now
                    print(
                        "gnss_observables: time0_ns unreadable at %s (%s); keeping the "
                        "latched anchor" % (args.frame0_url, _e),
                        file=sys.stderr,
                    )

        # Stale-geometry probe: a loaded eph whose newest toe has aged past the window yields
        # no satellites, so az/el/range silently vanish until the 2 h reload. Probe every
        # 5 min; an empty predict for this constellation forces a reload now.
        stale_geom = False
        if eph is not None and now - eph_probe_t > 300.0:
            eph_probe_t = now
            try:
                probe = predict_all(
                    eph,
                    args.lat,
                    args.lon,
                    args.alt,
                    datetime.fromtimestamp(now, tz=timezone.utc),
                    mask_deg=-90.0,
                    max_age=EPH_GEOM_WINDOW_S,
                )
                stale_geom = not any(k[0] == args.sys for k in probe)
            except Exception:
                stale_geom = True
        if eph is None or now - eph_t > 7200 or stale_geom:
            try:
                # Read-only (gnss_brdc_supply.cached_brdc): the broker is the nav cache's one
                # writer. An empty cache omits geometry; it never triggers a fetch.
                srcs = cached_brdc()
                if not srcs:
                    raise RuntimeError(
                        "nav cache empty -- the broker has not fetched yet"
                    )
                eph = parse_rinex_nav(srcs)
                # Klobuchar coefficients beside the ephemeris (gnss_atmos): the row's iono_m
                _iono = gnss_atmos.iono_params(srcs, write=False)
                eph_t = now
                if stale_geom:
                    print(
                        "BRDC geometry was stale for %s; reloaded the nav cache"
                        % args.sys,
                        file=sys.stderr,
                    )
            except Exception as e:
                print("BRDC unavailable (%s); geometry omitted" % e, file=sys.stderr)
                eph_t = now - 7200 + 600

        if isinstance(status, list):
            for r in status:
                prn = r.get("prn")
                if not prn:
                    continue
                prn = int(prn)
                arc = r.get("adr_arc") or 0
                nrec = r.get("adr_records") or 0
                sig = (
                    (max(r.get("deep_snr") or 0.0, r.get("amp_snr") or 0.0))
                    if (r.get("coherence_s") or 0) > 0
                    else (r.get("amp_snr") or 0.0)
                )
                if sig <= 0 and not (r.get("amplitude") or 0):
                    continue  # PRN not despread this emit: a gap, not a row
                if last.get(prn) == (arc, nrec):
                    continue  # same combiner emit, already recorded
                last[prn] = (arc, nrec)
                # Epoch = the emit's own capture time. Wall clock at poll is a different
                # instant: 0.1 s of pipeline latency and emit jitter smears a 2 kHz Doppler by
                # ~200 cycles, and every geometry term is evaluated at the epoch, so the error
                # would land in the ionosphere estimate. The hop count is an integer off the
                # F-engine counter, never (t_now - utc0) on two 1.79e9 floats. No hop, or an
                # epoch far from wall clock, is no row (row_epoch says why).
                t_epoch, hop, why = row_epoch(r, frame0, now)
                if why is not None:
                    dropped[why] = dropped.get(why, 0) + 1
                    if why == "skew":
                        _sk = frame0 + hop * SAMPLES_PER_HOP / SAMPLE_RATE_HZ - now
                        if abs(_sk) > abs(dropped.get("_skew_max", 0.0)):
                            dropped["_skew_max"] = _sk
                    last.pop(prn, None)  # never dedup a later good row against this one
                    continue
                v = None
                # While the broker says the epoch is suspect (gnss_broker/timebase) the row
                # carries no geometry: evaluated at a wrong instant it is not degraded geometry
                # but another position wearing this PRN's number. The row's own measurements
                # are kept.
                epoch_suspect = bool(r.get("time_base_suspect"))
                # Hand predict_all this PRN's records only: it evaluates every satellite it is
                # given, and the whole store (every constellation over the cached days) would
                # cost a full-sky solve per call, two calls per row.
                _k1 = (args.sys, prn)
                _eph1 = {_k1: eph[_k1]} if (eph and _k1 in eph) else {}
                if eph and not epoch_suspect:
                    try:
                        v = predict_all(
                            _eph1,
                            args.lat,
                            args.lon,
                            args.alt,
                            datetime.fromtimestamp(t_epoch, tz=timezone.utc),
                            mask_deg=-90.0,
                            max_age=EPH_GEOM_WINDOW_S,
                            tropo=True,
                            iono=(
                                (_iono["alpha"], _iono["beta"], args.carrier_hz)
                                if _iono
                                else None
                            ),
                        ).get((args.sys, prn))
                    except Exception:
                        v = None
                adr = r.get("adr_cycles")
                trim = r.get("trim_cycles")
                # CMC inputs. code_resid = c*dt_rx + I + multipath (range form). Pair it with
                # the carrier residual
                #     carr_resid = -adr*lambda - range   (our ADR counts +dop, so it measures
                #                                         minus range)
                # and CMC = code_resid - carr_resid = 2I + code multipath + a per-arc constant,
                # the deterministic multipath metric. Computed here because the fleet-ADR form
                # needs the model range at fadr_hop, which the row does not carry.
                code_resid_m = None
                carr_resid_m = None
                code_resid_src = None
                carr_resid_src = None
                _clk_in = False
                if v is not None and adr is not None:
                    # Prefer the tracker's residual over the search's. dr_integ is the search
                    # detection's code phase against the model, clock removed: the acquisition
                    # estimate, ~0.5 chips of white noise per satellite. The tracker residual
                    # is the closed fleet DLL's replica placement against the same model
                    # (gnss_broker/trkresid), resolves ~1e-2 chips, and exists on every chain,
                    # detectors or not. It is taken in seconds because the broker's chips are
                    # its own currency (CM chips on L2C). The range form (trk_range_s, receiver
                    # clock left in) is used so that code minus the carrier residual below,
                    # which carries the same clock, is clock-free.
                    code_resid_m, code_resid_src, _clk_in = code_resid(
                        r, args.chip_rate_hz
                    )
                    carr_resid_m = -adr * lam - v["range_m"]
                    carr_resid_src = "adr"
                    # The fleet ADR at its own hop (gnss_broker/fleetadr). adr_cycles above is
                    # one instance's accumulator with its epoch hidden, and a carrier phase
                    # moves ~1e9 cycles/s, so a residual built from it at the row's epoch is
                    # off by c*dt. fadr_dop_cycles is the Doppler-only phase at fadr_hop and
                    # the model range is evaluated at exactly that hop, so the residual is the
                    # model-removed carrier range on this arc, up to the arc's constant.
                    _fh = r.get("fadr_hop")
                    _fd = r.get("fadr_dop_cycles")
                    if _fh and _fd is not None and eph:
                        try:
                            _ta = frame0 + _fh * SAMPLES_PER_HOP / SAMPLE_RATE_HZ
                            _va = predict_all(
                                _eph1,
                                args.lat,
                                args.lon,
                                args.alt,
                                datetime.fromtimestamp(_ta, tz=timezone.utc),
                                mask_deg=-90.0,
                                max_age=EPH_GEOM_WINDOW_S,
                            ).get((args.sys, prn))
                        except Exception:
                            _va = None
                        if _va is not None:
                            carr_resid_m = -_fd * lam - _va["range_m"]
                            carr_resid_src = "fadr"
                row = {
                    "t": round(t_epoch, 4),
                    "t_gps": round(gpst_of_utc(t_epoch), 4),
                    "sys": args.sys,
                    "prn": prn,
                    "band": args.band,
                    "carrier_hz": args.carrier_hz,
                    # Code: commanded code phase and the DLL's unapplied sub-chip residual, kept
                    # separate: the trim is a loop state, the discriminator a measurement, and
                    # offline decides how much of it to believe.
                    "cp_chips": r.get("code_phase_chips"),
                    "dll_disc": r.get("dll_disc"),
                    "proj_b": r.get("proj_b"),
                    "proj_k": r.get("proj_k"),
                    "proj_src": r.get("proj_src"),
                    "code_len": args.code_length,
                    "chip_rate_hz": args.chip_rate_hz,
                    # Phase: cycles on this arc (0 at an arc start). Ambiguous by an integer;
                    # the arc id says when the ambiguity changed.
                    "adr_cycles": adr,
                    "trim_cycles": trim,
                    "adr_m": (adr * lam) if adr is not None else None,
                    "adr_arc": arc,
                    "adr_records": nrec,
                    "code_resid_m": code_resid_m,  # model-removed code range (CMC input)
                    "code_resid_src": code_resid_src,  # trk | dr_integ
                    "code_resid_clk": _clk_in,  # True for trk (range form, clock left in)
                    # the tracker residual's own scatter and support (metres, records)
                    "code_resid_sd_m": (
                        (r.get("trk_resid_sd_chips") or 0.0)
                        * C_LIGHT
                        / args.chip_rate_hz
                        if code_resid_src == "trk"
                        else None
                    ),
                    "code_resid_n": (
                        r.get("trk_resid_n") if code_resid_src == "trk" else None
                    ),
                    # the record's own (argument, Doppler, hop) triple, verbatim: the raw code
                    # observable, from which any residual can be re-derived offline against a
                    # better model. The argument is meaningful only with its Doppler and hop.
                    "cp_rec_chips": r.get("cp_rec_chips"),
                    "dop_rec_hz": r.get("dop_rec_hz"),
                    "rec_hop": r.get("rec_hop"),
                    "dll_trim_cpp": r.get("dll_trim_cpp"),
                    "carr_resid_m": carr_resid_m,  # model-removed carrier range (CMC input)
                    "carr_resid_src": carr_resid_src,  # fadr (exact hop) | adr (epoch hidden)
                    # the arc the carrier residual lives on: fleet ADR's when it is the source
                    "carr_arc": (
                        r.get("fadr_arc") if carr_resid_src == "fadr" else arc
                    ),
                    # The fleet ADR, verbatim: Doppler-only cycles and the full phase, both at
                    # fadr_hop, on arc fadr_arc that began at fadr_hop0. Hops are on the one
                    # F-engine axis every chain shares.
                    "fadr_dop_cycles": r.get("fadr_dop_cycles"),
                    "fadr_cycles": r.get("fadr_cycles"),
                    "fadr_hop": r.get("fadr_hop"),
                    "fadr_hop0": r.get("fadr_hop0"),
                    "fadr_arc": r.get("fadr_arc"),
                    "fadr_n_rec": r.get("fadr_n_rec"),
                    "fadr_n_inst": r.get("fadr_n_inst"),
                    "fadr_n_silent": r.get("fadr_n_silent"),
                    "fadr_trim_cycles": r.get("fadr_trim_cycles"),
                    "fadr_res_cycles": r.get("fadr_res_cycles"),
                    # the same ADR at the fleet-wide grid hop, equal across every chain, and
                    # the recent grid hops (fadr_g_hist) so bands pair whatever the poll phase
                    "fadr_g_hop": r.get("fadr_g_hop"),
                    "fadr_g_hist": r.get("fadr_g_hist"),
                    "fadr_g_dop_cycles": r.get("fadr_g_dop_cycles"),
                    "fadr_g_cycles": r.get("fadr_g_cycles"),
                    "adr_lock_s": r.get("adr_lock_s"),
                    # Power
                    "cn0_coh_dbhz": cn0_dbhz(
                        r, r.get("deep_snr"), r.get("coherence_s")
                    ),
                    # Is the prompt tap on the signal? cn0_coh_dbhz is blind to code error:
                    # deep_snr comes from the deep fold, whose ~10x integration gain on the
                    # prompt still reads a satellite the tap has slid off. Carried per row so
                    # offline does not average a blind window into a health number. prompt_lock
                    # False means cn0_coh_dbhz is a deep-fold significance, not a C/N0.
                    "prompt_lock": r.get("prompt_lock"),
                    "prompt_rayleigh": r.get("prompt_rayleigh"),
                    "fleet_present": r.get("fleet_present"),
                    "time_base_suspect": epoch_suspect,  # geometry withheld when true
                    "coh_src": r.get("coh_src"),
                    "cn0_inc_dbhz": cn0_inc_dbhz(
                        r.get("amplitude"), r.get("unbiased_amplitude"), t_rec
                    ),
                    # The served radiometry (combdll.prompt_cn0): per-record prompt power, kept
                    # where the prompt is the tallest tap, debiased against the below-horizon
                    # probes, with no fit in it (cn0_coh's deep fold re-searches a rate per
                    # integration). duty is the fraction of records kept: offline consumers
                    # decline low-duty rows rather than average them. split_db is the
                    # even/odd-record self-consistency. noise_probe marks the below-horizon
                    # noise references riding the same rows (never satellites).
                    "cn0_prompt_dbhz": r.get("cn0_prompt_db"),
                    "cn0_prompt_duty": r.get("cn0_prompt_duty"),
                    "cn0_prompt_split_db": r.get("cn0_prompt_split_db"),
                    "noise_probe": r.get("noise_probe"),
                    # The known-rate coherent fold (combdll.coh_cn0): the radiometry for
                    # deep-sidelobe satellites below the per-record gate, centred on the
                    # previous cycle's record-stream rate (or zero) with a one-parameter
                    # residual-rate fit the probes share. kcoh_sig is its significance over the
                    # probes' identical fold; kcoh_eta the coherence efficiency (n_rec = fully
                    # coherent, ~1 = no coherent gain, mostly None on noise). cn0_kcoh is a
                    # power mean over instances, so on strong satellites it reads below
                    # cn0_prompt by up to 10log10(n_inst); their per-row difference, that loss
                    # plus any decoherence, is the standing cross-check.
                    "cn0_kcoh_dbhz": r.get("cn0_kcoh_db"),
                    "cn0_kcoh_sky_dbhz": r.get("cn0_kcoh_sky_db"),
                    "kcoh_sig": r.get("kcoh_sig"),
                    "kcoh_eta": r.get("kcoh_eta"),
                    "kcoh_rate_hz": r.get("kcoh_rate_hz"),
                    "kcoh_t_coh_s": r.get("kcoh_t_coh_s"),
                    "sig": sig,
                    "coherence_s": r.get("coherence_s"),
                    "search_snr": det_snr.get(prn),
                    "doppler_hz": r.get("doppler_hz"),
                    "carrier_hz_resid": r.get("carrier_hz_resid"),
                    # boundary_f: the code-period boundary's offset inside the despread window.
                    # On an overlay signal |A|^2 nulls at f = 0.5; the combiner's incoherent
                    # power uses |head|^2 + |tail|^2, which does not. Logged so offline can check
                    # that C/N0 does not depend on f.
                    "boundary_f": r.get("boundary_f"),
                    "s4": r.get("s4"),
                    "s4_raw": r.get("s4_raw"),
                    "sigma_phi": r.get("sigma_phi"),
                }
                # Geometry (BRDC at this epoch). Receiver clock not removed: solving it is a
                # downstream job, and a row that has had a model subtracted cannot be
                # un-subtracted when the model improves.
                if v is not None:
                    row.update(
                        {
                            "az": round(v["az"], 3),
                            "el": round(v["el"], 3),
                            "range_m": round(v["range_m"], 3),
                            "range_rate_mps": round(v["range_rate_mps"], 4),
                            "sat_clk_s": v["sat_clk_s"],
                            "eph_age_s": round(v["toe_age_s"], 1),
                            # one-way delays, not subtracted from anything here
                            "tropo_m": round(v.get("tropo_m", 0.0), 3),
                            "iono_m": round(v.get("iono_m", 0.0), 3),
                        }
                    )
                write_row(t_epoch, row)
        # Sleep to the next multiple of POLL_S on the wall clock, so every chain's writer
        # polls on one shared grid. A row's epoch is its emit hop, not the poll time, so the
        # grid sets only when an emit is picked up.
        now = time.time()
        dt = POLL_S - (now % POLL_S)
        time.sleep(dt if dt > 0.05 else dt + POLL_S)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
