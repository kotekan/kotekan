#!/usr/bin/env python3
"""GNSS OBSERVABLES RECORD -- the instrument's primary data product (RINEX in spirit).

One JSONL row per satellite per band per epoch, carrying the two ranging observables and
the geometry needed to interpret them:

    CODE   code_resid_m: the tracker's model-removed code range (src trk), else the
           search's (dr_integ), else a reconstruction (airspy); + the record's raw
           (cp_rec_chips, dop_rec_hz, rec_hop) triple and dll_disc
    PHASE  adr_cycles       (accumulated carrier phase on an unbroken arc, + arc id/length)
    POWER  cn0_dbhz, sig    (coherent + incoherent)
    GEOM   az, el, range_m, sat_clk_s, range_rate  (BRDC, evaluated AT THE EPOCH)

Deliberately RAW: no receiver clock removed, no ionosphere modelled, no arcs levelled. Every
derived quantity (receiver clock, TEC, DCBs, scintillation indices) is a function of these
rows and is computed OFFLINE, where it can be re-derived when the model improves. The record
is the contract; everything downstream is opinion.

WHY BOTH OBSERVABLES: code is unambiguous but noisy and multipath-ridden (metres); carrier is
exquisite (millimetres) but ambiguous by a whole number of cycles, and that ambiguity resets
at every arc break. The standard recipe -- and the one the TEC work will use -- levels each
carrier arc onto the code within that arc, buying carrier precision with code absoluteness.
This is why the arc id matters as much as the phase itself: a levelling that spans a slip is
a wrong answer with a small formal error, the worst kind.

MULTI-BAND: run one instance per band/chain (--combiner gps_combiner, gal_combiner, ...) with
--band naming the SIGNAL (GPS_L1CA, GAL_E1C, ...). The geometry-free combination of two bands'
rows for the same satellite is the ionosphere -- which is the point of the exercise.

Usage:
    python3 gnss_observables.py --combiner gps_combiner --search gps_search \\
        --sys G --band GPS_L1CA --out /tmp/gpswipe/obs_gps_l1.jsonl
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnss_stages import resolve_stage, capture_clock
from gnss_ephemeris import (fetch_brdc, parse_rinex_nav, predict_all, gpst_of_utc, C_LIGHT)


def _get(url, timeout=3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return None


def cn0_dbhz(row, deep_snr, coherence_s):
    """Coherent C/N0 (dB-Hz): PREFER the broker's own `cn0_coh_db`, and only fall back to the
    local 20log10(deep_snr) - 10log10(T) when serving an older broker that does not publish it.

    ⚠️ THE FALLBACK IS THE BUG THIS FUNCTION USED TO BE (task #47, 2026-08-12). Recomputing from
    the ROW's deep_snr silently re-introduced everything task #35 fixed one layer upstream: that
    field is OVERRIDDEN in publish.py by the fleet-coherent value or the quadrature fallback, so
    it mixes estimator populations with different normalisations. publish.py deliberately builds
    cn0_coh_db from the BEST SINGLE INSTANCE's own deep_snr over that instance's own coherent
    span -- one estimator, one normalisation, continuous across serving states -- and this
    function threw that away and recomputed the mixture. MEASURED live on gal_e5a the same day:
    the recomputation runs up to +9.7 dB hot against the published value, worst on exactly the
    PRNs carrying `coh_src: quad:12` (the 10log10(12) = 10.8 dB inflation of docs 11.31).
    So every observables file and every viewer trace written since #35 landed carries the
    UN-fixed number, while the broker's own REST surface carried the fixed one.

    Coherence is still required: a floored deep is rectification noise. Whether the PROMPT was
    on the signal at all is a separate question this number cannot answer -- see prompt_lock."""
    pub = row.get("cn0_coh_db")
    if pub is not None:
        return pub
    if deep_snr and deep_snr > 0 and coherence_s and coherence_s > 0:
        return 20.0 * math.log10(deep_snr) - 10.0 * math.log10(coherence_s)
    return None


def cn0_inc_dbhz(amp, unbiased, t_rec):
    """Incoherent C/N0 (dB-Hz) from the single-record power ratio x = u^2/(a^2 - u^2). Needs
    no coherence at all, so it survives where the coherent estimator drops out -- BUT its noise
    split is the 4th-moment debias, which a BOC pilot's amplitude modulation fools (reads low by
    ~10log10(t_rec): E1C ~-7, B1C ~-13 dB). Kept as the phase-blind estimator + a commercial-Rx
    comparison; cross-check it against cn0_q (modulation-immune) and cn0_coh."""
    if amp and unbiased and amp > unbiased > 0 and t_rec > 0:
        x = (unbiased * unbiased) / (amp * amp - unbiased * unbiased)
        return 10.0 * math.log10(x / t_rec)
    return None


def cn0_q_dbhz(snr_q, t_rec):
    """Modulation-immune C/N0 (dB-Hz): snr_q is the combiner's per-record SNR with the noise taken
    from the QUADRATURE component of the carrier-removed frame, so BOC amplitude modulation does not
    bias it (unlike cn0_inc). Coherence-referenced (the combiner produces it on the LINEAR pilot
    path only, and only while the carrier fit holds), so it is None for data signals / decohered
    windows -- there, fall back to cn0_inc. C/N0 = 10 log10(snr_q / t_rec)."""
    if snr_q is not None and snr_q > 0 and t_rec > 0:
        return 10.0 * math.log10(snr_q / t_rec)
    return None


_warn_dop = [0]


def _phys_chips(cp_arg, comb_mult, hop, t_abs, dop_hz, args):
    """Physical code phase (chips, mod code_length) of the replica at the epoch.

    C(n) = comb_mult*cp_arg + n*cps(dop), n the ABSOLUTE sample index -- the generator's own
    definition, so this is the exact inverse of the argument the tracker was seeded with.
    Done in exact rationals when the hop is available: n*cps is ~4.5e12 chips, and taking the
    modulus of that in float64 leaves ~5e-4 chips (1.6 cm) -- fine, but the rational costs
    nothing at this rate and removes the question entirely.
    """
    scale = 1.0 + args.code_doppler_sign * dop_hz / args.carrier_hz
    if hop:
        n = Fraction(int(hop) * int(args.samples_per_hop))
        chips = (Fraction(comb_mult) * Fraction(cp_arg).limit_denominator(10 ** 12)
                 + n * Fraction(args.chip_rate_hz).limit_denominator(10 ** 9)
                 * Fraction(scale).limit_denominator(10 ** 15)
                 / Fraction(args.sample_rate_hz).limit_denominator(10 ** 9))
        return float(chips % Fraction(int(args.code_length)))
    return (comb_mult * cp_arg + t_abs * args.chip_rate_hz * scale) % args.code_length


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:12048")
    ap.add_argument("--airspy", default="airspy_in", help="airspy stage for the capture-clock anchor (adcstat); band-prefixed in a merged instance")
    ap.add_argument("--combiner", default="gps_combiner")
    ap.add_argument("--search", default="gps_search")
    ap.add_argument("--sys", default="G", choices=("G", "E", "C"),
                    help="RINEX constellation letter (selects the BRDC records)")
    ap.add_argument("--band", default="GPS_L1CA",
                    help="signal name (gnssSignal.hpp): identifies the observable's frequency")
    ap.add_argument("--carrier-hz", type=float, default=1575.42e6)
    ap.add_argument("--chip-rate-hz", type=float, default=1.023e6)
    ap.add_argument("--code-length", type=float, default=1023.0)
    ap.add_argument("--code-doppler-sign", type=float, default=1.0,
                    help="must match the search stage's code_doppler_sign (CMC reconstruction)")
    ap.add_argument("--lat", type=float, default=43.968697)
    ap.add_argument("--lon", type=float, default=-79.252106)
    ap.add_argument("--alt", type=float, default=260.0)
    ap.add_argument("--eph-geom-window-s", type=float, default=21600.0,
                    help="toe validity window for GEOMETRY (az/el/range). Wider than the "
                         "4 h dead-reckon default so a stale-in-memory or offline ephemeris "
                         "still yields az/el through an overnight network gap (Keplerian "
                         "propagation is sub-degree for many hours). toe_age_s is logged per "
                         "row so precision consumers can still gate on freshness.")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="poll period (s); rows are written once per COMBINER EMIT (deduped "
                         "on the arc/record counters), so polling faster than the emit is free")
    ap.add_argument("--out", default="/tmp/gpswipe/observables.jsonl",
                    help="obs-log path. %%Y%%m%%d etc are strftime-expanded and the file ROLLS at "
                         "UTC midnight -- a date baked in at launch keeps one day's name for as "
                         "long as the process lives, and consumers that open today's file "
                         "then find nothing.")
    # ── THE CHORD ANCHOR. The airspy path anchors absolute time on the dongle's
    # adcstat/utc0_sample0; CHORD has no such endpoint, and the broker answers an unknown
    # path with a 200 and a chain summary, so `.get("utc0_sample0") or 0.0` silently yields
    # zero and every code residual comes out null. The F-engine's own sample-0 epoch is
    # served exactly (nanoseconds, identical on every node) at /telescope/time0_ns.
    ap.add_argument("--frame0-url", default="",
                    help="node REST base (e.g. http://cx43:12048) serving /telescope/time0_ns, "
                         "the UTC of F-engine sample 0. Sets the absolute-time anchor for the "
                         "code residual. Without it (or --frame0-utc) this falls back to the "
                         "airspy adcstat anchor.")
    ap.add_argument("--frame0-utc", type=float, default=0.0,
                    help="the same anchor as a literal, for replay.")
    ap.add_argument("--hop-key", default="fleet_hop",
                    help="status field holding the emit's ABSOLUTE F-engine hop. The epoch is "
                         "built from this integer, never from a UTC difference: two 1.79e9 "
                         "floats subtract to 2.4e-7 s of resolution, which is 2.4 chips (73 m) "
                         "of code. Which hop field is used barely matters -- both sides of the "
                         "residual are evaluated at the SAME hop, so the choice cancels to "
                         "first order (5e-5 chips over 12288 hops of Doppler mismatch).")
    ap.add_argument("--integ-max-age-s", type=float, default=90.0,
                    help="how stale the broker's dead-reckon integrity residual may be before "
                         "this falls back to reconstructing one (it refreshes ~30 s).")
    ap.add_argument("--samples-per-hop", type=int, default=16384)
    ap.add_argument("--sample-rate-hz", type=float, default=3.2e9)
    ap.add_argument("--comb-mult", type=int, default=1,
                    help="replica comb multiplier: the generator forms "
                         "C(n) = comb_mult*code_phase_chips + n*cps(doppler). 2 on L2C (CM).")
    args = ap.parse_args()

    args.combiner = resolve_stage(args.url, args.combiner)
    args.search = resolve_stage(args.url, args.search)
    to_unix = capture_clock(args.url, args.airspy)  # capture clock -> unix (band-prefixed stage)
    utc0 = 0.0                          # capture sample-0 UTC (CMC needs the absolute age)
    frame0 = float(args.frame0_utc or 0.0)
    if not frame0 and args.frame0_url:
        # FATAL rather than 0.0: a missing anchor is not a degraded mode, it is every code
        # residual silently null -- which is how this went unnoticed for as long as it did.
        with urllib.request.urlopen(args.frame0_url.rstrip("/") + "/telescope/time0_ns",
                                    timeout=10) as _r:
            _t0 = json.loads(_r.read().decode()).get("time0_ns")
        if not _t0:
            raise SystemExit("no time0_ns at %s -- the code residual needs the F-engine "
                             "sample-0 epoch; refusing to write null residuals" % args.frame0_url)
        frame0 = float(_t0) * 1e-9
        print("gnss_observables: F-engine sample 0 at %.9f (%s)" % (frame0, args.frame0_url),
              file=sys.stderr)
    t_rec = args.code_length / args.chip_rate_hz
    lam = C_LIGHT / args.carrier_hz          # carrier wavelength (m/cycle)

    eph, eph_t, eph_probe_t = None, 0.0, 0.0
    last = {}   # prn -> (adr_arc, adr_records) of the last row written (emit dedup)
    n = 0
    def _outpath(t):
        return time.strftime(args.out, time.gmtime(t))
    out_now = _outpath(time.time())
    f = open(out_now, "a", buffering=1)
    print("gnss_observables: %s [%s/%s] -> %s" % (args.band, args.sys, args.combiner, out_now),
          file=sys.stderr)

    while True:
        t0 = time.time()
        status = _get("%s/%s/get_status" % (args.url, args.combiner))
        dets = _get("%s/%s/get_detections" % (args.url, args.search)) or []
        det_snr = {int(d["prn"]): d.get("snr") for d in dets if "prn" in d}
        now = time.time()
        if to_unix(1.0) == 1.0:         # anchor wasn't up at startup: retry until it is
            to_unix = capture_clock(args.url, args.airspy)
        if not utc0:
            a = _get("%s/%s/adcstat" % (args.url, args.airspy)) or {}
            utc0 = float(a.get("utc0_sample0") or 0.0)

        # STALE-GEOMETRY PROBE (2026-07-20): a loaded eph whose newest toe has aged past the
        # window yields ZERO sats -> az/el/range silently vanish, and the plain 2 h timer
        # would leave it that way for up to 2 h (measured: L1-GPS ran 4 h geometry-blind
        # while fresh BRDC sat in the cache). Probe once every 5 min; an empty predict for
        # THIS constellation forces a re-fetch now.
        stale_geom = False
        if eph is not None and now - eph_probe_t > 300.0:
            eph_probe_t = now
            try:
                probe = predict_all(eph, args.lat, args.lon, args.alt,
                                    datetime.fromtimestamp(now, tz=timezone.utc),
                                    mask_deg=-90.0, max_age=args.eph_geom_window_s)
                stale_geom = not any(k[0] == args.sys for k in probe)
            except Exception:
                stale_geom = True
        if eph is None or now - eph_t > 7200 or stale_geom:
            try:
                eph = parse_rinex_nav(fetch_brdc())
                eph_t = now
                if stale_geom:
                    print("BRDC geometry was stale for %s; re-fetched" % args.sys,
                          file=sys.stderr)
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
                sig = ((max(r.get("deep_snr") or 0.0, r.get("amp_snr") or 0.0))
                       if (r.get("coherence_s") or 0) > 0 else (r.get("amp_snr") or 0.0))
                if sig <= 0 and not (r.get("amplitude") or 0):
                    continue                       # PRN not despread this emit: a gap, not a row
                if last.get(prn) == (arc, nrec):
                    continue                       # same combiner emit, already recorded
                last[prn] = (arc, nrec)
                # EPOCH = the emit's own capture time. Wall-clock-at-poll is a different
                # (and wrong) instant: pipeline latency and emit jitter of 0.1 s smear a
                # 2 kHz Doppler by ~200 cycles, and every geometry term is evaluated at the
                # epoch, so the error would land straight in the ionosphere estimate.
                hop = r.get(args.hop_key) or r.get("pow_hop") or 0
                if frame0 and hop:
                    # EXACT: an integer hop count off the F-engine's own counter, scaled by
                    # two integers. Never (t_now - utc0) on two 1.79e9 floats.
                    t_abs = hop * args.samples_per_hop / args.sample_rate_hz
                    t_epoch = frame0 + t_abs
                else:
                    t_epoch = to_unix(r.get("utc") or 0.0)
                    t_abs = (t_epoch - utc0) if utc0 else 0.0
                if not (t_epoch > 1.0e9):
                    continue                   # no capture anchor yet: an untagged row is junk
                v = None
                if eph:
                    try:
                        v = predict_all(eph, args.lat, args.lon, args.alt,
                                        datetime.fromtimestamp(t_epoch, tz=timezone.utc),
                                        mask_deg=-90.0,
                                        max_age=args.eph_geom_window_s).get((args.sys, prn))
                    except Exception:
                        v = None
                adr = r.get("adr_cycles")
                trim = r.get("trim_cycles")
                # ---- CMC INPUTS. cp_chips alone is NOT a range: it is the seed CURRENCY
                # (cp0), back-referenced to sample 0 through the Doppler, so its geometry is
                # already removed -- it drifts at the receiver clock offset while the carrier
                # drifts at the range rate. Differencing them cancels nothing (measured: CMC
                # in KILOMETRES). Reconstruct the PHYSICAL code phase the same way
                # gnss_deadreckon_check does, then remove the model:
                #     cp_phys  = cp0 + t_abs*f_chip*(1 + sign*dop/f_carrier)   (mod L)
                #     cp_pred  = (t_tx_sv mod T_code)/T_code * L
                #     code_resid = wrap(cp_phys - cp_pred)  -> SUB-CHIP, unambiguous, metres
                # code_resid = c*dt_rx + I + multipath. Pair it with the carrier residual
                #     carr_resid = -adr*lambda - range   (our ADR counts +dop, so it measures
                #                                         MINUS range)
                # and CMC = code_resid - carr_resid = 2I + code multipath + a per-arc constant.
                # THAT is the deterministic multipath metric; it could not be built from the
                # logged fields alone, which is why it is computed here, at the source.
                code_resid_m = None
                carr_resid_m = None
                code_resid_src = None
                carr_resid_src = None
                _clk_in = False
                if v is not None and (utc0 or frame0) and adr is not None:
                    # ⚠️ THE DOPPLER MUST BE THE ONE THE GENERATOR WAS HANDED. code_phase_chips
                    # is an ARGUMENT back-referenced to sample 0 along a Doppler-scaled rate,
                    # so lifting it to a physical phase carries a lever of ~5095 chips per Hz
                    # at this uptime -- exact only if the Doppler is the replica's own
                    # (doppler_applied_hz). The reported doppler_hz differs by ~0.1 Hz, which
                    # is 500 chips: not a degraded residual, a meaningless one.
                    # ⚡ PREFER THE BROKER'S OWN INTEGRITY RESIDUAL. It is the same quantity
                    # this block reconstructs -- measured code minus the model, receiver clock
                    # removed -- but measured where the currency is unambiguous, so it does not
                    # ride the argument's Doppler lever. On a healthy chain it is +-0.1-0.5
                    # chips; the reconstruction below lands within a code period at best. The
                    # reconstruction stays as the fallback for the airspy prototype, which has
                    # no dead-reckon clock.
                    # ⚡⚡ PREFER THE TRACKER'S RESIDUAL OVER THE SEARCH'S. dr_integ is the
                    # search detection's code phase against the model -- the acquisition
                    # estimate, ~0.5 chips of white noise per satellite. trk_resid is the
                    # closed fleet DLL's replica placement against the same model and the
                    # same clock (gnss_broker/trkresid), resolves ~1e-2 chips, and exists on
                    # every chain, detectors or not. Taken in SECONDS: the broker's chips
                    # are its own currency (CM chips on L2C) and metres must not guess it.
                    # The RANGE form (receiver clock left in, trk_range_s) is the row's
                    # code_resid_m -- "receiver clock NOT removed" is this record's contract,
                    # and the carrier residual below carries the same clock, so code minus
                    # carrier is clock-free. Older brokers published only the clock-removed
                    # form; it is kept as the fallback, and code_resid_clk says which.
                    _ts = r.get("trk_range_s")
                    _clk_in = _ts is not None
                    if _ts is None:
                        _ts = r.get("trk_resid_s")
                    _ta = r.get("trk_resid_age_s")
                    _ic = r.get("dr_integ_chips")
                    _ia = r.get("dr_integ_age_s")
                    if _ts is not None and (_ta is None or _ta <= args.integ_max_age_s):
                        code_resid_m = float(_ts) * C_LIGHT
                        code_resid_src = "trk"
                    elif _ic is not None and (_ia is None or _ia <= args.integ_max_age_s):
                        code_resid_m = float(_ic) * C_LIGHT / args.chip_rate_hz
                        code_resid_src = "dr_integ"
                    dop_used = r.get("doppler_applied_hz")
                    if dop_used is None:
                        dop_used = r.get("doppler_hz") or 0.0
                        _warn_dop[0] = _warn_dop[0] + 1
                        if _warn_dop[0] == 1:
                            print("gnss_observables: no doppler_applied_hz -- falling back to "
                                  "the reported Doppler; the code residual carries the "
                                  "5095 chips/Hz argument lever and is NOT metre-good",
                                  file=sys.stderr)
                    cp_phys = _phys_chips(r.get("code_phase_chips") or 0.0, args.comb_mult,
                                          hop if (frame0 and hop) else None, t_abs,
                                          float(dop_used), args) if code_resid_src is None else 0.0
                    t_tx = (gpst_of_utc(t_epoch) - v["range_m"] / C_LIGHT + v["sat_clk_s"])
                    cp_pred = (t_tx % t_rec) / t_rec * args.code_length
                    d = ((cp_phys - cp_pred + args.code_length / 2.0) % args.code_length
                         - args.code_length / 2.0)
                    if code_resid_src is None and not frame0:
                        # AIRSPY ONLY. On CHORD the reconstruction rides the argument's
                        # ~5095 chips/Hz Doppler lever, so it lands within a code period at
                        # best -- measured 1.7 km of position error when it reached the PVT
                        # solve as if it were an observation. A null is the honest output:
                        # this chain has no dead-reckon clock yet, so it has no code residual.
                        code_resid_m = d * C_LIGHT / args.chip_rate_hz
                        code_resid_src = "reconstructed"
                    carr_resid_m = -adr * lam - v["range_m"]
                    carr_resid_src = "adr"
                    # ⚡ THE FLEET ADR, at ITS OWN hop (gnss_broker/fleetadr). adr_cycles above
                    # is one instance's accumulator with its epoch hidden; a carrier phase
                    # counts 1.17e9 cycles/s, so a residual built from it at the row's epoch is
                    # off by c*dt. fadr_dop_cycles is the Doppler-only phase at fadr_hop, and
                    # the model range is evaluated at exactly that hop -- so the residual is
                    # the geometry-free carrier range on this arc, up to the arc's constant.
                    _fh = r.get("fadr_hop")
                    _fd = r.get("fadr_dop_cycles")
                    if frame0 and _fh and _fd is not None and eph:
                        try:
                            _ta = frame0 + _fh * args.samples_per_hop / args.sample_rate_hz
                            _va = predict_all(eph, args.lat, args.lon, args.alt,
                                              datetime.fromtimestamp(_ta, tz=timezone.utc),
                                              mask_deg=-90.0,
                                              max_age=args.eph_geom_window_s).get((args.sys, prn))
                        except Exception:
                            _va = None
                        if _va is not None:
                            carr_resid_m = -_fd * lam - _va["range_m"]
                            carr_resid_src = "fadr"
                row = {
                    "t": round(t_epoch, 4),
                    "t_gps": round(gpst_of_utc(t_epoch), 4),
                    "sys": args.sys, "prn": prn, "band": args.band,
                    "carrier_hz": args.carrier_hz,
                    # --- CODE: commanded code phase + the DLL's unapplied sub-chip residual.
                    # Kept separate on purpose: the trim is a loop state, the discriminator is
                    # a measurement, and offline gets to decide how much of it to believe.
                    "cp_chips": r.get("code_phase_chips"),
                    "dll_disc": r.get("dll_disc"),
                    "code_len": args.code_length, "chip_rate_hz": args.chip_rate_hz,
                    # --- PHASE: cycles on THIS arc (0 at an arc start). Ambiguous by an
                    # integer; arc id is what tells you when the ambiguity changed.
                    "adr_cycles": adr,
                    "trim_cycles": trim,
                    "adr_m": (adr * lam) if adr is not None else None,
                    "adr_arc": arc, "adr_records": nrec,
                    "code_resid_m": code_resid_m,     # model-removed code range (CMC input)
                    "code_resid_src": code_resid_src,  # trk | dr_integ | reconstructed
                    "code_resid_clk": _clk_in,         # True: receiver clock IN (range form)
                    # the tracker residual's own scatter and support (metres, records)
                    "code_resid_sd_m": ((r.get("trk_resid_sd_chips") or 0.0) * C_LIGHT
                                        / args.chip_rate_hz
                                        if code_resid_src == "trk" else None),
                    "code_resid_n": (r.get("trk_resid_n") if code_resid_src == "trk" else None),
                    # the record's own (argument, Doppler, hop) triple, verbatim: the RAW code
                    # observable, from which any residual can be re-derived offline against a
                    # better model. Only meaningful as a triple (chord-cp-currency).
                    "cp_rec_chips": r.get("cp_rec_chips"),
                    "dop_rec_hz": r.get("dop_rec_hz"),
                    "rec_hop": r.get("rec_hop"),
                    "dll_trim_cpp": r.get("dll_trim_cpp"),
                    "carr_resid_m": carr_resid_m,     # model-removed carrier range (CMC input)
                    "carr_resid_src": carr_resid_src,  # fadr (exact hop) | adr (epoch hidden)
                    # the arc the carrier residual lives on: fleet ADR's when it is the source
                    "carr_arc": (r.get("fadr_arc") if carr_resid_src == "fadr" else arc),
                    # THE FLEET ADR, verbatim: Doppler-only cycles and the full phase, both at
                    # fadr_hop, on arc fadr_arc that began at fadr_hop0. Every chain's rows carry
                    # hops on the one F-engine axis, so two bands pair at EQUAL hops exactly.
                    "fadr_dop_cycles": r.get("fadr_dop_cycles"),
                    "fadr_cycles": r.get("fadr_cycles"),
                    "fadr_hop": r.get("fadr_hop"),
                    "fadr_hop0": r.get("fadr_hop0"),
                    "fadr_arc": r.get("fadr_arc"),
                    "fadr_n_rec": r.get("fadr_n_rec"),
                    "fadr_n_inst": r.get("fadr_n_inst"),
                    "fadr_trim_cycles": r.get("fadr_trim_cycles"),
                    "fadr_res_cycles": r.get("fadr_res_cycles"),
                    # the same ADR at the fleet-wide grid hop: EQUAL hops across every chain
                    "fadr_g_hop": r.get("fadr_g_hop"),
                    "fadr_g_dop_cycles": r.get("fadr_g_dop_cycles"),
                    "fadr_g_cycles": r.get("fadr_g_cycles"),
                    "adr_lock_s": r.get("adr_lock_s"),
                    # --- POWER
                    "cn0_coh_dbhz": cn0_dbhz(r, r.get("deep_snr"), r.get("coherence_s")),
                    # IS THE PROMPT TAP ACTUALLY ON THE SIGNAL? (task #47) Every C/N0 above is
                    # blind to code error -- deep_snr comes from the RE-SEARCHING deep fold, so
                    # it re-finds the satellite wherever the tap was commanded. Carried per-row
                    # so that offline can no longer average a blind window into a health number:
                    # 7-30% of bright-LOOKING samples on 2026-08-11/12 were blind.
                    # `prompt_lock` False => cn0_coh_dbhz is a deep-fold significance, not a C/N0.
                    "prompt_lock": r.get("prompt_lock"),
                    "prompt_rayleigh": r.get("prompt_rayleigh"),
                    "fleet_present": r.get("fleet_present"),
                    "coh_src": r.get("coh_src"),
                    "cn0_inc_dbhz": cn0_inc_dbhz(r.get("amplitude"),
                                                 r.get("unbiased_amplitude"), t_rec),
                    "cn0_q_dbhz": cn0_q_dbhz(r.get("snr_q"), t_rec),  # modulation-immune (BOC pilots)
                    # --- THE SERVED RADIOMETRY (task #57, 2026-08-15). Per-record prompt
                    # power, q-gated, debiased against the below-horizon probes; NO fit
                    # anywhere in it, unlike cn0_coh above (whose deep fold re-searches a
                    # rate per integration and carries ~20 dB of paired self-scatter).
                    # Validated on sky: split-half <= 0.1 dB, two-feed pairing <= 0.35 dB
                    # corrected (scripts/gnss/cn0_prompt_gate.py). duty is the lock duty
                    # the value is conditioned on -- offline consumers decline low-duty
                    # rows rather than average them; noise_probe marks the below-horizon
                    # noise references riding the same rows (never satellites).
                    "cn0_prompt_dbhz": r.get("cn0_prompt_db"),
                    "cn0_prompt_duty": r.get("cn0_prompt_duty"),
                    "cn0_prompt_split_db": r.get("cn0_prompt_split_db"),
                    "noise_probe": r.get("noise_probe"),
                    # The known-rate coherent fold (#57 step 3): the deep-sidelobe
                    # radiometry (+~10log10(n_rec) sensitivity over cn0_prompt), rate
                    # injected causally, never searched. kcoh_sig is its detection
                    # significance vs the probes' identical fold; kcoh_eta the coherence
                    # efficiency (n_rec = fully coherent, ~1 = noise). On strong
                    # satellites cn0_kcoh and cn0_prompt must agree -- their difference
                    # per row is the standing cross-check, archived here.
                    "cn0_kcoh_dbhz": r.get("cn0_kcoh_db"),
                    "cn0_kcoh_sky_dbhz": r.get("cn0_kcoh_sky_db"),
                    "kcoh_sig": r.get("kcoh_sig"),
                    "kcoh_eta": r.get("kcoh_eta"),
                    "kcoh_rate_hz": r.get("kcoh_rate_hz"),
                    "kcoh_t_coh_s": r.get("kcoh_t_coh_s"),
                    "sig": sig, "coherence_s": r.get("coherence_s"),
                    "search_snr": det_snr.get(prn),
                    "doppler_hz": r.get("doppler_hz"),
                    "carrier_hz_resid": r.get("carrier_hz_resid"),
                    "nh_phase": r.get("nh_phase"),
                    # boundary_f: the code-period boundary's offset inside the despread window.
                    # Pre-feab8b04 amplitude went as |2f-1|, so f~0.5 nulled the sat (the
                    # "bistable"); the segmented despread killed that dependence. Logged so a
                    # soak can PROVE C/N0 is independent of f (the direct regression test) --
                    # it was REST-only before, which is exactly why the 07-15 overnight
                    # couldn't be tested statistically.
                    "boundary_f": r.get("boundary_f"),
                    "s4": r.get("s4"),
                    "s4_raw": r.get("s4_raw"),
                    "sigma_phi": r.get("sigma_phi"),
                }
                # --- GEOMETRY (BRDC at this epoch). Receiver clock NOT removed: solving it is
                # a downstream job, and a row that has had a model subtracted can never be
                # un-subtracted when the model improves.
                if v is not None:
                    row.update({"az": round(v["az"], 3), "el": round(v["el"], 3),
                                "range_m": round(v["range_m"], 3),
                                "range_rate_mps": round(v["range_rate_mps"], 4),
                                "sat_clk_s": v["sat_clk_s"],
                                "eph_age_s": round(v["toe_age_s"], 1)})
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
                n += 1
        # PHASE-LOCK the cadence to the wall-clock interval grid (2026-07-19): dTEC pairing
        # intersects epochs EXACTLY across bands, and free-running loops land on arbitrary
        # per-process phases -- measured tonight: the L1 and L2C loggers both ran 2 s
        # cadences on OPPOSITE second parity, so the G:L1xL2C pair had literally zero
        # common epochs all day while L1xL5 worked by parity luck. Sleeping to the next
        # multiple of the interval puts every band's logger on one shared grid.
        now = time.time()
        dt = args.interval - (now % args.interval)
        time.sleep(dt if dt > 0.05 else dt + args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
