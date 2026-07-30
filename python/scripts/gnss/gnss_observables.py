#!/usr/bin/env python3
"""GNSS OBSERVABLES RECORD -- the instrument's primary data product (RINEX in spirit).

One JSONL row per satellite per band per epoch, carrying the two ranging observables and
the geometry needed to interpret them:

    CODE   code_phase_chips (+ dll_disc, the sub-chip residual the DLL is still working off)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnss_stages import resolve_stage, capture_clock
from gnss_ephemeris import (fetch_brdc, parse_rinex_nav, predict_all, gpst_of_utc, C_LIGHT)


def _get(url, timeout=3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return None


def cn0_dbhz(deep_snr, coherence_s):
    """Coherent C/N0 (dB-Hz) = 10 log10(deep_snr^2 / T_coh); None unless the combiner
    certified the coherence (coherence_s > 0 -- a floored deep is rectification noise)."""
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
    ap.add_argument("--out", default="/tmp/gpswipe/observables.jsonl")
    args = ap.parse_args()

    args.combiner = resolve_stage(args.url, args.combiner)
    args.search = resolve_stage(args.url, args.search)
    to_unix = capture_clock(args.url, args.airspy)  # capture clock -> unix (band-prefixed stage)
    utc0 = 0.0                          # capture sample-0 UTC (CMC needs the absolute age)
    t_rec = args.code_length / args.chip_rate_hz
    lam = C_LIGHT / args.carrier_hz          # carrier wavelength (m/cycle)

    eph, eph_t, eph_probe_t = None, 0.0, 0.0
    last = {}   # prn -> (adr_arc, adr_records) of the last row written (emit dedup)
    n = 0
    f = open(args.out, "a", buffering=1)
    print("gnss_observables: %s [%s/%s] -> %s" % (args.band, args.sys, args.combiner, args.out),
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
                t_epoch = to_unix(r.get("utc") or 0.0)
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
                if v is not None and utc0 and adr is not None:
                    t_abs = t_epoch - utc0
                    cp_phys = ((r.get("code_phase_chips") or 0.0)
                               + t_abs * args.chip_rate_hz
                                 * (1.0 + args.code_doppler_sign
                                    * (r.get("doppler_hz") or 0.0) / args.carrier_hz)
                               ) % args.code_length
                    t_tx = (gpst_of_utc(t_epoch) - v["range_m"] / C_LIGHT + v["sat_clk_s"])
                    cp_pred = (t_tx % t_rec) / t_rec * args.code_length
                    d = ((cp_phys - cp_pred + args.code_length / 2.0) % args.code_length
                         - args.code_length / 2.0)
                    code_resid_m = d * C_LIGHT / args.chip_rate_hz
                    carr_resid_m = -adr * lam - v["range_m"]
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
                    "carr_resid_m": carr_resid_m,     # model-removed carrier range (CMC input)
                    "adr_lock_s": r.get("adr_lock_s"),
                    # --- POWER
                    "cn0_coh_dbhz": cn0_dbhz(r.get("deep_snr"), r.get("coherence_s")),
                    "cn0_inc_dbhz": cn0_inc_dbhz(r.get("amplitude"),
                                                 r.get("unbiased_amplitude"), t_rec),
                    "cn0_q_dbhz": cn0_q_dbhz(r.get("snr_q"), t_rec),  # modulation-immune (BOC pilots)
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
