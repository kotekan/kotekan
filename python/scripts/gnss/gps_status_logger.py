#!/usr/bin/env python3
"""Persist the GPS combiner's per-PRN detection stats to a JSONL log, so C/N0 (and everything else
that only lives in REST / the browser -- deep_snr, coherence_s, amp_snr, carrier_hz_resid) survives
the run for offline analysis (captures/gps_beam_survey.py reads it).

The recorded /tmp/gpswipe/level_*.raw file carries Â but NOT deep_snr/coherence_s, so an absolute C/N0
can't be reconstructed from it. This logger closes that gap: it GETs <combiner>/get_status (+ the
search SNR and ADC rms) on a timer and appends one JSON object per ACTIVE PRN per poll, stamped with
REAL wall-clock UTC (so no capture-time anchoring is needed either). Tiny + dedup'd; rotate/delete
freely -- it's a monitoring side-log, not part of the pipeline.

Run alongside the pipeline (run_live.sh launches it automatically). Standalone:
  python3 gps_status_logger.py --url http://localhost:12048 --out /tmp/gpswipe/status_log.jsonl
"""
import argparse, json, os, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnss_stages import resolve_stage  # noqa: E402  (gps_* <-> bare stage-name aliasing)


def _get(url, timeout=3.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return None


def cn0_dbhz(deep_snr, coherence_s):
    """C/N0 (dB-Hz) = 10*log10(deep_snr^2 / T_coh). Matches the viewer's Â<->C/N0 toggle: squaring
    deep_snr undoes the sqrt-compression, /coherence_s makes it a density stable across the
    auto-coherence ladder's window choice. Absolute zero-point is approximate (nav-wipe loss)."""
    import math
    if deep_snr and deep_snr > 0:
        T = coherence_s if (coherence_s and coherence_s > 0) else 1.0
        return 20.0 * math.log10(deep_snr) - 10.0 * math.log10(T)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:12048", help="kotekan REST base")
    ap.add_argument("--combiner", default="combiner")
    ap.add_argument("--search", default="search")
    ap.add_argument("--airspy", default="airspy_in")
    ap.add_argument("--interval", type=float, default=2.0, help="poll period (s); 2 s is plenty for C/N0")
    ap.add_argument("--out", default="/tmp/gpswipe/status_log.jsonl")
    ap.add_argument("--no-dedup", action="store_true",
                    help="log every poll even if a PRN's deep_snr is unchanged (same combiner emit)")
    args = ap.parse_args(argv)

    # gps_* <-> bare stage-name aliasing (gnss_stages): the tri-constellation config names
    # the GPS chain gps_search/gps_combiner; the older benches use search/combiner.
    args.combiner = resolve_stage(args.url, args.combiner)
    args.search = resolve_stage(args.url, args.search)

    last = {}   # prn -> (deep_snr, deep_amplitude, t_logged) of the last logged sample (dedup)
    n = 0
    f = open(args.out, "a", buffering=1)   # line-buffered append
    print("gps_status_logger: %s -> %s (every %.1fs)" % (args.url, args.out, args.interval),
          file=sys.stderr)
    while True:
        t0 = time.time()
        status = _get("%s/%s/get_status" % (args.url, args.combiner))
        dets = _get("%s/%s/get_detections" % (args.url, args.search)) or []
        adc = _get("%s/%s/adcstat" % (args.url, args.airspy)) or {}
        det_snr = {int(d["prn"]): d.get("snr") for d in dets if "prn" in d}
        rms = adc.get("rms")
        now = time.time()   # REAL wall-clock UTC of this poll (no capture-time anchoring needed)
        if isinstance(status, list):
            for r in status:
                prn = r.get("prn")
                if not prn:
                    continue
                ds = r.get("deep_snr") or 0.0
                pk = r.get("peel_deep_snr") or 0.0   # peel moves independently of deep
                da = r.get("deep_amplitude") or 0.0
                asnr = r.get("amp_snr") or 0.0
                # Only ACTIVE PRNs (a real deep/incoherent presence), and dedup repeat polls of the
                # same emit (unchanged deep_snr+amplitude) -> keeps the log small. A 10 s heartbeat
                # overrides the dedup: noise probes sit at a constant floor for hours, and a
                # pedestal needs their rows to keep flowing (gps_cn0_map.py pedestal()).
                if ds <= 0 and asnr <= 0 and da <= 0:
                    continue
                prev = last.get(prn)
                # NB the dedup key carries peel_deep_snr too: the peel residual moves
                # INDEPENDENTLY of deep (that is the entire point of measuring it), so keying on
                # (deep_snr, deep_amplitude) alone would silently drop exactly the samples that
                # make a depth-vs-time series interesting.
                if (not args.no_dedup and prev is not None
                        and prev[:3] == (ds, da, pk) and now - prev[3] < 10.0):
                    continue
                last[prn] = (ds, da, pk, now)
                rec = {"t": round(now, 3), "prn": int(prn),
                       # raw biased incoherent |A| -- with unbiased_amplitude this makes the noise
                       # power exactly recoverable offline (N = amplitude^2 - unbiased^2), the
                       # input to the incoherent C/N0 beam map (gps_cn0_map.py).
                       "amplitude": r.get("amplitude") or 0.0,
                       "deep_snr": ds, "coherence_s": r.get("coherence_s") or 0.0,
                       "deep_records": r.get("deep_records") or 0,
                       "deep_amplitude": da, "amp_snr": asnr,
                       "unbiased_amplitude": r.get("unbiased_amplitude") or 0.0,
                       "cn0_dbhz": cn0_dbhz(ds, r.get("coherence_s")),
                       "doppler_hz": r.get("doppler_hz"), "code_phase_chips": r.get("code_phase_chips"),
                       "carrier_hz_resid": r.get("carrier_hz_resid"),
                       "dll_disc": r.get("dll_disc"),
                       # fixed-full-window noise-debiased coherent power (Hz, ~0 on noise) +
                       # the reported rung's rectification floor -- the coherent beam-map
                       # observable and its honesty certificate (gps_cn0_map.py)
                       "deep_pow_hz": r.get("deep_pow_hz"),
                       # B1C/L1C overlay alignment phase: physics pins every healthy sat to
                       # the FLEET CONSENSUS +-2 chips (secondaries are BDT-frame-synced;
                       # only differential range separates them) -- a sat away from
                       # consensus has a wrong overlay row or a false alignment.
                       "nh_phase": r.get("nh_phase"),
                       # boundary_f: where this sat's code-period boundary sits inside the
                       # despread window (0/1 = window edge, ~0.5 = the record straddles the
                       # secondary-overlay chip flip). Pre-feab8b04 the amplitude went as
                       # |2f-1| and f~0.5 NULLED the sat (the "bistable"); the segmented
                       # despread removed that, so C/N0 must now be INDEPENDENT of f -- log it
                       # to keep proving that (and to catch any regression) over a whole soak.
                       "boundary_f": r.get("boundary_f"),
                       # multipath/scintillation pair: S4 (amplitude fluctuation, thermal
                       # floor removed) and sigma_phi (carrier-phase jitter, rad). Multipath
                       # moves both; our two C/N0 estimators are each blind to one of them.
                       "s4": r.get("s4"),
                       "s4_raw": r.get("s4_raw"),
                       "sigma_phi": r.get("sigma_phi"),
                       "deep_floor": r.get("deep_floor"),
                       # VOLTAGE PEEL (P7b) -- the residual after subtraction, measured by the
                       # SAME estimator on the SAME window/phase as `deep_amplitude`, so the
                       # ratio is honest. Until 2026-07-26 this lived ONLY in REST and the
                       # browser: a couple of hours of G29 showing clear periodic structure was
                       # unanalysable because nothing persisted it, and the fallback (the peel
                       # health line) samples at 9.984 s against a 30 s GPS frame -- 3.005x,
                       # which aliases the frame into a ~104 MINUTE beat and would manufacture
                       # exactly the slow periodicity one is trying to measure.
                       #
                       # DEPTH IS DERIVED, NOT STORED: 20*log10(deep_amplitude/peel_deep), and
                       # it is a BOUND (>=) whenever peel_deep_snr sits at the detection floor,
                       # because a residual at the floor is not a measurement. Log the parts and
                       # let the analysis decide -- storing a single "depth" column would bake
                       # in the at-floor ambiguity that has already produced two retracted
                       # claims (>=61 dB and >=47 dB readings, both floor junk).
                       "peel_deep": r.get("peel_deep"),
                       "peel_deep_snr": r.get("peel_deep_snr"),
                       "peel_incoh": r.get("peel_incoh"),
                       "search_snr": det_snr.get(int(prn)), "adc_rms": rms}
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                n += 1
        # steady cadence regardless of request latency
        dt = args.interval - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
