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
import argparse, json, sys, time, urllib.request


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

    last = {}   # prn -> (deep_snr, deep_amplitude) of the last logged sample (dedup)
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
                da = r.get("deep_amplitude") or 0.0
                asnr = r.get("amp_snr") or 0.0
                # Only ACTIVE PRNs (a real deep/incoherent presence), and dedup repeat polls of the
                # same emit (unchanged deep_snr+amplitude) -> keeps the log small.
                if ds <= 0 and asnr <= 0 and da <= 0:
                    continue
                if not args.no_dedup and last.get(prn) == (ds, da):
                    continue
                last[prn] = (ds, da)
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
