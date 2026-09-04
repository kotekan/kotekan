#!/usr/bin/env python3
"""Minimal TEST-BENCH broker: closes the search -> peel seed loop for offline peel benches.

WHY THIS EXISTS. The replay bench (peel_test20-style: rawFileRead -> fengine -> search + peel)
has no broker, so GnssVoltagePeel's seeds are STATIC config values. That breaks peel validation
in two independent ways:

  1. fcar never refreshes  ->  f_ref == fcar forever  ->  the v2 common-reference correction's
     absolute-anchor bridging term is NEVER exercised. That term (the gain-phase jump at each
     broker dop refresh, ~15 cycles at t=30 s) is the DOMINANT mechanism the v2 fix addresses;
     the Doppler-rate ramp alone is negligible over a ~7 ms EMA window. A static-fcar bench is
     therefore structurally blind to the fix (measured 2026-07-23: smoothing+FF == smoothing).
  2. A hand-seeded code_phase_rate is only approximate, so with pullin_chips=0 (which the
     GnssVoltagePeel header REQUIRES when smoothing -- a per-record re-pick wrecks the phase
     track) the replica drifts off-peak and the peel becomes CODE-limited, masking any
     gain-smoothing effect entirely (measured: v1 depth 29.0 -> 11.8 dB).

This script fixes both: it polls the baseline search's detections and pushes consensus seeds to
the peel on an interval, exactly as the real broker does -- giving genuine periodic dop refreshes
AND a self-correcting cp track.

It deliberately mirrors only the broker behaviour the PEEL consumes (seed push); it is NOT a
stand-in for gps_distributed_broker.py (no carrier loop, no DLL trim, no clock solve, no coasting).

Usage (bench kotekan on -b 0.0.0.0:12060):
    ./peel_bench_broker.py --url http://localhost:12060 --prns 10,32 --interval 1.0 --duration 90
"""
import argparse
import json
import time
import urllib.request


def _get(url, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(url, payload, timeout=5.0):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def _slope(xs, ys):
    """Least-squares slope dy/dx; 0.0 if under-determined."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def _unwrap(prev, cur, period):
    """Bring `cur` onto the branch nearest `prev` (code phase wraps at `period` chips)."""
    if prev is None:
        return cur
    k = round((prev - cur) / period)
    return cur + k * period


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:12060", help="bench kotekan REST base")
    ap.add_argument("--search", default="search_base", help="stage name of the BASELINE search")
    ap.add_argument("--peel", default="peel", help="stage name of the GnssVoltagePeel")
    ap.add_argument("--trackers", default="",
                    help="comma-separated GnssChannelizedTracker stage names to seed as well "
                         "(e.g. 'track_base,track_peel' for the depth bench -- they take the same "
                         "set_seeds payload). Empty = peel only.")
    ap.add_argument("--prns", required=True, help="comma-separated PRNs the peel is configured for")
    ap.add_argument("--interval", type=float, default=1.0, help="seed push period (s) = dop refresh rate")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after this many s (0 = forever)")
    ap.add_argument("--sample-rate", type=float, default=20.0e6)
    ap.add_argument("--spectrum-length", type=int, default=10, help="fft_len = 2*this")
    ap.add_argument("--code-length", type=float, default=1023.0, help="chips per code period")
    ap.add_argument("--history", type=int, default=8, help="detections kept per PRN for rate fits")
    ap.add_argument("--doppler-rate", default="",
                    help="per-PRN Hz/s, e.g. '10:-0.147,32:-0.418'. This is the ORBIT-model rate, "
                         "the same way the real broker gets doppler_rate_hz_s from BRDC -- do NOT "
                         "estimate it by differencing detections: GnssChannelizedSearch reports a "
                         "GRID-QUANTIZED Doppler (no parabola-refine, see buglist), so a "
                         "short-baseline fit reads +-5 Hz/s of pure noise against a ~0.15 Hz/s truth.")
    ap.add_argument("--raw-doppler", action="store_true",
                    help="push the latest raw detection Doppler instead of the smoothed trajectory "
                         "(diagnostic: injects the search's ~20 Hz grid jitter into fcar)")
    ap.add_argument("--carrier-loop", default="",
                    help="combiner stage name (e.g. 'gps_combiner') to close a SHARED CARRIER "
                         "LOOP against: poll its carrier_hz_resid per PRN, integrate a trim, and "
                         "push carrier_trim_hz with the seeds -- the minimal mirror of "
                         "gps_distributed_broker's loop. WITHOUT this the GPU chain "
                         "(carrier_shared, fll_gain 0 by contract) has NO fine carrier at all: "
                         "deep coherence collapses to the shortest ladder rung, deep_snr pins at "
                         "the rectification floor on EVERY chain, and any depth measured from "
                         "deep amplitudes is a ratio of floor junk (measured 2026-07-24 -- "
                         "carr_resid 12-55 Hz, coherence_s stuck at 0.124 s, baseline and peeled "
                         "alike). Empty = off (the historical behaviour).")
    ap.add_argument("--trim-gain", type=float, default=0.4,
                    help="carrier-loop integrator gain per poll (on the measured residual Hz)")
    ap.add_argument("--trim-clamp", type=float, default=100.0, help="|trim| clamp (Hz)")
    ap.add_argument("--dry-run", action="store_true", help="log seeds, do not POST")
    args = ap.parse_args()

    prns = [int(p) for p in args.prns.split(",") if p.strip()]
    drate_cfg = {}
    for tok in args.doppler_rate.split(","):
        if ":" in tok:
            k, v = tok.split(":", 1)
            drate_cfg[int(k)] = float(v)
    dt_hop = (2.0 * args.spectrum_length) / args.sample_rate  # seconds per hop
    det_url = "%s/%s/get_detections" % (args.url.rstrip("/"), args.search.strip("/"))
    seed_urls = ["%s/%s/set_seeds" % (args.url.rstrip("/"), args.peel.strip("/"))]
    for t in args.trackers.split(","):
        if t.strip():
            seed_urls.append("%s/%s/set_seeds" % (args.url.rstrip("/"), t.strip().strip("/")))

    hist = {p: [] for p in prns}   # prn -> [(ref_hop, doppler_hz, cp_unwrapped)]
    trim = {p: 0.0 for p in prns}  # prn -> integrated carrier trim (Hz), --carrier-loop
    trim_last = {}                 # prn -> last integrated residual (dedup on changed value)
    locked_once = set()            # prns that have EVER cohered (bootstrap -> track mode)
    comb_url = ("%s/%s/get_status" % (args.url.rstrip("/"), args.carrier_loop.strip("/"))
                if args.carrier_loop else "")
    t0 = time.time()
    n_push = 0
    print("bench-broker: %s -> %s | prns=%s interval=%.2fs dt_hop=%.3g s"
          % (det_url, ",".join(u.rsplit("/", 2)[-2] for u in seed_urls), prns,
             args.interval, dt_hop), flush=True)

    while True:
        if args.duration > 0 and (time.time() - t0) >= args.duration:
            break
        try:
            dets = _get(det_url)
        except Exception as e:
            print("  detections fetch failed (%s); retrying" % e, flush=True)
            time.sleep(args.interval)
            continue

        by_prn = {int(d["prn"]): d for d in dets if int(d["prn"]) in hist}

        # Shared carrier loop: integrate the combiner's measured residual into the trim. The
        # residual is reported in the INTERNAL (r2c-flipped) convention like the trim itself, so
        # the sign is a straight accumulate (the same convention the real broker uses). Gated on
        # a sane magnitude -- an unlocked sat's "residual" is fit noise.
        if comb_url:
            try:
                for r in _get(comb_url):
                    p = int(r.get("prn", -1))
                    if p in trim:
                        resid = float(r.get("carrier_hz_resid", 0.0) or 0.0)
                        # Dedup on changed value, exactly like the real broker: the combiner
                        # emits a fresh residual every ~1 s window while this loop polls at the
                        # push cadence -- integrating the SAME emit repeatedly over-applies the
                        # gain and oscillates (measured: settled at ~1 Hz p-p instead of
                        # converging; the real broker's comment reports +-20 Hz swings from the
                        # same mistake at its 5 Hz poll rate).
                        if not (0.0 < abs(resid) < 60.0) or resid == trim_last.get(p):
                            continue
                        trim_last[p] = resid
                        # TWO-MODE, the real broker's shape in miniature. BOOTSTRAP (never
                        # cohered): full gain -- the true residual IS the seed error and gating
                        # would deadlock. TRACK (cohered once): innovation-gate + slew-clamp.
                        # The squared-phase residual fit is ambiguous mod 1/(2*T_window) (+-2 Hz
                        # at the 0.25 s rung), and integrating an ALIASED measurement at full
                        # gain is positive feedback: measured 2026-07-24, the loop pulled in,
                        # deep-locked at 103 sigma / 0.5 s coherence, then walked the NCO off at
                        # ~1.5 Hz/s to +15 Hz -- the broker comment's "alias-capture disease"
                        # reproduced in one bench run. Gate + clamp is its minimal cure.
                        if (r.get("coherence_s") or 0.0) > 0.0:
                            locked_once.add(p)
                        if p in locked_once and abs(resid) > 3.0:
                            continue  # innovation gate: an aliased/garbage fit, not a measurement
                        step = args.trim_gain * resid
                        step = max(-2.0, min(2.0, step))  # slew clamp (Hz per update)
                        trim[p] += step
                        trim[p] = max(-args.trim_clamp, min(args.trim_clamp, trim[p]))
            except Exception as e:
                print("  carrier-loop poll failed (%s)" % e, flush=True)

        seeds = []
        for prn in prns:
            d = by_prn.get(prn)
            h = hist[prn]
            if d is not None:
                ref_hop = float(d.get("ref_hop", 0))
                cp = float(d["code_phase_chips"])
                cp = _unwrap(h[-1][2] if h else None, cp, args.code_length)
                # only append genuinely new detections (ref_hop advances)
                if not h or ref_hop > h[-1][0]:
                    h.append((ref_hop, float(d["doppler_hz"]), cp))
                    del h[: max(0, len(h) - args.history)]
            if not h:
                continue  # nothing to seed this PRN with yet

            ref_hop, dop_raw, cp = h[-1]
            hops = [e[0] for e in h]
            # doppler_rate: from the ORBIT model (CLI), not from differencing quantized detections.
            drate = drate_cfg.get(prn, 0.0)
            # Smooth Doppler trajectory: SLOPE fixed at the orbit rate, OFFSET measured robustly
            # (median of dop_i - drate*t_i). This is what the real broker effectively pushes --
            # a smooth orbit-driven dop, not the search's grid-quantized argmax. Pushing the raw
            # detection instead injects ~20 Hz jitter into fcar (see --raw-doppler).
            if args.raw_doppler:
                dop = dop_raw
            else:
                offs = sorted(e[1] - drate * (e[0] * dt_hop) for e in h)
                b = offs[len(offs) // 2]
                dop = b + drate * (ref_hop * dt_hop)
            # code_phase_rate: chips per HOP (the peel extrapolates cp + rate*(win_hop - ref_hop)).
            # This one IS fit from detections -- it converges well (cp is sub-chip accurate).
            cprate = _slope(hops, [e[2] for e in h])
            seeds.append({
                "prn": prn,
                "doppler_hz": dop,
                "code_phase_chips": cp % args.code_length,
                "code_phase_rate": cprate,
                "doppler_rate_hz_s": drate,
                "carrier_trim_hz": trim.get(prn, 0.0),
                "ref_hop": int(ref_hop),
            })

        if seeds:
            n_push += 1
            desc = " ".join("PRN%d dop%+.1f drate%+.3f cpr%+.2e"
                            % (s["prn"], s["doppler_hz"], s["doppler_rate_hz_s"],
                               s["code_phase_rate"]) for s in seeds)
            if args.dry_run:
                print("  [dry] push #%d: %s" % (n_push, desc), flush=True)
            else:
                bad = []
                for u in seed_urls:
                    try:
                        _post(u, seeds)
                    except Exception as e:
                        bad.append("%s (%s)" % (u.rsplit("/", 2)[-2], e))
                print("  push #%d: %s%s" % (n_push, desc,
                                            ("  FAILED:" + ",".join(bad)) if bad else ""), flush=True)
        time.sleep(args.interval)

    print("bench-broker: done (%d pushes)" % n_push, flush=True)


if __name__ == "__main__":
    main()
