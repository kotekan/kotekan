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
    seed_url = "%s/%s/set_seeds" % (args.url.rstrip("/"), args.peel.strip("/"))

    hist = {p: [] for p in prns}   # prn -> [(ref_hop, doppler_hz, cp_unwrapped)]
    t0 = time.time()
    n_push = 0
    print("bench-broker: %s -> %s | prns=%s interval=%.2fs dt_hop=%.3g s"
          % (det_url, seed_url, prns, args.interval, dt_hop), flush=True)

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
                try:
                    _post(seed_url, seeds)
                    print("  push #%d: %s" % (n_push, desc), flush=True)
                except Exception as e:
                    print("  seed push failed (%s)" % e, flush=True)
        time.sleep(args.interval)

    print("bench-broker: done (%d pushes)" % n_push, flush=True)


if __name__ == "__main__":
    main()
