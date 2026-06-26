#!/usr/bin/env python3
"""Distributed GNSS broker: search detections -> consensus seeds -> trackers.

The control plane of the distributed-band channelized pipeline. It is wholly
decoupled from the sample dataflow -- pure REST glue + assignment policy -- so the
same broker drives the local single-kotekan comb and the real cross-node CHORD
layout; only the endpoint list changes.

Each iteration:
  1. GET <detector>/get_detections from every detection source; keep the best-SNR
     detection per PRN. A "detector" is a GnssSearchAggregator (cross-channel
     consensus already done) or, in the older fan-out, a GnssChannelizedSearch per
     subband -- either way the broker takes the strongest hit per PRN across them.
  2. (optional) gate by visibility: drop below-horizon PRNs.
  3. GET <combiner>/get_status for the full-band |A| per PRN; drop a tracked PRN
     after it stays below --drop-amplitude for --drop-hits consecutive polls
     (lock lost).
  4. POST the consensus seed set [{prn, doppler_hz, code_phase_chips}] to *every*
     tracker's /set_seeds, so all subbands despread the same (cp, Doppler) and the
     per-subband products recombine coherently.

Endpoints (--detectors / --trackers / --combiner) are comma-separated and support:
  * bare stage names         -> resolved against --rest-url (one kotekan instance)
  * absolute URLs            -> http://host:port/stage (per-node, mix freely)
  * brace ranges {a..b}      -> track_{00..49} expands to track_00..track_49,
                                bash-style (zero-padded iff an operand is padded);
                                http://nodeB:12048/track_{0..24} works too.

Seeds are refreshed from the latest detection every iteration, so the trackers'
code phase stays fresh against slow code-Doppler drift -- run --interval well below
the ~0.5 s decorrelation time. REST via stdlib urllib; visibility via skyfield
(optional, only if --lat/--lon given).
"""

import argparse
import json
import re
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone


def _get(url, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(url, payload, timeout=5.0):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def _log(msg):
    print("[broker] %s" % msg, file=sys.stderr, flush=True)


def expand_token(tok):
    """Expand the first bash-style {a..b} range in a token, recursing for more.

    Zero-pads to the operand width iff either operand is written zero-padded
    (e.g. {00..49} -> 00..49, but {0..49} -> 0..49), matching shell brace ranges.
    """
    m = re.search(r"\{(\d+)\.\.(\d+)\}", tok)
    if not m:
        return [tok]
    lo, hi = m.group(1), m.group(2)
    padded = (len(lo) > 1 and lo[0] == "0") or (len(hi) > 1 and hi[0] == "0")
    width = max(len(lo), len(hi)) if padded else 0
    a, b = int(lo), int(hi)
    step = 1 if b >= a else -1
    out = []
    for i in range(a, b + step, step):
        out.append(tok[:m.start()] + str(i).zfill(width) + tok[m.end():])
    res = []
    for o in out:  # handle any further ranges in the same token
        res.extend(expand_token(o))
    return res


def resolve_prefix(entry, default_base):
    """Endpoint prefix (everything before /<verb>) for a list entry.

    Absolute http(s) entries are used as-is; bare names hang off --rest-url.
    """
    entry = entry.strip()
    if entry.startswith("http://") or entry.startswith("https://"):
        return entry.rstrip("/")
    return default_base.rstrip("/") + "/" + entry.strip("/")


def parse_endpoints(csv, default_base):
    """Comma list -> resolved endpoint prefixes, with {a..b} ranges expanded."""
    prefixes = []
    for raw in csv.split(","):
        raw = raw.strip()
        if not raw:
            continue
        for tok in expand_token(raw):
            prefixes.append(resolve_prefix(tok, default_base))
    return prefixes


def visible_prns(lat, lon, alt_m, mask_deg, look_ahead_s):
    """PRNs above the elevation mask now (and look_ahead_s ahead). Needs skyfield."""
    try:
        from gps_beamtrack import load_gps_sats, sat_elevation  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        _log("visibility gating unavailable (%s); not gating" % e)
        return None
    sats = load_gps_sats()
    up = set()
    for prn, sat in sats.items():
        if sat_elevation(sat, lat, lon, alt_m, look_ahead_s) >= mask_deg:
            up.add(prn)
    return up


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rest-url", default="http://localhost:12048",
                    help="default kotekan REST base for bare stage names")
    ap.add_argument("--detectors", "--searches", dest="detectors", required=True,
                    help="detection endpoints: aggregator/search stage names, absolute "
                         "URLs, or {a..b} ranges (e.g. aggregate  or  search_{00..49})")
    ap.add_argument("--trackers", required=True,
                    help="tracker endpoints to seed (e.g. track_{00..49} or "
                         "http://nodeA:12048/track_{0..24},http://nodeB:12048/track_{0..24})")
    ap.add_argument("--combiner", default="combiner",
                    help="combiner endpoint for full-band |A| status (name or URL)")
    ap.add_argument("--interval", type=float, default=0.25,
                    help="control-loop period, s (keep below ~0.5 s drift time)")
    ap.add_argument("--acquire-snr", type=float, default=12.0,
                    help="min detection SNR to (re)seed a PRN")
    ap.add_argument("--drop-amplitude", type=float, default=0.3,
                    help="combined |A| below which a tracked PRN is a drop candidate")
    ap.add_argument("--drop-hits", type=int, default=8,
                    help="consecutive low-|A| polls before dropping a PRN")
    ap.add_argument("--lat", type=float, help="receiver latitude (enables gating)")
    ap.add_argument("--lon", type=float, help="receiver longitude")
    ap.add_argument("--alt", type=float, default=0.0, help="receiver altitude, m")
    ap.add_argument("--mask-deg", type=float, default=5.0, help="elevation mask, deg")
    ap.add_argument("--almanac", action="store_true",
                    help="orbit-predict each PRN's Doppler (needs --lat/--lon + skyfield): "
                         "seed trackers with the precise predicted Doppler (geometry + a common "
                         "clock-freq bias solved from the measured sats) instead of the coarse "
                         "search grid, and gate to visible sats. Code phase still from the search.")
    ap.add_argument("--tle", default=None, help="GPS TLE file/URL (default: Celestrak gps-ops)")
    ap.add_argument("--carrier-hz", type=float, default=1575.42e6, help="carrier for Doppler pred")
    ap.add_argument("--doppler-sign", type=float, default=1.0,
                    help="multiply predicted Doppler (set -1 if the convention is inverted)")
    ap.add_argument("--once", action="store_true",
                    help="run a single control-loop iteration and exit (for tests)")
    args = ap.parse_args(argv)

    base = args.rest_url.rstrip("/")
    detectors = parse_endpoints(args.detectors, base)
    trackers = parse_endpoints(args.trackers, base)
    combiner = resolve_prefix(args.combiner, base)
    gating = args.lat is not None and args.lon is not None

    # Almanac assist: load the GPS constellation once (TLEs), predict per-cycle.
    almanac_sats = None
    if args.almanac:
        if not gating:
            _log("--almanac needs --lat/--lon; disabling")
            args.almanac = False
        else:
            try:
                from gps_beamtrack import load_gps_satellites, predict_dopplers
                from gps_beamtrack import DEFAULT_TLE_URL
                almanac_sats = load_gps_satellites(args.tle or DEFAULT_TLE_URL)
                _log("almanac: loaded %d GPS TLEs; predicting Doppler @ (%.4f, %.4f)"
                     % (len(almanac_sats), args.lat, args.lon))
            except Exception as e:
                _log("almanac unavailable (%s); falling back to search Doppler" % e)
                args.almanac = False

    seeds = {}       # prn -> {"doppler_hz", "code_phase_chips"} (consensus)
    low_hits = {}    # prn -> consecutive low-|A| poll count
    _log("detectors=%d trackers=%d combiner=%s interval=%.2fs gating=%s"
         % (len(detectors), len(trackers), combiner, args.interval, gating))
    _log("trackers: %s" % (trackers if len(trackers) <= 6
                           else "%s ... %s (%d)" % (trackers[0], trackers[-1], len(trackers))))

    while True:
        t0 = time.time()
        # 1. collect best-SNR detection per PRN across all detection sources
        best = {}  # prn -> (snr, dop, cp)
        for d_ep in detectors:
            try:
                dets = _get("%s/get_detections" % d_ep)
            except Exception as e:
                _log("get_detections %s failed: %s" % (d_ep, e))
                continue
            for d in dets:
                prn, snr = int(d["prn"]), float(d["snr"])
                if snr < args.acquire_snr:
                    continue
                if prn not in best or snr > best[prn][0]:
                    best[prn] = (snr, float(d["doppler_hz"]), float(d["code_phase_chips"]))

        # 2. orbit-predicted Doppler + visibility (almanac assist), else plain gate
        pred = {}          # prn -> (doppler_hz, rate_hz_s, elev_deg) [sign-applied]
        clock_bias = 0.0   # common receiver clock-frequency bias, Hz
        up = None
        if args.almanac:
            try:
                from gps_beamtrack import predict_dopplers
                raw = predict_dopplers(args.lat, args.lon, args.alt,
                                       t_utc=datetime.now(tz=timezone.utc), _sats=almanac_sats,
                                       f_carrier_hz=args.carrier_hz)
                pred = {p: (args.doppler_sign * d, r, e) for p, (d, r, e) in raw.items()}
            except Exception as e:
                _log("predict_dopplers failed: %s" % e)
            up = {p for p, v in pred.items() if v[2] >= args.mask_deg}
            # Common clock-frequency bias = median(measured - predicted) over detected
            # sats. A tight residual spread confirms the sign convention; a wild spread
            # (resid ~ -2x predicted) means flip --doppler-sign.
            resid = [best[p][1] - pred[p][0] for p in best if p in pred]
            if resid:
                clock_bias = statistics.median(resid)
            for p in sorted(best):
                if p in pred:
                    _log("PRN %d: meas %+.0f  pred %+.0f  resid %+.0f Hz (elev %.0f)"
                         % (p, best[p][1], pred[p][0], best[p][1] - pred[p][0], pred[p][2]))
            if resid:
                _log("clock-freq bias %+.0f Hz (%d sats) -> seeding predicted Doppler"
                     % (clock_bias, len(resid)))
        elif gating:
            up = visible_prns(args.lat, args.lon, args.alt, args.mask_deg, 0.0)

        # refresh / add consensus seeds: code phase from the search, Doppler from the
        # orbit prediction when available (precise enough for coherent integration),
        # else the coarse search grid.
        for prn, (snr, dop, cp) in best.items():
            if up is not None and prn not in up:
                continue
            seed_dop = (pred[prn][0] + clock_bias) if (args.almanac and prn in pred) else dop
            seeds[prn] = {"doppler_hz": seed_dop, "code_phase_chips": cp}
            low_hits[prn] = 0

        # 3. drop on lost lock (sustained low combined |A|) or set below horizon
        try:
            status = {int(r["prn"]): float(r["amplitude"])
                      for r in _get("%s/get_status" % combiner)}
        except Exception as e:
            status = {}
            _log("get_status failed: %s" % e)
        for prn in list(seeds):
            if up is not None and prn not in up:
                _log("drop PRN %d (below horizon)" % prn)
                del seeds[prn]
                low_hits.pop(prn, None)
                continue
            amp = status.get(prn, 0.0)
            if prn not in best:  # not re-detected this poll; judge by combined |A|
                low_hits[prn] = low_hits.get(prn, 0) + (1 if amp < args.drop_amplitude else 0)
                if low_hits[prn] >= args.drop_hits:
                    _log("drop PRN %d (lock lost, |A|=%.2f)" % (prn, amp))
                    del seeds[prn]
                    low_hits.pop(prn, None)

        # 4. push consensus seeds to every tracker
        payload = [dict(prn=prn, **v) for prn, v in sorted(seeds.items())]
        ok = 0
        for t_ep in trackers:
            try:
                _post("%s/set_seeds" % t_ep, payload)
                ok += 1
            except Exception as e:
                _log("set_seeds %s failed: %s" % (t_ep, e))
        _log("active=%s (%d); seeded %d/%d trackers" % (sorted(seeds), len(seeds), ok, len(trackers)))

        if args.once:
            return
        dt = args.interval - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


if __name__ == "__main__":
    main()
