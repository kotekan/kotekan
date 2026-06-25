#!/usr/bin/env python3
"""Distributed GNSS broker: search detections -> consensus seeds -> trackers.

The control plane of the distributed-band channelized pipeline
(GnssChannelizedSearch x M subbands -> GnssChannelizedTracker x M -> GnssCoherentCombiner).

Each iteration:
  1. GET <search>/get_detections from every subband search; keep the best-SNR
     detection per PRN (a subband only has to find a sat in *one* slice).
  2. (optional) gate by visibility: drop below-horizon PRNs.
  3. GET <combiner>/get_status for the full-band |A| per PRN; drop a tracked PRN
     after it stays below --drop-amplitude for --drop-hits consecutive polls
     (lock lost).
  4. POST the consensus seed set [{prn, doppler_hz, code_phase_chips}] to *every*
     tracker's /set_seeds, so all subbands despread the same (cp, Doppler) and the
     per-subband products recombine coherently.

Seeds are refreshed from the latest detection every iteration, so the trackers'
code phase stays fresh against slow code-Doppler drift -- run --interval well below
the ~0.5 s decorrelation time. REST via stdlib urllib; visibility via skyfield
(optional, only if --lat/--lon given).
"""

import argparse
import json
import sys
import time
import urllib.request


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
                    help="kotekan REST base URL")
    ap.add_argument("--searches", required=True,
                    help="comma-separated search stage names (e.g. search_a,search_b)")
    ap.add_argument("--trackers", required=True,
                    help="comma-separated tracker stage names (e.g. track_a,track_b)")
    ap.add_argument("--combiner", default="combiner",
                    help="combiner stage name (for full-band |A| status)")
    ap.add_argument("--interval", type=float, default=0.25,
                    help="control-loop period, s (keep below ~0.5 s drift time)")
    ap.add_argument("--acquire-snr", type=float, default=12.0,
                    help="min search SNR to (re)seed a PRN")
    ap.add_argument("--drop-amplitude", type=float, default=0.3,
                    help="combined |A| below which a tracked PRN is a drop candidate")
    ap.add_argument("--drop-hits", type=int, default=8,
                    help="consecutive low-|A| polls before dropping a PRN")
    ap.add_argument("--lat", type=float, help="receiver latitude (enables gating)")
    ap.add_argument("--lon", type=float, help="receiver longitude")
    ap.add_argument("--alt", type=float, default=0.0, help="receiver altitude, m")
    ap.add_argument("--mask-deg", type=float, default=5.0, help="elevation mask, deg")
    args = ap.parse_args(argv)

    base = args.rest_url.rstrip("/")
    searches = [s.strip() for s in args.searches.split(",") if s.strip()]
    trackers = [s.strip() for s in args.trackers.split(",") if s.strip()]
    gating = args.lat is not None and args.lon is not None

    seeds = {}       # prn -> {"doppler_hz", "code_phase_chips"} (consensus)
    low_hits = {}    # prn -> consecutive low-|A| poll count
    _log("searches=%s trackers=%s combiner=%s interval=%.2fs gating=%s"
         % (searches, trackers, args.combiner, args.interval, gating))

    while True:
        t0 = time.time()
        # 1. collect best-SNR detection per PRN across subband searches
        best = {}  # prn -> (snr, dop, cp)
        for s in searches:
            try:
                dets = _get("%s/%s/get_detections" % (base, s))
            except Exception as e:
                _log("get_detections %s failed: %s" % (s, e))
                continue
            for d in dets:
                prn, snr = int(d["prn"]), float(d["snr"])
                if snr < args.acquire_snr:
                    continue
                if prn not in best or snr > best[prn][0]:
                    best[prn] = (snr, float(d["doppler_hz"]), float(d["code_phase_chips"]))

        # 2. visibility gate
        up = visible_prns(args.lat, args.lon, args.alt, args.mask_deg, 0.0) if gating else None

        # refresh / add consensus seeds from fresh detections
        for prn, (snr, dop, cp) in best.items():
            if up is not None and prn not in up:
                continue
            seeds[prn] = {"doppler_hz": dop, "code_phase_chips": cp}
            low_hits[prn] = 0

        # 3. drop on lost lock (sustained low combined |A|) or set below horizon
        try:
            status = {int(r["prn"]): float(r["amplitude"])
                      for r in _get("%s/%s/get_status" % (base, args.combiner))}
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
        for tname in trackers:
            try:
                _post("%s/%s/set_seeds" % (base, tname), payload)
            except Exception as e:
                _log("set_seeds %s failed: %s" % (tname, e))
        _log("active=%s (%d)" % (sorted(seeds), len(seeds)))

        dt = args.interval - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


if __name__ == "__main__":
    main()
