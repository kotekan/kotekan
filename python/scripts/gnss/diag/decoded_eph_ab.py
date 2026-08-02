#!/usr/bin/env python3
"""Standalone A/B: the decoded-eph BRDC fallback vs the network BRDC, on the LIVE node.

Validates `decoded_eph.predict_from_decoders` (the network-loss fallback) against
`gnss_ephemeris.predict_all` (the network BRDC truth) WITHOUT touching the production brokers: it
polls the data-channel combiners' `nav_obs` off the REST server, feeds them to FRESH decoder
instances (the same InavPredictor/FnavPredictor/CnavPredictor/Bcnav1/2Predictor the broker uses),
and once a decoder has extracted an ephemeris, predicts the satellite two ways and differences
them. Zero seed posting, zero risk to the soak -- and it exercises exactly what the offline test
could not: the seconds-of-week frame conversion and BeiDou's BDT = GPST-14 s offset in the
fallback wrapper, across every constellation.

The per-satellite dpos cross-check the broker already logs validates the PROPAGATORS; this
validates the fallback's geometry wrapper on top of them (range + range-rate / Doppler).

    python3 diag/decoded_eph_ab.py [--rest http://localhost:12048] [--duration 240]
                                   [--lat .. --lon .. --alt ..]
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import decoded_eph as de
import gnss_ephemeris as ge

# signal -> (combiner stage, sys, predictor module.Class, code module.sv_position fn, toe field)
REG = [
    ("E1B  I/NAV",  "l1_e1b_combiner",  "E", "inav_predictor.InavPredictor",
     "galileo_inav.sv_position_inav",  "t0e"),
    ("E5aI F/NAV",  "l5_e5a_i_combiner", "E", "fnav_predictor.FnavPredictor",
     "galileo_fnav.sv_position_fnav",  "t0e"),
    ("B1C  B-CNAV1", "l1_b1c_d_combiner", "C", "bcnav1_predictor.Bcnav1Predictor",
     "beidou_bcnav1.sv_position_bcnav1", "t_oe"),
    ("B2a  B-CNAV2", "l5_b2a_d_combiner", "C", "bcnav2_predictor.Bcnav2Predictor",
     "beidou_bcnav2.sv_position_bcnav2", "t_oe"),
    ("L2C  CNAV",   "l2c_gps_combiner", "G", "cnav_predictor.CnavPredictor",
     "gps_cnav.sv_position_cnav",      "toe"),
    ("L5   CNAV",   "l5_i_combiner",    "G", "cnav_predictor.CnavPredictor",
     "gps_cnav.sv_position_cnav",      "toe"),
]


def _imp(dotted):
    mod, name = dotted.rsplit(".", 1)
    return getattr(__import__(mod), name)


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read().decode())


def _toe_gpst(eph, toef, gpst_now):
    """Decoded toe (seconds-of-week) placed in NOW's continuous GPS week, nearest-week fold --
    identical to the broker's _decoded_entries."""
    wk = int(gpst_now // 604800)
    t = wk * 604800 + eph[toef]
    if t - gpst_now > 302400:
        t -= 604800
    elif t - gpst_now < -302400:
        t += 604800
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rest", default="http://localhost:12048")
    ap.add_argument("--duration", type=float, default=240.0, help="max poll seconds")
    ap.add_argument("--lat", type=float, default=43.968697)
    ap.add_argument("--lon", type=float, default=-79.252106)
    ap.add_argument("--alt", type=float, default=260.0)
    ap.add_argument("--min-sats", type=int, default=3, help="stop a signal once it has this many eph")
    args = ap.parse_args()

    decoders, svfn, toef, sysid, stage = {}, {}, {}, {}, {}
    for label, st, sy, dcl, svp, tf in REG:
        try:
            decoders[label] = _imp(dcl)(log=lambda m: None)
            svfn[label], toef[label], sysid[label], stage[label] = _imp(svp), tf, sy, st
        except Exception as e:
            print("skip %s: %s" % (label, e))

    print("polling %d data channels for nav_obs; decoding until >=%d sats have an ephemeris "
          "(max %.0fs)..." % (len(decoders), args.min_sats, args.duration))
    t0 = time.time()
    done = set()
    while time.time() - t0 < args.duration and len(done) < len(decoders):
        for label, dec in decoders.items():
            if label in done:
                continue
            try:
                rows = _get("%s/%s/get_status" % (args.rest, stage[label]))
            except Exception:
                continue
            for r in rows:
                if "nav_obs" in r and r.get("prn"):
                    try:
                        dec.ingest(int(r["prn"]), r["nav_obs"])
                    except Exception:
                        pass
            neph = sum(1 for prn in list(dec._p) if _safe_eph(dec, prn) is not None)
            if neph >= args.min_sats:
                done.add(label)
        el = time.time() - t0
        got = {lb: sum(1 for prn in list(d._p) if _safe_eph(d, prn) is not None)
               for lb, d in decoders.items()}
        print("  t=%3.0fs  eph/signal: %s" % (el, " ".join("%s:%d" % (lb.split()[0], n)
                                                            for lb, n in got.items())))
        if len(done) < len(decoders):
            time.sleep(10)

    # BRDC truth, once
    try:
        brdc = ge.parse_rinex_nav(ge.fetch_brdc())
    except Exception as e:
        print("BRDC fetch failed (%s) -- cannot A/B" % e)
        return
    now = ge.datetime.now(ge.timezone.utc)
    pa = ge.predict_all(brdc, args.lat, args.lon, args.alt, now, mask_deg=-90.0)
    gpst_now = ge.gpst_of_utc(now)

    print("\n=== A/B: decoded-eph fallback vs network BRDC ===")
    print("%-13s %5s %12s %14s %10s" % ("signal", "sats", "worst range", "worst rng-rate",
                                        "median rng"))
    for label, dec in decoders.items():
        ents = []
        for prn in sorted(dec._p):
            e = _safe_eph(dec, prn)
            if e is None or toef[label] not in e:
                continue
            ents.append((sysid[label], prn, label, e, svfn[label],
                         _toe_gpst(e, toef[label], gpst_now)))
        if not ents:
            print("%-13s %5s  (no ephemeris decoded)" % (label, 0))
            continue
        pf = de.predict_from_decoders(ents, args.lat, args.lon, args.alt, now, mask_deg=-90.0)
        pairs = []
        for (sy, prn, *_r) in ents:
            k = (sy, prn)
            if k in pf and k in pa:
                pairs.append((abs(pf[k]["range_m"] - pa[k]["range_m"]),
                              abs(pf[k]["range_rate_mps"] - pa[k]["range_rate_mps"])))
        if not pairs:
            print("%-13s %5d  (no BRDC overlap)" % (label, len(ents)))
            continue
        wr = max(p[0] for p in pairs)
        wrr = max(p[1] for p in pairs)
        med = sorted(p[0] for p in pairs)[len(pairs) // 2]
        flag = "  <-- CHECK" if wr > 100.0 else ""
        print("%-13s %5d %10.1f m %11.4f m/s %8.1f m%s"
              % (label, len(pairs), wr, wrr, med, flag))
    print("\nRange agreement to ~metres validates the fallback's frame handling (seconds-of-week; "
          "BeiDou BDT-14 s). A CHECK flag = the frame is wrong for that constellation.")


def _safe_eph(dec, prn):
    try:
        return dec.ephemeris(prn)
    except Exception:
        return None


if __name__ == "__main__":
    main()
