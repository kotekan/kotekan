#!/usr/bin/env python3
"""CALIBRATED peel-depth estimator: deep-coherent suppression from twin combiner chains.

Polls two GnssCoherentCombiner instances -- one fed by a tracker on the BASELINE channelized
voltage, one by an identical tracker on the PEEL RESIDUAL (config/peel_depth_bench.yaml) -- and
reports, per PRN,
    depth_dB = 20*log10(deep_snr_base / deep_snr_resid)
The T_coh term of cn0_coh = 20log10(deep_snr) - 10log10(coherence_s) is common to both chains and
cancels, so this is a calibrated ratio in dB with no threshold or grid-statistic artifacts.

WHY NOT the obvious alternatives (all tried and rejected 2026-07-23):
  * GnssChannelizedSearch snr -- a grid peak-to-noise DETECTION statistic. On identical configs it
    read base ~61/resid ~43-48 at acquire_windows=500 but ~67-70/~2-7 at aw=100 (both directions
    backwards), and the variant RANKING flipped between settings. Not a depth measurement.
  * The peel's own get_status peel_db -- DEGENERATE: it despreads the residual with the SAME
    replica just subtracted, so v1 (which removes the exact per-record projection) reads ~0
    residual = infinite depth. Any estimator must re-track the residual INDEPENDENTLY.

FLOOR HANDLING: a good peel drives the residual into the combiner's rectification floor. That is
reported explicitly ("resid AT FLOOR -> depth > X dB") rather than silently censored -- the whole
failure mode of the search-snr metric.

Usage:
    ./peel_depth.py --url http://localhost:12060 --prns 10,32 --samples 30 --interval 1.0
"""
import argparse
import json
import math
import statistics
import time
import urllib.request


def _get(url, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _by_prn(rows, prns):
    out = {}
    for r in rows:
        try:
            p = int(r.get("prn", -1))
        except (TypeError, ValueError):
            continue
        if p in prns:
            out[p] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:12060")
    ap.add_argument("--base", default="comb_base", help="combiner stage on the BASELINE stream")
    ap.add_argument("--peel", default="comb_peel", help="combiner stage on the RESIDUAL stream")
    ap.add_argument("--prns", required=True, help="comma-separated PRNs being peeled")
    ap.add_argument("--samples", type=int, default=30, help="status polls to accumulate")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--warmup", type=float, default=8.0,
                    help="seconds to discard first (deep integration + gain EMA must converge)")
    ap.add_argument("--json", default="", help="also write the summary to this path")
    args = ap.parse_args()

    prns = [int(p) for p in args.prns.split(",") if p.strip()]
    u_base = "%s/%s/get_status" % (args.url.rstrip("/"), args.base.strip("/"))
    u_peel = "%s/%s/get_status" % (args.url.rstrip("/"), args.peel.strip("/"))

    if args.warmup > 0:
        time.sleep(args.warmup)

    acc = {p: {"base": [], "peel": [], "floor_hits": 0, "n": 0} for p in prns}
    for _ in range(args.samples):
        try:
            b = _by_prn(_get(u_base), set(prns))
            q = _by_prn(_get(u_peel), set(prns))
        except Exception as e:
            print("  poll failed (%s)" % e, flush=True)
            time.sleep(args.interval)
            continue
        for p in prns:
            rb, rq = b.get(p), q.get(p)
            if not rb or not rq:
                continue
            sb, sq = rb.get("deep_snr"), rq.get("deep_snr")
            fq = rq.get("deep_floor")
            if not sb or sb <= 0:
                continue
            acc[p]["n"] += 1
            acc[p]["base"].append(float(sb))
            if sq and sq > 0:
                acc[p]["peel"].append(float(sq))
                # residual sitting at/below the combiner's rectification floor = depth is a bound
                if fq and float(sq) <= float(fq) * 1.05:
                    acc[p]["floor_hits"] += 1
        time.sleep(args.interval)

    print("\nCALIBRATED PEEL DEPTH (deep-coherent, twin combiner chains)")
    print("  PRN    n    deep_snr base    deep_snr resid    depth_dB     note")
    summary = {}
    for p in prns:
        a = acc[p]
        if not a["base"] or not a["peel"]:
            print("  %3d   %3d    (insufficient status samples)" % (p, a["n"]))
            continue
        mb = statistics.median(a["base"])
        mq = statistics.median(a["peel"])
        depth = 20.0 * math.log10(mb / mq) if mq > 0 else float("inf")
        # spread of the per-poll depth, for an honest error bar
        pairs = list(zip(a["base"], a["peel"]))
        ds = [20.0 * math.log10(x / y) for x, y in pairs if y > 0]
        sd = statistics.pstdev(ds) if len(ds) > 1 else 0.0
        sm = 1.253 * sd / math.sqrt(len(ds)) if ds else 0.0
        floored = a["floor_hits"] > 0.5 * len(a["peel"])
        note = "resid AT FLOOR -> depth is a LOWER BOUND" if floored else ""
        print("  %3d   %3d    %11.3f    %13.3f    %6.2f+-%.2f  %s"
              % (p, a["n"], mb, mq, depth, sm, note))
        summary[p] = {"n": a["n"], "deep_snr_base": mb, "deep_snr_resid": mq,
                      "depth_db": depth, "sigma_db": sm, "resid_at_floor": floored}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print("  -> %s" % args.json)


if __name__ == "__main__":
    main()
