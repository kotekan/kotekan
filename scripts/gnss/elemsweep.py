#!/usr/bin/env python3
"""Which antenna should the search use? Ask the array, don't guess.

    elemsweep.py                      # per-element power on every node, one table
    elemsweep.py --set 9              # point every node's search tap at element 9
    elemsweep.py --sweep 1,4,5,9,13 --dwell 90    # try each, rank by SEARCH SNR

WHY THIS EXISTS. On 2026-08-07 the search's reference element (0) was dark. Because the
search correlates a SINGLE element against a perfect replica, it despread literal zeros for
an entire session: no detections, and -- the tell nobody read correctly for hours -- a best
peak BELOW the pure-noise ceiling, because correlating against zeros gives a surface
smoother than noise. Finding it meant dumping a buffer and decoding base64 by hand; fixing
it meant regenerating seven configs and restarting six nodes.

POWER IS NOT HEALTH, and this is the point of --sweep. Mean power separates DARK from
NOT-DARK and nothing else: a self-oscillating amplifier is loud, an amplifier with the sky
disconnected is loud, and a healthy feed is loud. They differ in whether the samples contain
GPS -- which only a despread can answer. So --sweep re-points the fleet, waits, and ranks
candidates by the SEARCH SNR they actually deliver against the aggregator's own noise
ceiling. Use --set only once --sweep has told you what to set.
"""

import argparse
import json
import sys
import time
import urllib.request

NODES = ["cx19", "cx27", "cx42", "cx43", "cx44", "cx51"]
PORT = 12049
AGG = "cf06:12050"
TAPS = ["gnss0_srch_tap", "gnss1_srch_tap"]


def _get(url, timeout=8):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def _post(url, payload, timeout=8):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def power_table(nodes):
    """Per-element mean power, per node. The tracking tap reports EVERY element."""
    out = {}
    for n in nodes:
        try:
            # the tracking tap sees all elements; the search tap sees one
            d = _get("http://%s:%d/gnss0_tap/element_power" % (n, PORT))
            out[n] = d.get("element_power", [])
        except Exception as e:
            print("  %s: unreachable (%s)" % (n, e), file=sys.stderr)
    return out


def print_power(tbl):
    if not tbl:
        print("no nodes answered -- is element_power registered? (needs the 2026-08-07 build)")
        return
    ne = max(len(v) for v in tbl.values())
    nodes = sorted(tbl)
    print("element power (mean |sample|^2 per element, per node)")
    print("  elem " + "".join("%10s" % n for n in nodes) + "   verdict")
    ref = [sorted(v)[len(v) // 2] for v in tbl.values() if v]
    med = sorted(ref)[len(ref) // 2] if ref else 0.0
    for e in range(ne):
        row = [tbl[n][e] if e < len(tbl[n]) else float("nan") for n in nodes]
        lo = min(row)
        hi = max(row)
        verdict = "DARK" if hi < 0.02 * med else ("hot?" if lo > 4 * med else "live")
        print("  %4d " % e + "".join("%10.2f" % v for v in row) + "   " + verdict)
    print("\n  median %.2f. DARK = no signal path. 'hot?' = >4x median, which is as consistent")
    print("  with an oscillating amp as with a good feed -- POWER CANNOT TELL THEM APART.")
    print("  Rank candidates with --sweep, which uses search SNR.")


def set_element(nodes, elem):
    ok = 0
    for n in nodes:
        for tap in TAPS:
            try:
                r = _post("http://%s:%d/%s/set_element" % (n, PORT, tap), {"element": elem})
                print("  %s/%s: %s -> %s" % (n, tap, r.get("previous"), r.get("element_offset")))
                ok += 1
            except Exception as e:
                print("  %s/%s: FAILED (%s)" % (n, tap, e), file=sys.stderr)
    return ok


def search_snr(samples=6, gap=5.0):
    """Best pass SNR and the aggregator's own pure-noise ceiling, over a few passes."""
    best, ceil = [], []
    for _ in range(samples):
        try:
            d = _get("http://%s/gps_search/get_status" % AGG)
            if isinstance(d, dict):
                if "best_snr" in d:
                    best.append(float(d["best_snr"]))
                if "noise_ceiling" in d:
                    ceil.append(float(d["noise_ceiling"]))
        except Exception:
            pass
        time.sleep(gap)
    return (max(best) if best else float("nan"), max(ceil) if ceil else float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nodes", nargs="*", default=NODES)
    ap.add_argument("--set", type=int, default=None, help="point every search tap at this element")
    ap.add_argument("--sweep", default=None, help="comma-separated elements to rank by search SNR")
    ap.add_argument("--dwell", type=float, default=90.0,
                    help="seconds to hold each candidate before scoring (default 90)")
    a = ap.parse_args()

    if a.set is not None:
        print("setting search element -> %d" % a.set)
        set_element(a.nodes, a.set)
        return

    if a.sweep:
        cands = [int(x) for x in a.sweep.split(",") if x.strip()]
        print("sweeping %s, %.0f s each. Ranking by SEARCH SNR -- power cannot tell a healthy\n"
              "feed from an oscillating amp, a despread can.\n" % (cands, a.dwell))
        res = []
        for e in cands:
            if not set_element(a.nodes, e):
                print("  element %d: no node accepted the change, skipping" % e)
                continue
            time.sleep(a.dwell)
            snr, ceil = search_snr()
            res.append((snr, ceil, e))
            print("  element %2d: best search snr %.2f (noise ceiling %.2f)\n" % (e, snr, ceil))
        if res:
            res.sort(reverse=True)
            print("ranking (best first):")
            for snr, ceil, e in res:
                mark = "  <- above the ceiling" if snr == snr and ceil == ceil and snr > ceil \
                       else "  (at or below noise -- not usable)"
                print("  element %2d: %.2f vs %.2f%s" % (e, snr, ceil, mark))
        return

    print_power(power_table(a.nodes))


if __name__ == "__main__":
    main()
