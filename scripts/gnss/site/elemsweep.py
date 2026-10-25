#!/usr/bin/env python3
"""Which antenna should the search use? Ask the array, don't guess.

    elemsweep.py                      # per-element power on every node, one table
    elemsweep.py --set 9              # point every node's search tap at element 9

The nodes are --nodes, by default the fleet (config/gnss_fleet.py hosts).

The search correlates a single element, the one its tap copies, against a replica, so a dark
element gives no detections and a best peak below the pure-noise ceiling (correlating against
zeros gives a surface smoother than noise). This table finds a dark element from the running
nodes; --set re-points the running taps without a config change (a restart restores the
config's element_offset).

Power is not health. Mean power separates dark from not-dark and nothing else: a
self-oscillating amplifier is loud, an amplifier with the sky disconnected is loud, and a
healthy feed is loud. They differ in whether the samples contain GPS, which only a despread
can answer, and this table is not one.
"""

import argparse
import json
import os
import sys
import urllib.request

CONF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "config"
)
PORT = 12048
TAPS = ["gnss0_srch_tap", "gnss1_srch_tap"]


def fleet_hosts():
    """The fleet's hosts in slot order: config/gnss_fleet_chord.yaml nodes:, read by
    config/gnss_fleet.py."""
    sys.path.insert(0, CONF)
    try:
        import gnss_fleet

        return gnss_fleet.hosts()
    except Exception as e:
        sys.exit(
            "no fleet host list from %s/gnss_fleet.py (%s: %s); name the nodes explicitly"
            % (CONF, type(e).__name__, e)
        )


def _get(url, timeout=8):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def _post(url, payload, timeout=8):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def power_table(nodes):
    """Per-element mean power, per node. The search tap copies one element, but its
    element_power covers every element of the input frame."""
    out = {}
    for n in nodes:
        try:
            d = _get("http://%s:%d/gnss0_srch_tap/element_power" % (n, PORT))
            out[n] = d.get("element_power", [])
        except Exception as e:
            print("  %s: unreachable (%s)" % (n, e), file=sys.stderr)
    return out


def print_power(tbl):
    if not tbl:
        print(
            "no nodes answered gnss0_srch_tap/element_power -- are they up on a GNSS config?"
        )
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
    print(
        "\n  median %.2f. DARK = no signal path. 'hot?' = >4x median, which is as consistent"
    )
    print(
        "  with an oscillating amp as with a good feed -- power cannot tell them apart."
    )


def set_element(nodes, elem):
    ok = 0
    for n in nodes:
        for tap in TAPS:
            try:
                r = _post(
                    "http://%s:%d/%s/set_element" % (n, PORT, tap), {"element": elem}
                )
                print(
                    "  %s/%s: %s -> %s"
                    % (n, tap, r.get("previous"), r.get("element_offset"))
                )
                ok += 1
            except Exception as e:
                print("  %s/%s: FAILED (%s)" % (n, tap, e), file=sys.stderr)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--nodes", nargs="*", help="default: the fleet (config/gnss_fleet.py hosts)"
    )
    ap.add_argument(
        "--set",
        type=int,
        default=None,
        help="re-point the search taps (gnss0_srch_tap, gnss1_srch_tap) of every --nodes "
        "node at this element (absolute index in the input frame) at runtime; a node "
        "restart restores the config's element_offset",
    )
    a = ap.parse_args()
    a.nodes = a.nodes or fleet_hosts()

    if a.set is not None:
        print("setting search element -> %d" % a.set)
        set_element(a.nodes, a.set)
        return

    print_power(power_table(a.nodes))


if __name__ == "__main__":
    main()
