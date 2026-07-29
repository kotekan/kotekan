#!/usr/bin/env python3
"""Map the 1176.45 MHz GNSS signals onto the CHORD comb, per node.

A Python mirror of lib/stages/gnss/gnssBandPlan.cpp: same half_bandwidth /
occupied_band / covering_channels semantics, including the open-interval
passband overlap test (a channel counts if its passband overlaps the occupied
band at all, not merely if its centre falls inside).

Used to answer, before any config exists: which freq_ids does each node hold
that carry the signal, how much of the main lobe do a given set of nodes see,
and where does the delay response alias?

    python3 config/chord_band_plan.py                 # report for all nodes
    python3 config/chord_band_plan.py --nodes cx19 cx27
"""

import argparse
import math
import os

import yaml

CONF = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NODE_FILE = os.path.join(CONF, "chord_gnss_node.yaml")

# The GNSS base chipping/subcarrier unit that BOC orders multiply (gnssBandPlan.cpp).
BASE_RATE_HZ = 1.023e6


def half_bandwidth_hz(chip_rate_hz, boc_m=0, boc_n=0, modulation="BPSK"):
    """One-sided half-occupied bandwidth of the main spectral lobe(s)."""
    if modulation == "BOC":
        # BOC(m,n): main lobes at +/- m*1.023 MHz, upper outer null a further n out.
        return (boc_m + boc_n) * BASE_RATE_HZ
    # BPSK: first spectral null is one chip rate from the carrier.
    return chip_rate_hz


def occupied_band(carrier_hz, chip_rate_hz, max_doppler_hz, **kw):
    """Sky band the signal occupies, widened by worst-case |Doppler|."""
    half = half_bandwidth_hz(chip_rate_hz, **kw) + max_doppler_hz
    return carrier_hz - half, carrier_hz + half


def covering_channels(chans, carrier_hz, chip_rate_hz, max_doppler_hz, **kw):
    """freq_ids whose passband overlaps the occupied band, ordered by centre.

    `chans` is an iterable of (freq_id, centre_hz, width_hz).
    """
    lo, hi = occupied_band(carrier_hz, chip_rate_hz, max_doppler_hz, **kw)
    hits = []
    for fid, centre, width in chans:
        pass_lo = centre - 0.5 * width
        pass_hi = centre + 0.5 * width
        if pass_lo < hi and pass_hi > lo:  # open-interval overlap, as in the C++
            hits.append((centre, fid))
    hits.sort()
    return [fid for _, fid in hits]


def node_channels(cfg, node):
    """Every freq_id the node holds, as (freq_id, centre_hz, width_hz)."""
    sb = cfg["science_band"]
    width = cfg["fengine"]["channel_width_hz"]
    off = cfg["nodes"][node]["comb_offset_mod8"]
    return [
        (fid, fid * width, width)
        for fid in range(sb["min_freq_id"], sb["max_freq_id"] + 1)
        if fid % 8 == off
    ]


def all_band_channels(cfg):
    """Every science-band freq_id, regardless of which node holds it."""
    sb = cfg["science_band"]
    width = cfg["fengine"]["channel_width_hz"]
    return [(fid, fid * width, width) for fid in range(sb["min_freq_id"], sb["max_freq_id"] + 1)]


def delay_response(freq_ids, width_hz, carrier_hz, span_ns=3000.0, n=300001):
    """Peak sidelobe of the comb's delay response, and where it sits.

    R(tau) = |sum_c exp(2j*pi*(f_c - f_carrier)*tau)| / N. Excludes the main
    lobe (|tau| < 150 ns, ~3 main-lobe widths for a 20 MHz span).
    """
    import numpy as np

    f = np.asarray(freq_ids, dtype=float) * width_hz - carrier_hz
    tau = np.linspace(0.0, span_ns * 1e-9, n)
    r = np.abs(np.exp(2j * math.pi * np.outer(tau, f)).sum(axis=1)) / len(f)
    r[tau < 150e-9] = 0.0
    k = int(r.argmax())
    return tau[k] * 1e9, 20.0 * math.log10(max(r[k], 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--node-file", default=DEFAULT_NODE_FILE)
    ap.add_argument("--nodes", nargs="*", default=None,
                    help="nodes to combine (default: report each, then cumulative)")
    args = ap.parse_args()

    with open(args.node_file) as fh:
        cfg = yaml.safe_load(fh)

    sig = cfg["signals"]
    carrier = float(sig["carrier_hz"])
    chip = float(sig["chip_rate_hz"])
    dopp = float(sig["max_doppler_hz"])
    width = cfg["fengine"]["channel_width_hz"]

    lo, hi = occupied_band(carrier, chip, dopp)
    full = covering_channels(all_band_channels(cfg), carrier, chip, dopp)
    print(f"signal      : {sig['primary']} (+ {', '.join(sig['also_available'])}, same channels)")
    print(f"carrier     : {carrier/1e6:.2f} MHz, chip rate {chip/1e6:.2f} MHz")
    print(f"occupied    : {lo/1e6:.3f} .. {hi/1e6:.3f} MHz  (|Doppler| <= {dopp/1e3:.1f} kHz)")
    print(f"main lobe   : {len(full)} channels of {cfg['fengine']['num_bins']} "
          f"({width/1e3:.4f} kHz each)")
    print()

    order = args.nodes if args.nodes else cfg["nodes"]["bringup_order"]
    hdr = f"{'nodes':<22}{'ch':>4}{'% lobe':>9}{'dSNR':>8}{'1st alias':>12}{'height':>9}"
    print(hdr)
    print("-" * len(hdr))

    cumulative = []
    for node in order:
        got = covering_channels(node_channels(cfg, node), carrier, chip, dopp)
        cumulative.append(node)
        combined = sorted(
            set().union(*[set(covering_channels(node_channels(cfg, n), carrier, chip, dopp))
                          for n in cumulative]))
        frac = len(combined) / len(full)
        ns, db = delay_response(combined, width, carrier)
        label = "+".join(n[2:] for n in cumulative)
        print(f"{label:<22}{len(combined):>4}{frac*100:>8.1f}%{10*math.log10(frac):>+7.1f}dB"
              f"{ns:>10.0f} ns{db:>8.1f} dB")
        if len(cumulative) == 1:
            print(f"  {node} alone holds freq_ids {got[:3]}...{got[-1]} "
                  f"({got[0]*width/1e6:.2f}..{got[-1]*width/1e6:.2f} MHz)")

    print()
    print("dSNR is relative to seeing the whole main lobe; the alias is the first delay")
    print("ambiguity outside the main lobe. Compare against a BRDC model delay uncertainty")
    print("of roughly 10-40 ns -- every configuration above clears it by >=16x.")


if __name__ == "__main__":
    main()
