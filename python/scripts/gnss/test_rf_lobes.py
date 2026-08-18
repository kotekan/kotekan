#!/usr/bin/env python3
"""#8: rf_lobes -- does contiguity actually recover the two bands?

    python3 python/scripts/gnss/test_rf_lobes.py

The fixture is the real deployed layout: cx19/gnss0's monitor watches 14 LOCAL channels,
277-283 (the 1176.45 MHz lobe) and 287-293 (1207.14), because the generator hands it the
union of every chain's covering set on that GPU. Contiguity is the band grouping.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnss_broker.fits import rf_lobes  # noqa: E402

CX19_G0 = list(range(277, 284)) + list(range(287, 294))   # as generated, verified 08-18


def main():
    fails = []

    # 1. THE REAL LAYOUT -> two lobes, correctly split.
    n = len(CX19_G0)
    lob = rf_lobes(CX19_G0, [1.0] * n, [0.0] * n, [0.0] * n)
    if len(lob) != 2:
        fails.append("expected 2 lobes from the deployed layout, got %d" % len(lob))
    elif (lob[0]["chan0"], lob[0]["chan1"],
          lob[1]["chan0"], lob[1]["chan1"]) != (277, 283, 287, 293):
        fails.append("lobe edges wrong: %s" % [(l["chan0"], l["chan1"]) for l in lob])
    else:
        print("ok  14 channels -> 2 lobes, 277-283 and 287-293 (the two bands)")

    # 2. BAND-SELECTIVE POWER IS THE WHOLE POINT: +16 dB on one lobe, the other flat,
    #    which is what the 2026-08-18 event looked like. The lobes must NOT be blended.
    pw = [40.0] * 7 + [1.0] * 7
    lob = rf_lobes(CX19_G0, pw, [0.0] * 14, [0.0] * 14)
    if abs(lob[0]["power"] - 40.0) > 1e-9 or abs(lob[1]["power"] - 1.0) > 1e-9:
        fails.append("band-selective power blended: %s" % [l["power"] for l in lob])
    else:
        print("ok  +16 dB on lobe 0 only -> 40.0 vs 1.0, kept apart (the 08-18 signature)")

    # 3. CLIP IS MAXED, NOT MEANED. One railing channel among seven quiet ones is the
    #    narrowband case this exists to catch; a mean would report 0.14 and hide it.
    cl = [0.0] * 14
    cl[3] = 1.0                       # local channel 280 pinned at the rail
    lob = rf_lobes(CX19_G0, [1.0] * 14, cl, [0.0] * 14)
    if abs(lob[0]["clip_lo"] - 1.0) > 1e-9:
        fails.append("one railing channel was averaged away: clip_lo=%.4f (mean would be "
                     "%.4f)" % (lob[0]["clip_lo"], 1.0 / 7))
    elif lob[0]["clip_lo_chan"] != 280:
        fails.append("named the wrong channel: %s" % lob[0]["clip_lo_chan"])
    else:
        print("ok  1 railing channel of 7 -> clip_lo 1.00 and NAMED (280), not meaned to 0.14")

    # 4. THE RAILS STAY APART. -8 is what negate_4bit corrupts; +7 is only headroom.
    lo, hi = [0.0] * 14, [0.0] * 14
    lo[0], hi[8] = 0.5, 0.25
    lob = rf_lobes(CX19_G0, [1.0] * 14, lo, hi)
    if lob[0]["clip_lo"] != 0.5 or lob[0]["clip_hi"] != 0.0:
        fails.append("lobe 0 rails merged: %s" % lob[0])
    elif lob[1]["clip_hi"] != 0.25 or lob[1]["clip_lo"] != 0.0:
        fails.append("lobe 1 rails merged: %s" % lob[1])
    else:
        print("ok  low rail in lobe 0, high rail in lobe 1 -> reported separately")

    # 5. UNSORTED INPUT must not invent lobes -- the tap serves whatever order it walks.
    import random
    sh = list(CX19_G0)
    random.Random(7).shuffle(sh)
    pw = [40.0 if c < 285 else 1.0 for c in sh]
    lob = rf_lobes(sh, pw, [0.0] * 14, [0.0] * 14)
    if len(lob) != 2 or abs(lob[0]["power"] - 40.0) > 1e-9:
        fails.append("shuffled input broke the grouping: %s"
                     % [(l["chan0"], l["chan1"], l["power"]) for l in lob])
    else:
        print("ok  shuffled channel order -> same 2 lobes (sorted internally)")

    # 6. DEGENERATE: one channel, and nothing at all.
    if len(rf_lobes([5], [1.0], [0.0], [0.0])) != 1:
        fails.append("a single channel should be one lobe")
    elif rf_lobes([], [], [], []) != []:
        fails.append("empty input should be no lobes, not a crash")
    else:
        print("ok  single channel -> 1 lobe; empty -> [] (no crash)")

    print("-" * 70)
    if fails:
        for f in fails:
            print("FAIL: %s" % f)
        return 1
    print("GATE GOOD: 6 arms on the real deployed channel layout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
