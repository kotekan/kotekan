#!/usr/bin/env python3
"""#8: rf_lobes -- does it recover the fleet's BANDS, and refuse to invent one?

    python3 python/scripts/gnss/test_rf_lobes.py

Two fixtures, both the real deployed layout:

  CX19_G0    the two-band era -- 14 LOCAL channels, 277-283 (1176.45 MHz) and 287-293
             (1207.14), which contiguity alone splits correctly.
  CX19_G1    2026-09-02, eight chains: contiguity alone is NOT ENOUGH any more. gps_l2c
             adds one isolated channel that only 5 of 12 instances own, and bds_b3i and
             gal_e6 ABUT -- 306-314 is one contiguous run holding two bands. Splitting
             those needs the ABSOLUTE freq_id the tap now serves, plus the DECLARED band
             set (--rf-bands); with neither, the grouping must degrade to unnamed rows
             rather than guess.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnss_broker.fits import rf_lobes  # noqa: E402

CX19_G0 = list(range(277, 284)) + list(range(287, 294))   # as generated, verified 08-18

# cx19/gnss1, as generated 2026-09-02: L5 | E5b | L2C(one channel) | B3I+E6 (abutting).
CX19_G1 = (list(range(277, 284)) + list(range(287, 293)) + [296] + list(range(306, 315)))
FIRST16_G1 = 1548                       # freq_id = FIRST16 + 16*local on that instance
G1_FIDS = [FIRST16_G1 + 16 * c for c in CX19_G1]

# The bands the receiver actually flies, as the broker resolves --rf-bands.
from gnss_broker import signals  # noqa: E402
CHORD_CARRIERS = signals.carriers_for_chains(
    ["gps_l5", "gal_e5a", "bds_b2a", "gal_e5b", "bds_b2b", "gps_l2c", "bds_b3i", "gal_e6"])


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

    # 7. THE 8-CHAIN LAYOUT: freq_ids + a declared band set -> five NAMED bands, and the
    #    abutting B3I/E6 run is split, which no gap test can do.
    n = len(CX19_G1)
    lob = rf_lobes(CX19_G1, [1.0] * n, [0.0] * n, [0.0] * n, G1_FIDS, CHORD_CARRIERS)
    got = [(l.get("band"), l["chan0"], l["chan1"]) for l in lob]
    want = [("L5", 277, 283), ("E5b", 287, 292), ("L2", 296, 296),
            ("B3", 306, 310), ("E6", 311, 314)]
    if got != want:
        fails.append("8-chain layout mis-grouped:\n     got  %s\n     want %s" % (got, want))
    else:
        print("ok  8 chains -> 5 named bands; abutting B3I/E6 split by BAND, not by gap")

    # 8. ⚠️ THE FAILURE THAT NEARLY SHIPPED: without a declared band set, "nearest carrier
    #    in the full table" labels E5b's lower shoulder (1199.2, 1202.3 MHz) as GLONASS L3,
    #    which CHORD does not fly. No declaration must mean NO NAME, never a plausible one.
    lob = rf_lobes(CX19_G1, [1.0] * n, [0.0] * n, [0.0] * n, G1_FIDS, None)
    if any(l.get("band") for l in lob):
        fails.append("named bands without a declared set: %s"
                     % [l.get("band") for l in lob])
    elif [(l["chan0"], l["chan1"]) for l in lob] != [(277, 283), (287, 292), (296, 296),
                                                     (306, 314)]:
        fails.append("undeclared fallback did not reproduce plain contiguity: %s"
                     % [(l["chan0"], l["chan1"]) for l in lob])
    else:
        print("ok  no declared band set -> no names, plain contiguity (never a guess)")

    # 9. A RAGGED freq_ids ARRAY must not label channel k with channel j's frequency.
    lob = rf_lobes(CX19_G1, [1.0] * n, [0.0] * n, [0.0] * n, G1_FIDS[:3], CHORD_CARRIERS)
    if any(l.get("band") for l in lob):
        fails.append("a short freq_ids array still produced band names")
    else:
        print("ok  ragged freq_ids -> unnamed, not misaligned")

    # 10. OLD NODE BINARY (no freq_ids at all) on a fleet whose broker DOES declare bands:
    #     the rolling-restart case. Must be unnamed, and identical to arm 1's behaviour.
    lob = rf_lobes(CX19_G0, [1.0] * len(CX19_G0), [0.0] * len(CX19_G0), [0.0] * len(CX19_G0),
                   None, CHORD_CARRIERS)
    if [(l["chan0"], l["chan1"]) for l in lob] != [(277, 283), (287, 293)]:
        fails.append("no-freq_ids path changed the legacy grouping")
    elif any(l.get("band") for l in lob):
        fails.append("named a band with no freq_ids to name it from")
    else:
        print("ok  node without freq_ids -> legacy grouping unchanged, unnamed")

    print("-" * 70)
    if fails:
        for f in fails:
            print("FAIL: %s" % f)
        return 1
    print("GATE GOOD: 10 arms on the real deployed channel layouts (2-band and 8-chain)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
