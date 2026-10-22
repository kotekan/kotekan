#!/usr/bin/env python3
"""rf_lobes: does it recover the fleet's bands, and refuse to invent one?

    python3 python/scripts/gnss/test_rf_lobes.py

Two fixtures, both generated channel layouts:

  CX19_G0    two bands: 14 local channels, 277-283 (1176.45 MHz) and 287-293 (1207.14),
             which contiguity alone splits correctly.
  CX19_G1    eight chains, where contiguity alone is not enough: gps_l2c adds one isolated
             channel that only 5 of 12 instances own, and bds_b3i and gal_e6 abut -- 306-314
             is one contiguous run holding two bands. Splitting those needs the absolute
             freq_id the tap serves plus the declared band set (--rf-bands); with either
             missing, the grouping must degrade to unnamed rows rather than guess.

The last three arms cover the rail over the array's live elements (array.live_element_ranges):
fleet.poll_rf_stats restricts the tap's elem_clip to them, instr_rf_stats reads them from the
node yaml, and the viewer's stream-health panel (rendered under node) shows the row.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnss_broker.fits import rf_lobes  # noqa: E402

CX19_G0 = list(range(277, 284)) + list(range(287, 294))  # cx19/gnss0, two-band layout

# cx19/gnss1, eight-chain layout: L5 | E5b | L2C (one channel) | B3I+E6 (abutting).
CX19_G1 = list(range(277, 284)) + list(range(287, 293)) + [296] + list(range(306, 315))
FIRST16_G1 = 1548  # freq_id = FIRST16 + 16*local on that instance
G1_FIDS = [FIRST16_G1 + 16 * c for c in CX19_G1]

# The bands the receiver flies, resolved as the broker resolves --rf-bands.
from gnss_broker import signals  # noqa: E402

CHORD_CARRIERS = signals.carriers_for_chains(
    [
        "gps_l5",
        "gal_e5a",
        "bds_b2a",
        "gal_e5b",
        "bds_b2b",
        "gps_l2c",
        "bds_b3i",
        "gal_e6",
    ]
)


def main():
    fails = []

    # 1. The two-band layout -> two lobes, correctly split.
    n = len(CX19_G0)
    lob = rf_lobes(CX19_G0, [1.0] * n, [0.0] * n, [0.0] * n)
    if len(lob) != 2:
        fails.append("expected 2 lobes from the deployed layout, got %d" % len(lob))
    elif (lob[0]["chan0"], lob[0]["chan1"], lob[1]["chan0"], lob[1]["chan1"]) != (
        277,
        283,
        287,
        293,
    ):
        fails.append("lobe edges wrong: %s" % [(l["chan0"], l["chan1"]) for l in lob])
    else:
        print("ok  14 channels -> 2 lobes, 277-283 and 287-293 (the two bands)")

    # 2. Band-selective power: +16 dB on one lobe, the other flat, the signature of a
    #    band-selective external source. The lobes must not be blended.
    pw = [40.0] * 7 + [1.0] * 7
    lob = rf_lobes(CX19_G0, pw, [0.0] * 14, [0.0] * 14)
    if abs(lob[0]["power"] - 40.0) > 1e-9 or abs(lob[1]["power"] - 1.0) > 1e-9:
        fails.append("band-selective power blended: %s" % [l["power"] for l in lob])
    else:
        print(
            "ok  +16 dB on lobe 0 only -> 40.0 vs 1.0, kept apart (a band-selective source)"
        )

    # 3. Clip is maxed, not meaned. One railing channel among seven quiet ones is the
    #    narrowband case this exists to catch; a mean would report 0.14 and hide it.
    cl = [0.0] * 14
    cl[3] = 1.0  # local channel 280 pinned at the rail
    lob = rf_lobes(CX19_G0, [1.0] * 14, cl, [0.0] * 14)
    if abs(lob[0]["clip_lo"] - 1.0) > 1e-9:
        fails.append(
            "one railing channel was averaged away: clip_lo=%.4f (mean would be "
            "%.4f)" % (lob[0]["clip_lo"], 1.0 / 7)
        )
    elif lob[0]["clip_lo_chan"] != 280:
        fails.append("named the wrong channel: %s" % lob[0]["clip_lo_chan"])
    else:
        print(
            "ok  1 railing channel of 7 -> clip_lo 1.00 and NAMED (280), not meaned to 0.14"
        )

    # 4. The rails stay apart. -8 is what negate_4bit corrupts; +7 is only headroom.
    lo, hi = [0.0] * 14, [0.0] * 14
    lo[0], hi[8] = 0.5, 0.25
    lob = rf_lobes(CX19_G0, [1.0] * 14, lo, hi)
    if lob[0]["clip_lo"] != 0.5 or lob[0]["clip_hi"] != 0.0:
        fails.append("lobe 0 rails merged: %s" % lob[0])
    elif lob[1]["clip_hi"] != 0.25 or lob[1]["clip_lo"] != 0.0:
        fails.append("lobe 1 rails merged: %s" % lob[1])
    else:
        print("ok  low rail in lobe 0, high rail in lobe 1 -> reported separately")

    # 4b. `clip` is one channel's lo + hi. The separate maxima may come from two channels, and
    #     their sum would report a rail no single channel reaches; the larger of the two
    #     would miss a channel railing both ways.
    lo, hi = [0.0] * 14, [0.0] * 14
    lo[1], hi[2] = 0.01, 0.01  # 278 low rail only, 279 high rail only
    lo[5], hi[5] = 0.006, 0.006  # 282 both, 0.012 together: the worst channel
    lob = rf_lobes(CX19_G0, [1.0] * 14, lo, hi)
    c = lob[0].get("clip")
    if c is None or abs(c - 0.012) > 1e-12 or lob[0].get("clip_chan") != 282:
        fails.append(
            "clip must be the worst single channel (0.012 on 282), got %s on %s; "
            "clip_lo + clip_hi = %.3f" % (c, lob[0].get("clip_chan"), lo[1] + hi[2])
        )
    else:
        print(
            "ok  lo 0.01 on 278, hi 0.01 on 279, both 0.006 on 282 -> clip 0.012 on 282"
        )

    # 5. Unsorted input must not invent lobes: the tap serves whatever order it walks.
    import random

    sh = list(CX19_G0)
    random.Random(7).shuffle(sh)
    pw = [40.0 if c < 285 else 1.0 for c in sh]
    lob = rf_lobes(sh, pw, [0.0] * 14, [0.0] * 14)
    if len(lob) != 2 or abs(lob[0]["power"] - 40.0) > 1e-9:
        fails.append(
            "shuffled input broke the grouping: %s"
            % [(l["chan0"], l["chan1"], l["power"]) for l in lob]
        )
    else:
        print("ok  shuffled channel order -> same 2 lobes (sorted internally)")

    # 6. Degenerate: one channel, and nothing at all.
    if len(rf_lobes([5], [1.0], [0.0], [0.0])) != 1:
        fails.append("a single channel should be one lobe")
    elif rf_lobes([], [], [], []) != []:
        fails.append("empty input should be no lobes, not a crash")
    else:
        print("ok  single channel -> 1 lobe; empty -> [] (no crash)")

    # 7. The eight-chain layout: freq_ids + a declared band set -> five named bands, and the
    #    abutting B3I/E6 run is split, which no gap test can do.
    n = len(CX19_G1)
    lob = rf_lobes(CX19_G1, [1.0] * n, [0.0] * n, [0.0] * n, G1_FIDS, CHORD_CARRIERS)
    got = [(l.get("band"), l["chan0"], l["chan1"]) for l in lob]
    want = [
        ("L5", 277, 283),
        ("E5b", 287, 292),
        ("L2", 296, 296),
        ("B3", 306, 310),
        ("E6", 311, 314),
    ]
    if got != want:
        fails.append(
            "8-chain layout mis-grouped:\n     got  %s\n     want %s" % (got, want)
        )
    else:
        print(
            "ok  8 chains -> 5 named bands; abutting B3I/E6 split by BAND, not by gap"
        )

    # 8. Without a declared band set, "nearest carrier in the full table" would label E5b's
    #    lower shoulder (1199.2, 1202.3 MHz) as GLONASS L3, which CHORD does not fly. No
    #    declaration must mean no name, never a plausible one.
    lob = rf_lobes(CX19_G1, [1.0] * n, [0.0] * n, [0.0] * n, G1_FIDS, None)
    if any(l.get("band") for l in lob):
        fails.append(
            "named bands without a declared set: %s" % [l.get("band") for l in lob]
        )
    elif [(l["chan0"], l["chan1"]) for l in lob] != [
        (277, 283),
        (287, 292),
        (296, 296),
        (306, 314),
    ]:
        fails.append(
            "undeclared fallback did not reproduce plain contiguity: %s"
            % [(l["chan0"], l["chan1"]) for l in lob]
        )
    else:
        print("ok  no declared band set -> no names, plain contiguity (never a guess)")

    # 9. A ragged freq_ids array must not label channel k with channel j's frequency.
    lob = rf_lobes(
        CX19_G1, [1.0] * n, [0.0] * n, [0.0] * n, G1_FIDS[:3], CHORD_CARRIERS
    )
    if any(l.get("band") for l in lob):
        fails.append("a short freq_ids array still produced band names")
    else:
        print("ok  ragged freq_ids -> unnamed, not misaligned")

    # 10. A node that serves no freq_ids, with a broker that declares bands (a rolling
    #     restart): unnamed, and grouped exactly as in arm 1.
    lob = rf_lobes(
        CX19_G0,
        [1.0] * len(CX19_G0),
        [0.0] * len(CX19_G0),
        [0.0] * len(CX19_G0),
        None,
        CHORD_CARRIERS,
    )
    if [(l["chan0"], l["chan1"]) for l in lob] != [(277, 283), (287, 293)]:
        fails.append("no-freq_ids path changed the legacy grouping")
    elif any(l.get("band") for l in lob):
        fails.append("named a band with no freq_ids to name it from")
    else:
        print("ok  node without freq_ids -> legacy grouping unchanged, unnamed")

    fails += live_element_arms()

    print("-" * 70)
    if fails:
        for f in fails:
            print("FAIL: %s" % f)
        return 1
    print(
        "GATE GOOD: 14 arms on the generated channel layouts (2-band and 8-chain) "
        "and the live-element rail"
    )
    return 0


# A 128-element tap: the full-crate inputs outside the array rail at 8%, the array's own at
# 0.37%, except element 70 at 2%.
LIVE = list(range(0, 16)) + list(range(64, 80))
EC128 = [0.0037 if e in LIVE else 0.08 for e in range(128)]
EC128[70] = 0.02
LIVE_MEAN = (31 * 0.0037 + 0.02) / 32


# The array's dark inputs (broken LNAs) read zero power and so zero clip.
DARK = [4, 5, 12, 13]
EC128_DARK = [0.0 if e in DARK else c for e, c in enumerate(EC128)]
LIVE_MEAN_DARK = (27 * 0.0037 + 0.02) / 28


def _tap(ec, dark=()):
    n = len(CX19_G0)
    return {
        "enabled": True,
        "chans": CX19_G0,
        "power": [1.0] * n,
        "clip_lo": [0.0] * n,
        "clip_hi": [0.0] * n,
        "elem_clip": ec,
        "elem_power": [0.0 if e in dark else 1.0 for e in range(len(ec))],
    }


def live_element_arms():
    """The rail over the array's live elements: poll_rf_stats restricts elem_clip to them,
    instr_rf_stats takes them from the node yaml, and the viewer renders the row."""
    import json
    import shutil
    import subprocess
    import tempfile
    import types
    from unittest import mock

    import gnss_arraymap
    from gnss_broker import fleet, instruments

    fails = []
    url = "http://cx19:12048/gnss0_srch_tap"

    # 11. The live-element figures cover the array's elements only, leave out dark inputs
    #     (zero power) rather than count them as clean, and refuse a tap that does not serve
    #     every one of them rather than average a subset.
    with mock.patch.object(fleet, "_get", return_value=_tap(EC128)):
        v = fleet.poll_rf_stats([url], rf_lobes, live_elems=LIVE)[url]
        bare = fleet.poll_rf_stats([url], rf_lobes)[url]
    with mock.patch.object(fleet, "_get", return_value=_tap(EC128_DARK, DARK)):
        dark = fleet.poll_rf_stats([url], rf_lobes, live_elems=LIVE)[url]
    with mock.patch.object(fleet, "_get", return_value=_tap(EC128[:64])):
        short = fleet.poll_rf_stats([url], rf_lobes, live_elems=LIVE)[url]
    if v.get("elem_clip_live") is None or abs(v["elem_clip_live"] - LIVE_MEAN) > 1e-12:
        fails.append(
            "live-element rail %s, want %.5f (the array's elements only)"
            % (v.get("elem_clip_live"), LIVE_MEAN)
        )
    elif (v["elem_clip_live_max"], v["elem_clip_live_worst"]) != (0.02, 70):
        fails.append(
            "worst live element %s at %s, want 70 at 0.02"
            % (v["elem_clip_live_worst"], v["elem_clip_live_max"])
        )
    elif v["elem_clip_max"] != 0.08:
        fails.append("all-element max moved: %s" % v["elem_clip_max"])
    elif bare["elem_clip_live"] is not None or short["elem_clip_live"] is not None:
        fails.append(
            "live-element rail without a live set (%s) or from a 64-element tap (%s)"
            % (bare["elem_clip_live"], short["elem_clip_live"])
        )
    elif dark["elem_clip_live"] is None or (
        abs(dark["elem_clip_live"] - LIVE_MEAN_DARK) > 1e-12
    ):
        fails.append(
            "live-element rail with 4 dark inputs %s, want %.5f (the 28 powered only)"
            % (dark["elem_clip_live"], LIVE_MEAN_DARK)
        )
    else:
        print(
            "ok  rows outside the array at 8%%, the array at 0.37%% -> live rail %.2f%%, "
            "worst element 70; dark inputs left out; none without a live set or past "
            "the tap" % (100 * LIVE_MEAN)
        )

    # 12. The broker's poll takes the live set from the node yaml and logs the figure.
    want = gnss_arraymap._live_ranges(gnss_arraymap.NODE_CONFIG)
    ec = [0.0] * 128
    for e in want or []:
        ec[e] = 0.001 * (1 + e % 3)
    pub = types.SimpleNamespace(rf=None)
    pub.set_rf = lambda rf, t: setattr(pub, "rf", rf)
    ctx = types.SimpleNamespace(
        args=types.SimpleNamespace(
            rf_stats_endpoints=url,
            rf_stats_poll_s=30.0,
            rf_bands="",
            rfi_stats=False,
            drop_stats=False,
        ),
        publisher=pub,
        base="http://localhost:12048",
        t0=1000.0,
        rf_last=[0.0],
    )
    logs = []
    with mock.patch.object(fleet, "_get", return_value=_tap(ec)), mock.patch.object(
        instruments, "_log_rl", side_effect=lambda k, m, every_s=0: logs.append(m)
    ):
        instruments.instr_rf_stats(ctx)
    got = ((pub.rf or {}).get(url) or {}).get("elem_clip_live")
    exp = sum(ec[e] for e in want) / len(want) if want else None
    if not want:
        fails.append("no live_element_ranges in %s" % gnss_arraymap.NODE_CONFIG)
    elif got is None or abs(got - exp) > 1e-12:
        fails.append("instr_rf_stats published live rail %s, want %s" % (got, exp))
    elif not any("live-element rail %.4f" % exp in m for m in logs):
        fails.append("RF PATH does not carry the live-element rail: %s" % logs)
    else:
        print(
            "ok  instr_rf_stats: %d live elements from the node yaml -> live rail "
            "%.4f published and logged" % (len(want), exp)
        )

    # 13. The viewer's stream-health panel renders the live-element row (run under node).
    #     GNSS_REQUIRE_NODE (the test runner sets it): no node fails rather than skips.
    if not shutil.which("node"):
        if os.environ.get("GNSS_REQUIRE_NODE"):
            fails.append(
                "viewer live-element row: no node, and GNSS_REQUIRE_NODE is set"
            )
        else:
            print("skip  viewer live-element row: no node")
        return fails
    panel = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "js_viewer",
        "app",
        "panels",
        "airspy_stats.js",
    )
    d = tempfile.mkdtemp()
    try:
        shutil.copy(panel, os.path.join(d, "panel.mjs"))
        with open(os.path.join(d, "run.mjs"), "w") as f:
            f.write(
                'import {AirspyStatsPanel} from "./panel.mjs";\n'
                "const p = Object.create(AirspyStatsPanel.prototype);\n"
                "p._gnss = JSON.parse(process.argv[2]);\n"
                "console.log(p._gnss_block());\n"
            )
        rf = {"instances": {url: v, "http://cx27:12048/gnss0_srch_tap": bare}}
        r = subprocess.run(
            ["node", os.path.join(d, "run.mjs"), json.dumps(rf)],
            capture_output=True,
            text=True,
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)
    html = r.stdout
    if r.returncode:
        fails.append("panel render failed: %s" % r.stderr.strip())
    elif "live elements" not in html:
        fails.append("the stream-health panel has no live-element row")
    elif "%.2g%%" % (100 * LIVE_MEAN) not in html or "element 70 on cx19" not in html:
        fails.append("live-element row lacks the mean or the worst element")
    elif ">1/2<" not in html:
        fails.append("live-element row miscounts the reporting instances")
    else:
        print("ok  viewer: live-element row, mean, worst element 70, 1/2 reporting")
    return fails


if __name__ == "__main__":
    sys.exit(main())
