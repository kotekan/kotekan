#!/usr/bin/env python3
"""Run the viewer's F-engine health rows in a real JS engine.

    <venv>/bin/python scripts/gnss/rf_panel_gate.py

The viewer is served straight to the browser with no build step, so a mistake shows up as a
silently dead panel, and python/scripts/js_viewer/check_js.sh needs node. A syntax check also
cannot see what matters here: whether "not armed" renders differently from "quiet". A dark
strip that reads as "no RFI" when it means "nothing is measuring" is worse than no strip at
all, and that is behaviour, not parsing.

So this loads the real panel source (app/panels/airspy_stats.js) into QuickJS, calls the real
_gnss_block() and _update() against synthetic broker documents, and asserts on the HTML they
return.

It tests the renderer, not the page: no DOM, no CSS, no browser. Run check_js.sh on a host
with node before trusting the whole file.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # scripts/gnss -> scripts -> root
SRC = os.path.join(
    ROOT, "python", "scripts", "js_viewer", "app", "panels", "airspy_stats.js"
)

try:
    import quickjs
except ImportError:
    print("needs quickjs: python3 -m venv V && V/bin/pip install quickjs")
    sys.exit(2)


def make(doc):
    """Instantiate the real class body with a stubbed constructor and call _gnss_block."""
    src = open(SRC).read()
    body = src[src.index("export class AirspyStatsPanel") :]
    body = body.replace("export class", "class", 1)
    # Replace only the constructor: we want the real methods, not a real DOM.
    body = re.sub(
        r"constructor\(\{target, feed\}\) \{.*?\n    \}\n",
        "constructor() { this._gnss = null; }\n",
        body,
        count=1,
        flags=re.S,
    )
    ctx = quickjs.Context()
    ctx.eval(body)
    ctx.eval("var P = new AirspyStatsPanel();")
    ctx.eval("P._gnss = %s;" % json.dumps(doc))
    return ctx.eval("P._gnss_block()")


def make_update(payload):
    """Drive _update (the feed callback) with a stubbed DOM el, return innerHTML."""
    src = open(SRC).read()
    body = src[src.index("export class AirspyStatsPanel") :]
    body = body.replace("export class", "class", 1)
    body = re.sub(
        r"constructor\(\{target, feed\}\) \{.*?\n    \}\n",
        "constructor() { this.el = {innerHTML:''}; this._gnss = null; }\n",
        body,
        count=1,
        flags=re.S,
    )
    ctx = quickjs.Context()
    ctx.eval(body)
    ctx.eval("var P = new AirspyStatsPanel();")
    ctx.eval("P._update(%s);" % json.dumps(payload))
    return ctx.eval("P.el.innerHTML")


def inst(lobes, state="on"):
    return {"state": state, "lobes": lobes}


def lobe(
    i, c0, c1, lo=0.0, hi=0.0, pw=1.0, band=None, mhz=None, f0=None, f1=None, clip=None
):
    """One lobe as the broker's fits.rf_lobes emits it. `band`/`carrier_mhz` present = a node
    that serves freq_ids and a broker with --rf-bands declared; absent = the unnamed shape of
    a node that serves no freq_ids, which the panel must still render. Both rails sit on c0
    unless `clip` (the worst single channel's lo + hi) says they are on different channels."""
    d = {
        "lobe": i,
        "chan0": c0,
        "chan1": c1,
        "n_chan": c1 - c0 + 1,
        "power": pw,
        "power_max": pw,
        "clip_lo": lo,
        "clip_lo_chan": c0,
        "clip_hi": hi,
        "clip_hi_chan": c0,
        "clip": lo + hi if clip is None else clip,
        "clip_chan": c0,
    }
    if band is not None:
        f0 = f0 if f0 is not None else 1548 + 16 * c0
        f1 = f1 if f1 is not None else 1548 + 16 * c1
        d.update(
            {
                "band": band,
                "carrier_mhz": mhz,
                "freq_id0": f0,
                "freq_id1": f1,
                "mhz0": round(f0 * 0.1953125, 3),
                "mhz1": round(f1 * 0.1953125, 3),
            }
        )
    return d


# The unnamed shape (no band identity): the panel must render it as unnamed rows, never
# merged with named ones.
TWO = [lobe(0, 277, 283), lobe(1, 287, 293)]

# Eight chains collapse to five RF bands. L2 is the case lobe-index keying gets wrong: one
# channel, and only 5 of 12 instances own one.
BANDED = [
    lobe(0, 277, 283, band="L5", mhz=1176.45),
    lobe(1, 287, 292, band="E5b", mhz=1207.14),
    lobe(2, 306, 310, band="B3", mhz=1268.52),
    lobe(3, 311, 314, band="E6", mhz=1278.75),
]
BANDED_L2 = [
    BANDED[0],
    BANDED[1],
    lobe(2, 296, 296, band="L2", mhz=1227.60),
    lobe(3, 306, 310, band="B3", mhz=1268.52),
    lobe(4, 311, 314, band="E6", mhz=1278.75),
]
RED, AMBER, GREEN = "#d64550", "#e8a13c", "#3fb26f"


def main():
    fails = []

    # 1. Healthy fleet, unnamed shape: two lobe rows, both green, all instances.
    doc = {
        "t": 1.0,
        "instances": {
            "http://cx%d:12048/gnss%d_srch_tap" % (n, g): inst(TWO)
            for n in (19, 27, 42, 43, 44, 51)
            for g in (0, 1)
        },
    }
    h = make(doc)
    # Count row labels ("lobe N · ch "), not a bare "lobe N", so tooltip prose never counts
    # as a row.
    n0 = h.count("lobe 0 \u00b7 ch ")
    n1 = h.count("lobe 1 \u00b7 ch ")
    if (n0, n1) != (1, 1):
        fails.append("expected one row per lobe, got lobe0=%d lobe1=%d" % (n0, n1))
    elif "12/12" not in h:
        fails.append("armed count wrong; no 12/12 in output")
    elif RED in h or AMBER in h:
        fails.append("healthy fleet rendered a warning colour")
    else:
        print("ok  healthy fleet, legacy unnamed lobes -> 2 rows, 12/12, all green")

    # 1b. Five named bands, in carrier order, no warning colour. This arm fails if rows are
    #     keyed by lobe index: L2 sits at index 2 on 5 instances and B3 at index 2 on the
    #     other 7.
    insts = {}
    for n in (19, 27, 42, 43, 51):
        for g in (0, 1):
            insts["http://cx%d:12048/gnss%d_srch_tap" % (n, g)] = inst(
                BANDED_L2 if g == 1 else BANDED
            )
    for g in (0, 1):
        insts["http://cx44:12048/gnss%d_srch_tap" % g] = inst(
            BANDED
        )  # cx44 owns no L2C
    h = make({"t": 1.0, "instances": insts})
    order = [h.find("%s \u00b7 " % b) for b in ("L5", "E5b", "L2", "B3", "E6")]
    if any(o < 0 for o in order):
        fails.append(
            "band rows missing: %s" % dict(zip(("L5", "E5b", "L2", "B3", "E6"), order))
        )
    elif order != sorted(order):
        fails.append("band rows out of carrier order: %s" % order)
    elif h.count("L2 \u00b7 ") != 1:
        fails.append(
            "L2 rendered %d times, want exactly one row" % h.count("L2 \u00b7 ")
        )
    elif "5/12" not in h:
        fails.append("L2's partial coverage not shown as 5/12")
    elif AMBER in h:
        fails.append("a correctly-partial band (L2 on 5 of 12) rendered as a warning")
    elif "1176.45 MHz" not in h or "1227.60 MHz" not in h:
        fails.append("band rows do not carry the carrier frequency")
    else:
        print(
            "ok  8-chain fleet -> 5 BAND rows in carrier order; L2 on 5/12 is grey, not amber"
        )

    # 1c. Named and unnamed instances together (a rolling restart) must not merge, and the
    #     banner must say the unnamed rows are not comparable.
    mixed_i = dict(insts)
    mixed_i["http://cx19:12048/gnss0_srch_tap"] = inst(TWO)
    h = make({"t": 1.0, "instances": mixed_i})
    if "L5 \u00b7 " not in h:
        fails.append("named rows vanished when one instance served no band")
    elif "lobe 0 \u00b7 ch " not in h:
        fails.append(
            "unnamed instance's lobes were dropped instead of rendered separately"
        )
    elif "not comparable" not in h:
        fails.append("mixed named/unnamed fleet did not warn")
    else:
        print(
            "ok  mixed named/unnamed instances -> both render, banner says not comparable"
        )

    # 2. The one that matters: not armed must not look like quiet.
    doc_off = {
        "t": 1.0,
        "instances": {"http://cx19:12048/gnss0_srch_tap": {"state": "off"}},
    }
    h = make(doc_off)
    if GREEN in h:
        fails.append("an UNARMED monitor rendered green -- reads as 'no RFI'")
    elif "not armed" not in h:
        fails.append("unarmed state does not say so: %s" % h[:200])
    elif "NOT a measurement" not in h:
        fails.append("unarmed row does not disclaim the zero")
    else:
        print(
            "ok  monitor OFF -> grey, says 'not armed', disclaims the zero (not green)"
        )

    # 3. Unreachable is a third state, distinct from off and from quiet.
    doc_un = {
        "t": 1.0,
        "instances": {"http://cx19:12048/gnss0_srch_tap": {"state": "unreachable"}},
    }
    h = make(doc_un)
    if "unreachable" not in h or GREEN in h:
        fails.append("unreachable not distinguished: %s" % h[:200])
    else:
        print("ok  instance UNREACHABLE -> its own wording, still not green")

    # 4. A rail on one instance in twelve must go red and name that instance: the reason
    #    clip is maxed, not meaned.
    bad = {
        "http://cx%d:12048/gnss%d_srch_tap" % (n, g): inst(TWO)
        for n in (19, 27, 42, 43, 44, 51)
        for g in (0, 1)
    }
    bad["http://cx43:12048/gnss0_srch_tap"] = inst(
        [lobe(0, 277, 283, lo=0.20), lobe(1, 287, 293)]
    )
    h = make({"t": 1.0, "instances": bad})
    if RED not in h:
        fails.append("20%% rail on one instance did not go red")
    elif "cx43" not in h:
        fails.append("the worst instance was not named")
    else:
        print("ok  20% rail on 1 of 12 -> RED and cx43 named (max, not mean)")

    # 4b. The rail is one channel's: a low rail on one channel and a high rail on another,
    #     0.8% each, are each under the 1% red bar. Summing the separate maxima would read
    #     1.6% and go red; a broker that serves no `clip` still gets that sum.
    two = {
        "http://cx19:12048/gnss0_srch_tap": inst(
            [lobe(0, 277, 283, lo=0.008, hi=0.008, clip=0.008), lobe(1, 287, 293)]
        )
    }
    h = make({"t": 1.0, "instances": two})
    old = json.loads(json.dumps(two))
    del old["http://cx19:12048/gnss0_srch_tap"]["lobes"][0]["clip"]
    h_old = make({"t": 1.0, "instances": old})
    if RED in h or AMBER not in h or "0.80%" not in h:
        fails.append("rails on two channels summed: %s" % re.findall(r">[\d.]+%<", h))
    elif RED not in h_old:
        fails.append("a lobe without `clip` did not fall back to the sum")
    else:
        print("ok  0.8% low + 0.8% high on two channels -> 0.80% amber, not 1.6% red")

    # 5. Band-selective power must show as a difference between the two rows: that is what
    #    identifies a band-selective external source.
    sel = {
        "http://cx19:12048/gnss0_srch_tap": inst(
            [lobe(0, 277, 283, pw=40.0), lobe(1, 287, 293, pw=1.0)]
        )
    }
    h = make({"t": 1.0, "instances": sel})
    if "40.0" not in h or "1.00" not in h:
        fails.append(
            "band-selective power not shown per lobe: %s" % re.findall(r">[\d.]+<", h)
        )
    else:
        print(
            "ok  +16 dB on lobe 0 only -> 40.0 vs 1.00, one row each (a band-selective source)"
        )

    # 6. No RF document renders nothing: an airspy-only page must not grow an empty strip.
    if make({"t": None, "instances": {}}) != "" or make(None) != "":
        fails.append("empty document rendered a strip")
    else:
        print("ok  no RF document -> renders nothing (airspy pages unaffected)")

    # 7. Unnamed instances that disagree on lobe count must warn: lobe index is then the
    #    only key, and a count disagreement means rows may be different bands. With named
    #    bands, differing counts are normal (only 5 of 12 instances own an L2C channel; arm
    #    1b checks that).
    mix = {
        "http://cx19:12048/gnss0_srch_tap": inst(TWO),
        "http://cx27:12048/gnss0_srch_tap": inst([lobe(0, 277, 283)]),
    }
    h = make({"t": 1.0, "instances": mix})
    if "disagree on lobe count" not in h:
        fails.append("mixed lobe counts did not warn")
    else:
        print(
            "ok  unnamed instances disagree on lobe count -> warns instead of blending"
        )

    # 8. RFI SK: a jammed element (SK outside the flag band) -> red, worst SK shown, worst
    #    instance named.
    sk_doc = {
        "t": 1.0,
        "instances": {
            "http://cx43:12048/gnss0_srch_tap": {
                "state": "on",
                "lobes": TWO,
                "sk": {
                    "sk_n": 32,
                    "sk_flagged": 3,
                    "sk_max": 5.2,
                    "sk_worst": 7,
                    "sk_med": 1.0,
                    "ema_frames": 256,
                    "sk_lo": 0.625,
                    "sk_hi": 1.375,
                },
            },
            "http://cx44:12048/gnss0_srch_tap": {
                "state": "on",
                "lobes": TWO,
                "sk": {
                    "sk_n": 32,
                    "sk_flagged": 0,
                    "sk_max": 1.1,
                    "sk_worst": 2,
                    "sk_med": 1.0,
                    "ema_frames": 256,
                    "sk_lo": 0.625,
                    "sk_hi": 1.375,
                },
            },
        },
    }
    h = make(sk_doc)
    if "RFI · SK" not in h:
        fails.append("SK row absent when instances carry sk")
    elif RED not in h or "5.20" not in h:
        fails.append("flagged SK did not render red / worst value: %s" % h[-400:])
    elif "cx43" not in h:
        fails.append("SK row did not name the worst instance")
    else:
        print("ok  RFI SK jam -> RED, worst SK 5.20 and cx43 named")

    # 9. Clean SK on every instance -> SK row present, no red.
    clean = {
        "t": 1.0,
        "instances": {
            "http://cx43:12048/gnss0_srch_tap": {
                "state": "on",
                "lobes": TWO,
                "sk": {
                    "sk_n": 32,
                    "sk_flagged": 0,
                    "sk_max": 1.12,
                    "sk_worst": 1,
                    "sk_med": 1.0,
                    "ema_frames": 256,
                    "sk_lo": 0.625,
                    "sk_hi": 1.375,
                },
            }
        },
    }
    h = make(clean)
    if "RFI · SK" not in h or RED in h:
        fails.append("clean SK rendered red or absent")
    else:
        print("ok  clean SK -> green, no false RFI alarm")

    # 10. Drops: a nonzero drop counter renders amber with its count.
    dr = {
        "t": 1.0,
        "instances": {
            "http://cx27:12048/gnss0_srch_tap": {
                "state": "on",
                "lobes": TWO,
                "drops": {
                    "srch_send_drops": 80537,
                    "telem_send_drops": 92488,
                    "dpdk_missed": 44490,
                    "dpdk_ring_full": 0,
                },
            }
        },
    }
    h = make(dr)
    if "drops" not in h:
        fails.append("drops row absent when instances carry drops")
    elif AMBER not in h or "80537" not in h:
        fails.append("nonzero drops did not render amber / localeString: %s" % h[-400:])
    else:
        print("ok  drops -> amber, 80537 (browser adds commas) search-send drops shown")

    # 11. The feed callback paints the F-engine (GNSS) section and nothing else.
    gnss = {"t": 1.0, "instances": {"http://cx43:12048/gnss0_srch_tap": inst(TWO)}}
    h = make_update({"gnss_rf": gnss})
    if "front end" in h or "valve" in h or "ADC" in h:
        fails.append("the feed callback rendered an airspy strip: %s" % h[:200])
    elif "F-engine (GNSS)" not in h:
        fails.append("the feed callback dropped the GNSS section")
    else:
        print("ok  feed callback -> GNSS section only")

    print("-" * 70)
    if fails:
        for f in fails:
            print("FAIL: %s" % f)
        return 1
    print("GATE GOOD: 14 arms, run in QuickJS against the real panel source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
