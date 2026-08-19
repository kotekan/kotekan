#!/usr/bin/env python3
"""#8: run the viewer's F-engine health rows in a real JS engine.

    <venv>/bin/python scripts/gnss/rf_panel_gate.py

WHY THIS EXISTS. The viewer is served straight to the browser with no build step, so a
mistake shows up as a SILENTLY DEAD PANEL -- and check_js.sh needs node, which is not
installed on cx19. Worse, a syntax check would not have caught the thing that actually
matters here: whether "not armed" renders differently from "quiet". A dark strip that reads
as "no RFI" when it means "nothing is measuring" is worse than no strip at all, and that is
a BEHAVIOUR, not a parse.

So this loads the real panel source into QuickJS, calls the real _gnss_block() against
synthetic broker documents, and asserts on the HTML it returns.

⚠️ It tests the RENDERER, not the page: no DOM, no CSS, no browser. Run check_js.sh on a
host with node before trusting the whole file.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/gnss -> scripts -> root
SRC = os.path.join(ROOT, "python", "scripts", "js_viewer", "app", "panels", "airspy_stats.js")

try:
    import quickjs
except ImportError:
    print("needs quickjs: python3 -m venv V && V/bin/pip install quickjs")
    sys.exit(2)


def make(doc):
    """Instantiate the real class body with a stubbed constructor and call _gnss_block."""
    src = open(SRC).read()
    body = src[src.index("export class AirspyStatsPanel"):]
    body = body.replace("export class", "class", 1)
    # Replace only the constructor: we want the real methods, not a real DOM.
    body = re.sub(r"constructor\(\{target, feed\}\) \{.*?\n    \}\n",
                  "constructor() { this._rf = new Map(); this._gnss = null; }\n",
                  body, count=1, flags=re.S)
    ctx = quickjs.Context()
    ctx.eval(body)
    ctx.eval("var P = new AirspyStatsPanel();")
    ctx.eval("P._gnss = %s;" % json.dumps(doc))
    return ctx.eval("P._gnss_block()")


def inst(lobes, state="on"):
    return {"state": state, "lobes": lobes}


def lobe(i, c0, c1, lo=0.0, hi=0.0, pw=1.0):
    return {"lobe": i, "chan0": c0, "chan1": c1, "n_chan": c1 - c0 + 1,
            "power": pw, "power_max": pw, "clip_lo": lo, "clip_lo_chan": c0,
            "clip_hi": hi, "clip_hi_chan": c0}


TWO = [lobe(0, 277, 283), lobe(1, 287, 293)]
RED, AMBER, GREEN = "#d64550", "#e8a13c", "#3fb26f"


def main():
    fails = []

    # 1. HEALTHY FLEET: two lobe rows, both green, all instances counted.
    doc = {"t": 1.0, "instances": {"http://cx%d:12048/gnss%d_srch_tap" % (n, g): inst(TWO)
                                   for n in (19, 27, 42, 43, 44, 51) for g in (0, 1)}}
    h = make(doc)
    # Count ROW LABELS, not the substring: the label tooltip legitimately says "lobe 0 is
    # the lower band", so a bare h.count("lobe 0") counts prose as rows.
    rows = re.findall(r"lobe (\d) &middot; ch |lobe (\d) \u00b7 ch ", h)
    n0 = h.count("lobe 0 \u00b7 ch "); n1 = h.count("lobe 1 \u00b7 ch ")
    if (n0, n1) != (1, 1):
        fails.append("expected one row per lobe, got lobe0=%d lobe1=%d" % (n0, n1))
    elif "12/12" not in h:
        fails.append("armed count wrong; no 12/12 in output")
    elif RED in h or AMBER in h:
        fails.append("healthy fleet rendered a warning colour")
    else:
        print("ok  healthy 12-instance fleet -> 2 lobe rows, 12/12, all green")

    # 2. ⚠️ THE ONE THAT MATTERS: NOT ARMED must not look like QUIET.
    doc_off = {"t": 1.0, "instances": {"http://cx19:12048/gnss0_srch_tap": {"state": "off"}}}
    h = make(doc_off)
    if GREEN in h:
        fails.append("an UNARMED monitor rendered green -- reads as 'no RFI'")
    elif "not armed" not in h:
        fails.append("unarmed state does not say so: %s" % h[:200])
    elif "NOT a measurement" not in h:
        fails.append("unarmed row does not disclaim the zero")
    else:
        print("ok  monitor OFF -> grey, says 'not armed', disclaims the zero (not green)")

    # 3. UNREACHABLE is a third state, distinct from off and from quiet.
    doc_un = {"t": 1.0, "instances": {"http://cx19:12048/gnss0_srch_tap": {"state": "unreachable"}}}
    h = make(doc_un)
    if "unreachable" not in h or GREEN in h:
        fails.append("unreachable not distinguished: %s" % h[:200])
    else:
        print("ok  instance UNREACHABLE -> its own wording, still not green")

    # 4. A RAIL ON ONE INSTANCE MUST GO RED and name that instance -- the whole point of
    #    maxing rather than meaning. One bad instance in twelve.
    bad = {"http://cx%d:12048/gnss%d_srch_tap" % (n, g): inst(TWO)
           for n in (19, 27, 42, 43, 44, 51) for g in (0, 1)}
    bad["http://cx43:12048/gnss0_srch_tap"] = inst([lobe(0, 277, 283, lo=0.20), lobe(1, 287, 293)])
    h = make({"t": 1.0, "instances": bad})
    if RED not in h:
        fails.append("20%% rail on one instance did not go red")
    elif "cx43" not in h:
        fails.append("the worst instance was not named")
    else:
        print("ok  20% rail on 1 of 12 -> RED and cx43 named (max, not mean)")

    # 5. BAND-SELECTIVE POWER must be visible as a DIFFERENCE between the two rows --
    #    the 2026-08-18 signature. If both rows showed the same number it would be useless.
    sel = {"http://cx19:12048/gnss0_srch_tap":
           inst([lobe(0, 277, 283, pw=40.0), lobe(1, 287, 293, pw=1.0)])}
    h = make({"t": 1.0, "instances": sel})
    if "40.0" not in h or "1.00" not in h:
        fails.append("band-selective power not shown per lobe: %s"
                     % re.findall(r">[\d.]+<", h))
    else:
        print("ok  +16 dB on lobe 0 only -> 40.0 vs 1.00, one row each (the 08-18 look)")

    # 6. NOTHING AT ALL renders nothing -- an airspy-only page must not grow an empty strip.
    if make({"t": None, "instances": {}}) != "" or make(None) != "":
        fails.append("empty document rendered a strip")
    else:
        print("ok  no RF document -> renders nothing (airspy pages unaffected)")

    # 7. MIXED LOBE COUNTS must warn rather than silently compare different bands.
    mix = {"http://cx19:12048/gnss0_srch_tap": inst(TWO),
           "http://cx27:12048/gnss0_srch_tap": inst([lobe(0, 277, 283)])}
    h = make({"t": 1.0, "instances": mix})
    if "disagree on lobe count" not in h:
        fails.append("mixed lobe counts did not warn")
    else:
        print("ok  instances disagree on lobe count -> warns instead of blending")

    # 8. RFI SK: a jammed channel (SK outside band) -> RED, flagged count shown, worst named.
    sk_doc = {"t": 1.0, "instances": {
        "http://cx43:12048/gnss0_srch_tap": {"state": "on", "lobes": TWO,
            "sk": {"sk_n": 32, "sk_flagged": 3, "sk_max": 5.2, "sk_worst": 7,
                   "sk_med": 1.0, "ema_frames": 256, "sk_lo": 0.625, "sk_hi": 1.375}},
        "http://cx44:12048/gnss0_srch_tap": {"state": "on", "lobes": TWO,
            "sk": {"sk_n": 32, "sk_flagged": 0, "sk_max": 1.1, "sk_worst": 2,
                   "sk_med": 1.0, "ema_frames": 256, "sk_lo": 0.625, "sk_hi": 1.375}}}}
    h = make(sk_doc)
    if "RFI · SK" not in h:
        fails.append("SK row absent when instances carry sk")
    elif RED not in h or "5.20" not in h:
        fails.append("flagged SK did not render red / worst value: %s" % h[-400:])
    elif "cx43" not in h:
        fails.append("SK row did not name the worst instance")
    else:
        print("ok  RFI SK jam -> RED, worst SK 5.20 and cx43 named")

    # 9. CLEAN SK on every instance -> green, zero flagged, no false alarm.
    clean = {"t": 1.0, "instances": {"http://cx43:12048/gnss0_srch_tap": {"state": "on",
        "lobes": TWO, "sk": {"sk_n": 32, "sk_flagged": 0, "sk_max": 1.12, "sk_worst": 1,
        "sk_med": 1.0, "ema_frames": 256, "sk_lo": 0.625, "sk_hi": 1.375}}}}
    h = make(clean)
    if "RFI · SK" not in h or RED in h:
        fails.append("clean SK rendered red or absent")
    else:
        print("ok  clean SK -> green, no false RFI alarm")

    # 10. DROPS: a non-zero drop counter renders (amber) and the worst instance is named.
    dr = {"t": 1.0, "instances": {"http://cx27:12048/gnss0_srch_tap": {"state": "on",
        "lobes": TWO, "drops": {"srch_send_drops": 80537, "telem_send_drops": 92488,
        "dpdk_missed": 44490, "dpdk_ring_full": 0}}}}
    h = make(dr)
    if "drops" not in h:
        fails.append("drops row absent when instances carry drops")
    elif AMBER not in h or "80537" not in h:
        fails.append("nonzero drops did not render amber / localeString: %s" % h[-400:])
    else:
        print("ok  drops -> amber, 80537 (browser adds commas) search-send drops shown")

    print("-" * 70)
    if fails:
        for f in fails:
            print("FAIL: %s" % f)
        return 1
    print("GATE GOOD: 10 arms, run in QuickJS against the real panel source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
