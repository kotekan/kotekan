#!/usr/bin/env python3
"""The C/N0 history panel's sample dedup: a repeated C/N0 estimate gaps the C/N0 traces but must
not drop the sample, or every other observable is thinned to the estimator's cadence; a poll
that repeats everything adds nothing.

    PYTHON_GIL=0 /home/kvand/gnss/venv-ft/bin/python test_amp_history.py

It runs the shipped code: _blank() and _push() are extracted from
app/panels/gps_amp_history.js by brace matching and executed with dukpy, as test_lock_metrics
does for gps_feed.js.
"""
import json
import re
import sys

import dukpy

from test_lock_metrics import brace_block

HERE = __file__.rsplit("/", 1)[0]
PANEL = HERE + "/app/panels/gps_amp_history.js"

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def build_js():
    src = open(PANEL).read()
    consts = "\n".join(
        l for l in src.split("\n") if re.match(r"const (MAX_PTS|CN0_MIN_DUTY)\b", l)
    )
    blank = "function " + brace_block(src, r"_blank\(\) \{")
    push = "function " + brace_block(src, r"_push\(key, src, now, snr\) \{")
    return """
%s
var panel = {hist: new Map(), selected: null, _dirty: false};
panel._blank = %s;
panel._push = %s;
function run(samples) {
    for (var i = 0; i < samples.length; i++)
        panel._push("G11|gps_l5", samples[i], i, samples[i].snr);
    return panel.hist.get("G11|gps_l5");
}
""" % (
        consts,
        blank,
        push,
    )


def run(samples):
    return dukpy.evaljs(build_js() + "run(%s);" % json.dumps(samples))


def sample(**kw):
    s = dict(cn0=38.2, cn0_duty=0.9, cn0_kcoh=36.0, kcoh_eta=40.0, kcoh_n=96, q=3.1)
    s.update(sig=12.0, coh_s=1.0, dop=-1200.0, snr=None, beam_amp=1.0, beam_ph=20.0)
    s.update(kw)
    return s


def main():
    print("a repeated C/N0 estimate keeps the sample and gaps the C/N0")
    h = run([sample(), sample(q=3.4)])
    check(h["q"] == [3.1, 3.4], "q keeps both samples: %s" % h["q"])
    check(h["cn0"] == [38.2, None], "cn0 shows one point per estimate: %s" % h["cn0"])
    check(
        h["cn0_kcoh"] == [36.0, None],
        "cn0_kcoh shows one point per estimate: %s" % h["cn0_kcoh"],
    )

    print("a new estimate is plotted even when nothing else moved")
    h = run([sample(), sample(cn0=38.5, cn0_kcoh=36.1)])
    check(h["cn0"] == [38.2, 38.5], "cn0: %s" % h["cn0"])

    print("a duty-disqualified cn0 still dedups on the served value")
    h = run([sample(cn0_duty=0.1), sample(cn0_duty=0.1, sig=12.5)])
    check(h["sig"] == [12.0, 12.5], "sig keeps both samples: %s" % h["sig"])
    check(
        h["cn0_kcoh"] == [36.0, None],
        "the repeated coherent estimate is not plotted twice: %s" % h["cn0_kcoh"],
    )

    print("a poll that repeats every observable adds nothing")
    h = run([sample(), sample()])
    check(len(h["t"]) == 1, "%d sample(s)" % len(h["t"]))
    h = run([sample(), sample(beam_ph=25.0)])
    check(
        h["beam_ph"] == [20.0, 25.0], "beam_ph is part of the sample: %s" % h["beam_ph"]
    )

    print("\n%s (%d failure(s))" % ("FAIL" if _fails else "ALL PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
