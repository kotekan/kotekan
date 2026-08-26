#!/usr/bin/env python3
"""Does the viewer call a dead band 'locked'? Run the SHIPPED js against real rows.

    /home/kvand/gnss/venv-ft/bin/python test_lock_metrics.py [publisher_host:port]

WHY THIS EXISTS. On 2026-08-26 the analog front-ends were powered down for site work. With
nothing on the wire, every satellite on every chain still served sig 900..10,000 -- three
orders of magnitude past the 3-sigma lock bar -- so the sky panel drew a full constellation
of locked satellites, and the retired incoherent C/N0 fallback drew 80-100 dB-Hz beside them.
Both numbers were confidently wrong, and nothing in the viewer disagreed with them.

⚠️ IT PARSES THE REAL FILE. The lock rule and signal_metrics() are extracted from
app/panels/gps_feed.js by brace matching and executed, so this cannot drift from what the
browser runs the way a re-implementation would. A syntax error in the panel fails this too --
which matters, because the app is served with no build step and a syntax error is a silently
dead panel, and `node` is not installed on any host here.

⚠️ AND IT NEEDS BOTH DIRECTIONS. A test that only checks "noise reads unlocked" passes just
as well on a viewer that never reports a lock at all, so the healthy synthetic rows below are
not decoration: they are what makes the negative result mean something.

@author Keith Vanderlinde
"""
import json
import re
import subprocess
import sys

import dukpy

HERE = __file__.rsplit("/", 1)[0]
FEED = HERE + "/app/panels/gps_feed.js"
CHAINS = ("gps_l5", "gal_e5a", "gal_e5b", "bds_b2a", "bds_b2b")

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def brace_block(src, start_pat):
    """The source of a function, from its declaration to the matching close brace."""
    m = re.search(start_pat, src)
    if not m:
        raise SystemExit("cannot find %r in %s -- the panel was restructured" % (start_pat, FEED))
    i = src.index("{", m.start())
    depth, j = 0, i
    while j < len(src):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[m.start():j + 1]
        j += 1
    raise SystemExit("unbalanced braces after %r" % start_pat)


def build_js():
    src = open(FEED).read()
    consts = "\n".join(l for l in src.split("\n")
                       if re.match(r"const (AMP_LOCK|SNR_LOCK|Q_LOCK|Q_FLOOR_MARGIN|"
                                   r"CN0_INC_MAX_RATIO)\b", l))
    for name in ("AMP_LOCK", "SNR_LOCK", "Q_LOCK", "Q_FLOOR_MARGIN", "CN0_INC_MAX_RATIO"):
        if name not in consts:
            raise SystemExit("constant %s missing from %s" % (name, FEED))
    metrics = brace_block(src, r"export function signal_metrics")
    metrics = re.sub(r"^export\s+", "", metrics)
    # The lock rule lives inside _merge(); take it verbatim from `const have_sig` to the
    # statement before `return list;`.
    m = re.search(r"( *const have_sig = .*?)\n\s*return list;", src, re.S)
    if not m:
        raise SystemExit("cannot find the lock block in %s" % FEED)
    lock = m.group(1)
    return """
%s
%s
function run(rows, t_rec) {
    var list = [];
    for (var i = 0; i < rows.length; i++) {
        var m = signal_metrics(rows[i], t_rec);
        m.prn = rows[i].prn;
        list.push(m);
    }
%s
    var out = [];
    for (var k = 0; k < list.length; k++)
        out.push({prn: list[k].prn, active: !!list[k].active,
                  cn0: (list[k].cn0 === null || list[k].cn0 === undefined) ? null : list[k].cn0,
                  q: list[k].q === undefined ? null : list[k].q,
                  sig: list[k].sig});
    return out;
}
""" % (consts, metrics, lock)


def run_js(js, rows, t_rec=0.001):
    return dukpy.evaljs(js + "\nrun(dukpy['rows'], dukpy['t_rec']);", rows=rows, t_rec=t_rec)


def fetch(host, chain):
    try:
        raw = subprocess.run(
            ["ssh", "cf06", "curl -s --max-time 5 http://%s/%s/get_status" % (host, chain)],
            capture_output=True, timeout=25).stdout
        d = json.loads(raw)
        return d if isinstance(d, list) else []
    except Exception as e:
        print("   (%s unreachable: %s)" % (chain, e))
        return []


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:12060"
    js = build_js()
    print("extracted lock rule + signal_metrics from gps_feed.js (it parses)\n")

    # ---- POSITIVE CONTROL: a healthy satellite must still read locked -------------------
    print("healthy satellites (synthetic) -- the direction that makes a null mean something")
    healthy = [
        {"prn": 1, "fleet_q": 3.4, "fleet_q_floor": 1.1, "sig": 5000.0,
         "amplitude": 1e-6, "unbiased_amplitude": 8e-7, "cn0_prompt_db": 42.0,
         "cn0_prompt_duty": 0.98},
        {"prn": 2, "fleet_q": 2.3, "fleet_q_floor": 1.1, "sig": 40.0,
         "amplitude": 1e-6, "unbiased_amplitude": 5e-7, "cn0_prompt_db": 37.5,
         "cn0_prompt_duty": 0.8},
    ]
    r = {x["prn"]: x for x in run_js(js, healthy)}
    check(r[1]["active"] and r[2]["active"], "a satellite with q 3.4 / 2.3 reads LOCKED")
    check(abs(r[1]["cn0"] - 42.0) < 1e-9, "and its published C/N0 is passed through (42.0)")

    # ---- the marginal and at-noise cases ------------------------------------------------
    print("\nthe bars")
    edge = [
        {"prn": 3, "fleet_q": 1.9, "fleet_q_floor": 1.1, "sig": 9e3},   # marginal
        {"prn": 4, "fleet_q": 1.05, "fleet_q_floor": 1.1, "sig": 9e3},  # at the floor
        {"prn": 5, "fleet_q": 2.4, "fleet_q_floor": 2.3, "sig": 9e3},   # above bar, AT floor
    ]
    r = {x["prn"]: x for x in run_js(js, edge)}
    check(not r[3]["active"], "q 1.9 (under Q_LOCK) is NOT locked -- half-locked is not locked")
    check(not r[4]["active"], "q 1.05 against a 1.10 floor is NOT locked")
    check(not r[5]["active"],
          "q 2.4 clears the fixed bar but sits ON a raised floor (2.3) -- still not locked, "
          "because the published floor beats a hardcoded number")

    # ---- sig must no longer be able to manufacture a lock --------------------------------
    print("\nsig cannot manufacture a lock any more")
    r = {x["prn"]: x for x in run_js(js, [
        {"prn": 6, "fleet_q": 1.0, "fleet_q_floor": 1.1, "sig": 10000.0}])}
    check(not r[6]["active"], "sig 10,000 with q 1.0 reads UNLOCKED (the 08-26 case)")
    r = {x["prn"]: x for x in run_js(js, [{"prn": 7, "sig": 10000.0}])}
    check(r[7]["active"],
          "but a chain that publishes NO q still falls back to sig (old broker unchanged)")

    # ---- the C/N0 pole -------------------------------------------------------------------
    print("\nthe pre-#57 incoherent C/N0 and its pole at u -> a")
    pole = [
        # A #57 broker that declined: duty present, value null -> a GAP, never a fallback.
        {"prn": 8, "fleet_q": 1.0, "fleet_q_floor": 1.1, "sig": 9e3, "cn0_prompt_duty": 0.01,
         "amplitude": 2.8e-7, "unbiased_amplitude": 2.79999e-7},
        # A genuinely old broker (neither key) at the pole.
        {"prn": 9, "fleet_q": 1.0, "fleet_q_floor": 1.1, "sig": 9e3,
         "amplitude": 2.8e-7, "unbiased_amplitude": 2.79999e-7},
        # An old broker on a real signal: still renders.
        {"prn": 10, "fleet_q": 3.0, "fleet_q_floor": 1.1, "sig": 9e3,
         "amplitude": 1e-6, "unbiased_amplitude": 5e-7},
    ]
    r = {x["prn"]: x for x in run_js(js, pole)}
    check(r[8]["cn0"] is None,
          "a #57 broker that DECLINED renders a gap, not the old estimator's answer")
    check(r[9]["cn0"] is None, "an old broker at u/a = 0.999996 is refused (the pole)")
    check(r[10]["cn0"] is not None and 20.0 < r[10]["cn0"] < 45.0,
          "an old broker on real signal still renders a sane value (%.1f dB-Hz)"
          % (r[10]["cn0"] or 0))

    # ---- the noise probes ----------------------------------------------------------------
    print("\nnoise probes are never satellites")
    r = {x["prn"]: x for x in run_js(js, [
        {"prn": 11, "fleet_q": 3.5, "fleet_q_floor": 1.1, "sig": 9e3, "noise_probe": True,
         "cn0_prompt_db": 31.0, "cn0_prompt_duty": 0.9},
        {"prn": 12, "fleet_q": 3.5, "fleet_q_floor": 1.1, "sig": 9e3,
         "cn0_prompt_db": 31.0, "cn0_prompt_duty": 0.9}])}
    check(not r[11]["active"],
          "a noise_probe row cannot read locked even at q 3.5 -- it is the floor by "
          "construction, and publish.py flags it so no consumer plots it as a satellite")
    check(r[12]["active"], "the identical NON-probe row does read locked (the control)")

    # ---- THE LIVE ROWS -------------------------------------------------------------------
    print("\nlive rows from the publisher at %s" % host)
    tot = locked = withc = 0
    worst = None
    for ch in CHAINS:
        rows = fetch(host, ch)
        if not rows:
            continue
        out = run_js(js, rows)
        for o in out:
            tot += 1
            locked += 1 if o["active"] else 0
            withc += 1 if o["cn0"] is not None else 0
            if worst is None or (o["sig"] or 0) > worst[0]:
                worst = ((o["sig"] or 0), ch, o["prn"], o["q"], o["active"])
    if not tot:
        print("   no live rows -- skipping (this section needs a running publisher)")
    else:
        print("   %d satellite row(s): %d would render LOCKED, %d would show a C/N0"
              % (tot, locked, withc))
        if worst:
            print("   loudest sig: %s PRN %s  sig %.0f  q %s  active=%s"
                  % (worst[1], worst[2], worst[0], worst[3], worst[4]))
        # No assertion on the count: this file must stay useful WITH signal, where a nonzero
        # locked count is the correct answer. The number is printed so a human can read it
        # against what the sky is actually doing.

    print("\nFAILED (%d)" % len(_fails) if _fails else "\nOK")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
