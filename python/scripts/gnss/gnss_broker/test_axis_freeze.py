"""The two time-base guards, and the hole between them.

    python3 -m gnss_broker.test_axis_freeze

The broker carries two watches that partition the space:

  * instance_stall_verdict (fits.py): one instance frozen while the fleet advances. It
    refuses to accuse anyone when most of the fleet is also stalled (min_frac_advancing),
    because that is not a per-instance fault...
  * fe_axis_stale_s (gps_distributed_broker): ...and the global case is this one's job:
    "has the whole time base frozen?"

A partition is only as good as both halves. The global guard must time the staleness from the
last advance of the hop, not from the last poll: a stamp rewritten every poll measures an age
of one cycle, never crosses the bar, and leaves a whole-fleet freeze unreported while the
per-instance guard correctly stays silent.

The guard is modelled here, not imported: it lives inline in the broker's cycle loop and
cannot be called. `axis_stale_reference` below transcribes its stamp rule, and test 0 asserts
the transcription against the shipped source text so the two cannot drift silently.

@author Keith Vanderlinde
"""

import os
import sys


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def axis_stale_reference(hops_and_times, stale_s):
    """Feed [(newest_hop, wall)] in order; return the wall times at which it would warn.

    The rule, from the broker: keep (hop, wall time the hop last advanced); warn when the hop
    has not advanced and that stamp is older than stale_s.
    """
    fe_axis = None
    fired = []
    for hop, now in hops_and_times:
        if hop <= 0.0:
            continue
        prev = fe_axis
        if (
            prev is not None
            and hop <= prev[0]
            and stale_s > 0.0
            and now - prev[1] > stale_s
        ):
            fired.append(now)
        fe_axis = (hop, now) if (prev is None or hop > prev[0]) else (hop, prev[1])
    return fired


HPS = 195312.5
CYCLE = 2.0


def series(n_live, n_frozen, start_hop=36_373_000_000.0, t0=1000.0):
    """n_live advancing cycles, then n_frozen at a fixed hop."""
    out, hop, t = [], start_hop, t0
    for _ in range(n_live):
        out.append((hop, t))
        hop += HPS * CYCLE
        t += CYCLE
    for _ in range(n_frozen):
        out.append((hop, t))
        t += CYCLE
    return out


def main():
    # ---- 0. the transcription is the shipped rule ---------------------------------------
    src = open(
        os.path.join(os.path.dirname(__file__), "..", "gps_distributed_broker.py")
    ).read()
    rule = "fe_axis[0] = ((_fh, _now()) if (_fh_prev is None or _fh > _fh_prev[0]) else (_fh, _fh_prev[1]))"
    check(
        "".join(rule.split()) in "".join(src.split()),  # match code, not its formatting
        "the broker still keeps the stamp of the last ADVANCE (not of this poll)",
    )

    # ---- 1. the rule fires, and promptly -------------------------------------------------
    # An hour of total freeze at a 2 s cycle against a 30 s bar.
    s = series(10, 1800)
    fired = axis_stale_reference(s, 30.0)
    check(len(fired) > 0, "FIXED: the freeze is reported")
    freeze_began = 1000.0 + 10 * CYCLE
    check(
        fired and (fired[0] - freeze_began) <= 34.0,
        "... within one bar of the freeze (%.0f s after it began)"
        % ((fired[0] - freeze_began) if fired else -1),
    )

    # ---- 2. it stays quiet on a healthy stream ------------------------------------------
    check(
        axis_stale_reference(series(1800, 0), 30.0) == [],
        "an advancing time base is never accused",
    )

    # ---- 3. a brief stall under the bar is not reported ----------------------------------
    # 10 cycles = 20 s of freeze against a 30 s bar: a poll hiccup, not a frozen axis.
    brief = series(10, 10) + [(36_373_000_000.0 + HPS * CYCLE * 20, 1060.0)]
    check(
        axis_stale_reference(brief, 30.0) == [],
        "a 20 s hiccup under the 30 s bar is not reported",
    )

    # ---- 4. recovery re-arms it ----------------------------------------------------------
    # Freeze, warn, then the stream returns: the stamp must follow the new hop so a later
    # freeze is timed from the recovery, not from the original one.
    s2 = series(5, 40)  # freeze -> fires
    t_end = s2[-1][1] + CYCLE
    s2 += [(9e11, t_end)]  # re-based stream, far ahead: an advance
    s2 += [(9e11, t_end + CYCLE * k) for k in range(1, 6)]  # 10 s frozen again
    fired2 = axis_stale_reference(s2, 30.0)
    check(
        fired2 and max(fired2) <= t_end,
        "after recovery the clock restarts: a 10 s freeze that follows is NOT reported",
    )

    # ---- 5. the partition: the per-instance guard is silent here by design ---------------
    from gnss_broker.fits import instance_stall_verdict

    urls = ["cx%02d/%d" % (n, g) for n in (19, 27, 42, 43, 44, 51) for g in (0, 1)]
    frozen_hop = 36_373_000_000
    prev = {u: (frozen_hop, 1000.0) for u in urls}
    _new, stalled = instance_stall_verdict(
        prev, {u: frozen_hop for u in urls}, 1000.0 + 600.0, 90.0
    )
    check(
        stalled == [],
        "with ALL 12 frozen, instance_stall_verdict accuses nobody -- correct, and exactly "
        "why the global guard must work: it is the only cover for this case",
    )
    one = dict(prev)
    cur = {u: frozen_hop + int(HPS * 600) for u in urls}
    cur[urls[0]] = frozen_hop  # one wedged, eleven advancing
    _new, stalled = instance_stall_verdict(one, cur, 1000.0 + 600.0, 90.0)
    check(
        [u for u, _h, _d in stalled] == [urls[0]],
        "... and it DOES catch one-of-twelve, which the global guard cannot see",
    )

    # ---- 6. every help string must survive format_help() -------------------------------
    # argparse interpolates help strings, so a literal "75% of windows" is a format specifier.
    # Python 3.14 (the broker's venv-ft) rejects it in add_argument, so the broker dies at
    # startup; older interpreters build the parser and fail only when help is formatted.
    # This check builds the parser and formats its help, so it fails on either; a replay
    # never formats help, so on older interpreters it cannot see this.
    from gnss_broker.cli import build_parser

    try:
        build_parser("selftest").format_help()
        ok = True
    except Exception as e:
        ok = False
        print("      format_help raised: %r" % (e,))
    check(
        ok,
        "every --flag's help survives format_help() (a bare %% is a format specifier)",
    )

    print("\n%s (%d check(s) failed)" % ("FAIL" if _fails else "PASS", len(_fails)))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
