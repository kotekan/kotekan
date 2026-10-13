"""Arming the C++ trim loop and reading its standing trims back.

⚠️ AUTHORITY IS HELD BY EXACTLY ONE ARM. What this posts is what the gather actuates; the
Python integrator stands down per-PRN against the same set. The armed set is recorded BEFORE
the POST, not after, so a failed POST leaves both sides believing the C++ side is driving --
the safe direction, because a trim that keeps being driven is recoverable and one nobody
drives walks off.

⚠️ PRESENCE WITH A HOLD, NOT PRESENCE AT AN INSTANT. A satellite flickering across the deep
gate would otherwise be armed and released every cycle, and every arming change costs the
standing trim it just built.

⚠️ THE READBACK IS THE ONLY PLACE THE REAL TAP IS VISIBLE. seed + standing trim is where the
tracker actually despreads; every downstream judge of a seed that ignored the trim was judging
half the number.

@author Keith Vanderlinde
"""

import time

from gnss_broker.transport import _get, _post, _log_rl, log_tag


def stage_fleet_trim_arming(ctx):
    """4b: FLEET-TRIM ARMING -- tell the gather which PRNs its C++ loop may actuate.
        
    ⚠️ PRESENCE WITH A HOLD, NOT PRESENCE AT AN INSTANT. A satellite flickering across the deep
    gate would otherwise be armed and released every cycle, and an arming change costs the trim.
        
    ⚠️ THE HANDOVER'S HALF-STEP IS RECORDED BEFORE THE POST, NOT AFTER. `_ft_armed_last` is what
    next cycle's Python integrator stands down against, so authority is never held by both arms
    and never by neither. Recording after a POST would mean a failed POST leaves both sides
    believing the other is driving."""
    if ctx.args.fleet_trim_url:
        _now_present = [
            _p for _p in (ctx.dllp.fleet or {}) if ctx.dllp.fleet[_p].get("present")
        ]
        for _p in _now_present:
            ctx.dls.hold[_p] = time.time()
        # PRESENCE WITH A HOLD, not presence sampled at an instant -- see the flag.
        _armed = sorted(
            _p
            for _p, _t in ctx.dls.hold.items()
            if time.time() - _t < ctx.args.fleet_trim_hold_s
        )
        # THE HANDOVER'S HALF-STEP: record what we are about to hand the fast loop, so
        # NEXT cycle's slow integrator stands down for exactly the PRNs the C++ side is
        # actuating. Recorded before the POST rather than after, because a failed POST
        # leaves the controller running its LAST policy -- which is this one either way,
        # and the trims expire at the trackers if it never recovers.
        ctx.dls.armed_last.clear()
        ctx.dls.armed_last.update(_armed)
        # ── THE E3 FIX: ANCHOR THE FAST LOOP'S WINDOW GATE TO THE PROBES ────────────
        # The C++ integrate() gate was 3x the window's own population median -- a peer
        # competition, because on CHORD the armed rows are mostly REAL satellites (the
        # airspy premise "the median is the no-signal level" needed --noise-probes rows this
        # fleet never seeds into the loop's own window). Measured 2026-08-26: gal_e5b PRN 33
        # leak-only on 75.1% of its windows; E3's trim erased at ~20x the actual probe
        # floor, and a 60 s fade became a 12 min outage. This ships the same probe-anchored
        # absolute floor the PRESENCE gate has used since 08-14, margin already applied, so
        # the two gates finally reference the same noise.
        #
        # ⚠️ ONLY A PROBE-ANCHORED FLOOR IS SHIPPED. When presence itself fell back to the
        # peer bar (p_floor_src != probes:*), shipping that number would rebuild the exact
        # competition this removes, one level up -- so we ship 0 and the C++ side keeps its
        # local median, unchanged. And the floor rides EVERY policy POST rather than being
        # latched: probes come and go with the sky, and a stale floor held across a
        # brightness change is a gate nobody chose.
        _pol = {
            "chains": {
                ctx.telem_chain: {
                    "armed": _armed,
                    # BANDWIDTH, not per-update gain -- the controller converts with its measured
                    # rate. The slow DLL's dll_gain/dll_leak_present are NOT reused here: those
                    # constants are per-update at THIS process's cadence, and reusing them at
                    # 23.84 Hz is exactly the limit cycle of 2026-08-15. See the two flags.
                    "gain_per_s": ctx.args.fleet_trim_bandwidth,
                    "leak_per_s": ctx.args.fleet_trim_leak_per_s,
                    "clamp": 3.0,
                    "spacing": ctx.args.dll_spacing,
                    "targets": ["%s/set_trim" % t for t in ctx.trackers],
                }
            }
        }
        # ⚠️ ALWAYS SHIPPED NOW (2026-08-27). This used to be behind
        # --fleet-trim-floor-from-probes, with the key ABSENT when off so the equivalence
        # digests would not move -- absent parsed as 0.0 in the C++, which selected the
        # peer-median gate. That gate is deleted, so 0.0 no longer means "use the other
        # branch", it means REFUSE: the fast loop leaks every armed PRN and steps none.
        # That is the correct reading. A chain with no probe anchor has not measured its
        # noise, so no window of it carries usable information, and standing the loop down
        # is the conservative direction.
        _floor_abs = 0.0
        for _v in (ctx.dllp.fleet or {}).values():
            # Chain-level fact, identical on every row: read one, deliberately. And ONLY a
            # probe-anchored floor ships -- passing anything else through as absolute would
            # rebuild the competition this removes, one level up.
            if str(_v.get("p_floor_src", "")).startswith("probes:") and _v.get(
                "p_floor"
            ):
                _floor_abs = float(_v["p_floor"])
            break
        _pol["chains"][ctx.telem_chain]["p_floor_abs"] = _floor_abs
        try:
            _post(
                "%s/set_policy" % ctx.args.fleet_trim_url.rstrip("/"), _pol, timeout=2.0
            )
            ctx.dls.stat["posts"] += 1
        except Exception as _e:
            # NEVER take the cycle down for the fast loop. A controller that cannot be
            # reached simply stops being refreshed, and its trims EXPIRE at the trackers
            # (trim_ttl_s) rather than standing forever.
            ctx.dls.stat["fail"] += 1
            ctx.dls.stat["last_err"] = str(_e)
        _log_rl(
            "fleet-trim",
            "FLEET-TRIM %s: %d PRN(s) armed to %s, %d posts / %d failed%s"
            % (
                log_tag() or ctx.args.signal,
                len(_armed),
                ctx.args.fleet_trim_url,
                ctx.dls.stat["posts"],
                ctx.dls.stat["fail"],
                ("  last err %s" % ctx.dls.stat["last_err"])
                if ctx.dls.stat["last_err"]
                else "",
            ),
            every_s=30.0,
        )
        # #76 THE READBACK -- close the loop this block opened. GET the controller's
        # standing trims right after handing it policy, so this cycle's view of "where
        # does the tracker's tap actually sit" is seed + trim rather than seed alone.
        # READ-ONLY: nothing here feeds control yet (that is #83 2(d)); it fills
        # _ft_readback and the log. On failure the dict is CLEARED, not held: a stale
        # trim served as current is the exact blindness this exists to remove, and
        # "missing = unknown" is the truthful state.
        if ctx.args.fleet_trim_readback:
            try:
                _rb = _get(
                    "%s/get_dll" % ctx.args.fleet_trim_url.rstrip("/"), timeout=2.0
                )
                _rows = (_rb or {}).get(ctx.telem_chain) or {}
                ctx.dls.readback.clear()
                for _p, _r in _rows.items():
                    if isinstance(_r, dict) and "trim_chips" in _r:
                        ctx.dls.readback[int(_p)] = _r
                _log_rl(
                    "fleet-trim-rb",
                    "FLEET-TRIM READBACK %s: %s"
                    % (
                        log_tag() or ctx.args.signal,
                        " ".join(
                            "%d:%+.3f%s"
                            % (
                                _p,
                                ctx.dls.readback[_p]["trim_chips"],
                                "" if ctx.dls.readback[_p].get("armed") else "(rel)",
                            )
                            for _p in sorted(ctx.dls.readback)
                        )
                        or "no standing trim",
                    ),
                    every_s=30.0,
                )
            except Exception as _e:
                # Same rule as the POST above: never take the cycle down for the fast
                # loop. Counted and surfaced, and the dict stays empty until a poll
                # succeeds again.
                ctx.dls.readback.clear()
                ctx.dls.stat["last_err"] = str(_e)
