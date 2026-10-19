"""Arming the C++ fleet trim loop, and reading its standing trims back.

One arm drives each PRN's trim: the gather's loop actuates the armed set posted here, and the
Python integrator (codeloop.py) stands down per PRN against the same set (dls.armed_last,
recorded before the POST). A failed POST breaks this until a POST succeeds (see
stage_fleet_trim_arming).

Arming follows presence with a hold, not presence at an instant. A released PRN's trim leaks
away, so a satellite flickering across the gate would be armed and released every cycle and
lose the standing trim it built each time. A PRN present only on deep-fold evidence, with
prompt power below the probe floor the fast loop gates its windows on, is not armed: the
Python integrator trims it.

The readback is the only place the real tap position is visible: the tracker despreads at
seed + standing trim, so a judge of the seed that ignores the trim sees only part of the
offset.

@author Keith Vanderlinde
"""

from gnss_broker.transport import _get, _post, _log_rl, log_tag


def stage_fleet_trim_arming(ctx):
    """Tell the gather which PRNs its C++ loop may actuate (--fleet-trim-url), with the loop
    constants, the probe-anchored floor and the tracker targets; then read its standing trims
    back (--fleet-trim-readback).

    dls.armed_last is recorded before the POST: it is what next cycle's Python integrator
    stands down against. If the POST fails, the controller keeps actuating the last armed set
    it received: until a POST succeeds, a PRN armed after that set is driven by neither arm
    and one released after it by both."""
    if ctx.args.fleet_trim_url:
        _now_present = [
            _p for _p in (ctx.dllp.fleet or {}) if ctx.dllp.fleet[_p].get("present")
        ]
        # Present on deep-fold evidence with prompt power below the probe floor (the deep
        # gate, and the displaced admission's q+deep:probes+disp; fleet.apply_presence). The
        # fast loop leaks every window whose prompt power is below p_floor_abs, the floor
        # these rows already failed, and the Python integrator stands down for an armed PRN,
        # so armed they would be stepped by neither. They are left out of the armed set, hold
        # or not, and the per-cycle Python DLL trims them (codeloop.py).
        _below_floor = set(
            _p
            for _p in _now_present
            if ctx.dllp.fleet[_p].get("p_floor")
            and ctx.dllp.fleet[_p].get("p_pow", 0.0) < ctx.dllp.fleet[_p]["p_floor"]
        )
        # Stamped and aged on the frozen cycle clock, so a replay arms the same set at any
        # replay speed.
        for _p in _now_present:
            if _p not in _below_floor:
                ctx.dls.hold[_p] = ctx.t0
        # Armed: present within the last fleet_trim_hold_s (cli.py).
        _armed = sorted(
            _p
            for _p, _t in ctx.dls.hold.items()
            if ctx.t0 - _t < ctx.args.fleet_trim_hold_s and _p not in _below_floor
        )
        # What next cycle's Python integrator stands down against; recorded before the POST
        # (see the docstring).
        ctx.dls.armed_last.clear()
        ctx.dls.armed_last.update(_armed)
        # p_floor_abs anchors the fast loop's window gate to the probes: it is the presence
        # gate's absolute prompt-power floor (margin applied), so both gates reference the
        # same noise. It rides every policy POST rather than being latched, since probes come
        # and go with the sky.
        _pol = {
            "chains": {
                ctx.telem_chain: {
                    "armed": _armed,
                    # Per second, not per update: the controller divides by its measured
                    # update rate. The slow DLL's dll_gain / dll_leak_present are per update
                    # at this process's cadence and would put the ~23.8 Hz loop into a limit
                    # cycle. See --fleet-trim-bandwidth and fleet_trim_leak_per_s (cli.py).
                    "gain_per_s": ctx.args.fleet_trim_bandwidth,
                    "leak_per_s": ctx.args.fleet_trim_leak_per_s,
                    "clamp": 3.0,
                    "spacing": ctx.args.dll_spacing,
                    "targets": ["%s/set_trim" % t for t in ctx.trackers],
                }
            }
        }
        # 0 when the chain has no probe-anchored floor: the C++ loop then leaks every armed
        # PRN and steps none. A chain that has not measured its noise has no window carrying
        # usable information, so standing the loop down is the conservative direction.
        _floor_abs = 0.0
        for _v in (ctx.dllp.fleet or {}).values():
            # p_floor and p_floor_src are chain-level, identical on every row (fleet.py), so
            # one row is read. Only a probe-anchored floor ships; fleet.py marks a chain with
            # too few probes UNANCHORED.
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
            # Never take the cycle down for the fast loop. The controller keeps actuating the
            # last armed set it received; the trackers drop its trims (trim_ttl_s) only when
            # it stops posting them.
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
        # The readback: GET the controller's standing trims right after posting policy, so
        # the broker sees the tracker's tap at seed + trim, not the seed alone (seeding.py's
        # effective trim and CP_ERR line, fleetdll.py's published rows). On failure the dict
        # is cleared, not held: a stale trim served as current is the blindness this removes,
        # and missing means unknown.
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
                # As for the POST: never take the cycle down for the fast loop. The error
                # goes to last_err (the FLEET-TRIM line), and the dict stays empty until a
                # read succeeds.
                ctx.dls.readback.clear()
                ctx.dls.stat["last_err"] = str(_e)
