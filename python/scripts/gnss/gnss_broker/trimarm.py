"""Arming the C++ fleet trim loop, and reading its standing trims back.

One arm drives each PRN's trim: the gather's loop actuates the armed set posted here, and the
Python integrator (codeloop.py) stands down per PRN against the same set (dls.armed_last,
recorded before the POST). A failed POST breaks this until a POST succeeds (see
stage_fleet_trim_arming).

Arming follows presence with a hold, not presence at an instant. A released PRN's trim leaks
away, so a satellite flickering across the gate would be armed and released every cycle and
lose the standing trim it built each time.

The readback is the only place the real tap position is visible: the tracker despreads at
seed + standing trim, so a judge of the seed that ignores the trim sees only part of the
offset.

Neither loop may carry a tap onto a grating lobe of the tracker comb. The posted clamp stays
below half a lobe; a PRN whose seed is slewing is not trimmed, since the slew rails its
discriminator; and a present PRN whose instances have lost their common phase (the comb DLL's
xcoh, ~1 on the main peak and ~0 on a lobe) has both loops' trims released until they read ~0,
after which the loop starts again from the seed.

@author Keith Vanderlinde
"""

from gnss_broker.transport import _get, _log, _post, _log_rl, log_tag

# One GPU's channels lie on a stride-16 comb of 0.1953125-MHz channels, 3.125 MHz apart, so one
# instance's correlation response repeats every 320 ns: a grating lobe 3.27 chips out at
# 10.23 Mcps, 1.64 at E6's 5.115. On a lobe each instance keeps its prompt power but the
# instances lose their common phase, so the fleet sum cancels.
LOBE_S = 1.0 / (16 * 195312.5)
# The posted clamp is under half a lobe, so a trim alone cannot carry a tap onto a lobe's slope.
# Where the lobe falls inside the code's own +-1.5-chip peak (L2C, L2OC) it is not a separate
# peak, and the clamp keeps the full range.
TRIM_CLAMP_CHIPS = 3.0
LOBE_CLAMP_FRAC = 0.45
# Neither loop trims a PRN whose dead-reckon seed is further than this from its model: the
# slew moves the tap faster than the discriminator can follow, so it reads railed throughout. A
# gap counts while deadreckon.py refreshed it within SLEW_GAP_FRESH_S (each slew step does).
SLEW_ARM_MAX_CHIPS = 0.5
SLEW_GAP_FRESH_S = 30.0
# The lobe referee: an armed PRN whose xcoh stays below LOBE_XCOH, over at least
# LOBE_MIN_RECORDS records, for LOBE_HOLD_S, while its standing trim (fast + slow) has moved the
# tap at least LOBE_TRIM_FRAC of a lobe, is released until the fast trim reads under
# LOBE_RELEASED_CHIPS (at most LOBE_RELEASE_MAX_S), and not released again for LOBE_HOLDOFF_S.
# A low xcoh with a small trim is not a trim-made lobe, and releasing it would not move the tap.
LOBE_XCOH = 0.3
LOBE_MIN_RECORDS = 8
LOBE_HOLD_S = 30.0
LOBE_TRIM_FRAC = 0.25
LOBE_RELEASED_CHIPS = 0.02
LOBE_RELEASE_MAX_S = 60.0
LOBE_HOLDOFF_S = 120.0


def lobe_chips(chip_rate_hz):
    """The tracker comb's grating-lobe spacing in chips of this code."""
    return chip_rate_hz * LOBE_S


def trim_clamp(chip_rate_hz):
    """The clamp posted for a chain, chips of its own code."""
    lobe = lobe_chips(chip_rate_hz)
    if lobe < 1.5:
        return TRIM_CLAMP_CHIPS
    return min(TRIM_CLAMP_CHIPS, round(LOBE_CLAMP_FRAC * lobe, 3))


def slewing(ctx):
    """{prn: model-held gap, chips} for the PRNs whose seed is slewing further than
    SLEW_ARM_MAX_CHIPS, by a gap deadreckon.py stamped within SLEW_GAP_FRESH_S."""
    gaps = (getattr(ctx, "dr_state", None) or {}).get("slew_gap", {})
    return {
        p: g
        for p, (t, g) in gaps.items()
        if ctx.t0 - t <= SLEW_GAP_FRESH_S and abs(g) > SLEW_ARM_MAX_CHIPS
    }


def lobe_referee(ctx, skip):
    """Start and end lobe releases; return the PRNs released this cycle. Judges the PRNs armed
    by the hold, not those present this cycle: on a lobe the fleet prompt cancels, so presence
    flickers. A cycle without an xcoh measurement neither starts nor resets the timer. `skip`
    (the slewing PRNs) are not judged: a slew drops xcoh by itself. Nothing is judged where the
    lobe falls inside the code's own peak (trim_clamp)."""
    lobe = lobe_chips(ctx.args.chip_rate_hz)
    if lobe < 1.5:
        return set()
    fleet = ctx.dllp.fleet or {}
    held = set(
        p for p, t in ctx.dls.hold.items() if ctx.t0 - t < ctx.args.fleet_trim_hold_s
    )
    for p in list(ctx.dls.lobe_since):
        if p not in held or p in skip:
            del ctx.dls.lobe_since[p]
    for p in sorted(held - set(skip)):
        v = fleet.get(p) or {}
        x = v.get("xcoh")
        if x is None or (v.get("n_xcoh") or 0) < LOBE_MIN_RECORDS:
            continue
        if x < LOBE_XCOH:
            ctx.dls.lobe_since.setdefault(p, ctx.t0)
        else:
            ctx.dls.lobe_since.pop(p, None)
    for p, t in list(ctx.dls.lobe_since.items()):
        if ctx.t0 - t < LOBE_HOLD_S or p in ctx.dls.lobe_release:
            continue
        if ctx.t0 - ctx.dls.lobe_done.get(p, float("-inf")) < LOBE_HOLDOFF_S:
            continue
        fast = ctx.dls.readback.get(p, {}).get("trim_chips", 0.0)
        trim = fast + ctx.dls.trim.get(p, 0.0)
        if abs(trim) < LOBE_TRIM_FRAC * lobe:
            continue
        del ctx.dls.lobe_since[p]
        ctx.dls.lobe_release[p] = ctx.t0
        _log(
            "LOBE RELEASE PRN %d: xcoh below %.2f for %.0f s, trim %+.2f chips -- both "
            "trim loops released until it reads ~0, then the loop restarts from the seed"
            % (p, LOBE_XCOH, ctx.t0 - t, trim)
        )
    for p, t in list(ctx.dls.lobe_release.items()):
        # An erased trim leaves no row: absent from a read that succeeded is erased, absent
        # from a failed one is unknown.
        rb = ctx.dls.readback.get(p)
        erased = ctx.dls.readback_ok and (
            rb is None or abs(rb["trim_chips"]) <= LOBE_RELEASED_CHIPS
        )
        if (ctx.t0 > t and erased) or ctx.t0 - t >= LOBE_RELEASE_MAX_S:
            del ctx.dls.lobe_release[p]
            ctx.dls.lobe_done[p] = ctx.t0
            _log(
                "LOBE RELEASE PRN %d: re-armed after %.0f s (%s)"
                % (p, ctx.t0 - t, "trim ~0" if erased else "timed out")
            )
    return set(ctx.dls.lobe_release)


def stage_fleet_trim_arming(ctx):
    """Tell the gather which PRNs its C++ loop may actuate (--fleet-trim-url), with the loop
    constants, the probe-anchored floor and the tracker targets; then read its standing trims
    back (--fleet-trim-readback).

    dls.armed_last is recorded before the POST: it is what next cycle's Python integrator
    stands down against. If the POST fails, the controller keeps actuating the last armed set
    it received: until a POST succeeds, a PRN armed after that set is driven by neither arm
    and one released after it by both.

    dls.held_off is the set neither loop trims this cycle (slewing seeds and lobe releases);
    their Python trims are dropped here and codeloop.py does not integrate them."""
    if ctx.args.fleet_trim_url:
        _now_present = [
            _p for _p in (ctx.dllp.fleet or {}) if ctx.dllp.fleet[_p].get("present")
        ]
        # Stamped and aged on the frozen cycle clock, so a replay arms the same set at any
        # replay speed.
        for _p in _now_present:
            ctx.dls.hold[_p] = ctx.t0
        _slew = slewing(ctx)
        for _p, _g in sorted(_slew.items()):
            _log_rl(
                "slew-hold-%d" % _p,
                "SLEW HOLD PRN %d: seed %+.2f chips from its model (> %.2f), not trimmed "
                "until it arrives" % (_p, _g, SLEW_ARM_MAX_CHIPS),
                every_s=60.0,
            )
        ctx.dls.held_off = set(_slew) | lobe_referee(ctx, _slew)
        for _p in ctx.dls.held_off:
            ctx.dls.trim.pop(_p, None)
        # Armed: present within the last fleet_trim_hold_s (cli.py), and not held off.
        _armed = sorted(
            _p
            for _p, _t in ctx.dls.hold.items()
            if ctx.t0 - _t < ctx.args.fleet_trim_hold_s and _p not in ctx.dls.held_off
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
                    "clamp": trim_clamp(ctx.args.chip_rate_hz),
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
                ctx.dls.readback_ok = True
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
                ctx.dls.readback_ok = False
                ctx.dls.stat["last_err"] = str(_e)
