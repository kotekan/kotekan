"""Arming the C++ trim loop, reading its standing trims back, and the #93 ramp shadow.

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
        _now_present = [_p for _p in (ctx.dllp.fleet or {}) if ctx.dllp.fleet[_p].get("present")]
        for _p in _now_present:
            ctx.dls.hold[_p] = time.time()
        # PRESENCE WITH A HOLD, not presence sampled at an instant -- see the flag.
        _armed = sorted(_p for _p, _t in ctx.dls.hold.items()
                        if time.time() - _t < ctx.args.fleet_trim_hold_s)
        # THE HANDOVER'S HALF-STEP: record what we are about to hand the fast loop, so
        # NEXT cycle's slow integrator stands down for exactly the PRNs the C++ side is
        # actuating. Recorded before the POST rather than after, because a failed POST
        # leaves the controller running its LAST policy -- which is this one either way,
        # and the trims expire at the trackers if it never recovers.
        ctx.dls.armed_last.clear()
        ctx.dls.armed_last.update(_armed)
        _pol = {"chains": {ctx.telem_chain: {
            "armed": _armed,
            # BANDWIDTH, not per-update gain -- the controller converts with its measured
            # rate. The slow DLL's dll_gain/dll_leak_present are NOT reused here: those
            # constants are per-update at THIS process's cadence, and reusing them at
            # 23.84 Hz is exactly the limit cycle of 2026-08-15. See the two flags.
            "gain_per_s": ctx.args.fleet_trim_bandwidth,
            "leak_per_s": ctx.args.fleet_trim_leak_per_s,
            "clamp": 3.0,
            "spacing": ctx.args.dll_spacing,
            "targets": ["%s/set_trim" % t for t in ctx.trackers]}}}
        try:
            _post("%s/set_policy" % ctx.args.fleet_trim_url.rstrip("/"), _pol, timeout=2.0)
            ctx.dls.stat["posts"] += 1
            ctx.dls.stat["armed"] = len(_armed)
        except Exception as _e:
            # NEVER take the cycle down for the fast loop. A controller that cannot be
            # reached simply stops being refreshed, and its trims EXPIRE at the trackers
            # (trim_ttl_s) rather than standing forever.
            ctx.dls.stat["fail"] += 1
            ctx.dls.stat["last_err"] = str(_e)
        _log_rl("fleet-trim",
                "FLEET-TRIM %s: %d PRN(s) armed to %s, %d posts / %d failed%s"
                % (log_tag() or ctx.args.signal, len(_armed), ctx.args.fleet_trim_url,
                   ctx.dls.stat["posts"], ctx.dls.stat["fail"],
                   ("  last err %s" % ctx.dls.stat["last_err"])
                   if ctx.dls.stat["last_err"] else ""),
                every_s=30.0)
        # #76 THE READBACK -- close the loop this block opened. GET the controller's
        # standing trims right after handing it policy, so this cycle's view of "where
        # does the tracker's tap actually sit" is seed + trim rather than seed alone.
        # READ-ONLY: nothing here feeds control yet (that is #83 2(d)); it fills
        # _ft_readback and the log. On failure the dict is CLEARED, not held: a stale
        # trim served as current is the exact blindness this exists to remove, and
        # "missing = unknown" is the truthful state ([[chord-stale-artifacts]]).
        if ctx.args.fleet_trim_readback:
            try:
                _rb = _get("%s/get_dll" % ctx.args.fleet_trim_url.rstrip("/"), timeout=2.0)
                _rows = (_rb or {}).get(ctx.telem_chain) or {}
                ctx.dls.readback.clear()
                for _p, _r in _rows.items():
                    if isinstance(_r, dict) and "trim_chips" in _r:
                        ctx.dls.readback[int(_p)] = _r
                ctx.dls.stat["rb"] += 1
                _log_rl("fleet-trim-rb",
                        "FLEET-TRIM READBACK %s: %s"
                        % (log_tag() or ctx.args.signal,
                           " ".join("%d:%+.3f%s"
                                    % (_p, ctx.dls.readback[_p]["trim_chips"],
                                       "" if ctx.dls.readback[_p].get("armed") else "(rel)")
                                    for _p in sorted(ctx.dls.readback))
                           or "no standing trim"),
                        every_s=30.0)
            except Exception as _e:
                # Same rule as the POST above: never take the cycle down for the fast
                # loop. Counted and surfaced, and the dict stays empty until a poll
                # succeeds again.
                ctx.dls.readback.clear()
                ctx.dls.stat["rb_fail"] += 1
                ctx.dls.stat["last_err"] = str(_e)
            # -- GAP 3 SHADOW (#33, READ-ONLY): does the row PREDICT the trim ramp? --
            # Predicted per-sat code-rate aiding = rrate[m/s] * f_chip / c  (identically
            # K*residual_doppler_hz, K = chip_rate/carrier ~ 0.0087). Measured = LS slope
            # of the STANDING trim from consecutive readbacks. The window RESETS on any
            # step > 0.3 chips (a #92 re-anchor/wipe is a discontinuity, not a rate) and
            # on release (a released trim's slope is the LEAK, not the sky). Mean trim is
            # logged beside the slope so the leak drag (leak_per_s * trim) stays
            # computable offline -- in the tracking regime the equilibrium terms cancel
            # in the derivative and the raw slope ~ gain/(gain+leak) of the sky rate.
            # Nothing here feeds control; the consume step (GAP 3 step 3) is a separate
            # default-off arm with its own pre-registration.
            try:
                _t_rb = time.time()
                _jg = None
                try:
                    _jg = ctx.rx.joint_receiver(ctx.band_id, ctx.code_len,
                                            rereference=ctx.args.joint_rereference)
                except Exception:
                    _jg = None
                _g3_rows = []
                for _p in sorted(ctx.dls.readback):
                    _r = ctx.dls.readback[_p]
                    if not _r.get("armed"):
                        # A RELEASED trim's slope is the LEAK, not the sky: drop the
                        # series rather than pausing it.
                        ctx.g3_ramp.drop(_p)
                        continue
                    # #93: subtract the cumulative #92 handover deltas -- the
                    # gather applied them to the trim, so removing them makes the
                    # series continuous across re-bases (longer windows) and keeps
                    # sub-0.3-chip adjustments out of the slope.
                    ctx.g3_ramp.update(_p, _t_rb,
                                    ctx.handover.corrected(_p, float(_r["trim_chips"])))
                    _fit3 = ctx.g3_ramp.fit(_p)
                    if _fit3 is None:
                        continue
                    _msl, _ym, _spn, _n = _fit3
                    _prd = None
                    if _jg is not None:
                        _k3 = (ctx.args.dr_constellation, int(_p))
                        if _jg.rrate_sigma(_k3) < 99.0:
                            _prd = _jg.rrate(_k3) * ctx.args.chip_rate_hz / C_LIGHT
                    # #93 v2: the DIRECT ADR span rate, converted carrier Hz ->
                    # code chips/s (K = f_chip/f_carrier ~ 0.0087). Fresh within
                    # 60 s or absent.
                    _av = ctx.rf.adr_span_now.get(_p)
                    _adr = ((_av[0] * ctx.args.chip_rate_hz / ctx.args.carrier_hz)
                            if (_av is not None and _t_rb - _av[1] <= 60.0)
                            else None)
                    _g3_rows.append((_p, _prd, _adr, _msl, _ym, _spn))
                ctx.g3_ramp.retain(ctx.dls.readback)
                if _g3_rows:
                    _log_rl("gap3-shadow",
                            "GAP3-SHADOW %s (chips/s; p=row a=ADR-span "
                            "m=trim slope @mean trim): %s"
                            % (log_tag() or ctx.args.signal,
                               " ".join("%d:p%s/a%s/m%+.5f@%+.2f(%ds)"
                                        % (_p, ("%+.5f" % _pr)
                                           if _pr is not None else "--",
                                           ("%+.5f" % _ad)
                                           if _ad is not None else "--",
                                           _ms, _tmn, int(_sp))
                                        for _p, _pr, _ad, _ms, _tmn, _sp
                                        in _g3_rows)),
                            every_s=60.0)
            except Exception as _e:
                _log_rl("gap3-shadow-err", "GAP3-SHADOW error (shadow only): %s" % _e,
                        every_s=300.0)
