"""Feeding the carrier-rate observables to the joint receiver's per-satellite rrate rows.

Two feeds, differing in what they measure rather than merely how often: the coarse one carries
the deep-fold rate residual (deep_rate_full_hz), the fine one the ADR phase rate (res_cycles
over a span). With --rrate-state alone they only estimate and log a state (the JRR and JRRP
lines); the seeds act on it only with --rrate-command.

Each observable must be referenced to the sky, not to the command. The coarse feed adds the
standing command back only under --rrate-feed-applied 1, for trackers whose records still
carry the actuation; a record assembler that folds the applied trim out already reports the
whole residual, and adding the command back would make each row measure its own actuator.
The fine feed always adds its reference: the trim applied over the span (trim_cycles), or the
posted command's span mean. A mis-referenced feed is invisible in the fit quality: a row
that measures its own actuator fits it well.

@author Keith Vanderlinde
"""

from gnss_broker.transport import _log_rl
from gnss_broker.fits import adr_fine_rate


def stage_rate_feed_coarse(ctx):
    """Coarse rate feed: each cohering satellite's deep-rate residual into its rrate row.
        
    Under --rrate-feed-applied 1 the records carry the actuation, so the reported rate is what
    remains after the command, and the standing command is added back so `y` is referenced to
    the sky rather than to the current command. With the folded record assembler (0) the
    residual already is the whole rate and nothing is added."""
    if ctx.args.rrate_state and ctx.rf.resid2 and ctx.drp.t_now_abs is not None:
        try:
            _jrr = ctx.rx.joint_receiver(
                ctx.band_id, ctx.code_len, ctx.args.chip_rate_hz, gauge_mode="prior"
            )
            _n_ok = 0
            _n_gov = 0  # satellites in the phase-governed regime this poll
            for _p, _rv in sorted(ctx.rf.resid2.items()):
                # The reference. Under --rrate-feed-applied 1 the records were derotated by
                # the commanded trim, so the rate is what remains; adding the standing
                # command back makes y the same whatever the last command was. Fed bare, a
                # commanded satellite would read as solved and the filter would unlearn its
                # own correction. On the folded assembler (0) the observable never lost the
                # command, and adding it back would make the row integrate its own actuation.
                _y = _rv + (
                    ctx.rf.cmd_applied.get(_p, 0.0)
                    if ctx.args.rrate_feed_applied
                    else 0.0
                )
                _k = (ctx.args.dr_constellation, int(_p))
                # FLL->PLL handoff: a satellite with a fine or KCOH acceptance within
                # rrate_fine_hold_s takes its coarse measurements at sigma x
                # --rrate-coarse-deweight. Deweighted, never skipped: the coarse feed is what
                # re-acquires this row after a cycle slip, and a quiet fine lock expires.
                _sig_c = 0.2
                if ctx.args.rrate_coarse_deweight > 1.0 and (
                    ctx.t0 - ctx.rf.fine_t.get(_p, -1e9) <= ctx.args.rrate_fine_hold_s
                    or ctx.t0 - ctx.rf.kcoh_t.get(_p, -1e9)
                    <= ctx.args.rrate_fine_hold_s
                ):
                    _sig_c *= ctx.args.rrate_coarse_deweight
                    _n_gov += 1
                if (
                    _jrr.update_rrate(
                        _k, _y, ctx.drp.t_now_abs, ctx.args.carrier_hz, sigma_hz=_sig_c
                    )
                    is not None
                ):
                    _n_ok += 1
            _jrr.gauge_rrate()
            _rows = " ".join(
                "%d:%+.2f+-%s"
                % (
                    _p,
                    _jrr.rrate((ctx.args.dr_constellation, int(_p))),
                    (
                        "%.2f" % _s
                        if (
                            _s := _jrr.rrate_sigma((ctx.args.dr_constellation, int(_p)))
                        )
                        < 99.0
                        else "inf"
                    ),
                )
                for _p in sorted(ctx.rf.resid2)
            )
            _log_rl(
                "jrr",
                "JRR[%s%s] rrate m/s: %s | f_car %+.3f+-%.3f Hz "
                "(%d/%d accepted this poll%s; n=%d rej=%d)"
                % (
                    ctx.args.dr_constellation,
                    "" if ctx.rf.full_ok else " CAPPED-FALLBACK",
                    _rows,
                    _jrr.f_carrier(),
                    _jrr.f_carrier_sigma(),
                    _n_ok,
                    len(ctx.rf.resid2),
                    ((", %d PHASE-GOVERNED" % _n_gov) if _n_gov else "") + ", fold-fed",
                    _jrr.n_rrate,
                    _jrr.rrate_rejected,
                ),
                every_s=60.0,
            )
        except Exception as e:
            _log_rl("jrr-err", "rrate feed skipped: %s" % e, every_s=300.0)


def stage_rate_feed_fine(ctx):
    """Fine rate feed: the ADR phase rate (adr_fine_rate) into the rrate rows.
        
    Under --rrate-state the JRRP line logs it beside the coarse value; it feeds only with
    --rrate-phase-feed and a nonzero --rrate-phase-sign. The reference is always added back,
    so the observable describes the sky: the trim the tracker applied over the same span when
    the combiner serves trim_cycles, the posted command's span mean otherwise."""
    if ctx.args.rrate_state and ctx.drp.t_now_abs is not None:
        try:
            _rec_dt = 2048.0 / ctx.args.hops_per_sec
            _jpp = []
            _n_fine = 0
            _jrf = ctx.rx.joint_receiver(
                ctx.band_id, ctx.code_len, ctx.args.chip_rate_hz, gauge_mode="prior"
            )
            for _p, _rec in (ctx.status or {}).items():
                if not isinstance(_rec, dict):
                    continue
                _cmd_now = ctx.rf.cmd_applied.get(_p, 0.0)
                # With --rrate-phase-span-s set, the difference baseline is the newest ring
                # snapshot at least span_s old, not last poll's. res_cycles' error sits at
                # the two endpoints, so a longer baseline lowers the rate noise as 1/span,
                # while the staleness cost is only the drift over span/2. adr_fine_rate's
                # structural gates (same arc, counter advanced, span vs wall) apply
                # unchanged; a ring entry that predates an arc break fails the arc gate like
                # a last-poll one.
                _ring = ctx.rf.adr_ring.setdefault(_p, [])
                if ctx.args.rrate_phase_span_s > 0.0:
                    while (
                        _ring
                        and (ctx.t0 - _ring[0][2]) > 2.0 * ctx.args.rrate_phase_span_s
                    ):
                        _ring.pop(0)
                    _pv = None
                    for _e in reversed(_ring):
                        if (ctx.t0 - _e[2]) >= ctx.args.rrate_phase_span_s:
                            _pv = _e
                            break
                else:
                    _pv = ctx.rf.adr_prev.get(_p)
                if _pv is not None:
                    _snap = {
                        "adr_arc": _pv[0][0],
                        "adr_records": _pv[0][1],
                        "res_cycles": _pv[0][2],
                    }
                    _snap["trim_cycles"] = _pv[0][3]
                    # wall_dt arms adr_fine_rate's accumulator-identity gate: the row is
                    # best-of-instance and the serving instance can change, and a
                    # difference across two instances' accumulators can pass the arc gate.
                    _wdt = ctx.t0 - _pv[2]
                    _fr = adr_fine_rate(_rec, _snap, _rec_dt, wall_dt=_wdt)
                    if _fr is not None:
                        _fy, _nrec, _applied = _fr
                        _co = ctx.rf.resid2.get(_p) if ctx.rf.resid2 else None
                        if _co is not None:
                            _jpp.append((_p, _fy, _co))
                        # Command motion over the span enters the reference, not a
                        # stillness gate (the coarse loop moves the command every poll).
                        # The span mean is the best estimate of the applied command for
                        # an unknown application time within the span; its worst-case
                        # error dcmd/2 goes into sigma rather than into a silent bias.
                        # Gate only at 0.6 Hz, just over one slew step
                        # (rrate_cmd_slew_hz): beyond that the command jumped for a
                        # non-loop reason (re-seed, probe) and the span is not a
                        # measurement.
                        _dcmd = _cmd_now - _pv[1]
                        # Span-mode non-overlap throttle: at poll cadence consecutive
                        # span values share most of their data, and fed as independent
                        # they would make the row overconfident. In span mode a
                        # satellite feeds at most once per span, so fed measurements are
                        # disjoint windows. The JRRP line stays per-poll.
                        _span_ok = (
                            ctx.args.rrate_phase_span_s <= 0.0
                            or ctx.t0 - ctx.rf.span_fed_t.get(_p, 0.0)
                            >= ctx.args.rrate_phase_span_s
                        )
                        if (
                            ctx.args.rrate_phase_feed
                            and ctx.args.rrate_phase_sign != 0.0
                            and _span_ok
                            and abs(_dcmd) <= 0.6
                            and (
                                (_rec.get("coherence_s") or 0.0) > 0.0
                                or (_rec.get("coh_frac") or 0.0) >= 0.3
                            )
                        ):
                            _yf = ctx.args.rrate_phase_sign * _fy
                            # No convergence gate: res_cycles is unwrapped (summed
                            # per-record increments, no mod 2pi), so the fine value is
                            # valid at any residual below the per-record fold bound
                            # (0.25 cycles per 10.5 ms record, ~+-23 Hz). Gate only at
                            # fold safety.
                            if abs(_yf) < 20.0:
                                _k = (ctx.args.dr_constellation, int(_p))
                                # Reference: the measured applied command
                                # (trim_cycles) when the combiner serves it, exact over
                                # the same span and needing no motion term; the posted
                                # command's span mean otherwise. A plausible applied
                                # command is tens of Hz, so a larger one is not a
                                # measurement and falls back to the span mean.
                                if _applied is not None and abs(_applied) > 50.0:
                                    _applied = None
                                if _applied is not None:
                                    _cmd_mid = ctx.args.rrate_phase_sign * _applied
                                    _sig_f = ctx.args.rrate_phase_sigma
                                else:
                                    _cmd_mid = 0.5 * (_cmd_now + _pv[1])
                                    _sig_f = (
                                        ctx.args.rrate_phase_sigma ** 2
                                        + (0.5 * _dcmd) ** 2
                                    ) ** 0.5
                                if ctx.args.rrate_phase_span_s > 0.0:
                                    # rrate_phase_sigma is defined at a one-poll
                                    # (--interval) span and the noise falls as
                                    # 1/span; the staleness term prices the span mean
                                    # lagging a ~0.02 Hz/s drifting rate by span/2.
                                    # The measurement is stamped at t_now (the filter
                                    # only predicts forward), so the lag lives in the
                                    # weight, not the epoch.
                                    _span_s = _nrec * _rec_dt
                                    _sig_f = (
                                        (
                                            _sig_f
                                            * ctx.args.interval
                                            / max(_span_s, ctx.args.interval)
                                        )
                                        ** 2
                                        + (0.02 * 0.5 * _span_s) ** 2
                                    ) ** 0.5
                                if (
                                    _jrf.update_rrate(
                                        _k,
                                        _yf + _cmd_mid,
                                        ctx.drp.t_now_abs,
                                        ctx.args.carrier_hz,
                                        sigma_hz=_sig_f,
                                    )
                                    is not None
                                ):
                                    _n_fine += 1
                                    # Accepted fine measurements arm the handoff,
                                    # not attempts, so a satellite whose fine values
                                    # are rejected stays coarse-governed.
                                    ctx.rf.fine_t[_p] = ctx.t0
                                    ctx.rf.span_fed_t[_p] = ctx.t0
                ctx.rf.adr_prev[_p] = (
                    (
                        _rec.get("adr_arc"),
                        _rec.get("adr_records") or 0,
                        _rec.get("res_cycles"),
                        _rec.get("trim_cycles"),
                    ),
                    _cmd_now,
                    ctx.t0,
                )
                if ctx.args.rrate_phase_span_s > 0.0:
                    _ring.append(ctx.rf.adr_prev[_p])
            # A satellite that has left the seed set re-acquires on the coarse feed when it
            # returns, so its fine lock is dropped rather than left to deweight the
            # measurements that must pull it back in.
            for _dead in [k for k in ctx.rf.fine_t if k not in ctx.seeds]:
                ctx.rf.fine_t.pop(_dead, None)
                ctx.rf.adr_prev.pop(_dead, None)
                ctx.rf.adr_ring.pop(_dead, None)
            if _jpp:
                _log_rl(
                    "jrrp",
                    "JRRP[%s%s] fine|coarse Hz (fine in INTERNAL sign): %s%s"
                    % (
                        ctx.args.dr_constellation,
                        (" span %.0fs" % ctx.args.rrate_phase_span_s)
                        if ctx.args.rrate_phase_span_s > 0.0
                        else "",
                        " ".join("%d:%+.3f|%+.3f" % t for t in _jpp),
                        (" -- %d fine-fed" % _n_fine) if _n_fine else "",
                    ),
                    every_s=60.0,
                )
        except Exception as e:
            _log_rl("jrrp-err", "phase-step feed skipped: %s" % e, every_s=300.0)
