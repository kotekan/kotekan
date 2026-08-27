"""Feeding the carrier-rate observables to the joint receiver (#33).

Two feeds, coarse and fine, differing in what they measure rather than merely how often: the
coarse one carries the deep-fold rate residual, the fine one the per-record rate and the #93
ADR span. Both are SHADOW paths by default -- they estimate and publish a state; whether the
seeds act on it is a separate armed decision.

⚠️ ADD THE STANDING COMMAND BACK, ALWAYS. The tracker has already derotated its records by the
commanded trim, so the reported rate is the RESIDUAL. Referencing the observable to the sky
means adding the command back; feeding the residual raw makes the estimator measure its own
actuator. That is #33 GAP 2's mirror, and it is invisible in the fit quality -- a mirror looks
like an excellent model.

⚠️ THE ROW IS NOT THE OBSERVABLE. GAP 3's F1 verdict (2026-08-25) measured the rrate ROW as
~97% carrier-only, with per-satellite rates 30x the code-side ramps they were supposed to
predict. #93's v2 feeds the DIRECT ADR span instead, and the two are logged side by side so
the comparison stays honest.

@author Keith Vanderlinde
"""

from gnss_broker.transport import _log_rl
from gnss_broker.fits import adr_fine_rate


def stage_rate_feed_coarse(ctx):
    """#33 RATE FEED (coarse): feed the deep-rate residual to the joint receiver's per-sat rate state.
        
    ⚠️ THE REFERENCE IS THE WHOLE POINT. `deep_rate` is measured on records the tracker ALREADY
    derotated by the commanded trim, so what the search reports is only what REMAINS. The standing
    command is added back so `y` is referenced to the sky rather than to the current command --
    feeding the residual raw makes the estimator measure its own actuator (see #33 GAP 2, the
    mirror)."""
    if ctx.args.rrate_state and ctx.rf.resid2 and ctx.drp.t_now_abs is not None:
        try:
            _jrr = ctx.rx.joint_receiver(ctx.band_id, ctx.code_len, rereference=ctx.args.joint_rereference, gauge_mode=ctx.args.joint_gauge)
            _n_ok = 0
            _n_gov = 0   # sats in the PHASE-GOVERNED regime this poll
            _n_rec_fed = 0
            for _p, _rv in sorted(ctx.rf.resid2.items()):
                # THE REFERENCE. deep_rate is measured on records the tracker already
                # derotated by the commanded trim, so what the search reports is what
                # REMAINS. Add the standing command back so y is referenced to the
                # chain's BASE seed -- the same value regardless of what the last
                # command happened to be. A frequency from a reference, never from an
                # argument: feeding the bare residual would make every commanded sat
                # read as "solved" and the filter would unlearn its own correction.
                # ⚠️ ONLY on the command-AWARE plant (--rrate-feed-applied). On the
                # folded assembler the observable never lost the command, and adding
                # it back is the arm-12 integrator runaway. See the flag's help.
                _y = _rv + (ctx.rf.cmd_applied.get(_p, ctx.car.trim.get(_p, 0.0))
                            if ctx.args.rrate_feed_applied else 0.0)
                _k = (ctx.args.dr_constellation, int(_p))
                # FLL->PLL HANDOFF: a phase-governed satellite takes its coarse
                # measurements at inflated sigma (see --rrate-coarse-deweight). Never
                # SKIPPED -- the coarse feed is what re-acquires this row after a
                # cycle slip, and a fine lock that has gone quiet expires on its own.
                _sig_c = 0.2
                # ---- PREFER THE RECORD-STREAM RATE (2026-08-14) -----------------
                # deep_rate_full_hz is the DEEP FOLD's argmax, and its measured
                # structure function is FLAT with lag -- rms change 1.44 m/s at 3 s
                # rising to 2.07 at 24 s and no further. Flat means independent NOISE
                # on every sample (a genuinely moving state grows as sqrt(lag)), so
                # that feed carries ~1.4 m/s = 5.6 Hz of noise around a state that
                # barely moves. The filter was right to reject 98% of it, and
                # loosening q_rr would only have admitted the noise.
                #
                # fcoh's rate_hz fits the phase slope of the fleet-summed per-record
                # series instead, with a split-half sigma measured the same way --
                # 15-500 mHz on these satellites, 4-100x better. Same records, no
                # extra poll. Falls through to the fold's value when the arc is too
                # short to fit, so a thin poll degrades rather than starving the row.
                # ⚠️ OFF (2026-08-14, measured): the record-stream fit is NOT better.
                # Structure functions on the SAME satellites at the SAME instants, once
                # the 2048x units bug was fixed:
                #     lag      3 s    6 s   12 s   24 s
                #     record  1.21   1.83   2.37   2.40   m/s
                #     fold    1.26   1.87   2.28   2.15
                # Indistinguishable. Both pick a peak from a spectrum over the same ~1 s
                # window of the same records with deterministic algorithms, so when the
                # peak selection goes wrong they go wrong TOGETHER -- correlated errors,
                # identical structure functions.
                #
                # Its split-half sigma (1.13 Hz) said otherwise and was WRONG to trust:
                # two halves of one window fitted on one grid both land on the same wrong
                # peak and agree. Split-half measures PRECISION, never accuracy. The
                # structure function at 3 s reads 1.21 m/s where 0.29 m/s of independent
                # noise would give 0.41 -- that gap is the shared bias split-half cannot
                # see.
                #
                # Left in place, unused, because rec_rate_hz remains a useful published
                # diagnostic and because the next estimator worth trying is of a
                # DIFFERENT KIND (the fine feed's res_cycles phase accumulation), not a
                # better spectral fit over the same window. Set _use_rec to re-enable.
                _use_rec = False
                _fcr = (ctx.dllp.fcoh or {}).get(_p) or {}
                _rrec, _srec = _fcr.get("rate_hz"), _fcr.get("rate_sigma_hz")
                if _use_rec and _rrec is not None and _srec is not None:
                    _y = _rrec + (ctx.rf.cmd_applied.get(_p, ctx.car.trim.get(_p, 0.0))
                                  if ctx.args.rrate_feed_applied else 0.0)
                    # never claim better than the fold's grid can resolve, and never
                    # worse than the old blanket 0.2 -- a split-half of exactly 0 is
                    # two halves landing in one bin, not infinite precision.
                    _sig_c = min(max(_srec, 0.02), 0.2)
                    _n_rec_fed += 1
                if (ctx.args.rrate_coarse_deweight > 1.0
                        and (ctx.t0 - ctx.rf.fine_t.get(_p, -1e9) <= ctx.args.rrate_fine_hold_s
                             or ctx.t0 - ctx.rf.kcoh_t.get(_p, -1e9)
                             <= ctx.args.rrate_fine_hold_s)):
                    _sig_c *= ctx.args.rrate_coarse_deweight
                    _n_gov += 1
                if _jrr.update_rrate(_k, _y, ctx.drp.t_now_abs, ctx.args.carrier_hz,
                                     sigma_hz=_sig_c) is not None:
                    _n_ok += 1
            _jrr.gauge_rrate()
            _rows = " ".join(
                "%d:%+.2f+-%s" % (_p, _jrr.rrate((ctx.args.dr_constellation, int(_p))),
                                  ("%.2f" % _s if (_s := _jrr.rrate_sigma(
                                      (ctx.args.dr_constellation, int(_p)))) < 99.0
                                   else "inf"))
                for _p in sorted(ctx.rf.resid2))
            _log_rl("jrr",
                    "JRR[%s%s] rrate m/s: %s | f_car %+.3f+-%.3f Hz "
                    "(%d/%d accepted this poll%s; n=%d rej=%d)"
                    % (ctx.args.dr_constellation, "" if ctx.rf.full_ok else " CAPPED-FALLBACK",
                       _rows, _jrr.f_carrier(),
                       _jrr.f_carrier_sigma(), _n_ok, len(ctx.rf.resid2),
                       ((", %d PHASE-GOVERNED" % _n_gov) if _n_gov else "")
                       + ((", %d/%d from RECORD STREAM" % (_n_rec_fed, len(ctx.rf.resid2)))
                          if _n_rec_fed else ", fold-fed"),
                       _jrr.n_rrate, _jrr.rrate_rejected), every_s=60.0)
        except Exception as e:
            _log_rl("jrr-err", "rrate feed skipped: %s" % e, every_s=300.0)


def stage_rate_feed_fine(ctx):
    """#33 RATE FEED (fine): the per-record fine rate, and the #93 ADR-span capture.
        
    Same reference discipline as the coarse feed: the standing command is added back so the
    observable describes the sky. This is also where the shadow's direct per-satellite ADR span
    rate is captured (`_adr_span_now`), which #93's v2 compares against the rrate ROW -- the row
    was measured on 2026-08-25 to be ~97% carrier-only (GAP 3's F1 verdict)."""
    if ctx.args.rrate_state and ctx.drp.t_now_abs is not None:
        try:
            _rec_dt = 2048.0 / ctx.args.hops_per_sec
            _jpp = []
            _n_fine = 0
            _jrf = ctx.rx.joint_receiver(ctx.band_id, ctx.code_len, rereference=ctx.args.joint_rereference, gauge_mode=ctx.args.joint_gauge)
            for _p, _rec in (ctx.status or {}).items():
                if not isinstance(_rec, dict):
                    continue
                _cmd_now = ctx.rf.cmd_applied.get(_p, 0.0)
                # GAP-1 LONG SPAN (2026-08-25): with --rrate-phase-span-s set, the
                # difference baseline is the newest ring snapshot AT LEAST span_s old,
                # not last poll's. res_cycles' error TELESCOPES (measured 08-24,
                # fixtures/gap1_tau_scaling.py: per-sat sigma_rate 1.12 Hz at 2 s ->
                # 0.31 at 16 s -> 0.18 at 32 s, converging 1/tau on both instances
                # checked), so a longer baseline buys noise down linearly while the
                # staleness cost is only drift*(span/2) ~ 0.02 Hz/s * span/2 -- the
                # 16-32 s window nets ~0.35 Hz effective, under the 0.5 Hz quietness
                # bar that parked arms 16b/17. All of adr_fine_rate's structural gates
                # (same accumulator arc, counter advanced, span-vs-wall) apply
                # unchanged; a ring entry that predates an arc break is rejected by
                # the arc gate exactly like a last-poll one.
                _ring = ctx.rf.adr_ring.setdefault(_p, [])
                if ctx.args.rrate_phase_span_s > 0.0:
                    while _ring and (ctx.t0 - _ring[0][2]) > 2.0 * ctx.args.rrate_phase_span_s:
                        _ring.pop(0)
                    _pv = None
                    for _e in reversed(_ring):
                        if (ctx.t0 - _e[2]) >= ctx.args.rrate_phase_span_s:
                            _pv = _e
                            break
                else:
                    _pv = ctx.rf.adr_prev.get(_p)
                if _pv is not None:
                    _snap = {"adr_arc": _pv[0][0], "adr_records": _pv[0][1],
                             "res_cycles": _pv[0][2]}
                    _snap["trim_cycles"] = _pv[0][3] if len(_pv[0]) > 3 else None
                    # wall_dt arms the serving-churn discriminator (see
                    # adr_fine_rate): the row is best-of-instance and the winner
                    # churns; a cross-instance span is wrong by up to 12x while
                    # passing the arc gate.
                    _wdt = (ctx.t0 - _pv[2]) if len(_pv) > 2 else None
                    _fr = adr_fine_rate(_rec, _snap, _rec_dt, wall_dt=_wdt)
                    if _fr is not None:
                        _fy, _nrec, _applied = _fr
                        _co = ctx.rf.resid2.get(_p) if ctx.rf.resid2 else None
                        if _co is not None:
                            _jpp.append((_p, _fy, _co))
                        # ── #93 SHADOW v2 CAPTURE (read-only) ──────────────────
                        # The sat's total carrier residual rate vs the seeded
                        # model, reference-corrected EXACTLY as the feed would
                        # (measured applied command when served and plausible,
                        # else the posted-command span-mean) -- but on the shadow
                        # path, so the unfed chain (e5a) measures it too. Sign
                        # falls back to +1 where uncalibrated; the fold-safety
                        # bound is the same 20 Hz as the feed.
                        _sg93 = ctx.args.rrate_phase_sign or 1.0
                        if abs(_sg93 * _fy) < 20.0:
                            _ap93 = (_applied
                                     if (_applied is not None
                                         and abs(_applied) <= 50.0) else None)
                            _cm93 = ((_sg93 * _ap93) if _ap93 is not None
                                     else 0.5 * (_cmd_now + _pv[1]))
                            ctx.rf.adr_span_now[_p] = (_sg93 * _fy + _cm93, ctx.t0, _nrec)
                        # COMMAND MOTION over the span enters the REFERENCE, not a
                        # stillness gate (a strict gate starved the feed to ~1/min --
                        # the coarse loop nudges the command every poll). Span-mean
                        # is the best estimate of the applied command for an unknown
                        # application time within the span; its worst-case error is
                        # dcmd/2, which goes into sigma rather than into a silent
                        # bias. Gate only at ONE slew step (0.6): beyond that the
                        # command jumped for a non-loop reason (re-seed, probe) and
                        # the span is not a measurement.
                        _dcmd = _cmd_now - _pv[1]
                        # SPAN-MODE NON-OVERLAP THROTTLE (2026-08-25): at poll cadence
                        # consecutive 16 s-span values share 14/16 of their data --
                        # feeding them as independent makes the row ~sqrt(8)x
                        # overconfident. In span mode a sat feeds at most once per
                        # span, so fed measurements are disjoint windows. The SHADOW
                        # (JRRP) stays per-poll; only the feed is throttled.
                        _span_ok = (ctx.args.rrate_phase_span_s <= 0.0
                                    or ctx.t0 - ctx.rf.span_fed_t.get(_p, 0.0)
                                    >= ctx.args.rrate_phase_span_s)
                        if (ctx.args.rrate_phase_feed and ctx.args.rrate_phase_sign != 0.0
                                and _span_ok
                                and abs(_dcmd) <= 0.6
                                and ((_rec.get("coherence_s") or 0.0) > 0.0
                                     or (_rec.get("coh_frac") or 0.0) >= 0.3)):
                            _yf = ctx.args.rrate_phase_sign * _fy
                            # NO CONVERGENCE REGIME (00:2x, measured): res_cycles is
                            # UNWRAPPED -- summed per-record increments, no mod-2pi
                            # anywhere -- so the fine value is valid at ANY residual
                            # below the per-record fold bound (0.25 cyc / 10.5 ms
                            # ~ +-23 Hz). The old 0.3 gate was FLL->PLL folklore: it
                            # kept fine to ~1 sat/poll while the two sats inside it
                            # held their commands to +-0.02 Hz and everyone else
                            # wandered at the coarse floor. Gate only at fold safety.
                            if abs(_yf) < 20.0:
                                _k = (ctx.args.dr_constellation, int(_p))
                                # reference: the MEASURED applied command when the
                                # combiner serves it (same span, same stream, no
                                # assumption -- and no motion penalty needed, it is
                                # exact); the posted-command span-mean otherwise.
                                # TRIPWIRE: since the honest PrnCtl::ctrim_hz export
                                # (2026-08-14) trim_cycles integrates the trim the
                                # tracker ACTUALLY applied on every chain. A tracker
                                # still running an older binary serves the airspy
                                # identity's (ctrim - f_offset)/2 = MHz-scale garbage
                                # (arm 9's rail runaway) -- this bound rejects that
                                # loudly. A plausible applied command is tens of Hz.
                                if _applied is not None and abs(_applied) > 50.0:
                                    _applied = None
                                if _applied is not None:
                                    _cmd_mid = ctx.args.rrate_phase_sign * _applied
                                    _sig_f = ctx.args.rrate_phase_sigma
                                else:
                                    _cmd_mid = 0.5 * (_cmd_now + _pv[1])
                                    _sig_f = (ctx.args.rrate_phase_sigma ** 2
                                              + (0.5 * _dcmd) ** 2) ** 0.5
                                if ctx.args.rrate_phase_span_s > 0.0:
                                    # sigma is defined AT THE 1-POLL SPAN and the
                                    # noise telescopes (1/span, measured); the
                                    # staleness term prices the span-mean lagging a
                                    # ~0.02 Hz/s drifting rate by span/2. The
                                    # measurement is timestamped at t_now (the filter
                                    # predicts forward only), so the lag lives HERE,
                                    # in the weight, not in the epoch.
                                    _span_s = _nrec * _rec_dt
                                    _sig_f = ((_sig_f * ctx.args.interval
                                               / max(_span_s, ctx.args.interval)) ** 2
                                              + (0.02 * 0.5 * _span_s) ** 2) ** 0.5
                                if _jrf.update_rrate(
                                        _k, _yf + _cmd_mid, ctx.drp.t_now_abs, ctx.args.carrier_hz,
                                        sigma_hz=_sig_f) is not None:
                                    _n_fine += 1
                                    # ACCEPTED fine measurements arm the handoff --
                                    # not attempts, so a sat whose fine values the
                                    # gate keeps rejecting stays coarse-governed.
                                    ctx.rf.fine_t[_p] = ctx.t0
                                    ctx.rf.span_fed_t[_p] = ctx.t0
                ctx.rf.adr_prev[_p] = ((_rec.get("adr_arc"), _rec.get("adr_records") or 0,
                                 _rec.get("res_cycles"), _rec.get("trim_cycles")),
                                _cmd_now, ctx.t0)
                if ctx.args.rrate_phase_span_s > 0.0:
                    _ring.append(ctx.rf.adr_prev[_p])
            # A sat that has left the seed set is RE-ACQUIRING when it returns, which
            # is the coarse feed's job -- drop its fine lock rather than let a stale
            # one de-weight the very measurements that must pull it back in.
            for _dead in [k for k in ctx.rf.fine_t if k not in ctx.seeds]:
                ctx.rf.fine_t.pop(_dead, None)
                ctx.rf.adr_prev.pop(_dead, None)
                ctx.rf.adr_ring.pop(_dead, None)
            if _jpp:
                _log_rl("jrrp",
                        "JRRP[%s%s] fine|coarse Hz (fine in INTERNAL sign): %s%s"
                        % (ctx.args.dr_constellation,
                           (" span %.0fs" % ctx.args.rrate_phase_span_s)
                           if ctx.args.rrate_phase_span_s > 0.0 else "",
                           " ".join("%d:%+.3f|%+.3f" % t for t in _jpp),
                           (" -- %d fine-fed" % _n_fine) if _n_fine else ""),
                        every_s=60.0)
        except Exception as e:
            _log_rl("jrrp-err", "phase-step feed skipped: %s" % e, every_s=300.0)
