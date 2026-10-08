"""Orbit prediction and visibility: what the sky should look like right now.

Evaluates the broadcast ephemeris (or the almanac/TLE fallback) at this instant to produce
`ctx.pred` -- per-satellite Doppler, Doppler rate and elevation -- and `ctx.up`, the set above
the mask. Every other stage plans against those two: which satellites to seed, which to coast,
where to centre a search window.

⚠️ THE EOP TABLE LIVES IN THE CONFIG AND IT ROLLS. A stale table is not a degraded prediction,
it is a REJECTED one. Eighteen days of staleness put dUT1 5.007 ms out, every frame's time
metadata was refused, and it killed a receiver outright on 2026-08-19. If predictions vanish
rather than drift, suspect the table before the model.

⚠️ THIS STAGE ALSO MAINTAINS THE CLOCK-FREQUENCY BIAS, and the two jobs are less separable
than they look: the bias is solved from the same multi-satellite residuals the prediction
produces. See `clockbias.py` for the invariant that follows -- a stale bias must widen the
search rather than narrow it.

@author Keith Vanderlinde
"""

import statistics

from gnss_broker.transport import _now, _post, _log, _log_rl
from gnss_broker.sky import brdc_predict, nearest_boresight, visible_prns
from gnss_broker import timebase


def stage_almanac_predict(ctx):
    """2: ORBIT-PREDICTED DOPPLER AND VISIBILITY, and the receiver clock bias EMA.
        
    Evaluates the almanac/BRDC at this instant to produce `pred` (the per-satellite Doppler and
    elevation the whole cycle plans against) and maintains the clock-bias EMA the seeds ride on.
        
    ⚠️ THE EOP TABLE IS IN THE CONFIG AND IT ROLLS. A stale table is not a degraded prediction, it
    is a rejected one: 18 days of staleness put dUT1 5.007 ms out and every frame's time metadata
    was refused (2026-08-19, it killed a receiver outright).
        
    ⚠️ BOTH THE BIAS AND ITS STALENESS ARE OUTPUTS. Downstream, `bias_stale` decides whether the
    search hint may narrow -- a confidently wrong narrow window is worse than no hint at all."""
    if ctx.args.almanac:
        try:
            t_pred = ctx.alm_now()
            if ctx.brdc_alm is not None:
                ctx.brdc_alm["eph_rebase"] = bool(getattr(ctx.args, "eph_rebase", 0))
                raw = brdc_predict(
                    ctx.brdc_alm,
                    ctx.args.lat,
                    ctx.args.lon,
                    ctx.args.alt,
                    ctx.alm_sys,
                    ctx.alm_min_prn,
                    t_pred,
                    ctx.args.carrier_hz,
                )
                # ── EPH-REBASE (#101): an ephemeris refresh just STEPPED the model by a
                # KNOWN per-sat delta; hand the equal-and-opposite trim adjustment to the
                # gather through the #92 handover (same-cycle ledger transfer) instead of
                # letting the leak-limited loop rebuild through the step for ~15-30 min.
                # ⚠️ SIGN IS CALIBRATED BY THE INSTRUMENT, NOT BY THIS COMMENT: the
                # post-merge trim-kick superposed-epoch (fixtures/gap3_merge_epoch.py)
                # must COLLAPSE on the armed chain vs its band-sibling control. If it
                # GROWS, the sign is wrong -- disarm and flip. (08-29's position-fix
                # lesson, pre-applied.) Requires --fleet-trim-rebase-adjust: offer()
                # no-ops when the handover is disabled, so warn loudly once.
                _es = ctx.brdc_alm.pop("eph_step", None)
                if _es is not None and getattr(ctx.args, "eph_rebase", 0):
                    if not ctx.handover.enabled:
                        _log_rl(
                            "ephreb-noho",
                            "eph-rebase ARMED but --fleet-trim-rebase-adjust is OFF: "
                            "the handover transport is disabled, steps NOT posted "
                            "(armed-but-inert)",
                            every_s=600.0,
                        )
                    else:
                        _stepd, _ts = _es
                        # CENSUS FIRST, unconditionally: without this line, zero posts
                        # is uninterpretable -- steps-all-tiny, sats-unarmed, and
                        # steps-never-computed all look identical (the armed-but-inert
                        # trap). One line per refresh names which.
                        _mx = max(_stepd.values(), key=abs) * ctx.args.chip_rate_hz
                        _log(
                            "EPH-REBASE census: refresh stepped %d sat model(s), "
                            "largest %+.3f chips, %d over the 0.02 post floor"
                            % (
                                len(_stepd),
                                _mx,
                                sum(
                                    1
                                    for v in _stepd.values()
                                    if abs(v * ctx.args.chip_rate_hz) >= 0.02
                                ),
                            )
                        )
                        _n_post = 0
                        for _k, _ds in sorted(_stepd.items()):
                            _prn = _k[1] if isinstance(_k, tuple) else _k
                            _dchips = _ds * ctx.args.chip_rate_hz
                            if abs(_dchips) < 0.02:
                                continue  # sub-noise; not worth a post
                            if ctx.handover.offer(
                                _prn,
                                _dchips,
                                _prn in ctx.dls.armed_last,
                                ctx.telem_chain,
                                ctx.args.fleet_trim_url,
                                _post,
                                _log,
                            ):
                                _n_post += 1
                        if _n_post:
                            _log(
                                "EPH-REBASE: %d per-sat model step(s) handed to the "
                                "gather at the refresh (largest %+0.3f chips)"
                                % (
                                    _n_post,
                                    max(
                                        (
                                            v * ctx.args.chip_rate_hz
                                            for v in _stepd.values()
                                        ),
                                        key=abs,
                                    ),
                                )
                            )
            else:
                from gps_beamtrack import predict_dopplers

                raw = predict_dopplers(
                    ctx.args.lat,
                    ctx.args.lon,
                    ctx.args.alt,
                    t_utc=t_pred,
                    _sats=ctx.almanac_sats,
                    f_carrier_hz=ctx.args.carrier_hz,
                )
            # doppler_sign flips to the receiver's observed convention -- apply it to BOTH the
            # Doppler and its rate so the 2nd-order feed-forward ramps the right way. (Range
            # is geometry, no sign; it feeds the CL time-assist propagation delay. The 5th
            # element -- broadcast sat clock -- is BRDC-only; the TLE path has none.)
            ctx.pred = {
                p: (
                    ctx.args.doppler_sign * v[0],
                    ctx.args.doppler_sign * v[1],
                    v[2],
                    v[3],
                    (v[4] if len(v) > 4 else 0.0),
                    (v[5] if len(v) > 5 else 0.0),
                )
                + tuple(v[6:9])
                for p, v in raw.items()
            }
            # ── #102 GEOMETRY FEED (--post-sat-geometry): every ~30 s, post each sat's
            # az/el to the record assemblers' /set_sat_geometry so the element steering
            # (elem_positions_enu in the node config) has fresh directions. With the rates and
            # epoch (pred elements 6-8, BRDC path) the post is [az, el, az_rate, el_rate,
            # t_utc] and the assemblers follow the track between posts; without them it stays
            # a bare [az, el] snapshot, which they hold. Endpoint
            # derived from the combiner list (n2combine -> n2assemble). Harmless where
            # the assembler has no positions configured (the endpoint then does not
            # exist and the post fails quietly into the rate-limited log).
            if (
                getattr(ctx.args, "post_sat_geometry", 0)
                and ctx.dll_combiners
                and ctx.t0 - ctx.geom_post_t[0] >= 30.0
            ):
                ctx.geom_post_t[0] = ctx.t0
                _body = {}
                for _p2, _v2 in ctx.pred.items():
                    if len(_v2) > 5 and _v2[2] > 0.0:  # above horizon; az is element 5
                        _body[str(_p2)] = (
                            [
                                float(_v2[5]),
                                float(_v2[2]),
                                float(_v2[6]),
                                float(_v2[7]),
                                float(_v2[8]),
                            ]
                            if len(_v2) > 8
                            else [float(_v2[5]), float(_v2[2])]
                        )
                # THE TRANSIT GATE for the assemblers' shared element model: the POOLED nearest-to-
                # boresight separation over every constellation (ctx.dr_pd carries all three --
                # the same sky the railing veto reads). A bright satellite of ANOTHER system
                # captures this band's per-PRN learners just the same, so a per-chain answer
                # would miss exactly the transits that matter.
                _near = nearest_boresight(getattr(ctx, "dr_pd", None))
                if _body and _near is not None:
                    _body["_bore"] = [float(_near[0]), float(t_pred.timestamp())]
                if _body:
                    _okc = 0
                    for _u2 in ctx.dll_combiners:
                        try:
                            _post(
                                "%s/set_sat_geometry"
                                % _u2.replace("n2combine", "n2assemble"),
                                _body,
                                timeout=2.0,
                            )
                            _okc += 1
                        except Exception as _e2:
                            _log_rl(
                                "geom-post",
                                "sat-geometry post failed (%s): %s" % (_u2, _e2),
                                every_s=300.0,
                            )
                    _log_rl(
                        "geom-post-ok",
                        "SAT-GEOMETRY posted: %d sat(s) to %d/%d assembler(s)"
                        % (len(_body), _okc, len(ctx.dll_combiners)),
                        every_s=300.0,
                    )
            # SERVE THE SKY. The broker already knows every satellite's az/el; publishing it
            # is what lets the viewer stop deriving its own (and stop writing the shared nav
            # cache to do it -- 2026-08-27). Receiver-wide by construction: each chain
            # contributes its own constellation and the publisher unions them.
            # ⚠️ TLE-sourced rows carry no azimuth (element 5 defaults to 0.0), so they are
            # withheld rather than served as az=0 -- a satellite drawn due north because we
            # had no azimuth is worse than one absent from the plot.
            if ctx.publisher is not None and len(next(iter(raw.values()), ())) > 5:
                ctx.publisher.set_sky(
                    ctx.alm_sys, {p: (v[2], v[5]) for p, v in ctx.pred.items()}, _now()
                )
        except Exception as e:
            _log("almanac predict failed: %s" % e)
        ctx.up = {p for p, v in ctx.pred.items() if v[2] >= ctx.args.mask_deg}
        # Common clock-frequency bias = median(measured - predicted) over detected
        # sats. A tight residual spread confirms the sign convention; a wild spread
        # (resid ~ -2x predicted) means flip --doppler-sign.
        # FRESH detections only. The search stage's REST table keeps serving a detection
        # long after it was made, and best[] holds it; the prediction meanwhile ADVANCES
        # at the satellite's dop_rate, so a stale detection's (meas - pred) grows linearly
        # with its age -- stale-age x dop_rate wearing a clock-bias costume. Measured on
        # the L5 replay bench 2026-07-27: one ~90 s-stale PRN20 detection walked the
        # "solved" bias +4 -> +68 Hz at exactly dop_rate, the seeds chased it, and the
        # tracker was dragged off a 55-sigma satellite. Live mostly hides this because
        # strong sats re-detect every few seconds -- but any sparse-detection stretch
        # (fades, one-sat horizons) exposes the same feedback. det_fresh already tracks
        # exactly this (its comment even computes the hazard: "at 0.4 Hz/s slew, 100 s of
        # staleness fakes a 40 Hz mismatch") -- the bias solve just never consulted it.
        # SNR gate as well as freshness (2026-08-02). This median is a COMMON-MODE
        # estimate: its uncertainty is (per-sat Doppler error)/sqrt(N), so one satellite
        # whose Doppler is noise does real damage when N is 2. The acquire's own
        # interpolation error is ~1.2 Hz rms (measured sawtooth), which at N=2 predicts
        # ~0.8 Hz -- but the raw estimate scatters 10.5 Hz, because every detection above
        # acquire_snr enters, and ~half of CHORD's sit between the threshold (30) and the
        # noise ceiling (19), where the reported Doppler is near-random inside the hint
        # window. Gated, this is what decides whether predicted Doppler (BRDC 0.1 Hz +
        # this) beats the detection's own (~2 Hz): at N=8 it should reach ~0.4 Hz.
        # Default 0 keeps every point -- the prototype's behaviour, right for detections
        # that sit far above threshold.
        resid = [
            ctx.best[p][1] - ctx.pred[p][0]
            for p in ctx.best
            if p in ctx.pred
            and ctx.t0 - ctx.det_fresh.get(p, (None, 0.0))[1]
            < ctx.args.bias_det_fresh_s
            and ctx.best[p][0] >= ctx.args.bias_min_snr
        ]
        # ── TIME BASE: the same residuals, read for their SPREAD rather than their median.
        # The median is the clock bias; a spread that one epoch shift explains is a stale
        # sample-0 epoch on the nodes -- the fault that filed a day of data onto the wrong
        # day, twice, with nothing here complaining (gnss_broker/timebase.py).
        timebase.observe(ctx, _log_rl)
        _local_ok = len(resid) >= ctx.args.bias_min_sats
        if _local_ok:
            # The per-cycle median is quantized to the 500 Hz search grid and jumps
            # hundreds of Hz as the detected-sat set flickers; the TRUE bias is a slow
            # TCXO drift. EMA-smooth it (sub-grid dither across sats/cycles averages
            # out the quantization) so every sat's seed Doppler is stable -- a jittery
            # common bias was wrecking coherent integration (residual carrier +-260 Hz).
            # GATED on --bias-min-sats: one sat's residual is unfalsifiable and a bad
            # one narrows the search into a self-locking deadlock (see the arg help);
            # below the gate the EMA holds its last multi-sat value (or stays unsolved
            # -> margins stay WIDE, which is exactly what lets more sats in).
            # One LO, one number: fuse this chain's median with the siblings' persisted
            # estimates, sat-count weighted. With ZERO local sats the band consensus
            # stands alone -- that is not a guess, it is the same physical LO measured
            # by ~27 satellites on the neighbouring chains.
            raw_bias = statistics.median(resid)
            _cb_snap = (
                ctx.cb.ema is None or ctx.cb.stale
            )  # captured before the branch clears it
            if _cb_snap:
                # First solve, or stale-rescue re-solve: SNAP to the fresh median. An
                # EMA crawl (a=0.05) from a mid-walk latch kHz off truth would spend
                # minutes converging through exactly the hint region it just vacated.
                if ctx.cb.stale:
                    _log(
                        "CLOCK BIAS RE-SOLVED %+.1f Hz after %.0f s stale (held %+.1f, "
                        "%d sats)"
                        % (raw_bias, ctx.t0 - ctx.cb.meas_t, ctx.cb.ema, len(resid))
                    )
                    # ⚠️⚠️ A STARVED RE-SOLVE MUST NOT BECOME THE REFERENCE.
                    # This adopted `raw_bias` as the new warm-start calibration
                    # unconditionally -- ONE median, from however few satellites happened to
                    # be present, announced as "hardware news". Measured 2026-08-27: gps_l5
                    # went 842 s with no multi-sat solve (satellites are scarce), re-solved
                    # once at -17.9 Hz from a distribution whose median over 1222 samples is
                    # +0.0 with sd 12.7, and snapped `cal` from -2.3 to -17.9. Every later
                    # comparison was then against a reference that was pure noise, and the
                    # drift alarm fired once a minute for hours against a bias that was
                    # sitting correctly near zero.
                    #
                    # A reference deserves at least the evidence the ALARM demands before it
                    # cries wolf -- the alarm already widens its bar below 5 sats for exactly
                    # this reason. Below that, hold the old calibration and say so: an old
                    # reference is stale, a noise-derived one is WRONG, and wrong outranks
                    # stale for something every later judgement is measured against.
                    _n_cal = len(resid)
                    # ⚠️ THE FIRST CALIBRATION NEEDS THE EVIDENCE TOO. The guard below was
                    # written as "cal is not None and too few sats" -- so it protected
                    # RE-calibration and left the FIRST solve to take whatever was there.
                    # Measured minutes after shipping it (2026-08-27): a fresh broker set
                    # cal = -7.8 Hz from 3 sats and the alarm was back. With no calibration
                    # at all the alarm simply does not fire (it is gated on `cal is not
                    # None`), the EMA still seeds normally, and we wait for a reference worth
                    # having -- which is strictly better than adopting one that is noise.
                    if _n_cal < ctx.args.clock_bias_cal_min_sats:
                        _log(
                            "CLOCK BIAS: re-solve %+.1f Hz on only %d sat(s) (need %d) -- "
                            "%s. A starved re-solve is noise (sd ~13 Hz at this count), and "
                            "a wrong reference poisons every later comparison."
                            % (
                                raw_bias,
                                _n_cal,
                                ctx.args.clock_bias_cal_min_sats,
                                ("HELD the calibration at %+.1f" % ctx.cb.cal)
                                if ctx.cb.cal is not None
                                else "NO calibration set yet -- the drift alarm stays silent "
                                "until there is a reference worth comparing against",
                            )
                        )
                    else:
                        if (
                            ctx.cb.cal is not None
                            and abs(raw_bias - ctx.cb.cal)
                            > ctx.args.clock_bias_alarm_hz
                        ):
                            _log(
                                "CLOCK BIAS RECALIBRATED %+.1f -> %+.1f Hz on %d sats -- "
                                "hardware news (GPSDO re-settled?); new warm-start reference"
                                % (ctx.cb.cal, raw_bias, _n_cal)
                            )
                        ctx.cb.cal = raw_bias
                    ctx.cb.stale = False
                ctx.cb.ema = raw_bias
            else:
                ctx.cb.ema += ctx.args.bias_alpha * (raw_bias - ctx.cb.ema)
            ctx.cb.meas_t = ctx.t0
            ctx.cb.value = ctx.cb.ema
            # #105: the seed consumers' bias -- ClockBias.update_seed has the invariant.
            ctx.cb.update_seed(
                raw_bias, ctx.args.seed_bias_source, ctx.args.seed_bias_alpha, _cb_snap
            )
            # CONTRIBUTE (task #27 M3). Receiver scope -- one reference, one frequency
            # error -- so every co-hosted chain can have this without solving it. A
            # chain that HAS its own estimate never reads back, which is why publishing
            # here cannot change single-chain behaviour.
            ctx.rx.contribute_carrier_bias(ctx.chain_id, ctx.cb.ema, len(resid), ctx.t0)
            # ALARM LOG on a SAT-SCALED bar: the median-of-residuals noise is ~1/sqrt(n),
            # so the fixed bar (tuned for strong chains) cried wolf on the weak-sat chains
            # (L5/E5a/B2a ~730 false alarms/night 2026-07-20 while the strong chains were
            # silent -- and a fleet-wide GPSDO event hits ALL chains, so a weak chain's
            # solo alarm is almost always noise). Widen it below 5 sats; a real event is
            # large + sustained so it still trips. Persist above keeps the TIGHT bar.
            if ctx.cb.cal is not None:
                _abar = ctx.args.clock_bias_alarm_hz * max(
                    1.0, (5.0 / max(len(resid), 1)) ** 0.5
                )
                if abs(ctx.cb.ema - ctx.cb.cal) > _abar:
                    _log_rl(
                        "clkalarm",
                        "CLOCK DRIFT ALARM: carrier bias %+.1f Hz vs calibration %+.1f "
                        "(|d| > %.0f Hz, %d sats) -- GPSDO unlock / thermal event? INVESTIGATE"
                        % (ctx.cb.ema, ctx.cb.cal, _abar, len(resid)),
                        # ⚠️ 60 s WAS FAR TOO HOT for an advisory nobody can act on in a
                        # minute. Against a poisoned calibration it fired once a minute
                        # for hours (2026-08-27) and buried everything else in the log --
                        # an alarm that repeats faster than it can be investigated is
                        # noise, and it would have hidden a real one.
                        every_s=ctx.args.clock_bias_alarm_every_s,
                    )
        # S2 OBSERVER: publish the carrier-side estimate. OUTSIDE the solve gate on
        # purpose -- an unsolved chain is exactly the case the fused state exists to
        # rescue, so it has to be visible, and `null` is how that is said (never 0).
        # `raw_hz` is this chain's OWN median, computed here rather than reusing
        # `raw_bias` above, which is already sibling-FUSED. Scoring cross-chain
        # agreement on a fused number measures the fusion, not the estimator.
        if ctx.state_w is not None:
            try:
                _raw_local = statistics.median(resid) if resid else None
                ctx.state_w.observe(
                    "carrier",
                    hz=ctx.cb.ema,
                    raw_hz=_raw_local,
                    mad_hz=ctx.receiver_state.mad(resid, _raw_local),
                    n=len(resid),
                    cal_hz=ctx.cb.cal,
                    stale=bool(ctx.cb.stale),
                    meas_age_s=round(ctx.t0 - ctx.cb.meas_t, 2),
                )
            except Exception:
                pass
        for p in sorted(ctx.best):
            if p in ctx.pred:
                _log_rl(
                    "meas-%d" % p,
                    "PRN %d: meas %+.0f  pred %+.0f  resid %+.0f Hz (elev %.0f)"
                    % (
                        p,
                        ctx.best[p][1],
                        ctx.pred[p][0],
                        ctx.best[p][1] - ctx.pred[p][0],
                        ctx.pred[p][2],
                    ),
                )
        if _local_ok:
            _log_rl(
                "clkbias",
                "clock-freq bias %+.0f Hz (raw %+.0f, %d sats, EMA a=%.2f) "
                "-> hints; seeds ride %+.1f Hz (%s)"
                % (
                    ctx.cb.value,
                    raw_bias,
                    len(resid),
                    ctx.args.bias_alpha,
                    ctx.cb.seed,
                    "slow a=%.3f" % ctx.args.seed_bias_alpha
                    if ctx.args.seed_bias_source == "slow"
                    else "zero: fixed, the solve is hints-only"
                    if ctx.args.seed_bias_source == "zero"
                    else "= hint EMA",
                ),
            )
        else:
            # Say WHY, with both counts -- "1 sat" alone sent this investigation looking
            # at the clock, the sky and the front end before anyone asked whether the
            # BAND had already solved it (2026-07-27).
            _log_rl(
                "clkbias",
                "clock-freq bias %s (%d sats < --bias-min-sats %d: residual not "
                "trusted)"
                % (
                    "held %+.0f Hz" % ctx.cb.value
                    if ctx.cb.ema is not None
                    else "UNSOLVED (wide margins)",
                    len(resid),
                    ctx.args.bias_min_sats,
                ),
            )
    elif ctx.gating:
        ctx.up = visible_prns(
            ctx.args.lat, ctx.args.lon, ctx.args.alt, ctx.args.mask_deg, 0.0
        )
