"""Orbit prediction and visibility: what the sky should look like right now.

Evaluates the broadcast ephemeris (or the TLE fallback) at the wall clock to produce
`ctx.pred` -- per-satellite Doppler, Doppler rate and elevation -- and `ctx.up`, the set above
the mask. Every other stage plans against those two: which satellites to seed, which to coast,
where to centre a search window.

This stage also maintains the clock-frequency bias, solved from the residuals between the
search's detections and these predictions. See `clockbias.py` for the invariant that follows:
a stale bias must widen the search rather than narrow it.

@author Keith Vanderlinde
"""

import statistics

from gnss_broker.transport import _now, _post, _log, _log_rl, axis_now_utc, now_utc_s
from gnss_broker.sky import brdc_predict, nearest_boresight
from gnss_broker import timebase
from gnss_broker import prnmap


def _refresh_capable(ctx):
    """Re-read the signal-capability set from the cached IGS registry every
    prnmap.CAPABLE_REFRESH_S. The set is otherwise fixed at startup, and a slot that changes
    hands (GLONASS re-slots a few times a year) would stay seeded, or unseeded, until a
    restart. No fetch here: ensure_registry() runs at process start, outside the cycle."""
    if ctx.capable_t is None or _now() - ctx.capable_t < prnmap.CAPABLE_REFRESH_S:
        return
    ctx.capable_t = _now()
    try:
        new = prnmap.registry_capable(ctx.args.signal_capability)
    except Exception as e:
        _log(
            "signal-capability %s: registry re-read FAILED (%s); keeping the set"
            % (ctx.args.signal_capability, e)
        )
        return
    if new is None or new == ctx.capable:
        return
    old = ctx.capable or set()
    _log(
        "signal-capability %s: registry changed, now %d capable PRN(s): +%s -%s"
        % (
            ctx.args.signal_capability,
            len(new),
            sorted(new - old) or "none",
            sorted(old - new) or "none",
        )
    )
    ctx.capable = new


def stage_almanac_predict(ctx):
    """2: orbit-predicted Doppler and visibility, and the receiver clock-bias EMA (--almanac).
        
    Evaluates BRDC (or the TLEs) at this instant to produce `ctx.pred` (the per-satellite
    Doppler and elevation the whole cycle plans against) and `ctx.up`, and maintains the bias
    in `ctx.cb` that the hints and seeds ride on. Also posts satellite geometry to the record
    assemblers (--post-sat-geometry) and serves the sky to the publisher.
        
    The broker sets `ctx.cb.stale` before this stage; a multi-satellite re-solve here clears
    it. searchhint narrows the hint margins only for a solved bias that is not stale, because
    a confidently wrong narrow window is worse than no hint at all."""
    if ctx.args.almanac:
        try:
            _refresh_capable(ctx)
            raw_dark = {}
            # The epoch: the hop-axis now (#208). It is also what the time-base detector below
            # needs: a stale sample-0 epoch from the nodes moves this prediction, never the
            # measured Doppler, and that spread is its tell. The wall clock serves only before
            # the F-engine anchor is known, when there is no other time.
            t_pred = axis_now_utc(ctx) or ctx.alm_now()
            if ctx.brdc_alm is not None:
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
                if ctx.capable is not None:
                    # Capability by block type; the alm_min_prn floor above is the fallback.
                    # Under --probe-dark the excluded satellites are kept aside as probe
                    # candidates: signal-free on this signal at any elevation.
                    if getattr(ctx.args, "probe_dark", False):
                        raw_dark = {
                            p: v for p, v in raw.items() if p not in ctx.capable
                        }
                    raw = {p: v for p, v in raw.items() if p in ctx.capable}
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
            # doppler_sign maps to the receiver's observed convention; it applies to both the
            # Doppler and its rate so the second-order feed-forward ramps the right way. Range
            # is geometry (no sign) and gives the propagation delay of the overlay-epoch
            # predictions (searchhint's NH hint, seeding's NH-joint votes). Elements 4 on
            # (satellite clock, az, rates, epoch) are BRDC-only: the TLE path gets 0.0 for the
            # clock and az, and no rates.
            def _row(v):
                return (
                    ctx.args.doppler_sign * v[0],
                    ctx.args.doppler_sign * v[1],
                    v[2],
                    v[3],
                    (v[4] if len(v) > 4 else 0.0),
                    (v[5] if len(v) > 5 else 0.0),
                ) + tuple(v[6:9])

            ctx.pred = {p: _row(v) for p, v in raw.items()}
            ctx.pred_dark = {p: _row(v) for p, v in raw_dark.items()}
            # Geometry feed (--post-sat-geometry): every ~30 s, post each satellite's az/el to
            # the record assemblers' /set_sat_geometry for the element steering
            # (elem_positions_enu in the node config). With the rates and epoch (pred elements
            # 6-8, BRDC path) the post is [az, el, az_rate, el_rate, t_utc] and the assemblers
            # follow the track between posts; without them it is a bare [az, el] snapshot,
            # which they hold. The endpoints are --dll-combiners with n2combine -> n2assemble.
            # An assembler without steering acknowledges the post and drops it.
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
                # The transit gate for the assemblers' shared element model ("_bore"): the
                # nearest-to-boresight separation pooled over every constellation.
                # ctx.transit_pd is the geometry-only sky with the health gate off, the same sky
                # the railing veto reads; it is set later in the cycle, so this reads the
                # previous cycle's. A bright satellite of another system captures this band's
                # per-PRN learners just the same, so a per-chain answer would miss the transits
                # that matter.
                _near = nearest_boresight(getattr(ctx, "transit_pd", None))
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
            # Serve the sky (el, az) so the viewer draws the broker's rather than deriving its
            # own. Receiver-wide: each chain contributes its own constellation and the
            # publisher unions them. TLE rows carry no azimuth (element 5 defaults to 0.0), so
            # they are withheld rather than drawn due north.
            if ctx.publisher is not None and len(next(iter(raw.values()), ())) > 5:
                ctx.publisher.set_sky(
                    ctx.alm_sys, {p: (v[2], v[5]) for p, v in ctx.pred.items()}, _now()
                )
        except Exception as e:
            _log("almanac predict failed: %s" % e)
        ctx.up = {p for p, v in ctx.pred.items() if v[2] >= ctx.args.mask_deg}
        # Common clock-frequency bias = median(measured - predicted) over detected
        # satellites. A tight residual spread confirms the sign convention; a spread with
        # resid ~ -2x predicted means doppler_sign (cli._FROZEN) is inverted.
        # Only fresh detections (det_fresh, bias_det_fresh_s) enter: the search's REST table
        # keeps serving a detection after it was made while the prediction advances at the
        # satellite's Doppler rate, so a stale detection's residual grows with its age and
        # reads as a clock bias the seeds then chase. --bias-min-snr gates too: the median is
        # a common-mode estimate with uncertainty (per-satellite Doppler error)/sqrt(N), so
        # one satellite whose Doppler is noise does real damage when N is small.
        resid = [
            ctx.best[p][1] - ctx.pred[p][0]
            for p in ctx.best
            if p in ctx.pred
            and now_utc_s(ctx) - ctx.det_fresh.get(p, (None, 0.0))[1]
            < ctx.args.bias_det_fresh_s
            and ctx.best[p][0] >= ctx.args.bias_min_snr
        ]
        # Time base: the same residuals, read for their spread rather than their median. A
        # spread that one epoch shift explains is a stale sample-0 epoch on the nodes
        # (gnss_broker/timebase.py).
        timebase.observe(ctx, _log_rl)
        _local_ok = len(resid) >= ctx.args.bias_min_sats
        if _local_ok:
            # The per-cycle median is quantized to the search's Doppler grid and jumps as the
            # detected set changes, while the true bias is a slow oscillator drift. The EMA
            # (bias_alpha) averages the quantization out, so the hints (and the seeds under
            # --seed-bias-source ema) see a stable bias.
            # Gated on bias_min_sats (cli._FROZEN): one satellite's residual is unfalsifiable
            # and a bad one narrows the search into a deadlock. Below the gate the EMA holds
            # its last multi-satellite value, or stays unsolved with the margins wide, which
            # is what lets more satellites in.
            raw_bias = statistics.median(resid)
            _cb_snap = (
                ctx.cb.ema is None or ctx.cb.stale
            )  # captured before the branch clears it
            if _cb_snap:
                # First solve, or a stale-rescue re-solve: snap to the fresh median. An EMA
                # crawl (bias_alpha) from a value latched far from truth would spend minutes
                # converging through the hint region it just vacated.
                if ctx.cb.stale:
                    _log(
                        "CLOCK BIAS RE-SOLVED %+.1f Hz after %.0f s stale (held %+.1f, "
                        "%d sats)"
                        % (raw_bias, ctx.t0 - ctx.cb.meas_t, ctx.cb.ema, len(resid))
                    )
                    # A starved re-solve must not become the calibration, which every later
                    # drift-alarm comparison is measured against: a median of a few satellites
                    # is noise, and a stale reference beats a noise-derived one. Below
                    # clock_bias_cal_min_sats the old calibration is held, or none is set yet;
                    # with cal None the alarm is silent and the bias EMA is unaffected.
                    _n_cal = len(resid)
                    if _n_cal < ctx.args.clock_bias_cal_min_sats:
                        _log(
                            "CLOCK BIAS: re-solve %+.1f Hz on only %d sat(s) (need %d) -- "
                            "%s. A starved re-solve is noise, and "
                            "a wrong reference poisons every later comparison."
                            % (
                                raw_bias,
                                _n_cal,
                                ctx.args.clock_bias_cal_min_sats,
                                ("held the calibration at %+.1f" % ctx.cb.cal)
                                if ctx.cb.cal is not None
                                else "no calibration set yet -- the drift alarm stays silent "
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
                                "hardware news (GPSDO re-settled?); new drift-alarm reference"
                                % (ctx.cb.cal, raw_bias, _n_cal)
                            )
                        ctx.cb.cal = raw_bias
                    ctx.cb.stale = False
                ctx.cb.ema = raw_bias
            else:
                ctx.cb.ema += ctx.args.bias_alpha * (raw_bias - ctx.cb.ema)
            ctx.cb.meas_t = ctx.t0
            ctx.cb.value = ctx.cb.ema
            # The seed consumers' bias; ClockBias.update_seed has the invariant.
            ctx.cb.update_seed(ctx.args.seed_bias_source)
            # Drift alarm on a satellite-scaled bar: the median's noise goes as 1/sqrt(n), so
            # the bar widens by sqrt(5/n) below 5 satellites. A real event (GPSDO unlock,
            # thermal) is large, sustained and hits every chain, so it still trips; a weak
            # chain's solo alarm is almost always noise.
            if ctx.cb.cal is not None:
                _abar = ctx.args.clock_bias_alarm_hz * max(
                    1.0, (5.0 / max(len(resid), 1)) ** 0.5
                )
                if abs(ctx.cb.ema - ctx.cb.cal) > _abar:
                    _log_rl(
                        "clkalarm",
                        "CLOCK DRIFT ALARM: carrier bias %+.1f Hz vs calibration %+.1f "
                        "(|d| > %.0f Hz, %d sats) -- GPSDO unlock / thermal event? Investigate"
                        % (ctx.cb.ema, ctx.cb.cal, _abar, len(resid)),
                        every_s=ctx.args.clock_bias_alarm_every_s,
                    )
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
                    "zero: fixed, the solve is hints-only"
                    if ctx.args.seed_bias_source == "zero"
                    else "= hint EMA",
                ),
            )
        else:
            # Log both counts, so a held or unsolved bias reads as too few satellites rather
            # than a clock or front-end fault.
            _log_rl(
                "clkbias",
                "clock-freq bias %s (%d sats < bias_min_sats %d: residual not "
                "trusted)"
                % (
                    "held %+.0f Hz" % ctx.cb.value
                    if ctx.cb.ema is not None
                    else "UNSOLVED (wide margins)",
                    len(resid),
                    ctx.args.bias_min_sats,
                ),
            )
