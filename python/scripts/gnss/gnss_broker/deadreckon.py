"""The dead-reckon stage's clock pipeline: solve, judge, adopt.

The receiver code clock is what lets a chain seed a satellite it has never detected -- the
whole model-primary spine rests on it. These three steps run in order inside the dead-reckon
stage, and each has a failure mode that looks like success:

  SOLVE   a robust median over per-satellite code offsets.
  JUDGE   is this cycle's solve trustworthy at all (spread, drift, staleness)?
  ADOPT   if this chain cannot solve its own, take the band sibling's.

⚠️ THE MEMBERSHIP MATTERS AS MUCH AS THE MEDIAN. Which satellites are in the pool steps the
median by 1-2 chips when membership churns, on a ~600 s timescale, and that churn was THE
DECAY ROOT: a clock that moves because the population moved is indistinguishable, in the
value alone, from a clock that moved because the receiver did.

⚠️ A PRIME IS NOT A MEASUREMENT. Seeding the clock with a fixed value silences the alarm that
the solve is failing rather than making it succeed. `--dr-clock-chips 0.0` did exactly that
once, and the snap it hid was real.

⚠️ ADOPTION FAILS SILENTLY AND PER BAND. gal_e5b and bds_b2b adopted (l-a) ZERO times while
their 1176.45 MHz siblings adopted it 102 and 107 times -- 150 chips of error against a +-1
chip correlation peak, with nothing in the logs saying so. Gate adoption on whether the
quantity has STOPPED MOVING, never on its age alone: age alone once adopted a clock bouncing
8690 -> 2787 -> 9382 chips between reads.

@author Keith Vanderlinde
"""

import os

from gnss_broker.transport import _log, _log_rl


def dr_clock_solve(ctx):
    """3e-solve: THE RECEIVER CLOCK SOLVE -- the robust median over per-satellite offsets.
        
    ⚠️ MEDIAN, AND THE MEMBERSHIP MATTERS AS MUCH AS THE VALUE. Which satellites are in the pool
    steps the median by 1-2 chips when membership churns, on a ~600 s timescale, and that churn
    was THE DECAY ROOT (chord-clock-median-churn): a clock that moves because the population moved
    looks exactly like a clock that moved because the receiver did."""
    if len(ctx.drp.offs) >= ctx.args.dr_min_sats:
        ref = ctx.drp.offs[0][1]
        cen = sorted(((d - ref + ctx.code_len / 2) % ctx.code_len) - ctx.code_len / 2
                     for _, d in ctx.drp.offs)
        ctx.drp.raw_clk = (cen[len(cen) // 2] + ref) % ctx.code_len
        # DO THE SATELLITES AGREE? A median over >= dr_min_sats is only a
        # measurement if its inputs cluster. Detections from a starved fleet are
        # noise, their offsets scatter ~uniformly over CODE_LEN, and the circular
        # median of that is an arbitrary number -- which was then accepted
        # unconditionally, snapped straight into dr_state["clk"], and shipped as
        # every satellite's seed.
        #
        # THAT IS THE OTHER HALF OF THE 2026-08-10 LOOP (docs 11.33): a wrong
        # clock produced wrong seeds and wrong hints, which prevented the
        # detections that would have corrected it, and the only escape was the
        # clock random-walking back onto truth (6996 -> 5095 -> 1797 -> 1555 ->
        # 9210 -> 134 chips over ~15 min, then a snap to 150.8 and lock). Three
        # broker restarts that afternoon each re-rolled this dice on a thin fleet.
        #
        # THE SEPARATION IS ENORMOUS, so the bound is not a tuning knob: real
        # per-sat offsets cluster inside ~+-10 chips (the joint state measures the
        # biases at +-5, docs 11.22 quotes +-3-7), while uniform noise over a
        # 10230-chip code has a MAD of ~2557. Anything from 50 to 500 separates
        # them; 100 is the middle of that range in log terms.
        _mad = sorted(abs(c - cen[len(cen) // 2]) for c in cen)[len(cen) // 2]
        if _mad > ctx.args.dr_max_solve_mad_chips:
            # HOW LONG HAVE WE BEEN REFUSING? The guard alone is a LATCH: the
            # scatter that triggers it is sustained by the very clock it is
            # protecting, so once the model has drifted off the sky every
            # satellite reads as noise and MAD never comes back down on its own
            # (measured 2045 against a bound of 100, for 14 minutes, until the
            # run was stopped). Escaping needs a deliberate re-roll.
            _since = ctx.dr_state.get("mad_refused_since")
            if _since is None:
                _since = ctx.dr_state["mad_refused_since"] = ctx.drp.now_w
            _held_s = ctx.drp.now_w - _since
            if (ctx.args.dr_solve_refused_rebootstrap_s > 0.0
                    and _held_s >= ctx.args.dr_solve_refused_rebootstrap_s):
                # FORCED RE-BOOTSTRAP. Clearing clk sends the next accepted
                # median through the BOOTSTRAP branch below, which snaps rather
                # than EMAs. This is a re-roll, NOT a measurement -- it restores
                # the random walk that used to escape this state in ~15-20 min
                # and that the guard removed. raw is deliberately left standing.
                _log("clock solve REFUSED for %.0f s (MAD %.0f, bound %.0f) -- "
                     "FORCING A RE-BOOTSTRAP off %d sats. This is a RE-ROLL, not "
                     "a measurement: holding a clock the sky disagrees with is "
                     "self-sustaining, so a fresh draw is strictly better than a "
                     "latch. Was %s"
                     % (_held_s, _mad, ctx.args.dr_max_solve_mad_chips, len(ctx.drp.offs),
                        ("%.2f chips" % ctx.dr_state["clk"])
                        if ctx.dr_state.get("clk") is not None else "UNSET"))
                ctx.dr_state["clk"] = None
                ctx.dr_state.pop("raw_prev", None)
                ctx.dr_state["off_hist"] = {}
                ctx.dr_state["mad_refused_since"] = None
            else:
                _log_rl("clkmad",
                        "clock solve REFUSED: %d sats scatter MAD %.0f chips "
                        "(bound %.0f) -- this is a median over NOISE, not a "
                        "measurement; holding clk %s (%.0f s, re-bootstrap at "
                        "%.0f s)"
                        % (len(ctx.drp.offs), _mad, ctx.args.dr_max_solve_mad_chips,
                           ("%.2f" % ctx.dr_state["clk"]) if ctx.dr_state.get("clk")
                           is not None else "UNSET", _held_s,
                           ctx.args.dr_solve_refused_rebootstrap_s),
                        every_s=30.0)
                ctx.drp.raw_clk = None
        else:
            ctx.dr_state["mad_refused_since"] = None


def dr_clock_quality(ctx):
    """3e-quality: is the solved clock trustworthy this cycle? (MAD, drift, staleness)
        
    ⚠️ A PRIME IS NOT A MEASUREMENT. Seeding the clock with a fixed value silences the alarm that
    the solve is failing rather than making it succeed -- `--dr-clock-chips 0.0` did exactly that
    once, and the snap it hid was real."""
    if len(ctx.drp.offs) >= ctx.args.dr_min_sats and ctx.drp.raw_clk is not None:
        prev_raw = ctx.dr_state.get("raw_prev")
        # A primed drift is authoritative (the GPSDO rate is a band constant):
        # never EMA it toward pair-differences of solutions built from UNCHANGED
        # detections, which difference to ~zero and drag a correct prime away
        # (measured 2026-07-31: primed +0.0439 walked to -61 within a minute).
        if prev_raw is not None and 0.5 < ctx.drp.now_w - prev_raw[1] < 30.0                             and ctx.args.dr_clock_drift is None:
            d_est = (((ctx.drp.raw_clk - prev_raw[0] + ctx.code_len / 2) % ctx.code_len)
                     - ctx.code_len / 2) / (ctx.drp.now_w - prev_raw[1])
            # PLAUSIBILITY BOUND. d_est is a DIFFERENCE OF TWO CLOCK SOLVES, so
            # anything that displaces the solve -- a node restart, an F-engine
            # restart, detections straddling either -- lands here as clock
            # "motion" that never happened. Measured 2026-08-09: every node
            # restart poisoned this EMA (+223 chips/s once, -36 another), and
            # because the EMA is a=0.05 at ~30 s the poison then bleeds off over
            # ~10 MINUTES, during which `drift` sweeps every model-primary seed
            # off its peak. GPS survives (search-anchored); E5a/B2a/E5b/B2b do
            # not -- E5a measured 96 -> 5 while GPS sat at 352 in the same
            # minute. The bound is physics, not tuning: this clock is
            # GPS-disciplined and its true drift is ~4e-4 chips/s (the joint
            # state measures it), while the noisiest legitimate estimator we
            # have scatters +-0.07. A whole chip per second is ~2500x the truth
            # and 14x that scatter, so nothing real is being rejected.
            if abs(d_est) > ctx.args.dr_max_drift_chips_s:
                _log_rl("driftrej",
                        "clock drift estimate %+.1f chips/s REJECTED (bound "
                        "%.2f): the solve jumped, the clock did not -- holding "
                        "drift %+.4f" % (d_est, ctx.args.dr_max_drift_chips_s,
                                         ctx.dr_state.get("drift") or 0.0),
                        every_s=30.0)
            else:
                ctx.dr_state["drift"] = (d_est if ctx.dr_state.get("drift") is None
                                     else ctx.dr_state["drift"]
                                     + 0.05 * (d_est - ctx.dr_state["drift"]))
        ctx.dr_state["raw_prev"] = (ctx.drp.raw_clk, ctx.drp.now_w)
        # SNAP ON THE FIRST MEASUREMENT, whether or not a prime is standing.
        # `raw` is a circular median over >= --dr-min-sats satellites, so it is a
        # measurement of the same quantity the prime guessed; EMA-ing from the
        # guess buys nothing but the walk-in. Snapping lands within the median's
        # own scatter (a few chips at 6 sats) on cycle one instead of ~20 cycles.
        if ctx.dr_state["clk"] is None or ctx.dr_state.pop("clk_primed", False):
            was = ctx.dr_state["clk"]
            ctx.dr_state["clk"] = ctx.drp.raw_clk
            _log("dead-reckon: receiver clock BOOTSTRAP %.2f chips = %.3f us "
                 "(mod %.0f ms; %d sats%s)"
                 % (ctx.drp.raw_clk, ctx.drp.raw_clk / ctx.args.chip_rate_hz * 1e6, ctx.drp.t_code * 1e3, len(ctx.drp.offs),
                    "" if was is None else
                    "; REPLACES the %.2f-chip prime -- a prime is a seed, not a "
                    "measurement" % was))
        else:
            clk = (ctx.dr_state["clk"]
                   + ctx.drp.drift * (ctx.drp.now_w - ctx.dr_state["clk_t"])) % ctx.code_len
            step = ((ctx.drp.raw_clk - clk + ctx.code_len / 2) % ctx.code_len) - ctx.code_len / 2
            ctx.dr_state["clk"] = (clk + ctx.args.dr_clock_alpha * step) % ctx.code_len
        ctx.dr_state["clk_t"] = ctx.drp.now_w
        # CONTRIBUTE (task #27 M3). THIS IS THE SEAM --dr-clock-adopt PAPERS
        # OVER: dr_state straddles the boundary -- clk and drift are the
        # RECEIVER's, seeded/pin/pd are this chain's per-PRN bookkeeping. A
        # co-hosted chain with no detectors reads this directly instead of a
        # JSON file written at flush cadence and gated on a two-read slew test.
        # Carried WITH its code length, because chips are modular and a value
        # mod 10230 is meaningless to a 1023000-chip code.
        ctx.rx.contribute_dr_clock(ctx.chain_id, ctx.band_id, ctx.dr_state["clk"],
                               ctx.dr_state.get("drift"), ctx.drp.now_w, ctx.code_len)


def dr_clock_adopt(ctx):
    """3e-adopt: ADOPT the sibling band's receiver clock when this chain cannot solve its own.
        
    The two bands of one satellite share a receiver clock, so a chain with no detections can take
    its sibling's -- which is how the searchless chains bootstrap at all.
        
    ⚠️ AN ADOPTED CLOCK IS A MEASUREMENT, because the sibling measured it -- but gate on whether
    the quantity has STOPPED MOVING, not on its age alone. Age alone once adopted a clock bouncing
    8690 -> 2787 -> 9382 chips between reads.
        
    ⚠️ THE FAILURE MODE IS SILENT AND PER-BAND. gal_e5b and bds_b2b adopted (l-a) ZERO times while
    their 1176.45 MHz siblings adopted it 102 and 107 times -- 150 chips of error against a +-1
    chip peak, with nothing in the logs saying so."""
    if (ctx.drp.rx_sib is None and ctx.args.dr_clock_adopt and not ctx.drp.offs and ctx.xb_read_dir
            and ctx.args.state_dongle):
        try:
            import receiver_state as _rs  # optional module, imported where used
            sibs = _rs.read_dongle(
                ctx.xb_read_dir, ctx.args.state_dongle,
                max_age_s=ctx.args.dr_clock_adopt_max_age_s, t_now=ctx.t0,
                exclude=(os.path.basename(ctx.args.state_file).rsplit(".", 1)[0]
                         if ctx.args.state_file else None))
        except Exception:
            sibs = []
        # Freshest wins. Not a weighted mean: this is an adoption of ONE physical
        # number measured by a chain that can measure it, and averaging two siblings'
        # copies of the same quantity would just average their noise back in.
        best_sib = None
        _refused = False
        for rec in sibs:
            # GROUP NAME IS "rxclock", read off a live state file rather than
            # inferred from the observe() call site -- the first attempt guessed
            # "dr" from the surrounding variable names and silently found nothing,
            # logging "no fresh sibling" while a perfectly good record sat there.
            dr = (rec.get("rxclock") or {})
            if dr.get("chips") is None:
                continue
            if best_sib is None or float(rec.get("t", 0)) > float(best_sib[0]):
                best_sib = (rec.get("t", 0), rec, dr)
        # QUALITY GATE: JUDGE THE CLOCK BY WATCHING IT, not by the sibling's
        # aggregate quality fields. Two earlier versions of this gate were wrong in
        # opposite directions and both cost time:
        #
        #   * age alone -> adopted a clock bouncing 8690 -> 2787 -> 9382 chips
        #     minutes after the sibling restarted.
        #   * `untrusted >= n` -> refuses forever. Those count DIFFERENT
        #     populations: n = len(_ir), the satellites contributing an integrity
        #     residual THIS cycle; untrusted = len(dr_untrusted), the persistent
        #     demoted set. untrusted > n is normal after a restart, not impossible,
        #     and comparing them is meaningless.
        #   * integ_mad_chips > 1.0 -> also refuses a good clock. That MAD is taken
        #     over a mixed population including badly-tracked satellites; measured
        #     2026-08-08 it sat at 3.3-3.9 chips while the CLOCK ITSELF was stable
        #     to 0.2 chips across consecutive reads.
        #
        # So gate on the quantity being adopted: has it stopped moving? Two reads a
        # cycle apart answer that directly, in seconds, with no appeal to anyone's
        # self-reported quality. A converged clock moves by its drift; a
        # non-converged one moves by thousands of chips.
        if best_sib is not None:
            _, rec, dr = best_sib
            _cand = float(dr["chips"]) % ctx.code_len
            _prev = ctx.dr_state.get("adopt_prev")
            if _prev is None:
                ctx.dr_state["adopt_prev"] = (_cand, ctx.t0)
                _log_rl("clkadopt-watch",
                        "dead-reckon: watching sibling '%s' clock %.2f chips -- "
                        "adopting once a second read confirms it is not moving "
                        "(one cycle, not a burn-in)" % (rec.get("chain", "?"), _cand))
                best_sib = None
                _refused = True
            else:
                _pc, _pt = _prev
                _dt = max(ctx.t0 - _pt, 1e-6)
                _move = abs(((_cand - _pc + ctx.code_len / 2) % ctx.code_len)
                            - ctx.code_len / 2) / _dt
                ctx.dr_state["adopt_prev"] = (_cand, ctx.t0)
                if _move > ctx.args.dr_clock_adopt_max_slew:
                    _log_rl("clkadopt-q",
                            "dead-reckon: REFUSED sibling '%s' clock -- moving "
                            "%.1f chips/s (limit %.1f). It has not converged; "
                            "holding %.2f chips."
                            % (rec.get("chain", "?"), _move,
                               ctx.args.dr_clock_adopt_max_slew,
                               ctx.dr_state["clk"] if ctx.dr_state.get("clk") is not None
                               else float("nan")))
                    best_sib = None
                    _refused = True
        if best_sib is not None:
            _, rec, dr = best_sib
            new_clk = float(dr["chips"]) % ctx.code_len
            prev = ctx.dr_state.get("clk")
            moved = (abs(((new_clk - prev + ctx.code_len / 2) % ctx.code_len)
                         - ctx.code_len / 2) if prev is not None else None)
            ctx.dr_state["clk"] = new_clk
            ctx.dr_state["clk_t"] = ctx.t0
            if dr.get("drift_chips_s") is not None:
                ctx.dr_state["drift"] = float(dr["drift_chips_s"])
            # Loud on a real MOVE, quiet on the steady state. A jump is the
            # signature of an F-engine restart re-establishing frame 0, which is
            # precisely the event the hand-primed constant used to survive wrongly.
            if moved is None or moved > 0.5:
                _log("dead-reckon: clock ADOPTED %.2f chips from band sibling "
                     "'%s' (%s%s, age %.1f s)"
                     % (new_clk, rec.get("chain", "?"),
                        "cold" if moved is None else "moved %.2f chips" % moved,
                        ", drift %+.4f chips/s" % dr["drift_chips_s"]
                        if dr.get("drift_chips_s") is not None else "",
                        ctx.t0 - float(rec.get("t", ctx.t0))))
            else:
                _log_rl("clkadopt", "dead-reckon: clock adopted %.2f chips from "
                                    "'%s' (steady)" % (new_clk, rec.get("chain", "?")))
        elif ctx.args.dr_clock_adopt and not _refused:
            # NOT after a quality refusal -- that path logs its own reason. Saying
            # "no fresh sibling" when a sibling was found and REJECTED describes the
            # wrong failure, and the two want opposite responses: absent means check
            # the publisher, rejected means wait for it to converge.
            _log_rl("clkadopt-none",
                    "dead-reckon: --dr-clock-adopt found no fresh sibling for dongle "
                    "'%s' in %s (<%.0f s) -- HOLDING the primed clock %.2f chips, "
                    "which does not survive an F-engine restart"
                    % (ctx.args.state_dongle, ctx.xb_read_dir,
                       ctx.args.dr_clock_adopt_max_age_s,
                       ctx.dr_state.get("clk") if ctx.dr_state.get("clk") is not None else float("nan")))
