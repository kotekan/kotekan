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

import math
import os
import time
from datetime import datetime, timezone

from gnss_broker.transport import _now, _post, _log, _log_rl
from gnss_broker.seed import Seed
from gnss_broker.fits import (
    dr_seed_phys, dr_cp0, cp_rate_from_code_bias, seed_phase_at_ref, split_erratic_offsets,
)


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


def dr_joint_shadow(ctx):
    """3e-shadow: THE JOINT RECEIVER FEED (#33 GAP 2) -- feed per-sat offsets to the joint filter.
        
    Shadow by construction: it estimates, logs, and (where armed) publishes a state, but the
    seeding sub-stage below decides what actually reaches the trackers.
        
    ⚠️ FEEDING AND CONSUMING THE SAME QUANTITY IS A MIRROR, NOT A MEASUREMENT. The model-primary
    feed once measured its own seeded lock-and-arm rate (#33 GAP 2's nested self-reference); the
    spec-anchored feed exists so what goes in is not what this loop just put there.
        
    ⚠️ ITS `legacy clk` COLUMN IS ONE CYCLE STALE -- see the `_DrProducts` note. Do not compare it
    against the joint clock printed beside it without accounting for the lag."""
    if ctx.args.joint_shadow and ctx.drp.offs:
        try:
            # P3: ONE state for the whole receiver, band carried as a
            # measurement LABEL rather than as a separate filter. Two per-band
            # states cannot estimate the offset between them, which is exactly
            # why the second band needed a hand-wired cross-band clock bootstrap
            # (#34) -- tau_band now has somewhere to live.
            # WARM START (2026-08-11): hand the filter the legacy clock at
            # CREATION. Born at 0 against a true ~151 chips, every measurement
            # lands beyond escape_max_step and the escape hatch refuses the
            # correction as non-physical -- 100% rejection, forever (observed
            # 15:25 UTC: 34 rejections agreeing to 0.22 chips, all implying
            # +151.7, all refused). Only read on creation, so this cannot fight
            # the filter once it is running. #28's lesson, applied one filter on.
            _js = ctx.rx.joint_receiver(ctx.band_id, ctx.code_len,
                                    clk0=float(ctx.dr_state.get("clk") or 0.0))
            # #33 gap 3: refresh the coupling constant on the SHARED object
            # every cycle rather than only at construction -- any consumer
            # site can be the creator (thread startup order), and a kwarg
            # passed only here would silently lose that race. Idempotent:
            # the flag lives in the common yaml section, same value from
            # every chain. A change of value is logged by the filter's own
            # F-matrix taking effect (0.0 = identity, no code path change).
            _js.rr_bsat_chips_per_m = float(ctx.args.rr_bsat_chips_per_m)
            # SNR GATE, caller-side. The broker's own comment 60 lines up
            # measures it: below --period-check-snr a detection's phase is
            # noise, "~2000-chip within-period residuals against a few chips
            # above it". The estimator this replaces was a circular MEDIAN and
            # shrugged those off; a mean-gauged filter cannot, and feeding them
            # ungated on 2026-08-09 walked the clock rate to -0.028 ppm and put
            # 17 chips/min of fictitious drift into every unlocked seed.
            _snr = {p: v[0] for p, v in ctx.best.items()}
            # P2c: withhold the masked sats, then ask the state where it thinks
            # they are. `predicted` is clk + b_i, so the residual below is exactly
            # "what the shared state got wrong about a satellite it is no longer
            # being told about".
            # FEED WARMUP (2026-08-12, the zombie's root): no measurements
            # until the establishment window has passed -- see the flag help.
            if time.time() - ctx.broker_t0 < ctx.args.joint_feed_warmup_s:
                _log_rl("jwarm", "JFEED WARMUP: withholding the joint feed "
                        "(%.0f s of %.0f remain) -- establishment-phase "
                        "measurements must not become birth geometry"
                        % (ctx.args.joint_feed_warmup_s
                           - (time.time() - ctx.broker_t0),
                           ctx.args.joint_feed_warmup_s), every_s=60.0)
            else:
                # ── #83 P3-3a: THE MODEL INNOVATION (MINNOV) ──
                # The same residual P2C measures one coasted satellite at a
                # time -- wrap(d - predicted - tau), against the PRIOR state,
                # before this cycle's measurements are consumed -- computed
                # continuously for every ESTABLISHED satellite: "could the
                # joint state have placed this satellite without this
                # detection?". This is the per-PRN model-primacy flip gate's
                # number (INNOV judges the commanded seed; MINNOV judges the
                # MODEL). Established-rows-only: a birth row's prediction is
                # its own first measurement. SERVED ONLY (publisher + log).
                for _p3, _d3 in ctx.drp.offs:
                    _k3 = (ctx.drp.tag, _p3)
                    if (_snr.get(_p3, 0.0) >= ctx.args.joint_min_snr
                            and _k3 in _js._idx
                            and _js._n.get(_k3, 0) >= ctx.args.joint_mask_after):
                        _mi = _js.wrap(_d3 - _js.predicted(_k3)
                                       - _js.tau(ctx.band_id))
                        _mh = ctx.minnov_hist.setdefault(_p3, [])
                        _mh.append((ctx.t0, _mi))
                        del _mh[:-120]
                _js.cycle([((ctx.drp.tag, p), d, ctx.args.joint_sigma, ctx.band_id)
                           for p, d in ctx.drp.offs
                           if _snr.get(p, 0.0) >= ctx.args.joint_min_snr
                           and ctx.track_ok(p)
                           and not ctx.p2c_hold(_js, (ctx.drp.tag, p))],
                          ctx.drp.t_now_abs)
            # The filter has no logger; drain what it wants an operator to see.
            # An escape or an incoherent run is a tracking event worth a line --
            # on 2026-08-10 the single most damaging update of the day fired
            # completely silently and was only found by its consequences.
            for _n in _js.drain_notes():
                _log_rl("joint-note", "JOINT %s: %s" % (ctx.band_id, _n),
                        every_s=10.0)
            _drained = True
            ctx.p2c_tick(_js, ctx.drp.t_now_abs)
            for _p, _d in ctx.drp.offs:
                if ctx.p2c_hold(_js, (ctx.drp.tag, _p)):
                    _r = _js.wrap(_d - _js.predicted((ctx.drp.tag, _p)) - _js.tau(ctx.band_id))
                    if ctx.p2c["key"] == (ctx.drp.tag, _p):
                        ctx.p2c["samples"].append((ctx.drp.t_now_abs - ctx.p2c["t0"], _r))
                    _log_rl("p2c-%d" % _p,
                            "P2C %s PRN %d MASKED %.0fs: coast residual %+.3f chips "
                            "(b %+.3f, sigma %.3f, tau %+.4f) -- flat = the state "
                            "carries it"
                            % (ctx.band_id, _p, _js.age((ctx.drp.tag, _p), ctx.drp.t_now_abs) or 0.0,
                               _r, _js.bias((ctx.drp.tag, _p)), _js.sigma((ctx.drp.tag, _p)),
                               _js.tau(ctx.band_id)),
                            every_s=30.0)
            if ctx.drp.now_w >= ctx.dr_state.get("joint_log_next", 0.0):
                ctx.dr_state["joint_log_next"] = ctx.drp.now_w + 30.0
                # tau_band is reported WITH its observability count, never alone.
                # It separates from b_sat only through satellites seen in BOTH
                # bands, so a tau printed next to "dual 0" is an artefact of the
                # priors -- which is precisely the state the disjoint E5a/E5b PRN
                # lists put the instrument in while every chain looked healthy.
                _tb = "".join(
                    "  tau[%s] %+.3f+-%.3f (dual %d)"
                    % (_b, _js.tau(_b), _js.tau_sigma(_b),
                       _js.tau_observability(_b))
                    for _b in sorted(_js._band_idx))
                _amb = "  ⚠AMBIGUOUS(clk near wrap)" if ctx.rx.joint_ambiguous() else ""
                _log("JOINT[shadow] " + _js.summary(ctx.drp.t_now_abs) + _tb + _amb)
        except Exception as e:      # shadow must never take the broker down
            _log_rl("jointerr", "JOINT[shadow] disabled this cycle: %s" % e,
                    every_s=300.0)
    # -- P3: THE MODEL-PRIMARY MEASUREMENT (task #33) ------------------------
    # Everything above needs `offs`, which only a chain with DETECTIONS has. So
    # until now only GPS fed the joint state -- 4 shadow lines against 0 for every
    # model-primary chain -- and with one band contributing there was no second
    # band for tau_band to be a delay AGAINST. This is the other half of the same
    # measurement, and the plan has specified it since section 3a:
    #
    #     y_i = (where the replica actually is) - (pure model, no clock, no bias)
    #         = dr_seed_phys(seed) + dll_trim - cp_predicted
    #
    # dr_seed_phys is the tracker's own phase model evaluated at h1 (the same
    # function the slew block uses to ask "where does the tracker think the code
    # is"), dll_trim is the fleet loop's standing correction to it, and
    # cp_predicted is BRDC geometry alone. The difference satisfies the identical
    # y_i = clk + b_i + tau_band as the search-anchored form, which is what makes
    # two seeding disciplines poolable into one state at all.
    #
    # SEEDS ARE LAST CYCLE'S HERE, deliberately: this runs before the seed loop
    # rebuilds them, and "where the replica actually is" IS the seed the trackers
    # are flying right now. Reading a seed the broker has not yet shipped would
    # measure an intention rather than the instrument.
    #
    # No SNR gate to mirror -- a model-primary chain has no detection SNR. The
    # protection is the filter's own innovation gate plus birth_max, which is why
    # those were built with an escape hatch.
    elif ctx.args.joint_model_primary and ctx.args.joint_shadow and ctx.seeds and not ctx.drp.offs:
        try:
            _js = ctx.rx.joint_receiver(ctx.band_id, ctx.code_len,   # warm start, see above
                                    clk0=float(ctx.dr_state.get("clk") or 0.0))
            _js.rr_bsat_chips_per_m = float(ctx.args.rr_bsat_chips_per_m)  # see 3a
            _h1 = int(round(ctx.drp.t_now_abs * ctx.args.hops_per_sec))
            _th = _h1 / ctx.args.hops_per_sec
            _mm = []
            _fd_skip = 0
            for _prn, _sd in ctx.seeds.items():
                _v = ctx.drp.pd.get((ctx.drp.tag, _prn))
                if _v is None or "ref_hop" not in _sd:
                    continue
                # THE GATE THIS FEED NEVER HAD. A dead-reckoned seed carries no
                # detection SNR, which is why it was originally fed ungated --
                # but "no detection SNR" is not "no quality signal": the tracker
                # says plainly whether it is seeing anything. Without this the
                # feed offers EVERY seeded satellite every cycle, including ones
                # despreading pure noise, and on 2026-08-09 it walked clk to
                # +445 against a true ~150 with 67-82% of updates rejected.
                if not ctx.track_ok(_prn):
                    continue
                # ── THE DLL MUST BE IN ITS LINEAR RANGE (2026-08-22) ──────────
                # y = held + trim - cp_predicted. The tracker despreads at S+T
                # (seed + applied trim) and the DLL drives T so that S+T sits on
                # the peak -- so S+T is SKY-ANCHORED and y is a real measurement:
                # move S and the DLL moves T oppositely, leaving y invariant.
                # THAT is why feeding and consuming on one chain is legitimate in
                # principle, and it is the answer to "a Kalman filter is supposed
                # to feed and consume".
                # IT ONLY HOLDS WHILE THE DLL CAN DO ITS HALF. Past ~1 chip the
                # correlation triangle has no gradient, so T stops compensating,
                # S+T follows S, and y reports the consumer's own output -- the
                # loop closes with no observation in it. Measured 2026-08-22:
                # median |model - held| is 1.8-2.5 chips, i.e. ALREADY outside the
                # measurable range for most satellites, which is why the feed
                # diverged when it was armed with no gate at all.
                # So feed only satellites the loop is demonstrably holding: q at
                # the lock bar AND a trim well inside the clamp. A railed or
                # near-railed trim means the seed error exceeds what the
                # discriminator can see, and that satellite's y is not evidence.
                # ── #85: SPEC ANCHORING ──────────────────────────────
                # sky = held + applied_trim + spec_tau IDENTICALLY (the fit
                # measures sky-minus-replica and the replica IS held+trim), so
                # with a fresh, significant fit y is sky-anchored at ANY
                # displacement -- including past the DLL's linear range, where
                # the trim gate below would otherwise exclude the satellite.
                # A consumed clock moving the seed moves trim+spec_tau
                # oppositely; y is invariant. THAT is the mirror's removal.
                _sp = ((ctx.dr_state.get("spec_y") or {}).get(_prn)
                       if ctx.args.joint_feed_spec else None)
                _sp_ok = (_sp is not None
                          and ctx.drp.t_now_abs - _sp[2]
                          <= ctx.args.joint_feed_spec_max_age_s
                          and _sp[1] >= ctx.args.joint_feed_min_ratio)
                if ctx.args.joint_feed_max_trim > 0.0 and not _sp_ok:
                    _fl_i = (ctx.dllp.fleet or {}).get(_prn) or {}
                    _q_i = _fl_i.get("q")
                    _tr_i = abs(float((ctx.dls.readback.get(_prn) or {})
                                      .get("trim_chips") or 0.0))
                    if (_q_i is None or _q_i < ctx.args.lock_q
                            or _tr_i >= ctx.args.joint_feed_max_trim):
                        _fd_skip += 1
                        continue
                _held = dr_seed_phys(_sd, _h1, ctx.args.hops_per_sec,
                                     ctx.args.chip_rate_hz, ctx.args.carrier_hz,
                                     ctx.args.code_doppler_sign, ctx.drp.mod)
                # ⚠️ THE TRIM THE TRACKER ACTUALLY APPLIED, not the one this
                # process happens to hold (2026-08-21). Authority over the code
                # trim is per-PRN: Python integrates only for PRNs the C++ fleet
                # loop is NOT actuating (`if prn not in _ft_armed_last`), so on a
                # chain the fast loop owns -- gal_e5a owns nearly all of it --
                # dll_trim is ZERO for exactly the satellites carrying a real
                # standing trim. Measured 15:01 on gal_e5a: C++ held 11:+1.601
                # 19:+1.795 25:+2.920 29:+1.984 while dll_trim read +0.00/-0.03.
                # So this measurement was systematically wrong by up to ~3 chips
                # per satellite, and -- because the trims MOVE (PRN 11 3.000 ->
                # 1.601, PRN 25 1.652 -> 2.920 inside a minute) -- their drifting
                # mean was injected into the shared state as a spurious CLOCK
                # RATE (~0.011 chips/s from the readback alone). A feed that does
                # not know what the actuator did is measuring its own loop.
                # #76's readback is exactly this number; it is one cycle old
                # (populated later in the cycle), which is causal and correct.
                _trim_applied = (
                    float((ctx.dls.readback.get(_prn) or {}).get("trim_chips") or 0.0)
                    if _prn in ctx.dls.armed_last
                    else ctx.dls.trim.get(_prn, 0.0))
                _y = ((_held + _trim_applied
                       + (_sp[0] if _sp_ok else 0.0)
                       - ctx.cp_predicted(_v, _th)) % ctx.drp.mod)
                if ctx.p2c_hold(_js, (ctx.drp.tag, _prn)):
                    if True:
                        _r = _js.wrap(_y - _js.predicted((ctx.drp.tag, _prn)) - _js.tau(ctx.band_id))
                        if ctx.p2c["key"] == (ctx.drp.tag, _prn):
                            ctx.p2c["samples"].append((ctx.drp.t_now_abs - ctx.p2c["t0"], _r))
                        _log_rl("p2c-%d" % _prn,
                                "P2C %s PRN %d MASKED %.0fs: coast residual %+.3f "
                                "chips (b %+.3f, tau %+.4f)"
                                % (ctx.band_id, _prn,
                                   _js.age((ctx.drp.tag, _prn), ctx.drp.t_now_abs) or 0.0,
                                   _r, _js.bias((ctx.drp.tag, _prn)), _js.tau(ctx.band_id)), every_s=30.0)
                    continue
                _mm.append(((ctx.drp.tag, _prn), _y, ctx.args.joint_sigma, ctx.band_id))
            # ── #85: THE SET GATE. Eligibility is a property of the SET --
            # 1-2 measurements have spread ~ 0 by construction and a single
            # bad y IS the poll (the 01:xx degenerate feed). Withhold rather
            # than feed thin; the state coasts on clk_rate, which is exactly
            # what it is for.
            if _mm and len(_mm) < ctx.args.joint_feed_min_set:
                _log_rl("jfeed-thin",
                        "JFEED %s: only %d satellite(s) qualify (< %d) -- "
                        "WITHHELD; a thin set feeds its own noise as clock"
                        % (ctx.band_id, len(_mm), ctx.args.joint_feed_min_set),
                        every_s=60.0)
                _mm = []
            if _mm:
                # ── JFEED: THE FORK THIS EXISTS TO SETTLE (task #33, 2026-08-11)
                # Turning this feed on gave 9 Galileo b_sat all equal to +0.44
                # (0.01 chip spread) against 8.9 chips across 3 GPS ones. Two
                # explanations need opposite fixes and the log could not tell
                # them apart, because cycle() returns only a COUNT and update()
                # returns None on reject -- every per-satellite fact was thrown
                # away at the call site:
                #
                #   (a) DEGENERATE AT SOURCE. y = dr_seed_phys(seed) + dll_trim
                #       - cp_predicted, and on a model-primary chain the seed is
                #       itself BUILT from model + clock -- so y may just report
                #       back the clock that built it, carrying no per-sat
                #       information. Self-reference: right in the mean, wrong in
                #       the variance. Signature: spread(y - clk) ~ 0.
                #   (b) DISTINCT BUT REJECTED. The measurements differ per sat,
                #       the innovation gate throws them out, and b_sat stays
                #       frozen at its birth value while clk drags it along.
                #       Signature: spread(y - clk) is chips-scale, |r| large,
                #       acc low.
                #
                # So log the RAW y, the innovation r the gate actually tests,
                # and the accepted count -- all three before/around the update,
                # since predicted() moves once cycle() runs. Costs one line per
                # 10 s and touches no control flow.
                _diag = None
                if ctx.drp.now_w >= ctx.dr_state.get("jfeed_log_next", 0.0):
                    ctx.dr_state["jfeed_log_next"] = ctx.drp.now_w + 10.0
                    _diag = [(_k[1], _yy,
                              _js.wrap(_yy - _js.predicted(_k) - _js.tau(_bd)),
                              _js.bias(_k))
                             for _k, _yy, _sg, _bd in _mm]
                    # TERM DECOMPOSITION. The innovation came back at a common
                    # -135 chips on every satellite with y ramping 0.15 chips/s
                    # (300x the clock rate), so the question is no longer "which
                    # satellite" but "which TERM of y". Print the three summands
                    # against the LEGACY offset the same chain is successfully
                    # seeding from -- E5a tracks fine at deep_snr 80+, so the
                    # legacy number is the working reference and y has to be
                    # compared to it, not to zero.
                    # ⚠️⚠️ THIS DIAGNOSTIC PRINTED THE WRONG VARIABLE UNTIL
                    # 2026-08-21 23:3x, AND IT NEARLY BOUGHT A WRONG CONCLUSION.
                    # It logged `dll_trim` -- the PYTHON dict -- while the actual
                    # measurement `_y` a few lines above uses `_trim_applied`,
                    # the C++ READBACK. For an armed PRN dll_trim is zero BY
                    # DESIGN (Python stands down; see #51), so the line read
                    # "dll_trim +0.000" on 22 of 27 samples and invited the
                    # reading "there is no sky term in y" -- when the fleet DLL
                    # was reporting disc +0.13..+0.51 chips at q>3 on those very
                    # satellites. A diagnostic that prints a different quantity
                    # than the code under test is worse than no diagnostic.
                    #
                    # WHAT THIS NOW ANSWERS, and it is the question KV put:
                    # a Kalman loop is SUPPOSED to feed and consume -- that is
                    # vector tracking -- and closing the loop is stabilising
                    # PROVIDED the fed-back quantity is an OBSERVATION. A
                    # discriminator is an observation (it correlates the replica
                    # against the sky). `held - cp_pred` is a difference of two
                    # PREDICTIONS and observes nothing. So the diagnosis turns
                    # entirely on HOW MUCH SKY IS IN y, which is what `trim` and
                    # `disc` measure. Print them, per satellite, next to the term
                    # they are supposed to correct.
                    # `fleet` carries the previous cycle's per-PRN disc (it is
                    # rebuilt later in this cycle) -- one cycle old, the same
                    # causality the readback already has, and guarded because it
                    # does not exist on the first pass.
                    try:
                        _dfl = ctx.dllp.fleet
                    except NameError:
                        _dfl = {}
                    for _dk, _dyy, _dsg, _dbd in _mm[:4]:
                        _dsd = ctx.seeds.get(_dk[1]) or {}
                        _dv = ctx.drp.pd.get(_dk)
                        if _dv is None or "ref_hop" not in _dsd:
                            continue
                        _dheld = dr_seed_phys(_dsd, _h1, ctx.args.hops_per_sec,
                                              ctx.args.chip_rate_hz, ctx.args.carrier_hz,
                                              ctx.args.code_doppler_sign, ctx.drp.mod)
                        _dcp = ctx.cp_predicted(_dv, _th)
                        _darm = _dk[1] in ctx.dls.armed_last
                        _dtrim = (float((ctx.dls.readback.get(_dk[1]) or {})
                                        .get("trim_chips") or 0.0)
                                  if _darm else ctx.dls.trim.get(_dk[1], 0.0))
                        _drow = _dfl.get(_dk[1]) or {}
                        _ddisc = _drow.get("disc")
                        _dq = _drow.get("q")
                        _log("JFEED-TERMS %s PRN %d [%s]: held %+.3f  "
                             "trim_applied %+.4f (py %+.4f)  disc %s q %s  "
                             "cp_pred %+.3f -> y %+.3f | legacy clk %+.3f + b "
                             "%+.3f = %+.3f | joint clk %+.3f"
                             % (ctx.band_id, _dk[1],
                                "ARMED-cpp" if _darm else "python",
                                _dheld, _dtrim, ctx.dls.trim.get(_dk[1], 0.0),
                                ("%+.4f" % _ddisc) if _ddisc is not None else "-",
                                ("%.2f" % _dq) if _dq is not None else "-",
                                _dcp,
                                ((_dheld + _dtrim - _dcp) % ctx.drp.mod),
                                ctx.drp.clk_now, ctx.bsat.get(_dk[1], ctx.drp.now_w),
                                ctx.drp.clk_now + ctx.bsat.get(_dk[1], ctx.drp.now_w), _js.clk))
                _nok = _js.cycle(_mm, ctx.drp.t_now_abs)
                if _diag:
                    _ys = [_js.wrap(d[1] - _js.clk) for d in _diag]
                    _sp = max(_ys) - min(_ys)
                    _log("JFEED %s: %d meas, %d accepted (%.0f%%)  "
                         "spread(y-clk) %.4f chips  -> %s | %s"
                         % (ctx.band_id, len(_mm), _nok,
                            100.0 * _nok / max(1, len(_mm)), _sp,
                            "DEGENERATE (no per-sat info)" if _sp < 0.05
                            else "per-sat info PRESENT",
                            " ".join("%s%d y%+.3f r%+.3f b%+.3f"
                                     % (ctx.drp.tag, p, y, r, b)
                                     for p, y, r, b in sorted(_diag))))
            if ctx.drp.now_w >= ctx.dr_state.get("joint_log_next", 0.0):
                ctx.dr_state["joint_log_next"] = ctx.drp.now_w + 30.0
                _tb = "".join(
                    "  tau[%s] %+.3f+-%.3f (dual %d)"
                    % (_b, _js.tau(_b), _js.tau_sigma(_b),
                       _js.tau_observability(_b))
                    for _b in sorted(_js._band_idx))
                _amb = "  ⚠AMBIGUOUS(clk near wrap)" if ctx.rx.joint_ambiguous() else ""
                _log("JOINT[shadow] " + _js.summary(ctx.drp.t_now_abs) + _tb + _amb)
        except Exception as e:
            _log_rl("jointerr-mp",
                    "JOINT[shadow] model-primary feed skipped: %s" % e,
                    every_s=300.0)


def dr_seed(ctx):
    """3e-seed: BIRTH, SLEW or HAND BACK every visible satellite's seed from the model.
        
    The stage that actuates: it propagates the ephemeris and clock to a code phase and Doppler and
    writes the seed each tracker will despread with. It computes `_drp.clk_now`.
        
    ⚠️ NAME THE EPOCH OF EVERY TERM. Every walkoff this project has chased was born here, in a mix
    of two time bases: the F-engine axis against the wall clock (the 08-17 axis fix), and
    `cp_predicted` extrapolating from AXIS age while `predict_all` ran at WALL -- error = K*dop*lag,
    a defect born WITH the fix that preceded it (chord-trim-clamp-limit-cycle).
        
    ⚠️ A RE-BASE MOVES THE SEED WHILE THE C++ TRIM STILL CARRIES THE OLD CHIPS. That is #92's
    handover, and without it the tap leaves the sky and the trim rebuilds over ~25 minutes -- E3's
    sawtooth. The bound on it is a safety argument, not a tuning knob."""
    if ctx.dr_state["clk"] is not None:
        ctx.drp.clk_now = (ctx.dr_state["clk"]
                   + ctx.drp.drift * (ctx.drp.now_w - ctx.dr_state["clk_t"])) % ctx.code_len
        # -- P2b CONSUMER "clk" (2026-08-11, the decay root's fix) -------------
        # clk_now above is the circular MEDIAN of per-sat offsets whose per-sat
        # biases span ~11 chips; with 4-7 sats in the solve, every membership
        # change steps it 1-2 chips, and the set churns on the search's revisit
        # timescale -- measured 22:20-22:50: a +-1-2 chip ~600 s oscillation
        # that dragged every model-primary seed off-peak together while the
        # JOINT filter (clk + per-sat b estimated jointly) read the same clock
        # FLAT to +-0.3 (gnss_broker_20260811_drclk.log, DRCLK vs JOINT).
        #
        # So on chains that opt in (`joint-consume: clk`), replace the LEVEL
        # with the joint clock, applied as a WRAPPED DELTA to the median (the
        # lesson of the slew consumer: the two agree only mod CODE_LEN, and the
        # legacy path keeps ownership of the long-code segment). Stateless per
        # cycle -- nothing is written back to dr_state, so a refused/degraded
        # joint falls back to the median seamlessly on the next cycle.
        #
        # GATES: --joint-min-sats in the state (a thin fleet lets one bias leak
        # into the clock), sigma bound (P grows while unfed, so ONE gate covers both
        # estimator health and staleness), and a delta bound against wrap
        # aliases / a diverged filter. The log line prints BOTH clocks every
        # time so the counterfactual median stays visible on the treated chain
        # -- a consumer whose firing cannot be seen is how this month's gates
        # failed.
        if "clk" in ctx.joint_consume:
            _jrC = ctx.joint_state(ctx.rx, ctx.band_id, ctx.args)
            if _jrC is not None and len(_jrC._idx) >= ctx.args.joint_min_sats:
                # sigma <= 0 is DEGENERATE (a zero-gain zombie claims perfect
                # knowledge), not excellent -- refuse it explicitly. The first
                # version wrote `sigma() or inf`, which also flipped a
                # legitimate 0.0 to inf by truthiness accident; with the
                # state_filter P floor sigma cannot reach 0 anymore, but this
                # gate must not depend on that.
                _jsigC = _jrC.sigma()
                if _jsigC is None or _jsigC <= 0.0:
                    _jsigC = float("inf")
                _jdC = ((_jrC.clk - ctx.drp.clk_now + ctx.code_len / 2.0) % ctx.code_len
                        ) - ctx.code_len / 2.0
                _jokC = (_jsigC <= ctx.args.joint_clk_max_sigma
                         and abs(_jdC) <= ctx.args.joint_clk_max_chips)
                _log_rl("jclk",
                        "JOINT-CLK: legacy %.3f joint %.3f chips (delta %+.3f,"
                        " sigma %.3f, n %d) -> %s"
                        % (ctx.drp.clk_now, _jrC.clk % ctx.code_len, _jdC, _jsigC,
                           len(_jrC._idx),
                           "ADOPTED" if _jokC else
                           "REFUSED (bounds %.1f chips / %.2f sigma)"
                           % (ctx.args.joint_clk_max_chips,
                              ctx.args.joint_clk_max_sigma)),
                        every_s=30.0)
                if _jokC:
                    ctx.drp.clk_now = (ctx.drp.clk_now + _jdC) % ctx.code_len
        # DRCLK (2026-08-11): the "dead-reckon clock ..." line above is gated on
        # `offs`, i.e. on this chain having its OWN detections -- so the four
        # model-primary chains, the ones whose seeds ride clk_now raw, never log
        # the clock they are consuming. This line is the missing term in the
        # commanded = cp_predicted + clk_now + b_sat + trim decomposition of the
        # per-sat disc walk (BSAT and DLL lines carry the other two). Logged
        # AFTER the joint-clk adoption on purpose: DRCLK is the clock the seeds
        # actually consume; the JOINT-CLK line above keeps the pre-adoption
        # median visible.
        _log_rl("drclk",
                "DRCLK clk_now %.3f chips drift %+.4f chips/s (la %+.4f ppm)"
                % (ctx.drp.clk_now, ctx.dr_state.get("drift") or 0.0, ctx.drp.la * 1e6),
                every_s=30.0)
        # -- P2b CONSUMER 3, THE SHADOW ARM -------------------------------------
        # Compare the two offset ESTIMATORS directly, over every satellite the
        # joint state holds, independent of whether any seeding path ran this
        # cycle. The first cut logged inside the seeding loop and printed
        # nothing, because that loop skips `prn in best` -- every DETECTED
        # satellite -- while the joint state is fed from `best` and so holds
        # only detected satellites. The two sets are very nearly disjoint by
        # construction, and the overlap is only the handful in transition, so a
        # comparison gated on the loop measures the transition rate rather than
        # the estimators. An A/B must not depend on which code path happened to
        # execute.
        _jr3 = ctx.joint_state(ctx.rx, ctx.band_id, ctx.args)
        if _jr3 is not None and ctx.drp.now_w >= ctx.dr_state.get("jslew_log_next", 0.0):
            ctx.dr_state["jslew_log_next"] = ctx.drp.now_w + 30.0
            _cmp = []
            for (_ct, _p) in list(_jr3._idx):
                if _ct != ctx.drp.tag:
                    continue
                _lo = ctx.drp.clk_now + ctx.bsat.get(_p, ctx.drp.now_w)
                _jo = _jr3.predicted((_ct, _p))
                _dd = ((_jo - _lo + ctx.drp.mod / 2.0) % ctx.drp.mod) - ctx.drp.mod / 2.0
                _cmp.append((_p, _dd, _jr3.sigma((_ct, _p)) or 0.0))
            if _cmp:
                _cmp.sort(key=lambda x: -abs(x[1]))
                _log("SEED-OFFSET %s: joint-vs-legacy over %d sat(s), "
                     "median %+.3f chips | %s"
                     % (ctx.band_id, len(_cmp),
                        sorted(abs(c[1]) for c in _cmp)[len(_cmp) // 2],
                        " ".join("PRN%d %+.2f(s%.2f)" % c for c in _cmp[:6])))
        planned = []
        for (ctag, prn), v in sorted(ctx.drp.pd.items()):
            # SIGNAL CAPABILITY first (see --dr-min-prn / --signal-capability): a
            # satellite that does not broadcast this signal must never be seeded,
            # however visible and however well-predicted it is. The model will
            # happily hand us a code phase for a signal that isn't there.
            if prn < ctx.dr_min_prn:
                continue
            if ctx.capable is not None and prn not in ctx.capable:
                continue  # block does not carry this signal (GPS L1C/L5/L2C)
            # +0.5 deg hysteresis vs the drop mask: a sat riding the mask
            # otherwise flickers drop->reseed every few cycles
            # #83 P3-3b: a MODEL-PRIMARY PRN is dr-owned even while the search
            # detects it -- that is the whole point of the flip. (It can never
            # be in cp_held: the flip's ENTER discards the hold.)
            # dr_untrusted is the LEGACY a0 integrity (EMA clock, no b_sat,
            # ~1-chip bar); a flipped sat's seed is the JOINT model's and its
            # referee is MINNOV -- gating the slew on the legacy flag would
            # orphan the sat seedless (its detections bypass re-anchor).
            if (ctag != ctx.drp.tag or v["el"] < ctx.args.mask_deg + 0.5
                    or (prn in ctx.best and prn not in ctx.mp_flipped)
                    or prn in ctx.probe_set or prn in ctx.cp_held
                    or (prn in ctx.dr_untrusted
                        and prn not in ctx.mp_flipped)):  # model wrong for this sat
                continue
            if prn in ctx.seeds and prn not in ctx.dr_state["seeded"]:
                continue  # search-anchored coast: not ours to touch
            # TASK #30, LAYER 2: A LOCKED MODEL-PRIMARY SAT GETS A SLEWED REFRESH,
            # not a hold and not a repin. Both extremes are measured failures:
            #   HOLD  (search-fed logic on a chain with no search): the model's
            #         residual code rate is unmodeled, so the prompt drifts through
            #         the +-1 chip window in 3-11 min -- the rise-peak-fall deep_snr
            #         envelope, fleet disc railed +0.7, E >> L, lock lost, reseed.
            #   REPIN (8c7d176b3, reverted): re-anchoring from clk_now every 10 s
            #         injects the clock EMA's ~+-1 chip jitter as code-phase STEPS;
            #         median deep_snr fell 221 -> 17 and 21% of deep folds lost
            #         certification. Seed continuity beats seed freshness.
            # The slew keeps continuity AND tracks the model: converge the seed's
            # physical code phase toward the live model at <= _DR_SLEW_CAP chips
            # per cycle (0.025 chips/s at 2 s -- above every drift measured, an
            # order below the 0.25-chip DLL trims the GPS fold shrugs off), with
            # gain _DR_SLEW_K low-passing the clock EMA jitter the repin injected
            # raw. Search-FED chains keep the plain hold: their DLL owns code and
            # their search owns Doppler, and their digest is the control.
            # THE RE-PIN DECISION, and the one #58 makes wrong. `_held` adds a
            # fold-independent path: a satellite whose PROMPT is on the signal is
            # locked whatever the deep fold currently says about it. Without this
            # the fold's ~50%% certification flicker toggles lock state and
            # re-pins a satellite the despread never lost -- and the re-pin then
            # steps the phase reference the fold is trying to integrate across.
            # ⚠️ AND THE ONE METRIC THIS PROJECT ACTUALLY JUDGES LOCK ON.
            # The viewer's own q annotation says it: "Judge lock HERE, not on
            # sig/C-N0 -- the deep fold re-searches and will certify a satellite
            # whose prompt tap is on noise (#47). ~2.2 is the working lock bar."
            # This decision used neither q nor anything correlated with it:
            # sig_of is amp_snr (measured 0-1 on tracking satellites) upgraded by
            # a deep fold that certifies as little as 17% of polls, and
            # lock_prompt_hold wants 3x the FLEET MEDIAN prompt, which most of the
            # fleet cannot clear by construction (measured 0.76-1.61).
            # So both gates failed together, every cycle, and 24% of re-births
            # landed on satellites at q >= 2.2 -- five of them stepping the seed
            # 1.3-4.5 chips, i.e. clean outside the +-1 chip correlation triangle,
            # which destroys the very lock the gate exists to protect. PRN 13 was
            # thrown +2.42 chips at q 3.01 and +4.47 at q 2.24.
            _q_locked = (ctx.args.lock_q > 0.0
                         and ctx.hold.q.get(prn, 0.0) >= ctx.args.lock_q)
            _held = (_q_locked
                     or (ctx.args.lock_prompt_hold > 0.0
                         and ctx.hold.prev.get(prn, 0.0) >= ctx.args.lock_prompt_hold))
            if _held and ctx.sig_of(ctx.status.get(prn, {})) < ctx.args.lock_snr:
                _log_rl("hold-prompt",
                        "HOLD-BY-PROMPT: PRN %d held through a deep-fold dropout "
                        "(prompt %.1fx noise, sig %.1f < %.1f) -- no re-pin"
                        % (prn, ctx.hold.prev.get(prn, 0.0),
                           ctx.sig_of(ctx.status.get(prn, {})), ctx.args.lock_snr),
                        every_s=60.0)
            _slew = (prn in ctx.seeds
                     and (not ctx.detectors or prn in ctx.mp_flipped)
                     and prn in ctx.dr_state["seeded"]
                     and (ctx.sig_of(ctx.status.get(prn, {})) >= ctx.args.lock_snr or _held))
            if (not _slew and prn in ctx.seeds
                    and (ctx.sig_of(ctx.status.get(prn, {})) >= ctx.args.lock_snr or _held)):
                continue  # sub-threshold LOCK: the DLL owns the residual now
            if (not _slew and prn in ctx.dr_state["seeded"]
                    and ctx.drp.now_w - ctx.dr_state["pin"].get(prn, 0.0) < ctx.args.dr_repin_s):
                continue
            # doppler + rate from BRDC range-rate (NOT the TLE pred: the BDS
            # TLE<->PRN mapping mismaps some birds, and BRDC is the precision
            # source anyway); clock_bias still comes from the TLE-vs-measured
            # solve -- it's a receiver constant, common to both models.
            v2 = ctx.drp.pd2.get((ctag, prn))
            dop_geo = -v["range_rate_mps"] / ctx.dr_eph_mod.C_LIGHT * ctx.args.carrier_hz
            dop_seed = ctx.args.doppler_sign * dop_geo + ctx.cb.seed  # #105: seed bias, not the hint EMA
            # ⚠️ THE HALVED DRATE (found 2026-08-22, the per-sat ramp's root).
            # This line predates the task #52 pair centring: when pd2 sat at
            # now_w+4 the /4.0 was a correct forward difference, but pd2 moved
            # to now_w+2 (and pd0 to now_w-2) and this consumer kept /4.0 over
            # a 2 s span -- so every model-primary seed's doppler_rate_hz_s was
            # EXACTLY HALF the truth. The missing half leaks into held-vs-model
            # as (f_chip/f_c) * (drate/2) * T/2 per re-pin interval T: measured
            # ramp = 0.0549 chips/s per Hz/s of drate over Galileo (r=+0.68,
            # 29 chain-PRNs), implying T ~ 25 s -- the observed re-pin cadence.
            # drate < 0 for EVERY satellite (Doppler falls through a pass), so
            # the per-sat ramps all share a sign and pooled they masqueraded as
            # a constellation-common drift. The centred pair is what :4670 (the
            # dop_model seed slot) already uses; this is the same fix, same
            # first-cycle fallback.
            drate = 0.0
            v0 = (ctx.dr_state.get("pd0") or {}).get((ctag, prn))
            if v2 is not None and v0 is not None:
                drate = (ctx.args.doppler_sign
                         * (-(v2["range_rate_mps"] - v0["range_rate_mps"]) / 4.0)
                         / ctx.dr_eph_mod.C_LIGHT * ctx.args.carrier_hz)
            elif v2 is not None:
                drate = (ctx.args.doppler_sign
                         * (-(v2["range_rate_mps"] - v["range_rate_mps"]) / 2.0)
                         / ctx.dr_eph_mod.C_LIGHT * ctx.args.carrier_hz)
            # inverse of cp_loc above: physical cp -> sample-0 cp0 removes the
            # nominal advance AND the code-Doppler drift (the seed currency)
            # + b_sat (task #33): the per-sat slow bias the P1 fit measures --
            # iono/tropo/BRDC, the term holding seeds at a model that is right
            # about the orbit and wrong about the path. tau is sky-minus-replica,
            # so the replica moves BY tau to meet the sky: phys + b. Zero until
            # the fit has fed it (and exactly 0.0 in every transcript replay).
            # -- P2b CONSUMER 3: the clock+bias offset, ONE definition for both
            # the birth/re-pin phase (cp0, below) and the slew target (_model,
            # further down). They are the same physical quantity -- "what to add
            # to the pure model for this satellite" -- and must not be able to
            # disagree.
            #
            # ⚠️ WHICH CHAINS REACH WHICH PATH, because it decides what this
            # consumer can be tested on at all. The slew path requires `not
            # detectors`, so ONLY the model-primary chains (E5a/E5b/B2a/B2b)
            # slew; search-fed GPS takes cp0 at birth and re-pin and is then
            # skipped entirely once locked ("the DLL owns the residual now").
            # But the joint state is fed ONLY by search-anchored chains, so it
            # contains ONLY GPS satellites unless --joint-model-primary is on.
            # Scoped to the slew path alone this consumer could therefore NEVER
            # FIRE: the chains that slew have no satellites in the state, and
            # the chain with satellites in the state does not slew. Covering
            # cp0 as well is what gives it a live arm today, on GPS births.
            if ctx.drp.hold:
                continue      # clock is still a prime; see the withhold note
            _leg_off = ctx.drp.clk_now + ctx.bsat.get(prn, ctx.drp.now_w)
            _off = _leg_off
            _off_sigma = None      # set only when a JOINT offset is adopted
            _jr3 = ctx.joint_state(ctx.rx, ctx.band_id, ctx.args)
            _joff = (_jr3.predicted((ctx.drp.tag, prn))
                     if (_jr3 is not None and (ctx.drp.tag, prn) in _jr3._idx)
                     else None)
            if _joff is not None:
                # ⚠️ WRAP AT THE MODULUS THE TWO ACTUALLY SHARE, AND APPLY THE
                # RESULT AS A DELTA. clk_now is reduced mod CODE_LEN (10230 for
                # L5) and so is the joint clk, but the SEED lives mod _DR_MOD
                # (204600 with --dr-long-code -- 20 primary periods). Three
                # consequences, all learned the hard way today:
                #   * wrapping the difference at _DR_MOD does not fold out a
                #     CODE_LEN wrap, so when clk_now crosses zero the shadow
                #     read -10058 chips of "disagreement" on every satellite at
                #     once (observed 16:58, and the giveaway was that it was
                #     COMMON to all six -- a per-sat estimator error cannot be);
                #   * the plausibility bound would then REFUSE the joint offset
                #     for the whole wrap neighbourhood -- a gate failing
                #     periodically for a reason unrelated to estimator health;
                #   * and SUBSTITUTING _joff for _leg_off would displace the
                #     seed by a whole primary period, because the two are only
                #     equivalent mod CODE_LEN while the seed cares mod _DR_MOD.
                # So the joint state contributes only the SMALL correction it
                # actually measures, and the legacy path keeps ownership of
                # which long-code segment we are in.
                _d3 = ((_joff - _leg_off + ctx.code_len / 2.0) % ctx.code_len) - ctx.code_len / 2.0
                _ok3 = abs(_d3) <= ctx.args.joint_slew_max_chips
                _log_rl("jslew-%d" % prn,
                        "SEED-OFFSET PRN %d (%s): joint %+.3f vs legacy %+.3f "
                        "chips (diff %+.3f mod %.0f, sigma %.3f)%s"
                        % (prn, "slew" if _slew else "cp0", _joff, _leg_off,
                           _d3, ctx.code_len, _jr3.sigma((ctx.drp.tag, prn)) or 0.0,
                           "" if _ok3 else "  REFUSED (> %.1f chips)"
                           % ctx.args.joint_slew_max_chips),
                        every_s=60.0)
                if "slew" in ctx.joint_consume and _ok3:
                    _off = _leg_off + _d3
                    # ...and how well we know it, for the rate limit below.
                    _off_sigma = _jr3.sigma((ctx.drp.tag, prn))
            cp0 = ((ctx.cp_predicted(v, ctx.drp.t_fc_abs) + _off)
                   - ctx.drp.t_fc_abs * ctx.args.chip_rate_hz
                     * (1.0 + ctx.args.code_doppler_sign
                        * dop_seed / ctx.args.carrier_hz)) % ctx.drp.mod
            if ctx.args.dr_dry_run:
                planned.append("PRN %d el %.0f cp0 %.1f dop %+.0f rate %+.2f"
                               % (prn, v["el"], cp0, dop_seed, drate))
                continue
            if _slew:
                # Where the TRACKER's propagation puts this seed right now
                # (dr_seed_phys inverts the currency + extrapolation exactly --
                # round-trip 1e-4 chips, re-anchor continuity 0.0 even across a
                # Doppler change; see the selftest). The model side is the same
                # sum the birth cp0 uses, BEFORE back-referencing.
                #
                # ⚠️ EVERY TERM AT THE SAME EPOCH: t_h = h1/hps, the ROUNDED
                # hop's time -- never t_now_abs. The physical code phase runs at
                # ~52.4 chips per hop, so comparing a model evaluated at
                # t_now_abs against a held phase evaluated at h1/hps injects
                # (t_now_abs*hps - h1) * 52.4 = up to +-26 chips of PHANTOM
                # difference from the sub-hop rounding alone. Caught on the
                # first live cycle (2026-08-09 23:04): every fresh B2a seed
                # reported model-held = +23.335 chips two seconds after birth,
                # identical across PRNs -- rounding, wearing the clock's
                # common-mode costume. (Birth gets away with mixing t_now_abs
                # and a rounded ref_hop because there the SAME t builds the
                # subtraction, so the skew cancels against the tracker's
                # back-reference to first order. A cross-epoch DIFFERENCE has no
                # such cancellation.)
                h1 = int(round(ctx.drp.t_fc_abs * ctx.args.hops_per_sec))
                t_h = h1 / ctx.args.hops_per_sec
                _held = dr_seed_phys(
                    ctx.seeds[prn], h1, ctx.args.hops_per_sec, ctx.args.chip_rate_hz,
                    ctx.args.carrier_hz, ctx.args.code_doppler_sign, ctx.drp.mod)
                # The clock+bias offset is _off, computed once above so the birth
                # phase and the slew target cannot disagree. b_sat is "how wrong
                # the pure model is for THIS satellite", which is why this is the
                # consumer aimed at the ~600 s plant oscillation (slew-to-model
                # fighting trim-to-sky, with the model per-sat +-1-6 chips out).
                _model = (ctx.cp_predicted(v, t_h) + _off) % ctx.drp.mod
                _dcp = ((_model - _held + ctx.drp.mod / 2.0) % ctx.drp.mod
                        ) - ctx.drp.mod / 2.0
                # ── THE RATE LIMIT, AND WHY IT IS THE WHOLE STORY ──
                # _DR_SLEW_CAP is 0.05 chips per event and 47% of steps sat
                # exactly on it: satellites 5-8 chips from their model were
                # being corrected at 0.05 a go, every 10-30 s -- of order an
                # HOUR to close, against a ~600 s oscillation. The seed can
                # never reach the model within a cycle, so the plant is a
                # rate-limited integrator chasing a moving multi-chip error: a
                # limit cycle almost by construction, and a better account of
                # the ~600 s oscillation than "slew fights trim".
                #
                # It also made P2b's slew consumer UNOBSERVABLE. For a
                # satellite already railed at the cap, moving the target by
                # +3-8 chips leaves the step at +-0.05 in the same direction --
                # identical seed behaviour before and after, which is exactly
                # what the viewer showed. Measured 2026-08-11 16:52, and
                # visible in the FIRST log line read that morning (PRN 35:
                # model-held -10.571 chips, step -0.050). Dividing the two
                # numbers in any log line would have found it.
                #
                # ACQUISITION AUTHORITY, TRACKING RESTRAINT. A seed that is
                # far from its target needs the authority to pull in; one that
                # has arrived must not be yanked around by noise. So the
                # ceiling slides with DISTANCE, and drops back to the flat cap
                # once the seed is within dr_slew_near_chips.
                #
                # KEYED ON DISTANCE, NOT ON TRACK AGE, though "new satellites
                # slew hard, established ones tighten" is the behaviour either
                # would give -- because a new seed IS the far one. Distance
                # re-arms and age does not, and that difference is the whole
                # point here: the ~600 s oscillation happens on ESTABLISHED
                # tracks that have drifted off. An age schedule would tighten
                # exactly those satellites and then deny them the correction
                # they need, locking in the limit cycle it was meant to break.
                #
                # SIGMA IS A TRUST GATE, NOT A MULTIPLIER. The first cut scaled
                # the cap as n x sigma, borrowed from the escape guard -- but
                # there sigma bounds the size of a CLAIM, while here it is the
                # precision of a KNOWN target, so proportional scaling made a
                # well-measured offset move SLOWER. Inverted. What sigma
                # actually decides is whether to trust the target enough to
                # move fast toward it at all; past dr_slew_trust_sigma we keep
                # the crawl. An offset with no sigma (no joint state, or
                # refused) is untrusted by construction.
                _cap = ctx.drp.slew_cap
                if (ctx.args.dr_slew_cap_acq > ctx.drp.slew_cap
                        and _off_sigma is not None
                        and math.isfinite(_off_sigma)
                        and _off_sigma <= ctx.args.dr_slew_trust_sigma
                        and abs(_dcp) > ctx.args.dr_slew_near_chips):
                    _cap = ctx.args.dr_slew_cap_acq
                _step = max(-_cap, min(_cap, ctx.drp.slew_k * _dcp))
                # Re-anchor at h1 with the FRESH model doppler/rate (kills the
                # dt^2 linearization error a held seed accumulates) but at the
                # HELD phase plus the bounded step -- never at the model's own
                # phase, which carries the clock EMA jitter raw. NOT popping
                # dll_trim: same trajectory, later epoch; the trim's residual is
                # still valid (the lesson of the reverted repin).
                ctx.seeds[prn] = Seed.born(
                    "dr_slew", epoch=h1,
                    doppler_hz=dop_seed,
                    code_phase_chips=dr_cp0(
                        _held + _step, t_h, dop_seed,
                        ctx.args.chip_rate_hz, ctx.args.carrier_hz,
                        ctx.args.code_doppler_sign, ctx.drp.mod),
                    code_phase_rate=cp_rate_from_code_bias(
                        dop_seed, ctx.drp.la, ctx.args.hops_per_sec,
                        ctx.args.chip_rate_hz, ctx.args.carrier_hz),
                    ref_hop=h1, doppler_rate_hz_s=drate)
                # #45 STEP 6: ship the PHASE as well. propagate_seed prefers it
                # and it carries no sample-0 lever, so a later dop edit cannot
                # desynchronise the pair (#42's writer, #44's coast). Both are
                # emitted so a tracker that ignores the field is unaffected.
                if ctx.args.seed_phase_transport:
                    ctx.seeds[prn].put(
                        "phase_xport", epoch=h1,
                        code_phase_at_ref_chips=seed_phase_at_ref(
                            _held + _step, dop_seed, ctx.args.chip_rate_hz,
                            ctx.args.hops_per_sec, ctx.args.carrier_hz,
                            ctx.args.code_doppler_sign, ctx.drp.mod,
                            ctx.args.search_fft_len or None))
                ctx.dr_state["pin"][prn] = ctx.drp.now_w
                _log_rl("drslew-%d" % prn,
                        "dead-reckon SLEW PRN %d: model-held %+.3f chips, "
                        "step %+.3f (cap %.2f), dop %+.0f rate %+.2f"
                        % (prn, _dcp, _step, _cap, dop_seed, drate),
                        every_s=120.0)
                continue
            if prn not in ctx.seeds:
                _log("dead-reckon SEED PRN %d (elev %.0f, cp0 %.1f, dop %+.0f,"
                     " rate %+.2f)" % (prn, v["el"], cp0, dop_seed, drate))
            ctx.dls.trim.pop(prn, None)  # any old trim served the OLD anchor
            ctx.dls.last.pop(prn, None)
            _rh_birth = int(round(ctx.drp.t_fc_abs * ctx.args.hops_per_sec))
            # ── BIRTH-STEP DECOMPOSITION (2026-08-22) ────────────────────────
            # Measured overnight: when several satellites are re-born in the SAME
            # cycle they all step by the SAME amount -- E5 +142.84, E13 +142.76,
            # E16 +142.78, E26 +142.80, E31 +142.87 chips, agreeing to 0.11 chips
            # (0.08%) with completely unrelated ddop (-0.76..-6.96 Hz) and seed
            # ages. Nothing per-satellite can do that: it is ONE SHARED CONSTANT
            # entering every seed at once, and its size is the receiver clock.
            # Three candidate explanations were falsified BEFORE this line was
            # written, which is why it logs terms rather than a verdict:
            #   * a rate rewrite under a stale anchor -> implied rate change
            #     0.44 chips/s vs the 0.001-0.02 the filter carries, and
            #     r(|step|, seed age) = -0.11 when the mechanism needs
            #     proportionality;
            #   * a Doppler edit through the cp-currency lever -> r = +0.08,
            #     step-equivalent 0.055 Hz against an actual ddop of 4.08 Hz;
            #   * a feedback loop -> the steps are discrete and SHARED, and a
            #     loop cannot synchronise independent satellites to 0.08%.
            # So: print where the OLD seed said the signal was, where the NEW one
            # puts it, and every term that differs between the two branches. The
            # slew branch anchors on `_held` (the TRACKER's own propagation); this
            # branch anchors on model + `_off`. If the clock term is what flips,
            # `step` lands on `_off` (or on `_d3`, the joint-vs-legacy correction
            # that comes and goes with ADOPTED/REFUSED) and the line says so
            # outright instead of leaving it to be inferred from a magnitude.
            _pv = ctx.seeds.get(prn)
            if _pv is not None and "ref_hop" in _pv:
                try:
                    _oldphys = dr_seed_phys(_pv, _rh_birth, ctx.args.hops_per_sec,
                                            ctx.args.chip_rate_hz, ctx.args.carrier_hz,
                                            ctx.args.code_doppler_sign, ctx.drp.mod)
                    _newphys = (ctx.cp_predicted(v, ctx.drp.t_fc_abs) + _off) % ctx.drp.mod
                    _bstep = ((_newphys - _oldphys + ctx.drp.mod / 2.0) % ctx.drp.mod
                              ) - ctx.drp.mod / 2.0
                    # ── #92 THE HANDOVER (--fleet-trim-rebase-adjust) ──
                    # The seed is about to move by _bstep while the C++
                    # standing trim carries the SAME chips: post the
                    # compensating -_bstep to the gather in the SAME cycle
                    # so the tap (seed + trim) never leaves the sky (E3's
                    # ~25-min sawtooth). The bound, the transport and the
                    # cumulative-delta bookkeeping are in
                    # gnss_broker/handover.py. Gated on _ft_armed_last: a
                    # PRN the fleet loop is not actuating has no standing
                    # trim to hand over, and posting for seeding churn
                    # buries the one refusal that would MEAN something.
                    ctx.handover.offer(prn, _bstep, prn in ctx.dls.armed_last,
                                    ctx.telem_chain, ctx.args.fleet_trim_url,
                                    _post, _log)
                    # D3 reads this to tell a REBASE-coincident wipe from a bare
                    # one -- the #92 P2 metric is the rebase-coincident rate on an
                    # armed chain vs its unarmed band sibling, and without the
                    # stamp the two wipe classes (birth-step vs slew-transfer,
                    # 2026-08-26) superpose exactly as E3 did.
                    ctx.birth_steps[prn] = ctx.t0
                    _log_rl("birthstep-%d" % prn,
                            "BIRTH-STEP PRN %d: old_phys %+.3f -> new_phys %+.3f"
                            "  step %+.3f chips | off %+.3f = leg %+.3f"
                            " (clk %+.3f + b %+.3f) %s |"
                            " age %.1f s ddop %+.3f Hz"
                            " | WHY-BIRTH: sig_of %.2f vs lock_snr %.1f,"
                            " hold_prev %.2f vs %.1f -> held %s;"
                            " in_seeds %s in_seeded %s | prev [%s]"
                            % (prn, _oldphys, _newphys, _bstep, _off, _leg_off,
                               ctx.drp.clk_now, ctx.bsat.get(prn, ctx.drp.now_w),
                               ("+ d3 %+.3f [joint %s]"
                                % (_d3, "ADOPTED" if _ok3 else "REFUSED"))
                               if _joff is not None else "[joint absent]",
                               (_rh_birth - int(_pv["ref_hop"])) / ctx.args.hops_per_sec,
                               dop_seed - float(_pv.get("doppler_hz", dop_seed)),
                               # ⚠️ WHY WAS THE SLEW BRANCH NOT TAKEN? Reaching
                               # this line means `_slew` was False for a satellite
                               # the tracker may well be holding -- and on
                               # 2026-08-22 that cost E13 a 3.75-chip snap off the
                               # correlation triangle and 28 minutes of q < 1.
                               # sig_of() is amp_snr, or max(amp, deep) ONLY when
                               # the deep fold certifies (coherence_s > 0) -- and
                               # measured live, amp_snr sits at 0-1 against a
                               # lock_snr of 3.0 while the fold certifies as
                               # little as 17% of polls. So print every input to
                               # the decision rather than inferring which failed.
                               ctx.sig_of(ctx.status.get(prn, {})), ctx.args.lock_snr,
                               ctx.hold.prev.get(prn, 0.0), ctx.args.lock_prompt_hold,
                               _held, prn in ctx.seeds,
                               prn in ctx.dr_state["seeded"],
                               _pv.owners() if hasattr(_pv, "owners") else "?"),
                            every_s=20.0)
                except Exception as _e:      # diagnostics never break seeding
                    _log_rl("birthstep-err", "BIRTH-STEP unavailable: %s" % _e,
                            every_s=300.0)
            ctx.seeds[prn] = Seed.born(
                "dr_birth", epoch=_rh_birth,
                doppler_hz=dop_seed, code_phase_chips=cp0,
                code_phase_rate=cp_rate_from_code_bias(
                    dop_seed, ctx.drp.la, ctx.args.hops_per_sec,
                    ctx.args.chip_rate_hz, ctx.args.carrier_hz),
                ref_hop=_rh_birth,
                doppler_rate_hz_s=drate)
            # #45 STEP 6, birth/re-pin arm. cp0 was just built FROM this phase
            # (cp_predicted + _off at t_now_abs), so shipping it costs nothing
            # and removes the round trip the tracker would otherwise redo.
            if ctx.args.seed_phase_transport:
                ctx.seeds[prn].put(
                    "phase_xport", epoch=_rh_birth,
                    code_phase_at_ref_chips=seed_phase_at_ref(
                        (ctx.cp_predicted(v, ctx.drp.t_fc_abs) + _off) % ctx.drp.mod,
                        dop_seed, ctx.args.chip_rate_hz, ctx.args.hops_per_sec,
                        ctx.args.carrier_hz, ctx.args.code_doppler_sign, ctx.drp.mod,
                        ctx.args.search_fft_len or None))
            ctx.dr_state["seeded"].add(prn)
            ctx.dr_state["pin"][prn] = ctx.drp.now_w
        if planned:
            _log("dead-reckon DRY RUN, would seed: %s" % "; ".join(planned))
        # model-owned sats drop on the BRDC elevation (they're exempt from
        # the TLE horizon drop -- see the coast loop), or on capability
        for prn in list(ctx.dr_state["seeded"]):
            v = ctx.drp.pd.get((ctx.drp.tag, prn))
            if prn < ctx.dr_min_prn or (ctx.capable is not None and prn not in ctx.capable):
                _log("dead-reckon drop PRN %d (does not broadcast this signal)" % prn)
                ctx.seeds.pop(prn, None)
                ctx.hold.low_hits.pop(prn, None)
            elif v is None or v["el"] < ctx.args.mask_deg:
                _log("dead-reckon drop PRN %d (set below BRDC horizon)" % prn)
                ctx.seeds.pop(prn, None)
                ctx.hold.low_hits.pop(prn, None)


def stage_dead_reckon(ctx):
    """3e: DEAD-RECKONED SEEDING -- the model-primary spine, and the shell of its pipeline.
        
    Refreshes the ephemeris, builds the per-satellite predictions and code offsets, then runs the
    five sub-stages in order: joint shadow -> clock solve -> clock quality -> clock adopt -> seed.
    Its working set lives on `_drp`, so each arrow of that pipeline is a named field rather than
    an implicit shared local.
        
    On the searchless chains (Galileo, BeiDou) this IS the acquisition path: it can seed a
    satellite that has never been detected, from the broadcast ephemeris and the receiver clock
    alone.
        
    ⚠️ `t_now_abs` IS None, NEVER 0.0, until the first ephemeris refresh. A missing axis time is
    UNKNOWN, and a confident wrong timestamp is worse than a skipped measurement -- as 0.0 it
    killed chain threads through the #85 stash."""
    if (ctx.dr_state is not None and ctx.args.almanac and ctx.pred and ctx.utc0_sample0
            and _now() >= ctx.dr_state["next"]):
        ctx.drp.now_w = _now()
        ctx.dr_state["next"] = ctx.drp.now_w + ctx.args.dr_refresh_s
        # ── DO NOT SEED ON A GUESSED CLOCK (2026-08-22) ──────────────────────────────
        # Measured with the BIRTH-STEP diagnostic: every satellite born before the clock
        # bootstraps carries NO clock, and the first re-birth after BOOTSTRAP adds it --
        # so the whole fleet steps by one clock at once. 40 events, all within 57 s of
        # broker start and none after 60 s, with step/off = 0.9958: the step IS `_off`,
        # to 0.4%. Five satellites agreeing to 0.08% with unrelated Doppler is what gave
        # it away -- nothing per-satellite can do that.
        # The fix is to wait, because the wait is SHORT and the clock is imminent: the
        # solve snaps on its first median (gps_l5 bootstraps ~1 s after start) and the
        # model-primary chains adopt it cross-band within a few seconds of that.
        # ⚠️ BUT IT MUST NOT BE AN INDEFINITE WAIT, and that is the whole design of the
        # prime. --dr-clock-chips exists precisely so a DETECTOR-LESS chain can seed at
        # all: `offs` is always empty there, so the solve never runs and no bootstrap is
        # ever coming. Refusing to seed until a measurement arrives would leave exactly
        # those chains dark forever -- trading a 150-chip startup step for a dead chain.
        # So: withhold while the clock is admittedly a guess, but only for
        # --dr-clock-wait-s, then seed on the prime and SAY SO. A guard that can deadlock
        # the instrument is worse than the artefact it removes.
        ctx.drp.hold = False
        if ctx.dr_state.get("clk_primed") and ctx.args.dr_clock_wait_s > 0.0:
            _waited = ctx.drp.now_w - ctx.dr_state.get("clk_t", ctx.drp.now_w)
            if _waited < ctx.args.dr_clock_wait_s:
                _log_rl("clkwait", "dead-reckon: WITHHOLDING seeds -- the clock is still "
                                   "the %.2f-chip PRIME, not a measurement (%.0f of %.0f s"
                                   " waited). Seeding now would anchor every cp0 without "
                                   "a clock and step the whole fleet by ~%.0f chips at the"
                                   " first re-birth after BOOTSTRAP."
                        % (ctx.dr_state.get("clk") or 0.0, _waited, ctx.args.dr_clock_wait_s,
                           abs(ctx.dr_state.get("clk") or 0.0) or 150.0),
                        every_s=10.0)
                ctx.dr_state["next"] = ctx.drp.now_w + min(2.0, ctx.args.dr_refresh_s)
                ctx.drp.hold = True
            # ⚠️ elif, NOT a second if. Written as a bare `if` this fired in the SAME
            # MILLISECOND as the withhold above -- "seeding on the PRIME after waiting
            # 30 s" logged 0 s in, because reaching the warning never depended on the
            # wait having expired. An alarm that cannot distinguish "waiting" from
            # "gave up waiting" is worse than none: it would have taught us to ignore it.
            elif not ctx.dr_state.get("clk_wait_warned"):
                ctx.dr_state["clk_wait_warned"] = True
                _log("dead-reckon: ⚠️ seeding on the %.2f-chip PRIME after waiting %.0f s "
                     "-- no clock measurement arrived. Expected on a DETECTOR-LESS chain "
                     "with no sibling to adopt from; on a chain that HAS detectors it "
                     "means the solve is not running, and every seed below is anchored on "
                     "a guess." % (ctx.dr_state.get("clk") or 0.0,
                                   ctx.drp.now_w - ctx.dr_state.get("clk_t", ctx.drp.now_w)))
        # THE DEAD-RECKON MODEL MUST WORK AT THE CODE THE TRACKER ACTUALLY DESPREADS.
        # This was CODE_LEN / chip_rate -- ONE PRIMARY PERIOD -- so t0m and every predicted
        # phase were reduced mod 1 ms and the secondary segment was discarded before a seed
        # was ever built. A dead-reckoned PRN therefore always landed in segment 0: right
        # 1 time in LC_SEG.
        #
        # GPS survives that because the BLIND SEARCH re-seeds it with a measured `nh`, so
        # the model's missing segment is overwritten before it matters. E5a/B2a have no
        # blind search at all (per-PRN secondaries, CHORD_MULTIBAND.md section 5) -- the
        # dead-reckon seed IS the answer -- so for them this modulo is the whole bug:
        # measured 2026-08-08, every E5a seed had code_phase_chips < 10230 against a
        # 1,023,000-chip code, i.e. 1-in-100 of the right CS period, i.e. noise.
        #
        # Reducing mod the LONG period instead keeps the numbers small enough for double
        # precision (the reason the reduction exists) while carrying the segment. Placing
        # it needs absolute time to half a primary period, 0.5 ms; the F-engine anchor is
        # GPS-disciplined to microseconds and BRDC range/clock are nanosecond-class, so
        # there are three orders of margin.
        ctx.drp.t_code = (ctx.lc_seg * ctx.code_len) / ctx.args.chip_rate_hz if ctx.args.dr_long_code \
                 else ctx.code_len / ctx.args.chip_rate_hz
        # The seed is reduced at the SAME length the prediction was: one constant, used
        # twice, so they cannot drift apart.
        ctx.drp.mod = (ctx.lc_seg * ctx.code_len) if ctx.args.dr_long_code else ctx.code_len
        # Layer-2 slew constants (task #30). CAP: 0.05 chips per 2 s cycle = 0.025
        # chips/s of correction authority -- above the 0.003-0.02 chips/s drift band
        # measured on sky, an order below the 0.25-chip DLL trims a fold tolerates, and
        # 20x below the ~1-chip clock-EMA jitter the reverted 10 s repin injected whole.
        # K: first-order low-pass; with the cap it bounds, it is deliberately unfussy.
        # The per-event slew rate limit. 0.05 chips was hardcoded here and is the
        # single most consequential constant in the seeding path (see the call site):
        # at 0.05 a go, every 10-30 s, a satellite 5-8 chips from its model takes of
        # order an HOUR to arrive. A flag because the decisive experiment is
        # --dr-slew-cap 0 -- slewing OFF, the seed left where the loop put it -- which
        # is the one arm nobody has run and the one that separates "the seed is wrong"
        # from "the model is wrong and the slew drags good seeds onto it".
        ctx.drp.slew_cap = ctx.args.dr_slew_cap
        ctx.drp.slew_k = 0.25
        if ctx.dr_state["eph"] is None or ctx.drp.now_w - ctx.dr_state["eph_t"] > 7200:
            try:
                ctx.dr_state["eph"] = ctx.dr_eph_mod.parse_rinex_nav(ctx.dr_eph_mod.fetch_brdc())
                ctx.dr_state["eph_t"] = ctx.drp.now_w
                ctx.dr_state["t0m"] = ctx.dr_eph_mod.gpst_of_utc(ctx.utc0_sample0) % ctx.drp.t_code
                _log("dead-reckon: BRDC loaded (%d sats)" % len(ctx.dr_state["eph"]))
                # MEASURED CODE BIASES, refreshed on the ephemeris cadence (A0b, part 2).
                # Daily product, ~5 days of latency, biases stable over weeks -- so the
                # refresh rate is irrelevant and the fetch is cached. Optional by design:
                # no token or no network -> dcb stays None and group_delay_s falls back to
                # the broadcast term, which is what every run before 2026-08-23 did.
                if ctx.args.dcb_bias:
                    try:
                        import gnss_dcb as _dcbm
                        _p = _dcbm.fetch_dcb()
                        _t = _dcbm.parse_dcb(_p)
                        ctx.dr_state["dcb"] = _t or None
                        if _t:
                            _n = sum(1 for k in _t if k[0] == ctx.args.dr_constellation)
                            _log("dead-reckon: DCB loaded (%s; %d sats this "
                                 "constellation) -- measured code biases override the "
                                 "broadcast TGD/BGD per satellite"
                                 % (os.path.basename(_p or "?"), _n))
                        else:
                            _log("dead-reckon: no DCB product (no token/network) -- "
                                 "falling back to the broadcast group delay")
                    except Exception as _de:
                        ctx.dr_state["dcb"] = None
                        _log("dead-reckon: DCB load failed (%s); broadcast term only"
                             % _de)
            except Exception as e:
                _log("dead-reckon: BRDC unavailable (%s); retry in 10 min" % e)
                ctx.dr_state["eph_t"] = ctx.drp.now_w - 7200 + 600
        # DECODED-EPH FALLBACK: keep predicting off our own decode when the network BRDC is
        # gone (or always, under --decoded-eph-fallback-force, the live A/B harness).
        _use_decoded = ctx.decfb is not None and (
            ctx.args.decoded_eph_fallback_force
            or (ctx.args.decoded_eph_fallback and not ctx.dr_state["eph"]))
        if ctx.dr_state["eph"] or _use_decoded:
            ctx.drp.tag = ctx.args.dr_constellation
            ctx.drp.t_now_abs = ctx.drp.now_w - ctx.utc0_sample0
            # ── #83 THE AXIS FIX (see --dr-fengine-axis) ── "now" from the F-engine
            # hop counter: newest telemetry hop at its fetch instant, plus the wall
            # ELAPSED since -- so NTP's absolute offset never enters and its slew
            # contributes only (drift x sub-cycle seconds) = nanoseconds. Every label
            # this block stamps (h1, _rh_birth) and every phase it evaluates then
            # lives on the axis the tracker actually runs. The static difference
            # between this axis and the old wall one lands in the solved receiver
            # clock exactly as the old anchor error did (common-mode), so nothing
            # steps at the flip of the flag except the labels' MEANING.
            if ctx.args.dr_fengine_axis and ctx.fe_axis[0] is not None:
                # FILTERED offset, not the freshest sample (2026-08-23): the raw form
                # _feh/hps + (now_w - _few) re-samples the pipeline lag every poll and
                # the jitter lands in every rebirth as lag x range_rate -- see fe_off.
                if ctx.fe_off[0] is not None:
                    ctx.drp.t_now_abs = ctx.fe_off[0] + ctx.drp.now_w
                else:
                    _feh, _few = ctx.fe_axis[0]
                    ctx.drp.t_now_abs = _feh / ctx.args.hops_per_sec + (ctx.drp.now_w - _few)
            # ── THE FORECAST EPOCH (see --dr-forecast-lead-s) ──
            # A seed is a FORECAST WITH A LABEL: (ref_hop, phase, doppler, rate). Its
            # correctness is defined entirely on the hop axis, so producing one needs no
            # notion of "now" at all -- and asking for one is what dragged the wall clock,
            # the NTP offset and the telemetry lag into the seed path in the first place.
            # Instead choose the hop we are forecasting TO: H = newest telemetry hop +
            # lead. Then the pipeline lag never enters the arithmetic; it only sets how
            # large the lead must be (it must exceed transport + install, or the seed
            # lands after the epoch it describes), and the lag's jitter merely eats
            # margin. H is an exact integer hop, so int(round(t_fc_abs*hps)) == H and the
            # label carries no rounding of its own.
            # ⚠️ ONE EPOCH, USED CONSISTENTLY. The phase and the label must be built from
            # the SAME t or the tracker's back-reference no longer cancels (see the note
            # at h1 below). Filter/bookkeeping sites deliberately keep t_now_abs: ages,
            # update_rrate and the joint cycle are measurements at NOW, and shifting them
            # into the future would inflate every age by the lead.
            ctx.drp.t_fc_abs = ctx.drp.t_now_abs
            if ctx.args.dr_forecast_lead_s > 0.0 and ctx.fe_axis[0] is not None:
                # forecast from the FILTERED now (t_now_abs above), not the raw newest
                # hop -- same jitter, same fix. H stays an exact integer hop so the
                # label still carries no rounding of its own.
                _fch = int(round((ctx.drp.t_now_abs + ctx.args.dr_forecast_lead_s)
                                 * ctx.args.hops_per_sec))
                ctx.drp.t_fc_abs = _fch / ctx.args.hops_per_sec
            # ── THE EPHEMERIS EPOCH STAYS ON WALL TIME, AND HERE IS THE MEASUREMENT ──
            # Orbit evaluation (predict_all / predict_from_decoders below) is the ONE
            # consumer in this file that needs absolute UTC to be TRUE rather than
            # merely self-consistent: everything else either differences two wall reads
            # (immune to a slewing offset) or uses wall on both sides of a round trip
            # (it cancels). A wrong epoch here is a wrong satellite POSITION and no
            # round trip removes it, so moving it onto the F-engine axis looked
            # obviously right -- utc0_sample0 is GPS-disciplined, cf06's wall carries
            # ~1.45 ms of NTP error (chrony root dispersion 1.489 ms, measured).
            #
            # ⚠️ TRIED 2026-08-17, MEASURED WORSE BY ~65x, REVERTED. The F-engine "now"
            # we can OBSERVE is not the F-engine's now: it is the newest telemetry
            # pow_hop, which is behind the sky by the gather/merge/serve latency.
            # Measured over the 520 cycles of broker_onsky_e5a: (utc0 + hop/hps) - wall
            # = -99.6 ms MEDIAN with a 59 ms interquartile spread. Feeding that to the
            # ephemeris trades 1.45 ms of NTP error for ~100 ms of pipeline lag -- 80 m,
            # 2.7 chips at L5 -- and the armed replay showed exactly that: seed cp0
            # moved 0.6-2.4 chips per PRN, scaling with each satellite's range rate.
            # The axis is drift-free, which is why it is right for LABELS (a label and
            # its phase are stamped at the same t and the tracker propagates from the
            # label -- staleness cancels); it is wrong for an EPOCH, where staleness is
            # the whole error. Fixing this properly needs lag compensation, i.e. an
            # estimate of the pipeline delay -- not a substitution.
            #
            # What survives is the reason the question was asked: a wall STEP is bounded
            # by nothing, and would move every model seed on all five chains at once,
            # fleet-common and unattributable. The two axes are compared below as a
            # TRIPWIRE (log only, never a control input) -- it cannot see a step smaller
            # than the lag jitter, and does not pretend to.
            if ctx.fe_axis[0] is not None and ctx.utc0_sample0:
                # unpacked HERE, not borrowed from the axis-fix block above: that block
                # runs only under --dr-fengine-axis and this tripwire must work either way
                _axh, _axw = ctx.fe_axis[0]
                _dax = (ctx.utc0_sample0 + _axh / ctx.args.hops_per_sec) - _axw
                _dprev = ctx.dr_state.get("ax_off")
                if _dprev is not None and abs(_dax - _dprev) > ctx.args.clock_step_guard_s:
                    _log("*** WALL-vs-F-ENGINE OFFSET JUMPED %+.3f s (%.3f -> %.3f, "
                         "bar %.3f s). TWO CAUSES, and this line cannot tell them "
                         "apart: (a) cf06's wall clock stepped -- the F-engine axis is "
                         "GPS-disciplined, so every model-evaluated seed on every chain "
                         "just moved with it, ~%.0f chips at 800 m/s of range rate, "
                         "fleet-common; (b) the telemetry lag jumped -- the observable "
                         "F-engine 'now' is the newest pow_hop and trails the sky by "
                         "the gather/serve latency (-99.6 ms median, 59 ms IQR "
                         "measured), so anything near that scale is lag, not the clock. "
                         "Discriminate with chronyc tracking. Nothing here corrects "
                         "either -- this exists so the next hour is attributable."
                         % (_dax - _dprev, _dprev, _dax, ctx.args.clock_step_guard_s,
                            abs(_dax - _dprev) * 800.0 / 29.3))
                ctx.dr_state["ax_off"] = _dax
            ctx.drp.la = (ctx.args.code_bias_force * 1e-6 if ctx.args.code_bias_force is not None
                  else (ctx.cb.code_ema if ctx.cb.code_ema is not None else None))
            # TASK #30, LAYER 1: A DETECTOR-LESS CHAIN CANNOT MEASURE (l-a) -- BORROW THE
            # BAND SIBLING'S. code_bias_ema fills only from this chain's own cp-fit pool,
            # which needs detections, so on E5a/B2a it stayed None and every dead-reckon
            # seed shipped code_phase_rate = 0.0 (visible in SEEDDBG). The receiver code
            # clock is PER BAND, not per chain -- gps_l5 measures it continuously on the
            # same 1176.45 MHz and has contributed it to the Receiver since M3, but
            # rx.code_bias() had NO consumer until now. The unmodeled rate this leaves is
            # ~0.003-0.01 chips/s, which walks the prompt through the +-1 chip correlation
            # window in 3-11 minutes: measured on sky as E5a's rise-peak-fall deep_snr
            # envelope with the fleet disc railed at +0.7 and E >> L. Rates are SMOOTH --
            # this is the correction that cannot inject a step, unlike the reverted repin.
            if ctx.drp.la is None:
                _sh_cb = ctx.rx.code_bias(ctx.band_id, exclude=ctx.chain_id, t_now=ctx.drp.now_w)
                _sh_band = ctx.band_id
                # LAYER 2 (task #34): FALL BACK ACROSS THE BAND. The same-band lookup above
                # is a bootstrap trap for a band with no chain that can solve its own clock,
                # and 1207.14 MHz was exactly that: gal_e5b and bds_b2b adopted (l-a) ZERO
                # times while their 1176.45 siblings adopted it 102 and 107 times, so all 19
                # of their PRNs shipped code_phase_rate = 0.0 and walked open-loop.
                #
                # This is SOUND, not a compromise, and the reason is that (l-a) is a
                # FRACTIONAL FREQUENCY: tau_band -- the per-carrier group delay that makes
                # the code PHASE band-specific -- is a CONSTANT, and a constant contributes
                # nothing to a rate. See Receiver.code_bias_any_band. The code PHASE keeps
                # its per-band scoping (dr_clock is untouched); adopting a phase across
                # carriers would inject the very tau_band offset the second band exists to
                # measure.
                #
                # Logged distinctly from the same-band case: "cross-band" in the message is
                # how you tell, from the log alone, that a chain is running on a borrowed
                # rate rather than one measured in its own band.
                if _sh_cb is None:
                    _sh_cb = ctx.rx.code_bias_any_band(exclude=ctx.chain_id, t_now=ctx.drp.now_w)
                    _sh_band = "cross-band"
                if _sh_cb is not None:
                    ctx.drp.la = float(_sh_cb.value)
                    _log_rl("la-adopt",
                            "dead-reckon: (l-a) %+.4f ppm ADOPTED from in-process chain "
                            "'%s' (%s %s) -> seeds carry the code-clock rate"
                            % (ctx.drp.la * 1e6, _sh_cb.src,
                               "same band" if _sh_band == ctx.band_id else "CROSS-BAND, rate only;",
                               _sh_band if _sh_band == ctx.band_id else "phase stays per-band"),
                            every_s=300.0)
                else:
                    ctx.drp.la = 0.0
            # clock drift (chips/s): EMPIRICAL from consecutive raw solves (EMA'd
            # below), falling back to the f_chip*(l-a) model until measured -- the
            # modeled value left a persistent EMA lag (~0.6 chips at first deploy),
            # outside the BOC DLL capture range.
            ctx.drp.drift = ctx.dr_state.get("drift")
            if ctx.drp.drift is None:
                ctx.drp.drift = ctx.args.chip_rate_hz * ctx.drp.la
            try:
                # two epochs, 4 s apart: range_rate difference -> doppler RATE (the
                # TLE almanac's rate is unused here -- BRDC governs model-owned sats)
                if _use_decoded:
                    _ents = ctx.decoded_entries(ctx.drp.now_w)
                    ctx.drp.pd = ctx.decfb.predict_from_decoders(
                        _ents, ctx.args.lat, ctx.args.lon, ctx.args.alt,
                        datetime.fromtimestamp(ctx.drp.now_w, tz=timezone.utc), mask_deg=-90.0)
                    # CENTRED PAIR (task #52): +/-2 s about now_w, not [now, now+4].
                    ctx.drp.pd2 = ctx.decfb.predict_from_decoders(
                        _ents, ctx.args.lat, ctx.args.lon, ctx.args.alt,
                        datetime.fromtimestamp(ctx.drp.now_w + 2.0, tz=timezone.utc),
                        mask_deg=-90.0)
                    pd0 = ctx.decfb.predict_from_decoders(
                        _ents, ctx.args.lat, ctx.args.lon, ctx.args.alt,
                        datetime.fromtimestamp(ctx.drp.now_w - 2.0, tz=timezone.utc),
                        mask_deg=-90.0)
                    if _now() - ctx.decfb_log_t[0] > 60.0:
                        ctx.decfb_log_t[0] = _now()
                        ab = ""
                        if ctx.args.decoded_eph_fallback_force and ctx.dr_state["eph"]:
                            # A/B: compare decoded vs BRDC predict, worst common sat.
                            pb = ctx.dr_eph_mod.predict_all(
                                ctx.dr_state["eph"], ctx.args.lat, ctx.args.lon, ctx.args.alt,
                                datetime.fromtimestamp(ctx.drp.now_w, tz=timezone.utc),
                                mask_deg=-90.0)
                            cm = set(ctx.drp.pd) & set(pb)
                            if cm:
                                dr_m = max(abs(ctx.drp.pd[k]["range_m"] - pb[k]["range_m"])
                                           for k in cm)
                                dd = max(abs(ctx.drp.pd[k]["range_rate_mps"]
                                             - pb[k]["range_rate_mps"]) for k in cm)
                                ab = (" | A/B vs BRDC over %d sats: worst range %.1f m, "
                                      "range-rate %.3f m/s (%.2f Hz@fc)"
                                      % (len(cm), dr_m, dd,
                                         dd / 299792458.0 * ctx.args.carrier_hz))
                        _log("dead-reckon: predicting from DECODED eph (%s; %d entries -> "
                             "%d sats)%s"
                             % ("FORCE A/B" if ctx.args.decoded_eph_fallback_force
                                else "BRDC network DOWN -> fallback",
                                len(_ents), len(ctx.drp.pd), ab))
                else:
                    # A0b (2026-08-23): `signal=` makes the returned sat_clk_s refer to
                    # THIS chain's code rather than the constellation's own broadcast
                    # reference (GPS L1/L2, GAL E1/E5a or E1/E5b, BDS B3I). It is the
                    # clock cp_predicted consumes, so it moves the seed directly.
                    # Measured before arming: ~+0.15 chips common at L5, +-0.3 per-sat --
                    # a b_sat-scale correction, NOT a constellation-offset one.
                    ctx.drp.pd = ctx.dr_eph_mod.predict_all(
                        ctx.dr_state["eph"], ctx.args.lat, ctx.args.lon, ctx.args.alt,
                        datetime.fromtimestamp(ctx.drp.now_w, tz=timezone.utc), mask_deg=-90.0,
                        signal=ctx.args.signal, dcb=ctx.dr_state.get("dcb"))
                    # CENTRED PAIR (task #52): +/-2 s about now_w, not [now, now+4]. The
                    # old form was a FORWARD difference, so it estimated the rate at
                    # now+2 and handed it to the seed as if it were the rate at now -- the
                    # same first-order time-tag mistake as the velocity in
                    # gnss_ephemeris.sat_pos_clk (ced7f8b51), one level up. Bias ~1.6e-3
                    # Hz/s against a 29 mHz/s single-window budget: below budget, which is
                    # why it survived, but it is free to remove and the centred form also
                    # cuts the truncation error 3x at the same 4 s baseline.
                    ctx.drp.pd2 = ctx.dr_eph_mod.predict_all(
                        ctx.dr_state["eph"], ctx.args.lat, ctx.args.lon, ctx.args.alt,
                        datetime.fromtimestamp(ctx.drp.now_w + 2.0, tz=timezone.utc),
                        mask_deg=-90.0, signal=ctx.args.signal, dcb=ctx.dr_state.get("dcb"))
                    pd0 = ctx.dr_eph_mod.predict_all(
                        ctx.dr_state["eph"], ctx.args.lat, ctx.args.lon, ctx.args.alt,
                        datetime.fromtimestamp(ctx.drp.now_w - 2.0, tz=timezone.utc),
                        mask_deg=-90.0, signal=ctx.args.signal, dcb=ctx.dr_state.get("dcb"))
            except Exception as e:
                ctx.drp.pd, ctx.drp.pd2, pd0 = {}, {}, {}
                _log("dead-reckon: predict failed: %s" % e)
            if ctx.drp.pd:
                # cache for the SEED loop (next cycle): BRDC doppler/rate for
                # search-anchored sats, so both masters share one currency
                ctx.dr_state["pd"], ctx.dr_state["pd2"] = ctx.drp.pd, ctx.drp.pd2
                ctx.dr_state["pd0"] = pd0

            # THE EPHEMERIS'S OWN EPOCH, in capture age. predict_all above ran at
            # now_w (WALL), so in capture-age units its rows are valid at
            # now_w - utc0_sample0 -- which is t_now_abs on the WALL axis and is NOT
            # t_now_abs under --dr-fengine-axis, where t_now_abs lags wall by the
            # telemetry lag (~0.15 s live). cp_predicted used to extrapolate from
            # t_now_abs unconditionally; under the axis fix that misplaces every
            # satellite by rdot*lag metres = K*dop*lag chips, PER SAT, dop-signed --
            # measured 2026-08-24 as the standing trim law trim ~ -1.0e-3 chips/Hz
            # x dop (tau 115 ms = the live axis lag), the walkoff limit cycle's root.
            # The clock median hides the common part, which is why 08-23's eps read
            # +420 ms against an 8-18 s lag. Born WITH the axis fix (08-17): on the
            # wall axis the two epochs coincide and the old form was exact.
            ctx.drp.t_eph_age = ctx.drp.now_w - ctx.utc0_sample0

            # -- receiver-clock solve (the bootstrap) + per-sat integrity residuals:
            # physical cp at each detection hop (undo the sample-0 back-reference),
            # minus the prediction, epoch-normalized to now (the offset drifts at
            # f_chip*(l-a)); the circular median over sats is the receiver clock.
            ctx.drp.offs = []
            # DETECTION AGE per sat (task #33, docs 11.22 follow-up). The epoch
            # normalization below ages each latched detection by drift*(t_now - t_i)
            # with drift = f_chip*(l-a): the l-a EMA's measured noise is +-0.05-0.07
            # chips/s, and detections latch for up to the 1276 s per-PRN revisit --
            # so the normalization term can carry CHIPS of error that scale with THIS
            # SAT'S staleness. Logged next to each residual so "integrity" can be
            # judged against age: residuals that grow with age are measuring the
            # normalization, not the sky.
            det_age = {}
            off_inputs = {}
            for prn, (snr, dop, cp, ref_hop, _nh, _cpl, _car) in sorted(ctx.best.items()):
                v = ctx.drp.pd.get((ctx.drp.tag, prn))
                if v is None:
                    continue
                t_i = ref_hop / ctx.args.hops_per_sec
                det_age[prn] = ctx.drp.t_now_abs - t_i
                # The search's sample-0 back-reference subtracts BOTH the nominal code
                # advance (t*f_chip mod L -- the 'off' term) and the code-Doppler drift;
                # add BOTH back. Omitting the nominal term hid inside the solved clock
                # whenever all detections shared a snapshot hop (every offline check),
                # but across live scans it scatters by f_chip/hops_per_sec * (ref_hop
                # mod code-period) -- the wandering "13 chips/s clock" of the first
                # live deploy (2026-07-13).
                # #45 STEP 5 (2026-08-12): this reconstruction STAYS the clock-solve
                # measurement. The payload's cp_at_ref is better conditioned in
                # principle, but it is referenced at the hop's last sample AND carries
                # the replica anchor's Doppler term (+52.3711 + 1.39e-4*dop chips vs
                # this quantity, measured on 26,815 banked detections), so consuming it
                # means importing the search's fft_len and anchor geometry here -- the
                # coupling this pass removes -- to fix a conditioning problem the sky
                # says does not exist: the detection's (cp0, dop) pair is published
                # together and its embed cancels exactly, giving 0.27-chip measured
                # continuity. Better conditioned on paper is not better when the price
                # is a second component's geometry.
                cp_loc = (cp + t_i * ctx.args.chip_rate_hz
                          * (1.0 + ctx.args.code_doppler_sign * dop / ctx.args.carrier_hz)
                          ) % ctx.code_len
                d_i = (cp_loc - ctx.cp_predicted(v, t_i)
                       + ctx.drp.drift * (ctx.drp.t_now_abs - t_i)) % ctx.code_len
                ctx.drp.offs.append((prn, d_i))
                # WHICH INPUT MOVED. Record cp_loc (NOT raw cp): raw cp swings
                # ~uniform mod L between passes by construction -- the search embeds
                # -t_abs*chip_rate*sign*dop/carrier in cp0 and the detection Doppler
                # jitters +-60 Hz pass to pass, 1696 chips/Hz at t_abs ~2.2 days.
                # That embed cancels exactly in cp_loc (verified live 2026-08-11:
                # d(cp_loc) median 0.27 chips over 1423 consecutive detections), so
                # cp_loc is the ONLY interpretable form of "what the search said".
                # The 2026-08-11 WHAT-MOVED that printed raw dcp cost half a day:
                # its thousands-of-chips swings were read as a search fault.
                off_inputs[prn] = (cp_loc, t_i, dop)
            # ERRATIC-TRACK GUARD (#39). A satellite whose solve offset JUMPS between
            # consecutive cycles is not measuring anything: d_i = clk + b_i and both
            # terms are stable, so a real satellite moves far less than 10 chips per
            # cycle. MEASURED 2026-08-10: PRN 2 -- which is not L5-capable, so this was
            # a noise or cross-correlation track -- entered at -3226 chips and then read
            # +2430 +2997 -4988 +3024 -4193 +2897 +1297 +3231 on successive cycles while
            # every real satellite sat at 1-6. It dragged the median, the clock latched
            # at 19:47, fleet-coherent alignment collapsed 0.86 -> 0.15 and engagement
            # went to zero. Catching it HERE means one satellite is dropped instead of
            # the whole solve being refused, which is what makes the refusal guard a
            # last resort rather than the first line.
            if ctx.args.dr_max_off_jump_chips > 0.0 and ctx.drp.offs:
                _keep, _drop = split_erratic_offsets(
                    ctx.drp.offs, ctx.dr_state.setdefault("off_hist", {}), ctx.drp.now_w,
                    ctx.args.dr_max_off_jump_chips, ctx.args.dr_off_jump_max_age_s, ctx.code_len)
                if _drop:
                    _prevI = ctx.dr_state.setdefault("off_inputs_prev", {})
                    _det = []
                    for _p, _j in _drop:
                        # NOT `_now`: that name is a FUNCTION in this scope, and
                        # assigning it here made every reference to _now() in main()
                        # a local-before-assignment -- the broker died on startup at
                        # a line 2000 lines away from this one.
                        _cur = off_inputs.get(_p)
                        _was = _prevI.get(_p)
                        if _cur and _was:
                            _dcl = ((_cur[0] - _was[0] + ctx.code_len / 2.0) % ctx.code_len
                                    - ctx.code_len / 2.0)
                            _det.append("PRN %d: dcp_loc %+.1f  dt_i %+.3fs  "
                                        "ddop %+.3fHz"
                                        % (_p, _dcl, _cur[1] - _was[1],
                                           _cur[2] - _was[2]))
                    if _det:
                        _log_rl("offjumpwhy", "clock solve: WHAT MOVED -- " +
                                " | ".join(_det), every_s=60.0)
                ctx.dr_state["off_inputs_prev"] = dict(off_inputs)
                if _drop and len(_keep) >= ctx.args.dr_min_sats:
                    ctx.drp.offs = _keep
                    _log_rl("offjump",
                            "clock solve: EXCLUDED %s -- offset jumped %s chips since "
                            "the last cycle (bound %.0f). d_i = clk + b_i, both "
                            "stable; the detection-Doppler embed cancels exactly in "
                            "cp_loc (2026-08-11), so a jump is a real discontinuity "
                            "in what the search reported, the model, or the drift "
                            "normalization -- see WHAT MOVED"
                            % (", ".join("PRN %d" % p for p, _ in _drop),
                               ", ".join("%.0f" % j for _, j in _drop),
                               ctx.args.dr_max_off_jump_chips), every_s=60.0)
                elif _drop:
                    # DROPPING THEM WOULD STARVE THE SOLVE. Say so rather than silently
                    # keeping them: if EVERY satellite jumped, the clock itself moved
                    # (or the model did), and that is a different fault from one bad
                    # track -- the MAD guard below is the right net for it.
                    _log_rl("offjumpkeep",
                            "clock solve: %d PRN(s) jumped but excluding them leaves "
                            "%d < --dr-min-sats %d -- keeping all; if this persists the "
                            "CLOCK moved, not one track"
                            % (len(_drop), len(_keep), ctx.args.dr_min_sats), every_s=60.0)
            ctx.dr_state["offs_t"] = ctx.drp.now_w  # freshness stamp for the referee's integrity veto
            # -- P2a SHADOW: the joint receiver-state solve (task #33, section 3a) ----
            # `offs` IS the measurement, and always was. d_i = clk + b_i: the physical
            # code phase the search measured, minus the pure model, with no clock
            # removed. Today the next 40 lines take its circular median as the clock
            # and treat the per-sat spread (+-3-7 chips, docs 11.22) as an error to be
            # gated on; the joint solve reads the SAME numbers as clock PLUS per-sat
            # bias and estimates both, separated by process noise rather than by a
            # threshold. Shadow: logged beside the median it will replace, consumed by
            # NOTHING, so every transcript digest is untouched.


            # ⚠️ NOTES MUST ESCAPE EVEN WITH NO DETECTIONS (2026-08-21). The shadow
            # block below is gated on `offs` -- this chain's OWN detections -- so on a
            # model-primary chain the filter's notes were never drained and never
            # logged. Measured: the state rejected 494 measurements in a row with
            # updates frozen at 3, the DEAF latch fired exactly as designed, and not
            # one line reached the operator. Identical in shape to the DRCLK defect
            # recorded a few hundred lines below (the four model-primary chains never
            # log the clock they consume). Drain first, unconditionally.
            if ctx.args.rrate_state or ctx.args.joint_shadow:
                try:
                    _jsn = ctx.rx.joint_receiver(ctx.band_id, ctx.code_len,
                                             rereference=ctx.args.joint_rereference, gauge_mode=ctx.args.joint_gauge)
                    for _n in _jsn.drain_notes():
                        _log_rl("joint-note", "JOINT %s: %s" % (ctx.band_id, _n),
                                every_s=10.0)
                except Exception:
                    pass
            dr_joint_shadow(ctx)
            dr_clock_solve(ctx)
            dr_clock_quality(ctx)
            # ---- ADOPT A BAND SIBLING'S CLOCK (--dr-clock-adopt) -------------------------
            # Runs AFTER the EMA above on purpose: a chain that solved its own clock this
            # cycle keeps it, and only a chain that cannot solve one (no detectors -> `offs`
            # empty -> the block above never ran) takes the sibling's. So enabling the flag
            # on a detector-bearing chain is a no-op rather than a silent override.
            #
            # IN-PROCESS SIBLING FIRST (task #27 M3). A chain co-hosted in this process
            # has already CONTRIBUTED its clock to the Receiver, so there is nothing to
            # serialise, flush, age or slew-test: the number is the same object the
            # sibling is using this cycle. The file route below stays for genuinely
            # cross-process siblings (the airspy benches, and a transitional split
            # deployment) and is unchanged. With one chain the lookup returns None and
            # this branch does not exist.
            ctx.drp.rx_sib = (ctx.rx.dr_clock(ctx.band_id, exclude=ctx.chain_id, t_now=ctx.t0)
                       if (ctx.args.dr_clock_adopt and not ctx.drp.offs) else None)
            # CROSS-BAND BOOTSTRAP (task #34). Without this a band whose chains all lack
            # detectors NEVER gets a clock: measured on sky, gal_e5b and bds_b2b sat at the
            # startup prime of 0.00 chips while gps_l5 had bootstrapped 150.74 and both
            # 1176.45 siblings had adopted it. 150 chips of error against a +-1 chip peak,
            # so dll_disc read -0.0008 (despreading noise) where E5a railed at -0.59.
            #
            # Layer 2 (the cross-band RATE) was necessary and NOT sufficient, and the
            # distinction is the whole point: a rate keeps you on the peak, a phase puts
            # you there. With the clock 150 chips out there is no peak to hold.
            #
            # ⚠️ THE ADOPTED PHASE CARRIES tau_band -- that is accepted, not overlooked. It
            # replaces a 150-chip error with a tau_band-sized one, which is inside the
            # DLL's pull-in, and the loop's steady-state residual then IS tau_band. The
            # per-band scoping exists to protect that measurement, and by blocking the
            # bootstrap it was the reason the measurement could never be made. Logged as
            # BOOTSTRAP, never as "ADOPTED ... same band", so the log distinguishes a
            # borrowed phase from a measured one.
            #
            # MODULUS: a clock may be reduced to a SHORTER code (150.74 mod 10230 is
            # well-defined) but never lengthened -- a value known mod 10230 says nothing
            # about which of the 100 periods of a 1023000-chip code it sits in. So a donor
            # whose code is shorter than ours is refused, which is the same-length guard
            # generalised rather than dropped.
            _rx_xband = False
            if ctx.drp.rx_sib is None and ctx.args.dr_clock_adopt and not ctx.drp.offs:
                _cand = ctx.rx.dr_clock_any_band(exclude=ctx.chain_id, t_now=ctx.t0)
                if _cand is not None and (_cand.extra.get("code_length") or 0) >= ctx.code_len:
                    ctx.drp.rx_sib, _rx_xband = _cand, True
            if ctx.drp.rx_sib is not None and _rx_xband:
                _v = float(ctx.drp.rx_sib.value) % ctx.code_len
                if ctx.dr_state.get("clk") is None or abs(
                        ((_v - ctx.dr_state["clk"] + ctx.code_len / 2) % ctx.code_len)
                        - ctx.code_len / 2) > 0.5:
                    _log("dead-reckon: clock BOOTSTRAP %.2f chips from in-process chain "
                         "'%s' (CROSS-BAND -- carries tau_band; the DLL residual IS that "
                         "measurement)" % (_v, ctx.drp.rx_sib.src))
                ctx.dr_state["clk"] = _v
                ctx.dr_state["clk_t"] = ctx.drp.rx_sib.t
                ctx.dr_state.pop("clk_primed", None)
            elif ctx.drp.rx_sib is not None and ctx.drp.rx_sib.extra.get("code_length") == ctx.code_len:
                # ── #104 (--dr-clock-adopt-max-chips): BOUND THE ADOPTION STEP. During
                # #103's 2026-08-30 outage, gps_l5's churn ran its legacy clock solve away
                # (150 -> 292 chips) and THIS PATH relayed the poison to gal/bds every ~2 s
                # -- fleet-wide q floor for 13 min -- while JOINT-CLK, which HAS a bound
                # (5 chips / 0.5 sigma), REFUSED the identical values throughout. One
                # guard existed; the parallel path skipped it (the peer-relative-blindness
                # class). Refuse a sibling step beyond the bound while the LOCAL clock is
                # fresh; if the local goes stale (> 300 s -- refusals do not refresh
                # clk_t), adopt anyway: a questionable clock beats a dead one, and 300 s
                # of containment turns a fleet kill into a slow, loudly-logged leak while
                # the sibling heals. 0 disables (the pre-#104 behaviour).
                _sib_v = float(ctx.drp.rx_sib.value) % ctx.code_len
                _adopt_step = None
                if ctx.dr_state.get("clk") is not None:
                    _adopt_step = (((_sib_v - ctx.dr_state["clk"] + ctx.code_len / 2)
                                    % ctx.code_len) - ctx.code_len / 2)
                _adopt_bound = getattr(ctx.args, "dr_clock_adopt_max_chips", 0.0)
                _local_fresh = (ctx.dr_state.get("clk_t") is not None
                                and ctx.t0 - ctx.dr_state["clk_t"] < 300.0)
                # ⚠️ A PRIMED CLOCK IS NOT A MEASUREMENT TO DEFEND (learned live 15:51-15:56:
                # gal/bds start with a 0.00-chip PRIME, and the first form of this guard
                # refused the sibling's real +149.44 against the placeholder -- both
                # 1176 MHz chains sat seedless until the disarm). clk_primed marks
                # exactly this state; the guard only defends a clock that was MEASURED.
                if (_adopt_bound > 0.0 and _adopt_step is not None and _local_fresh
                        and not ctx.dr_state.get("clk_primed")
                        and abs(_adopt_step) > _adopt_bound):
                    _log_rl("clkadopt-refuse",
                            "dead-reckon: sibling clock from '%s' is %+.2f chips from the "
                            "local solve -- adoption REFUSED (--dr-clock-adopt-max-chips "
                            "%.1f; #104: a poisoned sibling must not overwrite a healthy "
                            "chain; adopts again if local goes stale > 300 s)"
                            % (ctx.drp.rx_sib.src, _adopt_step, _adopt_bound),
                            every_s=30.0)
                else:
                    if ctx.dr_state.get("clk") is None or abs(
                            ((_sib_v - ctx.dr_state["clk"] + ctx.code_len / 2)
                             % ctx.code_len) - ctx.code_len / 2) > 0.5:
                        _log("dead-reckon: clock ADOPTED %.2f chips from in-process chain "
                             "'%s' (same band %s, no file transport)"
                             % (_sib_v, ctx.drp.rx_sib.src, ctx.band_id))
                    ctx.dr_state["clk"] = _sib_v
                    ctx.dr_state["clk_t"] = ctx.drp.rx_sib.t
                    # An adopted clock IS a measurement -- the sibling measured it -- so the
                    # prime is spent. If this chain ever gains detectors it should refine by
                    # EMA from here, not snap away from a good number.
                    ctx.dr_state.pop("clk_primed", None)
                    if ctx.drp.rx_sib.extra.get("drift") is not None:
                        ctx.dr_state["drift"] = float(ctx.drp.rx_sib.extra["drift"])
            elif ctx.drp.rx_sib is not None:
                # Same band, different code length: the chips are modular in a different
                # period, so the number is numerically fine and physically meaningless.
                # Refuse loudly rather than adopt a plausible wrong value.
                _log_rl("clkadopt-len",
                        "dead-reckon: chain '%s' publishes a clock mod %.0f chips but "
                        "this chain's code is %.0f -- NOT adoptable across code lengths"
                        % (ctx.drp.rx_sib.src, ctx.drp.rx_sib.extra.get("code_length") or -1, ctx.code_len),
                        every_s=60.0)
            dr_clock_adopt(ctx)
            # ---- MODEL-HEALTH GATES (design (b)): the resilience the fence never gave.
            # A fence on Doppler MOTION cannot detect a model that is simply WRONG -- a
            # stale ephemeris, a manoeuvre, a bad orbit. These can, and they use signals we
            # already compute every cycle:
            #   (1) EPHEMERIS FRESHNESS -- toe age.
            #   (2) INTEGRITY RESIDUAL -- measured-vs-predicted code phase, normally
            #       +-0.2 chips. If it blows up, the model is lying about this satellite.
            # A demoted satellite falls back to the SEARCH-measured Doppler, and because a
            # Doppler-source change is just another currency translation, the fallback
            # costs nothing: no re-anchor, no lock loss.
            # A single bad sample must never demote a satellite, and a gate that flaps is
            # worse than no gate: at startup the clock EMA is still converging, so the
            # residuals legitimately sit near the threshold and the first version of this
            # flapped UNTRUSTED/TRUSTED every few seconds (observed). Same discipline the
            # escape referee already uses: PERSISTENCE (N consecutive) + HYSTERESIS
            # (demote high, restore low) + do not judge at all until the clock has settled.
            if ctx.dr_state["clk"] is not None and ctx.dr_state.get("drift") is not None:
                ctx.dr_state.setdefault("integ", {})
                for prn_i, d_i in ctx.drp.offs:
                    r_i = (((d_i - ctx.dr_state["clk"] + ctx.code_len / 2.0) % ctx.code_len)
                           - ctx.code_len / 2.0)
                    # exported for the escape referee's integrity veto (chips, this
                    # chain's code; search-vs-model with the solved clock removed)
                    ctx.dr_state["integ"][prn_i] = (r_i, ctx.drp.now_w)
                    v_i = ctx.drp.pd.get((ctx.drp.tag, prn_i))
                    age_i = abs(v_i["toe_age_s"]) if v_i else 1e9
                    why = None
                    if age_i > ctx.args.dr_max_eph_age_s:
                        why = "ephemeris %.1f h old" % (age_i / 3600.0)
                    elif abs(r_i) > ctx.args.dr_max_integrity_chips:
                        why = "integrity residual %+.2f chips" % r_i
                    if why:
                        ctx.dr_bad[prn_i] = ctx.dr_bad.get(prn_i, 0) + 1
                    elif abs(r_i) < 0.5 * ctx.args.dr_max_integrity_chips:
                        ctx.dr_bad[prn_i] = 0          # hysteresis: restore well INSIDE
                    if (ctx.dr_bad.get(prn_i, 0) >= 3) and prn_i not in ctx.dr_untrusted:
                        ctx.dr_untrusted[prn_i] = why
                        # ⚠️ THIS MESSAGE USED TO SAY "falling back to the SEARCH-measured
                        # Doppler", which is not what happens under the default
                        # --seed-doppler auto: the seed loop skips the v_dr branch for an
                        # untrusted sat and lands on `pred` (the almanac predictor, which
                        # is BRDC too when --almanac-source brdc), NOT on the search's
                        # `dop`. The search Doppler reaches the seed only with
                        # --seed-doppler det. Corrected 2026-08-10 while tracing where
                        # GPS's carrier seed picks up its jitter -- a log line that names
                        # the wrong source sends the next reader to the wrong code.
                        _log("MODEL-UNTRUSTED PRN %d (%s, 3 consecutive) -> Doppler source "
                             "falls back dr -> %s for this sat"
                             % (prn_i, why,
                                "pred (almanac)" if ctx.args.seed_doppler != "det" else "det"))
                    elif ctx.dr_bad.get(prn_i, 0) == 0 and prn_i in ctx.dr_untrusted:
                        del ctx.dr_untrusted[prn_i]
                        _log("MODEL-TRUSTED again PRN %d (integrity %+.2f chips)"
                             % (prn_i, r_i))
            if ctx.dr_state["clk"] is not None and ctx.drp.offs and ctx.drp.now_w >= ctx.dr_state["log_next"]:
                ctx.dr_state["log_next"] = ctx.drp.now_w + 30.0
                resid = ["PRN %d %+.2f a%.0f%s" % (p, r, det_age.get(p, -1),
                                                   " BAD" if abs(r) > 1.0 else "")
                         for p, d in ctx.drp.offs
                         for r in [((d - ctx.dr_state["clk"] + ctx.code_len / 2) % ctx.code_len)
                                   - ctx.code_len / 2]]
                _log("dead-reckon clock %.2f chips (%.3f us mod %.0f ms, drift "
                     "%+.3f chips/s); integrity: %s"
                     % (ctx.dr_state["clk"], ctx.dr_state["clk"] / ctx.args.chip_rate_hz * 1e6,
                        ctx.drp.t_code * 1e3, ctx.dr_state.get("drift") or 0.0, "; ".join(resid)))
            # -- seed / re-pin every visible, undetected, unlocked sat from the model --
            dr_seed(ctx)
        # a fresh detection re-anchors via the seed loop (search = fallback); a
        # dropped seed (set below horizon) clears the model-owned state with it.
        # #83 P3-3b: EXCEPT a model-primary PRN -- its detections feed the filter and
        # the referee, never the seed, so they must not evict its dr ownership (they
        # would every single cycle, orphaning the flip the moment it started).
        for prn in list(ctx.dr_state["seeded"]):
            if (prn in ctx.best and prn not in ctx.mp_flipped) or prn not in ctx.seeds:
                ctx.dr_state["seeded"].discard(prn)
                ctx.dr_state["pin"].pop(prn, None)
