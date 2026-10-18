"""The dead-reckon stage (stage_dead_reckon): the receiver clock and the seeds it drives.

The receiver code clock is what lets a chain seed a satellite it has never detected. Each
cycle dr_joint_shadow feeds this chain's per-satellite code offsets to the receiver-wide joint
state, the clock pipeline runs in this order, and dr_seed seeds from the result:

  SOLVE   dr_clock_solve: the circular median of the per-satellite code offsets, refused
          when they scatter.
  JUDGE   dr_clock_quality: whether that solve may move the clock (time base advancing, no
          boresight transit, the step bound); then snap or EMA it in, update the drift within
          its bound, and contribute the clock to the shared Receiver.
  ADOPT   dr_clock_adopt_rx: a chain with no offsets of its own takes a co-hosted chain's
          clock, from the same band first, else across bands.

The median is a cross-check, not the level. Pool membership steps it by 1-2 chips as
satellites come and go, and in the value alone that looks the same as receiver motion. So on
a chain with `joint-consume: clk` the seeds fly the joint state's clock
(state_filter.JointReceiverState under its prior gauge, which membership does not step),
applied by dr_joint_clk as a wrapped delta so the median still selects which period of a
longer code the clock is in. The median is the fallback when the joint clock is refused and
is printed as `legacy` on the JOINT-CLK line; a chain that does not consume the joint clock
flies the median.

A prime (--dr-clock-chips) is a placeholder, not a measurement: the first accepted solve
replaces it outright and no adoption bound defends it.

Adoption is bounded by the step the donor's clock would make (--dr-clock-adopt-max-chips),
not by the donor's age alone; see dr_clock_adopt_rx.

@author Keith Vanderlinde
"""

import threading
import os
import time
from datetime import datetime, timezone

from gnss_broker.transport import _now, _log, _log_rl
from gnss_broker.clockhealth import dr_solve_healthy
from gnss_broker.seed import Seed
from gnss_broker.fits import (
    dr_seed_phys,
    dr_cp0,
    cp_rate_from_code_bias,
    seed_phase_at_ref,
    split_erratic_offsets,
)
from gnss_broker.receiver import JOINT_REF_RATE_HZ
from gnss_broker.sky import nearest_boresight, transit_sky
from gnss_broker.trkresid import wrap


def _transit_pd(ctx):
    """The geometry-only sky (sky.transit_sky) the transit freeze reads: the last one the
    dead-reckon predict built, else the cycle context's, else the strict sky ctx.drp.pd."""
    return (
        getattr(ctx.drp, "transit_pd", None)
        or getattr(ctx, "transit_pd", None)
        or getattr(ctx.drp, "pd", None)
    )


def dr_clock_solve(ctx):
    """SOLVE: the circular median of this cycle's per-satellite offsets (ctx.drp.offs), left in
    ctx.drp.raw_clk for dr_clock_quality. Needs dr_min_sats offsets. raw_clk is None when the
    solve is refused (a forced re-bootstrap aside) or the clock is frozen for a transit.

    Pool membership steps the median by 1-2 chips, so it is the level only on a chain that
    does not consume the joint clock; elsewhere it is the cross-check and the fallback, and
    it keeps the long-code period (dr_joint_clk).

    Inside a boresight transit (dr_clock_transit) no median is taken and the MAD re-bootstrap
    timer is stopped: a re-roll inside a transit would draw from the transit's noise."""
    _tr = dr_clock_transit(ctx)
    if _tr is None and ctx.dr_state.get("clk_transit") is not None:
        _log(
            "dead-reckon: boresight transit CLEAR (nearest %s%d at %.1f deg) after %.0f s -- "
            "the clock solve RESUMES from the held %.2f chips"
            % (
                ctx.dr_state["clk_transit"][1][0],
                ctx.dr_state["clk_transit"][1][1],
                (nearest_boresight(_transit_pd(ctx)) or (float("nan"),))[0],
                ctx.drp.now_w - ctx.dr_state.get("clk_transit_since", ctx.drp.now_w),
                ctx.dr_state["clk"]
                if ctx.dr_state.get("clk") is not None
                else float("nan"),
            )
        )
    elif _tr is not None and ctx.dr_state.get("clk_transit") is None:
        ctx.dr_state["clk_transit_since"] = ctx.drp.now_w
        _log(
            "dead-reckon: receiver clock FROZEN for a boresight transit (%s%d at %.1f deg < "
            "--dr-clock-transit-freeze-deg %.1f) -- holding %.2f chips at zero rate and "
            "contributing it every cycle; no solve is applied, no drift is differenced and "
            "the MAD re-bootstrap clock is stopped until it clears"
            % (
                _tr[1][0],
                _tr[1][1],
                _tr[0],
                ctx.args.dr_clock_transit_freeze_deg,
                ctx.dr_state["clk"],
            )
        )
    ctx.dr_state["clk_transit"] = _tr
    if _tr is not None:
        ctx.dr_state["mad_refused_since"] = None
        ctx.drp.raw_clk = None
        return
    if len(ctx.drp.offs) >= ctx.args.dr_min_sats:
        ref = ctx.drp.offs[0][1]
        cen = sorted(
            ((d - ref + ctx.code_len / 2) % ctx.code_len) - ctx.code_len / 2
            for _, d in ctx.drp.offs
        )
        ctx.drp.raw_clk = (cen[len(cen) // 2] + ref) % ctx.code_len
        # The median is a measurement only if its inputs cluster. Real per-satellite offsets
        # sit within ~+-10 chips (the joint state measures the biases at +-5); offsets of
        # noise detections scatter uniformly over the code (MAD ~2557 chips on a 10230-chip
        # code), and their median is an arbitrary number that would be shipped as every
        # seed's clock. Any bound from 50 to 500 chips separates the two, so
        # dr_max_solve_mad_chips (100) is not a tuning knob.
        _mad = sorted(abs(c - cen[len(cen) // 2]) for c in cen)[len(cen) // 2]
        if _mad > ctx.args.dr_max_solve_mad_chips and _dr_clock_guarded(ctx):
            # A confirmed clock is not re-rolled. The offsets do not depend on this clock
            # (d_i = cp_loc - cp_predicted, and the search's hints never use it), so a
            # scattered sky under a confirmed clock is the sky's fault (a transit the pooled
            # veto misses, an outage) and a re-roll would draw from that noise.
            # dr_clock_quality holds the clock; a wrong one is corrected by a re-pin, which
            # needs solves that agree with each other.
            _log_rl(
                "clkmad",
                "clock solve REFUSED: %d sats scatter MAD %.0f chips (bound %.0f) -- "
                "this is a median over NOISE, not a measurement; holding the CONFIRMED "
                "clk %.2f at zero rate (no re-bootstrap: a confirmed clock re-pins on "
                "agreeing solves instead)"
                % (
                    len(ctx.drp.offs),
                    _mad,
                    ctx.args.dr_max_solve_mad_chips,
                    ctx.dr_state["clk"],
                ),
                every_s=30.0,
            )
            ctx.dr_state["mad_refused_since"] = None
            ctx.drp.raw_clk = None
        elif _mad > ctx.args.dr_max_solve_mad_chips:
            # An unconfirmed clock is not defended forever: only an accepted solve can move
            # it, so a refusal that persists would be a latch. After
            # dr_solve_refused_rebootstrap_s of continuous refusal it is re-rolled.
            _since = ctx.dr_state.get("mad_refused_since")
            if _since is None:
                _since = ctx.dr_state["mad_refused_since"] = ctx.drp.now_w
            _held_s = ctx.drp.now_w - _since
            if _held_s >= ctx.args.dr_solve_refused_rebootstrap_s:
                # Forced re-bootstrap: clear the clock and leave raw_clk standing, so
                # dr_clock_quality snaps to this refused median in the same cycle. A re-roll,
                # not a measurement.
                _log(
                    "clock solve REFUSED for %.0f s (MAD %.0f, bound %.0f) -- "
                    "FORCING A RE-BOOTSTRAP off %d sats. This is a RE-ROLL, not "
                    "a measurement: holding a clock the sky disagrees with is "
                    "self-sustaining, so a fresh draw is strictly better than a "
                    "latch. Was %s"
                    % (
                        _held_s,
                        _mad,
                        ctx.args.dr_max_solve_mad_chips,
                        len(ctx.drp.offs),
                        ("%.2f chips" % ctx.dr_state["clk"])
                        if ctx.dr_state.get("clk") is not None
                        else "UNSET",
                    )
                )
                ctx.dr_state["clk"] = None
                ctx.dr_state.pop("raw_prev", None)
                ctx.dr_state["off_hist"] = {}
                ctx.dr_state["mad_refused_since"] = None
            else:
                _log_rl(
                    "clkmad",
                    "clock solve REFUSED: %d sats scatter MAD %.0f chips "
                    "(bound %.0f) -- this is a median over NOISE, not a "
                    "measurement; holding clk %s (%.0f s, re-bootstrap at "
                    "%.0f s)"
                    % (
                        len(ctx.drp.offs),
                        _mad,
                        ctx.args.dr_max_solve_mad_chips,
                        ("%.2f" % ctx.dr_state["clk"])
                        if ctx.dr_state.get("clk") is not None
                        else "UNSET",
                        _held_s,
                        ctx.args.dr_solve_refused_rebootstrap_s,
                    ),
                    every_s=30.0,
                )
                ctx.drp.raw_clk = None
        else:
            ctx.dr_state["mad_refused_since"] = None


def dr_clock_quality(ctx):
    """JUDGE: decide whether this cycle's solve (ctx.drp.raw_clk) moves the receiver clock,
    apply it, and contribute the clock to the Receiver.

    Nothing moves while the time base is frozen, and a transit freeze holds the clock. The
    first solve (no clock yet, or a prime: a placeholder, never a measurement) snaps; a later
    one goes through _dr_clock_step_gate, which holds the clock, re-pins it, or lets the EMA
    take it at dr_clock_alpha. The drift is a 0.05 EMA of the rate between consecutive
    applied solves 0.5-30 s apart, bounded by dr_max_drift_chips_s."""
    # A frozen time base must freeze the clock state. When the newest telemetry hop stops
    # advancing, the orbit model keeps moving against frozen detections and the solve reads
    # that as clock drift; a wrong drift walks every seed off-peak, so no later solve would
    # re-measure it. While the hop has not advanced for dr_clock_freeze_s, no solve is
    # applied, no drift is differenced, and raw_prev is dropped so no pair straddles the gap.
    # fe_axis[0][1] is the wall time the hop last advanced, the same clock as now_w.
    _fe = ctx.fe_axis[0] if getattr(ctx, "fe_axis", None) else None
    _freeze_s = ctx.args.dr_clock_freeze_s
    if (
        _fe is not None
        and ctx.drp.now_w is not None
        and ctx.drp.now_w - _fe[1] > _freeze_s
    ):
        if ctx.dr_state.pop("raw_prev", None) is not None or not ctx.dr_state.get(
            "clk_frozen"
        ):
            _log(
                "dead-reckon: time base frozen %.0f s (hop %.0f) -- receiver clock solve "
                "and drift EMA HELD; raw_prev dropped so nothing is differenced across the "
                "gap (--dr-clock-freeze-s %.1f)"
                % (ctx.drp.now_w - _fe[1], _fe[0], _freeze_s)
            )
        ctx.dr_state["clk_frozen"] = True
        return
    if ctx.dr_state.pop("clk_frozen", False):
        _log(
            "dead-reckon: time base advancing again -- clock solve RESUMES from the held "
            "state (drift %s)"
            % (
                "%+.4f chips/s" % ctx.dr_state["drift"]
                if ctx.dr_state.get("drift") is not None
                else "unmeasured"
            )
        )
    # A transit freeze (decided in dr_clock_solve) holds the clock every cycle, whatever the
    # detections say.
    if ctx.dr_state.get("clk_transit") is not None:
        dr_clock_hold(ctx)
        return
    if len(ctx.drp.offs) >= ctx.args.dr_min_sats and ctx.drp.raw_clk is not None:
        # An established clock moves only on a solve of enough satellites whose step is
        # inside the bound, or on a confirmed re-pin (_dr_clock_step_gate). The bootstrap
        # below (no clock yet, or a prime) is exempt: there is nothing to defend, and waiting
        # for more satellites could leave a cold chain unseeded.
        if (
            ctx.dr_state["clk"] is not None
            and not ctx.dr_state.get("clk_primed")
            and _dr_clock_step_gate(ctx) != "update"
        ):
            return
        prev_raw = ctx.dr_state.get("raw_prev")
        if prev_raw is not None and 0.5 < ctx.drp.now_w - prev_raw[1] < 30.0:
            d_est = (
                ((ctx.drp.raw_clk - prev_raw[0] + ctx.code_len / 2) % ctx.code_len)
                - ctx.code_len / 2
            ) / (ctx.drp.now_w - prev_raw[1])
            # d_est is a difference of two clock solves, so anything that displaces the solve
            # (a node or F-engine restart, detections straddling one) arrives as clock motion
            # that never happened, and the 0.05 EMA would carry it for minutes. The true drift
            # of this GPS-disciplined clock is ~4e-4 chips/s and the noisiest legitimate
            # estimate scatters +-0.07 chips/s, so dr_max_drift_chips_s (1.0) rejects nothing
            # real.
            if abs(d_est) > ctx.args.dr_max_drift_chips_s:
                _log_rl(
                    "driftrej",
                    "clock drift estimate %+.1f chips/s REJECTED (bound "
                    "%.2f): the solve jumped, the clock did not -- holding "
                    "drift %+.4f"
                    % (
                        d_est,
                        ctx.args.dr_max_drift_chips_s,
                        ctx.dr_state.get("drift") or 0.0,
                    ),
                    every_s=30.0,
                )
            else:
                ctx.dr_state["drift"] = (
                    d_est
                    if ctx.dr_state.get("drift") is None
                    else ctx.dr_state["drift"] + 0.05 * (d_est - ctx.dr_state["drift"])
                )
                ctx.dr_state["drift_t"] = ctx.drp.now_w
        ctx.dr_state["raw_prev"] = (ctx.drp.raw_clk, ctx.drp.now_w)
        # Snap on the first solve, prime or not: a median over >= dr_min_sats satellites
        # measures what the prime guessed, and EMA-ing in from a guess only adds a walk-in.
        if ctx.dr_state["clk"] is None or ctx.dr_state.pop("clk_primed", False):
            was = ctx.dr_state["clk"]
            ctx.dr_state["clk"] = ctx.drp.raw_clk
            # Nothing has agreed with a snap yet: it is unconfirmed (no transit freeze holds
            # it), and the new clk_epoch tells adopting chains it is a deliberate new clock.
            ctx.dr_state["clk_confirmed"] = False
            ctx.dr_state.pop("repin", None)
            ctx.dr_state["clk_epoch"] = ctx.dr_state.get("clk_epoch", 0) + 1
            _log(
                "dead-reckon: receiver clock BOOTSTRAP %.2f chips = %.3f us "
                "(mod %.0f ms; %d sats%s)"
                % (
                    ctx.drp.raw_clk,
                    ctx.drp.raw_clk / ctx.args.chip_rate_hz * 1e6,
                    ctx.drp.t_code * 1e3,
                    len(ctx.drp.offs),
                    ""
                    if was is None
                    else "; REPLACES the %.2f-chip prime -- a prime is a seed, not a "
                    "measurement" % was,
                )
            )
        else:
            clk = (
                ctx.dr_state["clk"]
                + ctx.drp.drift * (ctx.drp.now_w - ctx.dr_state["clk_t"])
            ) % ctx.code_len
            step = (
                (ctx.drp.raw_clk - clk + ctx.code_len / 2) % ctx.code_len
            ) - ctx.code_len / 2
            ctx.dr_state["clk"] = (clk + ctx.args.dr_clock_alpha * step) % ctx.code_len
            ctx.dr_state["clk_confirmed"] = True
        ctx.dr_state["clk_t"] = ctx.drp.now_w
        # Contribute to the shared Receiver: clk and drift are the receiver's, the rest of
        # dr_state this chain's, and a co-hosted chain without offsets adopts from there.
        # Carried with its code length and chip rate, because chips are modular and a unit: a
        # value mod 10230 says nothing about a 1023000-chip code, and a consumer at another
        # chip rate must convert it as a time.
        ctx.rx.contribute_dr_clock(
            ctx.chain_id,
            ctx.band_id,
            ctx.dr_state["clk"],
            ctx.dr_state.get("drift"),
            ctx.drp.now_w,
            ctx.code_len,
            chip_rate_hz=ctx.args.chip_rate_hz,
            healthy=dr_solve_healthy(ctx.dr_state, ctx.drp.now_w),
            epoch=ctx.dr_state.get("clk_epoch"),
        )
    elif _dr_clock_guarded(ctx):
        # No usable solve this cycle (MAD-refused, or too few offsets). A confirmed clock is
        # held and contributed, as through a transit; a refused solve breaks a re-pin
        # candidate's run, a thin cycle leaves it standing.
        if len(ctx.drp.offs) >= ctx.args.dr_min_sats:
            ctx.dr_state.pop("repin", None)
        dr_clock_hold(ctx)


def _dr_clock_guarded(ctx):
    """True when this chain's receiver clock is solved and confirmed: a bootstrap followed by
    an in-bound update of dr_update_min_sats satellites, or a re-pin. A confirmed clock is
    never re-rolled, and while the time base advances it is held (contributed at zero rate)
    on every cycle without a usable solve. A prime, a bootstrap nothing has agreed with yet,
    and a chain that never solves are not guarded."""
    return (
        ctx.dr_state.get("clk") is not None
        and not ctx.dr_state.get("clk_primed")
        and bool(ctx.dr_state.get("clk_confirmed"))
    )


def dr_clock_transit(ctx):
    """(separation_deg, (sys, prn)) while this chain's confirmed receiver clock must freeze for
    a boresight transit (separation below dr_clock_transit_freeze_deg), else None.

    A satellite near boresight rails the quantiser every chain shares; the search then reports
    other satellites at noise code phases with SNR just over its bar, and two of them agreeing
    inside the MAD bound make a median hundreds of chips off. The receiver clock is
    GPS-disciplined and drifts ~4e-4 chips/s, a fraction of a chip over a transit, so inside
    a transit the best estimate is the one held at its entry, and no solve is consulted.

    Pooled over every constellation in the transit sky (sky.nearest_boresight), because the
    bright satellite need not be one this chain tracks. Only a confirmed clock freezes
    (_dr_clock_guarded): a prime or an unconfirmed bootstrap may be the transit's own noise and
    must stay free to re-pin, and a chain that never solves never confirms, so it keeps
    adopting."""
    _deg = ctx.args.dr_clock_transit_freeze_deg
    if (
        ctx.dr_state.get("clk") is None
        or ctx.dr_state.get("clk_primed")
        or not ctx.dr_state.get("clk_confirmed")
    ):
        return None
    _near = nearest_boresight(_transit_pd(ctx))
    return _near if (_near is not None and _near[0] < _deg) else None


def dr_clock_hold(ctx):
    """Hold this chain's clock at zero rate for this cycle and contribute it, marked held.

    Zero rate, not propagation at `drift`: the held value is the best estimate of a constant,
    and the drift EMA's noise (~0.01 chips/s) would integrate into a walk over a long hold
    (18 chips in 30 minutes). clk_t moves to now, so dr_seed's clk + drift*(now - clk_t) is
    the held value exactly.

    drift_t follows the hold, so the drift does not expire (dr_drift_max_age_s) during it:
    nothing consumes it meanwhile, and the value measured before the hold is a better rate
    afterwards than the f_chip*(l-a) fallback, which is fed by the code-rate fits of the very
    transit that caused the hold.

    The contribution carries this cycle's stamp and held=True, so a consumer takes the value
    as valid at its own now rather than extrapolating it at its own rate (a cross-band
    consumer carries no drift and would use f_chip*(l-a)). raw_prev is dropped so no drift
    pair straddles a held cycle."""
    ctx.dr_state["clk_t"] = ctx.drp.now_w
    if ctx.dr_state.get("drift") is not None:
        ctx.dr_state["drift_t"] = ctx.drp.now_w
    ctx.dr_state.pop("raw_prev", None)
    ctx.rx.contribute_dr_clock(
        ctx.chain_id,
        ctx.band_id,
        ctx.dr_state["clk"],
        ctx.dr_state.get("drift"),
        ctx.drp.now_w,
        ctx.code_len,
        chip_rate_hz=ctx.args.chip_rate_hz,
        epoch=ctx.dr_state.get("clk_epoch"),
        held=True,
        # a hold suspends the solve on purpose, so it makes no health claim
        healthy=None,
    )


def _dr_clock_newest_hop(ctx):
    """The newest detection ref_hop among this cycle's solve inputs, or None when unknown. The
    search's table latches: a stalled search hands the broker the same detections cycle after
    cycle, and their median is then the same measurement repeated, not agreement."""
    _best = getattr(ctx, "best", None) or {}
    _h = [_best[p][3] for p, _ in ctx.drp.offs if p in _best and len(_best[p]) > 3]
    return max(_h) if _h else None


def _dr_clock_step_gate(ctx):
    """Judge one MAD-accepted solve against an established clock.

    Returns "update" when the EMA may apply it; otherwise holds or re-pins the clock itself and
    returns "held" or "repinned".

      Few satellites (dr_update_min_sats): a median of 2-3 follows its members' biases (+-5
        chips) as much as the clock, and it is where two corrupted detections that agree
        become a clock, so such a solve never updates the clock; it is held.
      The step (dr_clock_step_max_chips) is the solve against the propagated clock, the
        quantity the EMA would take dr_clock_alpha of. Inside it the update is the plain EMA.
      A re-pin (dr_clock_repin_solves): a step beyond the bound is taken only when that many
        consecutive solves agree with each other within the bound, and then whole, as a snap.
        This is how a wrong clock is corrected, so it must stay reachable: a disagreeing solve
        restarts the count, a MAD-refused one breaks it on a confirmed clock, a thin cycle
        leaves it standing. A confirmed clock counts only solves of dr_update_min_sats; an
        unconfirmed one (a bootstrap or re-roll nothing has agreed with) counts any solve of
        dr_min_sats, so a thin but consistent sky can replace it. A solve whose newest
        detection is not newer than the last counted one's does not count
        (_dr_clock_newest_hop): agreement must come from independent measurements."""
    _nmin = ctx.args.dr_update_min_sats
    _smax = ctx.args.dr_clock_step_max_chips
    _n = len(ctx.drp.offs)
    _enough = _n >= _nmin
    L = ctx.code_len

    def _thin():
        _log_rl(
            "clkthin",
            "dead-reckon: receiver clock HELD at %.2f chips -- %d sat(s) in the solve "
            "(< --dr-update-min-sats %d): a thin median follows its members' biases, not "
            "the clock" % (ctx.dr_state["clk"], _n, _nmin),
            every_s=60.0,
        )
        dr_clock_hold(ctx)
        return "held"

    _prop = (
        ctx.dr_state["clk"] + ctx.drp.drift * (ctx.drp.now_w - ctx.dr_state["clk_t"])
    ) % L
    _step = ((ctx.drp.raw_clk - _prop + L / 2) % L) - L / 2
    if abs(_step) <= _smax:
        if not _enough:
            return _thin()
        ctx.dr_state.pop("repin", None)
        return "update"
    if not _enough and ctx.dr_state.get("clk_confirmed"):
        return _thin()
    _k = ctx.args.dr_clock_repin_solves
    _cand = ctx.dr_state.get("repin")
    _hop = _dr_clock_newest_hop(ctx)
    if (
        _cand is not None
        and abs(((ctx.drp.raw_clk - _cand["v"] + L / 2) % L) - L / 2) <= _smax
    ):
        if _hop is None or _cand.get("hop") is None or _hop > _cand["hop"]:
            _cand["n"] += 1
            _cand["hop"] = _hop
    else:
        _cand = ctx.dr_state["repin"] = {
            "v": ctx.drp.raw_clk,
            "n": 1,
            "t": ctx.drp.now_w,
            "hop": _hop,
        }
    if _cand["n"] >= _k:
        _was = ctx.dr_state["clk"]
        _conf = bool(ctx.dr_state.get("clk_confirmed"))
        ctx.dr_state["clk"] = ctx.drp.raw_clk
        ctx.dr_state["clk_t"] = ctx.drp.now_w
        ctx.dr_state["clk_confirmed"] = True
        ctx.dr_state.pop("raw_prev", None)
        ctx.dr_state.pop("repin", None)
        ctx.dr_state["clk_epoch"] = ctx.dr_state.get("clk_epoch", 0) + 1
        _log(
            "dead-reckon: receiver clock RE-PIN %.2f chips (%+.2f from the held %s%.2f): %d "
            "consecutive solves (this one %d sats) agree within %.1f chips over %.0f s -- a "
            "confirmed step, taken whole"
            % (
                ctx.drp.raw_clk,
                _step,
                "" if _conf else "UNCONFIRMED ",
                _was,
                _cand["n"],
                _n,
                _smax,
                ctx.drp.now_w - _cand["t"],
            )
        )
        ctx.rx.contribute_dr_clock(
            ctx.chain_id,
            ctx.band_id,
            ctx.dr_state["clk"],
            ctx.dr_state.get("drift"),
            ctx.drp.now_w,
            L,
            chip_rate_hz=ctx.args.chip_rate_hz,
            epoch=ctx.dr_state.get("clk_epoch"),
            healthy=dr_solve_healthy(ctx.dr_state, ctx.drp.now_w),
        )
        return "repinned"
    _log_rl(
        "clkstep",
        "dead-reckon: clock solve step %+.1f chips REFUSED (%d sats, bound %.1f) -- holding "
        "%.2f chips; a step is taken only after %d consecutive agreeing solves (this one "
        "is %d)" % (_step, _n, _smax, ctx.dr_state["clk"], _k, _cand["n"]),
        every_s=30.0,
    )
    dr_clock_hold(ctx)
    return "held"


def _dr_clock_local_fresh(ctx):
    """Is this clock-consuming chain's local clock fresh? A fresh clock refuses a donor's
    undeclared step beyond the adoption bound; once it is 300 s stale the step is adopted
    anyway (a questionable clock beats a dead one).

    The age runs from the newer of two stamps:
      clk_src_t  the donor's stamp on the clock last taken from it (same-band adoption or
                 cross-band bootstrap); clk_t before the first. A refusal holds the clock at
                 zero rate (clk_t moves to now) but never refreshes clk_src_t, so once a chain
                 has adopted, refusing cannot keep it fresh forever.
      jclk_t     when JOINT-CLK last adopted the joint clock on top of it. A chain the joint
                 is vouching for (healthy by its own gates, and within their bound of this
                 very clock) is not stale and must not escape onto a stepped donor; when the
                 joint goes deaf, unhealthy or disagrees, adoptions stop and the escape
                 returns 300 s later.

    A step the donor declared (a new clock_epoch: a bootstrap, re-roll or re-pin of its own
    solve) is not refused at all: the callers adopt it at any size (_dr_clock_declared)."""
    _src = ctx.dr_state.get("clk_src_t", ctx.dr_state.get("clk_t"))
    _ts = [_t for _t in (_src, ctx.dr_state.get("jclk_t")) if _t is not None]
    return bool(_ts) and ctx.t0 - max(_ts) < 300.0


def _dr_clock_declared(ctx):
    """True when the donor's contribution carries a clock_epoch this chain has not adopted yet:
    the donor snapped its clock on purpose (bootstrap, re-roll, re-pin) since our last
    adoption. False when the donor publishes no epoch, so every step is bounded."""
    _e = ctx.drp.rx_sib.extra.get("epoch") if ctx.drp.rx_sib is not None else None
    return _e is not None and _e != ctx.dr_state.get("clk_src_epoch")


def dr_clock_adopt_rx(ctx):
    """ADOPT: take a co-hosted chain's receiver clock when this chain has no offsets to solve
    its own: the same band first, else across bands as a bootstrap, both bounded by
    --dr-clock-adopt-max-chips. Leaves ctx.drp.rx_sib as the donor it looked at (None if
    none). Runs after dr_clock_quality."""
    # Only a chain with no offsets this cycle adopts, so --dr-clock-adopt on a chain with
    # detections never overrides its own solve. The donor's clock comes from the shared
    # Receiver: its latest contribution, at most 120 s old.
    ctx.drp.rx_sib = (
        ctx.rx.dr_clock(ctx.band_id, exclude=ctx.chain_id, t_now=ctx.t0)
        if (ctx.args.dr_clock_adopt and not ctx.drp.offs)
        else None
    )
    # Cross-band bootstrap: with no same-band donor, take any band's clock. The adopted phase
    # carries the band delay difference tau_band, on purpose: it replaces an error of the
    # whole clock (~150 chips against a +-1 chip peak) with a tau_band-sized one inside the
    # DLL's pull-in, and the loop's steady-state residual then measures tau_band. It is
    # logged as BOOTSTRAP, never as a same-band ADOPTED, so the log tells a borrowed phase
    # from a measured one.
    #
    # The donor publishes chips at its own chip rate, and the clock is a time: the value is
    # converted through the two chip rates (Receiver.clock_chips_convert). A clock can be
    # reduced to a shorter period but never extended to a longer one, so a donor whose code
    # window is shorter in time than our code period is refused unless the epoch extension
    # below resolves which of our periods it is in.
    _rx_xband = False
    _rx_xext = None
    if ctx.drp.rx_sib is None and ctx.args.dr_clock_adopt and not ctx.drp.offs:
        _cand = ctx.rx.dr_clock_any_band(exclude=ctx.chain_id, t_now=ctx.t0)
        if _cand is not None and _cand.extra.get("healthy") is False:
            # a donor whose own search solve is not established may have solved on noise
            _log_rl(
                "clkxboot-unhealthy",
                "dead-reckon: cross-band donor '%s' WITHHELD -- its search solve is not "
                "established; the clock stays as it is" % _cand.src,
                every_s=60.0,
            )
            _cand = None
        if _cand is not None:
            _don_rate = _cand.extra["chip_rate_hz"]
            _don_window_s = (_cand.extra.get("code_length") or 0) / _don_rate
            _our_window_s = ctx.code_len / ctx.args.chip_rate_hz
            if _don_window_s >= _our_window_s:
                ctx.drp.rx_sib, _rx_xband = _cand, True
            else:
                # Epoch extension: the donor's clock is known only mod its shorter window,
                # and the clock-mod-epoch record (gps_l5's NH joint fit, clock mod 20 ms)
                # resolves which of our periods it is in while the donor supplies the
                # precision. Candidates are a full donor window apart, so the record need
                # only be right to half a window.
                _ext = ctx.rx.clock_mod_epoch(
                    min_epoch_s=_our_window_s, exclude=ctx.chain_id, t_now=ctx.t0
                )
                if _ext is not None:
                    ctx.drp.rx_sib, _rx_xband, _rx_xext = _cand, True, _ext
                else:
                    _log_rl(
                        "clkxmod-none",
                        "dead-reckon: cross-band donor '%s' REFUSED on "
                        "modulus (its %.4f s window < our %.4f s period) and "
                        "no clock-mod-epoch record covers us -- the clock "
                        "stays primed. The NH joint fit (--nh-joint) on a "
                        "sibling chain is what contributes that record."
                        % (_cand.src, _don_window_s, _our_window_s),
                        every_s=300.0,
                    )
    if ctx.drp.rx_sib is not None and _rx_xband:
        _don_rate = ctx.drp.rx_sib.extra["chip_rate_hz"]
        if _rx_xext is None:
            _v = ctx.rx.clock_chips_convert(
                ctx.drp.rx_sib.value, _don_rate, ctx.args.chip_rate_hz, ctx.code_len
            )
        else:
            _don_window_s = (
                ctx.drp.rx_sib.extra.get("code_length") or ctx.code_len
            ) / _don_rate
            _our_window_s = ctx.code_len / ctx.args.chip_rate_hz
            _t_sec, _resid_s = ctx.rx.clock_extend_mod(
                float(ctx.drp.rx_sib.value) / _don_rate,
                _don_window_s,
                float(_rx_xext.value),
                _rx_xext.extra["epoch_s"],
                _our_window_s,
            )
            if abs(_resid_s) > 0.25 * _don_window_s:
                # The two sources disagree by a large fraction of a donor window, so the
                # choice of period itself is in doubt: refuse, and log it.
                _log_rl(
                    "clkxmod-resid",
                    "dead-reckon: epoch extension REFUSED -- donor '%s' and "
                    "clock-mod '%s' disagree by %+.1f us (%.0f%% of the donor "
                    "window). One of them is wrong; the clock stays primed."
                    % (
                        ctx.drp.rx_sib.src,
                        _rx_xext.src,
                        _resid_s * 1e6,
                        100.0 * abs(_resid_s) / _don_window_s,
                    ),
                    every_s=300.0,
                )
                ctx.drp.rx_sib = None
                _rx_xband = False
                _v = None
            else:
                _v = (_t_sec * ctx.args.chip_rate_hz) % ctx.code_len
    if ctx.drp.rx_sib is not None and _rx_xband:
        # Bounded like a same-band adoption, as a time: --dr-clock-adopt-max-chips of the
        # donor's chips (5 = 489 ns at 10.23 Mcps), converted into ours through the two chip
        # rates, since a chip count means a different time on each chain. Same freshness rule
        # (_dr_clock_local_fresh), prime exemption and declared-step exemption; a refusal
        # holds the clock at zero rate rather than extrapolating it at this chain's (l-a) rate.
        _xb_rate = ctx.drp.rx_sib.extra.get("chip_rate_hz") or ctx.args.chip_rate_hz
        _xb_bound = float(getattr(ctx.args, "dr_clock_adopt_max_chips", 0.0) or 0.0)
        _xb_ours = _xb_bound * ctx.args.chip_rate_hz / _xb_rate
        _xb_step = None
        if ctx.dr_state.get("clk") is not None:
            _xb_step = (
                (_v - ctx.dr_state["clk"] + ctx.code_len / 2) % ctx.code_len
            ) - ctx.code_len / 2
        _xb_decl = _dr_clock_declared(ctx)
        # A step of more than half the donor's window moves this clock by a whole donor
        # period, which only the epoch extension can justify, so the clock-mod record behind
        # it must be recent (120 s) and from an established solve.
        _xb_whole_refuse = False
        if (
            _xb_step is not None
            and _rx_xext is not None
            and not ctx.dr_state.get("clk_primed")
            and abs(_xb_step) > 0.5 * _don_window_s * ctx.args.chip_rate_hz
        ):
            _xb_whole_refuse = (ctx.t0 - _rx_xext.t > 120.0) or (
                _rx_xext.extra.get("healthy") is False
            )
        if (
            _xb_bound > 0.0
            and _xb_step is not None
            and not ctx.dr_state.get("clk_primed")
            and not _xb_decl
            and _dr_clock_local_fresh(ctx)
            and abs(_xb_step) > _xb_ours
        ):
            _log_rl(
                "clkxboot-refuse",
                "dead-reckon: cross-band clock from '%s' is %+.3f chips (%+.0f ns) "
                "from the local clock -- bootstrap REFUSED (bound %.3f chips = "
                "%.0f ns: --dr-clock-adopt-max-chips %.1f of the donor's %.3f-Mcps "
                "chips); holding %.3f chips at zero rate, adopts again once the "
                "local clock is 300 s stale with no JOINT-CLK adoption"
                % (
                    ctx.drp.rx_sib.src,
                    _xb_step,
                    _xb_step / ctx.args.chip_rate_hz * 1e9,
                    _xb_ours,
                    _xb_bound / _xb_rate * 1e9,
                    _xb_bound,
                    _xb_rate / 1e6,
                    ctx.dr_state["clk"],
                ),
                every_s=30.0,
            )
            ctx.dr_state["clk_t"] = ctx.t0
        elif _xb_whole_refuse:
            _log_rl(
                "clkxboot-whole",
                "dead-reckon: cross-band clock from '%s' would step %+.3f chips = %+.2f ms, "
                "a whole period of the donor's code, on a clock-mod record from '%s' that is "
                "%.0f s old%s -- bootstrap REFUSED, holding %.3f chips"
                % (
                    ctx.drp.rx_sib.src,
                    _xb_step,
                    _xb_step / ctx.args.chip_rate_hz * 1e3,
                    _rx_xext.src,
                    ctx.t0 - _rx_xext.t,
                    ""
                    if _rx_xext.extra.get("healthy") is not False
                    else " and from an unestablished solve",
                    ctx.dr_state["clk"],
                ),
                every_s=30.0,
            )
            ctx.dr_state["clk_t"] = ctx.t0
        else:
            if (
                ctx.dr_state.get("clk") is None
                or abs(
                    ((_v - ctx.dr_state["clk"] + ctx.code_len / 2) % ctx.code_len)
                    - ctx.code_len / 2
                )
                > 0.5
            ):
                _log(
                    "dead-reckon: clock BOOTSTRAP %.2f chips (donor %.2f @ %.3f Mcps, "
                    "ours %.3f) from in-process chain '%s' (CROSS-BAND -- carries "
                    "tau_band; the DLL residual IS that measurement)%s"
                    % (
                        _v,
                        float(ctx.drp.rx_sib.value),
                        _don_rate / 1e6,
                        ctx.args.chip_rate_hz / 1e6,
                        ctx.drp.rx_sib.src,
                        ""
                        if _rx_xext is None
                        else " [EPOCH-EXTENDED by '%s' clock-mod-%.0f-ms, resid %+.1f us]"
                        % (
                            _rx_xext.src,
                            _rx_xext.extra["epoch_s"] * 1e3,
                            _resid_s * 1e6,
                        ),
                    )
                )
            if (
                _xb_decl
                and _xb_step is not None
                and _xb_bound > 0.0
                and not ctx.dr_state.get("clk_primed")
                and abs(_xb_step) > _xb_ours
            ):
                _log(
                    "dead-reckon: cross-band clock from '%s' stepped %+.3f chips -- "
                    "ADOPTED: a DECLARED snap of its own solve (clock epoch %s)"
                    % (ctx.drp.rx_sib.src, _xb_step, ctx.drp.rx_sib.extra.get("epoch"))
                )
            ctx.dr_state["clk"] = _v
            # a held donor value is its clock at zero rate, valid now: never extrapolate it
            # from its stamp at this chain's own f_chip*(l-a)
            ctx.dr_state["clk_t"] = (
                ctx.t0 if ctx.drp.rx_sib.extra.get("held") else ctx.drp.rx_sib.t
            )
            ctx.dr_state["clk_src_t"] = ctx.drp.rx_sib.t
            ctx.dr_state["clk_src_epoch"] = ctx.drp.rx_sib.extra.get("epoch")
            ctx.dr_state.pop("clk_primed", None)
    elif (
        ctx.drp.rx_sib is not None
        and ctx.drp.rx_sib.extra.get("code_length") == ctx.code_len
    ):
        # Same band: refuse a donor step beyond --dr-clock-adopt-max-chips (0 = unbounded)
        # while the local clock is fresh (_dr_clock_local_fresh). Once the local clock is
        # 300 s stale the step is adopted anyway: a questionable clock beats a dead one, and
        # 300 s of refusals turns a donor's bad solve into a slow, logged leak instead of an
        # instant copy onto every chain.
        _sib_v = float(ctx.drp.rx_sib.value) % ctx.code_len
        _adopt_step = None
        if ctx.dr_state.get("clk") is not None:
            _adopt_step = (
                (_sib_v - ctx.dr_state["clk"] + ctx.code_len / 2) % ctx.code_len
            ) - ctx.code_len / 2
        _adopt_bound = getattr(ctx.args, "dr_clock_adopt_max_chips", 0.0)
        _local_fresh = _dr_clock_local_fresh(ctx)
        # A primed clock is a placeholder, not a measurement to defend: refusing a donor's
        # real clock against it would leave this chain seedless.
        _adopt_decl = _dr_clock_declared(ctx)
        if (
            _adopt_bound > 0.0
            and _adopt_step is not None
            and _local_fresh
            and not ctx.dr_state.get("clk_primed")
            and not _adopt_decl
            and abs(_adopt_step) > _adopt_bound
        ):
            _log_rl(
                "clkadopt-refuse",
                "dead-reckon: sibling clock from '%s' is %+.2f chips from the "
                "local solve -- adoption REFUSED (--dr-clock-adopt-max-chips "
                "%.1f; #104: a poisoned sibling must not overwrite a healthy "
                "chain; holding %.2f chips at zero rate, adopts again once the "
                "local clock is 300 s stale with no JOINT-CLK adoption)"
                % (ctx.drp.rx_sib.src, _adopt_step, _adopt_bound, ctx.dr_state["clk"]),
                every_s=30.0,
            )
            # Hold at zero rate rather than extrapolate at the drift the refused donor handed
            # over at its last adoption.
            ctx.dr_state["clk_t"] = ctx.t0
        else:
            if (
                ctx.dr_state.get("clk") is None
                or abs(
                    ((_sib_v - ctx.dr_state["clk"] + ctx.code_len / 2) % ctx.code_len)
                    - ctx.code_len / 2
                )
                > 0.5
            ):
                _log(
                    "dead-reckon: clock ADOPTED %.2f chips from in-process chain "
                    "'%s' (same band %s, no file transport)"
                    % (_sib_v, ctx.drp.rx_sib.src, ctx.band_id)
                )
            if (
                _adopt_decl
                and _adopt_step is not None
                and _adopt_bound > 0.0
                and not ctx.dr_state.get("clk_primed")
                and abs(_adopt_step) > _adopt_bound
            ):
                _log(
                    "dead-reckon: sibling clock from '%s' stepped %+.2f chips -- "
                    "ADOPTED: a DECLARED snap of its own solve (clock epoch %s)"
                    % (
                        ctx.drp.rx_sib.src,
                        _adopt_step,
                        ctx.drp.rx_sib.extra.get("epoch"),
                    )
                )
            ctx.dr_state["clk"] = _sib_v
            ctx.dr_state["clk_t"] = (
                ctx.t0 if ctx.drp.rx_sib.extra.get("held") else ctx.drp.rx_sib.t
            )  # a held donor value is valid now
            ctx.dr_state["clk_src_t"] = ctx.drp.rx_sib.t
            ctx.dr_state["clk_src_epoch"] = ctx.drp.rx_sib.extra.get("epoch")
            # An adopted clock is a measurement (the donor's), so the prime is spent: if this
            # chain gains detections it refines from here rather than snapping.
            ctx.dr_state.pop("clk_primed", None)
            # The drift travels with the clock, unknown included: keeping our own value when
            # the donor has none would let a stale drift outlive the donor's.
            _sd = ctx.drp.rx_sib.extra.get("drift")
            ctx.dr_state["drift"] = float(_sd) if _sd is not None else None
            ctx.dr_state["drift_t"] = ctx.drp.rx_sib.t
    elif ctx.drp.rx_sib is not None:
        # Same band, different code length: the value is modular in a different period,
        # numerically plausible and physically meaningless, so refuse it and log.
        _log_rl(
            "clkadopt-len",
            "dead-reckon: chain '%s' publishes a clock mod %.0f chips but "
            "this chain's code is %.0f -- NOT adoptable across code lengths"
            % (
                ctx.drp.rx_sib.src,
                ctx.drp.rx_sib.extra.get("code_length") or -1,
                ctx.code_len,
            ),
            every_s=60.0,
        )


def dr_joint_shadow(ctx):
    """Feed this chain's per-satellite offsets to the receiver-wide joint state.

    It touches no seed and not dr_state's clock: the joint state reaches the seeds only
    through the joint consumers (dr_joint_clk in dr_seed, and the rate and slew consumers).
    It also records each established satellite's MINNOV for the model-primacy gate. Runs
    when any joint consumer is configured (args.joint_shadow) and this chain has offsets.

    The state is in reference chips (receiver.JOINT_REF_RATE_HZ, 10.23 Mcps) whatever this
    chain's chip rate: the offsets, their sigma and the warm-start clock go in multiplied by
    _k = reference rate / ours, and the residuals read back against the state (MINNOV, P2C)
    come out divided by it, so the histories and log lines stay in this chain's chips. At
    10.23 Mcps _k is exactly 1.0 and nothing changes bit for bit."""
    if ctx.args.joint_shadow and ctx.drp.offs:
        try:
            _k = JOINT_REF_RATE_HZ / ctx.args.chip_rate_hz
            # One state for the whole receiver, warm-started from this chain's clock: born at
            # 0 against a true clock of ~151 chips, every measurement would land beyond
            # escape_max_step and be refused. clk0 is read only at creation, so it cannot
            # fight the running filter.
            _js = ctx.rx.joint_receiver(
                ctx.band_id,
                ctx.code_len,
                ctx.args.chip_rate_hz,
                clk0=float(ctx.dr_state.get("clk") or 0.0) * _k,
                gauge_mode="prior",
            )
            # Only detections at or above joint_min_snr feed the state: below it a detection's
            # code phase is noise (~2000-chip residuals), which a median shrugs off and the
            # filter cannot.
            _snr = {p: v[0] for p, v in ctx.best.items()}
            # The P2C coast (--joint-p2c-rotate) withholds one established satellite at a time
            # (p2c_hold) and logs its residual against the state's prediction (clk + b_i):
            # what the shared state gets wrong about a satellite it is no longer told about.
            # No measurement feeds the state for joint_feed_warmup_s after broker start:
            # establishment-phase offsets can be wrong yet self-consistent, and a birth has no
            # innovation to gate on.
            if time.time() - ctx.broker_t0 < ctx.args.joint_feed_warmup_s:
                _log_rl(
                    "jwarm",
                    "JFEED WARMUP: withholding the joint feed "
                    "(%.0f s of %.0f remain) -- establishment-phase "
                    "measurements must not become birth geometry"
                    % (
                        ctx.args.joint_feed_warmup_s - (time.time() - ctx.broker_t0),
                        ctx.args.joint_feed_warmup_s,
                    ),
                    every_s=60.0,
                )
            else:
                # MINNOV, the model innovation: for each established satellite (at least
                # joint_mask_after accepted updates), wrap(d - predicted) against the state
                # before this cycle's update, i.e. whether the joint state could have placed
                # the satellite without this detection. A birth row's prediction is its own
                # first measurement, hence established rows only. INNOV judges the commanded
                # seed, MINNOV the model; fleetdll publishes its p95, which the model-primacy
                # flip gate (instruments.instr_model_primacy) reads. Wrapped in the state's
                # reference chips, stored in ours.
                for _p3, _d3 in ctx.drp.offs:
                    _k3 = (ctx.drp.tag, _p3)
                    if (
                        _snr.get(_p3, 0.0) >= ctx.args.joint_min_snr
                        and _k3 in _js._idx
                        and _js._n.get(_k3, 0) >= ctx.args.joint_mask_after
                    ):
                        _mi = _js.wrap(_d3 * _k - _js.predicted(_k3)) / _k
                        _mh = ctx.minnov_hist.setdefault(_p3, [])
                        _mh.append((ctx.t0, _mi))
                        del _mh[:-120]
                _js.cycle(
                    [
                        ((ctx.drp.tag, p), d * _k, ctx.args.joint_sigma * _k)
                        for p, d in ctx.drp.offs
                        if _snr.get(p, 0.0) >= ctx.args.joint_min_snr
                        and ctx.track_ok(p)
                        and not ctx.p2c_hold(_js, (ctx.drp.tag, p))
                    ],
                    ctx.drp.t_now_abs,
                )
            # The filter has no logger: drain its notes (escapes, incoherent runs) into the log.
            for _n in _js.drain_notes():
                _log_rl("joint-note", "JOINT %s: %s" % (ctx.band_id, _n), every_s=10.0)
            ctx.p2c_tick(_js, ctx.drp.t_now_abs)
            for _p, _d in ctx.drp.offs:
                if ctx.p2c_hold(_js, (ctx.drp.tag, _p)):
                    # The residual, bias and sigma are read back in this chain's chips (the
                    # samples the P2C END summary averages are stated in them).
                    _r = _js.wrap(_d * _k - _js.predicted((ctx.drp.tag, _p))) / _k
                    if ctx.p2c["key"] == (ctx.drp.tag, _p):
                        ctx.p2c["samples"].append(
                            (ctx.drp.t_now_abs - ctx.p2c["t0"], _r)
                        )
                    _log_rl(
                        "p2c-%d" % _p,
                        "P2C %s PRN %d MASKED %.0fs: coast residual %+.3f chips "
                        "(b %+.3f, sigma %.3f; %.4f-Mcps chips) -- flat = the state "
                        "carries it"
                        % (
                            ctx.band_id,
                            _p,
                            _js.age((ctx.drp.tag, _p), ctx.drp.t_now_abs) or 0.0,
                            _r,
                            _js.bias((ctx.drp.tag, _p)) / _k,
                            _js.sigma((ctx.drp.tag, _p)) / _k,
                            ctx.args.chip_rate_hz / 1e6,
                        ),
                        every_s=30.0,
                    )
            if ctx.drp.now_w >= ctx.dr_state.get("joint_log_next", 0.0):
                ctx.dr_state["joint_log_next"] = ctx.drp.now_w + 30.0
                _amb = "  ⚠AMBIGUOUS(clk near wrap)" if ctx.rx.joint_ambiguous() else ""
                _log("JOINT[shadow] " + _js.summary(ctx.drp.t_now_abs) + _amb)
        except Exception as e:  # shadow must never take the broker down
            _log_rl(
                "jointerr", "JOINT[shadow] disabled this cycle: %s" % e, every_s=300.0
            )


def _joint_coast(ctx, ds, jrate, jlen):
    """The coast (dr_joint_clk's docstring): (delta in our chips, extrapolation, age, rate)
    from the stamps of the last adoption, or None when there is nothing to coast on: no
    adoption behind us, the window (--joint-clk-coast-max-s) closed or 0, or an
    extrapolation past joint_clk_max_chips, the bound an adoption itself may not exceed.

    The rate is the one stamped at adoption (jclk_last_rate), never the state's current one:
    the refusal that calls for a coast is the moment the state's rate is least trusted (a
    thin set walks it), and an unbounded live rate over 300 s could move the seeds by 6x
    the adoption bound. The level is built in joint chips, then converted and wrapped
    exactly as an adoption is (Receiver.joint_clk_delta)."""
    if not (ctx.args.joint_clk_coast_max_s > 0.0):
        return None
    last, t_last, rate = ds.get("jclk_last"), ds.get("jclk_last_t"), ds.get("jclk_last_rate")
    if last is None or t_last is None or rate is None:
        return None
    age = ctx.t0 - t_last
    if not (0.0 <= age <= ctx.args.joint_clk_coast_max_s):
        return None
    ext = rate * age
    if abs(ext) > ctx.args.joint_clk_max_chips:
        return None
    delta = ctx.rx.joint_clk_delta(
        (last + ext) % jlen, jrate, jlen, ctx.drp.clk_now, ctx.args.chip_rate_hz, ctx.code_len
    )[0]
    return delta, ext, age, rate


def dr_joint_clk(ctx):
    """The `joint-consume: clk` consumer: move this cycle's clk_now onto the joint clock, or
    leave it.

    The joint state is in reference chips (receiver.JOINT_REF_RATE_HZ, read back through
    Receiver.joint_unit), this chain's clock in its own; the two are one receiver time, so
    the joint is converted into our chips and the delta wrapped in the window both are
    defined in (Receiver.joint_clk_delta). Applied as that wrapped delta, so our clock keeps
    saying which period of a longer code we are in.

    Both bounds are in the joint's chips: joint_clk_max_chips and joint_clk_max_sigma mean the
    same time on every chain (5 chips = 489 ns at 10.23 Mcps), because what they guard (a
    wrap alias, a diverged filter, the legacy median's churn) is a clock error, not a count of
    any one chain's chips. The same 14.7 us is 150 joint chips, 75 E6 chips and 7.5 L2C-CM
    chips, so unconverted chips would refuse a correct joint clock on every chain whose chip
    rate differs from the feeders'.

    No tau(band) is added. The clock this replaces carries none (a same-band donor, or the
    cross-band bootstrap's converted donor), and the per-sat biases and DLL trims this chain
    runs on were built without it, so the band's delay already lives in the trims; adding
    tau would move every seed by it while the trims still hold it. tau is only defined where
    a satellite is fed in both bands, and a band that feeds nothing has no row (tau 0).

    Stateless for the clock: clk_now moves for this cycle only and dr_state["clk"] is
    untouched. It records when it adopted (dr_state["jclk_t"], the freshness
    _dr_clock_local_fresh reads) and, for the step guard and the coast, the joint clock it
    last adopted with the time it adopted it (dr_state["jclk_last"], joint chips, and
    dr_state["jclk_last_t"]) and the pending new level with the time it was first seen
    (dr_state["jclk_cand"]).

    A refused joint coasts on clk + clk_rate dt, never on the median (docs 6.2): when this
    chain adopted before and the adoption is younger than joint_clk_coast_max_s, clk_now
    goes onto jclk_last + jclk_last_rate (t0 - jclk_last_t), the rate stamped at that
    adoption (not the state's current one, which a thin set walks), bounded so the
    extrapolation never exceeds joint_clk_max_chips (_joint_coast); the rate in joint
    chips per second, applied as the same wrapped delta an adoption is. The legacy median
    churns by +-1-2 chips with its membership, and a refusal (a transit thinning the
    feeders, a sigma excursion, a level the step guard has not yet seen hold) is the moment
    the seeds would otherwise step back onto that churn. A coast never refreshes jclk_t,
    jclk_last or jclk_last_t, so it ages: past the window the refusal is a plain REFUSED on
    the legacy clock, the local clock goes stale and the cross-band bootstrap re-arms. A
    refusal with no adoption behind it is a REFUSED as well, and joint_clk_coast_max_s 0
    disables the coast.

    The step guard: a transit that blinds the feeder band thins the joint's satellite set and
    walks its clock for under a minute, inside the bounds. A joint clock more than
    joint_clk_max_step_chips from the one last adopted here is refused until it has held the
    new level, to the same tolerance, for joint_clk_step_hold_s; a bounds refusal in between
    restarts the hold. It tests the joint clock, not the delta: the legacy median's
    membership churn steps the delta while the joint stands still, and a hold on each of
    those would put the chain back on the churned clock the joint replaces. The receiver
    clock moves under 0.003 chips/s, so a walk never meets the guard and a real level change
    is a discontinuity, followed one hold late."""
    _jrC = ctx.joint_state(ctx.rx, ctx.band_id, ctx.args)
    if _jrC is None or len(_jrC._idx) < ctx.args.joint_min_sats:
        # Not fit to consume: thin, deaf, or absent. A transit that thins the feeders below
        # joint_min_sats is the case the coast exists for, so coast here too, on the stamps
        # of the last adoption, inside the same window and bound.
        _unit0 = ctx.rx.joint_unit()
        _c0 = _joint_coast(ctx, ctx.dr_state, _unit0[0], _unit0[1]) if _unit0 else None
        if _c0 is not None:
            _log_rl(
                "jclk",
                "JOINT-CLK: legacy %.3f chips, joint state not fit to consume (%s) -> COAST "
                "(%+.3f joint chips over %.0f s at clk_rate %+.4f joint chips/s)"
                % (
                    ctx.drp.clk_now,
                    "absent or deaf"
                    if _jrC is None
                    else "%d sats < %d" % (len(_jrC._idx), ctx.args.joint_min_sats),
                    _c0[1],
                    _c0[2],
                    _c0[3],
                ),
                every_s=30.0,
            )
            ctx.drp.clk_now = (ctx.drp.clk_now + _c0[0]) % ctx.code_len
        return
    # sigma <= 0 is degenerate (a state claiming perfect knowledge), not excellent: refuse it
    # explicitly. The state_filter P floor keeps sigma above 0, but this gate does not rely
    # on it.
    _jsigC = _jrC.sigma()
    if _jsigC is None or _jsigC <= 0.0:
        _jsigC = float("inf")
    _unit = ctx.rx.joint_unit()
    if _unit is None:
        return  # the receiver holds no joint state (ctx.joint_state handed us a stray one)
    _jrate, _jlen = _unit
    _r = ctx.args.chip_rate_hz / _jrate
    _jdC, _jdJ = ctx.rx.joint_clk_delta(
        _jrC.clk, _jrate, _jlen, ctx.drp.clk_now, ctx.args.chip_rate_hz, ctx.code_len
    )
    _jokC = (
        _jsigC <= ctx.args.joint_clk_max_sigma
        and abs(_jdJ) <= ctx.args.joint_clk_max_chips
    )
    # The delta is wrapped in the joint's window, so a clock off by whole periods of it reads
    # as a tiny delta and would be adopted every cycle, keeping this clock fresh and the
    # bootstrap that could fix it disarmed. When our code outlasts the joint's window, resolve
    # the joint across our period with the clock-mod-epoch record (the NH joint fit, published
    # only from an established search solve, at most 600 s old) and refuse while the two
    # disagree by more than half a joint window; the refusal lets the local clock go stale
    # and the cross-band bootstrap re-arm.
    _jwin_s = _jlen / _jrate
    _our_s = ctx.code_len / ctx.args.chip_rate_hz
    _jperiod_ms = None
    if _jokC and _our_s > 1.5 * _jwin_s:
        _jext = ctx.rx.clock_mod_epoch(
            min_epoch_s=_our_s,
            exclude=getattr(ctx, "chain_id", None),
            max_age_s=600.0,
            t_now=getattr(ctx, "t0", None),
        )
        if _jext is not None:
            _jt_sec, _jresid_s = ctx.rx.clock_extend_mod(
                (_jrC.clk / _jrate) % _jwin_s,
                _jwin_s,
                float(_jext.value),
                _jext.extra["epoch_s"],
                _our_s,
            )
            if abs(_jresid_s) <= 0.25 * _jwin_s:
                _jerr_s = (
                    (_jt_sec - ctx.drp.clk_now / ctx.args.chip_rate_hz + _our_s / 2.0)
                    % _our_s
                ) - _our_s / 2.0
                if abs(_jerr_s) > 0.5 * _jwin_s:
                    _jperiod_ms = (
                        _jerr_s * 1e3,
                        _jext.src,
                        _jext.extra["epoch_s"] * 1e3,
                    )
                    _jokC = False
    # The step guard, in joint chips within the joint's window (see the docstring).
    _ds = ctx.dr_state
    _jlvl = _jrC.clk % _jlen
    _jstep = None
    if _jokC and _ds.get("jclk_last") is not None:
        _jd = wrap(_jlvl - _ds["jclk_last"], _jlen)
        if abs(_jd) > ctx.args.joint_clk_max_step_chips:
            _jcand = _ds.get("jclk_cand")
            if (
                _jcand is None
                or abs(wrap(_jlvl - _jcand[0], _jlen))
                > ctx.args.joint_clk_max_step_chips
            ):
                _jcand = (_jlvl, ctx.t0)
            _ds["jclk_cand"] = _jcand
            if ctx.t0 - _jcand[1] < ctx.args.joint_clk_step_hold_s:
                _jstep = (_jd, ctx.t0 - _jcand[1])
                _jokC = False
    if _jstep is None:
        _ds.pop("jclk_cand", None)
    # The coast (see the docstring): a refusal with an adoption behind it, inside the
    # window, moves clk_now onto the last adopted joint clock carried forward at the joint's
    # rate. Built in joint chips, then converted and wrapped exactly as an adoption is.
    _jcoast = _joint_coast(ctx, _ds, _jrate, _jlen) if not _jokC else None
    # A coast the bound declined is named in the refusal: the stamps were there and the
    # window open, and the extrapolation alone put the seeds past what an adoption may.
    _jdecl = ""
    if not _jokC and _jcoast is None and _ds.get("jclk_last_rate") is not None:
        _jage0 = ctx.t0 - (_ds.get("jclk_last_t") or ctx.t0)
        if 0.0 <= _jage0 <= ctx.args.joint_clk_coast_max_s:
            _jext0 = _ds["jclk_last_rate"] * _jage0
            if abs(_jext0) > ctx.args.joint_clk_max_chips:
                _jdecl = (
                    "; coast declined: %+.1f joint chips of extrapolation over %.0f s is "
                    "past the %.1f-chip bound" % (_jext0, _jage0, ctx.args.joint_clk_max_chips)
                )
    if _jokC:
        _why = None
    elif _jperiod_ms is not None:
        _why = (
            "the clock-mod-%.0f-ms record from '%s' puts the joint %+.3f ms from the "
            "legacy clock: a whole-period error the %.0f-ms delta cannot see"
            % (_jperiod_ms[2], _jperiod_ms[1], _jperiod_ms[0], _jwin_s * 1e3)
        )
    elif _jstep is not None:
        _why = (
            "step %+.3f chips from the last adopted joint clock; the new level has held "
            "%.0f of %.0f s" % (_jstep[0], _jstep[1], ctx.args.joint_clk_step_hold_s)
        )
    else:
        _why = "bounds %.1f chips / %.2f sigma" % (
            ctx.args.joint_clk_max_chips,
            ctx.args.joint_clk_max_sigma,
        )
    # Fields before the bracket are in our chips; the bracket (other chip rates only) gives
    # the joint-chip numbers the bounds are tested in. A COAST's extrapolation and rate are
    # in joint chips too.
    _log_rl(
        "jclk",
        "JOINT-CLK: legacy %.3f joint %.3f chips (delta %+.3f,"
        " sigma %.3f, n %d)%s -> %s"
        % (
            ctx.drp.clk_now,
            (_jrC.clk * _r) % ctx.code_len,
            _jdC,
            _jsigC * _r,
            len(_jrC._idx),
            ""
            if _r == 1.0
            else " [x%.4f: joint %.3f delta %+.3f sigma %.3f in %.3f-Mcps chips, the"
            " bounds' unit]" % (_r, _jrC.clk, _jdJ, _jsigC, _jrate / 1e6),
            "ADOPTED"
            if _jokC
            else "COAST (refused: %s; %+.3f joint chips over %.0f s at clk_rate %+.4f "
            "joint chips/s)" % (_why, _jcoast[1], _jcoast[2], _jcoast[3])
            if _jcoast is not None
            else "REFUSED (%s%s)" % (_why, _jdecl),
        ),
        every_s=30.0,
    )
    if _jokC:
        ctx.drp.clk_now = (ctx.drp.clk_now + _jdC) % ctx.code_len
        _ds["jclk_t"] = ctx.t0
        _ds["jclk_last"] = _jlvl
        _ds["jclk_last_t"] = ctx.t0
        _ds["jclk_last_rate"] = _jrC.clk_rate  # the rate a coast runs on (_joint_coast)
    elif _jcoast is not None:
        # No stamp moves on a coast: it must age out, so the bootstrap can re-arm.
        ctx.drp.clk_now = (ctx.drp.clk_now + _jcoast[0]) % ctx.code_len


def dr_seed(ctx):
    """Birth, re-pin or slew the seed of every visible satellite this chain dead-reckons, and drop
    model-owned seeds that have set or whose satellite lacks the signal. Computes ctx.drp.clk_now.
    Does nothing until the receiver clock exists (a prime counts), and seeds nothing while
    stage_dead_reckon withholds seeding on the prime (drp.hold).

    Every term must name its epoch. The phase and its ref_hop label are built at the forecast epoch
    t_fc_abs, while the ephemeris was evaluated at wall time and cp_predicted propagates range from
    that epoch (t_eph_age); mixing the F-engine axis with wall time puts a per-satellite error of
    range rate x lag into the seed.

    A re-birth re-bases the seed while the C++ fleet trim still carries chips accumulated against
    the old seed, so the tap moves and the trim rebuilds; birth_steps lets D3 attribute the
    resulting trim wipe."""
    if ctx.dr_state["clk"] is not None:
        ctx.drp.clk_now = (
            ctx.dr_state["clk"]
            + ctx.drp.drift * (ctx.drp.now_w - ctx.dr_state["clk_t"])
        ) % ctx.code_len
        # On a chain with `joint-consume: clk`, dr_joint_clk moves clk_now (dr_state's clock
        # propagated at the drift) onto the joint clock for this pass only, unless its gates
        # refuse the joint clock.
        if "clk" in ctx.joint_consume:
            dr_joint_clk(ctx)
        # DRCLK is the clock the seeds consume, on every chain: the 'dead-reckon clock' line
        # (stage_dead_reckon) logs only on a chain with offsets of its own. Logged after the
        # joint-clock adoption, while JOINT-CLK keeps the legacy median visible. It is the clk_now
        # term of commanded = cp_predicted + clk_now + b_sat + trim (the BSAT and DLL lines carry
        # the other two).
        _log_rl(
            "drclk",
            "DRCLK clk_now %.3f chips drift %+.4f chips/s (la %+.4f ppm)"
            % (ctx.drp.clk_now, ctx.dr_state.get("drift") or 0.0, ctx.drp.la * 1e6),
            every_s=30.0,
        )
        # SEED-OFFSET summary: the joint offset against the legacy one (clk_now + b_sat) for every
        # satellite of this constellation in the joint state, every 30 s, outside the seeding loop.
        # The loop skips detected satellites (model-primary ones aside) and the joint state holds
        # only detected ones, so a comparison inside the loop would see almost none.
        _jr3 = ctx.joint_state(ctx.rx, ctx.band_id, ctx.args)
        # The joint is in reference chips (Receiver.joint_unit): at another chip rate the
        # comparison is made in ours, through the clock consumer's conversion and window.
        _u3 = ctx.rx.joint_unit() if _jr3 is not None else None
        if _u3 is not None and ctx.drp.now_w >= ctx.dr_state.get("jslew_log_next", 0.0):
            ctx.dr_state["jslew_log_next"] = ctx.drp.now_w + 30.0
            _r3 = ctx.args.chip_rate_hz / _u3[0]
            _cmp = []
            for (_ct, _p) in list(_jr3._idx):
                if _ct != ctx.drp.tag:
                    continue
                _lo = ctx.drp.clk_now + ctx.bsat.get(_p, ctx.drp.now_w)
                _jo = _jr3.predicted((_ct, _p))
                if _r3 == 1.0:
                    _dd = (
                        (_jo - _lo + ctx.drp.mod / 2.0) % ctx.drp.mod
                    ) - ctx.drp.mod / 2.0
                else:
                    _dd = ctx.rx.joint_clk_delta(
                        _jo, _u3[0], _u3[1], _lo, ctx.args.chip_rate_hz, ctx.code_len
                    )[0]
                _cmp.append((_p, _dd, (_jr3.sigma((_ct, _p)) or 0.0) * _r3))
            if _cmp:
                _cmp.sort(key=lambda x: -abs(x[1]))
                _log(
                    "SEED-OFFSET %s: joint-vs-legacy over %d sat(s), "
                    "median %+.3f chips | %s"
                    % (
                        ctx.band_id,
                        len(_cmp),
                        sorted(abs(c[1]) for c in _cmp)[len(_cmp) // 2],
                        " ".join("PRN%d %+.2f(s%.2f)" % c for c in _cmp[:6]),
                    )
                )
        for (ctag, prn), v in sorted(ctx.drp.pd.items()):
            # Signal capability first (--dr-min-prn, --signal-capability): the model predicts a code
            # phase for a signal the satellite does not broadcast, and seeding it would despread
            # noise.
            if prn < ctx.dr_min_prn:
                continue
            if ctx.capable is not None and prn not in ctx.capable:
                continue  # its block does not broadcast this signal
            # Seed above the mask + 0.5 deg, drop below the mask (the drop loop below): without the
            # hysteresis a satellite riding the mask flickers drop/reseed. A model-primary PRN stays
            # dead-reckon-owned while the search detects it, and is not skipped as untrusted:
            # dr_untrusted judges the legacy model (solved clock, no b_sat), while a flipped seed is
            # the joint model's and MINNOV referees it, so honouring the flag would leave it
            # seedless (its detections never re-anchor it). A flipped PRN is never in cp_held: the
            # flip's ENTER discards the hold.
            if (
                ctag != ctx.drp.tag
                or v["el"] < ctx.args.mask_deg + 0.5
                or (prn in ctx.best and prn not in ctx.mp_flipped)
                or prn in ctx.probe_set
                or prn in ctx.cp_held
                or (prn in ctx.dr_untrusted and prn not in ctx.mp_flipped)
            ):  # not this chain's to dead-reckon
                continue
            if prn in ctx.seeds and prn not in ctx.dr_state["seeded"]:
                continue  # search-anchored seed: not ours to touch
            # A locked dead-reckoned satellite on a chain with no search, or a model-primary PRN,
            # gets a slewed refresh rather than a hold or a re-pin. Held, its prompt drifts out of
            # the +-1-chip window in minutes, since the model's residual code rate is not in the
            # seed; re-pinned from clk_now, every refresh injects the clock's ~1-chip jitter as a
            # code-phase step. The slew keeps continuity and still follows the model (the slew block
            # below). Search-fed chains keep the plain hold: their DLL owns the code and their
            # search the Doppler.
            #
            # Locked means sig_of >= lock_snr, or _held, which does not depend on the deep fold:
            # fleet q >= lock_q (the metric lock is judged on elsewhere; 0 disables it) or prompt
            # power >= lock_prompt_hold x the noise probes' median. sig_of follows the fold's
            # certification, which flickers on a satellite the despread holds; judged unlocked, that
            # satellite is re-pinned, and the re-birth can step its seed outside the +-1-chip
            # correlation triangle, losing the lock the gate exists to protect.
            _q_locked = (
                ctx.args.lock_q > 0.0 and ctx.hold.q.get(prn, 0.0) >= ctx.args.lock_q
            )
            _held = (
                _q_locked or ctx.hold.prev.get(prn, 0.0) >= ctx.args.lock_prompt_hold
            )
            if _held and ctx.sig_of(ctx.status.get(prn, {})) < ctx.args.lock_snr:
                _log_rl(
                    "hold-prompt",
                    "HOLD-BY-PROMPT: PRN %d held through a deep-fold dropout "
                    "(prompt %.1fx noise, sig %.1f < %.1f) -- no re-pin"
                    % (
                        prn,
                        ctx.hold.prev.get(prn, 0.0),
                        ctx.sig_of(ctx.status.get(prn, {})),
                        ctx.args.lock_snr,
                    ),
                    every_s=60.0,
                )
            _slew = (
                prn in ctx.seeds
                and (not ctx.detectors or prn in ctx.mp_flipped)
                and prn in ctx.dr_state["seeded"]
                and (ctx.sig_of(ctx.status.get(prn, {})) >= ctx.args.lock_snr or _held)
            )
            if (
                not _slew
                and prn in ctx.seeds
                and (ctx.sig_of(ctx.status.get(prn, {})) >= ctx.args.lock_snr or _held)
            ):
                continue  # locked and not slewed: the DLL owns the residual
            if (
                not _slew
                and prn in ctx.dr_state["seeded"]
                and ctx.drp.now_w - ctx.dr_state["pin"].get(prn, 0.0)
                < ctx.args.dr_repin_s
            ):
                continue
            # Doppler and its rate come from the model's range rate (BRDC, or the decoded fallback),
            # never the TLE almanac, whose BeiDou PRN mapping is wrong for some satellites. The
            # clock-frequency bias is a receiver constant, common to every model.
            v2 = ctx.drp.pd2.get((ctag, prn))
            dop_geo = (
                -v["range_rate_mps"] / ctx.dr_eph_mod.C_LIGHT * ctx.args.carrier_hz
            )
            dop_seed = (
                ctx.args.doppler_sign * dop_geo + ctx.cb.seed
            )  # the seed-side bias, not the hint EMA
            # One epoch, the Doppler included: dop_geo is the model at now_w, but the seed is
            # labelled at the forecast hop (--dr-forecast-lead-s ahead), and the tracker propagates
            # the Doppler from ref_hop at doppler_rate, so an unforecast Doppler runs the replica
            # lead x drate off the sky for the whole seed. It is forecast to the label once drate is
            # known (below).
            #
            # drate is the centred difference of range rate between pd0 and pd2, 2 s either side of
            # now_w (a 4 s span); when pd0 lacks the satellite, the forward difference over the 2 s
            # from now_w to pd2. Each divisor must match its span: a halved drate leaks into the
            # seed as a ramp, and drate has one sign on every satellite (Doppler falls through a
            # pass), so the ramps add up to what looks like a constellation-common drift.
            drate = 0.0
            v0 = (ctx.dr_state.get("pd0") or {}).get((ctag, prn))
            if v2 is not None and v0 is not None:
                drate = (
                    ctx.args.doppler_sign
                    * (-(v2["range_rate_mps"] - v0["range_rate_mps"]) / 4.0)
                    / ctx.dr_eph_mod.C_LIGHT
                    * ctx.args.carrier_hz
                )
            elif v2 is not None:
                drate = (
                    ctx.args.doppler_sign
                    * (-(v2["range_rate_mps"] - v["range_rate_mps"]) / 2.0)
                    / ctx.dr_eph_mod.C_LIGHT
                    * ctx.args.carrier_hz
                )
            dop_seed += drate * (ctx.drp.t_fc_abs - ctx.drp.t_now_abs)
            # cp0, the seed currency: the model's physical code phase plus _off at the forecast
            # epoch, back-referenced to sample 0 (removing the nominal advance and the code-Doppler
            # drift: the inverse of cp_loc in stage_dead_reckon).
            #
            # _off = clk_now + b_sat. b_sat (SatBiasFilter, from the fleet tau fit) is the
            # per-satellite slow path bias: iono, tropo, broadcast orbit and clock. tau is sky minus
            # replica, so the replica moves by +b to meet the sky; b_sat is 0 until measured, and
            # once stale. _off is the one definition of what to add to the pure model for this
            # satellite, shared by cp0 (birth, re-pin) and the slew target _model so the two cannot
            # disagree. The joint `slew` consumer corrects it here, so the correction reaches births
            # and re-pins as well: only detector-less chains and model-primary PRNs slew, while the
            # joint state holds only satellites some chain detects.
            if ctx.drp.hold:
                continue  # withheld: the clock is still the prime (stage_dead_reckon)
            _leg_off = ctx.drp.clk_now + ctx.bsat.get(prn, ctx.drp.now_w)
            _off = _leg_off
            _jr3 = ctx.joint_state(ctx.rx, ctx.band_id, ctx.args)
            # The joint is in reference chips (Receiver.joint_unit), converted into ours below as
            # in dr_joint_clk.
            _u3 = ctx.rx.joint_unit() if _jr3 is not None else None
            _joff = (
                _jr3.predicted((ctx.drp.tag, prn))
                if (_u3 is not None and (ctx.drp.tag, prn) in _jr3._idx)
                else None
            )
            if _joff is not None:
                # Apply the joint offset as a wrapped delta on the legacy offset, never in its
                # place. The two agree only modulo the window Receiver.joint_clk_delta wraps in (the
                # shorter code), while the seed lives mod drp.mod (lc_seg code periods):
                # substituting _joff could move the seed by whole code periods, and a difference
                # wrapped at drp.mod would read a whole code period of disagreement on every
                # satellite whenever clk_now wraps, refusing the joint offset there. So the joint
                # contributes only the small correction it measures, and the legacy offset keeps
                # which long-code period the seed is in.
                #
                # At another chip rate the joint is converted into our chips; the bound is tested in
                # the joint's chips, the same time on every chain, as in dr_joint_clk. At the
                # joint's rate this is the same-rate arithmetic bit for bit, and the line carries no
                # bracket.
                _r3 = ctx.args.chip_rate_hz / _u3[0]
                _d3, _d3j = ctx.rx.joint_clk_delta(
                    _joff, _u3[0], _u3[1], _leg_off, ctx.args.chip_rate_hz, ctx.code_len
                )
                _ok3 = abs(_d3j) <= ctx.args.joint_slew_max_chips
                _s3 = _jr3.sigma((ctx.drp.tag, prn)) or 0.0
                _log_rl(
                    "jslew-%d" % prn,
                    "SEED-OFFSET PRN %d (%s): joint %+.3f vs legacy %+.3f "
                    "chips (diff %+.3f mod %g, sigma %.3f)%s%s"
                    % (
                        prn,
                        "slew" if _slew else "cp0",
                        _joff * _r3,
                        _leg_off,
                        _d3,
                        min(_u3[1] * _r3, ctx.code_len),
                        _s3 * _r3,
                        ""
                        if _r3 == 1.0
                        else " [x%.4f: diff %+.3f sigma %.3f in %.3f-Mcps chips, the"
                        " bound's unit]" % (_r3, _d3j, _s3, _u3[0] / 1e6),
                        ""
                        if _ok3
                        else "  REFUSED (> %.1f chips)" % ctx.args.joint_slew_max_chips,
                    ),
                    every_s=60.0,
                )
                if "slew" in ctx.joint_consume and _ok3:
                    _off = _leg_off + _d3
            cp0 = (
                (ctx.cp_predicted(v, ctx.drp.t_fc_abs) + _off)
                - ctx.drp.t_fc_abs
                * ctx.args.chip_rate_hz
                * (1.0 + ctx.args.code_doppler_sign * dop_seed / ctx.args.carrier_hz)
            ) % ctx.drp.mod
            if _slew:
                # _held: where the tracker's propagation puts this seed at hop h1 (dr_seed_phys, the
                # exact inverse of the seed currency plus the tracker's extrapolation; the selftest
                # checks the round trip). _model is the sum cp0 uses, before back-referencing.
                #
                # Both at t_h = h1/hps, the rounded hop's time, never the unrounded t_fc_abs: the
                # code advances ~52.4 chips per hop at 10.23 Mcps, so the sub-hop rounding alone
                # would put up to +-26 chips between them, common to every satellite. A birth may
                # build its phase at an unrounded t under the rounded ref_hop because the same t
                # builds the back-reference, so the skew cancels in the tracker to first order; a
                # difference of two epochs has no such cancellation.
                h1 = int(round(ctx.drp.t_fc_abs * ctx.args.hops_per_sec))
                t_h = h1 / ctx.args.hops_per_sec
                _held = dr_seed_phys(
                    ctx.seeds[prn],
                    h1,
                    ctx.args.hops_per_sec,
                    ctx.args.chip_rate_hz,
                    ctx.args.carrier_hz,
                    ctx.args.code_doppler_sign,
                    ctx.drp.mod,
                )
                # _off as computed above, so this target and the birth phase cannot disagree.
                _model = (ctx.cp_predicted(v, t_h) + _off) % ctx.drp.mod
                _dcp = (
                    (_model - _held + ctx.drp.mod / 2.0) % ctx.drp.mod
                ) - ctx.drp.mod / 2.0
                # The cap makes the slew a rate-limited integrator: a seed several chips from its
                # model closes at slew_cap per pass, and while the step sits on the cap it does not
                # depend on the target, so moving the target (the joint `slew` consumer) does not
                # show in the seed. The SLEW line prints model-held beside the step.
                _cap = ctx.drp.slew_cap
                _step = max(-_cap, min(_cap, ctx.drp.slew_k * _dcp))
                # Re-anchor at h1 with the fresh model Doppler and rate (removing the dt^2
                # linearization error a held seed accumulates) but at the held phase plus the
                # bounded step, never at the model's phase, which carries the clock's jitter. The
                # DLL trim stays: same trajectory, later epoch, so its residual still holds.
                ctx.seeds[prn] = Seed.born(
                    "dr_slew",
                    epoch=h1,
                    doppler_hz=dop_seed,
                    code_phase_chips=dr_cp0(
                        _held + _step,
                        t_h,
                        dop_seed,
                        ctx.args.chip_rate_hz,
                        ctx.args.carrier_hz,
                        ctx.args.code_doppler_sign,
                        ctx.drp.mod,
                    ),
                    code_phase_rate=cp_rate_from_code_bias(
                        ctx.drp.la, ctx.args.hops_per_sec, ctx.args.chip_rate_hz,
                    ),
                    ref_hop=h1,
                    doppler_rate_hz_s=drate,
                )
                # --seed-phase-transport: also ship the phase at ref_hop. propagate_seed prefers it,
                # and it has no sample-0 lever, so a later Doppler edit cannot desynchronise the
                # pair. code_phase_chips is still sent, for a tracker that ignores the field.
                if ctx.args.seed_phase_transport:
                    ctx.seeds[prn].put(
                        "phase_xport",
                        epoch=h1,
                        code_phase_at_ref_chips=seed_phase_at_ref(
                            _held + _step,
                            dop_seed,
                            ctx.args.chip_rate_hz,
                            ctx.args.hops_per_sec,
                            ctx.args.carrier_hz,
                            ctx.args.code_doppler_sign,
                            ctx.drp.mod,
                        ),
                    )
                ctx.dr_state["pin"][prn] = ctx.drp.now_w
                _log_rl(
                    "drslew-%d" % prn,
                    "dead-reckon SLEW PRN %d: model-held %+.3f chips, "
                    "step %+.3f (cap %.2f), dop %+.0f rate %+.2f"
                    % (prn, _dcp, _step, _cap, dop_seed, drate),
                    every_s=120.0,
                )
                continue
            if prn not in ctx.seeds:
                _log(
                    "dead-reckon SEED PRN %d (elev %.0f, cp0 %.1f, dop %+.0f,"
                    " rate %+.2f)" % (prn, v["el"], cp0, dop_seed, drate)
                )
            ctx.dls.trim.pop(prn, None)  # that trim was built against the old anchor
            ctx.dls.last.pop(prn, None)
            _rh_birth = int(round(ctx.drp.t_fc_abs * ctx.args.hops_per_sec))
            _pv = ctx.seeds.get(prn)
            if _pv is not None and "ref_hop" in _pv:
                # A re-birth: D3 calls a trim wipe within its rebase window of this stamp a
                # REBASE-WIPE, otherwise a BARE-WIPE (slew transfer or other).
                ctx.birth_steps[prn] = ctx.t0
            ctx.seeds[prn] = Seed.born(
                "dr_birth",
                epoch=_rh_birth,
                doppler_hz=dop_seed,
                code_phase_chips=cp0,
                code_phase_rate=cp_rate_from_code_bias(
                    ctx.drp.la, ctx.args.hops_per_sec, ctx.args.chip_rate_hz,
                ),
                ref_hop=_rh_birth,
                doppler_rate_hz_s=drate,
            )
            # Birth and re-pin arm: cp0 was built from this phase (cp_predicted + _off at t_fc_abs),
            # so shipping it costs nothing.
            if ctx.args.seed_phase_transport:
                ctx.seeds[prn].put(
                    "phase_xport",
                    epoch=_rh_birth,
                    code_phase_at_ref_chips=seed_phase_at_ref(
                        (ctx.cp_predicted(v, ctx.drp.t_fc_abs) + _off) % ctx.drp.mod,
                        dop_seed,
                        ctx.args.chip_rate_hz,
                        ctx.args.hops_per_sec,
                        ctx.args.carrier_hz,
                        ctx.args.code_doppler_sign,
                        ctx.drp.mod,
                    ),
                )
            ctx.dr_state["seeded"].add(prn)
            ctx.dr_state["pin"][prn] = ctx.drp.now_w
        # Model-owned seeds drop here, on the model's elevation (stage_coast_drop exempts them from
        # the TLE up-set) or on signal capability.
        for prn in list(ctx.dr_state["seeded"]):
            v = ctx.drp.pd.get((ctx.drp.tag, prn))
            if prn < ctx.dr_min_prn or (
                ctx.capable is not None and prn not in ctx.capable
            ):
                _log("dead-reckon drop PRN %d (does not broadcast this signal)" % prn)
                ctx.seeds.pop(prn, None)
                ctx.hold.low_hits.pop(prn, None)
            elif v is None or v["el"] < ctx.args.mask_deg:
                _log("dead-reckon drop PRN %d (set below BRDC horizon)" % prn)
                ctx.seeds.pop(prn, None)
                ctx.hold.low_hits.pop(prn, None)


# The seeding ephemeris reloads every 15 min, on the sky's cadence: a broadcast record is valid for
# 4 h from its toe, and the hourly station merge adds records late (BeiDou's more than an hour after
# toe), so on a slow reload an in-view satellite's newest record can expire first and its seed and
# transit-sky row vanish until the next reload, a whole constellation at once. fetch_brdc returns
# the cached merge and refreshes it in the background, so a reload is a parse. The DCB table reloads
# every 2 h: it is a daily product, and fetch_dcb's walk back to the newest day costs an HTTPS 404
# for every day younger than the product's latency.
_DR_EPH_REFRESH_S = 900.0
_DR_DCB_REFRESH_S = 7200.0


def _dr_reload(ctx, first, with_dcb=True):
    """Fetch and parse the dead-reckon ephemeris (and, when `with_dcb`, the DCB table): the slow
    half of a reload. It touches no ctx state, so it can run on the dr-reload thread (every reload
    but the first). Returns the result dict _dr_apply_reload takes; `first` (no ephemeris yet) makes
    fetch_brdc block on the network."""
    res = {
        "eph": None,
        "dcb": None,
        "has_dcb": False,
        "error": None,
        "t0": time.time(),
    }
    try:
        res["eph"] = ctx.dr_eph_mod.parse_rinex_nav(
            ctx.dr_eph_mod.fetch_brdc(block=first)
        )
        # Measured code biases (DCB), on their own slower cadence (_DR_DCB_REFRESH_S): a daily
        # product with ~5 days of latency, biases stable over weeks. Optional: with no token or no
        # network dcb is None and group_delay_s uses the broadcast term.
        if ctx.args.dcb_bias and with_dcb:
            try:
                import gnss_dcb as _dcbm

                _st = {}
                _p = _dcbm.fetch_dcb(status=_st)
                _t = _dcbm.parse_dcb(_p)
                res["dcb"], res["has_dcb"] = (_t or None), True
                # Warn on the product's age, not only its absence: once the credential stops
                # working, fetch_dcb serves the newest cached file for up to 14 days, so absence is
                # the late symptom (dcb_max_age_days).
                _age = _st.get("age_days")
                _texp = _st.get("token_expiry_days")
                _why = None
                if not _t:
                    _why = {
                        "no-token": "no Earthdata token configured "
                        "(~/.cache/kotekan_gps/.earthdata_token)",
                        "auth-rejected": "the Earthdata token was REJECTED "
                        "(HTTP %s) -- it has expired or been "
                        "revoked" % _st.get("http"),
                        "unreachable": "no product reachable and none cached "
                        "in the walk-back window",
                    }.get(_st.get("reason"), "no product (%s)" % _st.get("reason"))
                elif (
                    ctx.args.dcb_max_age_days > 0.0
                    and _age is not None
                    and _age > ctx.args.dcb_max_age_days
                ):
                    _why = (
                        "the product in use is %.1f days old (bar %.1f) -- the "
                        "fetch has stopped working and the cache is carrying "
                        "it" % (_age, ctx.args.dcb_max_age_days)
                    )
                if _why:
                    _log(
                        "⚠️ DCB: %s. Falling back to the BROADCAST group delay "
                        "per satellite -- the per-sat bias spread is gone, and "
                        "for BeiDou B2a there is no broadcast term at all. %s"
                        % (
                            _why,
                            "Token expires in %.1f days." % _texp
                            if _texp is not None
                            else "Token expiry unknown (not a JWT).",
                        )
                    )
                else:
                    _n = sum(1 for k in _t if k[0] == ctx.args.dr_constellation)
                    _log(
                        "dead-reckon: DCB loaded (%s, product %s; %d sats this "
                        "constellation) -- measured code biases override the "
                        "broadcast TGD/BGD per satellite%s"
                        % (
                            os.path.basename(_p or "?"),
                            "%.1f d old" % _age if _age is not None else "age ?",
                            _n,
                            ""
                            if _texp is None or _texp > 14.0
                            else "; ⚠️ TOKEN EXPIRES IN %.1f DAYS" % _texp,
                        )
                    )
            except Exception as _de:
                res["dcb"], res["has_dcb"] = None, True
                _log("dead-reckon: DCB load failed (%s); broadcast term only" % _de)
    except Exception as e:
        res["error"] = str(e)
    return res


def _dr_apply_reload(ctx, res):
    """Swap a finished reload into dr_state, from stage_dead_reckon before the predict. A failed
    reload keeps the current ephemeris and schedules the next attempt 10 min out (the first load,
    with no ephemeris yet, is retried every pass instead).

    The reload runs off the pass because the parse and the DCB walk-back take seconds, and a pass
    stalled that long holds up the telemetry fold: windows older than the telemetry ring's depth are
    evicted unfolded, which breaks the fleet-ADR arcs. Nothing here re-seeds: dr_seed reads the new
    model the same way it read the old one."""
    if res["error"]:
        _log("dead-reckon: BRDC unavailable (%s); retry in 10 min" % res["error"])
        ctx.dr_state["eph_t"] = ctx.drp.now_w - _DR_EPH_REFRESH_S + 600.0
        return
    ctx.dr_state["eph"] = res["eph"]
    ctx.dr_state["eph_t"] = ctx.drp.now_w
    ctx.dr_state["t0m"] = ctx.dr_eph_mod.gpst_of_utc(ctx.utc0_sample0) % ctx.drp.t_code
    if res["has_dcb"]:
        ctx.dr_state["dcb"] = res["dcb"]
        ctx.dr_state["dcb_t"] = ctx.drp.now_w
    _log(
        "dead-reckon: BRDC loaded (%d sats) -- %.1f s of fetch+parse, off the pass"
        % (len(res["eph"]), time.time() - res["t0"])
    )


def stage_dead_reckon(ctx):
    """Dead-reckoned seeding: refresh the ephemeris, predict every satellite, build this chain's
    per-satellite code offsets, then run dr_joint_shadow, dr_clock_solve, dr_clock_quality,
    dr_clock_adopt_rx, the model-health gates and dr_seed, in that order. The working set lives on
    ctx.drp, so each step's inputs are named fields.

    On a chain with no search this is the acquisition path: it seeds satellites that have never been
    detected, from the broadcast ephemeris and the receiver clock alone.

    ctx.drp.t_now_abs is None, never 0.0, until a model is first available: a missing axis time is
    unknown, and every consumer must check for None."""
    if (
        ctx.dr_state is not None
        and ctx.args.almanac
        and ctx.pred
        and ctx.utc0_sample0
        and _now() >= ctx.dr_state["next"]
    ):
        ctx.drp.now_w = _now()
        ctx.dr_state["next"] = ctx.drp.now_w + ctx.args.dr_refresh_s
        # Withhold seeding while the clock is still the --dr-clock-chips prime: a seed born without
        # a clock steps by the whole clock at its first re-birth after the bootstrap, and every
        # satellite steps at once. The wait is short where a solve or an adoption is coming, but a
        # detector-less chain with no donor never gets one (the prime exists for it), so after
        # dr_clock_wait_s seeding proceeds on the prime and logs that once.
        ctx.drp.hold = False
        if ctx.dr_state.get("clk_primed"):
            _waited = ctx.drp.now_w - ctx.dr_state.get("clk_t", ctx.drp.now_w)
            if _waited < ctx.args.dr_clock_wait_s:
                _log_rl(
                    "clkwait",
                    "dead-reckon: WITHHOLDING seeds -- the clock is still "
                    "the %.2f-chip PRIME, not a measurement (%.0f of %.0f s"
                    " waited). Seeding now would anchor every cp0 without "
                    "a clock and step the whole fleet by ~%.0f chips at the"
                    " first re-birth after BOOTSTRAP."
                    % (
                        ctx.dr_state.get("clk") or 0.0,
                        _waited,
                        ctx.args.dr_clock_wait_s,
                        abs(ctx.dr_state.get("clk") or 0.0) or 150.0,
                    ),
                    every_s=10.0,
                )
                ctx.dr_state["next"] = ctx.drp.now_w + min(2.0, ctx.args.dr_refresh_s)
                ctx.drp.hold = True
            # elif: the warning must fire only once the wait has expired, never in the same pass as
            # a withhold.
            elif not ctx.dr_state.get("clk_wait_warned"):
                ctx.dr_state["clk_wait_warned"] = True
                _log(
                    "dead-reckon: ⚠️ seeding on the %.2f-chip PRIME after waiting %.0f s "
                    "-- no clock measurement arrived. Expected on a DETECTOR-LESS chain "
                    "with no sibling to adopt from; on a chain that HAS detectors it "
                    "means the solve is not running, and every seed below is anchored on "
                    "a guess."
                    % (
                        ctx.dr_state.get("clk") or 0.0,
                        ctx.drp.now_w - ctx.dr_state.get("clk_t", ctx.drp.now_w),
                    )
                )
        # The model works modulo the full code the tracker despreads, lc_seg primary periods, not
        # one: a chain with no blind search gets the secondary-code segment only from this seed, and
        # a one-period reduction would put every seed in segment 0. The reduction keeps the numbers
        # small for double precision; placing the segment needs absolute time to half a primary
        # period, which the GPS-disciplined F-engine anchor and the BRDC range and clock give with
        # orders of magnitude to spare.
        ctx.drp.t_code = (ctx.lc_seg * ctx.code_len) / ctx.args.chip_rate_hz
        # The seed is reduced at the same length as the prediction.
        ctx.drp.mod = ctx.lc_seg * ctx.code_len
        # Slew constants for dr_seed: each pass moves a slewed seed slew_k of the way to the model,
        # at most dr_slew_cap chips. At 0.05 chips per 2 s pass that is 0.025 chips/s of authority,
        # above the 0.003-0.02 chips/s code drift seen on sky and an order below the 0.25-chip DLL
        # trims the deep fold tolerates; slew_k low-passes the clock jitter the target carries.
        ctx.drp.slew_cap = ctx.args.dr_slew_cap
        ctx.drp.slew_k = 0.25
        _rl = ctx.dr_state.setdefault("reload", {"thread": None, "done": None})
        if ctx.dr_state["eph"] is None:
            # The first load is synchronous: with no ephemeris there is nothing to seed from. Every
            # later reload runs on the dr-reload thread.
            _dr_apply_reload(ctx, _dr_reload(ctx, first=True))
        elif (
            ctx.drp.now_w - ctx.dr_state["eph_t"] > _DR_EPH_REFRESH_S
            and _rl["thread"] is None
        ):
            _with_dcb = (
                ctx.drp.now_w - ctx.dr_state.get("dcb_t", float("-inf"))
                > _DR_DCB_REFRESH_S
            )

            def _run(_rl=_rl, _with_dcb=_with_dcb):
                try:
                    _rl["done"] = _dr_reload(ctx, first=False, with_dcb=_with_dcb)
                except Exception as _e:  # _dr_reload catches its own; this catches the rest
                    _rl["done"] = {
                        "eph": None,
                        "dcb": None,
                        "has_dcb": False,
                        "error": str(_e),
                        "t0": time.time(),
                    }

            _rl["thread"] = threading.Thread(
                target=_run, name="dr-reload-%s" % ctx.chain_id, daemon=True
            )
            _rl["thread"].start()
        if _rl["done"] is not None:
            _res, _rl["done"], _rl["thread"] = _rl["done"], None, None
            _dr_apply_reload(ctx, _res)
        # Decoded-ephemeris fallback: predict from this broker's own decoded ephemeris while no BRDC
        # ephemeris is loaded.
        _use_decoded = (
            ctx.decfb is not None
            and ctx.args.decoded_eph_fallback
            and not ctx.dr_state["eph"]
        )
        if ctx.dr_state["eph"] or _use_decoded:
            ctx.drp.tag = ctx.args.dr_constellation
            ctx.drp.t_now_abs = ctx.drp.now_w - ctx.utc0_sample0
            # --dr-fengine-axis: "now" from the F-engine hop counter rather than wall minus the
            # anchor, so NTP's absolute offset never enters and wall time contributes only elapsed
            # seconds. Every label stamped here (h1, the birth ref_hop) and every phase evaluated
            # then lives on the axis the tracker runs. The static difference from the wall axis is
            # common-mode and lands in the solved receiver clock.
            if ctx.args.dr_fengine_axis and ctx.fe_axis[0] is not None:
                # The filtered offset (fe_off) when there is one: the raw newest hop re-samples the
                # pipeline lag every poll, and that jitter would land in every re-birth as lag x
                # range rate.
                if ctx.fe_off[0] is not None:
                    ctx.drp.t_now_abs = ctx.fe_off[0] + ctx.drp.now_w
                else:
                    _feh, _few = ctx.fe_axis[0]
                    ctx.drp.t_now_abs = _feh / ctx.args.hops_per_sec + (
                        ctx.drp.now_w - _few
                    )
            # The forecast epoch (--dr-forecast-lead-s). A seed is a forecast with a label (ref_hop,
            # phase, Doppler, rate), defined entirely on the hop axis, so it is built for the whole
            # hop H nearest t_now_abs + lead rather than for "now". The pipeline lag then never
            # enters the arithmetic; it only sets how large the lead must be (more than transport
            # plus install, or the seed lands after the epoch it describes). The phase and its label
            # must be built from the same t (see h1 in dr_seed). Ages, update_rrate and the joint
            # cycle stay on t_now_abs: they are measurements at now, and moving them forward would
            # inflate every age by the lead.
            ctx.drp.t_fc_abs = ctx.drp.t_now_abs
            if ctx.args.dr_forecast_lead_s > 0.0 and ctx.fe_axis[0] is not None:
                # From the filtered now (t_now_abs), not the raw newest hop. H is a whole hop, so
                # the label carries no rounding of its own.
                _fch = int(
                    round(
                        (ctx.drp.t_now_abs + ctx.args.dr_forecast_lead_s)
                        * ctx.args.hops_per_sec
                    )
                )
                ctx.drp.t_fc_abs = _fch / ctx.args.hops_per_sec
            # The orbit is evaluated at wall time (predict_all / predict_from_decoders at now_w),
            # never on the F-engine axis. It is the one consumer here that needs absolute UTC to be
            # true rather than self-consistent, but the observable F-engine "now" is the newest
            # telemetry pow_hop, which trails the sky by the gather/serve latency (~100 ms, tens of
            # ms of jitter). On that axis the orbit epoch would trade ~1.5 ms of NTP error for
            # ~100 ms of lag: chips of seed error at L5, scaling with range rate. The axis suits
            # labels (a label and its phase share a t and the tracker propagates from the label, so
            # the staleness cancels), not an epoch, where the staleness is the error. Below, the two
            # axes are compared as a tripwire (log only): a wall-clock step would move every model
            # seed on every chain at once.
            if ctx.fe_axis[0] is not None and ctx.utc0_sample0:
                # Unpacked here: the axis block above runs only under --dr-fengine-axis, and this
                # tripwire runs either way.
                _axh, _axw = ctx.fe_axis[0]
                _dax = (ctx.utc0_sample0 + _axh / ctx.args.hops_per_sec) - _axw
                _dprev = ctx.dr_state.get("ax_off")
                if (
                    _dprev is not None
                    and abs(_dax - _dprev) > ctx.args.clock_step_guard_s
                ):
                    _log(
                        "*** WALL-vs-F-ENGINE OFFSET JUMPED %+.3f s (%.3f -> %.3f, "
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
                        % (
                            _dax - _dprev,
                            _dprev,
                            _dax,
                            ctx.args.clock_step_guard_s,
                            abs(_dax - _dprev) * 800.0 / 29.3,
                        )
                    )
                ctx.dr_state["ax_off"] = _dax
            ctx.drp.la = ctx.cb.code_ema
            # A chain with no detections cannot measure (l-a): code_ema fills only from its own
            # cp-fit pool. (l-a) is a property of the receiver clock, so take another in-process
            # chain's from the Receiver, same band first. Without it every dead-reckon seed ships
            # code_phase_rate 0 and the prompt walks out of the +-1-chip correlation window in
            # minutes. A rate cannot step the seed.
            if ctx.drp.la is None:
                _sh_cb = ctx.rx.code_bias(
                    ctx.band_id, exclude=ctx.chain_id, t_now=ctx.drp.now_w
                )
                _sh_band = ctx.band_id
                # Fall back across bands: otherwise a band with no chain that measures its own (l-a)
                # never gets one. A rate needs no band: (l-a) is a fractional frequency, and the
                # per-band group delay tau_band is a constant that contributes nothing to it
                # (Receiver.code_bias_any_band). A phase is different: the cross-band clock
                # bootstrap (dr_clock_adopt_rx) carries tau_band. The log says CROSS-BAND when the
                # rate is borrowed from another band.
                if _sh_cb is None:
                    _sh_cb = ctx.rx.code_bias_any_band(
                        exclude=ctx.chain_id, t_now=ctx.drp.now_w
                    )
                    _sh_band = "cross-band"
                if _sh_cb is not None:
                    ctx.drp.la = float(_sh_cb.value)
                    _log_rl(
                        "la-adopt",
                        "dead-reckon: (l-a) %+.4f ppm ADOPTED from in-process chain "
                        "'%s' (%s %s) -> seeds carry the code-clock rate"
                        % (
                            ctx.drp.la * 1e6,
                            _sh_cb.src,
                            "same band"
                            if _sh_band == ctx.band_id
                            else "CROSS-BAND, rate only;",
                            _sh_band
                            if _sh_band == ctx.band_id
                            else "phase stays per-band",
                        ),
                        every_s=300.0,
                    )
                else:
                    ctx.drp.la = 0.0
            # Clock drift (chips/s): the measured drift (dr_clock_quality's, from consecutive raw
            # solves, or a donor's, adopted with its clock in dr_clock_adopt_rx), else the
            # f_chip*(l-a) model. A measured drift expires after dr_drift_max_age_s: raw solves
            # refresh it, and a wrong drift prevents them (it walks the seeds off peak), so a value
            # that has stood long is more likely unmeasurable than steady.
            _dmax = ctx.args.dr_drift_max_age_s
            _dt = ctx.dr_state.get("drift_t")
            if (
                ctx.dr_state.get("drift") is not None
                and _dt is not None
                and ctx.drp.now_w - _dt > _dmax
            ):
                _log(
                    "dead-reckon: clock drift %+.4f chips/s last MEASURED %.0f s ago -- "
                    "EXPIRED (--dr-drift-max-age-s %.0f); back to the (l-a) model %+.4f "
                    "until a fresh pair of solves"
                    % (
                        ctx.dr_state["drift"],
                        ctx.drp.now_w - _dt,
                        _dmax,
                        ctx.args.chip_rate_hz * ctx.drp.la,
                    )
                )
                ctx.dr_state["drift"] = None
                ctx.dr_state.pop("drift_t", None)
            elif ctx.dr_state.get("drift") is not None and _dt is None:
                # an unstamped drift gets one full lifetime from now, not forever
                ctx.dr_state["drift_t"] = ctx.drp.now_w
            ctx.drp.drift = ctx.dr_state.get("drift")
            if ctx.drp.drift is None:
                ctx.drp.drift = ctx.args.chip_rate_hz * ctx.drp.la
            try:
                # Three epochs, now_w and now_w +- 2 s: the centred range-rate difference is the
                # Doppler rate dr_seed ships (the TLE almanac's rate is not used here).
                if _use_decoded:
                    _ents = ctx.decoded_entries(ctx.drp.now_w)
                    ctx.drp.pd = ctx.decfb.predict_from_decoders(
                        _ents,
                        ctx.args.lat,
                        ctx.args.lon,
                        ctx.args.alt,
                        datetime.fromtimestamp(ctx.drp.now_w, tz=timezone.utc),
                        mask_deg=-90.0,
                    )
                    # Centred pair, +-2 s about now_w, as in the BRDC branch.
                    ctx.drp.pd2 = ctx.decfb.predict_from_decoders(
                        _ents,
                        ctx.args.lat,
                        ctx.args.lon,
                        ctx.args.alt,
                        datetime.fromtimestamp(ctx.drp.now_w + 2.0, tz=timezone.utc),
                        mask_deg=-90.0,
                    )
                    pd0 = ctx.decfb.predict_from_decoders(
                        _ents,
                        ctx.args.lat,
                        ctx.args.lon,
                        ctx.args.alt,
                        datetime.fromtimestamp(ctx.drp.now_w - 2.0, tz=timezone.utc),
                        mask_deg=-90.0,
                    )
                    if _now() - ctx.decfb_log_t[0] > 60.0:
                        ctx.decfb_log_t[0] = _now()
                        _log(
                            "dead-reckon: predicting from DECODED eph (BRDC network "
                            "DOWN -> fallback; %d entries -> %d sats)"
                            % (len(_ents), len(ctx.drp.pd))
                        )
                else:
                    # signal= refers sat_clk_s to this chain's code instead of the constellation's
                    # broadcast clock reference (GPS L1/L2, Galileo E1/E5a or E1/E5b, BeiDou B3I),
                    # using the DCB table's biases when loaded (gnss_ephemeris.group_delay_s).
                    # cp_predicted consumes that clock, so it moves the seed directly.
                    ctx.drp.pd = ctx.dr_eph_mod.predict_all(
                        ctx.dr_state["eph"],
                        ctx.args.lat,
                        ctx.args.lon,
                        ctx.args.alt,
                        datetime.fromtimestamp(ctx.drp.now_w, tz=timezone.utc),
                        mask_deg=-90.0,
                        signal=ctx.args.signal,
                        dcb=ctx.dr_state.get("dcb"),
                    )
                    # Centred pair, +-2 s about now_w: a forward difference over [now, now+4] would
                    # estimate the rate at now+2 and hand it to the seed as the rate at now, and the
                    # centred form has the smaller truncation error.
                    ctx.drp.pd2 = ctx.dr_eph_mod.predict_all(
                        ctx.dr_state["eph"],
                        ctx.args.lat,
                        ctx.args.lon,
                        ctx.args.alt,
                        datetime.fromtimestamp(ctx.drp.now_w + 2.0, tz=timezone.utc),
                        mask_deg=-90.0,
                        signal=ctx.args.signal,
                        dcb=ctx.dr_state.get("dcb"),
                    )
                    pd0 = ctx.dr_eph_mod.predict_all(
                        ctx.dr_state["eph"],
                        ctx.args.lat,
                        ctx.args.lon,
                        ctx.args.alt,
                        datetime.fromtimestamp(ctx.drp.now_w - 2.0, tz=timezone.utc),
                        mask_deg=-90.0,
                        signal=ctx.args.signal,
                        dcb=ctx.dr_state.get("dcb"),
                    )
            except Exception as e:
                ctx.drp.pd, ctx.drp.pd2, pd0 = {}, {}, {}
                _log("dead-reckon: predict failed: %s" % e)
            # The geometry-only sky (sky.transit_sky) for the transit freezes and the D3 transit
            # veto: health gate off, a day of toe validity, every constellation. Built from the same
            # ephemeris as the seeds' sky and kept separate from it on purpose.
            try:
                if ctx.dr_state.get("eph"):
                    ctx.drp.transit_pd = transit_sky(
                        ctx.dr_eph_mod,
                        ctx.dr_state["eph"],
                        ctx.args.lat,
                        ctx.args.lon,
                        ctx.args.alt,
                        datetime.fromtimestamp(ctx.drp.now_w, tz=timezone.utc),
                    )
                    ctx.dr_state["transit_pd"] = ctx.drp.transit_pd
            except Exception as e:
                _log(
                    "dead-reckon: transit sky failed (%s); the freezes keep the last one"
                    % e
                )
            if ctx.drp.pd:
                # Cached for the next cycle's seed loop (seeding.py), which takes the BRDC Doppler
                # and rate for search-anchored satellites from it, so both seed paths use one model;
                # dr_seed reads pd0 from here this cycle.
                ctx.dr_state["pd"], ctx.dr_state["pd2"] = ctx.drp.pd, ctx.drp.pd2
                ctx.dr_state["pd0"] = pd0

            # The ephemeris's own epoch, in capture age: predict_all ran at now_w (wall), so its
            # rows hold at now_w - utc0_sample0. That is t_now_abs on the wall axis, but under
            # --dr-fengine-axis t_now_abs trails wall by the residual telemetry lag. cp_predicted
            # propagates range from this epoch: propagating from t_now_abs would misplace each
            # satellite by range_rate x lag, a per-satellite error proportional to Doppler of which
            # the clock solve absorbs only the common part.
            ctx.drp.t_eph_age = ctx.drp.now_w - ctx.utc0_sample0

            # Receiver-clock offsets and integrity residuals: the physical code phase at each
            # detection's hop (the sample-0 back-reference undone), minus the model, normalized to
            # t_now_abs at the drift. dr_clock_solve takes their circular median.
            ctx.drp.offs = []
            # Detection age per satellite, logged beside each integrity residual. A detection
            # latches for up to the 1276 s per-PRN revisit and the normalization ages it by drift x
            # age, so drift error grows with age: residuals that grow with age measure the
            # normalization, not the sky.
            det_age = {}
            off_inputs = {}
            for prn, (snr, dop, cp, ref_hop, _nh, _cpl, _car) in sorted(
                ctx.best.items()
            ):
                v = ctx.drp.pd.get((ctx.drp.tag, prn))
                if v is None:
                    continue
                t_i = ref_hop / ctx.args.hops_per_sec
                det_age[prn] = ctx.drp.t_now_abs - t_i
                # The search's sample-0 back-reference subtracts both the nominal code advance and
                # the code-Doppler drift; add both back. Without the nominal term, detections from
                # different scans scatter by their ref_hop's place in the code period.
                #
                # The solve uses this reconstruction, not the detection's cp_at_ref: that is
                # referenced at the hop's last sample and carries the replica anchor's Doppler term,
                # so using it would import the search's fft_len and anchor geometry. The (cp0, dop)
                # pair is published together and its Doppler embed cancels exactly here.
                cp_loc = (
                    cp
                    + t_i
                    * ctx.args.chip_rate_hz
                    * (1.0 + ctx.args.code_doppler_sign * dop / ctx.args.carrier_hz)
                ) % ctx.code_len
                d_i = (
                    cp_loc
                    - ctx.cp_predicted(v, t_i)
                    + ctx.drp.drift * (ctx.drp.t_now_abs - t_i)
                ) % ctx.code_len
                ctx.drp.offs.append((prn, d_i))
                # WHAT MOVED records cp_loc, not raw cp: the search's cp0 embeds
                # -t_abs*chip_rate*sign*dop/carrier, so raw cp swings by thousands of chips with the
                # detection Doppler's pass-to-pass jitter, while the embed cancels exactly in
                # cp_loc.
                off_inputs[prn] = (cp_loc, t_i, dop)
            # Erratic-track guard (split_erratic_offsets): d_i = clk + b_i, both stable, so a real
            # satellite's offset moves far less than dr_max_off_jump_chips between cycles, while a
            # noise or cross-correlation track jumps by thousands of chips and drags the median.
            # Dropping that one satellite keeps the solve, and leaves the MAD refusal in
            # dr_clock_solve as the last resort.
            if ctx.drp.offs:
                _keep, _drop = split_erratic_offsets(
                    ctx.drp.offs,
                    ctx.dr_state.setdefault("off_hist", {}),
                    ctx.drp.now_w,
                    ctx.args.dr_max_off_jump_chips,
                    ctx.args.dr_off_jump_max_age_s,
                    ctx.code_len,
                )
                if _drop:
                    _prevI = ctx.dr_state.setdefault("off_inputs_prev", {})
                    _det = []
                    for _p, _j in _drop:
                        # Not `_now`: that is the imported clock function, and assigning the name
                        # anywhere in this function would make every _now() call in it an unbound
                        # local.
                        _cur = off_inputs.get(_p)
                        _was = _prevI.get(_p)
                        if _cur and _was:
                            _dcl = (
                                _cur[0] - _was[0] + ctx.code_len / 2.0
                            ) % ctx.code_len - ctx.code_len / 2.0
                            _det.append(
                                "PRN %d: dcp_loc %+.1f  dt_i %+.3fs  "
                                "ddop %+.3fHz"
                                % (_p, _dcl, _cur[1] - _was[1], _cur[2] - _was[2])
                            )
                    if _det:
                        _log_rl(
                            "offjumpwhy",
                            "clock solve: WHAT MOVED -- " + " | ".join(_det),
                            every_s=60.0,
                        )
                ctx.dr_state["off_inputs_prev"] = dict(off_inputs)
                if _drop and len(_keep) >= ctx.args.dr_min_sats:
                    ctx.drp.offs = _keep
                    _log_rl(
                        "offjump",
                        "clock solve: EXCLUDED %s -- offset jumped %s chips since "
                        "the last cycle (bound %.0f). d_i = clk + b_i, both "
                        "stable; the detection-Doppler embed cancels exactly in "
                        "cp_loc (2026-08-11), so a jump is a real discontinuity "
                        "in what the search reported, the model, or the drift "
                        "normalization -- see WHAT MOVED"
                        % (
                            ", ".join("PRN %d" % p for p, _ in _drop),
                            ", ".join("%.0f" % j for _, j in _drop),
                            ctx.args.dr_max_off_jump_chips,
                        ),
                        every_s=60.0,
                    )
                elif _drop:
                    # Dropping them would leave fewer than dr_min_sats: keep all and say so. If
                    # every satellite jumped, the clock or the model moved, not one track, and the
                    # MAD refusal in dr_clock_solve is the net for that.
                    _log_rl(
                        "offjumpkeep",
                        "clock solve: %d PRN(s) jumped but excluding them leaves "
                        "%d < --dr-min-sats %d -- keeping all; if this persists the "
                        "CLOCK moved, not one track"
                        % (len(_drop), len(_keep), ctx.args.dr_min_sats),
                        every_s=60.0,
                    )
            ctx.dr_state[
                "offs_t"
            ] = ctx.drp.now_w  # freshness stamp of the offsets built this cycle
            # offs is the measurement: d_i = clk + b_i, the search's physical code phase minus the
            # pure model, with no clock removed. dr_clock_solve takes its circular median as the
            # clock; the joint state (dr_joint_shadow) estimates clk and every b_i together.

            # Drain the joint state's notes (escapes, the DEAF latch) on every chain, before
            # dr_joint_shadow, which drains them only when this chain has offsets of its own.
            if ctx.args.rrate_state or ctx.args.joint_shadow:
                try:
                    _jsn = ctx.rx.joint_receiver(
                        ctx.band_id, ctx.code_len, ctx.args.chip_rate_hz, gauge_mode="prior"
                    )
                    for _n in _jsn.drain_notes():
                        _log_rl(
                            "joint-note",
                            "JOINT %s: %s" % (ctx.band_id, _n),
                            every_s=10.0,
                        )
                except Exception:
                    pass
            dr_joint_shadow(ctx)
            dr_clock_solve(ctx)
            dr_clock_quality(ctx)
            dr_clock_adopt_rx(ctx)
            # Model-health gates. A fence on Doppler motion cannot see a model that is simply wrong
            # (stale ephemeris, manoeuvre, bad orbit); these can: toe age over dr_max_eph_age_s, and
            # the integrity residual (search code phase against the model, solved clock removed,
            # normally +-0.2 chips) over dr_max_integrity_chips. A satellite is demoted to
            # dr_untrusted after 3 consecutive failures and restored once its residual is back
            # inside half the bar; nothing is judged until the clock and a measured drift exist,
            # since residuals sit near the bar while the clock converges. An untrusted satellite is
            # not dead-reckoned (model-primary PRNs aside) and its seed Doppler leaves the BRDC
            # model (seeding.py; --dr-doppler-ignores-integrity keeps it for an integrity demotion).
            if (
                ctx.dr_state["clk"] is not None
                and ctx.dr_state.get("drift") is not None
            ):
                ctx.dr_state.setdefault("integ", {})
                for prn_i, d_i in ctx.drp.offs:
                    r_i = (
                        (d_i - ctx.dr_state["clk"] + ctx.code_len / 2.0) % ctx.code_len
                    ) - ctx.code_len / 2.0
                    # Read by the escape referee's integrity veto, dr_solve_healthy and the
                    # publisher (dr_integ_chips): chips of this chain's code, search against the
                    # model, solved clock removed.
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
                        ctx.dr_bad[prn_i] = 0  # hysteresis: reset well inside the bar
                    if (
                        ctx.dr_bad.get(prn_i, 0) >= 3
                    ) and prn_i not in ctx.dr_untrusted:
                        ctx.dr_untrusted[prn_i] = why
                        # The log names the Doppler the seed loop (seeding.py) falls back to: the
                        # almanac predictor's, or the search's under --seed-doppler det. Under
                        # --dr-doppler-ignores-integrity an integrity demotion keeps the BRDC
                        # Doppler instead, which this line does not reflect (seeding.py logs it).
                        _log(
                            "MODEL-UNTRUSTED PRN %d (%s, 3 consecutive) -> Doppler source "
                            "falls back dr -> %s for this sat"
                            % (
                                prn_i,
                                why,
                                "pred (almanac)"
                                if ctx.args.seed_doppler != "det"
                                else "det",
                            )
                        )
                    elif ctx.dr_bad.get(prn_i, 0) == 0 and prn_i in ctx.dr_untrusted:
                        del ctx.dr_untrusted[prn_i]
                        _log(
                            "MODEL-TRUSTED again PRN %d (integrity %+.2f chips)"
                            % (prn_i, r_i)
                        )
            if (
                ctx.dr_state["clk"] is not None
                and ctx.drp.offs
                and ctx.drp.now_w >= ctx.dr_state["log_next"]
            ):
                ctx.dr_state["log_next"] = ctx.drp.now_w + 30.0
                resid = [
                    "PRN %d %+.2f a%.0f%s"
                    % (p, r, det_age.get(p, -1), " BAD" if abs(r) > 1.0 else "")
                    for p, d in ctx.drp.offs
                    for r in [
                        ((d - ctx.dr_state["clk"] + ctx.code_len / 2) % ctx.code_len)
                        - ctx.code_len / 2
                    ]
                ]
                # The median is the level only where the joint clock is not consumed; the
                # parenthesis says which, so a reader does not take it for the seeds' clock.
                _log(
                    "dead-reckon clock %.2f chips (median: %s; %.3f us mod %.0f ms, drift "
                    "%+.3f chips/s); integrity: %s"
                    % (
                        ctx.dr_state["clk"],
                        "a cross-check, the adopted joint clock is the level"
                        if "clk" in ctx.joint_consume
                        else "the level on this chain",
                        ctx.dr_state["clk"] / ctx.args.chip_rate_hz * 1e6,
                        ctx.drp.t_code * 1e3,
                        ctx.dr_state.get("drift") or 0.0,
                        "; ".join(resid),
                    )
                )
            # Birth, re-pin or slew every visible dead-reckoned satellite's seed.
            dr_seed(ctx)
        # Hand a satellite back to the search once it is detected (the seed loop re-anchors it) or
        # its seed is gone. A model-primary PRN stays model-owned while detected: its detections
        # feed the joint state and MINNOV, never the seed.
        for prn in list(ctx.dr_state["seeded"]):
            if (prn in ctx.best and prn not in ctx.mp_flipped) or prn not in ctx.seeds:
                ctx.dr_state["seeded"].discard(prn)
                ctx.dr_state["pin"].pop(prn, None)
