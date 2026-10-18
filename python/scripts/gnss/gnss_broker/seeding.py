"""Detections into seeds, and the coast/drop decision that retires them.

`stage_detections_to_seeds` turns each PRN's best detection this cycle (snr, Doppler, code
phase, ref_hop) into the seed the trackers consume; `stage_coast_drop` coasts a seed through a
fade and drops it when the coast runs out or its satellite sets; `stage_push_seeds` posts the
seeds to the trackers.

An alias-bin detection is harmless; do not "fix" it. The search's Doppler is ambiguous mod
1/(2*t_rec), 25 Hz on 20 ms records. The search back-projects cp0 to sample 0 with the same
reported Doppler, and `cp_to_seed_currency` adds that projection back with the same numbers,
so the round trip is exact whatever bin the Doppler rode. Replacing the Doppler (folding it
into another bin) before the currency conversion breaks that cancellation, by
t_abs*f_chip/f_carrier chips per Hz of change.

code_phase_chips is an argument, not a transportable quantity: it means something only
against the Doppler and epoch it was measured at (the lever is t_abs*f_chip/f_carrier chips
per Hz, thousands of chips at a multi-day t_abs). Never transport or difference one; only
`cp_at_ref` is comparable.

Dead-reckoned (model-owned) seeds are exempt from the almanac up-set: the dead-reckon stage
drops them on its own BRDC elevation, and the up-set can disagree with it (the TLE almanac
mismaps some BeiDou PRNs). Dropping a satellite the model still tracks leaves the chain to
re-acquire it from nothing.

@author Keith Vanderlinde
"""

import math
import statistics

from gnss_broker.sky import C_LIGHT
from gnss_broker.transport import _log, _log_rl
from gnss_broker.clockhealth import dr_solve_healthy
from gnss_broker.seed import Seed
from gnss_broker.fits import (
    retag_seed_doppler,
    code_clock_bias_sample,
    fit_cp_rate,
    fit_dop_rate,
    tracker_phase_at,
    cp_rate_from_code_bias,
)


def _present_streak(ctx, prn):
    """Trailing run of "present" cycles for PRN in ctx.qpop, the population-honest series.

    This is the per-cycle presence the fleet controller admits trims on, not the `DLL:` log
    line (survivors only) and not amp_snr (the coherent arc). Returns 0 when the series has
    never seen the PRN.
    """
    hist = ctx.qpop.hist.get(prn)  # qpop is always set on the context, so no getattr:
    if not hist:  # a missing qpop must raise, not run inert
        return 0
    n = 0
    for _t, _q, state in reversed(hist):
        if state != "present":
            break
        n += 1
    return n


def _nh_joint_pred_chips(ctx, prn, t_utc):
    """Predicted phase (chips, mod the overlay epoch) of PRN's code at receiver wall time
    t_utc, transmit-aligned: gpst(t) - tau + sat_clock. There is deliberately no receiver-clock
    term: it is the one common unknown the votes resolve. Same convention as the search's nh
    hint (searchhint.py); the broadcast satellite clock matters because an overlay segment is
    one code period (1 ms at L5)."""
    import gnss_ephemeris as _eph

    v = ctx.pred[prn]
    t_tx = _eph.gpst_of_utc(t_utc) - v[3] / C_LIGHT + v[4]
    return (t_tx % ctx.lc_epoch) * ctx.args.chip_rate_hz


def _nh_joint_vote(ctx, prn, ph_ref, ref_hop, snr, now):
    """Record one detection's vote: the receiver-common overlay offset implied by its phase at
    its own epoch. delta = (measured - predicted) mod LLc is the receiver clock in chips mod
    the overlay epoch, the same number for every satellite when the model is right, so the
    per-satellite spread in the NH-JOINT log checks the model. Only phases at their own epoch
    vote: a period carried through cp0 rides the t_abs Doppler lever and is noise."""
    t_ref = ctx.utc0_sample0 + ref_hop / ctx.args.hops_per_sec
    LLc = ctx.lc_seg * ctx.code_len
    delta = (ph_ref - _nh_joint_pred_chips(ctx, prn, t_ref)) % LLc
    ctx.cpt.nh_votes[prn] = (now, delta, min(float(snr), 200.0))
    return delta


def _nh_joint_consensus(ctx, now):
    """Resolve (or keep) the common offset from the standing votes: a circular weighted median
    over chips mod LLc. Continuous, not an integer vote: the sub-ms part of the receiver clock
    is common too, and an integer vote splits when the offset sits near a segment boundary.
    A resolved consensus is a run constant: it is kept when votes thin out and only replaced by
    a newer resolution, never expired."""
    a = ctx.args
    LLc = ctx.lc_seg * ctx.code_len
    votes = ctx.cpt.nh_votes
    for p in [p for p, (t, _, _) in votes.items() if now - t > a.nh_joint_window_s]:
        votes.pop(p)
    if len(votes) < a.nh_joint_min_prns:
        return ctx.cpt.nh_common
    # circular mean as the pivot, then a weighted median of the wrapped deviations
    two_pi = 2.0 * math.pi
    cs = sum(w * math.cos(two_pi * d / LLc) for _, d, w in votes.values())
    sn = sum(w * math.sin(two_pi * d / LLc) for _, d, w in votes.values())
    if cs == 0.0 and sn == 0.0:
        return ctx.cpt.nh_common
    mu = (math.atan2(sn, cs) / two_pi * LLc) % LLc
    devs = sorted(
        (((d - mu + LLc / 2) % LLc) - LLc / 2, w, p) for p, (_, d, w) in votes.items()
    )
    half = 0.5 * sum(w for _, w, _ in devs)
    acc = 0.0
    med = devs[-1][0]
    for dv, w, _ in devs:
        acc += w
        if acc >= half:
            med = dv
            break
    dstar = (mu + med) % LLc
    inliers = [
        (p, ((d - dstar + LLc / 2) % LLc) - LLc / 2, w)
        for p, (_, d, w) in votes.items()
    ]
    inl = [(p, dv, w) for p, dv, w in inliers if abs(dv) <= a.nh_joint_tol_chips]
    w_tot = sum(w for _, _, w in inliers)
    w_in = sum(w for _, _, w in inl)
    spread = max((abs(dv) for _, dv, _ in inl), default=0.0)
    _log_rl(
        "nhjoint",
        "NH-JOINT: delta %+.1f chips (%.3f ms, seg %d) from %d/%d sats "
        "(weight %.0f/%.0f), max inlier dev %.1f chips"
        % (
            dstar,
            dstar / a.chip_rate_hz * 1e3,
            int(round(dstar / ctx.code_len)) % ctx.lc_seg,
            len(inl),
            len(votes),
            w_in,
            w_tot,
            spread,
        ),
        every_s=60.0,
    )
    if len(inl) >= a.nh_joint_min_prns and w_in >= 0.6 * w_tot:
        prev = ctx.cpt.nh_common
        moved = (
            prev is None
            or abs(((dstar - prev[0] + LLc / 2) % LLc) - LLc / 2) > 0.5 * ctx.code_len
        )
        if moved:
            _log(
                "NH-JOINT: common overlay offset %s %+.1f chips (%.3f ms) from %d sats%s"
                % (
                    "RESOLVED" if prev is None else "MOVED to",
                    dstar,
                    dstar / a.chip_rate_hz * 1e3,
                    len(inl),
                    "" if prev is None else " -- was %+.1f" % prev[0],
                )
            )
        ctx.cpt.nh_common = (dstar, now, len(inl), (w_in / w_tot) if w_tot else 0.0)
        # Publish the resolved clock mod the overlay epoch to the Receiver, as a time: a chain whose
        # code period is longer than its donor's window (gps_l2c's 20 ms against gps_l5's 1 ms)
        # needs it to bootstrap from the donor's clock. Published only while this chain's search
        # solve is established (dr_solve_healthy not False): votes from noise detections can walk
        # the record, and a consumer cannot tell a walked record from a measured one. The last
        # published record stands; consumers bound its age.
        if getattr(ctx, "rx", None) is not None:
            if dr_solve_healthy(getattr(ctx, "dr_state", None), now) is not False:
                ctx.rx.contribute_clock_mod_epoch(
                    ctx.chain_id,
                    dstar / a.chip_rate_hz,
                    ctx.lc_epoch,
                    len(inl),
                    now,
                    healthy=True,
                )
            else:
                _log_rl(
                    "nhjoint-unhealthy",
                    "NH-JOINT: clock-mod-epoch record NOT published -- this chain's search "
                    "solve is not established (fewer than 2 fresh integrity residuals inside "
                    "a chip, or one outside); consumers keep the last published one",
                    every_s=300.0,
                )
    return ctx.cpt.nh_common


def _nh_joint_snap(ctx, prn, ph_ref, ref_hop, dstar):
    """Snap a measured phase's overlay segment to the consensus, keeping its fine phase: the
    segment comes from (prediction + common offset), the sub-chip phase from the measurement.
    Returns (snapped_ph, k_segments_moved)."""
    t_ref = ctx.utc0_sample0 + ref_hop / ctx.args.hops_per_sec
    LLc = ctx.lc_seg * ctx.code_len
    want = (_nh_joint_pred_chips(ctx, prn, t_ref) + dstar) % LLc
    k = int(round((((want - ph_ref + LLc / 2) % LLc) - LLc / 2) / ctx.code_len))
    return (ph_ref + k * ctx.code_len) % LLc, k


def stage_coast_drop(ctx):
    """COAST / DROP: drop seeds whose satellite has set, and coast the ones merely fading.

    A visible seed that was not re-detected keeps its anchor and has its Doppler forecast
    forward; it is dropped once its lock metric has stayed below the bar for the coast budget.
    Noise probes are left alone, and dead-reckoned seeds are exempt from both drops (see the
    module docstring)."""
    for prn in list(ctx.seeds):
        if prn in ctx.probe_set:
            continue
        if (
            ctx.up is not None
            and prn not in ctx.up
            and not (ctx.dr_state is not None and prn in ctx.dr_state["seeded"])
        ):
            _log("drop PRN %d (set below horizon)" % prn)
            del ctx.seeds[prn]
            ctx.cp_held.discard(prn)
            ctx.hold.miss.pop(prn, None)
            ctx.hold.low_hits.pop(prn, None)
            continue
        if (
            prn in ctx.best
        ):  # re-detected: stage_detections_to_seeds re-anchored it and reset the coast
            continue
        # not re-detected this poll but still visible -> coast: forecast the Doppler forward.
        if ctx.dr_state is not None and prn in ctx.dr_state["seeded"]:
            # Dead-reckoned seed: its Doppler is frozen between re-pins, each of which refreshes
            # Doppler, code phase and rate together; the almanac forecast must not touch it.
            pass
        elif ctx.args.almanac and prn in ctx.pred:
            # The seed's Doppler is valid at ref_hop, not at now: propagate_seed applies
            #     doppler_hz + doppler_rate_hz_s * (t - t(ref_hop))
            # so the forecast is stored back-propagated to ref_hop, and the tracker rebuilds it
            # exactly at now for any age. Storing the now-value beside the old ref_hop would add
            # rate*age on top, running the applied Doppler at ~2x the true rate until the next
            # detection re-anchors ref_hop. For the same reason the code-phase retag compares the
            # old and new Doppler both at now, so a standing rate cancels. With no sample-0 anchor
            # (utc0_sample0 unset) age is 0.
            _rate_new = ctx.pred[prn][1]
            # The rate the tracker propagates with: none without the key, so nothing to undo.
            _rate_eff = _rate_new if "doppler_rate_hz_s" in ctx.seeds[prn] else 0.0
            _age = 0.0
            if ctx.utc0_sample0:
                _age = max(
                    0.0,
                    (ctx.drp.now_w - ctx.utc0_sample0)
                    - ctx.seeds[prn].get("ref_hop", 0) / ctx.args.hops_per_sec,
                )
            new_dop = (
                ctx.pred[prn][0] + ctx.cb.seed
            )  # the forecast at now, with the seed-side clock bias
            _old_rate = ctx.seeds[prn].get("doppler_rate_hz_s", 0.0)
            # the Doppler the tracker is applying now
            old_dop = ctx.seeds[prn].get("doppler_hz", new_dop) + _old_rate * _age
            if new_dop != old_dop:
                # Re-express cp0 in the new Doppler's currency at every forecast update: cp0 means
                # something only paired with its Doppler, and changing the Doppler alone walks the
                # despread by t_abs*f_chip*ddop/f_carrier chips. retag_seed_doppler preserves the
                # phase at now; without a sample-0 anchor (utc0_sample0 unset) it works at the
                # seed's ref_hop epoch instead, a partial correction.
                _t_retag = (
                    (ctx.drp.now_w - ctx.utc0_sample0)
                    if ctx.utc0_sample0
                    else ctx.seeds[prn].get("ref_hop", 0) / ctx.args.hops_per_sec
                )
                ctx.seeds[prn].put(
                    "coast_retag",
                    epoch=ctx.seeds[prn].get("ref_hop"),
                    code_phase_chips=retag_seed_doppler(
                        ctx.seeds[prn].get("code_phase_chips", 0.0),
                        old_dop,
                        new_dop,
                        _t_retag,
                        ctx.args.chip_rate_hz,
                        ctx.args.carrier_hz,
                        ctx.args.code_doppler_sign,
                        ctx.code_len,
                    ),
                    # stored at ref_hop, not at now (see above)
                    doppler_hz=new_dop - _rate_eff * _age,
                )
            if "doppler_rate_hz_s" in ctx.seeds[prn]:
                ctx.seeds[prn].put(
                    "coast_retag",
                    epoch=ctx.seeds[prn].get("ref_hop"),
                    doppler_rate_hz_s=_rate_new,
                )
        rec = ctx.status.get(prn, {})
        if ctx.have_sig:
            metric, thresh = ctx.sig_of(rec), ctx.args.lock_snr
        else:
            metric, thresh = float(rec.get("amplitude", 0.0)), ctx.args.drop_amplitude
        # Fold-independent lock (lock_prompt_hold), OR-ed against its own bar and never max()-ed
        # into `metric`: the prompt ratio is a power ratio, `metric` a debiased sigma (or |A|).
        if metric >= thresh or ctx.hold.prev.get(prn, 0.0) >= ctx.args.lock_prompt_hold:
            ctx.hold.low_hits[
                prn
            ] = 0  # lock holding through the dropout -> reset coast
        else:
            ctx.hold.low_hits[prn] = ctx.hold.low_hits.get(prn, 0) + 1
            # Dead-reckoned seeds are model-owned and never coast out: they exist for satellites
            # with no signal above the search threshold.
            if ctx.hold.low_hits[prn] >= ctx.coast_polls and not (
                ctx.dr_state is not None and prn in ctx.dr_state["seeded"]
            ):
                _log(
                    "drop PRN %d (coast %.0fs expired, %s=%.2f)"
                    % (
                        prn,
                        ctx.args.coast_budget,
                        "sig" if ctx.have_sig else "|A|",
                        metric,
                    )
                )
                del ctx.seeds[prn]
                ctx.hold.low_hits.pop(prn, None)


def stage_detections_to_seeds(ctx):
    """1: DETECTIONS -> SEEDS. Turn this cycle's best detection per PRN into a tracker seed.

    Converts the search's (snr, Doppler, code phase, ref_hop) into the seed the trackers
    consume, via cp_to_seed_currency, refines it from the per-PRN fits, and applies the
    hold/escape policy that decides whether the detection may move a standing seed. An
    alias-bin detection is harmless (see the module docstring)."""
    for prn, (snr, dop, cp, ref_hop, det_nh, cp_long, cp_at_ref) in ctx.best.items():
        v_dr = ctx.dr_pd.get((ctx.args.dr_constellation, prn)) if ctx.dr_pd else None
        if ctx.up is not None and prn not in ctx.up:
            # Accept it anyway if the dead-reckon BRDC sky has it above the mask: the almanac up-set
            # can disagree (the TLE almanac mismaps some BeiDou PRNs).
            if v_dr is None or v_dr["el"] < ctx.args.mask_deg:
                continue
        # The search is the admission authority for the trim presence gate
        # (--dll-deep-gate-from-search). Stamped after the visibility filter, so a spurious
        # below-horizon detection cannot arm a correction, and before every seeding-policy filter
        # below: eligibility asks whether the satellite is up and detectable, not whether this cycle
        # re-seeds from the detection.
        if (
            ctx.args.dll_deep_gate_from_search > 0.0
            and snr >= ctx.args.dll_deep_gate_from_search
        ):
            ctx.dls.deep_gate_seen[prn] = ctx.t0
        _dop_src = "pred" if (ctx.args.almanac and prn in ctx.pred) else "DET(grid)"
        seed_dop = (
            (ctx.pred[prn][0] + ctx.cb.seed)
            if (ctx.args.almanac and prn in ctx.pred)
            else dop
        )
        # With dead-reckoning, prefer the BRDC Doppler for every seed: the same model owns the
        # undetected satellites, and mixing sources would step the seed Doppler by the
        # almanac-vs-dead-reckon difference at every handoff between the two.
        #
        # dr_untrusted has two unlike reasons: a stale ephemeris (doubtful for range and range rate
        # alike) or a code-phase integrity residual over dr_max_integrity_chips (1 chip, ~30 m at
        # 10.23 Mcps: ordinary iono plus b_sat, which says nothing about range rate). Under
        # --dr-doppler-ignores-integrity only the first demotes the Doppler, so a satellite flipping
        # trust does not switch its seed, which is the replica's carrier phase, between two Doppler
        # models.
        _unt = ctx.dr_untrusted.get(prn)
        _dop_trusted = _unt is None or (
            ctx.args.dr_doppler_ignores_integrity
            and not str(_unt).startswith("ephemeris")
        )
        if v_dr is not None and _dop_trusted:
            _dop_src = "dr" if _unt is None else "dr(code-untrusted)"
            seed_dop = (
                ctx.args.doppler_sign
                * (-v_dr["range_rate_mps"] / C_LIGHT * ctx.args.carrier_hz)
                + ctx.cb.seed
            )
            if _unt is not None:
                _log_rl(
                    "dopkeep-%d" % prn,
                    "PRN %d: code model untrusted (%s) but KEEPING the BRDC Doppler "
                    "-- an integrity residual is a code-phase statement, and "
                    "switching the seed switches the replica's carrier phase"
                    % (prn, _unt),
                    every_s=120.0,
                )
        # --seed-doppler det: the measured Doppler has the last word, over the model and the
        # dead-reckon value. Those exist to keep the seed smooth and to own undetected satellites;
        # for a measured satellite a model error rides the seed's whole extrapolation age at
        # f_chip/f_carrier chips per Hz per second.
        if ctx.args.seed_doppler == "det":
            _dop_src = "det"
            seed_dop = dop
        # Seed Doppler step tripwire: a step over 10 Hz from the satellite's previous seed is logged
        # with its source; a step of almost exactly one search-grid Doppler step points at a grid or
        # quantization slip upstream.
        _prev_sd = ctx.seeds.get(prn, {}).get("doppler_hz")
        if _prev_sd is None and ctx.args.almanac and not ctx.cb.available:
            # No first seed until this chain's clock-frequency bias is solved (cb.available: a bias
            # EMA from its own multi-satellite solve). Under --seed-bias-source ema a first seed
            # sent earlier carries the receiver's whole frequency offset, and a strong satellite can
            # lock and enter the hold freeze in that wrong currency within seconds. The guard stays
            # in every mode: cb.available is false on a chain's first cycle, which runs before the
            # dead-reckon stage stamps drp.now_w, and a first seed then would raise in
            # _nh_joint_consensus (test_seed_bias.py pins this). Detections do not need seeds, and
            # the bias solves from the same detections.
            continue
        if _prev_sd is None:
            # First seed: log it in full; the step tripwire has no previous value to compare.
            _log(
                "PRN %d FIRST SEED dop %.1f (src=%s, det=%.1f, pred=%s, bias %+.1f)"
                % (
                    prn,
                    seed_dop,
                    _dop_src,
                    dop,
                    ("%.1f" % (ctx.pred[prn][0]))
                    if (ctx.args.almanac and prn in ctx.pred)
                    else "n/a",
                    ctx.cb.seed,
                )
            )
        elif abs(seed_dop - _prev_sd) > 10.0 and prn in ctx.cp_held:
            # Held satellite: the candidate Doppler walks while the emitted seed stays frozen or
            # translated, so this step is never applied as is, and logging it would flood.
            pass
        elif abs(seed_dop - _prev_sd) > 10.0:
            _log(
                "PRN %d SEED DOP STEP %+.1f Hz (%.1f -> %.1f, src=%s, det=%.1f)"
                % (prn, seed_dop - _prev_sd, _prev_sd, seed_dop, _dop_src, dop)
            )

        # Doppler-rate history. The seed's doppler_rate_hz_s drives both the tracker's carrier NCO
        # and the quadratic code term, so its error counts twice. The BRDC rate is a numerical
        # derivative (range rate over a 4 s epoch pair), weakest near zenith where the curvature
        # peaks; the Doppler is measured every pass, so its slope is fitted instead (fit_dop_rate
        # below). Both histories keep only distinct snapshots (the search serves a detection until
        # it updates it), and a gap beyond max_gap_hops is a re-acquisition that drops the old
        # history.
        dh_ = ctx.cpt.dop_hist.get(prn, [])
        if dh_ and (ref_hop - dh_[-1][0]) > ctx.max_gap_hops:
            dh_ = []
        if not dh_ or ref_hop != dh_[-1][0]:
            dh_.append((ref_hop, dop))
            dh_ = dh_[-ctx.hist_len :]
        ctx.cpt.dop_hist[prn] = dh_

        h = ctx.cpt.hist.get(prn, [])
        if h and (ref_hop - h[-1][0]) > ctx.max_gap_hops:
            h = []  # gap too large -> re-acquisition, old slope is stale
        # Only detections at or above --fit-min-snr enter the cp history: the fit resolves a slope
        # of ~0.01 chips/s, and a near-threshold detection's phase is noise, so one such point
        # destroys the fit rather than degrading it.
        if not h or ref_hop != h[-1][0]:
            if snr >= ctx.args.fit_min_snr:
                h.append((ref_hop, cp, dop))
                h = h[-ctx.hist_len :]
            elif h:
                _log_rl(
                    "fitsnr-%d" % prn,
                    "PRN %d cp-fit: skipping snr %.0f point (< --fit-min-snr %.0f); "
                    "%d in history" % (prn, snr, ctx.args.fit_min_snr, len(h)),
                    every_s=120.0,
                )
        ctx.cpt.hist[prn] = h

        # The detection's cp is in its own Doppler's currency; the tracker despreads at seed_dop.
        # Convert, or a satellite first acquired at t_abs starts off peak by
        #     t_abs*f_chip*(dop - seed_dop)/f_carrier chips.
        ((_, cp_seed_cur),) = ctx.cp_to_seed_currency([(ref_hop, cp, dop)], seed_dop)
        seed = Seed.born(
            "det",
            epoch=ref_hop,
            doppler_hz=seed_dop,
            code_phase_chips=cp_seed_cur,
            code_phase_rate=0.0,
            ref_hop=ref_hop,
        )
        # Re-attribute the Doppler to the source chosen above (_dop_src): same value, provenance
        # only, so SEEDAUDIT's owner trail separates a model-seeded Doppler from a measured one.
        seed.put("dop_sel:" + _dop_src, epoch=ref_hop, doppler_hz=seed_dop)
        # Doppler-rate feed-forward (Hz/s, sign applied like doppler_hz). The tracker integrates it
        # in its NCO and the quadratic code term, never by retuning the replica, which would walk
        # the absolutely anchored code and carrier off peak.
        v2_dr = ctx.dr_pd2.get((ctx.args.dr_constellation, prn)) if ctx.dr_pd2 else None
        v0_dr = ctx.dr_pd0.get((ctx.args.dr_constellation, prn)) if ctx.dr_pd0 else None
        if v2_dr is not None and v0_dr is not None:
            # BRDC Doppler rate, central difference over the pd0/pd2 pair at now_w -/+ 2 s, so the
            # rate is tagged at now_w.
            seed.put(
                "dop_model",
                epoch=ref_hop,
                doppler_rate_hz_s=(
                    ctx.args.doppler_sign
                    * (-(v2_dr["range_rate_mps"] - v0_dr["range_rate_mps"]) / 4.0)
                    / C_LIGHT
                    * ctx.args.carrier_hz
                ),
            )
        elif v_dr is not None and v2_dr is not None:
            # Fallback when pd0 lacks the PRN (pd0 is cached with pd and pd2, so this is rare): a
            # forward difference over [now_w, now_w + 2 s], i.e. the rate 1 s after its tag.
            seed.put(
                "dop_model",
                epoch=ref_hop,
                doppler_rate_hz_s=(
                    ctx.args.doppler_sign
                    * (-(v2_dr["range_rate_mps"] - v_dr["range_rate_mps"]) / 2.0)
                    / C_LIGHT
                    * ctx.args.carrier_hz
                ),
            )
        elif ctx.args.almanac and prn in ctx.pred:
            seed.put("dop_model", epoch=ref_hop, doppler_rate_hz_s=ctx.pred[prn][1])
        # A measured rate (fit_dop_rate) has the last word, as --seed-doppler det does, once it has
        # dop_rate_min_pts points over dop_rate_min_span_s and lies inside dop_rate_max. It must
        # also agree with the model to dop_rate_model_tol: the fit rides the detections' Doppler
        # scatter, and dop_rate_max bounds only the magnitude, so a wrong-signed or half-size fit
        # would pass it.
        _model_dr = seed.get("doppler_rate_hz_s")
        _dr = fit_dop_rate(
            ctx.cpt.dop_hist.get(prn, []),
            ctx.args.hops_per_sec,
            ctx.args.dop_rate_min_pts,
            ctx.args.dop_rate_min_span_s,
            ctx.args.dop_rate_max,
        )
        if (
            _dr is not None
            and _model_dr is not None
            and abs(_dr - _model_dr) > ctx.args.dop_rate_model_tol
        ):
            # The fit and the model disagree beyond the model's accuracy: keep the model's rate,
            # which comes from an orbit rather than from detection noise.
            ctx.dop_rate_rejected[prn] = (_dr, _model_dr)
        elif _dr is not None:
            seed.put("dop_fit", epoch=ref_hop, doppler_rate_hz_s=_dr)
            ctx.dop_rate_fitted[prn] = _dr
        _hc = ctx.cp_to_seed_currency(
            h, seed_dop, float(seed.get("doppler_rate_hz_s", 0.0) or 0.0)
        )
        fit = fit_cp_rate(_hc, ctx.code_len)
        if fit is not None:
            rate, h0, cp_ref = fit
            # Code-rate cross-check (--cp-rate-model-tol). The residual code rate is common mode:
            # propagate_seed feeds each satellite's geometric code Doppler forward, so what this
            # seed carries is the receiver's l-a clock, one number for the chain (satellite clock
            # drift, af1 ~1e-12 s/s, is three orders smaller). Only a chain with detectors reaches
            # this fit (gps_l5 in production); the chains without detectors borrow its l-a through
            # the Receiver (the dead-reckon stage).
            #
            # fit_cp_rate is unbounded and unwraps its history to the nearest code period, so one
            # mis-wrap puts a whole period into the slope. Bound the deviation from the pooled
            # clock, not the magnitude: the clock runs ~3.45 chips/s uncalibrated (codeloop), and an
            # absolute ceiling would reject the feed-forward the trim needs. A rejection is logged
            # and counts toward --fit-flush-on-reject; it does not change the seeded rate, which is
            # the clock's only under --cp-rate-model-primary (below). The fit's position (cp_ref at
            # h0) is seeded either way. 0 disables.
            _seed_rate = rate
            _tol = getattr(ctx.args, "cp_rate_model_tol", 0.0)
            if _tol > 0.0 and ctx.cb.code_ema is not None:
                _model = cp_rate_from_code_bias(
                    ctx.cb.code_ema, ctx.args.hops_per_sec, ctx.args.chip_rate_hz,
                )
                _dev = (rate - _model) * ctx.args.hops_per_sec  # chips/s
                if abs(_dev) > _tol:
                    ctx.cp_rate_rejected[prn] = (
                        rate * ctx.args.hops_per_sec,
                        _model * ctx.args.hops_per_sec,
                    )
                    _log_rl(
                        "cprate-rej-%d" % prn,
                        "PRN %d cp-rate REJECTED: fit %+.3f chips/s vs pooled clock "
                        "%+.3f (dev %+.3f > --cp-rate-model-tol %.3f) -- position kept, "
                        "clock rate seeded"
                        % (
                            prn,
                            rate * ctx.args.hops_per_sec,
                            _model * ctx.args.hops_per_sec,
                            _dev,
                            _tol,
                        ),
                        every_s=60.0,
                    )
                    # --fit-flush-on-reject: the seed's position is the same fit evaluated at
                    # ref_hop, so a wrap-blown slope also moves the command by (slope error x
                    # history span), and the poison sustains itself (off-peak command, weak
                    # detections, re-poisoned history) for as long as the history lasts. After N
                    # consecutive rejections the history is flushed and the satellite is seeded from
                    # its detections until a clean fit rebuilds. 0 disables.
                    _n = ctx.cpt.rej_streak.get(prn, 0) + 1
                    _flush_n = getattr(ctx.args, "fit_flush_on_reject", 0)
                    if _flush_n > 0 and _n >= _flush_n:
                        ctx.cpt.hist.pop(prn, None)
                        ctx.cpt.rej_streak.pop(prn, None)
                        _log(
                            "PRN %d cp-fit history FLUSHED: %d consecutive rejected "
                            "rates (last %+.2f chips/s vs clock %+.2f) -- wrap-poisoned "
                            "history dropped, sat rides the birth path while a clean "
                            "fit rebuilds"
                            % (
                                prn,
                                _n,
                                rate * ctx.args.hops_per_sec,
                                _model * ctx.args.hops_per_sec,
                            )
                        )
                        # The fit's input samples are logged nowhere else, so the flush logs the
                        # history's tail in the fit's own currency with each detection's reported
                        # Doppler (the alias bin it rode), to name the sample that dragged the
                        # slope.
                        _log(
                            "PRN %d cp-fit history at flush, newest last "
                            "(hop,cp chips in seed currency,det dop Hz): %s"
                            % (
                                prn,
                                " ".join(
                                    "%d,%.1f,%+.0f" % (hh, cc, dd)
                                    for (hh, cc), (_, _, dd) in zip(_hc[-12:], h[-12:])
                                ),
                            )
                        )
                    else:
                        ctx.cpt.rej_streak[prn] = _n
                else:
                    ctx.cpt.rej_streak.pop(prn, None)
            # --cp-rate-model-primary: the command rides the pooled-clock model rate for every fit
            # and the fitted rate is monitor-only, as on the model-primary chains. The fit's
            # position is kept, the tol/flush guard above still protects it, and `rate` still feeds
            # the l-a pool below as a measurement. Before the clock has an EMA (code_ema None) the
            # fitted rate is seeded.
            if (
                getattr(ctx.args, "cp_rate_model_primary", 0)
                and ctx.cb.code_ema is not None
            ):
                _seed_rate = cp_rate_from_code_bias(
                    ctx.cb.code_ema, ctx.args.hops_per_sec, ctx.args.chip_rate_hz,
                )
            # Substitute the command only, never the measurement: `rate` stays the fitted slope,
            # because code_clock_bias_sample() below contributes it to the very l-a pool `_model`
            # came from, and feeding the pool its own output would reinforce whatever it believes.
            seed.put(
                "cp_fit",
                epoch=h0,
                code_phase_rate=_seed_rate,
                ref_hop=h0,
                code_phase_chips=cp_ref,
            )
            ctx.fitted.add(prn)
            # This fit's (l-a) sample: its slope over f_chip. The slope is residual (the replica
            # feeds the geometric code Doppler forward), so no carrier term is subtracted.
            la = code_clock_bias_sample(
                rate, ctx.args.hops_per_sec, ctx.args.chip_rate_hz,
            )
            # Per-sample gate: a noisy or unwrap-blown slope is a large l-a outlier that a
            # few-satellite median cannot reject, and a wandering pooled l-a moves every seeded code
            # rate (1 ppm = f_chip*1e-6 chips/s), walking the deep integration off peak within its
            # ~1 s window. Bounded by code_bias_max (ppm), and only from detections at acquire_snr
            # or above.
            if snr >= ctx.args.acquire_snr and abs(la) < ctx.args.code_bias_max * 1e-6:
                ctx.la_samples.append(la)
            _log_rl(
                "cpfit-%d" % prn,
                "PRN %d cp-fit: %.2f chips @ hop %d, slope %+.3f chips/s "
                "(%d pts, l-a %+.3f ppm)"
                % (prn, cp_ref, h0, rate * ctx.args.hops_per_sec, len(h), la * 1e6),
            )
        # Measured overlay alignment: the search reports the long-code phase. cp0 and the seed are
        # both arguments in the C = arg + n*cps convention, so matching the primary phase and the
        # overlay index gives
        #     cp_long = cp_primary + nh * CODE_LEN   (mod LC_SEG*CODE_LEN)
        # per satellite, with no wall clock or range model.
        if cp_long >= 0.0 and ctx.lc_seg > 1:
            # The search already reduced this at the overlaid code's own length, so it carries the
            # period; re-deriving it here (from nh or absolute time) would duplicate a convention
            # the search owns.
            seed.put(
                "nh_lift",
                epoch=ref_hop,
                code_phase_chips=cp_long % (ctx.lc_seg * ctx.code_len),
            )
            ctx.cl_report.append("PRN %d long-cp (search)" % prn)
            # Carry the phase at the search's own epoch too. cp0 refers back to sample 0 through the
            # reported Doppler, which turns Doppler error into whole code periods on the t_abs
            # lever; a phase at its own epoch has no such lever.
            if cp_at_ref >= 0.0:
                LLc = ctx.lc_seg * ctx.code_len
                ph = cp_at_ref % LLc
                # Joint overlay fit (--nh-joint): vote on the raw measured phase, before the
                # debounce or any offset touches it (_nh_joint_vote).
                _nhj_ok = (
                    ctx.args.nh_joint != "off"
                    and ctx.utc0_sample0
                    and ctx.args.almanac
                    and prn in ctx.pred
                )
                _ph_raw = ph  # the measured phase, before the debounce can move it:
                # the vote, the referee and the apply-side snap read this, never the debounced ph
                if _nhj_ok and snr >= ctx.args.nh_joint_min_snr:
                    _nh_joint_vote(ctx, prn, _ph_raw, ref_hop, snr, ctx.drp.now_w)
                # Period continuity is a check, not an authority. The search measures the overlay
                # period (AcquisitionResult::peak_tau_samples), so overriding it from history can
                # only add error, and a resolver that stored its own corrections would make one bad
                # correction permanent. This predicts the period from the previous strong detection,
                # logs any disagreement (a nonzero m on a strong satellite means the source broke)
                # and stores the measured phase; the only change it may make is the debounce below.
                _nh_deferred = False
                prev = ctx.cpt.ph_hist.get(prn)
                if prev is not None:
                    h0, ph0, dop0 = prev
                    dh = ref_hop - h0
                    gap_s = dh / ctx.args.hops_per_sec
                    if 0 < gap_s <= 900.0:
                        rate = (
                            ctx.args.chip_rate_hz
                            / ctx.args.hops_per_sec
                            * (
                                1.0
                                + ctx.args.code_doppler_sign
                                * 0.5
                                * (dop0 + dop)
                                / ctx.args.carrier_hz
                            )
                        )
                        ph_pred = (
                            ph0 + dh * rate
                        ) % LLc  # predicted phase at this detection's hop
                        m = (
                            int(round(((ph_pred - ph) % LLc) / ctx.code_len))
                            % ctx.lc_seg
                        )
                        if m:
                            resid = (
                                (ph + m * ctx.code_len - ph_pred + LLc / 2) % LLc
                            ) - LLc / 2
                            # Only a strong disagreement is evidence about the source; a marginal
                            # detection disagreeing tells us about the detection.
                            sev = (
                                "SOURCE PERIOD DISAGREES"
                                if snr >= ctx.args.period_check_snr
                                else "weak det"
                            )
                            _log_rl(
                                "phcont-%d" % prn,
                                "PRN %d period continuity %s: %+d periods "
                                "(snr %.0f, gap %.0f s, residual %+.1f chips) "
                                "-- NOT applied" % (prn, sev, m, snr, gap_s, resid),
                                every_s=60.0,
                            )
                        if m and ctx.args.nh_period_debounce > 0:
                            # Period debounce (--nh-period-debounce): the search's measured period
                            # can toggle +-1 on strong satellites with the fine phase right, and
                            # each single-detection flip would step every node's seed by a whole
                            # code period. A changed period is adopted only after this many
                            # consecutive detections with the same m; until then the seed carries
                            # the measured fine phase with the standing period. ph_hist is not
                            # updated while a period is pending (storing the flipped phase would
                            # make the next, correct detection alarm), and a real change passes
                            # after N detections, so a wrong standing value cannot deadlock.
                            _pm, _pc = ctx.cpt.nh_pending.get(prn, (0, 0))
                            _pc = _pc + 1 if _pm == m else 1
                            ctx.cpt.nh_pending[prn] = (m, _pc)
                            if _pc < ctx.args.nh_period_debounce:
                                ph = (ph + m * ctx.code_len) % LLc
                                _nh_deferred = True
                                _log_rl(
                                    "phdeb-%d" % prn,
                                    "PRN %d period DEBOUNCED: measured %+d period(s) "
                                    "off the standing one (%d/%d consecutive) -- "
                                    "standing period kept, measured fine phase seeded"
                                    % (prn, m, _pc, ctx.args.nh_period_debounce),
                                    every_s=60.0,
                                )
                            else:
                                ctx.cpt.nh_pending.pop(prn, None)
                                _log(
                                    "PRN %d period ADOPTED: %+d period(s), confirmed by "
                                    "%d consecutive detection(s)" % (prn, m, _pc)
                                )
                        elif not m:
                            ctx.cpt.nh_pending.pop(prn, None)
                # Feed ph_hist only from detections whose phase means something: below
                # period_check_snr the phase is noise and would poison every comparison until the
                # PRN is seen again (a PRN's first entry is kept whatever its SNR). A
                # debounce-deferred period feeds nothing: the corrected phase is our own word and
                # the measured one is unconfirmed.
                if not _nh_deferred and (
                    snr >= ctx.args.period_check_snr or prn not in ctx.cpt.ph_hist
                ):
                    ctx.cpt.ph_hist[prn] = (ref_hop, ph, dop)
                # Joint overlay fit, apply side: with a resolved consensus the overlay segment is
                # derived (prediction + common offset) and the measurement keeps only its sub-chip
                # fine phase. ph_hist was fed above from measured phases only, so nothing derived is
                # stored as the satellite's own word.
                if _nhj_ok:
                    _nhc = _nh_joint_consensus(ctx, ctx.drp.now_w)
                    if _nhc is not None:
                        # The derived segment applies to every satellite, strong included: the
                        # consensus is the weighted median of the strong population's raw phases, so
                        # no single satellite's label toggle (or a stale ph_hist standing period the
                        # debounce keeps deferring) can move its own seed. A nonzero k on a strong
                        # detection is the referee's alarm about the source.
                        _ph_j, _k_j = _nh_joint_snap(
                            ctx, prn, _ph_raw, ref_hop, _nhc[0]
                        )
                        if _k_j:
                            _log_rl(
                                (
                                    "nhjref-%d"
                                    if snr >= ctx.args.period_check_snr
                                    else "nhjapp-%d"
                                )
                                % prn,
                                "PRN %d overlay segment DERIVED from consensus: %+d "
                                "period(s) off the %s measurement (snr %.0f)%s"
                                % (
                                    prn,
                                    _k_j,
                                    "STRONG"
                                    if snr >= ctx.args.period_check_snr
                                    else "weak",
                                    snr,
                                    " -- if this persists across sats the consensus "
                                    "or the prediction is wrong"
                                    if snr >= ctx.args.period_check_snr
                                    else "",
                                ),
                                every_s=60.0,
                            )
                        ph = _ph_j
                ph = ph % LLc
                seed.put("nh_lift", epoch=ref_hop, code_phase_at_ref_chips=ph)
        # Hold on lock: once a PRN shows a real lock, freeze its seed tuple and let the DLL own the
        # sub-chip residual. The search's per-fix cp is good to ~1-2 chips, so re-anchoring on every
        # fit would inject that jitter into the despread; the fit is for capture, the frozen anchor
        # plus the DLL for tracking.
        #
        # (cp0, rate, ref_hop, Doppler) is the tracker's code currency: cp0 shifts by
        # t_abs*f_chip/f_carrier chips per Hz of Doppler change, so the currency must be piecewise
        # constant while locked (a merely smooth Doppler multiplies t_abs and walks the despread).
        # The whole tuple is frozen; a Doppler move beyond the hold_max_dop_hz fence is translated
        # into the new currency, not released. The hold is released after 3 consecutive reads below
        # the lock gate or by the escape referee, and the released seed re-anchors on this cycle's
        # detection.
        prev = ctx.seeds.get(prn)
        # Track-vs-search referee (the escape below): the freeze trusts the DLL, but a sharp-ACF
        # (BOC) discriminator has a finite capture range (~1/3 chip at 0.5-chip effective spacing)
        # with stable false equilibria beyond it (prompt R ~ -0.25, -12 dB). A track pushed past
        # capture servoes the wrong lobe while deep stays coherent and amp bleeds ~12 dB, and the
        # hold never lets go. The search is the referee: a persistent disagreement beyond the
        # capture range releases the freeze and re-anchors the seed on the search.
        cp_err = None
        # The effective trim the tracker adds to the model phase: the Python integrator's dls.trim
        # plus the C++ fleet loop's standing trim from the readback (one cycle old: the trim-arming
        # stage refreshes it after this one). Normally only one arm is nonzero per PRN, but the sum
        # is right in every regime, including a handover where both are.
        _trim_eff = ctx.dls.trim.get(prn, 0.0) + (
            (ctx.dls.readback.get(prn) or {}).get("trim_chips", 0.0)
        )
        # The detection's physical phase at its epoch, rebuilt from the (cp0, dop) pair the search
        # published together, so no translation enters. (cp_at_ref is the same phase in the C++
        # last-sample convention, one hop less a sample later.)
        _cpe_recon = (
            cp
            + (ref_hop / ctx.args.hops_per_sec)
            * ctx.args.chip_rate_hz
            * (1.0 + ctx.args.code_doppler_sign * dop / ctx.args.carrier_hz)
        ) % ctx.code_len
        # The innovation: measurement minus forecast (chips, wrapped to one code period), for every
        # accepted detection, before this cycle overwrites the seed with it. It is published
        # (innov_hist: the INNOV log line and the publisher rows) and, for a held PRN, it is the
        # escape referee's statistic below, so the two cannot disagree.
        _inv = None
        # Dead-reckoned seeds are excluded; MINNOV is their referee. Their ref_hop comes from the
        # dead-reckon time base, which is the wall clock unless --dr-fengine-axis, and a
        # wall-clock ref_hop against a detection's F-engine hop reads the axis offset as chips.
        if (
            prev is not None
            and (ctx.dr_state is None or prn not in ctx.dr_state["seeded"])
            and all(
                k in prev
                for k in (
                    "code_phase_chips",
                    "code_phase_rate",
                    "ref_hop",
                    "doppler_hz",
                )
            )
        ):
            # Forecast what the tracker runs: tracker_phase_at mirrors propagate_seed, including its
            # preference for code_phase_at_ref_chips over cp0. A forecast from cp0 would swing on
            # the t_abs lever, because seed writers legitimately move doppler_hz without
            # re-projecting cp0. The measurement moves to the same last-sample convention with its
            # own Doppler (_hop_off_det).
            _fc = tracker_phase_at(
                prev,
                ref_hop,
                ctx.args.hops_per_sec,
                ctx.args.chip_rate_hz,
                ctx.args.carrier_hz,
                ctx.args.code_doppler_sign,
                ctx.code_len,
            )
            _hop_off_det = (
                ctx.args.chip_rate_hz
                / ctx.args.hops_per_sec
                * (1.0 + ctx.args.code_doppler_sign * dop / ctx.args.carrier_hz)
            )
            _inv = (
                (_cpe_recon + _hop_off_det - _fc - _trim_eff + ctx.code_len / 2.0)
                % ctx.code_len
            ) - ctx.code_len / 2.0
            _ih = ctx.innov_hist.setdefault(prn, [])
            _ih.append((ctx.t0, _inv))
            # Keeps the newest 120 samples; the publish block cuts the 10-minute statistic by time
            # from these, so above one sample per 5 s the cap, not 10 minutes, sets the window.
            del _ih[:-120]
        # Model-primary PRNs (mp_flipped) stop here. The detection has done all it may: the
        # deep-gate stamp above, and the joint feed and MINNOV in the dead-reckon stage (the PRN is
        # dead-reckon-owned, so INNOV is skipped). The seed stays the model's, slewed by the
        # dead-reckon stage, so the re-anchor, hold/escape and commit below are bypassed. MINNOV's
        # exit gate (p95 hysteresis or starvation) hands the PRN back to the search, whose next
        # detection re-anchors it.
        if prn in ctx.mp_flipped:
            ctx.mp_last_det[prn] = ctx.t0
            continue
        if (
            prev is not None
            and prn in ctx.cp_held
            and all(
                k in seed for k in ("code_phase_chips", "code_phase_rate", "ref_hop")
            )
        ):
            # At-epoch comparison: the escape referee consumes the innovation above (the detection's
            # physical phase against tracker_phase_at, net of the trim), so the seed-vs-detection
            # Doppler bias does not enter, and the referee and the published INNOV are the same
            # number.
            cp_err = _inv
            if cp_err is not None and abs(cp_err) > ctx.args.hold_max_cp_err:
                _log_rl(
                    "cperr-%d" % prn,
                    "CP_ERR PRN %d: %+.2f chips at det hop %d (at-epoch: "
                    "search cp_at_ref vs held propagation; trim %+.2f "
                    "= py %+.2f + cpp %+.2f, hold_age %.0f s)"
                    % (
                        prn,
                        cp_err,
                        ref_hop,
                        _trim_eff,
                        ctx.dls.trim.get(prn, 0.0),
                        (ctx.dls.readback.get(prn) or {}).get("trim_chips", 0.0),
                        (ref_hop - prev["ref_hop"]) / ctx.args.hops_per_sec,
                    ),
                    every_s=60.0,
                )
            # Median gate: per-detection cp noise is 0.03-0.5 chips depending on the satellite
            # (multipath, BOC refine), so the 5-consecutive-sign rule alone lets a noisy satellite
            # sustain a false accusation; the median of the last 9 cycles' values must also exceed
            # the bar.
            if cp_err is not None:
                ctx.cpt.err_hist.setdefault(prn, []).append(cp_err)
                del ctx.cpt.err_hist[prn][:-9]
        # Escape counting: only sign-consistent disagreements count. A real wrong-lobe park is
        # one-signed; search-fit noise on a weak satellite alternates, and every escape re-injects
        # seed jitter and forces an overlay re-align.
        #
        # fit_trusted gates both the escape referee and hold admission: >= 6 history points (fit
        # noise ~ per-fix/sqrt(n)), a span of at least fit_maturity_span_s (point count alone is not
        # maturity when snapshots arrive fast; the span makes the code-Doppler quadratic
        # observable), and a detection at twice acquire_snr.
        fit_span_s = (
            ((h[-1][0] - h[0][0]) / ctx.args.hops_per_sec) if len(h) >= 2 else 0.0
        )
        fit_trusted = (
            fit is not None
            and len(h) >= 6
            and fit_span_s >= ctx.args.fit_maturity_span_s
            and snr >= 2.0 * ctx.args.acquire_snr
        )
        # Amp veto (escape_amp_veto): a hold despreading at full amplitude is on the main peak,
        # since the false lobes sit ~12 dB down; refuse the fit's accusation.
        amp_now = float((ctx.status.get(prn) or {}).get("amp_snr", 0) or 0)
        amp_veto = amp_now > ctx.args.escape_amp_veto
        # Integrity veto: never re-anchor onto a fit the BRDC model disputes. A fresh (< 10 s)
        # dead-reckon integrity residual (search against model, solved clock removed, normally +-0.2
        # chips) beyond hold_max_cp_err says the search's own position is off, so the fit is the
        # suspect, not the track.
        integ_veto = False
        if ctx.dr_state is not None and ctx.dr_state.get("integ"):
            _iv = ctx.dr_state["integ"].get(prn)
            if _iv is not None and ctx.t0 - _iv[1] < 10.0:
                if abs(_iv[0]) > ctx.args.hold_max_cp_err:
                    integ_veto = True
        cp_err_med_ok = (
            cp_err is not None
            and len(ctx.cpt.err_hist.get(prn, [])) >= 5
            and abs(statistics.median(ctx.cpt.err_hist[prn])) > ctx.args.hold_max_cp_err
        )
        if (
            cp_err is not None
            and abs(cp_err) > ctx.args.hold_max_cp_err
            and cp_err_med_ok
            and fit_trusted
            and not amp_veto
            and not integ_veto
        ):
            n_prev = ctx.cpt.escape.get(prn, 0)
            same_sign = (n_prev == 0) or (
                cp_err * ctx.cpt.escape_sign.get(prn, 0.0) > 0
            )
            ctx.cpt.escape[prn] = n_prev + 1 if same_sign else 1
            ctx.cpt.escape_sign[prn] = cp_err
        else:
            ctx.cpt.escape[prn] = 0
        # This monitor must stay a standalone statement above the escape `if`: placed between that
        # `if` and the hold `elif`, it would re-chain the `elif` to its own condition and silently
        # turn the hold into a per-detection re-anchor. test_holdchain.py pins it.
        #
        # Track-vs-model monitor (log only): the model-referenced track residual is r_i - cp_err
        # (search-vs-model minus search-vs-track, both in chips). It logs when the model puts the
        # track past the escape bar, notably while the fit-referenced referee is quiet (vetoed or
        # immature fit).
        if (
            cp_err is not None
            and ctx.dr_state is not None
            and ctx.dr_state.get("integ")
        ):
            _iv2 = ctx.dr_state["integ"].get(prn)
            if _iv2 is not None and ctx.t0 - _iv2[1] < 10.0:
                _tm = _iv2[0] - cp_err
                if abs(_tm) > ctx.args.hold_max_cp_err:
                    _log_rl(
                        "tvm-%d" % prn,
                        "TRACK-vs-MODEL PRN %d: %+.2f chips past the escape bar "
                        "(fit-ref cp_err %+.2f, integ %+.2f; fit-referee %s) -- "
                        "monitor only"
                        % (
                            prn,
                            _tm,
                            cp_err,
                            _iv2[0],
                            "AMP-VETOED"
                            if amp_veto
                            else "INTEG-VETOED"
                            if integ_veto
                            else "fit-untrusted"
                            if not fit_trusted
                            else "active",
                        ),
                        every_s=120.0,
                    )
        if ctx.cpt.escape.get(prn, 0) >= 5:
            _log(
                "ESCAPE PRN %d: track %+.2f chips off the search fit (5 consecutive,"
                " sign-consistent) -> release hold + DLL trim, re-anchor on the fit"
                % (prn, cp_err)
            )
            ctx.cpt.escape[prn] = 0
            ctx.cpt.err_hist.pop(prn, None)
            ctx.dls.trim.pop(prn, None)
            ctx.dls.last.pop(prn, None)
            ctx.cp_held.discard(prn)
            ctx.hold.miss.pop(prn, None)
        # Hold admission requires a mature fit (fit_trusted, the escape referee's own predicate): a
        # birth-window anchor (wide margins, unsolved bias, a short fit) can be chips wrong, and
        # protecting it would freeze the error. Until then the seed rides the maturing fit, which
        # corrects itself. Already-held PRNs are unaffected (the cp_held alternatives below).
        elif prev is not None and (
            (
                ctx.sig_of_last(ctx.status.get(prn)) >= ctx.args.hold_snr
                and (prn in ctx.cp_held or fit_trusted)
            )
            # --hold-on-present: also hold on fleet presence (ctx.qpop, the series that admits
            # trims). amp_snr rides the coherent arc and flickers with the deep fold, which leaves
            # locked satellites un-held; presence marks exactly the satellites whose trims the fleet
            # controller already trusts. fit_trusted is still required.
            or (
                ctx.args.hold_on_present > 0
                and fit_trusted
                and _present_streak(ctx, prn) >= ctx.args.hold_on_present
            )
            or (prn in ctx.cp_held and ctx.hold.miss.get(prn, 0) < 3)
        ):
            # Persistent-loss release: a single blank or stale status read (sig 0.0 from a poll
            # racing the emit) must not release the hold, since every release re-fits the seed and
            # pays a carrier re-anchor transient. A held PRN rides through up to 3 consecutive
            # sub-gate reads.
            if ctx.sig_of_last(ctx.status.get(prn)) >= ctx.args.hold_snr:
                ctx.hold.miss[prn] = 0
            elif ctx.args.hold_on_present > 0 and _present_streak(ctx, prn) >= 1:
                # Presence sustains a hold the sig path would starve: the sig read races the emit,
                # while the fleet gate is the statistic the trims already ride.
                ctx.hold.miss[prn] = 0
            else:
                ctx.hold.miss[prn] = ctx.hold.miss.get(prn, 0) + 1
            ddop = seed["doppler_hz"] - prev["doppler_hz"]
            # Safety net: bound one cycle's Doppler move to dop_max_rate_hz (3x the fence, at least
            # 5 Hz). A real MEO Doppler moves under ~1 Hz/s, so this fires only on a bad model; it
            # bounds the damage rather than forbidding motion.
            if abs(ddop) > ctx.args.dop_max_rate_hz:
                if prn not in ctx.cpt.dop_clamped:
                    ctx.cpt.dop_clamped.add(prn)
                    _log(
                        "DOP-CLAMP PRN %d: model wanted %+.1f Hz in one cycle (max %.1f)"
                        " -- clamping. A real MEO moves <1 Hz/cycle: SUSPECT THE MODEL."
                        % (prn, ddop, ctx.args.dop_max_rate_hz)
                    )
                ddop = math.copysign(ctx.args.dop_max_rate_hz, ddop)
                seed.put("dop_clamp", doppler_hz=prev["doppler_hz"] + ddop)
            # Freeze the currency while the Doppler move is inside the hold_max_dop_hz fence (0.1
            # cycle per record); translate it once the move exceeds the fence.
            if abs(ddop) <= ctx.args.hold_max_dop_hz or ddop == 0.0:
                # Currency frozen: the whole tuple rides unchanged.
                seed.put(
                    "hold_freeze",
                    epoch=prev["ref_hop"],
                    doppler_hz=prev["doppler_hz"],
                    code_phase_chips=prev["code_phase_chips"],
                    code_phase_rate=prev["code_phase_rate"],
                    ref_hop=prev["ref_hop"],
                )
                # The at-ref phase is part of the tuple and propagate_seed prefers it over cp0, so
                # it rides from prev, where it is paired with prev's ref_hop. The detection's fresh
                # phase beside the frozen ref_hop would command the despread off peak by the code
                # advance between the two epochs. With none in prev the field is dropped and the
                # tracker uses the frozen (cp0, dop) pair.
                if "code_phase_at_ref_chips" in prev:
                    seed.put(
                        "hold_freeze",
                        epoch=prev["ref_hop"],
                        code_phase_at_ref_chips=prev["code_phase_at_ref_chips"],
                    )
                else:
                    seed.pop("code_phase_at_ref_chips", None)
            else:
                # Currency translation: a Doppler update is not a loss of lock. The tracker builds
                # the code phase from the sample-0 anchor as
                #     cp(t) = cp0 + t*f_chip*(1 + sign*dop/f_carrier)
                # so the Doppler applies over the whole elapsed time: changing dop by ddop with cp0
                # fixed jumps the physical phase by t*f_chip*ddop/f_carrier chips, a lever that
                # grows with t. Releasing the hold here would trade a good anchor for a measured
                # one, whose noise costs a sharp BOC peak C/N0 and seconds of DLL re-convergence. So
                # keep the anchor and re-express it in the new Doppler's currency: pure arithmetic,
                # no fit noise. code_phase_rate is left alone; only the retroactive part needs
                # absorbing.
                t_now = ref_hop / ctx.args.hops_per_sec
                # doppler_hz keeps its new value and is re-attributed here: the translation is what
                # makes the (cp0, dop) pair valid at the shipped ref_hop.
                seed.put(
                    "translate",
                    epoch=prev["ref_hop"],
                    code_phase_chips=(
                        prev["code_phase_chips"]
                        - t_now
                        * ctx.args.chip_rate_hz
                        * ctx.args.code_doppler_sign
                        * ddop
                        / ctx.args.carrier_hz
                    )
                    % ctx.code_len,
                    code_phase_rate=prev["code_phase_rate"],
                    ref_hop=prev["ref_hop"],
                    doppler_hz=seed["doppler_hz"],
                )
                # As in the freeze arm: prev's at-ref phase is a physical phase at prev's ref_hop,
                # which a Doppler update does not move (the new Doppler enters the forward
                # propagation, not the anchor), so it rides unchanged.
                if "code_phase_at_ref_chips" in prev:
                    seed.put(
                        "translate",
                        epoch=prev["ref_hop"],
                        code_phase_at_ref_chips=prev["code_phase_at_ref_chips"],
                    )
                else:
                    seed.pop("code_phase_at_ref_chips", None)
                if prn not in ctx.cpt.translated:
                    ctx.cpt.translated.add(prn)
                    _log(
                        "TRANSLATE PRN %d: dop %+.0f -> %+.0f (%+.2f Hz) -> cp0 shifted "
                        "%+.2f chips; SAME physical code phase, anchor KEPT"
                        % (
                            prn,
                            prev["doppler_hz"],
                            seed["doppler_hz"],
                            ddop,
                            -t_now
                            * ctx.args.chip_rate_hz
                            * ctx.args.code_doppler_sign
                            * ddop
                            / ctx.args.carrier_hz,
                        )
                    )
            if prn not in ctx.cp_held:
                _sig = ctx.sig_of_last(ctx.status.get(prn))
                _via = (
                    "amp_snr %.1f >= %.1f" % (_sig, ctx.args.hold_snr)
                    if _sig >= ctx.args.hold_snr
                    else "PRESENT x%d cycles (fleet gate)" % _present_streak(ctx, prn)
                )
                _log(
                    "HOLD PRN %d: seed currency frozen (%s, dop %+.0f)"
                    % (prn, _via, prev["doppler_hz"])
                )
            ctx.cp_held.add(prn)
        else:
            if prn in ctx.cp_held:
                ddop_rel = (seed["doppler_hz"] - prev["doppler_hz"]) if prev else 0.0
                _log(
                    "RELEASE PRN %d: seed currency unfrozen (amp_snr %.1f, ddop %+.0f)"
                    % (prn, ctx.sig_of_last(ctx.status.get(prn)), ddop_rel)
                )
            ctx.cp_held.discard(prn)
            ctx.hold.miss.pop(prn, None)
        ctx.seeds[prn] = seed
        ctx.hold.low_hits[prn] = 0


def stage_push_seeds(ctx):
    """4: PUSH the consensus seeds to every tracker.

    The cycle's actuation point: whatever the stages above decided, this is what the trackers
    receive. The Python DLL trim is added to the posted copy only, so the stored seed stays the
    model's view and the trim stays the loop's, and the two ledgers remain separable.

    The published Doppler is the command, not an estimate: continuity of what the tracker is
    told beats freshness of what was last measured, and a seed that jumped to every new
    estimate would make the tracker chase the estimator's noise.

    ctx.rf.railed and ctx.rf.released are `+=` counters for the JRR-CMD line; the broker zeroes
    them before each call."""
    for prn, v in sorted(ctx.seeds.items()):
        d = dict(prn=prn, **v)
        if ctx.dls.trim.get(prn):
            d["code_phase_chips"] = d["code_phase_chips"] + ctx.dls.trim[prn]
            # The trim must move the at-ref phase too: propagate_seed prefers
            # code_phase_at_ref_chips whenever the payload carries it (search-fed seeds on an
            # overlaid code, and dead-reckoned ones under --seed-phase-transport), so a trim written
            # only into cp0 would be ignored. One trim, both currencies, same instant.
            if (
                d.get("code_phase_at_ref_chips", -1.0) is not None
                and d.get("code_phase_at_ref_chips", -1.0) >= 0.0
            ):
                _tmod = (ctx.lc_seg * ctx.code_len) if ctx.lc_seg > 1 else ctx.code_len
                d["code_phase_at_ref_chips"] = (
                    d["code_phase_at_ref_chips"] + ctx.dls.trim[prn]
                ) % _tmod
        if ctx.jrc is not None and prn not in ctx.probe_set:
            # Probes are excepted: no carrier, and a moving trim moves the reported Doppler, which
            # the beam map's churn gate reads as sky. A satellite whose rrate row has not converged
            # carries no trim: commanding from a wide row would inject the filter's own transient
            # into the NCO.
            _k = (ctx.args.dr_constellation, int(prn))
            # --rrate-cmd-min-sig: the row's own satellite must be detected this cycle (kcoh sig at
            # or above the bar) before it may command; the sigma gate alone admits a confident
            # estimate of noise. No kcoh row counts as not detected.
            _cmd_sig_ok = True
            if ctx.args.rrate_cmd_min_sig > 0.0:
                _cmd_sig_ok = (
                    ((ctx.dllp.kcoh or {}).get(prn) or {}).get("sig") or 0.0
                ) >= ctx.args.rrate_cmd_min_sig
            if _cmd_sig_ok and ctx.jrc.rrate_sigma(_k) <= ctx.args.rrate_cmd_max_sigma:
                _cmd = ctx.jrc.carrier_correction_hz(_k, ctx.args.carrier_hz)
                if ctx.args.carrier_max_hz > 0.0:
                    _cmd = max(
                        -ctx.args.carrier_max_hz, min(ctx.args.carrier_max_hz, _cmd)
                    )
                # Slew toward the target from the command posted last poll (rrate_cmd_slew_hz): the
                # feed's reference is exact only for a command that holds still over the emit lag,
                # so bounding the step is what keeps the closed loop stable. Railed steps are
                # counted into the JRR-CMD line; a rail that never clears is a target out of reach,
                # not convergence in progress.
                _prev = ctx.rf.cmd_applied.get(prn, 0.0)
                if ctx.args.rrate_cmd_slew_hz > 0.0:
                    _stp = max(
                        -ctx.args.rrate_cmd_slew_hz,
                        min(ctx.args.rrate_cmd_slew_hz, _cmd - _prev),
                    )
                    if abs(_cmd - _prev) > ctx.args.rrate_cmd_slew_hz:
                        ctx.rf.railed += 1
                    _cmd = _prev + _stp
                d["carrier_trim_hz"] = _cmd
                ctx.rr_cmd_new[prn] = _cmd
        # Release slew: a PRN that leaves the command set (sig bar lost, sigma widened, row gone)
        # walks its standing command back to 0 at the same bounded rate instead of stepping, the
        # reference discontinuity the slew bound exists to prevent. It drops out once within one
        # step of zero.
        if (
            ctx.args.rrate_command
            and prn not in ctx.rr_cmd_new
            and prn not in ctx.probe_set
            and ctx.rf.cmd_applied.get(prn)
        ):
            _prev = ctx.rf.cmd_applied[prn]
            _stp = (
                ctx.args.rrate_cmd_slew_hz if ctx.args.rrate_cmd_slew_hz > 0.0 else 0.0
            )
            if _stp > 0.0 and abs(_prev) > _stp:
                _cmd = _prev - math.copysign(_stp, _prev)
                d["carrier_trim_hz"] = _cmd
                ctx.rr_cmd_new[prn] = _cmd
                ctx.rf.released += 1
            # Otherwise (within one step of zero, or slew disabled) no carrier_trim_hz is posted,
            # which the tracker reads as 0, and the PRN leaves ctx.rf.cmd_applied.
        ctx.payload.append(d)
