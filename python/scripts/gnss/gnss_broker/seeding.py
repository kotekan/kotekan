import os
"""Detections into seeds, and the coast/drop decision that retires them.

`stage_detections_to_seeds` is the longest stage in the cycle and the one every other depends
on: it turns the search's (snr, doppler, code phase, ref_hop) into the seed triple the trackers
consume. `stage_coast_drop` is its other end -- when a satellite sets, or merely fades.

⚠️ AN ALIAS-BIN DETECTION IS HARMLESS. DO NOT "FIX" IT. The search's Doppler is ambiguous mod
1/(2*t_rec) -- 25 Hz on 20 ms records. The search back-projects cp0 to sample 0 using the SAME
reported Doppler, and `cp_to_seed_currency` adds that projection back with the same numbers, so
the round trip is exact whatever bin the Doppler rode. A "fold" that replaced dop before the
currency conversion broke exactly that cancellation, by K*t_abs*k*q. It was live for part of a
day in July 2026, which is why this paragraph exists.

⚠️ code_phase_chips IS AN ARGUMENT, NOT A TRANSPORTABLE QUANTITY -- meaningful only against
the epoch it was measured at (~5095 chips per Hz of Doppler). Never transport or DIFFERENCE
one; only `cp_at_ref` is comparable.

⚠️ MODEL-OWNED SATELLITES ARE EXEMPT FROM THE TLE UP-SET when dropping. The up-set mismaps some
BeiDou birds, so for dead-reckoned satellites the BRDC elevation governs -- in the dead-reckon
stage, not here. Dropping a bird the model still tracks hands the chain a hole it then has to
re-acquire from nothing.

@author Keith Vanderlinde
"""

import math
import statistics

from gnss_broker.sky import C_LIGHT
from gnss_broker.transport import _log, _log_rl
from gnss_broker.seed import Seed
from gnss_broker.fits import (
    retag_seed_doppler, code_clock_bias_sample, fit_cp_rate, fit_dop_rate, tracker_phase_at,
    cp_rate_from_code_bias, dr_seed_phys, dr_cp0, seed_phase_at_ref,
)


def _present_streak(ctx, prn):
    """Trailing run of PRESENT cycles in the population-honest series (D0's ctx.qpop).

    Reads the same per-cycle presence the fleet controller trims on -- NOT the `DLL:` log
    line (survivors only) and NOT amp_snr (the coherent arc). Returns 0 when the series
    has never seen the PRN.
    """
    hist = ctx.qpop.hist.get(prn)   # qpop is an unconditional ctx slot: no swallow --
    if not hist:                    # a missing attribute must CRASH, not run inert (#93)
        return 0
    n = 0
    for _t, _q, state in reversed(hist):
        if state != "present":
            break
        n += 1
    return n


def stage_coast_drop(ctx):
    """COAST / DROP: retire seeds for satellites that have set, and coast the ones merely fading.
        
    ⚠️ MODEL-OWNED SATS ARE EXEMPT FROM THE TLE UP-SET. The up-set mismaps some BeiDou birds, so
    for dead-reckoned satellites the BRDC elevation governs the drop instead -- in the dead-reckon
    stage, not here. Dropping a bird the model is still tracking would hand the chain a hole it
    would then try to re-acquire from scratch."""
    for prn in list(ctx.seeds):
        if prn in ctx.probe_set:
            continue
        if (ctx.up is not None and prn not in ctx.up
                and not (ctx.dr_state is not None and prn in ctx.dr_state["seeded"])):
            # (model-owned sats are exempt: the TLE up-set mismaps some BDS birds;
            # their BRDC elevation governs the drop, in the dead-reckon block)
            _log("drop PRN %d (set below horizon)" % prn)
            del ctx.seeds[prn]
            ctx.cp_held.discard(prn)
            ctx.hold.miss.pop(prn, None)
            ctx.hold.low_hits.pop(prn, None)
            continue
        if prn in ctx.best:  # re-detected -> re-anchored in the seed loop above (coast reset there)
            continue
        # not re-detected this poll but still visible -> COAST: forecast the Doppler forward.
        if ctx.dr_state is not None and prn in ctx.dr_state["seeded"]:
            # model-owned (dead-reckoned) seed: its doppler is FROZEN between re-pins
            # (each pin refreshes dop+cp+rate TOGETHER, currency-consistent); the TLE
            # pred must not touch it -- the BDS TLE<->PRN mapping mismaps some birds
            # (PRN 39: TLE el <5 vs BRDC el 10), so BRDC governs these sats entirely.
            pass
        elif ctx.args.almanac and prn in ctx.pred:
            # ⚠️ THE SEED'S DOPPLER IS VALID AT ref_hop, NOT AT NOW (2026-08-16).
            # gnss::propagate_seed computes
            #     dop_applied(t) = sd.doppler_hz + sd.dop_rate * (t - ref_hop)
            # so writing the forecast's NOW value into `doppler_hz` while leaving
            # `ref_hop` at the last re-detection makes the tracker add dop_rate*age on
            # top of a value that is ALREADY current. The applied Doppler then advances
            # at ~2x the true rate until the next detection re-anchors ref_hop, and
            # snaps back when it does -- the sawtooth measured on sky: drift -0.63 Hz/s
            # against a model -0.35, gap 60-81 Hz just before the snap, and
            # 69 Hz = 0.60 chips/s of CODE rate, which drags the prompt tap off-peak in
            # under two seconds. That is the churn (deep sig 1600 -> 3 on G23).
            #
            # Same disease as REC_PHI0 and as #44 itself: a VALUE updated to now while
            # its EPOCH says otherwise ([[#45]] value/currency/epoch). Two consequences,
            # both fixed here:
            #   1. store the Doppler BACK-PROPAGATED to ref_hop, so propagate_seed
            #      reconstructs the forecast exactly at now for ANY age;
            #   2. compare like with like when re-tagging cp -- the old EFFECTIVE
            #      Doppler at now, not the ref_hop-epoch number, or the ddop handed to
            #      retag_seed_doppler is inflated by that same dop_rate*age and kicks
            #      the code phase as well.
            # age = 0 whenever the sample-0 anchor is unknown, which reduces this to the
            # previous behaviour rather than guessing.
            _rate_new = ctx.pred[prn][1]
            # the rate the TRACKER will actually propagate with: absent key => 0, and
            # then there is no double-count to undo.
            _rate_eff = _rate_new if "doppler_rate_hz_s" in ctx.seeds[prn] else 0.0
            _age = 0.0
            if ctx.utc0_sample0:
                _age = max(0.0, (ctx.drp.now_w - ctx.utc0_sample0)
                           - ctx.seeds[prn].get("ref_hop", 0) / ctx.args.hops_per_sec)
            new_dop = ctx.pred[prn][0] + ctx.cb.value           # the forecast AT NOW
            _old_rate = ctx.seeds[prn].get("doppler_rate_hz_s", 0.0)
            # what the tracker is APPLYING at this instant, i.e. the currency cp is in
            old_dop = ctx.seeds[prn].get("doppler_hz", new_dop) + _old_rate * _age
            if new_dop != old_dop:
                # CURRENCY-CORRECT the coast, UNCONDITIONALLY (2026-07-19 audit A4): cp0
                # is meaningful only in its doppler's currency -- updating the forecast
                # dop WITHOUT re-expressing cp walks the despread by t_abs*f_chip*ddop/
                # f_c chips at soak age (the t_abs lever the code-currency rule forbids;
                # why long coasts silently lost the code peak).
                #
                # #44 (2026-08-12): the first fix translated at the SEED'S ANCHOR epoch
                # (ref_hop) -- which preserves the phase at the anchor and steps the
                # phase NOW by anchor_age * k_c * ddop per forecast update, i.e. the
                # residual half of the very symptom it targeted (~k_c*r*age^2/2 chips
                # of silent walk-off over a coast). The correct epoch is NOW, same as
                # the hold-path TRANSLATE has always used; both now go through
                # retag_seed_doppler so the algebra lives once, beside dr_cp0/
                # dr_seed_phys, with its own regression test. When the sample-0 anchor
                # is not known (no utc0_sample0), keep the old anchor-epoch behaviour:
                # a partial correction still beats the raw-dop overwrite it replaced.
                _t_retag = ((ctx.drp.now_w - ctx.utc0_sample0) if ctx.utc0_sample0
                            else ctx.seeds[prn].get("ref_hop", 0) / ctx.args.hops_per_sec)
                ctx.seeds[prn].put(
                    "coast_retag", epoch=ctx.seeds[prn].get("ref_hop"),
                    code_phase_chips=retag_seed_doppler(
                        ctx.seeds[prn].get("code_phase_chips", 0.0), old_dop, new_dop,
                        _t_retag, ctx.args.chip_rate_hz, ctx.args.carrier_hz,
                        ctx.args.code_doppler_sign, ctx.code_len),
                    # STORE AT ref_hop, not at now (see the note above).
                    doppler_hz=new_dop - _rate_eff * _age)
            if "doppler_rate_hz_s" in ctx.seeds[prn]:
                ctx.seeds[prn].put("coast_retag", epoch=ctx.seeds[prn].get("ref_hop"),
                               doppler_rate_hz_s=_rate_new)
        rec = ctx.status.get(prn, {})
        if ctx.have_sig:
            metric, thresh = ctx.sig_of(rec), ctx.args.lock_snr
        else:
            metric, thresh = float(rec.get("amplitude", 0.0)), ctx.args.drop_amplitude
        # FOLD-INDEPENDENT HOLD (#58). OR-ed against its own bar, never max()-ed into
        # `metric`: prompt hold is a power ratio and `metric` is a debiased sigma.
        if metric >= thresh or (ctx.args.lock_prompt_hold > 0.0
                                and ctx.hold.prev.get(prn, 0.0) >= ctx.args.lock_prompt_hold):
            ctx.hold.low_hits[prn] = 0  # lock holding through the dropout -> reset coast
        else:
            ctx.hold.low_hits[prn] = ctx.hold.low_hits.get(prn, 0) + 1
            # dead-reckoned seeds are MODEL-owned: visible + predicted = keep despreading
            # (their whole point is sats with no signal above the search threshold)
            if (ctx.hold.low_hits[prn] >= ctx.coast_polls and not ctx.args.coast_to_horizon
                    and not (ctx.dr_state is not None and prn in ctx.dr_state["seeded"])):
                _log("drop PRN %d (coast %.0fs expired, %s=%.2f)"
                     % (prn, ctx.args.coast_budget, "sig" if ctx.have_sig else "|A|", metric))
                del ctx.seeds[prn]
                ctx.hold.low_hits.pop(prn, None)


def stage_detections_to_seeds(ctx):
    """1: DETECTIONS -> SEEDS. Turn this cycle's best-SNR detection per PRN into a tracker seed.
        
    The longest single stage in the cycle, and the one every other stage depends on: it converts
    the search's (snr, doppler, code phase, ref_hop) into the seed triple the trackers consume,
    via cp_to_seed_currency.
        
    ⚠️ AN ALIAS-BIN DETECTION IS HARMLESS -- DO NOT "FIX" IT. The search's Doppler is ambiguous
    mod 1/(2*t_rec) (25 Hz on 20 ms records). The search back-projects cp0 to sample 0 with the
    SAME reported dop, and cp_to_seed_currency adds that projection back with the same numbers, so
    the round trip is exact whatever bin the Doppler rode. A "fold" that replaced dop before the
    currency conversion broke exactly that cancellation, by K*t_abs*k*q -- it was live for part of
    a day in 2026-07-20 and is the reason this comment is here."""
    for prn, (snr, dop, cp, ref_hop, det_nh, cp_long, cp_at_ref) in ctx.best.items():
        # DETECTION ALIAS CENSUS (2026-07-20; was briefly a FOLD, corrected same day):
        # the search's Doppler estimate is ambiguous mod 1/(2*t_rec) -- 25 Hz on L2C's
        # 20 ms records, 50 Hz B1C. An alias-bin detection is HARMLESS to the cp
        # bookkeeping: the search back-projects cp0 to sample 0 with the SAME reported
        # dop (GnssChannelizedSearch: det.doppler_hz and the drift term share one
        # variable), and cp_to_seed_currency adds that projection back with the same
        # numbers -- the round trip is exact whatever bin the dop rode. The v1 fold
        # REPLACED dop before the currency conversion, breaking exactly that
        # cancellation by K*t_abs*k*q: the TRACK-vs-MODEL monitor caught held-sat
        # candidates 12-57 chips off their healthy tracks within the hour (L2C
        # 18/23/32, 15:20) -- and a candidate that wrong silently DISABLES the escape
        # referee (sign-flipping cp_err never sustains 5 consecutive). So: measure,
        # never modify. The census still maps which chains/sats ride alias bins (the
        # B1C zombie-birth investigation continues on that data).
        if (ctx.args.det_alias_fold and ctx.args.almanac and prn in ctx.pred
                and ctx.cb.ema is not None and not ctx.cb.stale):
            _aref = ctx.pred[prn][0] + ctx.cb.value
            _k = round((dop - _aref) / ctx.q_alias_hz)
            if _k != 0 and abs(dop - _aref) < 3.5 * ctx.q_alias_hz:
                _log_rl("afold-%d" % prn,
                        "ALIAS BIN PRN %d: det dop %+.1f = model %+.1f %+d bin(s) of "
                        "%.0f Hz (census only; cp round-trip is exact)"
                        % (prn, dop, _aref, _k, ctx.q_alias_hz),
                        every_s=30.0)
        v_dr = ctx.dr_pd.get((ctx.args.dr_constellation, prn)) if ctx.dr_pd else None
        if ctx.up is not None and prn not in ctx.up:
            # accept the detection anyway if BRDC says it's up: the TLE up-set
            # mismaps some BDS birds (PRN 39: TLE el<5 vs BRDC el 10)
            if v_dr is None or v_dr["el"] < ctx.args.mask_deg:
                continue
        # #79: THE SEARCH IS THE ADMISSION AUTHORITY for the trim presence gate. Stamped
        # here -- after the visibility filter, so a spurious below-horizon detection
        # cannot arm a correction, and before every seeding-policy filter below, because
        # eligibility asks "is this satellite up and detectable", not "did this cycle
        # like the detection well enough to re-seed from it".
        if ctx.args.dll_deep_gate_from_search > 0.0 and snr >= ctx.args.dll_deep_gate_from_search:
            ctx.dls.deep_gate_seen[prn] = ctx.t0
        _dop_src = "pred" if (ctx.args.almanac and prn in ctx.pred) else "DET(grid)"
        seed_dop = (ctx.pred[prn][0] + ctx.cb.value) if (ctx.args.almanac and prn in ctx.pred) else dop
        # Dead-reckon armed: prefer the BRDC doppler for EVERY seed -- the same model
        # that owns the undetected sats. Mixing sources stepped the seed doppler by
        # the TLE-vs-BRDC error at every DR<->search handoff (~25 Hz on a stale TLE
        # = the whole E1 hold fence; observed E32 RELEASE ddop -25, 2026-07-13).
        # THE DOPPLER AND THE CODE PHASE DO NOT HAVE TO SHARE A TRUST DECISION.
        # dr_untrusted is set for two unlike reasons: a stale ephemeris (the orbit
        # itself is doubtful -- true for range AND range-rate) or a code-phase
        # integrity residual over --dr-max-integrity-chips (1.0 chip ~ 30 m, which is
        # ordinary iono + b_sat and says nothing about the range rate). Under
        # --dr-doppler-ignores-integrity only the first demotes the Doppler, so a
        # satellite flipping trust no longer switches its seed between two BRDC
        # evaluations -- and the seed IS the replica's carrier phase.
        _unt = ctx.dr_untrusted.get(prn)
        _dop_trusted = (_unt is None
                        or (ctx.args.dr_doppler_ignores_integrity
                            and not str(_unt).startswith("ephemeris")))
        if v_dr is not None and _dop_trusted:
            _dop_src = "dr" if _unt is None else "dr(code-untrusted)"
            seed_dop = (ctx.args.doppler_sign * (-v_dr["range_rate_mps"] / C_LIGHT
                                             * ctx.args.carrier_hz) + ctx.cb.value)
            if _unt is not None:
                _log_rl("dopkeep-%d" % prn,
                        "PRN %d: code model untrusted (%s) but KEEPING the BRDC Doppler "
                        "-- an integrity residual is a code-phase statement, and "
                        "switching the seed switches the replica's carrier phase"
                        % (prn, _unt), every_s=120.0)
        # ...unless the measured Doppler is explicitly preferred (--seed-doppler det). Last
        # word, so it overrides the model AND the DR: those two exist to keep the seed
        # smooth and to own undetected sats, but neither helps a sat we HAVE measured, and
        # a model error rides the seed's whole extrapolation age at 0.0087 chips/Hz/s.
        if ctx.args.seed_doppler == "det":
            _dop_src = "det"
            seed_dop = dop
        # SEED-STEP ATTRIBUTION (2026-07-18, the one-grid-step NCO disease): any seed
        # doppler step > 10 Hz vs the sat's previous seed is loud, with its source --
        # a ~exact-doppler_step jump here is the smoking gun for a grid/quantization
        # slip upstream (the hint-anchored search grid was one such; fixed same day).
        _prev_sd = ctx.seeds.get(prn, {}).get("doppler_hz")
        if _prev_sd is None and ctx.args.almanac and not ctx.cb.available:
            # S2d: the condition is now "no bias FROM ANY SOURCE", not "this chain has
            # not solved its own". Same guard, wider supply. It is the exact deadlock
            # that cost 2h45m on 2026-07-27: L5 GPS could measure one satellite, never
            # reached --bias-min-sats, so this line withheld every first seed -- while
            # its two siblings sat in the same band holding the right answer (+32 Hz)
            # and its OWN l-a fit held a better one still. With the fused state that
            # chain is `bias_available` from cycle 1 and never enters this branch.
            #
            # BIAS-UNSOLVED FIRST-SEED GUARD (2026-07-18): a first seed sent before the
            # clock-freq bias solves carries the FULL dongle offset (~150 Hz at L1,
            # measured: PRN 5 first seed 1146.5 vs det 974.1 at bias +0.0). A strong sat
            # can lock and enter hold-freeze on that wrong currency within seconds --
            # the startup race behind the stochastic diseased-node births (rails at
            # first seeding, fleet-dependent). Detections don't need seeds, so holding
            # off costs ~2 s of acquisition and delays nothing else: the bias solves
            # from the same detections this cycle already collected.
            continue
        if _prev_sd is None:
            # FIRST seed: full attribution (the rail onsets coincide with first seeding,
            # and a first seed has no previous value for the step tripwire to fire on).
            _log("PRN %d FIRST SEED dop %.1f (src=%s, det=%.1f, pred=%s, bias %+.1f, trim %+.1f)"
                 % (prn, seed_dop, _dop_src, dop,
                    ("%.1f" % (ctx.pred[prn][0])) if (ctx.args.almanac and prn in ctx.pred) else "n/a",
                    ctx.cb.value, ctx.car.trim.get(prn, 0.0)))
        elif abs(seed_dop - _prev_sd) > 10.0 and prn in ctx.cp_held:
            # HELD sat: the candidate walks while the emitted tuple stays frozen /
            # translated -- this "step" is never applied as-is, and logging it every
            # cycle flooded the logs (195k lines on L1 GPS in 3 h, 2026-07-18). The
            # tripwire's purpose (grid/quantization slips on LIVE seeds) is served by
            # the un-held branch below.
            pass
        elif abs(seed_dop - _prev_sd) > 10.0:
            _log("PRN %d SEED DOP STEP %+.1f Hz (%.1f -> %.1f, src=%s, det=%.1f)"
                 % (prn, seed_dop - _prev_sd, _prev_sd, seed_dop, _dop_src, dop))

        # Maintain a per-PRN cp0-vs-hop history (only distinct snapshots; the search
        # holds its detection between updates) and fit the first-order code drift.
        # DOPPLER-RATE HISTORY. The seed's doppler_rate_hz_s drives BOTH the tracker's
        # carrier NCO (out.doppler_hz = doppler + dop_rate*dt) and the quadratic CODE term
        # (quad = 0.5*(chip/f_c)*dop_rate*dt^2), so its error shows up twice. BRDC computes
        # it by range-rate differencing over a 4 s epoch pair -- a numerical derivative, and
        # we use it at the worst moment, near zenith where the curvature peaks. Measured
        # 2026-08-04: BRDC gave PRN 3 -0.4699 Hz/s where the Doppler track said -0.578, and
        # that 0.108 Hz/s residual is ~0.7 rad of phase curvature inside the 1.05 s deep
        # window -- a direct contributor to the 2.55 rad that costs the coherent sum 29 dB.
        #
        # We measure the Doppler every pass, so fit its slope instead. Well-conditioned at
        # an 11 s revisit: four points span ~44 s over which the Doppler moves ~25 Hz
        # against ~1.5 Hz per-detection noise. Same gap rule as the cp history -- a gap
        # means re-acquisition and the old slope is stale.
        dh_ = ctx.cpt.dop_hist.get(prn, [])
        if dh_ and (ref_hop - dh_[-1][0]) > ctx.max_gap_hops:
            dh_ = []
        if not dh_ or ref_hop != dh_[-1][0]:
            dh_.append((ref_hop, dop))
            dh_ = dh_[-ctx.hist_len:]
        ctx.cpt.dop_hist[prn] = dh_

        h = ctx.cpt.hist.get(prn, [])
        if h and (ref_hop - h[-1][0]) > ctx.max_gap_hops:
            h = []  # gap too large -> re-acquisition, old slope is stale
        # SNR gate (2026-08-02). The slope this fit is trying to resolve is ~0.0148
        # chips/s -- the drift from a ~1.7 Hz Doppler error. A detection near the
        # acquire threshold has a phase that is simply noise (measured on CHORD: below
        # snr ~60 the within-period residual runs ~2000 chips against a few chips above
        # it), so ONE such point does not degrade the fit, it destroys it. Default 0
        # keeps every point, which is the prototype's behaviour and right there: its
        # detections sit far above threshold and its revisit is seconds.
        if not h or ref_hop != h[-1][0]:
            if snr >= ctx.args.fit_min_snr:
                h.append((ref_hop, cp, dop))
                h = h[-ctx.hist_len:]
            elif h:
                _log_rl("fitsnr-%d" % prn,
                        "PRN %d cp-fit: skipping snr %.0f point (< --fit-min-snr %.0f); "
                        "%d in history" % (prn, snr, ctx.args.fit_min_snr, len(h)),
                        every_s=120.0)
        ctx.cpt.hist[prn] = h

        # The bare-detection cp is in the DETECTION's Doppler currency; the tracker will
        # despread at seed_dop. Convert (else a sat first acquired at t_abs inherits a
        # t_abs*f_chip*(dop-seed_dop)/f_c offset -- chips off-peak for any mid-run
        # acquisition, before tracking even starts).
        ((_, cp_seed_cur),) = ctx.cp_to_seed_currency([(ref_hop, cp, dop)], seed_dop)
        seed = Seed.born("det", epoch=ref_hop,
                         doppler_hz=seed_dop, code_phase_chips=cp_seed_cur,
                         code_phase_rate=0.0, ref_hop=ref_hop)
        # The doppler SOURCE was arbitrated above (_dop_src: pred | dr | det), BEFORE
        # the tuple existed, so the constructor's blanket "det" under-describes the
        # one field three estimators fight over. Re-attribute -- same value, provenance
        # only: the audit trail then separates a model-seeded doppler from a measured
        # one, the very distinction Phase 3's per-PRN primacy gate decides on.
        seed.put("dop_sel:" + _dop_src, epoch=ref_hop, doppler_hz=seed_dop)
        # 2nd-order carrier feed-forward: hand the tracker the almanac Doppler RATE (Hz/s, sign-
        # applied like doppler_hz); the tracker integrates it in its NCO (never a replica
        # retune -- that walks the absolutely-anchored code/carrier off-peak) so the deep-
        # integration residual stays flat even at zenith (max Doppler acceleration).
        v2_dr = ctx.dr_pd2.get((ctx.args.dr_constellation, prn)) if ctx.dr_pd2 else None
        v0_dr = ctx.dr_pd0.get((ctx.args.dr_constellation, prn)) if ctx.dr_pd0 else None
        if v2_dr is not None and v0_dr is not None:
            # BRDC doppler rate, CENTRAL difference over the +/-2 s pair straddling now_w
            # (task #52). Centred, so the rate is tagged at now_w rather than 2 s late.
            seed.put("dop_model", epoch=ref_hop,
                     doppler_rate_hz_s=(ctx.args.doppler_sign
                                        * (-(v2_dr["range_rate_mps"]
                                             - v0_dr["range_rate_mps"]) / 4.0)
                                        / C_LIGHT * ctx.args.carrier_hz))
        elif v_dr is not None and v2_dr is not None:
            # Fallback for the first cycle, before pd0 exists: the OLD forward form, and
            # it is deliberately still here rather than silently emitting nothing -- but it
            # is 2 s mis-tagged, so it must not be the steady state.
            seed.put("dop_model", epoch=ref_hop,
                     doppler_rate_hz_s=(ctx.args.doppler_sign
                                        * (-(v2_dr["range_rate_mps"]
                                             - v_dr["range_rate_mps"]) / 2.0)
                                        / C_LIGHT * ctx.args.carrier_hz))
        elif ctx.args.almanac and prn in ctx.pred:
            seed.put("dop_model", epoch=ref_hop, doppler_rate_hz_s=ctx.pred[prn][1])
        # MEASURED rate beats the model's, and it is the LAST word here for the same reason
        # --seed-doppler det is: the model exists to own sats we have not measured. Gated on
        # enough points over enough baseline that the slope is real rather than fitted to
        # detection noise.
        # CROSS-CHECK THE FIT AGAINST THE MODEL BEFORE ADOPTING IT. Measured on sky
        # 2026-08-05, seeded rate vs the Doppler actually observed over 140 s: PRN 8 seeded
        # 2.18x too small (-0.195 against -0.424), PRN 30 seeded the WRONG SIGN (+0.205
        # against -0.155), and PRNs 5/7/16 seeded None while drifting at -0.15..-0.45 Hz/s.
        # The rate is fitted from the MEASURED Doppler, which carries ~6 Hz of per-pass
        # search scatter (8.20.5), and it was the last word unconditionally -- a noisy
        # estimator overriding a model one, the same trade --seed-doppler det got wrong.
        # dop_rate_max only bounds the MAGNITUDE, so a wrong-signed or half-size fit inside
        # +-0.8 sails through. Its error costs twice: the carrier NCO extrapolation AND the
        # quadratic code term both use it.
        _model_dr = seed.get("doppler_rate_hz_s")
        _dr = fit_dop_rate(ctx.cpt.dop_hist.get(prn, []), ctx.args.hops_per_sec,
                           ctx.args.dop_rate_min_pts, ctx.args.dop_rate_min_span_s,
                           ctx.args.dop_rate_max)
        if (_dr is not None and _model_dr is not None and ctx.args.dop_rate_model_tol > 0.0
                and abs(_dr - _model_dr) > ctx.args.dop_rate_model_tol):
            # The two disagree by more than the model's own accuracy: trust the MODEL, which
            # comes from an orbit rather than from detection noise, and say so.
            ctx.dop_rate_rejected[prn] = (_dr, _model_dr)
        elif _dr is not None:
            seed.put("dop_fit", epoch=ref_hop, doppler_rate_hz_s=_dr)
            ctx.dop_rate_fitted[prn] = _dr
        elif ctx.args.force_doppler_rate is not None:
            # Replay-bench override: a recorded capture's sky is at another epoch (no almanac),
            # so inject a known rate into every seed to exercise the NCO feed-forward offline.
            seed.put("dop_force", epoch=ref_hop,
                     doppler_rate_hz_s=ctx.args.force_doppler_rate)
        fit = fit_cp_rate(
            ctx.cp_to_seed_currency(h, seed_dop,
                                float(seed.get("doppler_rate_hz_s", 0.0) or 0.0)),
            ctx.code_len)
        if fit is not None:
            rate, h0, cp_ref = fit
            # ── THE CODE-RATE CROSS-CHECK (#96, --cp-rate-model-tol) ──────────────────
            # The residual code rate is COMMON-MODE. propagate_seed feeds the geometry
            # forward itself (gnssSeedTransport: chips_per_hop scales by the sat's own
            # Doppler), so what is left for this seed to carry is the receiver's l-a
            # clock -- one number for every satellite on the chain. Satellite clock drift
            # (af1 ~ 1e-12 s/s) is three orders below it. Measured 2026-08-28: every
            # present gal/bds sat was commanded ONE value (0.0486 chips/s, SD 0.019 shared)
            # while gps_l5 was commanded a DIFFERENT value per satellite, SD 0.052-0.064,
            # 3x the scatter.
            #
            # WHY ONLY gps_l5, and it is NOT "the search chain vs the dead-reckon chains":
            # `dead-reckon` is set in the config's COMMON block, so all five chains run it.
            # The discriminator is `detectors`, which only gps_l5 has -- this stage is
            # stage_detections_to_seeds, so a chain with no detector never arrives here at
            # all (17304 cp-fits on gps_l5 that day against 4 on gal_e5a and 1 on bds_b2a).
            # The other four also carry `dr-clock-adopt`, so their code rate comes from the
            # adopted/joint clock as one shared number. gps_l5 has neither: it is the clock
            # MASTER the others adopt from (#75), so it cannot adopt, and its rate is
            # whatever this fit last returned.
            #
            # fit_cp_rate is unbounded -- it returns whatever least squares gives, and its
            # cp history is unwrapped mod code_len by nearest-wrap, so one mis-wrap injects
            # a whole code period into the slope: p90 2.16, p99 177, max 994 chips/s that
            # day. THE SIBLING FIT ALREADY HAS THIS GUARD (fit_dop_rate + the
            # --dop-rate-model-tol cross-check below it, added 2026-08-05 for exactly this
            # failure); the code rate never got one.
            #
            # Bound the DEVIATION FROM THE POOLED CLOCK, never the magnitude: the receiver
            # clock is ~0.047 chips/s calibrated but runs ~3.45 chips/s uncalibrated
            # (codeloop's feed-forward note), so an absolute ceiling would reject the very
            # feed-forward the trim cannot live without. The pooled l-a tracks whatever the
            # clock actually is; only the per-sat departure from it is unphysical.
            #
            # REJECT THE RATE, KEEP THE POSITION. cp_ref/h0 are a re-anchor and stay good
            # even when the slope is noise -- dropping the whole fit would trade a bad rate
            # for a stale phase.
            #
            # Measured on 2026-08-28's log, 15058 fits with an in-sample control and
            # stratified by pre-fit prompt/noise (the confounder: a fading sat makes a
            # noisy fit AND drops out on its own). At matched strength p/noise 5-15 the
            # 40 s dropout rate was 0.4% after a clean fit and 21.4% after a rejected one
            # -- and rejected-fit sats at 5-15 dropped MORE than CLEAN-fit sats that were
            # genuinely weaker (8.8% at p/noise 0-5), which is the ordering weakness alone
            # cannot produce. tol 0.5 sits mid-plateau: 0.3-1.0 all reject 2.3% of fits and
            # catch 34.3% of every gps_l5 dropout, so the number is not a tuned edge.
            # 0 disables the cross-check (the pre-2026-08-28 behaviour).
            _seed_rate = rate
            _tol = getattr(ctx.args, "cp_rate_model_tol", 0.0)
            if _tol > 0.0 and ctx.cb.code_ema is not None:
                _model = cp_rate_from_code_bias(seed_dop, ctx.cb.code_ema,
                                                ctx.args.hops_per_sec,
                                                ctx.args.chip_rate_hz, ctx.args.carrier_hz)
                _dev = (rate - _model) * ctx.args.hops_per_sec      # chips/s
                if abs(_dev) > _tol:
                    ctx.cp_rate_rejected[prn] = (rate * ctx.args.hops_per_sec,
                                                 _model * ctx.args.hops_per_sec)
                    _log_rl("cprate-rej-%d" % prn,
                            "PRN %d cp-rate REJECTED: fit %+.3f chips/s vs pooled clock "
                            "%+.3f (dev %+.3f > --cp-rate-model-tol %.3f) -- position kept, "
                            "clock rate seeded"
                            % (prn, rate * ctx.args.hops_per_sec,
                               _model * ctx.args.hops_per_sec, _dev, _tol),
                            every_s=60.0)
                    _seed_rate = _model
                    # ── #100 (--fit-flush-on-reject): a rejected fit is not just a bad
                    # RATE -- the seed's POSITION is the same fit EVALUATED at ref_hop,
                    # so a wrap-blown slope moves the command by (slope error x history
                    # span): SEEDAUDIT measured a +4067-chip step on G9 while the rate
                    # guard stood (2026-08-28 22:45). And the poison is self-sustaining:
                    # off-peak command -> weak detections -> re-poisoned history, for as
                    # long as the history remembers (~8 min at fit-hist-len 256). After
                    # N consecutive rejections the fit has left physics: flush the
                    # history and let the sat ride the birth path while a clean fit
                    # rebuilds (6 pts + 30 s). 0 disables.
                    _n = ctx.cpt.rej_streak.get(prn, 0) + 1
                    _flush_n = getattr(ctx.args, "fit_flush_on_reject", 0)
                    if _flush_n > 0 and _n >= _flush_n:
                        ctx.cpt.hist.pop(prn, None)
                        ctx.cpt.rej_streak.pop(prn, None)
                        _log("PRN %d cp-fit history FLUSHED: %d consecutive rejected "
                             "rates (last %+.2f chips/s vs clock %+.2f) -- wrap-poisoned "
                             "history dropped, sat rides the birth path while a clean "
                             "fit rebuilds"
                             % (prn, _n, rate * ctx.args.hops_per_sec,
                                _model * ctx.args.hops_per_sec))
                    else:
                        ctx.cpt.rej_streak[prn] = _n
                else:
                    ctx.cpt.rej_streak.pop(prn, None)
            # ── #103 (--cp-rate-model-primary): the fitted RATE becomes monitor-only and
            # the COMMAND rides the pooled-clock model rate for every fit, not just the
            # rejected ones -- gps_l5 joins the rate policy the four model-primary chains
            # already run. Measured basis (2026-08-30, TRACK-vs-MODEL census): gps_l5
            # tracks drift off-model at ~0.006-0.03 chips/s per-sat random-sign (the
            # fitted-rate supply's error), saturate the fleet trim's 1.25-chip ceiling in
            # minutes, and churn through escape/rebirth ~445x/night -- while the
            # model-primary chains' trim drift (0.09-0.26 mchips/s) shows the model rate
            # is ~100x better post-#99. The fit's POSITION is kept (the tol guard's own
            # rule: reject the rate, keep the position); the tol/flush machinery above
            # still protects the position against wrap-poisoned histories; and `rate`
            # still feeds fit_slope and the l-a pool below (measurements, not commands).
            # Cold start (code_ema None) falls back to the fitted rate, exactly as the
            # tol guard does.
            if (getattr(ctx.args, "cp_rate_model_primary", 0)
                    and ctx.cb.code_ema is not None):
                _seed_rate = cp_rate_from_code_bias(seed_dop, ctx.cb.code_ema,
                                                    ctx.args.hops_per_sec,
                                                    ctx.args.chip_rate_hz,
                                                    ctx.args.carrier_hz)
            # ⚠️ SUBSTITUTE THE COMMAND ONLY, NEVER THE MEASUREMENT. `rate` stays the
            # FITTED slope below this line, because the two consumers underneath are
            # measurements: ctx.cpt.fit_slope feeds CARRIER-FROM-CODE (a shadow), and
            # code_clock_bias_sample() contributes this satellite's l-a sample to the very
            # pool `_model` was computed from. Overwriting `rate` here would feed the clock
            # a sample derived from the clock -- a self-reference that reinforces whatever
            # the pool already believes, which is the mirror #33/GAP-2 was.
            seed.put("cp_fit", epoch=h0,
                     code_phase_rate=_seed_rate, ref_hop=h0, code_phase_chips=cp_ref)
            ctx.fitted.add(prn)
            ctx.cpt.fit_slope[prn] = rate * ctx.args.hops_per_sec   # chips/s, for CARRIER-FROM-CODE
            # This fit contributes an (l-a) sample: its code_frac minus the sat's carrier_frac.
            # Only strong, geometry-clean detections (SNR gate) -- weak/noisy slopes would bias it.
            la = code_clock_bias_sample(rate, seed_dop, ctx.args.hops_per_sec,
                                        ctx.args.chip_rate_hz, ctx.args.carrier_hz)
            # PER-SAMPLE gate: a single noisy/unwrap-blown slope fit is a large l-a outlier
            # that the few-sat median can't reject -- and a wandering pooled l-a swings the
            # seeded code rate (+-1 ppm = +-1 chip/s), walking the deep integration off-peak
            # within its ~1 s window (the 2026-07-07 L1 deep decay). Bound to --code-bias-max.
            if snr >= ctx.args.acquire_snr and abs(la) < ctx.args.code_bias_max * 1e-6:
                ctx.la_samples.append(la)
            _log_rl("cpfit-%d" % prn,
                    "PRN %d cp-fit: %.2f chips @ hop %d, slope %+.3f chips/s "
                    "(%d pts, l-a %+.3f ppm)"
                    % (prn, cp_ref, h0, rate * ctx.args.hops_per_sec, len(h), la * 1e6))
        # L2C CL TIME-ASSIST: the trackers despread the CL pilot, whose absolute code phase
        # is cp_CL = cp_CM + k*10230 with k the CL segment index -- COMPUTED, not searched.
        # CL's 1.5 s epoch is locked to GPS time in Z-count (1.5 s) units, and GPS-UTC (18 s)
        # and the GPS epoch (unix 315964800) are both exact multiples of 1.5 s, so unix time
        # mod 1.5 IS GPS time mod 1.5 (--cl-time-adjust is the escape hatch if that ever
        # changes). cp is absolute-anchored at capture sample 0, so evaluate the TRANSMIT
        # time of sample 0 (utc0_sample0 - range/c) and SNAP the predicted CL chip count to
        # the measured cp_CM (mod 10230): the code phase supplies the fine time; the wall
        # clock only picks the segment (needs ~10 ms absolute accuracy, and the logged
        # residual MEASURES the anchor+host-clock error). Second-order cp0/range drift keeps
        # this valid for ~hour-long runs.
        # MEASURED overlay alignment (search reports `nh`) beats the time-assist below.
        # Both cp0 and the seed are arguments in the same C = arg + n*cps convention, so
        # matching the primary phase AND the overlay index gives simply
        #     cp_long = cp_primary + nh * CODE_LEN   (mod LC_SEG*CODE_LEN)
        # -- no wall clock, no range model, no half-period accuracy requirement, and
        # per-satellite rather than one fitted constant for the whole sky.
        if cp_long >= 0.0 and ctx.lc_seg > 1:
            # The SEARCH already reduced this at the overlaid code's own length, so it
            # carries the period. Reconstructing it here -- from `nh`, or from absolute
            # time via --cl-assist -- means re-deriving a convention the search already
            # knows, which is where every previous attempt went wrong.
            seed.put("nh_lift", epoch=ref_hop,
                     code_phase_chips=((cp_long + ctx.args.nh_period_offset * ctx.code_len)
                                       % (ctx.lc_seg * ctx.code_len)))
            ctx.cl_report.append("PRN %d long-cp (search)" % prn)
            # And carry the PHASE at the search's own epoch. cp0 back-references to sample
            # 0 through a Doppler-scaled rate, which multiplies the reported Doppler's
            # error by ~5900 chips/Hz = 0.58 overlay PERIODS per Hz -- so the period that
            # survives that route is noise. A phase at its own epoch has no such lever.
            if cp_at_ref >= 0.0:
                LLc = ctx.lc_seg * ctx.code_len
                ph = cp_at_ref % LLc
                # PERIOD CONTINUITY IS A CHECK, NOT AN AUTHORITY (2026-08-02).
                #
                # This existed because the search could not report the overlay period: it
                # lifted by best_nh alone and dropped the coarse lag's whole-period count,
                # so the reported period was wrong ~half the time. Since 4371ff4eb the
                # search MEASURES it (from AcquisitionResult::peak_tau_samples), verified
                # 16/16 on injection and 9/9 on sky above snr 60. The source is now right,
                # so overriding it from history is strictly a way to be wrong.
                #
                # And this loop could only ever be wrong in one direction. It stored its
                # OWN correction in ph_hist and predicted the next pass from that, so a
                # single bad correction was permanent: every later measurement got snapped
                # into agreement with the original error. Self-consistent, absolutely
                # wrong -- and with no residual gate, a marginal detection whose phase is
                # noise still yielded a confident-looking integer m. Measured 2026-08-02:
                # PRN 21 was period-consistent at the source across three consecutive
                # detections and collected four "corrections" anyway.
                #
                # So: compute it, log the disagreement, apply NOTHING, and store the
                # MEASURED phase. That turns the resolver into a regression detector for
                # the search's period -- a nonzero m on a STRONG satellite now means the
                # source broke, which is worth an alarm rather than a silent repair.
                # --period-continuity correct restores the old override if ever needed.
                _nh_deferred = False
                prev = ctx.cpt.ph_hist.get(prn)
                if prev is not None and ctx.args.period_continuity != "off":
                    h0, ph0, dop0 = prev
                    dh = ref_hop - h0
                    gap_s = dh / ctx.args.hops_per_sec
                    if 0 < gap_s <= 900.0:
                        rate = (ctx.args.chip_rate_hz / ctx.args.hops_per_sec
                                * (1.0 + ctx.args.code_doppler_sign * 0.5 * (dop0 + dop)
                                   / ctx.args.carrier_hz))
                        ph_pred = (ph0 + dh * rate) % LLc  # NB not `pred` -- that is the
                        # almanac prediction dict in this scope, and shadowing it breaks
                        # the alias census a hundred lines down with a TypeError.
                        m = int(round(((ph_pred - ph) % LLc) / ctx.code_len)) % ctx.lc_seg
                        if m:
                            resid = ((ph + m * ctx.code_len - ph_pred + LLc / 2) % LLc) - LLc / 2
                            # Only a STRONG disagreement is evidence about the source; a
                            # marginal detection disagreeing tells us about the detection.
                            sev = ("SOURCE PERIOD DISAGREES"
                                   if snr >= ctx.args.period_check_snr else "weak det")
                            _log_rl("phcont-%d" % prn,
                                    "PRN %d period continuity %s: %+d periods "
                                    "(snr %.0f, gap %.0f s, residual %+.1f chips) "
                                    "-- NOT applied (%s)"
                                    % (prn, sev, m, snr, gap_s, resid,
                                       ctx.args.period_continuity),
                                    every_s=60.0)
                        if ctx.args.period_continuity == "correct":
                            ph = (ph + m * ctx.code_len) % LLc
                        elif m and ctx.args.nh_period_debounce > 0:
                            # ── THE PERIOD DEBOUNCE (#97, --nh-period-debounce) ──────────
                            # The search's measured overlay period TOGGLES +-1 on strong
                            # satellites (snr 284-413 measured 2026-08-28, fine phase right
                            # to ~0.5 chip, period alone wrong; ~37-49 alarms/h fleet-wide
                            # and present in the 08-19 archive). Each single-detection flip
                            # rewrote the seed by a whole code period, all 12 nodes stepped
                            # together (SEEDAUDIT +-10230-chip step-back pairs 2 s apart),
                            # q cratered for 1-2 polls, and the per-sat whole-period step
                            # rate ranked exactly with the observed q churn (G18 39/55min
                            # worst, G26 8 best).
                            #
                            # A CHANGED period must be confirmed by this many CONSECUTIVE
                            # detections (same m) before it is adopted. Until then the seed
                            # carries the MEASURED fine phase with the STANDING period --
                            # we never invent a phase, we only refuse to move the ambiguity
                            # integer on one detection's word.
                            #
                            # This is NOT the 2026-08-02 self-poisoning override that
                            # period_continuity == "check" retired. That loop stored its
                            # OWN correction in ph_hist, so one bad fix was permanent.
                            # Here, ph_hist is NOT updated while a period is pending
                            # (neither with the correction nor with the unconfirmed
                            # measurement -- storing the flipped phase is what made the
                            # NEXT, correct detection alarm and step back), and a real
                            # period change is adopted after N consecutive detections
                            # (~2N s at the revisit), so there is no deadlock at a wrong
                            # value. Mutually exclusive with "correct" by the elif.
                            _pm, _pc = ctx.cpt.nh_pending.get(prn, (0, 0))
                            _pc = _pc + 1 if _pm == m else 1
                            ctx.cpt.nh_pending[prn] = (m, _pc)
                            if _pc < ctx.args.nh_period_debounce:
                                ph = (ph + m * ctx.code_len) % LLc
                                _nh_deferred = True
                                _log_rl("phdeb-%d" % prn,
                                        "PRN %d period DEBOUNCED: measured %+d period(s) "
                                        "off the standing one (%d/%d consecutive) -- "
                                        "standing period kept, measured fine phase seeded"
                                        % (prn, m, _pc, ctx.args.nh_period_debounce),
                                        every_s=60.0)
                            else:
                                ctx.cpt.nh_pending.pop(prn, None)
                                _log("PRN %d period ADOPTED: %+d period(s), confirmed by "
                                     "%d consecutive detection(s)" % (prn, m, _pc))
                        elif not m:
                            ctx.cpt.nh_pending.pop(prn, None)
                # Feed history only from detections whose phase means something. Below the
                # bar the phase is noise (measured: snr < 60 gives ~2000-chip within-period
                # residuals against a few chips above it), and a noise entry poisons every
                # comparison until that PRN is seen again -- 90-270 s at CHORD's revisit.
                # A DEBOUNCE-DEFERRED period feeds nothing: the corrected phase is our own
                # word (the 2026-08-02 poison) and the measured one is unconfirmed.
                if not _nh_deferred and (snr >= ctx.args.period_check_snr
                                         or prn not in ctx.cpt.ph_hist):
                    ctx.cpt.ph_hist[prn] = (ref_hop, ph, dop)
                # --nh-period-offset: applied HERE, after the continuity check has had its
                # say, and to the phase rather than the argument -- propagate_seed prefers
                # phase_ref_chips whenever it is >= 0, so offsetting only code_phase_chips
                # would change nothing the tracker ever reads. ph_hist keeps the UNSHIFTED
                # phase so the continuity check still compares like with like.
                ph = (ph + ctx.args.nh_period_offset * ctx.code_len) % LLc
                seed.put("nh_lift", epoch=ref_hop, code_phase_at_ref_chips=ph)
        elif det_nh >= 0 and ctx.lc_seg > 1:
            seed.put("nh_lift", epoch=ref_hop,
                     code_phase_chips=((seed["code_phase_chips"] % ctx.code_len)
                                       + (det_nh % ctx.lc_seg) * ctx.code_len)
                                      % (ctx.lc_seg * ctx.code_len))
            ctx.cl_report.append("PRN %d nh=%d (measured)" % (prn, det_nh))
        elif ctx.args.cl_assist and ctx.utc0_sample0 and ctx.args.almanac and prn in ctx.pred:
            tau = ctx.pred[prn][3] / C_LIGHT
            cl_chips = (((ctx.utc0_sample0 - tau + ctx.args.cl_time_adjust) % ctx.lc_epoch)
                        * ctx.args.chip_rate_hz)
            cp_cm = seed["code_phase_chips"]
            k = int(round((cl_chips - cp_cm) / ctx.code_len))
            fine_ms = (cl_chips - cp_cm - k * ctx.code_len) / ctx.args.chip_rate_hz * 1e3
            seed.put("cl_assist", epoch=ref_hop,
                     code_phase_chips=(cp_cm + (k % ctx.lc_seg) * ctx.code_len)
                                      % (ctx.lc_seg * ctx.code_len))
            ctx.cl_report.append("PRN %d k=%d fine %+.1f ms" % (prn, k % ctx.lc_seg, fine_ms))
        # HOLD-ON-LOCK: once a PRN shows a real lock, FREEZE its cp anchor + rate and let the
        # DLL trim own the sub-chip residual. The search's per-fix cp is only good to ~1-2
        # chips (hop-resolution coarse + refine), so re-anchoring from the fit at every cycle
        # injects that jitter into the despread 5x per second -- the replay signature: the
        # incoherent amplitude flaps full-scale<->zero on seconds timescales and the deep
        # never builds past the lucky windows (measured 2026-07-11, 07-04 capture). Classic
        # acquisition->track handoff: the fit is for CAPTURE, the (frozen anchor + DLL) for
        # TRACKING. Doppler keeps refreshing (carrier fence semantics unchanged); on lock
        # loss the coast/drop path clears the seed and re-acquisition seeds fresh from the fit.
        # The seed's (cp0, rate, ref_hop, doppler) form the tracker's code CURRENCY: cp0 is
        # an absolute-sample-0 phase whose meaning shifts by t_abs*f_chip/f_c chips PER HZ
        # of doppler change. The currency must therefore be PIECEWISE-CONSTANT while
        # locked -- a merely-smooth doppler (detection grid jitter, clock-bias EMA wobble)
        # multiplies t_abs and walks/jitters the despread off-peak (blind replay: +-12 Hz
        # grid jitter -> +-0.5 chip at t=60 s; live soak: 0.2 Hz EMA wobble -> 0.5 chip at
        # t=1 h). Freeze the WHOLE tuple; release on amp collapse or when the frozen
        # doppler goes stale enough to decohere the 1-record despread itself.
        prev = ctx.seeds.get(prn)
        # TRACK-vs-SEARCH consistency (the hold's independent referee): the freeze
        # trusts the DLL to own the sub-chip residual, but a sharp-ACF (BOC)
        # discriminator has a FINITE capture range (~1/3 chip at 0.5-eff-chip
        # spacing) with further STABLE equilibria beyond it (prompt R ~ -0.25,
        # -12 dB). An excursion past capture (a jittery era, one bad re-anchor)
        # leaves the DLL contentedly servoing the WRONG lobe: deep stays coherent
        # (and floor-certified), amp bleeds ~12 dB, and the hold never lets go --
        # observed 2026-07-12 pm: C34 amp 440->0 over 35 min at deep 100+, the
        # track parked ~0.75 chips off while the search kept finding the true
        # peak at >100 sigma. The search FIT (sub-0.1-chip stable at 20 MSPS) is
        # the referee: persistent disagreement beyond the capture range releases
        # the freeze so the seed re-anchors on the true peak.
        cp_err = None
        # #76/#83 2(d): the EFFECTIVE trim -- what the tracker is actually adding to
        # the model phase right now: the Python slow integrator's value PLUS the C++
        # fleet loop's standing trim, read back last cycle (#76; ~one cycle old, since
        # the readback rides the policy block which runs after this loop). The handover
        # (180429f07) means at most one arm is normally nonzero per PRN, but the sum is
        # correct in every regime, including the release ramp when both exist. Until
        # #76 every consumer here used dll_trim alone: an armed PRN was judged as if
        # untrimmed while up to 3 chips of command stood at the trackers.
        _trim_eff = (ctx.dls.trim.get(prn, 0.0)
                     + ((ctx.dls.readback.get(prn) or {}).get("trim_chips", 0.0)))
        # THE DETECTION'S OWN PHYSICAL PHASE at its epoch: cp0 and dop were published
        # together, so undoing the pair reintroduces no translation -- which is the
        # entire #42 fix. (cp_at_ref would be better conditioned but lives in the C++
        # last-sample convention and carries the anchor Doppler term; see
        # track_vs_fit_chips.) Hoisted out of the hold branch: the innovation below
        # wants it for every detection.
        _cpe_recon = (cp + (ref_hop / ctx.args.hops_per_sec) * ctx.args.chip_rate_hz
                      * (1.0 + ctx.args.code_doppler_sign * dop / ctx.args.carrier_hz)
                      ) % ctx.code_len
        # ── #83 2(d): THE INNOVATION, computed for EVERY accepted detection ──
        # Measurement minus forecast (chips, wrapped to one period), evaluated BEFORE
        # this cycle overwrites the seed with the very detection being judged. SERVED
        # (log + publisher rows) and, when the PRN is held, it IS the escape referee's
        # statistic below -- one number, two consumers, so the referee and the
        # published innovation can never disagree. This is also the number Phase 3's
        # per-PRN model-primacy gate reads (p95 |innov| < ~1 chip over 10 min) and the
        # one-controller carrier design's code-side measurement.
        _inv = None
        # ⚠️ DR-OWNED SEEDS ARE EXCLUDED (2026-08-17 13:20, found by the flip's first
        # arm): their ref_hop is stamped from WALL time (int(t_now_abs*hops_per_sec))
        # while a detection's ref_hop rides the F-ENGINE hop counter -- a cross-axis
        # dh, so the "innovation" reads the wall-vs-F-engine offset's sub-second part
        # times the chip rate (ms -> thousands of chips; integer seconds vanish, 1 s =
        # 100.0 periods exactly). Measured on flipped G26: INNOV p95 2598 while MINNOV
        # p95 1.27, q 3.12 and trim -1.4 all said the tap was fine. The same axis
        # mismatch is the leading suspect for the 2(b) DR-chain audit steps. MINNOV is
        # the dr-owned satellites' referee; INNOV resumes when the search re-anchors.
        # Under --dr-fengine-axis the dr stamps ride the F-engine axis and
        # --innov-dr-seeds re-admits them (both flags, or the exclusion stands).
        if (prev is not None
                and ((ctx.args.innov_dr_seeds and ctx.args.dr_fengine_axis)
                     or ctx.dr_state is None or prn not in ctx.dr_state["seeded"])
                and all(k in prev for k in ("code_phase_chips", "code_phase_rate",
                                            "ref_hop", "doppler_hz"))):
            # FORECAST WHAT THE TRACKER RUNS, not the cp0 fiction. The first deploy of
            # this block used track_vs_fit_chips (= dr_seed_phys, the cp0-argument
            # path): every PRN read thousands of chips, sign-flipping, p95 ~5000 --
            # wrap-uniform, two uncorrelated phases -- while q and the trims said the
            # taps were fine. The tracker prefers code_phase_at_ref_chips
            # (propagate_seed), and seed writers legitimately move doppler_hz without
            # re-projecting cp0, so the cp0-implied phase swings by the t_abs lever
            # (~5600 chips/Hz). tracker_phase_at picks the same reference
            # propagate_seed does (#45 step 7 -- same lesson as #43's 90,000-chip
            # fiction). The measurement moves to the same LAST-SAMPLE convention with
            # its OWN doppler (the hop-epoch convention: 52.37 chips if mixed).
            _fc = tracker_phase_at(prev, ref_hop, ctx.args.hops_per_sec,
                                   ctx.args.chip_rate_hz, ctx.args.carrier_hz,
                                   ctx.args.code_doppler_sign, ctx.code_len,
                                   ctx.args.search_fft_len or None)
            _hop_off_det = (ctx.args.chip_rate_hz / ctx.args.hops_per_sec
                            * (1.0 + ctx.args.code_doppler_sign * dop
                               / ctx.args.carrier_hz))
            if ctx.args.search_fft_len:
                _hop_off_det *= 1.0 - 1.0 / ctx.args.search_fft_len
            _inv = ((_cpe_recon + _hop_off_det - _fc - _trim_eff + ctx.code_len / 2.0)
                    % ctx.code_len) - ctx.code_len / 2.0
            _ih = ctx.innov_hist.setdefault(prn, [])
            _ih.append((ctx.t0, _inv))
            # Bounds MEMORY only: the 10-minute statistic is cut by time at read
            # (the publish block), so this cap can never shorten the window.
            del _ih[:-120]
        # ── #83 P3-3b: MODEL-PRIMARY PRNs stop here ──
        # This detection has already done everything it is allowed to do: the #79
        # deep-gate stamp (above), INNOV (above), and the joint feed + MINNOV run in
        # their own block from `offs`. The SEED stays the MODEL's -- the dr slew path
        # owns it via the eligibility exceptions there -- so the re-anchor, the
        # hold/escape machinery and the commit below are all bypassed. The search
        # remains the referee THROUGH MINNOV: its exit gate (p95 hysteresis +
        # starvation) hands the PRN back, and the next detection re-anchors normally.
        if prn in ctx.mp_flipped:
            ctx.mp_last_det[prn] = ctx.t0
            continue
        if (prev is not None and prn in ctx.cp_held
                and all(k in seed for k in ("code_phase_chips", "code_phase_rate",
                                            "ref_hop"))):
            # AT-EPOCH COMPARISON (#42 -> #45 step 1, 2026-08-12). The sample-0
            # currency comparison that lived here manufactured -t_abs*k*d(clock_bias)
            # chips of phantom whenever the seed-vs-detection dop bias moved (an EMA
            # DESIGNED to move): 145 CP_ERR-DECOMP specimens in 7 min, seven false
            # ESCAPES in one evening against tracks healthy at 40 dB-Hz. The
            # candidate tuple reaching this point can be PAIR-INCONSISTENT (label
            # stepped, cp0 unmoved) and no translation survives that. So: compare
            # physical phases at the DETECTION's epoch instead -- the search's own
            # cp_at_ref (dt=0, its Doppler estimate does not enter) against where
            # dr_seed_phys puts the tracker's despread. test_track_vs_fit.py pins
            # the discriminating pair (bias step: old ~1700/Hz, new ~0; real lobe
            # park: both read +-3.27) and drives the SHIPPED function.
            # #83 2(d) ADDENDUM (2026-08-17): the referee now CONSUMES THE INNOVATION
            # above -- forecast by tracker_phase_at, the reference propagate_seed
            # actually prefers -- instead of track_vs_fit_chips' dr_seed_phys (the
            # cp0-argument path). On sky the cp0 fiction read wrap-uniform thousands
            # of chips (p95 ~5000) on satellites tracking cleanly: seed writers move
            # doppler_hz without re-projecting cp0, and the implied phase swings on
            # the t_abs lever (~5600 chips/Hz). That is the named mechanism behind
            # the +-700-4500-chip CP_ERR reports of 08-16/17 -- the referee had been
            # accusing healthy tracks with a number drawn from a uniform
            # distribution, and only the sign-consistency + median gates kept it
            # from escaping constantly. Confirmed as a discriminating pair on the
            # holds fixture: same replay, forecast swapped, innovations collapse to
            # p95 1.6-2.3 chips.
            cp_err = _inv
            if cp_err is not None and abs(cp_err) > ctx.args.hold_max_cp_err:
                _log_rl("cperr-%d" % prn,
                        "CP_ERR PRN %d: %+.2f chips at det hop %d (at-epoch: "
                        "search cp_at_ref vs held propagation; trim %+.2f "
                        "= py %+.2f + cpp %+.2f, hold_age %.0f s)"
                        % (prn, cp_err, ref_hop, _trim_eff,
                           ctx.dls.trim.get(prn, 0.0),
                           (ctx.dls.readback.get(prn) or {}).get("trim_chips", 0.0),
                           (ref_hop - prev["ref_hop"]) / ctx.args.hops_per_sec),
                        every_s=60.0)
            if cp_err is not None:
                ctx.cpt.err_hist.setdefault(prn, []).append(cp_err)
                del ctx.cpt.err_hist[prn][:-9]
            # MEDIAN GATE (2026-07-19): the per-detection cp noise is 0.03-0.5 chips
            # (per-sat conditions -- multipath/BOC refine; measured same-instrument at
            # t_abs 100 s AND 27000 s, i.e. FLAT in run age: the earlier 'growth law'
            # was the logged cp_ref coordinate wobbling with dop_seed x t_abs, which
            # the currency translation above cancels in cp_err by construction). The
            # 5-consecutive-sign rule alone still lets a noisy-conditions sat sustain
            # a false accusation; a 9-sample median cannot be dragged over the bar by
            # single-point noise, only by a persistent physical walk.
            ctx.cpt.err_hist.setdefault(prn, []).append(cp_err)
            del ctx.cpt.err_hist[prn][:-9]
        # Count only SIGN-CONSISTENT disagreements: a real wrong-lobe park is
        # one-signed; search-fit noise on a weak sat alternates (first deploy:
        # weak GPS sats escaped every minute on -0.5/+1.3/-0.5 flip-flops, and
        # each escape re-injects the seed jitter + forces an overlay re-align).
        # FIT-QUALITY gate (2026-07-12 evening): only a trustworthy fit may accuse the
        # track -- >=6 history points (fit noise ~ per-fix/sqrt(n)) and a solid current
        # detection (2x the acquire gate). Ungated, weak-sat fit noise drove 627 escapes
        # in 2.7 h and each one re-anchored the seed (the churn behind the GPS "wobble").
        # + FIT SPAN >= 30 s (2026-07-19 eve): point COUNT is not maturity -- L2C's
        # snapshots arrive fast enough that 6 points span ~13 s, over which the code-
        # Doppler QUADRATIC is unresolvable, so the fit carries the curvature-bias class
        # (the birth zombies the watchdog had to keep cleaning). A 30 s floor makes the
        # curvature term observable on every chain; on L1 (6 points ~ 60-80 s) it is a
        # no-op. This gate feeds BOTH the escape referee and hold admission.
        fit_span_s = ((h[-1][0] - h[0][0]) / ctx.args.hops_per_sec) if len(h) >= 2 else 0.0
        fit_trusted = (fit is not None and len(h) >= 6
                       and fit_span_s >= ctx.args.fit_maturity_span_s
                       and snr >= 2.0 * ctx.args.acquire_snr)
        # AMP VETO (see --escape-amp-veto): a full-amplitude hold is on the main peak
        # by construction -- refuse the fit's accusation rather than drag it off.
        amp_now = float((ctx.status.get(prn) or {}).get("amp_snr", 0) or 0)
        amp_veto = (ctx.args.escape_amp_veto > 0.0
                    and amp_now > ctx.args.escape_amp_veto)
        # INTEGRITY VETO (2026-07-19 eve, audit follow-up): never re-anchor onto a fit
        # the BRDC model itself disputes. The dead-reckon machinery already computes a
        # per-sat integrity residual (search-vs-model, solved clock removed, normally
        # +-0.2 chips): if a FRESH residual says the search's own position is off by
        # more than the escape bar, the fit is the suspect, not the track.
        integ_veto = False
        if ctx.dr_state is not None and ctx.dr_state.get("integ"):
            _iv = ctx.dr_state["integ"].get(prn)
            if _iv is not None and ctx.t0 - _iv[1] < 10.0:
                _iv_dev = _iv[0]
                # #98/#99 RELATIVE FORM (--integ-veto-baseline-s): a CHRONIC per-sat model
                # offset (#99, +-5 chips) is that sat's NORMAL, not evidence against this
                # fit -- the absolute test vetoed G28's escape all evening while its bad
                # hold walked +28 chips. Judge the EXCURSION from the sat's own recent
                # median; a real search lobe-jump still moves integ instantly. Falls back
                # to the absolute test until >=5 baseline samples exist (cold start).
                _bl_s = ctx.args.integ_veto_baseline_s
                if _bl_s > 0.0:
                    _ih = ctx.cpt.integ_hist.setdefault(prn, [])
                    if not _ih or _ih[-1][0] != _iv[1]:
                        _ih.append((_iv[1], _iv[0]))
                        del _ih[:-64]
                    _base = [v for (_t, v) in _ih if ctx.t0 - _t <= _bl_s]
                    if len(_base) >= 5:
                        _iv_dev = _iv[0] - statistics.median(_base)
                if abs(_iv_dev) > ctx.args.hold_max_cp_err:
                    integ_veto = True
        cp_err_med_ok = (cp_err is not None and len(ctx.cpt.err_hist.get(prn, [])) >= 5
                         and abs(statistics.median(ctx.cpt.err_hist[prn]))
                         > ctx.args.hold_max_cp_err)
        if (cp_err is not None and abs(cp_err) > ctx.args.hold_max_cp_err
                and cp_err_med_ok and fit_trusted and not amp_veto
                and not integ_veto):
            n_prev = ctx.cpt.escape.get(prn, 0)
            same_sign = (n_prev == 0) or (cp_err * ctx.cpt.escape_sign.get(prn, 0.0) > 0)
            ctx.cpt.escape[prn] = n_prev + 1 if same_sign else 1
            ctx.cpt.escape_sign[prn] = cp_err
        else:
            ctx.cpt.escape[prn] = 0
        # ⚠ CHAIN HAZARD (2026-08-28): this monitor lived BETWEEN the escape `if`
        # below and the hold `elif` from 2026-07-20 to 2026-08-28, silently re-chaining
        # the elif to THIS condition: whenever a held sat had a fresh detection and
        # dr integrity was populated, freeze/translate/release all skipped and the seed
        # became a per-detection re-anchor (G28's q drops, 08-28 evening). It sits ABOVE
        # the escape `if` now, as a standalone statement, and must never be moved into
        # that chain. test_holdchain.py pins the structure.
        # TRACK-vs-MODEL MONITOR (2026-07-20, log-only; audit follow-up census): the
        # referee's reference is the search FIT; the model-referenced track residual
        # is r_i - cp_err (search-vs-model minus search-vs-track, same chip units).
        # Log when the MODEL says the track is past the escape bar -- especially when
        # the fit-referenced referee stays quiet (veto / immature fit): those are the
        # cases an upgraded model-referenced referee would catch. Decide on enforcement
        # from this census, not from theory (the referee has bitten guessers before).
        if (cp_err is not None and ctx.dr_state is not None and ctx.dr_state.get("integ")):
            _iv2 = ctx.dr_state["integ"].get(prn)
            if _iv2 is not None and ctx.t0 - _iv2[1] < 10.0:
                _tm = _iv2[0] - cp_err
                if abs(_tm) > ctx.args.hold_max_cp_err:
                    _log_rl("tvm-%d" % prn,
                            "TRACK-vs-MODEL PRN %d: %+.2f chips past the escape bar "
                            "(fit-ref cp_err %+.2f, integ %+.2f; fit-referee %s) -- "
                            "monitor only"
                            % (prn, _tm, cp_err, _iv2[0],
                               "AMP-VETOED" if amp_veto else
                               "INTEG-VETOED" if integ_veto else
                               "fit-untrusted" if not fit_trusted else "active"),
                            every_s=120.0)
        if ctx.cpt.escape.get(prn, 0) >= 5:
            _log("ESCAPE PRN %d: track %+.2f chips off the search fit (5 consecutive,"
                 " sign-consistent) -> release hold + DLL trim, re-anchor on the fit"
                 % (prn, cp_err))
            ctx.cpt.escape[prn] = 0
            ctx.cpt.err_hist.pop(prn, None)
            ctx.dls.trim.pop(prn, None)
            ctx.dls.last.pop(prn, None)
            ctx.cp_held.discard(prn)
            ctx.hold.miss.pop(prn, None)
            # The re-anchor refreshes the seed doppler next cycle = an NCO f_ref step
            # the TRACK-mode trim was not built for (same latch as the hold release,
            # and it bypasses that branch because cp_held is discarded HERE): demote
            # to BOOTSTRAP so the carrier re-pulls instead of parking off-frequency.
            if prn in ctx.car.locked:
                ctx.car.locked.discard(prn)
                ctx.car.fade.pop(prn, None)
                _log("CARRIER REACQ PRN %d: escape re-anchor -> BOOTSTRAP re-pull" % prn)
        # HOLD ADMISSION REQUIRES FIT MATURITY (2026-07-19 eve, the Tier-3 burn-in fix):
        # a birth-window anchor (wide margins, unsolved bias, <6-point fit) can be chips
        # wrong, and granting it hold protection created the zombie cohorts that made
        # every relaunch take 5-20 min to heal (or forever, pre-watchdog: the L2C 18%
        # mornings). Until the sat's own cp fit is trusted -- the SAME predicate the
        # escape referee requires before it may accuse a track -- the seed keeps riding
        # the maturing fit (the weak-sat path, measured born-clean tonight), which is
        # self-correcting. Only a mature anchor earns protection; expected burn-in
        # collapses to the fit-maturation time (~6 search snapshots, ~60-80 s).
        # Already-held sats are unaffected (the cp_held alternative below).
        elif (prev is not None
                and ((ctx.sig_of_last(ctx.status.get(prn)) >= ctx.args.hold_snr
                      and (prn in ctx.cp_held or fit_trusted))
                     # ── #96/#97 CLOSURE: HOLD ON THE LOCK STATISTIC (--hold-on-present) ──
                     # The freeze below IS the architecture -- frozen tuple, DLL owns the
                     # residual, CP_ERR referee -- but amp_snr rides the coherent arc,
                     # which flickers with the deep fold (#58), so locked satellites sat
                     # un-held taking per-detection REPLACEs. Presence here is the
                     # population-honest fleet gate (ctx.qpop, the series that admits
                     # trims): the sat this branch protects is exactly the sat whose
                     # trims the fleet controller is already trusting. fit_trusted still
                     # required: a birth-window anchor must mature before it earns
                     # protection (the 2026-07-19 zombie-cohort lesson).
                     or (ctx.args.hold_on_present > 0 and fit_trusted
                         and _present_streak(ctx, prn) >= ctx.args.hold_on_present)
                     or (prn in ctx.cp_held and ctx.hold.miss.get(prn, 0) < 3))):
            # PERSISTENT-loss release (2026-07-12 evening): a single blank/stale status
            # read (sig 0.0 -- a poll racing the emit, a slow combiner cycle) used to
            # release the hold instantly: 562 of 736 releases in 2.7 h fired at
            # amp_snr < 8, mostly 0.0, and every release re-fit the seed (dop jump) and
            # paid the ~0.9 Hz x ~5 s carrier re-anchor transient -- the 2 s-median
            # churn behind the GPS coherence wobble (settled sats measure 0.06 Hz).
            # A held sat now rides through up to 3 consecutive sub-gate reads; doppler
            # STALENESS still releases immediately (real currency decoherence).
            if ctx.sig_of_last(ctx.status.get(prn)) >= ctx.args.hold_snr:
                ctx.hold.miss[prn] = 0
            elif ctx.args.hold_on_present > 0 and _present_streak(ctx, prn) >= 1:
                # Presence sustains a hold the sig path would starve: the sig read races
                # the emit (562/736 releases fired at sig ~0.0, 2026-07-12), while the
                # fleet gate is the statistic the trims already ride.
                ctx.hold.miss[prn] = 0
            else:
                ctx.hold.miss[prn] = ctx.hold.miss.get(prn, 0) + 1
            # ── #103 ROOT FIX (--hold-rate-from-clock): A HOLD FREEZES THE MEASUREMENT,
            # NOT THE PHYSICS. The la_rate writer refreshes every non-held sat's residual
            # code rate from the pooled clock each cycle and SKIPS held sats ("changing
            # the rate under a stale ref_hop jumps the extrapolated cp" -- true, and that
            # exemption is what let holds dead-reckon on whatever residual rate they
            # froze: measured 0.02-0.04 chips/s of LINEAR cp_err growth inside hold
            # episodes, the C++ trim chasing to its 1.25-chip ceiling, escape at ~3.3
            # chips, immediate re-hold freezing the next slightly-wrong rate -- gps_l5's
            # 445-escape/night churn. The four model-primary chains are immune because
            # the rates their holds freeze ARE the model's.) So refresh WITH the full
            # currency dance instead of skipping: re-express the anchor at the PRESENT
            # epoch under the OLD labels -- pure arithmetic, command-continuous, nothing
            # measured is adopted, so the hold's contract stands -- then swap in the
            # clock's residual rate. All three transport directions come from fits.py's
            # adjacent trio (dr_seed_phys / dr_cp0 / seed_phase_at_ref) so the
            # conventions cannot drift apart.
            # v2 (--hold-rate-source dr): v1 sourced this from the pooled l-a/joint
            # estimate and was REVERT-TRIGGERED within 30 min (2026-08-30 13:26-13:57):
            # that estimate swings +-50 mchips/s on minute timescales -- fine as a
            # per-cycle seed for measurement-anchored sats, but a HELD sat dead-reckons
            # on the rate for 30-60 s stretches and INTEGRATES the jitter (G18's q went
            # to noise; PRN 23 oscillated at 20-30 s). THE BAR, derived: a held sat
            # escapes when the C++ trim ramp hits its 1.25-chip ceiling at
            # t = 1.25/rate_err, so a 10-min hold needs the held rate good to
            # ~2 mchips/s. Sources measured that day: pooled l-a +-50 mchips/s; DR clock
            # drift sd 6.9, cycle-to-cycle median 0.4 mchips/s. So v2 slaves held rates
            # to the DR CLOCK DRIFT and SLEW-BOUNDS each refresh to 5 mchips/s: real
            # clock wander (slow) is tracked, transients (dr p90 12 mchips/s) are capped,
            # steady-state injected jitter is the source's own ~0.4 mchips/s.
            _hrs = getattr(ctx.args, "hold_rate_source", "none")
            if (_hrs == "dr" and ctx.dr_state is not None
                    and ctx.dr_state.get("drift") is not None
                    and ctx.dr_state.get("clk") is not None
                    and prev.get("ref_hop") is not None and ref_hop > prev["ref_hop"]):
                # dr drift is the receiver-clock code drift in chips/s; the seed's
                # code_phase_rate is the same RESIDUAL in chips/hop (the replica applies
                # the geometric code Doppler itself -- cp_rate_from_code_bias's note).
                _hr_tgt = ctx.dr_state["drift"] / ctx.args.hops_per_sec
                _hr_cur = prev.get("code_phase_rate", 0.0)
                _hr_d = (_hr_tgt - _hr_cur) * ctx.args.hops_per_sec  # chips/s
                _slew = 0.005 / ctx.args.hops_per_sec               # 5 mchips/s per refresh
                _hr_new = _hr_cur + max(-_slew, min(_slew, _hr_tgt - _hr_cur))
                # Plausibility bound (chips/s) on the TARGET: a dr drift further than this
                # from the held rate is the estimator misbehaving, not the satellite.
                # Skip, keep the frozen rate, and the referee still stands behind it.
                if abs(_hr_d) <= 0.5 and _hr_new != _hr_cur:
                    # ⚠️ RE-EXPRESS THE STREAM THE TRACKER READS (#45 step 7 / #43): the
                    # tracker prefers the at-ref phase over the cp0 argument, and the two
                    # come from different broker paths. Starting the re-expression from
                    # the cp0 side while the tracker reads the phase would snap the
                    # command by their disagreement -- the continuity gate in
                    # test_track_vs_fit.py::test_hold_retag_continuity caught exactly
                    # that in this fix's first draft.
                    _fft = getattr(ctx.args, "search_fft_len", 0) or None
                    _t_now = ref_hop / ctx.args.hops_per_sec
                    _ar = prev.get("code_phase_at_ref_chips")
                    if _ar is not None and _ar >= 0.0:
                        # last-sample commanded phase at the present, from the preferred
                        # reference -- becomes the new at-ref field verbatim
                        _ph_last = tracker_phase_at(prev, ref_hop, ctx.args.hops_per_sec,
                                                    ctx.args.chip_rate_hz,
                                                    ctx.args.carrier_hz,
                                                    ctx.args.code_doppler_sign,
                                                    ctx.code_len, _fft)
                        _per_hop = (ctx.args.chip_rate_hz / ctx.args.hops_per_sec
                                    * (1.0 + ctx.args.code_doppler_sign
                                       * prev["doppler_hz"] / ctx.args.carrier_hz))
                        _hop_off = _per_hop * (1.0 - 1.0 / _fft) if _fft else _per_hop
                        _phys_first = (_ph_last - _hop_off) % ctx.code_len
                        prev.put("hold_retag", epoch=ref_hop,
                                 code_phase_chips=dr_cp0(_phys_first, _t_now,
                                                         prev["doppler_hz"],
                                                         ctx.args.chip_rate_hz,
                                                         ctx.args.carrier_hz,
                                                         ctx.args.code_doppler_sign,
                                                         ctx.code_len),
                                 code_phase_at_ref_chips=_ph_last,
                                 code_phase_rate=_hr_new,
                                 ref_hop=ref_hop)
                    else:
                        # argument-branch tuple: the cp0 stream IS what the tracker reads
                        _phys_first = dr_seed_phys(prev, ref_hop, ctx.args.hops_per_sec,
                                                   ctx.args.chip_rate_hz,
                                                   ctx.args.carrier_hz,
                                                   ctx.args.code_doppler_sign,
                                                   ctx.code_len)
                        prev.put("hold_retag", epoch=ref_hop,
                                 code_phase_chips=dr_cp0(_phys_first, _t_now,
                                                         prev["doppler_hz"],
                                                         ctx.args.chip_rate_hz,
                                                         ctx.args.carrier_hz,
                                                         ctx.args.code_doppler_sign,
                                                         ctx.code_len),
                                 code_phase_rate=_hr_new,
                                 ref_hop=ref_hop)
                    if abs(_hr_d) > 0.005:
                        _log_rl("holdrate-%d" % prn,
                                "HOLD-RATE PRN %d: residual rate %+.4f -> %+.4f chips/s "
                                "(dr-drift target %+.4f, slew-bounded; anchor "
                                "re-expressed at the present, command continuous)"
                                % (prn, _hr_cur * ctx.args.hops_per_sec,
                                   _hr_new * ctx.args.hops_per_sec,
                                   _hr_tgt * ctx.args.hops_per_sec), every_s=300.0)
            ddop = seed["doppler_hz"] - prev["doppler_hz"]
            # SAFETY NET (design (b)): bound a single cycle's Doppler move. A real MEO
            # Doppler moves <1 Hz per 0.2 s cycle; this only fires on a bad model, and it
            # bounds the damage rather than forbidding motion.
            if abs(ddop) > ctx.args.dop_max_rate_hz:
                if prn not in ctx.cpt.dop_clamped:
                    ctx.cpt.dop_clamped.add(prn)
                    _log("DOP-CLAMP PRN %d: model wanted %+.1f Hz in one cycle (max %.1f)"
                         " -- clamping. A real MEO moves <1 Hz/cycle: SUSPECT THE MODEL."
                         % (prn, ddop, ctx.args.dop_max_rate_hz))
                ddop = math.copysign(ctx.args.dop_max_rate_hz, ddop)
                seed.put("dop_clamp", doppler_hz=prev["doppler_hz"] + ddop)
            # DESIGN (b): translate EVERY cycle (no fence). The freeze branch survives only
            # for --no-dop-continuous, and for the zero-motion case where it is a no-op.
            if (not ctx.args.dop_continuous and abs(ddop) <= ctx.args.hold_max_dop_hz) or ddop == 0.0:
                # Currency frozen: the whole tuple rides unchanged.
                seed.put("hold_freeze", epoch=prev["ref_hop"],
                         doppler_hz=prev["doppler_hz"],
                         code_phase_chips=prev["code_phase_chips"],
                         code_phase_rate=prev["code_phase_rate"],
                         ref_hop=prev["ref_hop"])
                # #80 FIX (2026-08-16): the at-ref phase is PART of the tuple and the
                # tracker PREFERS it (gnssSeedTransport.cpp:325 -- phase_ref_chips >= 0
                # wins over cp_chips unconditionally). Leaving the DETECTION's fresh
                # phase beside the frozen ref_hop commanded the despread off-peak by
                # the full inter-snapshot code advance (~4665 chips mod-period per
                # 6.29 s revisit, wrapping): measured on sky 2026-08-16 23:15 --
                # CP_ERR +1005..+4516 chips on ALL six held sats at hold_age 12-15 s,
                # prompt amps collapsed 21-77 -> 3-16 while the re-searching deep fold
                # kept sig above the release bar, so the mispaired hold NEVER let go
                # (#48's "prompt on noise", mechanism). Freeze means freeze: the phase
                # rides from prev (where it is paired with prev's ref_hop), or ships
                # not at all -- the tracker then falls back to the frozen (cp0, dop)
                # argument pair, which is the hold's original contract.
                if "code_phase_at_ref_chips" in prev:
                    seed.put("hold_freeze", epoch=prev["ref_hop"],
                             code_phase_at_ref_chips=prev["code_phase_at_ref_chips"])
                else:
                    seed.pop("code_phase_at_ref_chips", None)
            else:
                # ---- CURRENCY TRANSLATION (2026-07-14): a DOPPLER UPDATE IS NOT A LOSS
                # OF LOCK. The replica is anchored at sample 0, so the tracker builds the
                # code phase as
                #     cp(t) = cp0 + t*f_chip*(1 + sign*dop/f_carrier)
                # -- the Doppler is applied over the ENTIRE elapsed time, RETROACTIVELY.
                # cp0 is therefore not "the code phase": it is a coordinate that only
                # means anything PAIRED WITH A DOPPLER. Change dop by ddop and leave cp0
                # alone and the physical code jumps by t*f_chip*ddop/f_carrier -- ~20
                # chips for a 10 Hz fence step at an hour of run age, and the lever arm
                # GROWS with t.
                #
                # Historically the fence RELEASED the hold and re-anchored cp on the
                # search fit. The 20-chip currency jump was bookkeeping and harmless; the
                # damage was throwing away a good anchor and replacing it with a MEASURED
                # one. The fit's noise is nothing on GPS's broad triangular peak, but a
                # BOC(1,1) peak is ~3x sharper: measured 2026-07-14, C19 collapsed 42 ->
                # 20 dB-Hz and took ~5 s of DLL re-convergence, EVERY
                # hold_max_dop_hz/|dop_rate| = 10 Hz / 0.55 Hz/s = 18 s. ~25% of every
                # emit, on every locked BOC satellite, on all three constellations.
                #
                # So: keep the anchor and re-express it in the new Doppler's currency.
                # Pure arithmetic, no measurement, no fit noise, nothing for the deep
                # integration to trip over. The new slope is the BETTER one (dop is a
                # better estimate of the truth), so code_phase_rate is left alone -- only
                # the retroactive part needs absorbing.
                t_now = ref_hop / ctx.args.hops_per_sec
                # doppler_hz keeps its NEW value -- that is the point -- and is
                # re-attributed here because the translation is what makes the
                # (cp0, dop) pair valid at the shipped ref_hop by construction.
                seed.put("translate", epoch=prev["ref_hop"],
                         code_phase_chips=(
                             prev["code_phase_chips"]
                             - t_now * ctx.args.chip_rate_hz * ctx.args.code_doppler_sign
                             * ddop / ctx.args.carrier_hz) % ctx.code_len,
                         code_phase_rate=prev["code_phase_rate"],
                         ref_hop=prev["ref_hop"],
                         doppler_hz=seed["doppler_hz"])
                # #80 FIX, translate arm (the LIVE arm under --dop-continuous): same
                # as the freeze arm above. prev's at-ref phase is a PHYSICAL phase at
                # prev's ref_hop -- a doppler update does not move it (the new doppler
                # enters the forward propagation, not the anchor), so it rides
                # unchanged where the fresh detection's phase must not.
                if "code_phase_at_ref_chips" in prev:
                    seed.put("translate", epoch=prev["ref_hop"],
                             code_phase_at_ref_chips=prev["code_phase_at_ref_chips"])
                else:
                    seed.pop("code_phase_at_ref_chips", None)
                if prn not in ctx.cpt.translated:
                    ctx.cpt.translated.add(prn)
                    _log("TRANSLATE PRN %d: dop %+.0f -> %+.0f (%+.2f Hz) -> cp0 shifted "
                         "%+.2f chips; SAME physical code phase, anchor KEPT%s"
                         % (prn, prev["doppler_hz"], seed["doppler_hz"], ddop,
                            -t_now * ctx.args.chip_rate_hz * ctx.args.code_doppler_sign
                            * ddop / ctx.args.carrier_hz,
                            " (continuous: every cycle, no fence)"
                            if ctx.args.dop_continuous else ""))
            if prn not in ctx.cp_held:
                _sig = ctx.sig_of_last(ctx.status.get(prn))
                _via = ("amp_snr %.1f >= %.1f" % (_sig, ctx.args.hold_snr)
                        if _sig >= ctx.args.hold_snr
                        else "PRESENT x%d cycles (fleet gate)" % _present_streak(ctx, prn))
                _log("HOLD PRN %d: seed currency frozen (%s, dop %+.0f)"
                     % (prn, _via, prev["doppler_hz"]))
            ctx.cp_held.add(prn)
        else:
            if prn in ctx.cp_held:
                ddop_rel = (seed["doppler_hz"] - prev["doppler_hz"]) if prev else 0.0
                _log("RELEASE PRN %d: seed currency unfrozen (amp_snr %.1f, ddop %+.0f)"
                     % (prn, ctx.sig_of_last(ctx.status.get(prn)), ddop_rel))
                # A release used to STEP the tracker's f_ref by ddop while the TRACK-mode
                # trim carried the hold-era compensation -> instant residual ~ -ddop,
                # latched by the coh/innovation gates (C20 parked at -6.2 Hz for 40 min,
                # 2026-07-18). The arithmetic pre-shift (--trim-precomp-carrier) was
                # bench-rejected in both signs and DELETED (07-19 audit A4); the safe
                # rescuer below stands: the broker KNOWS the NCO stepped -- demote to
                # BOOTSTRAP and re-pull the trim at full gain (seconds, no arithmetic).
                # Under --dop-continuous ddop_rel is ~0 and this never fires.
                if abs(ddop_rel) > 1.0 and prn in ctx.car.locked:
                    ctx.car.locked.discard(prn)
                    ctx.car.fade.pop(prn, None)
                    _log("CARRIER REACQ PRN %d: hold released with dop step %+.1f Hz "
                         "-> BOOTSTRAP re-pull" % (prn, ddop_rel))
            ctx.cp_held.discard(prn)
            ctx.hold.miss.pop(prn, None)
        ctx.seeds[prn] = seed
        ctx.hold.low_hits[prn] = 0


def stage_push_seeds(ctx):
    """4: PUSH the consensus seeds to every tracker.
        
    The cycle's actuation point: whatever the stages above decided, this is what the trackers
    receive. The DLL trim is applied at POST time only, so the stored seed stays the model's view
    and the trim stays the loop's -- keeping the two ledgers separable is what makes #92's handover
    expressible at all.
        
    ⚠️ PUBLISHED DOPPLER IS THE COMMAND, NOT AN ESTIMATE. Continuity of what the tracker is told
    beats freshness of what we last measured; a seed that jumps to the newest estimate every cycle
    makes the tracker chase the estimator's noise.

    ⚠️ THE RAILED/RELEASED COUNTERS ARE `+=` ON A COUNTER INITIALISED BY THE CYCLE, so they
    must be nonlocal. An augmented assignment READS before it writes -- which is exactly what
    broker_iface missed when it cleared this stage for promotion, because it recorded only
    the Store. It cost gal_e5a a chain death at 12:49 on 2026-08-26."""
    for prn, v in sorted(ctx.seeds.items()):
        d = dict(prn=prn, **v)
        if ctx.dls.trim.get(prn):
            d["code_phase_chips"] = d["code_phase_chips"] + ctx.dls.trim[prn]
            # ⚠️ AUDIT §4.6 (#83 2(b) precondition): THE TRIM MUST MOVE THE PHASE TOO.
            # propagate_seed PREFERS code_phase_at_ref_chips whenever the payload
            # carries it -- which the search-fed path always does -- so a trim written
            # only into cp0 was written into the field the tracker ignores: the Python
            # slow trim has been a NO-OP on every phase-carrying seed, and enabling
            # --seed-phase-transport on the DR chains would have silently disabled
            # their only code loop. One trim, both currencies, same instant.
            if d.get("code_phase_at_ref_chips", -1.0) is not None \
                    and d.get("code_phase_at_ref_chips", -1.0) >= 0.0:
                _tmod = (ctx.lc_seg * ctx.code_len) if ctx.lc_seg > 1 else ctx.code_len
                d["code_phase_at_ref_chips"] = (
                    d["code_phase_at_ref_chips"] + ctx.dls.trim[prn]) % _tmod
        if ctx.car.trim.get(prn):
            d["carrier_trim_hz"] = ctx.car.trim[prn]
        if ctx.jrc is not None and prn not in ctx.probe_set:
            # Probes excepted for the trim loop's own reason: no carrier, and a moving
            # trim moves the REPORTED Doppler, which the beam map's churn gate reads
            # as sky. A sat whose row has not converged keeps the trim-loop value
            # (usually 0 on CHORD): commanding from a wide row would inject the
            # filter's own transient into the NCO.
            _k = (ctx.args.dr_constellation, int(prn))
            # ARM-12 GUARD: the row's own satellite must be DETECTED this cycle
            # (kcoh sig >= --rrate-cmd-min-sig) before it may command. The sigma
            # gate alone admitted E4's confident-noise row (see the flag's help).
            # No kcoh row (throttled estimator at startup, or the sat absent from
            # the fold) counts as NOT detected: no evidence, no command.
            _cmd_sig_ok = True
            if ctx.args.rrate_cmd_min_sig > 0.0:
                _cmd_sig_ok = (((ctx.dllp.kcoh or {}).get(prn) or {}).get("sig") or 0.0) \
                              >= ctx.args.rrate_cmd_min_sig
            if _cmd_sig_ok and ctx.jrc.rrate_sigma(_k) <= ctx.args.rrate_cmd_max_sigma:
                _cmd = ctx.jrc.carrier_correction_hz(_k, ctx.args.carrier_hz)
                if ctx.args.carrier_max_hz > 0.0:
                    _cmd = max(-ctx.args.carrier_max_hz, min(ctx.args.carrier_max_hz, _cmd))
                # SLEW toward the target from the command actually POSTED last poll
                # (--rrate-cmd-slew-hz): the feed's reference is only exact for a
                # command that holds still over the emit lag, so the step is bounded
                # and the bound is what makes the closed loop stable. Railed steps
                # are counted into the JRR-CMD line -- a rail that never clears is
                # a target out of reach, not convergence in progress.
                _prev = ctx.rf.cmd_applied.get(prn, 0.0)
                if ctx.args.rrate_cmd_slew_hz > 0.0:
                    _stp = max(-ctx.args.rrate_cmd_slew_hz,
                               min(ctx.args.rrate_cmd_slew_hz, _cmd - _prev))
                    if abs(_cmd - _prev) > ctx.args.rrate_cmd_slew_hz:
                        ctx.rf.railed += 1
                    _cmd = _prev + _stp
                d["carrier_trim_hz"] = _cmd
                ctx.rr_cmd_new[prn] = _cmd
        # RELEASE-SLEW (arm 12). A sat that exits the command set -- sig bar lost,
        # sigma widened, receiver row gone -- used to fall back to car_trim/0 in ONE
        # poll: an instant step of its whole standing command, the exact reference
        # discontinuity the slew bound exists to prevent, taken at release instead
        # of at pull-in. Walk it back at the same bounded rate; drop out only once
        # within one step of zero.
        if (ctx.args.rrate_command and prn not in ctx.rr_cmd_new and prn not in ctx.probe_set
                and ctx.rf.cmd_applied.get(prn)):
            _prev = ctx.rf.cmd_applied[prn]
            _stp = ctx.args.rrate_cmd_slew_hz if ctx.args.rrate_cmd_slew_hz > 0.0 else 0.0
            if _stp > 0.0 and abs(_prev) > _stp:
                _cmd = _prev - math.copysign(_stp, _prev)
                d["carrier_trim_hz"] = _cmd
                ctx.rr_cmd_new[prn] = _cmd
                ctx.rf.released += 1
            # else: within one step of zero (or slew disabled) -- final sub-slew
            # step back to car_trim/0, and the sat leaves rr_cmd_applied.
        if prn in ctx.car.repin_pending:
            # ONE-SHOT trim-bleed re-pin: the tracker does f_ref += this amount this frame.
            # Consume it here so it rides exactly this post (car_trim was zeroed above, so no
            # carrier_trim_hz accompanies it -- the trim moves wholly into f_ref, leaving the
            # combined carrier invariant).
            d["carrier_repin"] = ctx.car.repin_pending.pop(prn)
        # Peel sign source. PILOTS (P7b): the combiner publishes bit_pred directly --
        # its secondary overlay is DETERMINISTIC, so the chips are projected from the
        # pinned dead-reckon anchor with no decode and no round trip; forward verbatim.
        # DATA signals (P7a): the LNAV predictor's output. bit_pred wins where both
        # exist (a known overlay beats a decoded guess).
        _row = ctx.status.get(prn) or {}
        _bsrc = "none"
        # A source the health monitor has condemned for THIS satellite is skipped at
        # SELECTION time, so the chain genuinely falls back (pred -> lnav -> brdc)
        # instead of re-picking the vetoed source and going dark every cycle.
        _src_ok = (lambda _s: ctx.nav.health is None
                   or ctx.nav.health.verdict(prn, _s) != "bad")
        # Wire schema is COMPONENT-KEYED: nav_bits = {"P": table, "D": table, ...},
        # "P" = the component this chain's replica correlates (relational, not a signal
        # name -- see docs/navbit_supply_architecture.md C1). The tracker also accepts a
        # bare table as "P"; we publish the keyed form so the first data-channel producer
        # is an ADDITION ("D": ...) rather than a schema change.
        if _row.get("bit_pred", {}).get("bits") and _src_ok("pred"):
            # Attach only when the table CHANGED (utc0 moves once per combiner emit);
            # on other cycles the tracker keeps its stored copy -- see bp_pushed above.
            if ctx.bp_pushed.get(prn) != _row["bit_pred"].get("utc0"):
                d["nav_bits"] = {"P": _row["bit_pred"]}
                ctx.bp_pushed[prn] = _row["bit_pred"].get("utc0")
            _bsrc = "pred"
        elif ctx.nav.navbits is not None:
            # Predict from the freshest capture-clock UTC this PRN has reported: the
            # tracker consumes by ITS record UTC (same capture clock), so wall-clock
            # never enters. 4 s horizon >> the ~1 s status staleness + push cadence.
            _utc = _row.get("utc")
            if _utc:
                nb_lnav = ctx.nav.navbits.predict(prn, float(_utc), horizon_s=4.0)
                nb_brdc = (ctx.nav.brdc.predict(prn, float(_utc), horizon_s=30.0)
                           if ctx.nav.brdc is not None else None)
                if ctx.nav.health is not None:      # shadow-remember BOTH candidates
                    ctx.nav.health.remember(prn, nb_lnav, "lnav")
                    ctx.nav.health.remember(prn, nb_brdc, "brdc")
                nb = nb_lnav if (nb_lnav is not None and _src_ok("lnav")) else None
                if nb is None and _src_ok("brdc"):
                    # CONSTRUCTED fallback: this PRN never synced, so the decoder has
                    # nothing. Bits come from BRDC instead; alignment comes from the
                    # tracking geometry (range + sat clock) plus the common clock offset
                    # calibrated off a satellite that DID sync -- never from decoding,
                    # which is precisely what this satellite cannot do.
                    # 30 s, not the decoder's 4 s: only sf1-3 are constructible, so a 4 s
                    # window that lands inside sf4/5 is entirely unknown, predict() returns
                    # None, and the PRN reads `nobits` forever. A full frame (30 s) always
                    # spans the 18 s of sf1-3, so every push carries usable bits. Costs
                    # 1500 int8 per PRN.
                    nb = nb_brdc
                    _bsrc = "brdc" if nb is not None else "none"
                elif nb is not None:
                    _bsrc = "lnav"
                if nb is not None:
                    d["nav_bits"] = {"P": nb}
                    _bsrc = _bsrc if _bsrc != "none" else "lnav"
        elif ctx.nav.cnav is not None:
            # CNAV serve: the L2C-CM chain's replica correlates the DATA component, so a
            # decoded CNAV table is its "P" (the direct L1CA/LNAV analog). But this SERVES
            # already-decoded spans, it does not PREDICT the future (the CNAV type schedule
            # is not fixed -- see cnav_predictor), so at a forward horizon predict() usually
            # returns None and nothing is attached. That is correct: CL, not CNAV, is L2C's
            # peel win; CNAV's prizes are the live ephemeris + the S4 L5 cross-band feed.
            # Shadow-remembered and vetoed like every other source before it feeds a wipe.
            _utc = _row.get("utc")
            if _utc:
                nb = ctx.nav.cnav.predict(prn, float(_utc), horizon_s=4.0)
                if ctx.nav.health is not None:
                    ctx.nav.health.remember(prn, nb, "cnav")
                if nb is not None and _src_ok("cnav"):
                    d["nav_bits"] = {"P": nb}
                    _bsrc = "cnav"
        # VETO: a source that does not match the air for this satellite must not feed the
        # subtracter. Wrong bits are worse than no bits -- no bits means no subtraction,
        # wrong bits mean subtracting at the wrong sign on ~40% of records (measured
        # 2026-07-26: HURTING 0-1 -> 7-8). Then remember what we are about to publish, so
        # next cycle's observations score THIS table rather than a re-derived one.
        if "nav_bits" in d and ctx.nav.health is not None and ctx.nav.health.veto(prn, _bsrc):
            # FALL BACK, do not go dark: a bad decoded table must not darken a PRN whose
            # constructed table is fine. The chain re-runs with the vetoed source skipped
            # next cycle via the per-(prn,source) verdict; this cycle just drops the bits
            # (one cycle of nobits is the safe direction).
            d.pop("nav_bits")
            _bsrc = "vetoed:" + _bsrc
        elif "nav_bits" in d and ctx.nav.health is not None:
            ctx.nav.health.remember(prn, d["nav_bits"].get("P"), _bsrc)
        ctx.bit_src[_bsrc] = ctx.bit_src.get(_bsrc, 0) + 1
        if _bsrc != "none" and "nav_bits" in d:
            ctx.bit_known[_bsrc] = ctx.bit_known.get(_bsrc, 0) + sum(
                1 for t in d["nav_bits"].values() for b in t["bits"] if b)
        ctx.payload.append(d)
