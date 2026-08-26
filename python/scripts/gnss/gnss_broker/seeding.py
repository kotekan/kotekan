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

from gnss_broker.transport import _log, _log_rl
from gnss_broker.seed import Seed
from gnss_broker.fits import (
    retag_seed_doppler, code_clock_bias_sample, fit_cp_rate, fit_dop_rate, tracker_phase_at,
)


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
