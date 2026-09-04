"""The shared carrier loop: integrate the full-band carrier residual into a per-PRN trim.

⚠️ IT IS OFF ON CHORD. `--carrier-gain` is 0.0 in production, so nothing below runs on the
live instrument and no fixture reaches it. That is deliberate -- closing this loop was
measured to make tracking worse (#71) -- but it means the digest gate is silent about every
line here, and a latent state-clobbering bug lived in it undisturbed until 2026-08-26.

⚠️ NO FRESH DETECTION MEANS NO EVIDENCE. Without one there is nothing to say the estimator has
a signal at all; its residual is noise, and integrating noise random-walks the trim. Hold and
coast instead (`--carrier-det-gate-s`).

⚠️ THE BLEED SHADOW IS LOG-ONLY. If the trim holds a STANDING value across the stability
window then f_ref is pinned somewhere it should not be, and the trim is quietly absorbing a
reference error rather than a satellite one. Reading that as a satellite correction is how the
loop would make things worse while looking like it was working.

@author Keith Vanderlinde
"""

from gnss_broker.transport import _log


def stage_carrier_loop(ctx):
    """3d: THE SHARED CARRIER LOOP -- integrate the combiner's full-band carrier residual into a
    per-PRN trim, and command it.
        
    The code twin of 3c. Nothing here leaves the stage: the trim is delivered through `car_trim`
    and the seeds, so this block has ZERO live outputs into the rest of the cycle.
        
    ⚠️ IT IS OFF IN PRODUCTION. `--carrier-gain` is 0.0 on CHORD, so on the live instrument this
    entire routine is dead code, and has been since the loop was plumbed. That is a deliberate
    state (the loop was measured to make things worse -- see #71), not an oversight, but it means
    NOTHING HERE IS EXERCISED BY THE FIXTURES EITHER: the digest gate is silent about every line
    below. A latent state-clobbering bug lived here undisturbed until 2026-08-26 for exactly that
    reason. Treat changes to this routine as unverified until the loop is armed on sky."""
    if ctx.args.carrier_gain > 0.0:
        for _p in [p for p in ctx.car.trim_force if p in ctx.seeds]:
            ctx.car.trim[_p] = ctx.car.trim_force.pop(_p)
            _log("TRIM FORCE (bench): PRN %d car_trim POISONED to %+.1f Hz"
                 % (_p, ctx.car.trim[_p]))
        # (computed once above -- see the shared-call note; {} when carrier_source
        #  is not "rate", which preserves the old carrier_hz_resid fallback below)
        rate_resid, rate_consensus = ctx.rf.resid, ctx.rf.cons
        car_report = []
        for prn in list(ctx.seeds):
            if prn in ctx.probe_set:
                # NOISE PROBES ARE EXEMPT. A probe is a satellite we deliberately point at
                # empty sky: there IS no carrier, so its "residual" is pure noise, and with
                # no lock gate the loop integrates that noise straight into the trim until
                # it random-walks to the clamp. Harmless to the probe (it measures noise
                # either way) -- but the trim moves the REPORTED Doppler, which made the
                # probes look like the churniest objects in the sky, and the beam map's
                # churn gate then excised 100% of them (2026-07-14). The pedestal
                # calibrator was being destroyed by a loop chasing noise it should never
                # have been fed. Nothing downstream wants a carrier trim on a probe.
                continue
            rec = ctx.status.get(prn, {})
            # (ALIAS ESCAPE v1/v2 lived here until the 07-19 audit A4. The alias-capture
            #  disease it targeted -- resid estimator ambiguous mod 1/(2*T_rec), NCO
            #  parked on the alias for 40+ min -- is owned by two surviving mechanisms:
            #  a stale f_ref offset is snapped by the tight tracker fence (free under
            #  --dop-continuous), and a walked trim latch is the watchdog's lifecycle
            #  rescue. v1's fleet-kill postmortem lives in git history at 069e8770.)
            resid = float(rec.get("carrier_hz_resid", 0.0))
            if ctx.args.carrier_source == "rate":
                # THE MEASUREMENT THE LOOP WAS ALWAYS MISSING (2026-08-04). carrier_hz_resid
                # is signal-free -- 0.519 Hz on signal vs 0.492 Hz on noise -- which is why
                # closing this loop on it made every metric worse (8.18.4) and why leaving it
                # open left the carrier free-running (the ramp). deep_rate_hz comes from the
                # combiner's phase-rate search: peak/median 17.9-22.0 on signal against
                # 2.8-6.1 on noise, and split-half agreement of 0.0-0.5 Hz on strong sats
                # against the <1 Hz a 1.05 s window needs.
                #
                # It is a DELTA, not an absolute: the search runs on records the tracker
                # already derotated by car_trim, so what it reports is what remains. That is
                # exactly the residual this loop integrates -- no reference change.
                rr = rate_resid.get(prn)
                if rr is None and ctx.args.carrier_rate_inherit:
                    rr = rate_consensus # failed its own gate: take the fleet's answer
                if rr is None:
                    continue
                resid = rr
            if resid == 0.0:
                continue
            # TWO-MODE LOOP. BOOTSTRAP (never certified since seed): legacy behavior --
            # accept any residual at full gain, because at seed the trim is 0, the true
            # residual IS the ~30 Hz clock offset, and the window cannot cohere until the
            # trim pulls in (gating on certification here would deadlock acquisition).
            # TRACK (certified once): a residual is only a measurement when its window
            # was COHERENT -- gate on certification AND significance, and slew-clamp.
            # The sig gate alone let re-lock transition rows (amp back, phase not) kick
            # converged trims 30 Hz a shot: E36 at 48 dB-Hz visited the +-100 Hz rails.
            # Dedup FIRST -- at most one gate-check + integration per fresh emit: the
            # combiner emits a new residual every ~1.5 s window while this loop polls at
            # ~5 Hz; integrating a stale value 5-7x over-applies the gain and oscillates
            # (observed +-20 Hz swings), and the --carrier-refade counter below must
            # count EMITS, not polls. A changed value marks a fresh emit.
            if resid == ctx.car.last.get(prn):
                continue
            ctx.car.last[prn] = resid
            coh_ok = (rec.get("coherence_s") or 0.0) > 0.0
            # ---- VERIFYING: an applied step hypothesis is judged by OUTCOME ----
            # (explain-apply-verify, 2026-07-22): a trim correction is a falsifiable
            # hypothesis -- either coherence returns / the residual collapses within
            # CARRIER_VERIFY_EMITS, or it was WRONG and is reverted + escalated to a
            # full re-acquire. This bounded closed loop is exactly what the two
            # retracted open-loop escapes (v2 EMA unwrap, loose step-accept) were
            # missing: a wrong correction costs one reverted step, never compounds.
            if prn in ctx.car.verify:
                v = ctx.car.verify[prn]
                v["emits"] += 1
                if coh_ok or abs(resid) < ctx.carrier_explain_hz:
                    del ctx.car.verify[prn]
                    ctx.car.fade.pop(prn, None)
                    _log("CARRIER STEP VERIFIED PRN %d: healed in %d emit(s) "
                         "(coh=%s, resid %+.2f Hz)" % (prn, v["emits"], coh_ok, resid))
                    # fall through: this emit integrates normally below
                elif v["emits"] >= ctx.carrier_verify_emits:
                    ctx.car.trim[prn] = v["prev_trim"]  # revert the refuted hypothesis
                    del ctx.car.verify[prn]
                    ctx.car.locked.discard(prn)         # escalate: BOOTSTRAP re-acquire
                    ctx.car.step_t[prn] = ctx.t0 + 50.0     # ~60 s hypothesis lockout
                    ctx.car.fade.pop(prn, None)
                    _log("CARRIER STEP REFUTED PRN %d: no heal after %d emits (resid "
                         "%+.2f Hz) -> trim reverted to %+.2f, BOOTSTRAP re-pull"
                         % (prn, ctx.carrier_verify_emits, resid, v["prev_trim"]))
                    continue
                else:
                    continue  # verdict pending: hold, no further corrections
            # ---- POST-BLEED VERIFY (trim-bleed, explain-apply-verify) ----
            # A re-pin is a falsifiable hypothesis too: after folding the trim into f_ref, the
            # sat must stay coherent. Unlike a step, there is NOTHING to revert (the re-pin is
            # phase-continuous and car_trim correctly re-grows from 0 via the normal loop), so
            # this only OBSERVES the outcome and lets the lockout prevent churn. It falls
            # through to integrate normally either way (trim re-grows to the small remnant).
            if prn in ctx.car.bleed_verify:
                bv = ctx.car.bleed_verify[prn]
                bv["emits"] += 1
                if bv["emits"] >= ctx.args.carrier_bleed_verify_emits:
                    del ctx.car.bleed_verify[prn]
                    # Judge by the SETTLED residual, not coh_ok (which blips for ~1 emit on the
                    # deep-window reset a re-pin causes, good bleed or not). A residual at/under
                    # the bar means the fold left the carrier aligned; a large one is a real
                    # miss (the loop re-grows the trim from 0 either way -- and even a mild miss
                    # already REDUCED the standing trim, so the bar is generous).
                    if abs(resid) <= ctx.args.carrier_bleed_ok_hz:
                        _log("CARRIER BLEED VERIFIED PRN %d: resid settled %+.2f Hz "
                             "(<= %.2f) over %d emits, trim now %+.2f"
                             % (prn, resid, ctx.args.carrier_bleed_ok_hz, bv["emits"],
                                ctx.car.trim.get(prn, 0.0)))
                    else:
                        _log("CARRIER BLEED REFUTED PRN %d: resid %+.2f Hz (> %.2f) after "
                             "%d emits -- loop re-grows trim, %.0f s lockout"
                             % (prn, resid, ctx.args.carrier_bleed_ok_hz, bv["emits"],
                                ctx.args.carrier_bleed_lockout_s))
            sig = (max(rec.get("deep_snr") or 0.0, rec.get("amp_snr") or 0.0)
                   if coh_ok else 0.0)
            tracking = prn in ctx.car.locked
            if coh_ok and sig >= ctx.args.carrier_min_sig > 0.0:
                ctx.car.locked.add(prn)
            gated = (tracking and ctx.args.carrier_min_sig > 0.0
                     and (not coh_ok or sig < ctx.args.carrier_min_sig))
            fade_gated = gated  # incoherent/weak: the resid estimator is NOT trusted
            if not gated and tracking and ctx.args.carrier_innov_hz > 0.0 \
                    and abs(resid) > ctx.args.carrier_innov_hz:
                gated = True  # certified-but-implausible residual: the estimator is lying
            if gated:
                # Presence first (shared by the hypothesis stage AND refade below):
                # amp OR a fresh strong detection -- see the refade note.
                _df = ctx.det_fresh.get(prn)
                present = ((rec.get("amp_snr") or 0.0) >= ctx.args.hold_snr
                           or (_df is not None and ctx.t0 - _df[1] < 10.0
                               and prn in ctx.best
                               and ctx.best[prn][0] >= 2.0 * ctx.args.acquire_snr))
                # ---- STRONG-INCOHERENT HYPOTHESIS (explain-apply-verify) ----
                # The innovation gate above is COHERENT physics (a cohering sat cannot
                # carry a multi-Hz residual, so such a reading is a lie). An INCOHERENT
                # sat has no such bound -- and when the observables close on one story
                # (signal PRESENT + M consecutive residuals AGREE + the agreed value is
                # big enough to EXPLAIN the decoherence), the residual is the
                # explanation, not a lie (type specimen C19 2026-07-22: full amp, dark,
                # parked at +3.03 Hz for minutes while every gate held). Apply the FULL
                # agreed correction ONCE and enter VERIFYING (top of loop): heal within
                # 3 emits or be reverted + escalated. Applies to BOTH gate flavors --
                # the verify stage is what makes fade-gated acceptance safe (the loose
                # v1 escape lacked it and churned the fleet; a wrong hypothesis now
                # costs one reverted step + a 60 s lockout).
                if ctx.args.carrier_step_accept > 0 and present:
                    hist = ctx.car.step_hist.setdefault(prn, [])
                    hist.append((ctx.t0, resid))
                    del hist[:-ctx.args.carrier_step_accept]
                    band = max(2.0, ctx.args.carrier_innov_hz)
                    if (len(hist) >= ctx.args.carrier_step_accept
                            and ctx.t0 - hist[0][0] < 30.0
                            and ctx.t0 - ctx.car.step_t.get(prn, 0.0) >= 10.0):
                        vals = sorted(r for _, r in hist)
                        med = vals[len(vals) // 2]
                        if (vals[-1] - vals[0] < band
                                and abs(med) >= ctx.carrier_explain_hz):
                            prev_trim = ctx.car.trim.get(prn, 0.0)
                            ctx.car.trim[prn] = max(-ctx.args.carrier_max_hz,
                                                min(ctx.args.carrier_max_hz,
                                                    prev_trim + med))
                            ctx.car.step_t[prn] = ctx.t0
                            ctx.car.step_hist[prn] = []
                            ctx.car.verify[prn] = {"prev_trim": prev_trim, "emits": 0}
                            _log("CARRIER STEP HYPOTHESIS PRN %d: %d agreeing gated "
                                 "resids (med %+.2f Hz, spread %.2f) -> trim %+.2f, "
                                 "VERIFYING (heal in %d emits or revert)"
                                 % (prn, ctx.args.carrier_step_accept, med,
                                    vals[-1] - vals[0], ctx.car.trim[prn],
                                    ctx.carrier_verify_emits))
                            continue
                # --carrier-refade: the two gates otherwise form an ABSORBING state for a
                # sat whose NCO really stepped (hold release / escape re-anchor without
                # trim precomp): coherence never returns while the residual exceeds the
                # innovation gate, so the trim can never unwind. Sustained gating WITH
                # the sat still present = carrier genuinely lost, not a fade: demote to
                # BOOTSTRAP and let the next residual re-pull at full gain (seconds).
                # A genuinely faded sat (no amp AND no detection) keeps coasting on the
                # feed-forward -- the pathology --carrier-min-sig exists to protect.
                # Presence = amp OR a fresh strong detection (2026-07-19): the amp-only
                # bar left WEAK-chain sats (L2C at cn0 27-31: amp 5-20, under the
                # hold bar while decohered) latched with a STANDING sub-innovation
                # residual (~2 Hz measured: kills 1-s deep windows, too small for the
                # innov gate, coherence gate holds the trim, refade never fired) --
                # the C20 absorbing state one level down. The search still sees these
                # sats fine; det presence is the same test the watchdog trusts.
                # (`present` computed at the top of the gated branch, shared with the
                # hypothesis stage.)
                ctx.car.fade[prn] = ctx.car.fade.get(prn, 0) + 1 if present else 0
                # FLICKER GUARD (2026-07-20): a SUB-innovation residual on a sat that
                # cohered seconds ago is certification-bar sig flicker, not a stepped
                # NCO -- the re-pull has nothing to pull (settled-era E1/B1C: ~700
                # REACQs/3 h at mean |resid| 1.7 Hz, all no-ops). Suppress the demotion
                # for those; a STANDING decoherence (the L2C C20 absorbing state:
                # sub-gate resid, dark for minutes) still demotes once the sat has been
                # incoherent longer than the window. Inactive when the watchdog is off
                # (wd_coh_t empty -> old behavior).
                _flicker = (ctx.args.refade_flicker_s > 0.0
                            and abs(resid) < ctx.args.carrier_innov_hz
                            and ctx.t0 - ctx.wd.coh_t.get(prn, 0.0) < ctx.args.refade_flicker_s)
                if (ctx.args.carrier_refade > 0 and not _flicker
                        and ctx.car.fade.get(prn, 0) >= ctx.args.carrier_refade):
                    ctx.car.locked.discard(prn)
                    ctx.car.fade.pop(prn, None)
                    _log("CARRIER REACQ PRN %d: %d consecutive gated emits at full amp "
                         "(last resid %+.2f Hz) -> BOOTSTRAP re-pull"
                         % (prn, ctx.args.carrier_refade, resid))
                continue  # this emit stays held: coast on the feed-forward
            ctx.car.fade.pop(prn, None)
            ctx.car.step_hist.pop(prn, None)  # ungated emit: gated-run agreement is stale
            if not tracking and ctx.args.carrier_det_gate_s > 0.0:
                # BOOTSTRAP WALK GATE: no fresh detection = no evidence the estimator
                # has a signal to measure; its residual is noise and integrating it
                # random-walks the trim (see --carrier-det-gate-s). Hold and coast.
                _fr = ctx.det_fresh.get(prn)
                if _fr is None or ctx.t0 - _fr[1] > ctx.args.carrier_det_gate_s:
                    continue
            if prn not in ctx.car.trim and ctx.args.carrier_fleet_seed:
                # start on the fleet's clock, not at 0: the converged trim is the chain's
                # deterministic frac-N LO offset, common-mode across sats
                #
                # ⚠️ THIS LOCAL WAS CALLED `fleet` UNTIL 2026-08-26, AND THAT WAS A LATENT
                # BUG (found by the refactor's interface analysis, never seen on sky).
                # `fleet` is the DLL's per-PRN state dict for the whole cycle; assigning a
                # sorted LIST of carrier trims over it left every later consumer --
                # including the FLEET-TRIM arming block's `fleet[_p].get("present")` and
                # the fast-trim PRN set -- indexing a list of floats by PRN. It has never
                # fired only because --carrier-gain is 0.0 in production, so this whole
                # block is dead; arming the carrier loop (which is exactly what #52 wants)
                # would have taken the C++ trim loop down with it, and the traceback would
                # have pointed at the trim code rather than here.
                _car_seed_vals = sorted(ctx.car.trim.values())
                if len(_car_seed_vals) >= 3:
                    ctx.car.trim[prn] = _car_seed_vals[len(_car_seed_vals) // 2]
            prev_trim = ctx.car.trim.get(prn, 0.0)
            trim = (1.0 - ctx.args.carrier_leak) * prev_trim + ctx.args.carrier_gain * resid
            if tracking and ctx.args.carrier_max_step > 0.0:
                trim = prev_trim + max(-ctx.args.carrier_max_step,
                                       min(ctx.args.carrier_max_step, trim - prev_trim))
            ctx.car.trim[prn] = max(-ctx.args.carrier_max_hz, min(ctx.args.carrier_max_hz, trim))
            car_report.append("PRN %d resid %+.2f Hz trim %+.2f" % (prn, resid, ctx.car.trim[prn]))
            # ---- f_ref TRIM-BLEED SHADOW (log-only, no action) ----
            # This emit is COHERENT and TRACKING (it reached the integrator ungated). If the
            # trim has held a STANDING value across the stability window, f_ref is pinned
            # off-true by ~that trim and a re-pin (f_ref += trim) would clear it. Log the
            # candidate so the trigger can be validated on live data before it is ever armed.
            # Recency-windowed like car_step_hist (a decoherence gap ages the window out ->
            # not "converged"), so no per-gate-branch cleanup is needed.
            if (ctx.args.carrier_bleed_shadow or ctx.args.carrier_bleed) and tracking and coh_ok:
                bh = ctx.car.bleed_hist.setdefault(prn, [])
                bh.append((ctx.t0, ctx.car.trim[prn]))
                del bh[:-ctx.args.carrier_bleed_stable_emits]
                vals = [v for _, v in bh]
                # FLAT-TRIM gate (2026-08-03): a truly converged trim is FLAT; a still-settling
                # one DRIFTS (low spread but a monotonic climb toward a higher plateau). Bleeding
                # a drifting trim folds a mid-convergence value into f_ref and leaves the
                # remainder as a residual -> the REFUTED class from the L2C live arm (PRN21 bled
                # at +5.37 while climbing -> +2.86 resid). Spread alone can't tell drift from
                # noise; the least-squares SLOPE can (noise averages out, a drift does not).
                slope = 0.0
                if len(bh) >= 2:
                    tb = sum(t for t, _ in bh) / len(bh)
                    vb = sum(vals) / len(vals)
                    den = sum((t - tb) ** 2 for t, _ in bh)
                    if den > 0.0:
                        slope = sum((t - tb) * (v - vb) for t, v in bh) / den
                converged = (len(bh) >= ctx.args.carrier_bleed_stable_emits
                             and ctx.t0 - bh[0][0] < 90.0
                             and abs(ctx.car.trim[prn]) >= ctx.args.carrier_bleed_hz
                             and max(vals) - min(vals) <= ctx.args.carrier_bleed_stable_hz
                             and abs(slope) <= ctx.args.carrier_bleed_max_slope)
                # ARMED: re-pin f_ref and zero the trim (one bleed per lockout, never while a
                # step- or bleed-hypothesis is already under verify for this PRN).
                if (converged and ctx.args.carrier_bleed and prn not in ctx.car.verify
                        and prn not in ctx.car.bleed_verify
                        and ctx.t0 >= ctx.car.bleed_lock_t.get(prn, 0.0)):
                    prev_trim = ctx.car.trim[prn]
                    ctx.car.trim[prn] = 0.0             # f_ref re-pin absorbs the offset
                    ctx.car.repin_pending[prn] = prev_trim  # tracker does f_ref += prev_trim
                    ctx.car.bleed_verify[prn] = {"emits": 0, "prev_trim": prev_trim, "t": ctx.t0}
                    ctx.car.bleed_lock_t[prn] = ctx.t0 + ctx.args.carrier_bleed_lockout_s
                    ctx.car.bleed_hist[prn] = []
                    ctx.car.bleed_log_t[prn] = ctx.t0
                    _log("CARRIER BLEED PRN %d: re-pinning f_ref (%+.2f Hz absorbed, slope "
                         "%+.3f Hz/s), trim->0, VERIFYING (heal in %d emits)"
                         % (prn, prev_trim, slope, ctx.args.carrier_bleed_verify_emits))
                elif converged and ctx.t0 - ctx.car.bleed_log_t.get(prn, 0.0) >= 60.0:
                    ctx.car.bleed_log_t[prn] = ctx.t0
                    _log("CAR-BLEED CANDIDATE PRN %d: trim %+.2f Hz stable %d emits "
                         "(spread %.2f, slope %+.3f Hz/s), coherent -> %s"
                         % (prn, ctx.car.trim[prn], len(bh), max(vals) - min(vals), slope,
                            "locked out" if ctx.args.carrier_bleed
                            else "would re-pin f_ref, predict trim->~0 (shadow, no action)"))
        if car_report:
            _log("CAR: " + "; ".join(car_report))
        for k in list(ctx.car.trim):
            if k not in ctx.seeds:
                del ctx.car.trim[k]
                ctx.car.locked.discard(k)  # a re-seeded sat re-enters via BOOTSTRAP
                ctx.car.verify.pop(k, None)  # a dropped sat's hypothesis dies with it
                ctx.car.step_hist.pop(k, None)
                ctx.car.fade.pop(k, None)
                ctx.car.bleed_hist.pop(k, None)  # its convergence history dies with it
                ctx.car.bleed_log_t.pop(k, None)
                ctx.car.bleed_verify.pop(k, None)
                ctx.car.bleed_lock_t.pop(k, None)
                ctx.car.repin_pending.pop(k, None)
