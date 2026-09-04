"""The code loop's per-satellite decisions: the DLL control loop, and the track watchdog.

The DLL control loop is the only part of the fleet delay-lock stage that ACTUATES -- everything
else hung off that poll is instrumentation (see `instruments.py`). The watchdog watches the
same satellites from the other side: how long has each seeded PRN gone without ever cohering.

⚠️ THE RE-SEED IS EVALUATED BEFORE THE PRESENCE / RATE / HOP GATES. Those `continue`s exist to
stop the TRIM integrating on a bad basis, and they used to skip the #50 re-seed along with it --
which re-created #79's one-way door for exactly the satellites the re-seed exists for (measured
on E32: five consecutive qualifying fits, zero strikes logged). A policy decision about an
off-peak satellite must not sit behind an on-peak gate.

⚠️ NO LIVE code_phase_rate MEANS HOLD, NOT INTEGRATE. The receiver clock runs ~3.45 chips/s,
which is ~11 chips per loop round trip, fed forward by the seed. With no live rate the trim
would face the whole 11 chips -- far outside the +-0.5 chip pull-in, unrecoverable -- and it
would read as the DLL diverging rather than as the feed-forward being absent.

⚠️ THE `DLL:` LINE BUILT HERE LISTS ONLY PRESENCE-PASSING SATELLITES. Any per-satellite
statistic computed over it measures SURVIVORS: a satellite whose q craters leaves the sample
entirely. That produced a wrong "no harm" verdict on 2026-08-25, where E4 was absent from the
line for exactly the 85 minutes it was sick. Judge on a source that keeps the sick ones.

@author Keith Vanderlinde
"""

import time

from gnss_broker.transport import _log, _log_rl
from gnss_broker import combdll
from gnss_broker.admission import reseed_step


def stage_dll_control(ctx):
    """3c: THE DLL CONTROL LOOP -- per satellite, decide and integrate the code trim.
        
    The only part of the DLL stage that ACTUATES. Everything above it is instrumentation.
        
    ⚠️ THE RE-SEED IS EVALUATED BEFORE THE PRESENCE/RATE/HOP GATES. Those `continue`s exist to stop
    the TRIM integrating on a bad basis, and they used to skip the #50 re-seed with it -- which
    re-created #79's one-way door for exactly the satellites the re-seed exists for (measured on
    E32: five consecutive qualifying fits, zero strikes logged). A policy decision about an
    off-peak satellite must not sit behind an on-peak gate.
        
    ⚠️ NO LIVE code_phase_rate MEANS HOLD, NOT INTEGRATE. The clock offset is ~3.45 chips/s = 11
    chips per loop round trip, fed forward by the seed. With no live rate the trim would face the
    whole 11 chips -- far outside the +-0.5 chip pull-in, unrecoverable, and it would read as the
    DLL diverging rather than as the feed-forward being absent.
        
    ⚠️ THE `DLL:` LINE THIS BUILDS LISTS ONLY PRESENCE-PASSING SATELLITES. Any per-satellite
    statistic computed over it measures SURVIVORS. That cost a wrong "no harm" verdict on
    2026-08-25; judge on a source that keeps the sick ones."""
    for prn in list(ctx.seeds):
        rec = ctx.status.get(prn, {})
        fl = ctx.dllp.fleet.get(prn)
        # #90 (2026-08-25): the re-seed is evaluated BEFORE the presence/rate/hop
        # gates below -- those `continue`s exist to stop the TRIM integrating on a
        # bad basis, and they used to skip the re-seed with it, which re-created
        # #79's one-way door: the absent-admission clause was unreachable for the
        # exact PRNs it exists for (measured on E32, 00:06-00:11 UTC: five
        # consecutive qualifying fits, zero strikes logged). A policy decision
        # about an off-peak satellite must not sit behind an on-peak gate.
        if fl is not None:
            # ---- FAR-REGIME RE-SEED (task #50) ------------------------------------------
            # WHEN THE DISCRIMINATOR HAS NO GRADIENT, THE TRIM CANNOT HELP. Far off the
            # peak E, P and L are all noise, so q -> 1.0 and disc -> 0: #49's deep gate
            # admits the satellite to this loop but there is nothing for the loop to
            # follow. Measured immediately on deploying #49 (E33 on gal_e5a: q 0.96-1.04,
            # disc -0.02..-0.07, trim ~0, while deep_snr was 23-36).
            #
            # spec_tau measures the offset a different way -- the phase ramp ACROSS
            # CHANNELS (task #32) -- so it does not need E/P/L to straddle the peak, which
            # is exactly the regime that kills the discriminator.
            #
            # VALIDATED BEFORE BEING WIRED IN (2026-08-12): sign predicted A PRIORI from
            # the physics (disc<0 => L>E => peak later => tau>0, so anti-correlated) and
            # confirmed on the shoulder where disc is trustworthy, r -0.47..-0.62 over 222
            # samples, slope -0.67 chips per unit disc. In the far regime, judged against
            # cn0_inc (the prompt-based witness, disc never used) WITHIN satellite so the
            # off-peak/faint confound is removed: r -0.230 all, -0.375 strong fits. The
            # direction is established; THE SIZE IS NOT, which is why this steps by a
            # fraction and re-measures rather than jumping to the fitted value.
            #
            # ⚠️ A SEED STEP, NOT A TRIM INCREMENT. The slew cap (0.05 chips/event, already
            # railing 67-100% of the time) would swallow this whole if it went through the
            # trim -- see chord-slew-cap-saturation.
            #
            # spec_peak_ratio IS a shuffled-null significance (fit_spectrum_delay builds
            # the null from the same points, values reassigned within each instance), so
            # >= the bar means the fold beat its own null -- not a tuned constant.
            _rs = None
            _rs_qual = (ctx.dls.reseed_prns and fl is not None
                        and (ctx.dls.reseed_prns is True or prn in ctx.dls.reseed_prns)
                        and fl.get("q", 9.9) < ctx.args.reseed_q_max  # taps carry no gradient
                        and (fl.get("spec_ratio") or 0.0) >= ctx.args.reseed_min_ratio
                        and fl.get("spec_tau") is not None)
            # #90 ADMISSION CLAUSE: `present` is the #49 deep gate, and on the searchless
            # chains it is a one-way door -- off-peak kills presence, and with no search
            # to re-admit (#79 is gps_l5-only) the PRN can never earn its correction
            # back. A SEEDED absent PRN (the model says it is up; drop-on-set already
            # pruned the ones that are not) may fire the re-seed anyway, but only on the
            # SECOND consecutive qualifying fit that agrees on tau's SIGN: presence was
            # the noise guard, and on noise the sign is a coin flip, so same-sign
            # agreement is the replacement guard. One strike is recorded per qualifying
            # ABSENT fit; presence or a sign change resets the count.
            _rs_admit = False
            if _rs_qual and not fl.get("present"):
                # #90 ADMISSION CLAUSE: `present` is the #49 deep gate, and on the
                # searchless chains it is a one-way door -- off-peak kills presence,
                # and with no search to re-admit (#79 is gps_l5-only) the PRN can
                # never earn its correction back. A SEEDED absent PRN may fire the
                # re-seed anyway, on evidence that replaces what presence guarded.
                # The four flights' guards and their history are in
                # gnss_broker/admission.py; the strike clock is the REAL wall clock
                # (not the frozen cycle clock) because decorrelation is a statement
                # about fold windows, not about cycles.
                _npn = sum(1 for _f0 in ctx.dllp.fleet.values()
                           if isinstance(_f0, dict) and _f0.get("present"))
                _adm = ctx.adm_gate.decide(prn, float(fl["spec_tau"]), prn in ctx.seeds,
                                        time.time(), ctx.t0, _npn,
                                        time.time() - ctx.broker_t0)
                _rs_admit = _adm.fire
                for _ak, _am, _ae in _adm.logs:
                    _log_rl(_ak, _am, every_s=_ae)
                if _adm.reason == "strike1":
                    _log_rl("rs-admit-%d" % prn,
                            "RESEED-ADMIT PRN %d: strike 1 (tau %+.3f, "
                            "pk/fl %.2f, absent) -- fires on a consistent "
                            "(|dtau|<=0.5) qualifying fit 60-600 s from now"
                            % (prn, float(fl["spec_tau"]),
                               fl.get("spec_ratio") or 0.0),
                            every_s=60.0)
            elif fl.get("present"):
                ctx.adm_gate.note_present(prn)
            if _rs_qual and (fl.get("present") or _rs_admit):
                _t = float(fl["spec_tau"])
                # The span-edge refusal and the fractional step are in
                # gnss_broker/admission.py: the direction is validated, the MAGNITUDE
                # is not, so this converges over several opportunities rather than
                # betting the correction on one unproven number.
                _step, _rs = reseed_step(_t, ctx.args.spec_span_chips,
                                         ctx.args.reseed_gain, ctx.args.reseed_max_chips)
                if _step is not None:
                    ctx.seeds[prn].put(
                        "reseed", epoch=ctx.seeds[prn].get("ref_hop"),
                        code_phase_chips=(ctx.seeds[prn].get("code_phase_chips", 0.0)
                                          + _step) % ctx.args.code_length)
                    # The at-ref phase is a DERIVED leg of the same triple; leaving it
                    # stale would ship a seed whose two phases disagree, which is the
                    # transport disease of #45 in miniature. Drop it and let the normal
                    # path rebuild it from the value we just moved.
                    ctx.seeds[prn].pop("code_phase_at_ref_chips", None)
                    # #90: the admission is seed step + ARMING WINDOW, not seed step
                    # alone. On the dead-reckon chains the slew returns the seed to the
                    # model at the cap rate (~7 s for this step), so the DURABLE per-sat
                    # actuator is the C++ trim -- which only pulls while armed, and
                    # arming rides the presence hold this stamp opens. The spectrum
                    # admitted the PRN; give the trim loop the same 90 s the presence
                    # path would have given it.
                    if _rs_admit and ctx.args.fleet_trim_url:
                        ctx.dls.hold[prn] = time.time()
                    _rs = ("tau %+.3f pk/fl %.2f q %.2f -> seed %+.3f chips%s"
                           % (_t, fl.get("spec_ratio") or 0.0, fl.get("q", 0.0), _step,
                              " [#90 ADMIT: absent, 2-strike]" if _rs_admit else ""))
            if _rs:
                _log("RESEED PRN %d: %s" % (prn, _rs))
        if fl is not None:
            # THE FLEET PATH. Gate on the summed q against a floor MEASURED from this
            # cycle's own q population, not against a constant and not against the
            # single-instance significance. sig_of(rec) is one node's 6.7% view and
            # would keep vetoing precisely the satellites this exists for; a fixed bar
            # is worse still, because summing tightens the noise distribution instead
            # of raising q, so the correct bar FALLS as instances are added and any
            # constant is right for exactly one fleet size (see fleet_dll).
            #
            # THE RATE GATE (design section 7, and it is not optional): the clock offset
            # is 3.45 chips/s = 11 chips per loop round trip, fed forward by the seed's
            # code_phase_rate. The trim only ever absorbs the residual. With no live
            # rate the trim would face the whole 11 chips -- far outside the +-0.5 chip
            # pull-in, unrecoverable, and it would read as the DLL diverging rather than
            # the feed-forward being absent. So HOLD instead of integrating.
            if not float(ctx.seeds[prn].get("code_phase_rate", 0.0) or 0.0):
                _log_rl("dll-norate-%d" % prn,
                        "fleet DLL PRN %d: no live code_phase_rate, holding trim" % prn)
                continue
            if not fl["present"]:
                continue
            # One integration per new WINDOW -- an exact integer test, where the
            # single-instance path below can only watch for a changed float.
            if fl["hop"] == ctx.dls.last_hop.get(prn):
                continue
            ctx.dls.last_hop[prn] = fl["hop"]
            disc = fl["disc"]
        else:
            disc = float(rec.get("dll_disc", 0.0))
            if ctx.sig_of(rec) < ctx.args.lock_snr or disc == 0.0:
                continue
            # One integration per NEW measurement: the combiner emits a fresh disc each
            # integration window (~1 s) while this loop polls at ~5 Hz -- integrating the
            # stale value 5x per emit over-applies the gain (part of the 2026-07-07 L1
            # runaway). A changed value marks a fresh emit.
            if disc == ctx.dls.last.get(prn):
                continue
            ctx.dls.last[prn] = disc
        tau = -max(-1.0, min(1.0, disc)) / 4.0 * (ctx.args.dll_spacing / 0.5)
        # LEAKY integrator: mean-reverts, so discriminator NOISE cannot random-walk the
        # trim to the clamp (a pure integrator did -- L1 trims reached +-1 to 3 chips
        # over 2 min, dragging the code off the +-0.5-chip peak and collapsing the deep
        # while the search stayed strong). Steady state gain*tau/leak still nulls a real
        # static fit bias; noise averages over ~1/leak windows.
        # LEAK GATE. The leak exists because a PURE integrator random-walked the trim
        # into the clamp on discriminator noise (L1, 2026-07-07). But it also CAPS the
        # correction: steady state is trim = (gain/leak)*tau and |tau| <= 0.25 chips, so
        # at gain 0.25 / leak 0.05 nothing beyond 1.25 chips is reachable. Measured
        # 2026-08-04: PRN 9 parked at trim +1.00 with disc railed at -0.984 -- the loop
        # pushing as hard as it can and still not arriving.
        #
        # What the leak was standing in for was a signal test, and the fleet path now
        # HAS one: `present` is a self-calibrating significance on the summed prompt
        # power (fleet_dll), and a PRN that fails it is skipped entirely rather than
        # leaked. So on the fleet path the leak can be much smaller -- it keeps the slow
        # mean-reversion that stops a long-lived bias accumulating, without capping the
        # correction below the errors we actually have. The single-combiner fallback
        # keeps the original leak: it has no such gate, and that is where the runaway
        # happened.
        # (FAR-REGIME RE-SEED moved above the presence/rate/hop gates
        #  2026-08-25 -- see the #90 note at its new site.)

        leak = ctx.args.dll_leak_present if fl is not None else ctx.args.dll_leak
        # ONE EXPRESSION, ONE PLACE (2026-08-15). `tau` above and this recurrence used
        # to be inline here AND in the fast-trim thread AND, now, in C++. All three call
        # combdll.dll_integrate / gnss::dll_integrate, which are the same expression,
        # and scripts/gnss/fleetdll_gate.py compares the C++ and Python arms on
        # identical bytes. `tau` is still computed above because the far-regime re-seed
        # block reads it.
        #
        # ⚠️ NOT WHEN THE C++ FLEET LOOP OWNS THIS CHAIN. Two integrators correcting
        # one code error double-apply during pull-in and fight at the null -- and
        # this one bakes its trim into the SEED cp while the fleet loop's arrives at
        # the tracker, so they stack invisibly. The slow loop still MEASURES (the
        # disc/q log lines and the far-regime re-seed stay); it just stops actuating.
        # THE HANDOVER, not a flag-day (#79; the condition eec1d2f12 attached to any
        # re-arm). The old test was `not args.fleet_trim_url` -- chain-wide, so setting
        # the flag silenced the Python integrator for EVERY PRN while the C++ loop
        # trimmed only those already armed. On a chain whose search hands presence to
        # the C++ side that is harmless; on the four DEAD-RECKON chains it removed the
        # only route from off-peak to on-peak and left 0-1 armed with no code loop at
        # all -- measured, and the reason #49 was disarmed there.
        #
        # So: authority is per-PRN and follows the LAST POSTED armed set (what the C++
        # loop is actuating this instant, not what this cycle is about to decide).
        # Python integrates for PRNs the fast loop is not touching -- acquisition
        # authority -- and stands down for each one as the fast loop takes it. Never
        # both (two integrators on one state is the #52 disease), never neither.
        #
        # The standing trim is NOT popped at handover: it is baked into the seed's
        # code_phase_chips at post time and stays a valid offset, which the C++ loop
        # then corrects the residual of, starting from zero. That keeps the phase
        # continuous across the transition in the direction that matters (acquire ->
        # track). ⚠️ The reverse transition still steps: a disarmed C++ trim expires at
        # the tracker's 4 s TTL while Python resumes from its standing value. That is
        # the pre-existing expiry-is-a-step hazard (audit section 6), not new here.
        if prn not in ctx.dls.armed_last:
            ctx.dls.trim[prn] = combdll.dll_integrate(
                ctx.dls.trim.get(prn, 0.0), disc, ctx.args.dll_gain, leak, 3.0,
                ctx.args.dll_spacing)
        ctx.dllp.report.append(
            "PRN %d disc %+.3f trim %+.2f%s"
            # .get: when the C++ fleet loop owns this chain the integrator above is
            # skipped and dll_trim may have no entry -- indexing it killed the gps_l5
            # chain thread with KeyError(20) at 13:17:11 on 2026-08-15, and the seeds
            # expired 60 s later. The DLL line still reports disc/q; trim reads 0.
            % (prn, disc, ctx.dls.trim.get(prn, 0.0),
               "" if fl is None
               else " [fleet %d/%d q %.2f p %.1fx%s]"
                    % (fl["n_src"], len(ctx.dll_combiners), fl["q"],
                       fl["p_pow"] / fl["p_med"] if fl.get("p_med") else 0.0,
                       # Which gate admitted this PRN (#49). Printed only for the
                       # deep gate, so the opt-in set is identifiable in the log
                       # without diffing against the prompt-gated majority.
                       "" if fl.get("present_gate") != "deep" else
                       " DEEP %.1f/%.1f" % (fl.get("deep_gate_snr", 0.0),
                                            fl.get("deep_gate_floor", 0.0)))))


def stage_watchdog(ctx):
    """WATCHDOG: how long has each seeded PRN gone without coherence, and what to do about it.
        
    Tracks per-PRN birth and last-coherent stamps so a satellite that is seeded but never locks
    gets noticed rather than sitting in the table forever. Probe PRNs are exempt -- they are seeded
    deliberately to make the combiner emit noise records and are never expected to lock."""
    if ctx.args.watchdog_s > 0.0:
        for prn in list(ctx.seeds):
            if prn in ctx.probe_set:
                continue
            ctx.wd.birth.setdefault(prn, ctx.t0)
            _r = ctx.status.get(prn) or {}
            if (_r.get("coherence_s") or 0.0) > 0.0:
                ctx.wd.coh_t[prn] = ctx.t0
            else:
                ctx.wd.coh_t.setdefault(prn, ctx.t0)
            _fr = ctx.det_fresh.get(prn)
            # TRIM-RAIL RESCUE (2026-07-20): a trim parked at the +-carrier-max-hz rail
            # is pathology by construction -- converged trims are a few Hz around the
            # chain's common LO offset, and a railed loop plus the tracker fence can
            # self-sustain an NCO alias (E1: 4 ms records = 125 Hz ambiguity; the GPSDO
            # walk railed every trim at +100 and the fleet sat incoherent for 4 h while
            # the det bar (100) hid them from this watchdog at E1's det snr ~45). The
            # rail IS the evidence, so a railed sat is judged at the ordinary presence
            # bar (2x acquire) instead of the strong det bar.
            _railed = (ctx.args.carrier_max_hz > 0.0
                       and abs(ctx.car.trim.get(prn, 0.0)) >= 0.95 * ctx.args.carrier_max_hz)
            _det_bar = (2.0 * ctx.args.acquire_snr if _railed else ctx.args.watchdog_det_snr)
            _reseed = None
            if (ctx.t0 - ctx.wd.birth[prn] > ctx.args.watchdog_s
                    and ctx.t0 - ctx.wd.coh_t.get(prn, ctx.t0) > ctx.args.watchdog_s
                    and _fr is not None and ctx.t0 - _fr[1] < 10.0
                    and prn in ctx.best and ctx.best[prn][0] >= _det_bar):
                _reseed = ("det snr %.0f but ZERO coherent emits for >%.0f s%s"
                           % (ctx.best[prn][0], ctx.args.watchdog_s,
                              " (trim RAILED %+.0f Hz)" % ctx.car.trim[prn]
                              if _railed else ""))
            # WEAK-TRACK RESEED (2026-07-20): the coherent-but-weak zombie -- track
            # correlating ~20 dB off-peak with just enough coherence to hide from the
            # zero-coherence test above (C21/C42: sig 11-18 vs det snr strong, 70 min,
            # every rescuer blind). Judge track significance against the det bar:
            # strong det + persistently floor-level track = broken by construction.
            _tsig = max(_r.get("deep_snr") or 0.0, _r.get("amp_snr") or 0.0)
            if (ctx.args.watchdog_weak_sig > 0.0
                    and _fr is not None and ctx.t0 - _fr[1] < 10.0
                    and prn in ctx.best and ctx.best[prn][0] >= ctx.args.watchdog_det_snr):
                if _tsig >= ctx.args.watchdog_weak_sig:
                    ctx.wd.strong_t[prn] = ctx.t0
                    ctx.wd.weak_n.pop(prn, None)  # cleared the bar -> backoff resets
                elif (_reseed is None
                      # 3x birth grace (2026-07-20 13:12 soak): a reseed resets the
                      # deep ladder and sig takes 60-120 s to rebuild past the bar, so
                      # a 1x window re-fired on its own aftermath -- metronomic churn
                      # on healthy ramping sats (E3 at 50 dB-Hz reseeded 3x at birth).
                      # A real zombie (70 min) doesn't care about a 135 s judgment.
                      # EXPONENTIAL BACKOFF (14:05 soak): track sig alone cannot
                      # separate a zombie from a LEGIT-WEAK sat (E1 PRN 8: 29 dB-Hz,
                      # det snr 112 -- det snr barely scales with C/N0 on E1, so the
                      # weak-det exemption fails there) and the bar churned weak sats
                      # at exactly grace cadence. A real zombie is cured by fire #1;
                      # a sat that fires AGAIN earns doubled grace each time (135 s ->
                      # 270 -> 540 -> ... capped 16x), so persistent-weak sats are
                      # left alone while one-shot rescues stay fast.
                      and ctx.t0 - ctx.wd.birth[prn] > (3.0 * ctx.args.watchdog_s
                                                * (2 ** min(ctx.wd.weak_n.get(prn, 0), 4)))
                      and ctx.t0 - ctx.wd.strong_t.get(prn, ctx.wd.birth[prn]) > ctx.args.watchdog_s):
                    ctx.wd.weak_n[prn] = ctx.wd.weak_n.get(prn, 0) + 1
                    _reseed = ("det snr %.0f but track sig %.0f < %.0f for >%.0f s "
                               "(coherence %.2f, fire #%d -- WEAK-TRACK zombie)"
                               % (ctx.best[prn][0], _tsig, ctx.args.watchdog_weak_sig,
                                  ctx.args.watchdog_s, _r.get("coherence_s") or 0.0,
                                  ctx.wd.weak_n[prn]))
            if _reseed is not None:
                _log("WATCHDOG RESEED PRN %d: %s -> drop + fresh seed (tracker "
                     "state resets via the active-list gap)" % (prn, _reseed))
                del ctx.seeds[prn]
                ctx.dls.trim.pop(prn, None)
                ctx.dls.last.pop(prn, None)
                ctx.cp_held.discard(prn)
                ctx.hold.miss.pop(prn, None)
                ctx.cpt.escape.pop(prn, None)
                ctx.cpt.err_hist.pop(prn, None)
        for k in list(ctx.wd.birth):
            if k not in ctx.seeds:   # any unseeding path re-stamps birth on re-entry
                ctx.wd.birth.pop(k, None)
                ctx.wd.coh_t.pop(k, None)
                ctx.wd.strong_t.pop(k, None)
