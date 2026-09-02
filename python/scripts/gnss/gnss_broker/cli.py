"""The broker's command-line surface: the parser, and the frozen constants (task #89).

WHY THIS IS A SEPARATE FILE. 1,938 lines of `add_argument` and 567 lines of frozen tuning were
sitting in the middle of `main()`, between `install_dns_cache()` and the first line of actual
work. Nothing in either block depends on anything in the broker -- they reference `argparse`
and nothing else -- so their only effect on the broker file was to put ~2,500 lines between a
reader and the algorithm.

⚠️ `--help` PRINTS THE BROKER'S DOCSTRING, NOT THIS ONE. The parser used to be built with
`description=__doc__`, which in this file would silently mean the module docstring you are
reading. The description is passed in by the caller for exactly that reason.

⚠️ THE FLAG COUNT IS ITSELF A DEFECT (#89, and the sweep is not finished). 220 live flags
against 135 settings used in the production yaml. The method that retired the first 72 is
here: freeze, prove equivalence over every fixture, delete.

@author Keith Vanderlinde
"""

import argparse


# ── FROZEN TUNING (task #89) ─────────────────────────────────────────────────────────────
# Every name below WAS a broker flag. Not one of them was set in the production config, in
# any launch script, or in any gate or fixture in this repo -- so each has only ever taken
# the value it is given here, and this block is what the instrument has always run.
#
# THIS IS NOT A DEFAULTS TABLE. A default is a value you get when you decline to choose; a
# frozen constant is a value nobody may choose. The distinction is #86's, which cost a day:
# `sigma_rate0` defaulted to `rate_max`, so a PRIOR and a garbage CEILING shared one number
# and one 0.94-chip innovation could teach any clock rate. With 278 flags that collision was
# unreadable. The CLI now names only what we actually decide (~30 numbers in production), and
# anything that is really a physical constant lives here, next to the reason it has its value.
#
# TO CHANGE ONE: edit it here, in a commit that says why, and run
#     scripts/gnss/broker_equiv.py check <fixture>   # all four must stay EQUIVALENT
# Re-adding a flag is also fine -- but only alongside the A/B that needs it, never "just in
# case". The 278 got there one just-in-case at a time.
#
# The help text each one carried is kept verbatim: it is incident history, not documentation.
_FROZEN = dict(

    # --almanac-epoch-utc0
    #   the tracker's capture_utc0 (gnss_node.yaml shared value, 1.0): subtracted from combiner
    #   row utc to get the FILE POSITION that advances the --almanac-epoch clock at the data's
    #   own rate.
    almanac_epoch_utc0=1.0,

    # --bias-alpha
    #   EMA weight for the clock-freq bias (smaller = steadier seed Doppler; ~0.05 =>
    #   few-second time constant, dithers out the 500 Hz grid)
    bias_alpha=0.05,

    # --carrier-bleed-hz
    #   Shadow: minimum standing |trim| (Hz) to consider a bleed candidate. Below this the sinc
    #   loss is negligible and not worth a re-pin.
    carrier_bleed_hz=2.0,

    # --carrier-bleed-lockout-s
    #   Minimum seconds between bleeds of the same PRN (anti-churn).
    carrier_bleed_lockout_s=120.0,

    # --carrier-bleed-max-slope
    #   Shadow: max |least-squares trim slope| (Hz/s) over the window. A still-settling trim
    #   drifts (low spread but a monotonic climb); bleeding it folds a mid-convergence value
    #   and leaves a residual (the 2026-08-03 REFUTED class). Requiring a FLAT slope, not just
    #   low spread, rejects those.
    carrier_bleed_max_slope=0.02,

    # --carrier-bleed-ok-hz
    #   Post-bleed |residual| (Hz) at the end of the verify window to count VERIFIED. Judged on
    #   the SETTLED residual, not coh_ok -- a re-pin resets the deep coherent window for ~1
    #   emit (coherence_s blips) even on a good bleed, so coh_ok would false-refute; and a mild
    #   remainder still reduced the standing trim (net win).
    carrier_bleed_ok_hz=1.5,

    # --carrier-bleed-shadow
    #   f_ref TRIM-BLEED SHADOW (1 = on, log-only): report where a converged, coherent,
    #   STANDING carrier trim WOULD be re-pinned into f_ref (which would drop the
    #   sinc(ctrim*T_rec) despread loss, ~0.13-0.26 dB on L2C). Takes NO action -- it validates
    #   the TRIGGER on live data before any tracker change (the fleet-collapse lesson: a
    #   carrier correction that fires on the wrong sat is catastrophic). Gated on coherent +
    #   converged so it can never flag the BOOTSTRAP/incoherent noise-walk that killed
    #   alias-escape v1/v2.
    carrier_bleed_shadow=1,

    # --carrier-bleed-stable-emits
    #   Shadow: consecutive coherent emits the trim must hold steady over to count as CONVERGED
    #   (a moving trim is still settling -- not a candidate). Longer window makes the drift
    #   SLOPE (below) measurable above noise.
    carrier_bleed_stable_emits=8,

    # --carrier-bleed-stable-hz
    #   Shadow: max peak-to-peak trim spread (Hz) over the stability window to call it
    #   converged.
    carrier_bleed_stable_hz=0.4,

    # --carrier-bleed-verify-emits
    #   Emits to watch after a bleed before judging the re-pin by its residual.
    carrier_bleed_verify_emits=3,

    # --carrier-rate-clip-hz
    #   clip rate residuals this far from the fleet MEDIAN before the weighted consensus (<=0
    #   disables). Median first, because a wrong-bin outlier is arbitrarily far and would drag
    #   any mean.
    carrier_rate_clip_hz=25.0,

    # --carrier-rate-max-gap
    #   max window gap, in RECORDS, across which a rate measurement is still believed.
    #   Measured: contiguous emits step -0.24..-0.72 Hz, while any gap steps by tens of Hz
    #   because the tracker re-anchored. A gap re-baselines rather than integrating a reference
    #   change.
    carrier_rate_max_gap=2.0,

    # --carrier-rate-max-step
    #   reject a rate residual that jumped more than this since the last believed sample, and
    #   re-baseline. Catches the tracker's f_ref FENCE re-anchor, which fires mid-tracking with
    #   no window gap to mark it: the phase-continuity fold keeps the phase smooth but the
    #   FREQUENCY steps. Measured separation -- contiguous emits step 0.24-0.72 Hz, the
    #   smallest re-anchor jump was 7.6 Hz.
    carrier_rate_max_step=3.0,

    # --carrier-refade
    #   TRACK-mode DEMOTION: after this many consecutive gated residuals (fade-hold or
    #   innovation-reject) while the sat is still PRESENT (amp_snr >= --hold-snr), drop it back
    #   to BOOTSTRAP so the loop re-acquires at full gain (0 = never). Without this the two
    #   TRACK gates form an ABSORBING state: a seed-doppler step (un-precomped hold release /
    #   escape re-anchor) leaves a residual above the innovation gate, decoherence turns off
    #   coh_ok, and the sat parks carrier-dead with a perfectly measurable residual forever --
    #   measured 2026-07-18 on B1C: C20 latched at -6.2 Hz for 40 min at full amp while deep
    #   sat on the floor; the strongest (fastest-slewing) sats latch first, the weak ones never
    #   certify into TRACK and self-heal, inverting the fleet. The presence bar keeps a
    #   genuinely faded sat coasting on the feed-forward (the pathology --carrier-min-sig
    #   exists for); the innov gate's designed escape ('re-seed -> BOOTSTRAP') never fires for
    #   a held strong sat.
    carrier_refade=10,

    # --clock-bias-alarm-hz
    #   CLOCK DRIFT ALARM bar: loud log if the live bias EMA departs the warm-start calibration
    #   by more than this (GPSDO unlock / thermal event -- hardware news, not something to
    #   silently absorb)
    clock_bias_alarm_hz=10.0,

    # --code-bias-alarm-ppm
    #   same alarm for the code-rate clock (l-a) vs its warm-start value
    code_bias_alarm_ppm=0.05,

    # --code-bias-max
    #   reject per-sat l-a samples beyond +-this (ppm) before the median -- a
    #   noisy/unwrap-blown slope fit is an outlier a few-sat median can't reject; the seeded
    #   code rate must stay ~0.1 ppm stable or the deep decoheres
    code_bias_max=3.0,

    # --coh-floor-margin
    #   fleet coherent: multiple of the MEASURED null floor a fleet deep_snr must clear to be
    #   published as a detection. The floor is re-measured every cycle by shuffling each
    #   instance's records (real amplitudes, no common phase), exactly as the fleet DLL
    #   measures its q floor -- never a constant, because the right bar depends on fleet size
    #   and window length. Characterised: null 99th pct 4.15 / max 5.73 vs a weakest real
    #   detection of 22.2, so 3x sits in a clean gap.
    coh_floor_margin=3.0,

    # --coh-min-instances
    #   fleet coherent: instances that must see a PRN before it is combined. Below this the PRN
    #   keeps its single-instance coherent numbers.
    coh_min_instances=3,

    # --decoded-eph-fallback
    #   when the network BRDC fetch fails, keep the dead-reckon predict alive off THIS broker's
    #   own on-node decoded ephemeris (decoded_eph.py, which reuses each decoder's
    #   BRDC-validated propagator). Geometry+Doppler are exact; precise sat-clock is
    #   Galileo-complete, best-effort elsewhere. 1=on (default), 0=off (pre-fallback behaviour:
    #   predict goes dark).
    decoded_eph_fallback=1,

    # --det-alias-fold
    #   ALIAS-BIN CENSUS (diagnostic only): log detections whose Doppler sits a record-alias
    #   quantum 1/(2*t_rec) off the model+bias reference (25 Hz L2C, 50 Hz B1C). The v1 of this
    #   flag FOLDED the dop before the cp currency conversion -- WRONG: the search
    #   back-projects cp0 with the same reported dop, so the round trip is exact for any bin,
    #   and folding broke the cancellation by K*t_abs*k*q (12-57 chip candidates on held L2C
    #   sats, caught by the track-vs-model monitor within the hour). 0 silences the log.
    det_alias_fold=1,

    # --dll-hop-window-s
    #   fleet DLL: how far back from the newest window an instance's measurement may be and
    #   still join the sum, in seconds (converted once to an integer hop count -- the key
    #   itself is never a float). Instances free-run, so their emit phases differ by up to one
    #   emit period; the code moves 0.0033 chips/s (measured), so even a full second of spread
    #   is 0.003 chips, far below anything the discriminator resolves. DEFAULT 5 s, NOT the
    #   0.084 s spread measured on 2026-08-03: instances free-run, so their emit phases
    #   random-walk apart without bound, and within hours that spread had grown to 0.503 s. A
    #   0.5 s window then straddled it and the fleet flapped 14 -> 8 instances sample to
    #   sample, silently halving the combined bandwidth. Size this for DRIFT, not for a
    #   snapshot -- 5 s costs 0.017 chips.
    dll_hop_window_s=5.0,

    # --dll-quality-min
    #   fleet DLL: FALLBACK q bar, used only when fewer than 8 PRNs report and the noise
    #   population cannot be characterised. 2.2 is the measured single-instance bar (noise mean
    #   ~1.0 with a tail to 1.87 on sky 2026-08-03), so the fallback is the conservative
    #   one-node answer.
    dll_quality_min=2.2,

    # --dop-max-rate-hz
    #   SAFETY NET (not a fence): clamp how far the seed Doppler may move in ONE cycle. Bounds
    #   the damage from a garbage prediction slamming the tracker, without imposing discrete
    #   steps. A real MEO Doppler moves <1 Hz per 0.2 s cycle, so this only ever fires on a bad
    #   model.
    dop_max_rate_hz=None,

    # --dop-rate-max
    #   reject a fitted doppler rate beyond this (Hz/s) and fall back to the almanac's.
    #   PHYSICAL, not tuned: GPS Doppler acceleration peaks near 0.94 Hz/s at L1, so ~0.70 at
    #   L5 (x 1176.45/1575.42); 0.8 leaves margin. Seeding a noise-fitted rate ADDS curvature
    #   error -- observed on deploy, PRN 20 fitted at -1.16 Hz/s.
    dop_rate_max=0.8,

    # --dop-rate-min-pts
    #   detections needed before the MEASURED doppler rate is seeded in place of the almanac's.
    #   Below this the model is the better bet.
    dop_rate_min_pts=4,

    # --dop-rate-min-span-s
    #   baseline the doppler-rate fit must span (s). Slope error goes as sigma/(T*sqrt(N/12)),
    #   so at ~1.5 Hz detection scatter 4 points over 44 s give ~0.06 Hz/s and 8 over 88 s
    #   ~0.02 Hz/s, against BRDC's measured 0.108 Hz/s error. A short baseline fits noise.
    dop_rate_min_span_s=30.0,

    # --dop-rate-model-tol
    #   reject a FITTED doppler rate that disagrees with the model's by\nmore than this (Hz/s)
    #   and keep the model. --dop-rate-max only\nbounds magnitude, so a wrong-SIGNED or
    #   half-size fit passes it;\nmeasured on sky, PRN 8 was seeded 2.18x small and PRN 30 with
    #   the\nwrong sign. BRDC range-rate differencing is good to ~0.05 Hz/s, so\n0.12 admits
    #   real fit noise and rejects a fit that is simply wrong.\n0 disables the cross-check (the
    #   pre-2026-08-05 behaviour).
    dop_rate_model_tol=0.12,

    # --dr-clock-adopt-max-age-s
    #   with --dr-clock-adopt: refuse a sibling record older than this. Staleness is a REFUSAL,
    #   not a fallback (receiver_state.read_state) -- an old clock is a different epoch's
    #   number, and after an F-engine restart it is exactly the wrong one. On refusal the chain
    #   keeps what it has and says so.
    dr_clock_adopt_max_age_s=60.0,

    # --dr-clock-adopt-max-slew
    #   with --dr-clock-adopt: refuse a sibling whose clock is MOVING faster than this
    #   (chips/s), measured across two consecutive reads one cycle apart. Deliberately NOT the
    #   sibling's own quality fields: integ_mad_chips is a MAD over a mixed satellite
    #   population and read 3.3-3.9 chips while the clock itself was stable to 0.2
    #   (2026-08-08), and `untrusted` counts a persistent demoted set not comparable to `n`.
    #   Watching the number is the honest test. Default 2.0 chips/s: real drift is ~0.02, a
    #   non-converged estimate moves by thousands.
    dr_clock_adopt_max_slew=2.0,

    # --dr-clock-alpha
    #   EMA weight for the solved receiver clock (wrap-aware chips; the held value propagates
    #   at f_chip*(l-a) between solves)
    dr_clock_alpha=0.2,

    # --dr-long-code
    #   dead-reckon the code phase at the FULL secondary-overlaid code length
    #   (--long-code-segments * code length) rather than one primary period. Required for any
    #   chain without a blind search: E5a/B2a carry per-PRN secondaries, so nothing measures
    #   the segment and the model must supply it. With this off (the historical behaviour)
    #   every dead-reckoned seed lands in segment 0 -- correct 1 time in LC_SEG, which for
    #   E5a's CS100 is 1 in 100. GPS masked the bug because its blind search re-seeds with a
    #   measured nh. Needs absolute time to half a primary period (0.5 ms); the GPS-disciplined
    #   anchor is microsecond-class.
    dr_long_code=True,

    # --dr-max-eph-age-s
    #   do not trust the BRDC Doppler for a satellite whose ephemeris toe is older than this (4
    #   h). Stale/absent model -> fall back to the search-measured Doppler. The fallback is
    #   SEAMLESS: switching Doppler SOURCE is just another currency translation, not a loss of
    #   lock.
    dr_max_eph_age_s=14400.0,

    # --dr-max-integrity-chips
    #   demote a satellite to search-anchored when its dead-reckon integrity residual
    #   (measured-vs-predicted code phase, already computed every cycle and normally +-0.2
    #   chips) exceeds this. THIS is the resilience the fence never provided: it detects a
    #   model that is WRONG, which a fence on Doppler MOTION cannot.
    dr_max_integrity_chips=1.0,

    # --dr-max-off-jump-chips
    #   exclude a satellite from the receiver-clock solve when its offset jumps by more than
    #   this between consecutive cycles. d_i = clk + b_i, and BOTH terms are stable (clk drifts
    #   ~4e-4 chips/s, b_i is a per-sat constant of +-3-7 chips), so a real satellite moves far
    #   less than 10 chips per cycle while a noise or cross-correlation track scatters
    #   ~uniformly over the code. Same physics as --dr-max-solve-mad-chips, applied per
    #   satellite instead of to the population, so ONE bad track is dropped instead of the
    #   whole solve being refused. 0 disables.
    dr_max_off_jump_chips=100.0,

    # --dr-off-jump-max-age-s
    #   ignore the --dr-max-off-jump-chips test when the previous cycle's offset for that
    #   satellite is older than this: across a long gap the clock really can have moved, so a
    #   stale comparison would reject a satellite that is fine.
    dr_off_jump_max_age_s=120.0,

    # --dr-slew-cap-acq
    #   Per-event slew ceiling (chips) for a seed that is still FAR from its target --
    #   acquisition authority. Falls back to --dr-slew-cap (0.05) once within
    #   --dr-slew-near-chips, so an established, converged track keeps today's restraint. 0
    #   disables. ⚠️ THE FLAT CAP IS WHY P2b's SLEW CONSUMER WAS UNOBSERVABLE: 47%% of steps
    #   sat exactly on it while satellites 5-8 chips out closed at 0.05 a go every 10-30 s --
    #   of order an HOUR, against a ~600 s oscillation. Moving the TARGET does nothing while
    #   the step is railed. Keep this at or below the E/P/L span (~0.5): past that the seed
    #   outruns the loop that must follow it (the reverted 10 s re-pin, deep_snr 221 -> 17).
    dr_slew_cap_acq=0.0,

    # --dr-slew-near-chips
    #   Inside this distance the seed has arrived and the flat --dr-slew-cap applies. Distance
    #   rather than track AGE is the maturity test on purpose: it re-arms when an established
    #   track drifts off, which is exactly the ~600 s oscillation case an age-based schedule
    #   would lock in.
    dr_slew_near_chips=0.2,

    # --dr-slew-trust-sigma
    #   Only move at the acquisition ceiling toward a target known this well (chips, 1-sigma on
    #   the joint clk+b_sat for that satellite). A trust GATE, not a multiplier -- scaling the
    #   cap by n x sigma is inverted, it makes a better-measured offset move slower.
    dr_slew_trust_sigma=0.5,

    # --dr-solve-refused-rebootstrap-s
    #   after the clock solve has been REFUSED continuously for this long, force a one-shot
    #   re-bootstrap (clear the held clock and let the next median snap). 0 disables. THE
    #   REFUSAL GUARD IS A LATCH WITHOUT THIS: MEASURED 2026-08-10, one non-L5 PRN poisoned the
    #   median at 19:39, the clock froze at 19:47 and never re-solved, because the scatter that
    #   triggers the refusal is itself sustained BY the frozen clock (the model drifts off the
    #   sky, every sat reads as noise, MAD stays at 2045). Before the guard existed this same
    #   failure random-walked out in ~15-20 min (docs 11.33); the guard stopped the corruption
    #   and removed the escape with it. A forced re-roll is explicitly a RE-ROLL, not a
    #   measurement -- it restores the random walk rather than pretending to solve.
    dr_solve_refused_rebootstrap_s=300.0,

    # --drop-amplitude
    #   combined |A| below which a tracked PRN is coasting (fallback metric when the combiner
    #   reports no deep significance / nav-bit wipe is off)
    drop_amplitude=0.3,

    # --element-archive-every-s
    #   archive cadence; the gains move on the cal EMA (~1 s) but the beam traces move on the
    #   transit timescale, so 60 s loses nothing downstream while keeping the file ~MB/day
    #   scale.
    element_archive_every_s=60.0,

    # --escape-amp-veto
    #   VETO a hold-escape while the held track's incoherent amp_snr exceeds this ABSOLUTE
    #   value. Physics: the false lobes the escape exists to catch sit at prompt -12 dB, so a
    #   hold despreading at full amplitude CANNOT be on one -- an accusing fit is wrong by
    #   construction (2026-07-18: the phantom-sloped L2C fit dragged healthy 200-800-amp holds
    #   off-peak every ~60 s; the strongest observed sats' false lobes read <~50). amp_snr
    #   ONLY, never deep (deep stays coherent ON the wrong lobe -- the C34 signature). 0
    #   disables the veto.
    escape_amp_veto=100.0,

    # --fit-maturity-span-s
    #   cp-fit HISTORY SPAN required before the fit is trusted (escape referee + hold
    #   admission). 30 s makes the code-Doppler quadratic observable on every chain. BENCH NOTE
    #   (2026-07-19): 100-s replay legs cannot afford 30 s of maturity + overlay consensus --
    #   short legs go bimodal on B1C deep (some sats never sync, deep ~15 vs 220). Benches pass
    #   ~10; the A/B verdict discipline requires it.
    fit_maturity_span_s=30.0,

    # --fleet-coherent
    #   CROSS-NODE COHERENT COMBINE (fleet_coherent). Polls /get_records from the
    #   --dll-combiners endpoints and removes the per-record COMMON SKY PHASE -- ~0.75 rad,
    #   0.984-coherent across instances but only ~0.57 autocorrelated in time, which is why no
    #   within-node temporal estimator can touch it and why the deep folds sat at ~14 sigma
    #   against a ~100 sigma thermal ceiling. Each instance is corrected against the OTHERS
    #   (leave-one-out), and the fleet total uses a one-way split so the noise estimate
    #   survives. Measured 2026-08-05: 14 -> 170-700 on the bright sats, against a
    #   shuffled-null floor of ~4. Costs one extra REST poll per combiner and ~0.2 s per cycle;
    #   publishes into the coherent fields the FleetPublisher previously had to take
    #   best-of-one-instance.
    fleet_coherent=True,

    # --fleet-trim-hold-s
    #   keep a PRN armed to the C++ fleet loop this long after it was last PRESENT. ⚠️ Without
    #   this the loop is disarmed BY THE ERROR IT EXISTS TO FIX: the +-1-chip clock breathing
    #   (~19 s period, measured in the fleet disc 2026-08-15) sweeps the tap off-peak, prompt
    #   power collapses, presence drops, the PRN disarms and its trim is released -- so arming
    #   sampled the breathing's phase each ~12 s cycle and held ~30%% duty. The hold must
    #   exceed the breathing period by margin; a satellite that truly set is still released
    #   within ~1.5 min, and the E/L discriminator it leaves behind is noise-mean-zero, so a
    #   held arm on a dead PRN random-walks nothing (the leak reverts it).
    fleet_trim_hold_s=90.0,

    # --hold-max-dop-hz
    #   release a held seed when the fresh Doppler departs the FROZEN one by more than this:
    #   the stale replica carrier decoheres the SINGLE-RECORD despread. Default = 0.1 cycle per
    #   record = 0.1*chip_rate/code_length -- it MUST scale with the record period (100 Hz for
    #   GPS 1 ms records, 25 Hz for E1C 4 ms, 10 Hz for B1C 10 ms; the GPS-calibrated 100 Hz on
    #   B1C let the despread walk into the sinc NULL: amp oscillated 778<->0 on ~1 min cycles,
    #   the first tri-constellation night's BDS symptom).
    hold_max_dop_hz=None,

    # --integ-veto-baseline-s
    #   #98/#99 (2026-08-28): judge the escape referee's INTEGRITY VETO on the EXCURSION from
    #   the sat's own recent integ median (window this many seconds), not the absolute value.
    #   The veto's premise -- "a FRESH integ excursion past the bar means the SEARCH is the
    #   suspect" -- rots when a sat carries a CHRONIC model offset (#99: per-sat +-5 chips,
    #   drifting on minutes; G28 read -8 all evening): the absolute test then vetoes EVERY
    #   escape for that sat forever, and a hold whose frozen entry rate is bad WALKS unbounded
    #   (G28 21:24-21:30: CP_ERR +0.5 -> +28 chips, p 60x -> 12x, veto pinned). Relative form:
    #   a chronic offset cancels in the median; a real search lobe-jump still moves integ
    #   instantly and vetoes. Needs >=5 baseline samples in the window (~2.5 min at the ~30 s
    #   integrity cadence); until then the ABSOLUTE test stands (cold-start protection
    #   unchanged). 0 disables (absolute veto, pre-#98 behaviour).
    integ_veto_baseline_s=0.0,

    # --lock-prompt-hold
    #   HOLD-ON-LOCK, FOLD-INDEPENDENT PATH (2026-08-14, #58). A satellite also counts as
    #   locked when its FLEET PROMPT POWER over the live noise median reaches this, regardless
    #   of what the deep fold says. 0 disables, restoring the pre-#58 gate.\nWHY: the gate
    #   metric was max(amp_snr, deep_snr-if-certified), so it asks the DEEP FOLD whether we are
    #   locked -- and #58 measured that fold failing to certify ~50%% of the time on satellites
    #   the despread is holding continuously (prompt power flat through every collapse; all 12
    #   instances failing together; unchanged with the broker SIGSTOPped). When it fails, sig
    #   falls back to amp_snr, which is moment-debiased and routinely below 3, so a perfectly
    #   tracked satellite reads UNLOCKED and gets re-pinned on the --dr-repin-s timer. That
    #   closes a loop: fold flickers -> re-pin -> phase discontinuity -> fold fails. And
    #   re-pinning is expensive -- measured 2026-08-09, 10 s re-pin vs held seed on the same
    #   satellite: median deep_snr 16.7 vs 221.2, deep-uncertified 21%% vs 0%%.\nSCALE, and it
    #   is NOT the same units as --lock-snr: this is a POWER RATIO against the noise median
    #   (1.0 = nothing there), while lock_snr is a debiased sigma. They are OR-ed, each against
    #   its own bar, rather than max()-ed, because max() over two different units is
    #   meaningless. Live reference values: PRN 23 hold 22.0 (amp_snr 7.8), PRN 26 4.7 (2.8),
    #   PRN 20 1.0 (3.3) -- note the last one, where the prompt tap is ON THE NOISE while the
    #   fold reports the satellite: that is #48/#47 and it SHOULD re-pin, which this gate gets
    #   right.
    lock_prompt_hold=3.0,

    # --model-primacy-exit-p95
    #   EXIT gate (hysteresis above the enter gate): MINNOV p95 beyond this hands the PRN back
    #   to the search, which re-anchors on its next detection. MINNOV is the flip's own
    #   referee: a walking model (the 2(b) DR-chain class, ~0.6 chips/min) crosses this in
    #   minutes.
    model_primacy_exit_p95=3.0,

    # --model-primacy-min-n
    #   ENTER gate: minimum MINNOV samples in the 10-min window. Births read thousands of chips
    #   at n=1 (cold model vs first fix); a p95 needs a population before it means anything.
    model_primacy_min_n=30,

    # --model-primacy-p95
    #   ENTER gate: a PRN is flip-eligible when its 10-min MINNOV p95 is below this (chips)
    #   with at least --model-primacy-min-n samples. From the measured distribution (1.4-1.9 on
    #   healthy established sats), not the old 1-chip guess.
    model_primacy_p95=2.0,

    # --model-primacy-starve-s
    #   EXIT gate: no MINNOV sample for this long (satellite faded, search stopped seeing it,
    #   joint row evicted) also exits the flip -- a model-primary sat whose referee has gone
    #   silent is the #48 noise-parking shape, and nothing may hold a seed unrefereed.
    model_primacy_starve_s=600.0,

    # --nh-hint-min-samples
    #   DISTINCT detections (by ref_hop) pooled before the nh offset is trusted and hints are
    #   sent. The constant is common to all satellites, so this fills quickly. Counts distinct
    #   detections, not cycles: re-counting an unchanged detection every --interval made this
    #   measure UPTIME rather than evidence (fixed 2026-08-10).
    nh_hint_min_samples=6,

    # --period-continuity
    #   what to do when the search's reported overlay period disagrees with the one predicted
    #   from the previous pass. 'check' (default) LOGS the disagreement and applies nothing --
    #   correct since the search began measuring the period from the acquire's coarse lag
    #   (4371ff4eb); a nonzero disagreement on a strong satellite then means the SOURCE
    #   regressed, which wants an alarm, not a silent repair. 'correct' restores the old
    #   override (it stored its own correction and predicted from that, so one bad call was
    #   permanent). 'off' skips the comparison entirely.
    period_continuity='check',

    # --q-stall-bar
    #   q at or above this counts as a locked satellite for the duty (the working lock bar,
    #   chord-trim-quality-gate)
    q_stall_bar=2.2,

    # --q-stall-frac
    #   notice when the trailing duty falls below this fraction of the chain's best. 0.6 is
    #   deliberately loose: #87's regression was a 0.44 -> 0.19 collapse (43%% of best), and a
    #   tighter bar on a statistic with this much transit-driven variance would cry wolf.
    q_stall_frac=0.6,

    # --q-stall-min-best
    #   do not judge a chain whose best duty is below this -- bds_b2b lives near 0.2 for
    #   structural reasons (#31 nav bits) and has no headroom to fall from. A guard that fires
    #   on a chain doing its normal thing is a guard that gets ignored.
    q_stall_min_best=0.25,

    # --q-stall-notice-s
    #   rate limit (s) on the stall notice
    q_stall_notice_s=300.0,

    # --q-stall-window
    #   #70/#87 THE q STALL GUARD: trailing window (s) over which this chain's q duty is judged
    #   against its own best this session. 0 disables. 600 s is ~30-45 broker cycles -- long
    #   enough that a transit or one bad poll cannot trip it, short enough to catch a
    #   degradation in minutes rather than the 3.5 h #87 ran for.
    q_stall_window=600.0,

    # --refade-flicker-s
    #   suppress the --carrier-refade demotion when the residual is SUB-innovation AND the sat
    #   cohered within this many seconds: that is certification-bar sig flicker (settled-era
    #   E1/B1C: ~700 no-op re-pulls/3 h at |resid| ~1.7 Hz), not a stepped NCO. A STANDING
    #   decoherence (the L2C C20 absorbing state, dark for minutes at a sub-gate resid) still
    #   demotes after the window. Needs the track watchdog on for coherence timestamps; 0
    #   disables the guard.
    refade_flicker_s=30.0,

    # --reseed-gain
    #   task #50: fraction of the fitted tau applied per opportunity. The SIGN of spec_tau is
    #   validated (r -0.47..-0.62 against disc on the shoulder); the MAGNITUDE is not
    #   (within-satellite r -0.375 in the far regime), so converge over several fits rather
    #   than betting the whole correction on one unproven number.
    reseed_gain=0.5,

    # --reseed-max-chips
    #   task #50: hard cap on a single re-seed step, chips.
    reseed_max_chips=0.75,

    # --reseed-min-ratio
    #   task #50: minimum spec_peak_ratio. This is a SHUFFLED-NULL significance
    #   (fit_spectrum_delay builds the null from the same points), so 1.5 means the fold beat
    #   its own null by 50%%. Measured 2026-08-12: 11.3%% of emissions clear it -- fine for a
    #   one-shot re-seed (~1 opportunity per 5 min against 25-45 min excursions), and NOT
    #   enough for a continuous loop.
    reseed_min_ratio=1.5,

    # --reseed-q-max
    #   task #50: only re-seed when the fleet q is BELOW this. q = 2P/(E+L) is 1.0 with no peak
    #   in the taps and ~4 at a clean lock, so this says 'the discriminator has nothing to
    #   follow'. Above it the DLL can do the job and a seed step would fight it.
    reseed_q_max=1.2,

    # --reseed-spec-tau
    # (--reseed-spec-tau UNFROZEN 2026-08-24 for #90 -- the E32 off-peak disarm latch needed
    #  it armed per-chain, which is exactly the "alongside the A/B that needs it" clause
    #  above. It is a real flag again; see the argparse entry next to --dll-deep-gate.)

    # --rf-stats-poll-s
    #   Seconds between RF-health polls (default 30). The node refreshes every ~10 s, so a
    #   faster poll re-reads the same pass; the point of the bound is that the broker's cycle
    #   is on cf06's single 1 GbE, which telemetry has already inflated once (#64).
    rf_stats_poll_s=30.0,

    # --state-flush-s
    #   --state-file publish cadence (s). Current state only; history is the scorer's job (8
    #   brokers appending at 1 Hz is ~200 MB/day in a cache directory that has to survive
    #   reboots).
    state_flush_s=1.0,

    # --state-fuse
    #   with --state-file: also compute and PUBLISH this dongle's fused fractional LO estimate
    #   (S2c). STILL WRITE-ONLY -- no seed, gate or estimator consumes it; the broker logs what
    #   the fused prior WOULD have said beside what the chain actually uses, so the flip is
    #   decided on a soak of evidence rather than on argument. Fuses in ppm because carrier (Hz
    #   at this band) and code (l-a in ppm) measure the same FRACTIONAL error, and only from
    #   siblings' RAW values -- fusing their smoothed ones would feed the estimate back on
    #   itself and its covariance would be fiction.
    state_fuse=1,

    # --state-fuse-floor-ppm
    #   covariance FLOOR: no source may claim a standard error below this. ON by default, and
    #   the default was chosen from a live capture, not from theory. The 15-min cross-chain
    #   scan said a floor was unnecessary (pairwise |z| = 0.2 -- pure noise), but that scan
    #   compared MEDIANS over 150 samples and the fuser runs PER CYCLE: on the very first live
    #   fusion the L5 GPS chain claimed se = 0.00018 ppm (0.21 Hz, 100x tighter than any
    #   sibling) while sitting 22.7 sigma off, because its handful of satellites happened to
    #   agree that cycle. Its inverse-variance weight alone would have dragged the dongle's
    #   answer from +32.3 Hz to +14.7 Hz with all three chains actually at +31.5..+33.8. MAD
    #   over 2-6 satellites is a poor scatter estimate and can come out near zero by chance;
    #   the floor is the statement that no chain can beat ~1.5 Hz from a few sats no matter
    #   what its MAD says.
    state_fuse_floor_ppm=0.001,

    # --state-fuse-reject-sigma
    #   drop a source this far from the ROBUST (median) centre, then refit. Judged against the
    #   median, never the inverse-variance mean: with one bad source among three the mean is
    #   dragged far enough that the GOOD sources also exceed the bar, the survivor list comes
    #   back empty and a naive implementation then rejects nothing while publishing the
    #   contaminated estimate. 0 disables.
    state_fuse_reject_sigma=5.0,

    # --watchdog-weak-sig
    #   WEAK-TRACK RESEED bar: a sat the search sees at full det snr (>= --watchdog-det-snr)
    #   whose TRACK significance stays under this for a whole --watchdog-s window is a
    #   coherent-but-weak zombie (correlating ~20 dB off-peak; nonzero coherence evades the
    #   zero-coherence watchdog, correct cp evades the referee, folded resid evades refade --
    #   the C21/C42 class). Healthy weak chains are exempt via the det bar (their dets are weak
    #   too). 0 disables.
    watchdog_weak_sig=30.0,

    # --xband-hint-margin-hz
    #   search margin for a cross-band RESCUE hint (Hz): the cross-band seed accuracy is the
    #   inter-band MAD (~10 Hz) plus this band's own unsolved-LO width, so wider than a BRDC
    #   hint but far tighter than the blind grid
    xband_hint_margin_hz=60.0,

    # --bsat-gain
    #   Per-cycle gain of the b_sat loop (task #33): the per-satellite path-bias filter
    #   fed by the fleet phase-slope tau. 0 disables the UPDATES (b freezes at 0 / its
    #   last value) while the fit and its publishing keep running -- the control setting
    #   for the oscillation causality experiment: the 2026-08-09 tau test found the
    #   closed system limit-cycling at ~10 min with the DLL trim swinging IN PHASE with
    #   b (corr +0.65, trim +-1.1 chips vs b +-0.5), so 'does the plant oscillate with b
    #   frozen?' is what separates 'b causes it' from 'b follows a pre-existing
    #   trim/slew oscillation'.
    bsat_gain=0.02,

    # --carrier-rate-inherit
    #   a PRN failing the q gate takes the fleet's amp_snr-weighted consensus instead of
    #   no correction. OFF since 2026-08-04: this was built on 'the dominant term is
    #   common-mode', which the sky refutes. Measured across 131 emits with >=3 PRNs
    #   each, the spread of deep_rate_hz BETWEEN PRNs within a single emit has median
    #   60.8 Hz -- near the full +-47.7 Hz range of the search. That is expected in
    #   hindsight: the residual is measured against each tracker's OWN f_ref, which
    #   re-pins per PRN on its own schedule, so there is no shared zero to average
    #   towards. (The claim that IS true, and a different one, is that different NODES
    #   agree on a given PRN.) Inheriting hands a satellite a number belonging to
    #   someone else's reference; free-running is better.
    carrier_rate_inherit=False,

    # --carrier-rate-min-q
    #   hard gate on deep_rate_q before a rate residual is believed. The 8.18.5 gap:
    #   17.9-22.0 on signal, 2.8-6.1 on noise. A weak sat does not merely scatter, it
    #   lands on the WRONG spectral bin (measured: amp_snr 9.5 was 41.7 Hz out by
    #   split-half, where amp_snr 83.7 was 0.000).
    carrier_rate_min_q=10.0,

    # --carrier-source
    #   what the shared carrier loop integrates. 'rate' (default, 2026-08-04) = the
    #   combiner's phase-rate search (deep_rate_hz), measured at peak/median 17.9-22.0
    #   on signal vs 2.8-6.1 on noise and pinned to ~0.2 Hz by split-half on strong
    #   sats. 'resid' = the legacy carrier_hz_resid, which is SIGNAL-FREE (0.519 Hz on
    #   signal, 0.492 on noise) -- kept only to reproduce the old behaviour, never for
    #   science.
    carrier_source='rate',

    # --carrier-trim-const
    #   DIAGNOSTIC: command this fixed carrier_trim_hz to every seeded PRN, independent
    #   of --carrier-gain. Use with --carrier-gain 0 to measure the open-loop step
    #   response of deep_rate_hz: sweep 0 / +X / -X and the measured rate should move by
    #   exactly -X. Same sign => the loop is inverted; no movement => the trim never
    #   reaches the despread.
    carrier_trim_const=None,

    # --cl-autoseg
    #   CL segment AUTO-SEARCH (default ON; the durable fix for the ~40%%-of-launches CL
    #   failure): when the CL-vs-CM verify reads a dead fleet under strong CM, step an
    #   integer-segment correction through 0,-1,+1,-2,... (one 20 ms segment per step)
    #   and LATCH on green. Compensates the whole-segment utc0_sample0 anchor error
    #   (stamped from system_clock::now() on the first USB transfer, tens of ms of
    #   per-launch jitter; the auto-center absorbs only the fractional part). A working
    #   launch latches 0 immediately.
    cl_autoseg=1,

    # --cl-autoseg-dwell
    #   seconds per correction step (tracker re-lock + combiner deep build; the k-scan
    #   measured green appearing well inside 30 s)
    cl_autoseg_dwell=30.0,

    # --cl-kscan-chips
    #   FRACTIONAL scan mode: CSV of CHIP offsets added to the probe PRN's seeded cp
    #   (e.g. '0,0.25,-0.25,0.5,-0.5,0.75,-0.75,1,-1') instead of whole-segment steps.
    #   The comb/sub-chip test: CM/CL are chip-interleaved at 1.023 Mcps (one comb slot
    #   = 0.5 chip of the 511.5 kcps code), and slot parity couples with code phase when
    #   the replica timeline shifts, so scan a fine grid rather than betting on +-0.5 --
    #   a half-chip code offset degrades ~6 dB rather than nulling, so any partial
    #   despread stands far above the ~2 noise floor and names the true offset.
    cl_kscan_chips='',

    # --cl-kscan-dwell
    #   CL k-scan: broker cycles to dwell per offset (the CL combiner's deep integration
    #   must respond before stepping). 20 cycles ~= 4 s at the 0.2 s interval, matching
    #   L2C's coherence window.
    cl_kscan_dwell=20,

    # --cl-kscan-prn
    #   DIAGNOSTIC (default 0 = OFF): step the CL segment for THIS probe PRN through {k,
    #   k-1, k+1, k-2, k+2} and log which offset despreads best. Convention-free test
    #   for the whole-segment anchor bug that fine_ms cannot see (fine is the residual
    #   after round()). Only the probe PRN's SEED is shifted; the fleet's pin, the fine,
    #   and the auto-center are untouched, so this is safe to leave off and harmless
    #   when on. Pick a strong CM sat as the probe.
    cl_kscan_prn=0,

    # --cl-kscan-segs
    #   explicit CSV of SEGMENT offsets for the scan (e.g. the full-75 sweep
    #   '0,-1,1,...,-37,37'). The default +-2 neighbourhood only exonerates SMALL anchor
    #   errors; utc0_sample0 is stamped from system_clock::now() on the FIRST USB
    #   transfer and carries tens of ms of per-launch startup latency -- several 20 ms
    #   segments.
    cl_kscan_segs='',

    # --cl-time-adjust
    #   seconds added to the CL time-assist clock -- escape hatch for a future
    #   non-multiple-of-1.5s GPS-UTC offset or a known host-clock bias
    cl_time_adjust=0.0,

    # --code-bias-force
    #   DIAGNOSTIC: pin the (l-a) code-rate clock offset to this PPM -- the live fit/EMA
    #   still runs and logs (so the fit stays observable) but the SEEDED rate uses this
    #   value only. 2026-07-18: built to test the L2C phantom-l-a hypothesis (fit says
    #   +0.022 ppm, air truth says 0.000).
    code_bias_force=None,

    # --decoded-eph-fallback-force
    #   ALWAYS predict from decoded eph, even when BRDC is available -- the live A/B
    #   validation harness (compare against the BRDC predict in the log). Exercises the
    #   BeiDou BDT frame + CNAV clock the offline test can't. Do NOT leave on in
    #   production.
    decoded_eph_fallback_force=0,

    # --dll-deep-gate-margin
    #   fleet DLL: the deep gate's bar in units of deep_floor (default 3.0). The floor
    #   is the combiner's own rectification level, so this is a detection significance,
    #   not a tuned constant.
    dll_deep_gate_margin=3.0,

    # --dll-quality-sigma
    #   fleet DLL: how many sigma above the MEASURED q noise floor a PRN must sit before
    #   its trim integrates. q = 2P/(E+L) is a peak-SHARPNESS metric, not an SNR:
    #   exactly 1.0 with no peak (all three taps see equal noise power), 4.0 for a clean
    #   lock at 0.5-chip spacing. Summing instances does NOT raise it -- every tap's
    #   mean scales alike -- it SHRINKS its spread as 1/sqrt(K), so the right bar falls
    #   as the fleet grows and no constant can be correct for more than one fleet size.
    #   The floor is therefore re-measured each cycle as median + this many MAD-sigma
    #   over the live q population (most tracked PRNs are signal-free at any moment, so
    #   the median IS the no-peak value), and logged every time it is used.
    dll_quality_sigma=3.0,

    # --dll-spacing
    # (--dll-spacing UNFROZEN 2026-09-02 for the L2C tap widening -- the first chain whose
    #  correlation is NOT chip-narrow needs a per-chain value, which is the "alongside the
    #  A/B that needs it" clause above. It is a real flag again; see the argparse entry next
    #  to --dll-deep-gate. Every other chain still runs the 0.5 it always ran.)

    # --dr-clock-drift
    #   CHORD: prime the dead-reckon clock DRIFT (chips/s). The drift estimator needs
    #   consecutive multi-sat solutions 0.5-30 s apart; a search whose passes take
    #   minutes never provides them, so drift pins to zero on stale repeats and the
    #   clock freezes while the true receiver clock walks (measured 0.044 chips/s = +5
    #   Hz at L5 on the CHORD GPSDO). Priming it makes the age-correction terms treat
    #   minutes-old detections consistently, which is what lets a slow-cadence search
    #   bootstrap the fast tracker loop.
    dr_clock_drift=None,

    # --dr-dry-run
    #   compute + log the clock solve, integrity residuals and planned dead-reckoned
    #   seeds WITHOUT injecting any (validation mode)
    dr_dry_run=False,

    # --dr-max-drift-chips-s
    #   reject a dead-reckon clock DRIFT estimate beyond this (chips/s). The estimate is
    #   a difference of two clock solves, so any discontinuity -- a node or F-engine
    #   restart -- enters as motion that never happened, and the a=0.05 EMA then bleeds
    #   it off over ~10 min while sweeping every model-primary seed off peak. The true
    #   drift is ~4e-4 chips/s on this GPS-disciplined clock, so 1.0 rejects nothing
    #   real (2026-08-09: +223 and -36 chips/s observed after node restarts).
    dr_max_drift_chips_s=1.0,

    # --dr-refresh-s
    #   dead-reckon cadence (clock solve + integrity + pin checks)
    dr_refresh_s=2.0,

    # --dr-repin-s
    #   re-anchor a dead-reckoned (undetected, unlocked) seed from the model this often:
    #   fresh cp/doppler/rate together (a DR seed's doppler is FROZEN between pins --
    #   currency-consistent by construction -- so this also bounds the doppler
    #   staleness; 10 s * max MEO rate ~0.6 Hz/s = 6 Hz, under every band's fence)
    dr_repin_s=10.0,

    # --dr-slew-cap
    #   Per-event ceiling (chips) on the dead-reckon slew -- how far a seed may be
    #   dragged toward the model in one cycle. ⚠️ 0 DISABLES SLEWING: the seed is still
    #   re-anchored at a fresh epoch with fresh Doppler, but its PHASE is left where the
    #   loop put it, never pulled toward the model. That is a real experiment, not a
    #   degradation: on 2026-08-11 tracking was healthy while seeds sat 234-307 chips
    #   from the model and degraded as they ARRIVED on it, which would mean the seed is
    #   right and the model is wrong. Run it on ONE chain against an otherwise identical
    #   sibling (gal_e5a vs gal_e5b sees the same satellites on a different band) and
    #   compare decay at MATCHED time-since-restart -- a restart transiently fixes every
    #   chain for ~20-30 min, so before/after across one is not evidence.
    dr_slew_cap=0.05,

    # --element-poll-every-s
    #   minimum seconds between /get_elements polls per chain (0 = every cycle). THE
    #   POLL IS NOT IN ANY LOOP: the per-element gain table is served for display and
    #   appended to the beam archive, and the archive is ALREADY throttled to
    #   --element-archive-every-s (60). Polling it every cycle costs 12 GETs of ~62 kB
    #   and, profiled live, 197 ms per chain per cycle -- ~1.0 s of the fleet's
    #   serialised cycle, for a product nothing consumes at that rate. The gains move on
    #   the cal EMA (~1 s) but the beam traces move on the TRANSIT timescale, which is
    #   the same argument that set the archive cadence at 60 s.
    element_poll_every_s=20.0,

    # --fleet-trim-leak-per-s
    #   leak (1/s) published to the C++ fleet loop. Sets BOTH the noise mean-reversion
    #   time (1/this ~ 8 s) and the reachable-correction ceiling: trim_max = 0.25 x
    #   bandwidth / leak_per_s = 2.9 chips at the defaults, i.e. the +-3 clamp finally
    #   is the bound. The old pairing (gain 0.25, leak 0.05 per update) capped it at
    #   1.25 chips -- measured max |trim| 1.140 over 8 h, the clamp unreachable BY
    #   CONSTRUCTION.
    fleet_trim_leak_per_s=0.12,

    # --instance-stall-s
    #   Say so when an INSTANCE serves rows whose newest pow_hop has not advanced for
    #   this many seconds while the rest of the fleet does (#70; 0 disables). Costs no
    #   extra polling -- the hop comes from the fleet DLL poll already being made. ⚠️
    #   THIS IS NOT --fe-axis-stale-s. That one watches the MAXIMUM hop across instances
    #   and answers 'has the whole time base frozen?'. It cannot see one instance of
    #   twelve wedged, because the other eleven keep the maximum climbing -- on
    #   2026-08-18 FOUR wedged at once and it stayed correctly silent throughout. ⚠️ AND
    #   A REST ENDPOINT ANSWERING 200 IS NOT A LIVE INSTANCE. Every wedged instance
    #   served plausible, well-formed rows the entire time; cx42/port 0 was dropping the
    #   whole 195,313 pkt/s stream while doing so. Reachability said nothing, which is
    #   why this keys on a COUNTER. Healthy advance is ~5.9M hops/30 s, wedged is
    #   exactly 0. 90 s is ~7 broker cycles: conservative against a poll race, and still
    #   1000x faster than the 25 h the cx19 wedge ran undetected.
    instance_stall_s=90.0,

    # --joint-clk-max-chips
    #   P2b consumer 'clk': refuse the joint CLOCK if it disagrees with the legacy
    #   median by more than this. The median's measured churn oscillation is +-1-2 chips
    #   (the very thing being replaced), so 5 keeps the whole plausible envelope while
    #   refusing a wrap alias or a diverged filter.
    joint_clk_max_chips=5.0,

    # --joint-clk-max-sigma
    #   P2b consumer 'clk': refuse a joint clock with 1-sigma above this (chips). P
    #   grows while the state is unfed, so this one gate covers estimator health AND
    #   staleness; healthy runs measure 0.05-0.08.
    joint_clk_max_sigma=0.5,

    # --joint-feed-min-ratio
    #   minimum spec_peak_ratio for a spectrum fit to anchor the joint feed
    #   (--joint-feed-spec). Same shuffled-null significance the reseed gate uses, same
    #   default as --reseed-min-ratio: locked satellites read ~1.0-1.3 on residuals near
    #   zero (a small tau does not need a tall peak) but a DISPLACED satellite's peak
    #   stands well clear -- and the displaced ones are the ones the trim gate would
    #   otherwise exclude, which is the whole point.
    joint_feed_min_ratio=1.5,

    # --joint-feed-min-set
    #   do not feed the joint state at all unless at least this many satellites qualify
    #   THIS POLL. Eligibility is a property of the SET: a 1-2 satellite poll has
    #   spread(y-clk) ~ 0 by construction, and the 2026-08-23 01:xx arm fed exactly one
    #   rotating satellite per poll -- 19/19 polls DEGENERATE, single innovations to
    #   -6.8 chips, and the clock got WORSE (sd 1.635 -> 2.587). A thinned set must
    #   self-disarm, not feed degenerately.
    joint_feed_min_set=3,

    # --joint-feed-spec-max-age-s
    #   maximum age of the spectrum fit used to anchor the feed. The fit refreshes on
    #   the spectrum-poll cadence (~30 s); a stale tau re-anchors y on where the sky
    #   WAS.
    joint_feed_spec_max_age_s=90.0,

    # --joint-min-sats
    #   a JOINT consumer refuses to act on a state carrying fewer than this many
    #   satellites: with the mean(b)=0 gauge a thin fleet lets one satellite's bias leak
    #   into the clock at 1/N.
    joint_min_sats=4,

    # --joint-p2c-hold-s
    #   how long each P2c coast runs. ⚠️ Must stay below the state's max_age_s (900 s):
    #   a withheld satellite is not fed, so _t_seen stops advancing and _drop() evicts
    #   it -- a hold at or beyond that would end every coast by eviction and report
    #   nothing.
    joint_p2c_hold_s=600.0,

    # --joint-p2c-skip
    #   how many recently-tested satellites to skip when rotating, so the sample spans
    #   the fleet instead of re-testing the strongest sat.
    joint_p2c_skip=4,

    # --joint-sigma
    #   measurement sigma (chips) for --joint-shadow. The search cp noise is 0.03-0.5
    #   chips per-sat-conditions; 0.3 is the middle of that and NOT tuned against the
    #   answer -- the state's own covariance reports whether it was right.
    joint_sigma=0.3,

    # --kcoh-rate-from-row
    #   ARM-17 (the churn killer): inject the JOINT ROW's y-space prediction
    #   (carrier_correction_hz) into the KCOH fold instead of the previous cycle's
    #   record-stream fit, per seeded sat whose row is converged. The fold's headline
    #   (sig/eta) duty-cycles on the fit's own +-10 Hz cycle-to-cycle noise -- the
    #   fleet-wide 'sig oscillation' -- while the row (e5b fine feed + kcoh, arms 15/16)
    #   is 5-10x smoother. The record stream is command-blind on this plant, so the
    #   row's estimate IS the stream's residual rate on commanded and uncommanded chains
    #   alike. Probes have no rows and keep their zero/fit entries: the floor stays
    #   self-calibrated. 0 = off (default).
    kcoh_rate_from_row=0,

    # --kcoh-row-max-sigma
    #   row y-space 1-sigma bar (Hz) for --kcoh-rate-from-row: (f_band/c)*rrate_sigma +
    #   f_carrier_sigma must be at or under this before a row's prediction replaces the
    #   fit. An unmeasured row reads inf and never injects.
    kcoh_row_max_sigma=1.0,

    # --lock-snr
    #   detection significance (sigma above noise) above which a sat counts as locked --
    #   the primary, noise-relative lock metric (vs the noise-biased |A|; noise sits at
    #   ~1, a real lock at >>3)
    lock_snr=3.0,

    # --nh-hint-max-age-s
    #   drop nh-offset samples older than this, and stop hinting entirely when fewer
    #   than --nh-hint-min-samples remain. THE HINT MUST BE ABLE TO EXPIRE. It narrows
    #   the search to +-nh_hint_span of 20 overlay phases, so a WRONG offset points the
    #   scan away from the signal and no detection can arrive to correct it -- a closed
    #   loop whose only escape was the code clock random-walking back onto truth by
    #   chance (observed 2026-08-10: ~15 min of self-reinforcing outage that read as a
    #   frontend sensitivity loss, docs 11.33). Default 600 s is ~half the 1276 s
    #   per-PRN revisit, so a healthy fleet always has fresh samples and only a
    #   genuinely starved one widens.
    nh_hint_max_age_s=600.0,

    # --once
    #   run a single control-loop iteration and exit (for tests)
    once=False,

    # --period-check-snr
    #   detections below this SNR do not enter the period-continuity history and their
    #   disagreements are logged as 'weak det' rather than as source regressions.
    #   Measured on CHORD 2026-08-02: above ~60 the within-period phase is
    #   self-consistent to a few chips across a 400 s gap, below it the residuals are
    #   ~2000 chips, i.e. noise.
    period_check_snr=60.0,

    # --reseed-spec-tau
    #   task #50: PRNs allowed a FAR-REGIME RE-SEED from spec_tau (the cross-channel
    #   phase-ramp delay fit, #32). Comma-separated PRNs or 'all'. DEFAULT OFF. Fires
    #   only where the discriminator cannot help: the #49 deep gate says present, q <
    #   reseed_q_max (E/P/L carry no gradient), and spec_peak_ratio clears
    #   reseed_min_ratio. Applied as a SEED step, not a trim increment -- the slew cap
    #   would swallow a trim. (Was FROZEN off by #89; unfrozen 2026-08-24 for #90's arm.
    #   The gain/cap/ratio/q-max constants stay frozen.)
    reseed_spec_tau='',

    # --rrate-cmd-max-sigma
    #   command a sat's carrier only when its rrate row's 1-sigma (m/s) is below this.
    #   0.5 m/s is ~2 Hz at 1176 MHz -- an UNMEASURED row reads sigma=inf by design (a
    #   dead feed must not pass for a healthy one), so this also gates births: a sat is
    #   commanded only after its row has actually converged.
    rrate_cmd_max_sigma=0.5,

    # --rrate-fine-hold-s
    #   how long a fine acceptance keeps a satellite in the phase-governed regime. On
    #   expiry the coarse feed returns to full weight automatically -- the drop-back an
    #   FLL/PLL handoff needs when the fine gate stops passing (weak sat, arc break,
    #   command slewing hard). Sized well above the observed fine cadence (~1/min) so
    #   ordinary gaps do not flap the regime, and well under the timescale on which an
    #   unsupervised row could drift anywhere interesting.
    rrate_fine_hold_s=240.0,

    # --rrate-kcoh-min-sig
    #   minimum KCOH fold significance for the kcoh rrate feed: below this the fold has
    #   not caught the satellite and its residual is the fit chasing noise (the probe
    #   floor sits at sig ~1; 20 is well clear of the tail). The per-measurement sigma
    #   also scales with the fold: min(0.3, max(0.03, 2/sqrt(sig))) Hz -- a first cut
    #   anchored at the synthetic gate's ~0.03 Hz strong-signal recovery accuracy, to be
    #   re-derived from the innovation stream once this has flown.
    rrate_kcoh_min_sig=20.0,

    # --rrate-phase-sigma
    #   base measurement sigma (Hz) for the fine phase-step feed at the 1-poll span. NOT
    #   0.02: res_cycles TELESCOPES (consecutive dres share a record), so the span noise
    #   is sqrt(2)*sigma_phi/2pi ~ 0.07 cycles regardless of span length -- ~35-50 mHz
    #   over a 2 s poll. (The 08-21 'long baselines are not the fix' note is SUPERSEDED
    #   by measurement: at 16-32 s the staleness term is 0.16-0.32 Hz against a 3-6x
    #   noise win -- see --rrate-phase-span-s. The mHz class still needs the
    #   rate-of-rate state carrying the lag.) The feed inflates this by the command's
    #   motion over the span: sigma_eff = sqrt(sigma^2 + (0.5*dcmd)^2), the worst-case
    #   span-mean reference error for an unknown application time.
    rrate_phase_sigma=0.05,

    # --search-fft-len
    #   the search/tracker fft_len, used only to refine the one-hop epoch conversion in
    #   --seed-phase-transport by its '-1 sample' term (0.0064 chips at CHORD). 0 =
    #   omit; the residual is constant and two orders below the DLL's pull-in.
    search_fft_len=0,
)
# ── end frozen tuning ────────────────────────────────────────────────────────────────────


def build_parser(description):
    """The broker's argument parser. `description` is the BROKER module's docstring."""
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rest-url", default="http://localhost:12048",
                    help="default kotekan REST base for bare stage names")
    ap.add_argument("--detectors", "--searches", dest="detectors", default="",
                    help="detection endpoints: aggregator/search stage names, absolute "
                         "URLs, or {a..b} ranges (e.g. aggregate  or  search_{00..49}). "
                         "OPTIONAL: leave empty to run PURELY MODEL-PRIMARY, seeding every "
                         "visible satellite from the BRDC model with no acquisition search at "
                         "all (--almanac --dead-reckon). That is the CHORD configuration -- the "
                         "chain there has no search stage, because the station position is known "
                         "to millimetres and the model already lands inside the DLL capture "
                         "range, so a blind search would only re-derive what the ephemeris "
                         "already says.")
    ap.add_argument("--trackers", required=True,
                    help="tracker endpoints to seed (e.g. track_{00..49} or "
                         "http://nodeA:12048/track_{0..24},http://nodeB:12048/track_{0..24})")
    ap.add_argument("--combiner", default="combiner",
                    help="combiner endpoint for full-band |A| status (name or URL)")
    ap.add_argument("--interval", type=float, default=0.25,
                    help="control-loop period, s (keep below ~0.5 s drift time)")
    ap.add_argument("--acquire-snr", type=float, default=12.0,
                    help="min detection SNR to (re)seed a PRN")
    # (--trim-precomp / --trim-precomp-carrier / --trim-precomp-coast DELETED, 07-19 audit
    #  A4: the carrier pre-shift was bench-rejected in both signs -- the BOOTSTRAP re-pull
    #  owns step recovery, and under --dop-continuous steps no longer occur; the coast cp
    #  currency translation became unconditional -- it is the same algebra as the hold-path
    #  TRANSLATE, and the flag's OFF default was shipping the known-bad legacy overwrite.)
    ap.add_argument("--coast-to-horizon", action="store_true",
                    help="never drop a visible sat for low signal -- coast on the pure model "
                         "(almanac doppler + pooled code rate, currency-corrected) until it "
                         "SETS. The beam-map mode: sidelobe/null transits keep despreading on "
                         "the predicted trajectory so the unbiased incoherent/coherent power "
                         "observables sample the WHOLE beam, not just where the sat is locked. "
                         "The model holds the code peak for ~1-2 min per the pooled l-a "
                         "uncertainty; the search re-anchors whenever the signal returns.")
    ap.add_argument("--noise-probes", type=int, default=0,
                    help="keep this many deepest-below-horizon PRNs seeded as NOISE PROBES: "
                         "the combiner then emits genuine signal-free records for the beam "
                         "map's pedestal calibration (an almanac-gated broker otherwise never "
                         "tracks one and the pedestal falls back to a signal percentile). "
                         "Probes fail every lock gate naturally; ~2 is plenty.")
    ap.add_argument("--hold-max-cp-err", type=float, default=0.4,
                    help="release a HELD seed when the tracked code phase (held cp + DLL "
                         "trim) disagrees with the search FIT by more than this (chips) on "
                         "3 consecutive fixes. This is the DLL's capture half-range: a "
                         "sharp-ACF (BOC) power discriminator has stable FALSE equilibria "
                         "~0.75 chips out (prompt -12 dB) that the hold would otherwise "
                         "servo forever while the search sees the true peak.")
    ap.add_argument("--post-sat-geometry", type=int, default=0,
                    help="#102: post each satellite's az/el to the record assemblers' "
                         "/set_sat_geometry every ~30 s (endpoint derived from "
                         "--dll-combiners), feeding the per-element steering. 0 = off.")
    ap.add_argument("--eph-rebase", type=int, default=0,
                    help="#101: at each ephemeris refresh, hand the KNOWN per-sat model step "
                         "to the gather as a trim adjustment through the #92 handover "
                         "(--fleet-trim-rebase-adjust must be on), instead of letting the "
                         "leak-limited loop rebuild through the step for ~15-30 min (the "
                         "GAP 3 shadow's measured post-merge trim kicks). Steps below 0.02 "
                         "chips are skipped; the 2.5-chip handover bound applies. 0 = off.")
    ap.add_argument("--fit-flush-on-reject", type=int, default=0,
                    help="#100: after this many CONSECUTIVE cp-rate rejections for a PRN, "
                         "FLUSH its cp-fit history. A rejected fit also poisons the seed's "
                         "POSITION (the fit evaluated at ref_hop: slope error x history span "
                         "= thousands of chips), and the poison self-sustains -- off-peak "
                         "command, weak detections, re-poisoned history -- for as long as "
                         "the history remembers (~8 min at fit-hist-len 256; G9 2026-08-28 "
                         "22:36-23:00). Flushed, the sat rides the birth path while a clean "
                         "fit rebuilds (~30 s). 0 disables (pre-#100 behaviour).")
    ap.add_argument("--integ-veto-baseline-s", type=float, default=0.0,
                    help="#98/#99: judge the escape referee's INTEGRITY VETO on the EXCURSION "
                         "from the sat's own recent integ median over this window (s), not the "
                         "absolute value -- a chronic per-sat model offset (#99) is that sat's "
                         "normal, not evidence against this fit, and the absolute test vetoes "
                         "every escape for exactly the sats that need them (G28's bad hold "
                         "walked +28 chips un-evictable, 2026-08-28). A real search lobe-jump "
                         "still moves integ instantly and vetoes. Absolute test stands until "
                         ">=5 baseline samples exist. 0 = absolute veto (pre-#98 behaviour).")
    ap.add_argument("--hold-snr", type=float, default=8.0,
                    help="incoherent amp_snr above which a tracked PRN's cp anchor is FROZEN "
                         "(hold-on-lock: DLL owns the sub-chip residual; fit re-anchors only on "
                         "loss). Uses amp_snr ONLY -- deep_snr's off-peak value is the nav-wipe "
                         "rectification floor (~7), which would freeze bad anchors instantly.")
    ap.add_argument("--coast-budget", type=float, default=30.0,
                    help="seconds a VISIBLE sat is coasted (seed held + Doppler forecast forward) "
                         "through a signal dropout before dropping it -- so a radar sweep / brief "
                         "fade doesn't lose the lock. The code prediction stays good for ~tens of s "
                         "on a free-running TCXO; raise it with a disciplined clock (OCXO). |A| "
                         "recovering resets the coast; setting below the horizon drops immediately.")
    ap.add_argument("--lat", type=float, help="receiver latitude (enables gating)")
    ap.add_argument("--lon", type=float, help="receiver longitude")
    ap.add_argument("--alt", type=float, default=0.0, help="receiver altitude, m")
    ap.add_argument("--mask-deg", type=float, default=5.0, help="elevation mask, deg")
    ap.add_argument("--almanac", action="store_true",
                    help="orbit-predict each PRN's Doppler (needs --lat/--lon + skyfield): "
                         "seed trackers with the precise predicted Doppler (geometry + a common "
                         "clock-freq bias solved from the measured sats) instead of the coarse "
                         "search grid, and gate to visible sats. Code phase still from the search.")
    ap.add_argument("--tle", default=None, help="GPS TLE file/URL (default: Celestrak gps-ops)")
    ap.add_argument("--almanac-source", default="brdc", choices=("brdc", "tle"),
                    help="orbit source for the almanac assist. brdc (default) = the IGS "
                         "broadcast-ephemeris file (gnss_ephemeris): PRN-INDEXED (immune to the "
                         "celestrak name->PRN label rot that mislabels BeiDou IGSOs -- C31/C38/"
                         "C39 ranges were 16-20k km off on 2026-07-17, corrupting nh hints by "
                         "5-7 overlay chips and hiding C38's true 84-deg pass), ~m orbits vs "
                         "TLE's ~km, includes the sat clock, and needs no skyfield. tle = the "
                         "legacy celestrak path (also the automatic fallback when BRDC is "
                         "unreachable at startup).")
    ap.add_argument("--constellation", default=None, choices=("G", "E", "C", "R"),
                    help="constellation letter for the BRDC almanac (which file covers ALL "
                         "systems, unlike the per-group TLE URLs). Default: --dr-constellation. "
                         "C keeps only PRN >= 19 (BDS-3): the BDS-2 birds transmit B1I at "
                         "1561 MHz, outside our band -- same capability cut the BEIDOU-3 "
                         "TLE name filter encodes.")
    ap.add_argument("--carrier-hz", type=float, default=1575.42e6, help="carrier for Doppler pred")
    ap.add_argument("--doppler-sign", type=float, default=1.0,
                    help="multiply predicted Doppler (set -1 if the convention is inverted)")
    ap.add_argument("--narrow-search", action="store_true",
                    help="push per-PRN predicted Doppler to the detectors' /set_doppler_hints so the "
                         "search scans only doppler +- margin (almanac-narrowed acquisition) instead "
                         "of its blind grid; needs --almanac. Far cheaper + more sensitive.")
    ap.add_argument("--search-margin-hz", type=float, default=500.0,
                    help="narrow search half-window once the clock-freq bias is solved (Hz)")
    ap.add_argument("--search-margin-wide-hz", type=float, default=3000.0,
                    help="wider search half-window BEFORE the bias is solved (covers the unknown "
                         "TCXO offset; shrinks to --search-margin-hz once a few sats pin the clock)")
    ap.add_argument("--clock-bias-file", default=None,
                    help="persist the solved carrier clock-frequency bias (Hz, plain text) "
                         "across runs, and warm-start from it: the bias is then SOLVED from "
                         "cycle 1 (narrow margins, seeding enabled immediately). Audit rec D: "
                         "on the GPSDO this is a per-chain constant, not an estimate.")
    ap.add_argument("--state-read-dir", default=None,
                    help="directory of sibling receiver-state JSON to READ (--dr-clock-adopt). "
                         "Defaults to --state-file's directory. Separate because an adopting "
                         "chain publishes nothing: it contributes no estimate, so requiring it "
                         "to name a write path just to read one would be backwards.")
    ap.add_argument("--dr-clock-adopt", action="store_true",
                    help="ADOPT the receiver clock from a BAND SIBLING instead of holding the "
                         "primed --dr-clock-chips constant forever. Reads the sibling state "
                         "files receiver_state.StateWriter already publishes (same --state-dir "
                         "and --state-dongle scope as --clock-bias-siblings) and refreshes "
                         "dr_state['clk'] and its drift from the freshest one, every cycle.\n"
                         "\n"
                         "WHY: a chain with --detectors EMPTY cannot solve its own clock. The "
                         "bootstrap takes a median of measured code-phase residuals from "
                         "/get_detections, so with no detector `offs` is always empty, the EMA "
                         "never runs, and the primed constant stands for the WHOLE RUN -- which "
                         "the --dr-clock-chips help already says in as many words. That is "
                         "every multi-constellation chain (E5a/B2a have no blind acquisition at "
                         "all), so today the number is read out of the GPS broker's log by a "
                         "human and pasted in, and it is wrong after any F-engine restart.\n"
                         "\n"
                         "THIS IS AN ADOPTION, NOT A FUSION, and that is why it is safe to turn "
                         "on now while receiver_state's covariance-weighted fuser is still "
                         "deferred ('score, don't steer'). The adopting chain has NO estimate of "
                         "its own, so there is nothing to weight and no manufactured agreement "
                         "-- the failure mode that module's docstring warns about (siblings "
                         "reading each other, so their spread measures the fusion rather than "
                         "the estimator) cannot arise. A chain WITH detections ignores this "
                         "flag; it solves its own.\n"
                         "\n"
                         "SCOPE IS THE LO, exactly as receiver_state defines it: E5a and GPS L5 "
                         "share one carrier, one cable and one F-engine, so the clock transfers "
                         "EXACTLY. It does NOT transfer across a retune -- E5b at 1207 MHz has a "
                         "different PFB group delay -- which is why the dongle key gates it and "
                         "why this must never be pointed across bands.")
    ap.add_argument("--clock-bias-siblings", nargs="*", default=None,
                    help="BAND-SHARED bias (2026-07-22): the other chains' --clock-bias-file "
                         "paths on the SAME band. All chains of a band despread the SAME "
                         "carrier frequency through ONE LO, so the clock-freq bias is one "
                         "physical number measured independently per chain; a chain with 2 "
                         "tracked sats estimates it at +-30 Hz swing (measured: the L5-GPS "
                         "bias wandered -16..+44 Hz in 70 s while L1-GPS sat at -152 +-1 Hz "
                         "with 5 sats), and that swing feeds EVERY seed's predicted Doppler "
                         "-> fence re-pins -> the 30-45 s NCO kick cycle that decoheres the "
                         "chain. Fusing the siblings' persisted estimates (sat-count "
                         "weighted, <60 s fresh) gives each chain the band's full sat count "
                         "(~8-10) -- an identifiability fix, not a tuning knob. Chains in "
                         "drift-alarm stop persisting, so poisoned estimates go quiet "
                         "automatically.")
    ap.add_argument("--state-file", default=None,
                    help="S2 / Mechanism B OBSERVER: publish this chain's receiver-state "
                         "estimates as JSON (atomically replaced, ~1 Hz) for cross-chain "
                         "comparison. WRITE-ONLY -- exporting changes no estimate and no "
                         "seed. It exists because eight brokers estimate the same four "
                         "physical quantities independently, four of them measure the SAME "
                         "per-dongle clock error by different routes (search-Doppler median, "
                         "carrier-loop trim, l-a slope, DR drift), nothing has ever compared "
                         "carrier-side against code-side, and NOTHING anywhere carries a "
                         "variance -- so the covariance-weighted fuser this is a prerequisite "
                         "for cannot yet be written. Exports each chain's PRE-fusion value "
                         "and its scatter, because the persisted .hz files already read each "
                         "other and their agreement is therefore partly manufactured.")
    ap.add_argument("--decode-health-file", default=None,
                    help="publish this broker's NAV-DECODE health as JSON (atomically "
                         "replaced) for the viewer + the eventual decoded-eph BRDC fallback: "
                         "per (signal, PRN) whether SYNCED, whether a full ephemeris was "
                         "EXTRACTED, the BRDC-agreement residual dpos_m, and decode freshness. "
                         "WRITE-ONLY, sibling of --state-file; the viewer reads the directory "
                         "and merges. Off when unset.")
    ap.add_argument("--state-dongle", default=None,
                    help="fusion scope key for --state-file: what physically shares an LO "
                         "(one airspy per band, so this is the band tag, identical for every "
                         "chain on it). Do NOT fuse across dongles -- the per-band offsets "
                         "(-151/-15/+31 Hz) are frac-N synthesis constants, not a common "
                         "reference error, and averaging them is meaningless.")
    ap.add_argument("--state-consume", type=int, default=0,
                    help="S2d, RESCUE-ONLY (revised 2026-07-29): consume the dongle's fused "
                         "LO estimate EXACTLY when this chain has no estimate of its own "
                         "(cold start, below min-sats, warm-start file lost). The original "
                         "always-on scope was tried and REVERTED the same day -- car_trim "
                         "rose 30-36%% at matched node age, because the LO is a CONSTANT and "
                         "the chain's own EMA (minutes of time-averaging) beats one cycle "
                         "of cross-chain averaging; rescored against the EMA, fusion lost "
                         "7 of 8 chains. In the rescue case there is no EMA to lose to, and "
                         "the fused state's unique value is the cross-FAMILY rescue "
                         "(code->carrier) that --clock-bias-siblings structurally cannot "
                         "provide. With the chain solved, this flag is a PROVEN no-op. The "
                         "'untested rescue path' worry is answered by scoring it always "
                         "(the SHADOW log line) and exercising it deliberately "
                         "(diag/receiver_state_rescue_test.py + the isolated-broker "
                         "method), not by running it always.")
    ap.add_argument("--bias-stale-s", type=float, default=300.0,
                    help="STALE-BIAS RESCUE: if the solved bias EMA has gone this long without "
                         "a multi-sat measurement, widen the search margins and RE-SOLVE from "
                         "the next detections (snap to the fresh median, recalibrate, persist). "
                         "Closes the 2026-07-20 lockout: a GPSDO unlock walked the EMA -2 ppm, "
                         "every lock died, and the EMA latched mid-walk -- hints then sat kHz "
                         "off truth at narrow margins with nothing left to update them. The "
                         "held value still centers the (wide) hints and seeds, so a healthy "
                         "chain that merely has a sparse sky loses nothing. 0 disables.")
    ap.add_argument("--bias-min-sats", type=int, default=2,
                    help="detected sats needed before a cycle's median residual may update the "
                         "clock-freq bias (and hence NARROW the search). A single sat's residual "
                         "is unfalsifiable: one bad prediction (wrong TLE mapping, non-transmitting "
                         "sat cross-corr) gets swallowed as 'clock bias' and shifts every other "
                         "hint out of the narrow window -- a self-locking deadlock where no second "
                         "sat can ever acquire to correct it (2026-07-12: BDS-2 C14 froze the whole "
                         "B1C constellation at -1550 Hz).")
    ap.add_argument("--seed-bias-source", choices=("ema", "slow"), default="ema",
                    help="which clock-freq bias the SEED consumers ride (#105). 'ema' = the "
                         "hint EMA, the old shared-number behaviour. 'slow' = a long-memory "
                         "EMA (--seed-bias-alpha) of the same raw medians: search hints keep "
                         "the fast EMA (a wandering window centre is harmless against "
                         "hundreds-of-Hz margins), but seeds stop commanding the +-10 Hz "
                         "quantization wander into every replica's code rate (measured: the "
                         "wander integrates ~1 chip off-peak fleet-wide every ~5 min = the "
                         "#105 q-crash bursts; gal/bds were immune only because they never "
                         "solve a local bias).")
    ap.add_argument("--seed-bias-alpha", type=float, default=0.005,
                    help="EMA weight for the seed-side clock-freq bias under "
                         "--seed-bias-source=slow (per solve, ~10 s cadence: 0.005 => tau "
                         "~30 min -- above the minute-scale quantization wander, below "
                         "hour-scale GPSDO thermal drift).")
    ap.add_argument("--dr-cs-scan", action="store_true",
                    help="BRING-UP INSTRUMENT (2026-08-31, gal_e6): step every dead-reckon "
                         "birth cp by one primary period per pass (re-birthing, never "
                         "slewing) to scan the secondary-code phase hypothesis; the CS-SCAN "
                         "log line pairs each pass's k with the previous pass's best tracker "
                         "amp_snr. One wrap = lc_seg passes. A winner k = a measured "
                         "anchoring offset; no winner = the CS phase is exonerated. "
                         "NEVER leave armed: the chain cannot track while scanning.")
    ap.add_argument("--tle-name-filter", type=str, default=None,
                    help="regex on the TLE NAME; almanac keeps only matching sats. Encodes "
                         "signal capability the TLE group can't (e.g. 'BEIDOU-3' for B1C: "
                         "BDS-2 birds don't transmit it -- their predictions poison the clock "
                         "bias and their PRNs only manufacture cross-correlation locks).")
    ap.add_argument("--code-length", type=float, default=1023.0,
                    help="spreading-code length (chips) for cp0 unwrap/fit (L1 C/A = 1023)")
    ap.add_argument("--hops-per-sec", type=float, default=125000.0,
                    help="F-engine hops/s (Fs/fft_len) -- cp slope chips/s log + the code clock-bias estimate")
    ap.add_argument("--chip-rate-hz", type=float, default=1.023e6,
                    help="spreading chip rate (L1 C/A 1.023e6) -- for the code-rate clock-bias estimate")
    ap.add_argument("--code-doppler-sign", type=float, default=1.0,
                    help="must match the search stage's code_doppler_sign: the sign of the "
                         "Doppler-dependent cp0 back-reference the search applies. Used to "
                         "re-express cp0 history in the seed's Doppler currency before the "
                         "slope fit (cp_to_seed_currency).")
    ap.add_argument("--code-bias-alpha", type=float, default=0.05,
                    help="EMA weight for the receiver code-rate clock offset (l-a); slow -> tracks TCXO/OCXO drift")
    ap.add_argument("--code-bias-min-sats", type=int, default=2,
                    help="fitted sats needed before the pooled code-rate clock offset is trusted + seeded to weak sats")
    # ---- #91: THE BROWNOUT POLICY (default OFF; D1 supplies the trigger) -----------------
    # A band-wide carrier-coherence collapse takes PRESENCE down chain-wide (E3, 2026-08-25:
    # gal_e5a's KCOH sig_sum 43,255 -> 97 -> 14, every sat at once, ~9 min, while gal_e5b
    # held ~100k). Two things then make a 9-minute fade into a full per-sat re-pull:
    #   (b) every standing trim is RELEASED, and an unarmed trim leaks to erasure in ~5.6 s;
    #   (c) the (l-a) clock, re-fitted from the COLLAPSING population, swings (+-96 chips
    #       measured) and its garbage rate is adopted into every seed -- positive feedback
    #       that sustains the outage.
    ap.add_argument("--presence-admit-displaced", action="store_true",
                    help="THE E3 FIX, presence half: a row that FAILS the q bar is re-admitted "
                         "when the (E/P, L/P) decomposition says its q deficit is a CODE OFFSET "
                         "rather than weakness -- bright (same probe-anchored p bar), pedestal "
                         "<= --presence-disp-pedestal-max, |offset| <= --presence-disp-off-max. "
                         "q ~ 1 is degenerate (off-peak OR weak) and the bar reads it as weak, "
                         "so the displaced-but-strong satellite -- the one the DLL exists to "
                         "rescue, and whose offset SUPPRESSES q so the fault latches (E3's 12 "
                         "min outage; the bar shuts at ~0.17 chips) -- is exactly the one it "
                         "throws away. ⚠️ Only ever active under the probe-anchored q+p gate: "
                         "without probes the p bar is a peer competition and this would admit "
                         "peers' noise. Rows it admits carry present_gate = "
                         "'q+p:probes+disp' (bright) or 'q+deep:probes+disp' (seen only by "
                         "the offset-blind detector).")
    ap.add_argument("--presence-disp-pedestal-max", type=float, default=0.3,
                    help="admit-displaced: max fitted noise pedestal (relative to the peak). "
                         "Measured 2026-08-26: displaced-but-strong rows sit at 0.03-0.08; "
                         "genuinely weak rows and probes read >= ~1 (E=P=L fits pedestal inf).")
    ap.add_argument("--presence-disp-off-max", type=float, default=0.6,
                    help="admit-displaced: max |fitted offset| in chips. Inside the DLL's "
                         "pull-in range (the early tap leaves the correlation triangle at "
                         "1 - spacing/2 = 0.75 chips); past it the discriminator carries only "
                         "its sign and admission would arm a loop with no gradient to follow.")
    ap.add_argument("--presence-disp-deep-margin", type=float, default=3.0,
                    help="admit-displaced: the OFFSET-BLIND evidence bar, as a multiple of "
                         "the combiner's deep_floor. ⚠️ THE POINT OF THIS FLAG IS THAT THE "
                         "EVIDENCE MUST NOT BE AN ON-PEAK STATISTIC. The admission originally "
                         "required prompt power above the probe floor -- but prompt power is "
                         "suppressed by exactly the offset being rescued, so it is q's disease "
                         "one step along, and it was the THIRD time the same kind of statistic "
                         "was swapped for another of its kind. Measured on sky 2026-08-27: 7 "
                         "satellites detected at deep_snr 18-56 (C/N0 up to 26.8 dB-Hz) with "
                         "prompt at 0.5-2.7x noise; the prompt bar admitted 2 of them. "
                         "deep_snr RE-SEARCHES code phase, so it sees the satellite wherever "
                         "the tap sits, and deep_floor is the combiner's own rectification "
                         "level -- not a population of peers. Prompt power stays as an "
                         "ALTERNATIVE (either suffices); the pedestal and offset tests, not "
                         "this one, are what still refuse a centred noise realisation.")
    ap.add_argument("--fleet-trim-brownout-hold-s", type=float, default=0.0,
                    help="#91(b): during a D1 brownout, keep PRNs armed this long after last "
                         "presence AND freeze the C++ loop (gain=leak=0) for the duration. "
                         "0 = OFF (release as before). ⚠️ THE HOLD IS A FREEZE, NOT AN ARM: "
                         "armed alone would integrate a NOISE discriminator (the quality "
                         "gates are policy and live here, not in the C++ loop), and disarmed "
                         "erases the trim in ~5.6 s. Only zeroing BOTH gain and leak retains "
                         "the value. This flag doubles as the CAP -- past it the hold expires "
                         "and a genuinely set satellite is released normally.")
    ap.add_argument("--detector-transit-veto-deg", type=float, default=5.0,
                    help="suppress D2/D3 while ANY visible satellite (all constellations) is "
                         "within this many degrees of boresight. 0 = off. A near-boresight "
                         "satellite rails the 4+4b quantiser for every chain at once -- "
                         "measured 2026-08-26: inside 3 deg the tracked population drops from "
                         "7 per epoch to 4 and the survivors read 2-3 dB HIGH -- so satellites "
                         "vanish across all five chains together and a per-satellite detector "
                         "reads that as a constellation of individual faults. ON by default "
                         "because it is a SUPPRESSION (the conservative direction) and both "
                         "detectors are read-only; it costs recall on real events during a "
                         "transit and buys a base rate that is not mostly geometry.")
    # ---- LIVE PRN MEMBERSHIP (docs/CHORD_LIVE_PRN_RECONFIG.md, gnss_broker/prnmap.py) ----
    # The node's slot->PRN list was hand-written in the manifest and this broker reads live
    # BRDC; nothing reconciled them. Measured 2026-08-26: 5 dead Galileo slots while E36, which
    # transits at 83 deg, had no slot at all -- and, because the noise-probe selector picks the
    # deepest below-horizon PRN, it kept picking the one satellite the node could not represent,
    # dropping both gal chains to brightness-only presence.
    ap.add_argument("--clock-bias-cal-min-sats", type=int, default=5,
                    help="satellites a STALE-RESCUE re-solve needs before it may replace the "
                         "warm-start calibration. Below this the old calibration is HELD and "
                         "said so. The re-solve used to be adopted unconditionally: one median "
                         "from however few sats were present became the reference and was "
                         "announced as hardware news -- measured 2026-08-27, a 2-sat re-solve "
                         "moved it -2.3 -> -17.9 Hz against a population whose median is +0.0 "
                         "(sd 12.7 over 1222 samples), and every later comparison was then "
                         "against noise. An old reference is stale; a noise-derived one is "
                         "WRONG, and wrong outranks stale for something everything else is "
                         "measured against. 0 disables the guard (the old behaviour).")
    ap.add_argument("--clock-bias-alarm-every-s", type=float, default=600.0,
                    help="rate limit for CLOCK DRIFT ALARM. Was 60 s, which against a poisoned "
                         "calibration fired once a minute for hours and buried the log -- an "
                         "alarm repeating faster than it can be investigated is noise, and it "
                         "would have hidden a real one.")
    ap.add_argument("--probe-require-slot", action="store_true",
                    help="only pick noise probes the NODES ACTUALLY HOLD A SLOT FOR, by asking "
                         "them (/get_prns). The selector takes the deepest below-horizon PRNs "
                         "from the almanac and never checked the trackers could represent "
                         "them, so it kept choosing satellites that get seeded, logged as "
                         "seeded, and NEVER report. Measured 2026-08-27: BeiDou probed PRN 2 "
                         "against a 19-42 slot list, leaving 2 live probes -- below the >= 3 "
                         "the q+p presence gate needs -- so presence fell back to the PEER "
                         "COMPETITION and about HALF the population passed by construction "
                         "(q floor 4.72, above the q ~ 4 ceiling a real satellite can reach). "
                         "⚠️ Not the rejected discovery filter: this ASKS, so there is no "
                         "sidereal-day bootstrap. ⚠️ Fails OPEN -- no unanimous map yet, or a "
                         "split fleet, leaves the selector exactly as it was. ⚠️ Enables the "
                         "/get_prns sweep even with --prn-reconfig off, which is a NEW GET: "
                         "recorded transcripts predate it, so it must stay off by default or "
                         "replay diverges.")
    ap.add_argument("--prn-reconfig", choices=["off", "report", "apply"], default="off",
                    help="keep the nodes' slot->PRN map in step with live BRDC. off (default) "
                         "= this stage does nothing. report = GET each node's map, log the "
                         "swap it WOULD make, post nothing -- a live version of `gen_fleet.py "
                         "--check-prns`, with no risk. apply = actually POST /set_prns. "
                         "⚠️ A SWAP COSTS A FULL RE-ACQUISITION on that slot (the node resets "
                         "code table, Phi cache, carrier NCO, seed, trim, power averages and "
                         "element cal COLD -- correctly, since none of it describes the new "
                         "satellite), so every threshold below exists to make swaps RARE.")
    ap.add_argument("--prn-reconfig-poll-s", type=float, default=60.0,
                    help="how often to complete a full sweep of the nodes' live maps. Read "
                         "back rather than remembered, so what this diffs against is what the "
                         "nodes ACTUALLY hold -- including after a restart reverted them to "
                         "the config list. ⚠️ THE SWEEP IS ONE ENDPOINT PER CYCLE, not a burst: "
                         "a sweep of all 12 costs (n_dead x timeout) on a SINGLE cycle, which "
                         "is a full minute of broker stall the first time site work takes the "
                         "fleet down -- #81's failure exactly.")
    ap.add_argument("--prn-reconfig-interval-s", type=float, default=900.0,
                    help="minimum seconds between swaps, fleet-wide. ONE slot moves per "
                         "interval: re-acquisitions are the cost, and a mechanism that can "
                         "churn several at once is the #92 disease one level up.")
    ap.add_argument("--prn-reconfig-admit-deg", type=float, default=10.0,
                    help="a satellite with no slot must be ABOVE this to claim one. Not "
                         "'rises eventually': the swap is only worth paying for a satellite we "
                         "can use immediately, and one that is up now is up again tomorrow.")
    ap.add_argument("--prn-reconfig-evict-deg", type=float, default=0.0,
                    help="a slot's incumbent counts as DOWN below this elevation.")
    ap.add_argument("--prn-reconfig-down-hold-s", type=float, default=10800.0,
                    help="how long an incumbent must have been continuously down before its "
                         "slot may be reclaimed (default 3 h). Hysteresis, not caution for its "
                         "own sake: BRDC visibility flickers, and a slot that flickers is a "
                         "satellite that never locks.")
    ap.add_argument("--prn-reconfig-gone-hold-s", type=float, default=7200.0,
                    help="how long an incumbent must have been ABSENT FROM BRDC ENTIRELY "
                         "before its slot may be reclaimed (default 2 h). Shorter than the "
                         "down hold because a dead slot costs capacity and gains nothing -- "
                         "but not zero, because an ephemeris gap is not a decommissioning.")
    ap.add_argument("--prn-reconfig-heartbeat-s", type=float, default=900.0,
                    help="say ARMED AND HEALTHY this often, even with nothing to propose. "
                         "⚠️ NOT COSMETIC: every other branch of the stage speaks only when it "
                         "has a swap in mind, and having nothing to do is the NORMAL state -- "
                         "E36, the satellite this whole mechanism exists for, is below the "
                         "admit mask ~13 h a day. Without this, armed-and-idle is "
                         "indistinguishable from not-running.")
    ap.add_argument("--prn-reconfig-lead-s", type=float, default=5.0,
                    help="schedule a PRN swap this far ahead, as an ABSOLUTE F-engine HOP "
                         "(set_prns {\"at_hop\": N}), so every node crosses the discontinuity "
                         "on the SAME FRAME. Without it each node swaps on whatever frame it "
                         "happens to be building, and the combiner folds one window whose "
                         "instances disagree about which satellite slot p is -- an "
                         "accumulator-identity error that is invisible downstream because "
                         "every row is individually well-formed. 5 s is many frames (41.94 ms "
                         "each) of margin against POST scatter and well inside the "
                         "--prn-reconfig-interval-s cadence. 5 s, not the original 2: the map "
                         "goes to 12 endpoints in sequence and one unresponsive node costs a "
                         "whole --prn-reconfig-timeout-s, so the lead must cover the SCATTER "
                         "of the posts, not just their mean. The in-cycle delay between the "
                         "status poll and this post is handled separately, by advancing the "
                         "axis sample to post time -- do not spend the lead on it. "
                         "The lead is capped by the node: "
                         "a deadline it can never reach (an F-engine re-base moves the hop "
                         "BACKWARDS) is applied immediately with a warning rather than "
                         "wedging the slot forever. ⚠️ HOPS, NOT SAMPLES -- the first version "
                         "posted hops under a key the node read as samples (hop x 16384), so "
                         "every deadline was already past and every swap silently took the "
                         "apply-immediately degrade.")
    ap.add_argument("--prn-reconfig-axis-max-age-s", type=float, default=30.0,
                    help="refuse to schedule a swap when this cycle's F-engine axis sample is "
                         "older than this. The deadline is built by advancing the polled hop "
                         "to the instant of the POST, and an extrapolation over a long gap is "
                         "a guess dressed as a coordination point: past this age the swap "
                         "posts UNSCHEDULED and says so, rather than naming a frame the fleet "
                         "may already have passed.")
    ap.add_argument("--prn-reconfig-timeout-s", type=float, default=2.0,
                    help="per-endpoint HTTP timeout for the map GET/POST. With the one-per-"
                         "cycle sweep above this is the WHOLE cost a dead node can impose on "
                         "any one cycle.")
    ap.add_argument("--code-bias-brownout-hold", action="store_true",
                    help="#91(c): during a D1 brownout, HOLD the last (l-a) EMA instead of "
                         "re-fitting it from the collapsed population. --code-bias-min-sats "
                         "is an ABSOLUTE floor (2), which a chain that fell from 7 sats to 2 "
                         "still passes -- and that is exactly the population whose fit swung "
                         "the clock. The relative collapse is what D1 measures.")
    ap.add_argument("--code-bias-init", type=float, default=None,
                    help="warm-start the receiver code-rate clock offset (l-a) in PPM, e.g. from a prior "
                         "strong-signal (L1 C/A) run -- so a weak band (L1C) seeds on-peak from cycle 1 "
                         "instead of self-calibrating. Live samples still refine it if any sats fit.")
    ap.add_argument("--code-bias-file", type=str, default=None,
                    help="persist the converged (l-a) ppm here: read at startup (unless --code-bias-init "
                         "is set) and rewritten each update, so the offset carries across runs/bands")
    ap.add_argument("--fit-gap-s", type=float, default=16.0,
                    help="reset the cp-fit history across a detection gap longer than this "
                         "(seconds of capture time; converted via --hops-per-sec)")
    ap.add_argument("--carrier-gain", type=float, default=0.0,
                    help="SHARED carrier loop gain (0 = off): integrate the combiner's full-band "
                         "carrier_hz_resid into a per-PRN carrier_trim_hz commanded to every "
                         "subband tracker's NCO -- one loop at full-band SNR instead of N "
                         "noise-driven per-channel FLLs. Trackers need carrier_shared: true.")
    ap.add_argument("--carrier-max-hz", type=float, default=40.0,
                    help="clamp on the shared carrier trim (Hz)")
    ap.add_argument("--carrier-leak", type=float, default=0.05,
                    help="shared carrier integrator leak (same role as --dll-leak)")
    ap.add_argument("--carrier-min-sig", type=float, default=0.0,
                    help="HOLD the trim (skip the update) when the combiner's lock significance "
                         "is below this (0 = old behavior). The probe exemption above, "
                         "generalized: a FADED real satellite is the same pathology -- its "
                         "residual is noise, and integrating it at full gain random-walks the "
                         "trim, which drags the model phase off, which DEEPENS the fade. "
                         "Measured 2026-07-17 on the 1176 MHz chains: a ~4 s noise-driven "
                         "limit cycle (dip -> trim walk 1-9 Hz -> decoherence -> dip), median "
                         "certified stretch 4 s, every dip an ADR phase break of ~8 cycles -- "
                         "the gf-TEC floor. A held trim coasts on the almanac Doppler-rate "
                         "feed-forward, which carries the true dynamics through the fade. "
                         "Updates additionally require a CERTIFIED coherent window "
                         "(coherence_s > 0): a residual measured on a decohered window is "
                         "garbage at ANY amplitude (the sig gate alone let re-lock transition "
                         "rows kick converged trims to the +-100 Hz rails -- E36 at 48 dB-Hz).")
    ap.add_argument("--carrier-max-step", type=float, default=0.0,
                    help="slew clamp on the trim, Hz per update (0 = unclamped). A healthy "
                         "converged loop corrects 0.02-0.2 Hz per update and the clock it "
                         "tracks is GPSDO-smooth, so any large requested step is a bad "
                         "measurement by construction; clamping bounds the damage a single "
                         "garbage residual can do to less than a deep window can absorb. "
                         "Convergence from a fleet-seeded start needs only a few Hz total.")
    ap.add_argument("--carrier-step-accept", type=int, default=0,
                    help="EXPLAIN-APPLY-VERIFY hypothesis stage (0 = off): M = this many "
                         "consecutive fresh GATED residuals that agree (spread < max(2 Hz, "
                         "innov)). For a PRESENT-but-gated sat, when the agreed median is "
                         "also large enough to EXPLAIN the decoherence (>= ~1/(2*T_emit) = "
                         "0.5 Hz), the observables close on one story -- 'the NCO is off by "
                         "med' -- and the FULL correction is applied ONCE, entering a "
                         "VERIFY window: coherence returns / residual collapses within 3 "
                         "emits, or the hypothesis is REVERTED, the sat escalated to a "
                         "BOOTSTRAP re-acquire, and hypotheses locked out 60 s. The "
                         "coherent-state innovation gate is untouched (it is a physics "
                         "bound: a cohering sat cannot carry a multi-Hz residual). The "
                         "closed verify loop is what the two retracted open-loop escapes "
                         "lacked: a wrong correction costs one bounded, reverted step. "
                         "Type specimen: C19 2026-07-22, parked at +3.03 Hz / full amp / "
                         "dark for minutes while every gate held.")
    ap.add_argument("--carrier-innov-hz", type=float, default=0.0,
                    help="TRACK-mode innovation gate (0 = off): REJECT any residual larger "
                         "than this outright. After feed-forward, a converged sat's true "
                         "residual is sub-Hz (the trim tracks a slowly-drifting almanac-"
                         "Doppler error); the resid estimator nevertheless emits tens-of-Hz "
                         "values that pass certification (measured 07-17: 'certified' +40 Hz "
                         "-- impossible for a genuinely coherent window). A slew clamp only "
                         "slows the poisoning (E36 at 49 dB-Hz walked to 18 Hz off the fleet "
                         "at 1 Hz/update and collapsed 20 dB); rejection stops it. Real step "
                         "changes re-enter via re-seed -> BOOTSTRAP.")
    # (ALIAS ESCAPE v1/v2 DELETED, 07-19 audit A4. v1 killed the fleet in 15 min
    #  (8208dba6/069e8770); v2 shipped gated-off and never armed. Its two jobs are owned
    #  by surviving mechanisms: a stale/aliased f_ref offset is snapped by the TIGHT
    #  tracker fence (fll_reacq_hz ~15 Hz, free under --dop-continuous), and a walked/
    #  aliased TRIM latch is the watchdog's lifecycle rescue below.)
    ap.add_argument("--carrier-bleed", type=int, default=0,
                    help="ARM the trim-bleed (0 = shadow-only). On a verified candidate: zero "
                         "car_trim and flag the tracker to re-adopt the seed (f_ref = dop, phase-"
                         "continuous reanchored==2) -- folding the frozen sub-fence pin offset into "
                         "f_ref so the despread runs on-true. EXPLAIN-APPLY-VERIFY: heal (stay "
                         "coherent) or it is logged REFUTED and the loop re-grows the trim from 0. "
                         "Default OFF -- validate on the replay bench (GNSS_TRIM_FORCE) first.")
    ap.add_argument("--watchdog-s", type=float, default=0.0,
                    help="TRACK WATCHDOG (0 = off): a sat with a fresh detection at "
                         ">= --watchdog-det-snr that has ZERO coherent emits for this many "
                         "seconds (and has been seeded at least that long) is dropped from "
                         "seeds entirely -- the full re-seed lifecycle (fresh dop blend, "
                         "fleet trim prior, tracker state reset via the active[] gap) is "
                         "the only rescue that fixes every cause (aliased NCO, walked "
                         "trim, poisoned anchor) without guessing which one it is. The "
                         "2026-07-18 targeted-correction attempts (trim-step v1/v2) both "
                         "guessed and both lost; the lifecycle rescue never did.")
    ap.add_argument("--watchdog-det-snr", type=float, default=50.0,
                    help="watchdog presence bar: only judge sats the search currently "
                         "sees at this significance -- a sat this strong that cannot "
                         "cohere is broken by definition; weak sats legitimately take "
                         "minutes and must never be churned by the watchdog.")
    ap.add_argument("--carrier-det-gate-s", type=float, default=0.0,
                    help="BOOTSTRAP walk gate (0 = off): in BOOTSTRAP mode, integrate a "
                         "residual only if a fresh detection exists within this many "
                         "seconds. A never-detected (almanac-only) or long-undetected seed "
                         "has no signal for the estimator: its 'residual' is noise, and "
                         "integrating it random-walks the trim to the clamp (C40 walked to "
                         "-42 Hz over the 07-18 evening; the E36 innovation gate protects "
                         "only TRACK mode). Held trims coast on the fleet prior + Doppler-"
                         "rate feed-forward, which is the better model anyway.")
    ap.add_argument("--carrier-fleet-seed", action="store_true",
                    help="initialize a new (or re-seeded) sat's trim to the MEDIAN of the "
                         "converged fleet trims instead of 0. The converged trim is the "
                         "chain's deterministic frac-N LO offset (same for every sat, stable "
                         "across restarts -- e.g. the L5 chain sits ~+30 Hz), so the fleet "
                         "median is the right prior; the carrier twin of the code-bias "
                         "seeding above (strong sats calibrate the clock so weak ones start "
                         "on it).")
    ap.add_argument("--force-doppler-rate", type=float, default=None,
                    help="REPLAY BENCH ONLY: attach this doppler_rate_hz_s to every seed (a "
                         "recorded capture's sky is at another epoch, so no almanac rate) to "
                         "exercise the tracker's NCO Doppler-rate feed-forward offline.")
    ap.add_argument("--dll-gain", type=float, default=0.25,
                    help="code delay-lock-loop gain (0 = off): each poll, nudge a persistent "
                         "per-PRN cp TRIM by gain * tau_est from the combiner's E/L discriminator. "
                         "The trim rides on top of the search-fit cp, converging to the fit's "
                         "grid-quantization bias -- sub-chip code tracking with no per-record "
                         "decisions (R1, docs/gnss_architecture_audit.md).")
    ap.add_argument("--dll-leak-present", type=float, default=0.05,
                    help="DLL integrator leak on the FLEET path, where `present` has already "
                         "confirmed signal (--dll-combiners). The ordinary --dll-leak caps the "
                         "reachable correction at (gain/leak)*0.25 = 1.25 chips, which is below "
                         "the residuals actually seen and leaves the loop pushing at a railed "
                         "discriminator without arriving. 0.01 lifts that ceiling to 6.25 chips "
                         "-- past the 3.27-chip tracker grating-lobe spacing -- while still "
                         "mean-reverting over ~100 updates. The leak was a stand-in for a signal "
                         "test; the fleet path now has a real one, so it does not need to be "
                         "this strong. The single-combiner fallback keeps --dll-leak.\n"
                         "MEASURED 2026-08-04 AND REVERTED: 0.01 made locks strictly WORSE -- "
                         "best-q median 2.05 -> 1.03 and samples above q 2.0 went 4/8 -> 0/8, "
                         "every satellite driven to noise. The leak is not only a ceiling: when "
                         "the discriminator is RAILED (|disc| ~ 0.9, which is most of the time "
                         "here) tau saturates at 0.25 chips and carries only the SIGN, not the "
                         "distance -- so the integrator pushes one way indefinitely and the leak "
                         "is what bounds the excursion. Default restored to 0.05; the knob is "
                         "kept so the experiment is repeatable, not because it should be moved.")
    ap.add_argument("--dll-leak", type=float, default=0.05,
                    help="DLL integrator leak (0 = pure integrator): trim mean-reverts each "
                         "update so discriminator NOISE can't random-walk it to the clamp. DC "
                         "loop gain = dll_gain/dll_leak; ~1/leak windows of smoothing.")
    ap.add_argument("--nh-hint", action="store_true",
                    help="hint the search's secondary-code alignment from the EPHEMERIS, so the "
                         "acquire scans nh_hint_span alignments instead of all 20 (~92%% of a "
                         "pass). nh at transmit is round((gpst - range/c + clk_sv)/period) mod "
                         "overlay_len -- the same convention --nh-assist uses for the combiner "
                         "-- leaving ONE global constant, the receiver clock reference, which "
                         "any detection measures and every satellite shares. Works for sats "
                         "NEVER detected, which is where a full 20-way scan hurts most. "
                         "Requires --almanac and the capture time anchor.")
    ap.add_argument("--nh-hint-span", type=int, default=2,
                    help="alignments scanned either side of the prediction. Measured 2026-08-04: "
                         "the offset is 16 with a +-2 spread, so span 2 (5 of 20 = 4x) covered "
                         "100%% of samples while span 1 (6.7x) covered 85%%. The stage keeps its "
                         "own nh_hint_span; this only sizes the LOG. The +-2 spread on a "
                         "deterministic quantity is an OPEN anomaly, not an accepted tolerance.")
    ap.add_argument("--joint-p2c-rotate", action="store_true",
                    help="P2c, SELF-DRIVING. Withhold the best-established satellite from the "
                         "joint solve, coast it for --joint-p2c-hold-s, log the "
                         "residual-vs-age curve, release, and rotate to one not tested "
                         "recently. Supersedes hand-picking a PRN with --joint-mask-prn, "
                         "which is a sample of ONE satellite at ONE geometry: on a transit "
                         "instrument the first two runs disagreed outright (flat -0.19+-0.44 "
                         "chips over 848 s, then -2.70 over 900 s on the same satellite) and "
                         "a third produced nothing because the named PRN had set. What this "
                         "accumulates instead is a DISTRIBUTION over satellites and "
                         "elevations -- and the residual's growth with age IS the b_sat "
                         "random walk, so it MEASURES q_b (currently the guessed 0.013).")
    ap.add_argument("--dr-max-solve-mad-chips", type=float, default=100.0,
                    help="refuse the receiver-clock solve when the per-satellite offsets "
                         "scatter by more than this (MAD, chips). A circular median over "
                         "--dr-min-sats satellites is a MEASUREMENT only if its inputs "
                         "agree; on a starved fleet the detections are noise, scatter "
                         "~uniformly over the code, and their median is an arbitrary number "
                         "that used to be snapped straight into the clock and shipped as "
                         "every seed. Not a tuning knob -- real offsets cluster inside ~+-10 "
                         "chips against a uniform-noise MAD of ~2557 on a 10230-chip code. "
                         "Refusing leaves the clock UNSET, which is strictly better than "
                         "wrong: a wrong clock also acquires nothing, and additionally "
                         "poisons the search hints that would have recovered it (docs 11.33).")
    ap.add_argument("--spectrum-archive", default="",
                    help="write the RAW per-channel spectrum points to this JSONL (task "
                         "#25). fleet_spectrum already returns (freq_id, amplitude, energy, "
                         "instance) per PRN and today every point is collapsed to one tau "
                         "and discarded; this persists them BEFORE the collapse. One line "
                         "per (prn, channel, instance) per poll -- a few kB/poll, no "
                         "recomputation, no node-side change. Empty = off. THE PER-SUBBAND "
                         "AXIS IS THE SCIENCE PRODUCT: a combined number can always be "
                         "rebuilt from the parts, the parts can never be recovered from a "
                         "combined number, so the archive stores parts and every combine "
                         "(beam map, band health, per-element gain) is an offline function "
                         "of them. %%Y/%%m/%%d are expanded, so a long run rolls by day. "
                         "(The %% are DOUBLED because argparse runs help through percent-format: "
                         "a bare %%Y here makes --help itself crash with 'unsupported format "
                         "character', which is how this file's --help was dead from the #25 "
                         "commit until 2026-08-11 -- and --help is how anyone finds a flag.)")
    ap.add_argument("--publish-port", type=int, default=0,
                    help="serve the FLEET-MERGED per-PRN state on this port (0 = off): "
                         "GET /get_status returns rows in GnssCoherentCombiner's schema, so "
                         "the viewer consumes them unchanged, but built from ALL "
                         "--dll-combiners instead of one instance's 6.7%% of the L5 lobe. The "
                         "browser cannot merge 14 origins across 8 hosts itself, and the "
                         "broker is already the shared-knowledge node (pooled l-a, clock-freq "
                         "bias, fused LO, cross-band assist) -- this is the same kind of "
                         "object. Coherent statistics are best-of-instance, not merged, and "
                         "say so via coh_src; see FleetPublisher.")
    ap.add_argument("--carrier-from-code", action="store_true",
                    help="SHADOW: log the carrier error DERIVED from the fitted code slope "
                         "(x f_carrier/f_chip) beside the measured carrier_hz_resid, without "
                         "applying it. carrier_hz_resid is signal-free at CHORD SNR, so the "
                         "carrier loop was integrating noise; the code side is strong. Compare "
                         "the two before letting the derived value drive anything.")
    ap.add_argument("--spectrum-aligned", action="store_true",
                    help="TASK #53: gather /get_spectrum by WINDOW INDEX instead of taking "
                         "whatever each instance has accumulated since its own last poll. "
                         "Every instance is asked for the same window, each reply is rotated "
                         "onto one phase reference (phi0), PRNs whose accumulator re-anchored "
                         "mid-window are dropped, and the 'same index == same samples' "
                         "invariant is asserted across instances rather than assumed. "
                         "Instances on a pre-#53 node config cannot address windows and are "
                         "EXCLUDED, not mixed in -- one unaligned member reintroduces exactly "
                         "the free per-instance phase this exists to remove (#52). Off by "
                         "default only because the aligned path issues two GETs per instance "
                         "where the legacy path issues one, and replay is strict-ordered: an "
                         "old transcript replayed with it on would diverge.")
    ap.add_argument("--spectrum-stale-margin", type=int, default=0,
                    help="#84 THE SPEC-WINDOW PIN: exclude instances whose newest available "
                         "window trails the fleet's newest by more than this many windows "
                         "BEFORE choosing the common index (which is min over the rest). "
                         "0 = off (the replay gate's requirement): plain min(hi), which "
                         "hands the index to the slowest member forever -- measured "
                         "2026-08-17 as SPEC-WINDOW pinned at 605351 on all five chains by "
                         "cx19's bench-state n2assemble (frozen buffer, bit-identical "
                         "amp/energy every poll), 11/12 healthy instances dropped, and "
                         "spec_tau a noise fit over the one dead instance. Healthy "
                         "inter-instance lag is a few records (#46); ~8 is a generous "
                         "margin. Excluded instances are named in the dropped list.")
    ap.add_argument("--spectrum-endpoints", default="",
                    help="FLEET PHASE-SLOPE DELAY FIT (task #32, docs/CHORD_JOINT_TRACKING.md "
                         "P1): comma-separated GnssGpuRecordAssemble endpoints ({a..b} ranges "
                         "expanded) whose /get_spectrum -- the per-channel prompt sums BEFORE "
                         "the cross-channel combine -- feed the joint (tau, phi_i) fit. A "
                         "delay is a phase ramp across frequency; the fleet's UNION of combs "
                         "is what suppresses the 3.27-chip grating lobes a single instance's "
                         "stride-16 comb cannot resolve. MEASUREMENT ONLY today: logged as "
                         "SPEC-FIT and published as spec_tau_chips/spec_peak_ratio, feeding "
                         "no loop. Empty = off, and every recorded transcript replays "
                         "byte-identically (replay is strict-ordered; this flag not being in "
                         "an old transcript's argv is what keeps the new GETs out of replay).")
    ap.add_argument("--joint-shadow", action="store_true",
                    help="P2a (task #33, docs/CHORD_JOINT_TRACKING.md section 3a): run the "
                         "JOINT receiver-state solve -- one [clk, clk_rate, b_sat[i]] "
                         "estimated together from the dead-reckon integrity residuals, "
                         "receiver-scope so all three constellations feed ONE clock -- and "
                         "LOG it beside the circular-median clock it is meant to replace. "
                         "Consumed by nothing: this flag cannot change a seed, a POST or a "
                         "transcript digest, by construction. The comparison it exists to "
                         "make: the median treats the +-3-7 chip per-sat spread (docs "
                         "11.22) as error to be gated on; the joint solve reads the same "
                         "numbers as clock PLUS per-sat bias. If the biases hold steady "
                         "while the clock stays smooth, the revised P2 is right and the "
                         "consumers can start switching one per commit (P2b).")
    ap.add_argument("--joint-feed-max-trim", type=float, default=1.0,
                    help="only feed the joint state from satellites whose APPLIED DLL trim is "
                         "inside this many chips, and whose fleet q is at least --lock-q "
                         "(0 = no gate, the pre-2026-08-22 behaviour). The model-primary "
                         "measurement y = held + trim - cp_predicted is sky-anchored ONLY "
                         "while the DLL can hold the tap on the peak: move the seed and the "
                         "trim moves oppositely, leaving y invariant. Past ~1 chip the "
                         "correlation triangle has no gradient, the trim stops compensating, "
                         "and y degenerates into a readback of the consumer's own output. "
                         "Measured 2026-08-22: median |model - held| 1.8-2.5 chips, so most "
                         "satellites were ALREADY outside that range when the feed was armed "
                         "ungated -- which is why it diverged rather than converged.")
    ap.add_argument("--lock-q", type=float, default=2.2,
                    help="treat a satellite as LOCKED for the dead-reckon slew/re-birth "
                         "decision when its fleet discriminator q is at least this "
                         "(0 = ignore q, the pre-2026-08-22 behaviour). q is the metric this "
                         "project judges lock on everywhere else -- the viewer's own note "
                         "says 'judge lock HERE, not on sig/C-N0' with 2.2 as the working "
                         "bar -- but the seeder's lock test used amp_snr/deep_snr and a "
                         "prompt-vs-fleet-median ratio instead. Measured 2026-08-22: both of "
                         "those read below their bars on tracking satellites (sig_of 0.00, "
                         "hold_prev 0.76-1.61 against 3.0), so 24%% of re-births hit sats at "
                         "q >= 2.2 and five stepped the seed 1.3-4.5 chips -- outside the "
                         "+-1 chip correlation triangle, where the DLL has no gradient left "
                         "to recover with. That is the bimodal q collapse.")
    ap.add_argument("--dr-clock-wait-s", type=float, default=30.0,
                    help="withhold dead-reckon seeding while the receiver clock is still the "
                         "--dr-clock-chips PRIME rather than a measurement (0 = never wait). "
                         "A seed born without a clock steps by the WHOLE clock at its first "
                         "re-birth after BOOTSTRAP -- measured 2026-08-22: 40 events, all "
                         "inside 57 s of broker start, step/off = 0.9958, the entire fleet "
                         "moving together to 0.08%%. The wait is short (gps_l5 bootstraps ~1 s "
                         "in, the model-primary chains adopt cross-band a few seconds later). "
                         "⚠️ IT IS A TIMEOUT, NOT A PRECONDITION: a DETECTOR-LESS chain never "
                         "solves a clock at all -- that is what the prime is for -- so after "
                         "this many seconds we seed on the prime and log that we did. A guard "
                         "that can leave a chain dark forever is worse than the step it "
                         "removes.")
    ap.add_argument("--dcb-bias", action="store_true",
                    help="use the MGEX (CAS) measured differential code biases in place of "
                         "the broadcast TGD/BGD, per satellite, where the product covers the "
                         "signal. THE POINT IS BeiDou B2a, whose broadcast term (TGD_B2ap) is "
                         "a B-CNAV2 parameter that RINEX 3 does not carry at all -- so without "
                         "this it gets NO group-delay correction. ⚠️ The product is ZERO-MEAN "
                         "PER CONSTELLATION by construction, so it supplies the per-satellite "
                         "spread and can never explain a constellation-common offset; measured "
                         "2026-08-23 the BeiDou spread is -0.006..+0.121 chips against a "
                         "+0.74..+1.3 chip common trim. Needs an Earthdata token "
                         "($EARTHDATA_TOKEN or ~/.cache/kotekan_gps/.earthdata_token); without "
                         "one it logs and falls back to the broadcast term.")
    ap.add_argument("--joint-feed-spec", action="store_true",
                    help="task #85: anchor the model-primary joint feed on the SPECTRUM FIT. "
                         "spec_tau is sky-minus-replica measured by the fleet slope fit, and "
                         "the replica IS held+applied_trim -- so y = held + trim + spec_tau "
                         "- cp_pred equals (sky - model) IDENTICALLY, at any displacement "
                         "the fit can see. That is what breaks the mirror: a consumed clock "
                         "moving the seed moves trim+spec_tau oppositely (the sky does not "
                         "move), leaving y invariant -- so a chain may feed AND consume. "
                         "With a fresh, significant spec fit the DLL-linear-range trim gate "
                         "is bypassed (spec_tau measures past it); without one the satellite "
                         "falls back to the trim-gated path unchanged.")
    ap.add_argument("--joint-consume", default="",
                    help="P2b: comma-separated JOINT-state consumers to switch LIVE, one "
                         "name per commit so each is A/B-able on its own. Empty (default) "
                         "= pure shadow. Names: 'rate' (seeded code-rate clock from "
                         "clk_rate instead of the l-a EMA -- which measured ~80x biased, "
                         "see the call site); 'slew' (the dead-reckon SLEW TARGET's "
                         "clock+bias offset from clk + b_sat instead of the clock EMA plus "
                         "the SatBiasFilter -- the half of the state P2c validates, and the "
                         "half that survived the 2026-08-10 gauge collapse); later 'clock'. "
                         "The shadow comparison for each is logged whether or not it is "
                         "consumed, so the A/B exists before the switch. Implies "
                         "--joint-shadow.")
    ap.add_argument("--joint-mask-prn", default="",
                    help="P2c, THE DISCRIMINATING TEST FOR THE ARCHITECTURE: comma-separated "
                         "PRNs whose measurements are WITHHELD from the joint solve while "
                         "their residual is still logged. Vector tracking's claim is that a "
                         "satellite rides the SHARED receiver state -- so a masked sat must "
                         "COAST on clk + clk_rate*dt with its b_sat frozen, and its "
                         "predicted-minus-actual must stay flat. If it drifts instead, the "
                         "state is not carrying it and the per-sat loops are doing the work "
                         "the joint solve claims to have replaced. Measurement-only: the "
                         "masked sat is still SEEDED normally, so nothing on sky changes. "
                         "Mask a STRONG sat -- masking a weak one tests nothing, since its "
                         "own measurements were not holding it up anyway.")
    ap.add_argument("--rrate-state", action="store_true",
                    help="P3 step 2 (task #33): FEED the receiver-wide joint state's "
                         "per-satellite ORBITAL range-rate rows (m/s, band-shared) from "
                         "this chain's deep_rate_hz residuals -- the validated carrier "
                         "measurement (q-gated, continuity-gated, slew-gated by "
                         "rate_residuals; NEVER carrier_hz_resid, which is documented "
                         "signal-free at CHORD SNR). Measurement-only: cannot change a "
                         "seed or a POST. This is the shadow phase, and the JRR log line "
                         "is where sky validation happens. Per-CHAIN key on purpose: "
                         "E5a+E5b feeding lands both bands on ONE row per Galileo sat, "
                         "which is the cross-band combination the state exists for.")
    ap.add_argument("--rrate-command", action="store_true",
                    help="P3 step 3 (task #33): the seed's carrier command comes from the "
                         "joint state's carrier_correction_hz() -- receiver-wide f_carrier "
                         "plus this sat's orbital rrate, scaled to THIS band -- posted as "
                         "carrier_trim_hz, REPLACING the trim-loop value rather than "
                         "adding to it. One command per (satellite, band) is the point: "
                         "#52's root was two controllers on one state. Turning this on is "
                         "what closes the carrier loop. Implies --rrate-state. Per-CHAIN "
                         "key so half the fleet can run each arm and be compared IN ONE "
                         "POLL (the sky churns faster than a restart A/B can resolve). "
                         "Posts via carrier_trim_hz (NCO derotation, phase-continuous in "
                         "the tracker) and never touches doppler_hz: seed continuity "
                         "beats freshness, measured (deep_snr 221->17 on a 10 s re-pin).")
    ap.add_argument("--rrate-cmd-slew-hz", type=float, default=0.5,
                    help="per-POLL bound on how far the posted carrier command may move "
                         "(Hz). THE STABILITY TERM, not a nicety: the measured residual "
                         "reflects the command posted 1-2 polls earlier, while the feed "
                         "adds back the LATEST command -- so during a transient the "
                         "reconstruction is off by (cmd_now - cmd_then), which is "
                         "self-reinforcing and walked arm 3 at full slew. Bounding the "
                         "step bounds that misreference at ~slew*lag and it DECAYS: at "
                         "the fixed point the command stops moving and the reference is "
                         "exact. 0.5/poll = full 5 Hz pull-in in ~20 s. Railed-step "
                         "count is in the JRR-CMD line -- a rail that never clears means "
                         "the target is out of reach, not converging (the dr slew cap's "
                         "lesson).")
    ap.add_argument("--rrate-feed-applied", type=int, default=1,
                    help="ARM-13 (the E25 lesson, 2026-08-20): how the rrate feeds "
                         "reference the standing carrier command. 1 (default) = add the "
                         "posted command back to the measured residual -- CORRECT for the "
                         "command-AWARE plant (pre-08-14 trackers, whose records carried "
                         "the actuation, so measured = remaining and y = remaining + "
                         "applied). 0 = the command-BLIND plant: since the 08-14 tracker "
                         "cycle the assembler folds the applied ctrim out of every record "
                         "(e2e [4e] MEASURED commanded == control), so the fold/record "
                         "observables already read the FULL standing residual and the "
                         "add-back DOUBLE-COUNTS: the row integrates its own actuation "
                         "and every sat with a real residual ramps at exactly slew rate "
                         "(E25: sig 12776, resid pinned +5..+15 while its command walked "
                         "1 -> 10). Set 0 on any chain whose trackers run the folded "
                         "assembler -- i.e., every CHORD chain today. The fine PHASE feed "
                         "is untouched either way: its reference handles the command "
                         "explicitly (command-HELD gate + measured applied trim).")
    ap.add_argument("--rrate-cmd-min-sig", type=float, default=0.0,
                    help="ARM-12 GUARD (the E4 lesson, 2026-08-20): a row may COMMAND only "
                         "when its satellite's own kcoh detection significance meets this "
                         "bar this cycle. The sigma gate cannot catch a CONFIDENT estimate "
                         "of noise -- E4 (kcoh sig 10-36, raw rate swinging +-13 Hz) was "
                         "served sigma 0.03-0.30, passed rrate-cmd-max-sigma, and chased "
                         "its own actuation +2 -> +15.5 Hz in 12 min at exactly slew rate, "
                         "while every sig-thousands sat held +-1 Hz commands. Sigma "
                         "describes the FIT; this bar demands the SIGNAL. ~100 cleanly "
                         "separates the two populations. 0 = off (default). A sat that "
                         "loses this bar (or any command gate) now SLEWS its standing "
                         "command back to 0 instead of stepping -- see the release-slew "
                         "below the command site.")
    ap.add_argument("--rrate-phase-feed", action="store_true",
                    help="#33 PLL fine stage: feed the joint state's rrate rows from the "
                         "ADR's residual half (res_cycles per adr_records -- ~5 mHz over "
                         "a 20 s span) on sats that pass the fine gates: coarse loop "
                         "converged (|coarse resid| < 0.3 Hz), command HELD over the span "
                         "(the fine value's reference is only exact under a constant "
                         "command), same arc, counter advanced. The coarse deep_rate_full "
                         "feed stays as acquisition. SIGN IS CALIBRATED, NOT ASSUMED: "
                         "requires --rrate-phase-sign nonzero, which is set from the "
                         "JRRP shadow line's live fine-vs-coarse comparison -- closing a "
                         "loop on an uncalibrated observable is how this day started.")
    ap.add_argument("--rrate-phase-sign", type=float, default=0.0,
                    help="sign (+1/-1) mapping the combiner's internal (r2c-flipped) "
                         "res_cycles rate onto the deep_rate_full_hz convention. 0 = "
                         "uncalibrated: the JRRP shadow comparison still logs, the fine "
                         "feed stays off regardless of --rrate-phase-feed.")
    ap.add_argument("--rrate-phase-span-s", type=float, default=0.0,
                    help="GAP-1 long baseline (2026-08-25): difference res_cycles against "
                         "the newest snapshot at least this many seconds old, instead of "
                         "last poll's. 0 = per-poll (legacy). MEASURED (gap1_tau_scaling.py "
                         "08-24, two instances): per-sat sigma_rate 1.12 Hz at 2 s -> 0.31 "
                         "at 16 s -> 0.18 at 32 s, converging 1/tau -- the error "
                         "telescopes, so the long difference realizes what per-poll "
                         "feeding cannot (the filter cannot know consecutive fine values "
                         "share endpoints, so N per-poll feeds buy only 1/sqrt(N); this "
                         "buys 1/span from the same data). The span-mean lags a drifting "
                         "rate by span/2; that staleness is priced into sigma (drift class "
                         "0.02 Hz/s), NOT into the measurement epoch -- the filter only "
                         "predicts forward. 16-32 is the measured sweet spot (~0.35 Hz "
                         "effective vs the 0.5 quietness bar); 60+ hands the gain back to "
                         "staleness. Applies to the SHADOW (JRRP) and, where "
                         "--rrate-phase-feed is armed, the feed -- same computed value.")
    ap.add_argument("--rf-stats-endpoints", default="",
                    help="Comma/brace list of VOLTAGE TAP endpoints to poll for RF-path "
                         "health (#8), e.g. http://cx19:12048/gnss{0..1}_srch_tap. Empty = "
                         "OFF. Serves clip fraction and per-band power at the publisher's "
                         "get_rf.\n"
                         "\n"
                         "⚠️ ARM THIS ON ONE CHAIN ONLY. The voltage tap is per GPU and "
                         "serves every signal on it, so the RF path is a property of the "
                         "RECEIVER, not of a chain -- arming it in `common:` would poll the "
                         "same twelve endpoints five times a cycle for five identical "
                         "answers, and print the alarm five times.\n"
                         "\n"
                         "The node side is default-off too (config band_power_chans): an "
                         "unarmed instance reports enabled=false, which is recorded as OFF "
                         "and never published as zeros -- a dark panel must read 'not armed', "
                         "not 'no RFI'.")
    ap.add_argument("--rfi-stats", action="store_true",
                    help="Also poll each rf-stats instance's spectral-kurtosis RFI metrics "
                         "(rfi_sk_metrics/sk_metrics_{gpu}) and fold an SK summary into get_rf. "
                         "Default off; rides the SAME endpoints as --rf-stats-endpoints (the SK "
                         "url is derived per instance), so no second endpoint list.")
    ap.add_argument("--rf-bands", default="",
                    help="Comma list of CHAIN NAMES whose RF bands the receiver actually "
                         "flies, e.g. gps_l5,gal_e5b,gps_l2c,bds_b3i,gal_e6. Names the rows "
                         "of the Stream health panel: each monitored channel is labelled with "
                         "the nearest DECLARED carrier, and the carriers come from "
                         "gnssSignal.hpp so no frequency is typed twice.\n"
                         "\n"
                         "⚠️ WITHOUT THIS THE PANEL FALLS BACK TO UNNAMED 'lobe N' ROWS, ON "
                         "PURPOSE. The obvious shortcut -- label each channel with whatever "
                         "carrier in the band table is nearest -- INVENTS BANDS: measured "
                         "2026-09-02, cx19/gnss1's channels 287-288 (1199.2/1202.3 MHz) came "
                         "back as GLONASS 'L3', which CHORD does not fly, because E5b's lower "
                         "shoulder is nearer L3OC's carrier than E5b's own. Declaring the set "
                         "removes the ambiguity instead of hiding it.\n"
                         "\n"
                         "⚠️ Rides the SAME arming as --rf-stats-endpoints -- one chain only. "
                         "The list is the FLEET's bands, not this chain's: the tap is per GPU "
                         "and monitors the union of every chain on it.")
    ap.add_argument("--drop-stats", action="store_true",
                    help="Also poll each rf-stats instance's node /metrics for drop counters "
                         "(gnss{gpu} srch/telem buffer-send drops, node dpdk rx-missed and "
                         "ring-full) and fold them into get_rf. Default off. One /metrics fetch "
                         "per HOST per poll, shared across its two GPU instances.")
    ap.add_argument("--rr-bsat-chips-per-m", type=float, default=0.0,
                    help="#33 gap 3, THE CARRIER-AIDED CODE LOOP: couple the joint "
                         "state's per-sat range-rate rows into its code-bias rows, "
                         "d(b_sat)/dt = this * rrate (chips per m/s). 0.0 (default) is "
                         "the uncoupled filter, F identity -- state_filter.py implements "
                         "the link and demands the SIGN be measured on sky before arming "
                         "(|value| = f_chip/c = 0.03412 on the 10.23 Mcps bands). "
                         "MEASURED 2026-08-18 03:00 on 4 h of sky: with the fold-fed "
                         "(Hz-class) rrate feed, d(b_sat)/dt vs rrate regresses to "
                         "+0.00017 chips per m/s, rho +0.07 -- 200x below physics, i.e. "
                         "that feed carries no code-predictive information and arming on "
                         "it would inject noise-driven drift into b_sat at up to "
                         "~4 chips/min. DO NOT ARM until --rrate-phase-feed (mHz class) "
                         "is live and the regression is re-run with a measurable input. "
                         "⚠️ The joint state is ONE object per receiver, created by "
                         "whichever chain touches it first: arm in the COMMON yaml "
                         "section, never per-chain, or the value silently depends on "
                         "thread startup order (the _joint_state near-miss class).")
    ap.add_argument("--rrate-coarse-deweight", type=float, default=8.0,
                    help="THE FLL->PLL HANDOFF. While a satellite holds a fresh fine lock "
                         "(see --rrate-fine-hold-s), its COARSE measurements enter at "
                         "sigma x this factor. Both feeds keep running -- the coarse one "
                         "is what re-acquires after a cycle slip or a re-seed -- but a "
                         "60 mHz observable must not out-vote a 16 mHz one at equal "
                         "weight, which is exactly what pinned phi0 at 0.4-1.1 rad while "
                         "the fine feed fired once a minute. 8x sigma = 64x variance, so "
                         "one fine sample outweighs ~64 coarse ones; the row is then "
                         "governed by phase and merely SUPERVISED by frequency. 1.0 "
                         "disables the handoff (both feeds at face value).")
    ap.add_argument("--dr-fengine-axis", type=int, default=0,
                    help="#83 THE AXIS FIX: derive the dead-reckon time base from the "
                         "F-ENGINE hop counter (newest telemetry pow_hop + wall-ELAPSED "
                         "since its fetch) instead of wall-minus-anchor. MEASURED basis "
                         "(2026-08-17): the nodes' hop->utc mapping is PPS-true to 3 us "
                         "(implied utc0 spread 0.000 ms across 32 sats), while the "
                         "broker's wall axis carries ~1.45 ms of absolute NTP error "
                         "(chrony root dispersion 1.5 ms) DRIFTING at ~1 us/min. The "
                         "static part hides in the solved receiver clock (common-mode) "
                         "and only bites at axis crossings -- the flip's 14865-chip slew "
                         "target = 1.4531 ms exactly, the INNOV giants on dr seeds -- but "
                         "the DRIFT is the 2(b) phase-transport walk (~0.6 chips/min = "
                         "1 us/min x 10.23 Mcps): under cp0 transport the back-reference "
                         "round trip cancels it, under phase transport nothing does, and "
                         "on chains whose clock is ADOPTED (no detections) nothing "
                         "re-solves it away. With this axis, wall time enters only as a "
                         "sub-cycle DIFFERENCE (NTP slew x seconds = ns). Default OFF for "
                         "the replay gate; arming it moves every dr seed's epoch, so "
                         "fixtures that exercise dr paths re-bless.")
    ap.add_argument("--dr-forecast-lead-s", type=float, default=0.0,
                    help="#46/A1 THE FORECAST EPOCH. Build dead-reckon seeds for the hop "
                         "H = (newest telemetry pow_hop) + this many seconds, instead of for "
                         "\"now\". 0 = off (the replay gate's requirement, and exactly the "
                         "historical behaviour). A seed is a forecast WITH A LABEL "
                         "(ref_hop, phase, doppler, rate) and is defined entirely on the hop "
                         "axis, so it never needed a notion of \"now\" -- asking for one is "
                         "what dragged the wall clock, NTP and the telemetry lag into the "
                         "seed path, and it is why an ephemeris-epoch substitution measured "
                         "65x WORSE (reverted 2db3eec1f) and --innov-dr-seeds was falsified "
                         "in 20 minutes (09a8dc52e). Forecasting to a CHOSEN hop removes the "
                         "question: the measured ~217 ms lag (105 ms window quantisation + "
                         "~100 ms pipeline, 59 ms IQR, occasional 1 s excursions) no longer "
                         "enters the arithmetic and only sets how large this lead must be. "
                         "Pick it to exceed transport + install with margin -- 1-2 s is "
                         "ample against a 0.2-0.3 s lag -- but not so large that the "
                         "linear range-rate propagation in cp_predicted degrades (range "
                         "acceleration ~0.2 m/s^2 gives ~0.4 m = 0.013 chips at 2 s).")
    ap.add_argument("--fe-axis-stale-s", type=float, default=30.0,
                    help="Log loudly when the newest telemetry hop -- the thing t_now_abs is "
                         "built from -- has not advanced for this many seconds. 0 disables. "
                         "LOG ONLY. Motivation (2026-08-18): cx19/gnss0 deadlocked its DPDK "
                         "capture window, its pow_hop froze, and because the broker takes the "
                         "time base from that ONE combiner without a staleness test, "
                         "t_now_abs froze too. det_age went negative and grew at 1 s/s, the "
                         "clock solve saw every residual as enormous and latched at a frozen "
                         "value, and all four clock-ADOPTING chains lost tracking together "
                         "(#75) -- with nothing in the log naming the cause. Judged as a "
                         "tracking failure it is unfixable; named as a stalled instance it is "
                         "a node restart.")
    ap.add_argument("--clock-step-guard-s", type=float, default=0.5,
                    help="Log loudly when the offset between cf06's wall clock and the "
                         "F-engine's GPS-disciplined axis JUMPS by more than this (s). "
                         "LOG ONLY -- never a control input, and deliberately not a "
                         "correction. Motivation: the orbit model is evaluated at wall time "
                         "(the one consumer needing absolute UTC to be true, not merely "
                         "self-consistent), and nothing bounds what that clock may do; a "
                         "step moves every model seed on every chain at once, fleet-common "
                         "and otherwise unattributable. ⚠️ RESOLUTION LIMIT, measured: the "
                         "observable F-engine 'now' is the newest telemetry pow_hop, which "
                         "trails the sky by the gather/serve latency -- -99.6 ms median with "
                         "a 59 ms IQR over broker_onsky_e5a's 520 cycles. So this sees "
                         "STEPS well above that jitter (the seconds-scale ones NTP actually "
                         "makes) and is blind below it. That same lag is why the epoch was "
                         "NOT moved onto this axis: it would trade 1.45 ms of NTP error for "
                         "~100 ms of pipeline lag (measured 0.6-2.4 chips of seed motion).")
    ap.add_argument("--innov-dr-seeds", type=int, default=0,
                    help="#83 2(d): compute INNOV for DR-OWNED seeds too. The exclusion "
                         "exists because dr writers historically stamped ref_hop from "
                         "WALL time while detections ride the F-engine hop counter -- a "
                         "cross-axis dh reads the sub-second wall-vs-F-engine offset x "
                         "chip rate (measured: flipped G26 INNOV p95 2598 while MINNOV "
                         "1.27 and q 3.12 said the tap was fine). Under --dr-fengine-axis "
                         "the dr stamps ride the F-engine axis and the exclusion's "
                         "premise is gone; this flag re-admits them. HARD-TIED to "
                         "--dr-fengine-axis: without it this flag is ignored, because a "
                         "cross-axis INNOV would feed the escape referee garbage "
                         "(cp_err = _inv in the hold branch -- one number, two "
                         "consumers). ⚠️ ARMED AND FALSIFIED ON SKY 2026-08-17 22:30, "
                         "DISARMED: first flip after arming (PRN 4) served INNOV "
                         "+2201.31 chips / p95 2198.42 while unflipped sats in the SAME "
                         "line read 2.39-2.64. The axis fix removed the NTP DRIFT but not "
                         "the STALENESS: 2201 chips = 215 us = the sub-millisecond part of "
                         "the telemetry lag (integer ms vanish, 1 ms = one L5 period), and "
                         "t_now_abs rides the newest pow_hop, which trails the sky by "
                         "-99.6 ms median / 59 ms IQR. Re-arming requires LAG COMPENSATION "
                         "on t_now_abs, not a flag flip. MINNOV referees dr-owned sats.")
    ap.add_argument("--model-primacy-max", type=int, default=0,
                    help="#83 P3-3b, THE FLIP: at most this many PRNs run MODEL-PRIMARY on "
                         "a search-backed chain -- their seeds come from the dr-slew path "
                         "(BRDC + joint clk + b_sat, the 'slew' consumer), their "
                         "detections feed ONLY the filter, the innovations and the "
                         "referee, never the seed. Admission is per-PRN on the MEASURED "
                         "model innovation (see --model-primacy-p95); the best-p95 "
                         "eligible PRNs are flipped and the remaining eligible ones are "
                         "the in-poll controls. 0 = off (the default, and the replay "
                         "gate's requirement). MEASURED BASIS for the flip (2026-08-17 "
                         "12:08): MINNOV p95 1.4-1.9 chips vs INNOV p95 2.2-2.5 on the "
                         "SAME established satellites -- the model forecasts the sky "
                         "BETTER than the search-anchored seed, whose every re-anchor "
                         "injects the search's own 1-2 chip per-fix scatter; and the "
                         "rotating coast test carried a withheld sat at -0.13/-0.19/-0.31 "
                         "chips banded over 600 s.")
    ap.add_argument("--rrate-kcoh-feed", type=int, default=0,
                    help="#83 Phase 3 step 1: feed the joint rrate state the KCOH fold's "
                         "remaining rate (rate_hz + rate_resid_hz, task #57) -- a "
                         "PHASE-PROGRESSION estimator (segment-phasor lag products over "
                         "the fleet-summed prompt), which is the different KIND the "
                         "coarse feed's own post-mortem asked for: deep_rate is a "
                         "spectral argmax with a flat structure function (~5.6 Hz of "
                         "pure noise) and fcoh's spectral fit failed for sharing its "
                         "peak selection. This one is probe-calibrated and alive on "
                         "satellites a single tracker cannot hold -- the DR chains' "
                         "first credible carrier-rate measurement. Referenced like the "
                         "coarse feed: REC_P is already derotated by the applied "
                         "command, so remaining + applied re-references y to the BASE "
                         "seed. While a satellite has a fresh kcoh acceptance its "
                         "coarse measurements are deweighted exactly like the fine "
                         "hold. Default OFF for the replay gate (recorded argv rules); "
                         "armed in the yaml globals.")
    ap.add_argument("--joint-model-primary", action="store_true",
                    help="Feed MODEL-PRIMARY chains (E5a/B2a/E5b/B2b) into the joint solve. "
                         "⚠️ DEFAULT OFF AFTER A MEASURED REGRESSION (2026-08-10). The "
                         "model-primary measurement carries NO QUALITY GATE -- a detection has "
                         "an SNR to threshold on, a dead-reckoned seed does not -- so a "
                         "satellite whose tracker has not converged contributes a y of hundreds "
                         "of chips. birth_max does not stop it: that gate is conditioned on "
                         "P[0,0] < 100, which is false for minutes after a restart, so early "
                         "births are unvetted. Measured with it on: clk walked to +445 chips "
                         "against a true ~150, biases reached -427/-330/+281 where they should "
                         "be a few, tau_band blew to +792 chips, and 67-82%% of updates were "
                         "rejected. It looked healthy for the first hour only because the "
                         "satellites then up happened to be good; the failure arrives with the "
                         "next rise. Re-enable once the measurement is gated on tracking "
                         "quality (deep_snr/coh_frac), which is what #33 P2b needs anyway. "
                         "── THAT PRECONDITION IS NOW MET (2026-08-11). The gate landed in "
                         "2d5bccacd, 9 h after this default was set: _track_ok requires "
                         "coherence_s > 0 AND deep_snr >= --joint-min-deep-snr AND coh_frac "
                         ">= --joint-min-coh-frac, and it is applied to THIS feed. The text "
                         "above is kept because it is the reason the flag exists, not because "
                         "the gate is still missing -- read as-is it says 'do not turn this "
                         "on', which is no longer the advice. Still default-off because the "
                         "remaining risk is different and unbounded: this admits ~11 sats at "
                         "once per chain, and mass birth is the filter's known divergence "
                         "trigger (#39). Enable ONE chain at a time and watch the first "
                         "minutes; the state is shared, so gps_l5's consumers see it too.")
    ap.add_argument("--joint-mask-after", type=int, default=50,
                    help="P2c: engage --joint-mask-prn only once that satellite has this many "
                         "ACCEPTED updates. Masking from startup is a vacuous test -- the sat "
                         "is never born into the state, so there is no b_sat to freeze and "
                         "nothing to coast on. It must be established FIRST and withheld "
                         "after, which is the whole point: we are testing whether the state "
                         "REMEMBERS a satellite, not whether it can invent one. "
                         "⚠️ THE TWO SEEDING DISCIPLINES FEED AT VERY DIFFERENT RATES, which is "
                         "why this defaults low: a model-primary chain offers every seeded sat "
                         "EVERY cycle, so the count accrues in seconds, while a search-anchored "
                         "GPS sat only enters when a detection clears --joint-min-snr and can "
                         "take hours. Calibrated on the wrong path a threshold either fires "
                         "instantly or never -- 200 fired in 6 s on Galileo and had not fired "
                         "on GPS in 14 min. 50 is reachable on both and ample to establish a "
                         "bias (sigma_b0 10 against a 0.3-chip measurement converges in tens).")
    ap.add_argument("--joint-min-snr", type=float, default=60.0,
                    help="minimum detection SNR for a residual to enter the JOINT solve. "
                         "The default matches --period-check-snr, whose docstring records "
                         "the measurement: below it a detection's phase is noise with "
                         "~2000-chip residuals. Ungated, ONE such detection walked the "
                         "state to -0.028 ppm on 2026-08-09 (docs 11.23).")
    ap.add_argument("--joint-min-deep-snr", type=float, default=10.0,
                    help="minimum TRACKER significance (deep_snr, floor-cleared) for a "
                         "residual to enter the joint solve. --joint-min-snr gates on the "
                         "SEARCH SNR carried in `best`, which is a different and much "
                         "staler number: detections latch for up to the 1276 s per-PRN "
                         "revisit, so a satellite that has left the beam keeps presenting "
                         "the strong SNR of its last good scan while its tracker output is "
                         "already noise. That is exactly how PRN 25 got in on 2026-08-10 -- "
                         "elevation RISING 45->65 deg (it transited out of the beam, it did "
                         "not set), deep_snr 30->2.4, coh_frac 0.94->0.19, code phase "
                         "uniform over the 10230-chip code -- and destroyed the state. "
                         "MEASURED, not guessed: over 6827 GPS rows the population is "
                         "sharply bimodal, noise at deep_snr 2-3 / coh_frac 0.2 and signal "
                         "at 24+ / 0.92+; 10 sits in the empty gap and admits 70.5%%.")
    ap.add_argument("--joint-min-coh-frac", type=float, default=0.3,
                    help="minimum coherent fraction for a residual to enter the joint solve. "
                         "Nearly redundant with --joint-min-deep-snr (both gates select the "
                         "same 70.5%% of rows, which is the two measuring the same physics) "
                         "and kept as a cheap independent guard, not as a second opinion.")
    ap.add_argument("--joint-slew-max-chips", type=float, default=5.0,
                    help="P2b consumer 3: refuse the joint slew target if it disagrees with "
                         "the legacy (clock EMA + SatBiasFilter) target by more than this. "
                         "Both describe the same receiver, so a few chips is the whole "
                         "plausible range -- b_sat runs +-3-6 chips and the clock is common "
                         "to both. A larger gap means one estimator is broken, and the "
                         "legacy path is the one with years behind it. Same discipline as "
                         "--joint-max-rate-ppm on consumer 1, which would have refused the "
                         "-0.028 ppm runaway that froze the trackers no matter what went "
                         "wrong upstream.")
    ap.add_argument("--seed-phase-transport", action="store_true",
                    help="#45 step 6: dead-reckon/slew seeds carry code_phase_at_ref_chips "
                         "(the PHASE at ref_hop) in addition to the sample-0 argument. "
                         "propagate_seed prefers the phase, which has no t_abs lever, so a "
                         "producer that edits the Doppler without re-projecting cp0 can no "
                         "longer walk the despread (~1700 chips/Hz). The search-fed path has "
                         "shipped a phase all along; this closes the model-primary gap. "
                         "MEASURED before enabling (scripts/gnss/e2e_phase_transport.py, real "
                         "propagate_seed + GPU): arg 0.788 chips, phase+ 0.785, and the "
                         "UNCONVERTED phase 51.999 -- the broker references a hop's first "
                         "sample and the C++ side its last, so the conversion is what makes "
                         "this safe. OFF by default until it has flown.")
    ap.add_argument("--joint-feed-warmup-s", type=float, default=240.0,
                    help="withhold ALL measurements from the joint state for this many "
                         "seconds after broker start. Birth is the one path with no "
                         "innovation to gate on, and the establishment window is when the "
                         "y-feed is at its worst: the 2026-08-12 00:08 zombie was born "
                         "from establishment-phase garbage (biases +-20-46 chips), locked "
                         "it in, and no escape could fire because the garbage was "
                         "self-consistent. Seeds/holds settle in ~3 min; 240 is that plus "
                         "margin. The filter object itself is still created (warm-started) "
                         "immediately -- only the feed waits.")
    ap.add_argument("--joint-gauge", choices=["median", "prior"], default="median",
                    help="#94/S2: what pins the joint filter's unobservable common mode. "
                         "'median' (the shipped convention): a median(b)=0 pseudo-measurement "
                         "over the ACTIVE sats -- a POPULATION statement, so bare clk steps "
                         "when the middle element changes and b_i is fleet-relative by "
                         "construction. 'prior': each b_i carries its own absolute prior "
                         "b ~ N(0, 2 chips) -- the physical envelope of a per-sat code bias "
                         "(ephemeris residual + DCB + differential iono/tropo at 1176 MHz; "
                         "one chip is 29.3 m) -- Huber R-inflated past 3 sigma so a wild "
                         "bias's weight falls as 1/y^2. No satellite's gauge mentions any "
                         "other, so a rise or set is a NON-EVENT: measured in the selftest, "
                         "at-event clk step 0.009 chips vs 1.05 for the median pin. The "
                         "trade, chosen deliberately: a uniformly displaced comb heals "
                         "SLOWLY (clk+b_i stays correct throughout, so sum-reading "
                         "consumers ride it; only the bare-clk consumer degrades, and it "
                         "has its own 5-chip refusal).")
    ap.add_argument("--joint-rereference", action="store_true",
                    help="JOINT FILTER: treat a membership change as a change of "
                         "COORDINATES rather than as an observation. On a join/leave the "
                         "gauge (mean(b)=0) is re-referenced EXACTLY -- b_i -= mean(b), "
                         "clk += mean(b), P carried through the same linear map -- so "
                         "clk + b_i, the only quantity a consumer reads, is invariant and "
                         "mean(b)=0 holds without gauge() having to move anything. The "
                         "legacy path instead inflates P[0,0] by 4 chips^2 per event, which "
                         "makes the clock uncertain, lets the next measurement yank it, and "
                         "(via the clk<->rate correlation predict() rebuilds) feeds that "
                         "yank into clk_rate. Sound at 6+ sats; self-defeating at 4, where "
                         "it produced the 15-chip clock swing that made both Galileo chains "
                         "refuse the state. MEASURED (bench_gauge_churn.py): observable "
                         "error p95 2.77 -> 0.76 chips under churn, and a mass birth of 11 "
                         "satellites steps the clock +0.011 instead of +1.794 chips -- the "
                         "operation that diverged the state on 2026-08-10 and 08-18. The "
                         "legacy rate shield (P[0,1] = 0) is kept either way.")
    ap.add_argument("--joint-max-rate-ppm", type=float, default=0.01,
                    help="a JOINT rate consumer REFUSES a |clk_rate| beyond this (ppm) and "
                         "falls back to the l-a EMA. CHORD's reference is GPS-disciplined "
                         "and measures 4e-5 ppm, so 0.01 is ~250x headroom and still "
                         "refuses the -0.028 ppm runaway that froze the trackers.")
    ap.add_argument("--dll-combiners", default="",
                    help="FLEET-COMBINED DLL (docs/CHORD_GNSS_SHARED_DLL.md): comma-separated "
                         "combiner endpoints ({a..b} ranges expanded) whose RAW Early/Prompt/"
                         "Late powers are SUMMED before the discriminator is formed, closing "
                         "ONE code loop at full signal bandwidth. On CHORD the F-engine comb "
                         "spreads L5 across every node, so a single instance correlates 7 x "
                         "195.3 kHz of a 20.46 MHz lobe -- 6.7%%, -11.8 dB -- and no instance "
                         "can do better; that gap, not sensitivity or the quality gate, is why "
                         "the code lock is episodic. The combine is legitimate because the "
                         "discriminator is NON-COHERENT: (E-L)/(E+L) is built from POWERS, "
                         "which add with no phase alignment and no sample sync, only the same "
                         "window (pow_hop, exact). Empty = today's behaviour, --combiner alone. "
                         "REQUIRES trim_gain 0 on the trackers: E/L are measured relative to "
                         "the phase each instance despread at, so independent local trims make "
                         "the sum SMEAR instead of sharpen (design section 5).")
    ap.add_argument("--reseed-admit-absent", type=int, default=0,
                    help="task #90: let the #50 re-seed fire on a PRN the presence gate calls "
                         "ABSENT, provided it is still SEEDED (the model says it is up) and "
                         "TWO consecutive qualifying spec fits within 600 s agree on the SIGN "
                         "of tau. DEFAULT OFF. THE HOLE THIS CLOSES (E32, 2026-08-24 22:17): "
                         "off-peak -> q floored -> presence lost -> fleet-trim disarmed -> "
                         "trim released -> LATCHED, while SPEC-FIT measured a significant "
                         "peak at tau -0.5 every minute. On gps_l5 the narrowed search "
                         "re-admits (#79); the dead-reckon chains have NO search, so absent "
                         "was absorbing forever. The two-strike same-sign gate is the noise "
                         "guard the presence gate used to provide: on a truly faded sat the "
                         "fit is noise and tau's sign is a coin flip, so consecutive "
                         "agreement halves the false-fire rate per strike and a random walk "
                         "cannot build.")
    ap.add_argument("--dll-spacing", type=float, default=0.5, metavar="CHIPS",
                    help="the tracker's Early/Late tap offset from Prompt, in the signal's own "
                         "(component) chips. MUST MATCH the node's dll_spacing for this chain "
                         "(config/gen_chord_gnss_config.py dll_spacing_chips): the taps live on "
                         "the node, but the trim law tau = -disc/4 * (D/0.5) here and the C++ "
                         "fleet loop's copy (armed from this value) assume it -- a mismatch is a "
                         "pure gain error. Set by the CORRELATION WIDTH, not the chip: on a "
                         "chip-narrow correlation (L5/E5/B2: a 7-channel comb) 0.5 sits on the "
                         "slope; GPS L2C despreads ONE 195 kHz channel, whose |sinc|^2 nulls at "
                         "2.6 CM chips, so at 0.5 E and L sit 90%% up the peak and q = 2P/(E+L) "
                         "is pegged at 1.11 for a perfect lock (harness, 2026-09-02) -- every "
                         "q-gated instrument read the first L2C locks as dead. L2C runs 2.0: q "
                         "6.7 on peak, slope 1.10/chip, monotonic to +-2 chips.")
    ap.add_argument("--dll-deep-gate", default="",
                    help="fleet DLL (task #49): PRNs whose trim PRESENCE is gated on "
                         "deep_snr >= margin x deep_floor instead of on summed prompt power. "
                         "Comma-separated PRNs, or 'all'. DEFAULT OFF. The prompt gate is "
                         "ON-PEAK-BIASED and therefore a LATCH: prompt power is suppressed "
                         "precisely when the tap is off-peak, so an off-peak satellite fails "
                         "the gate and is never trimmed back. Measured 2026-08-12: 26 of 36 "
                         "well-detected satellites (72%%) were excluded, and the excluded ones "
                         "carried the LARGER discriminators. deep_snr is immune because the "
                         "deep fold re-searches rate and phase, so it detects the satellite "
                         "wherever the tap sits. Opt in ONE PRN AT A TIME: enabling the whole "
                         "fleet at once makes 72%% of it newly trimmable against a slew cap "
                         "already railing 67-100%% of the time, which trades the latch for an "
                         "oscillation and makes the A/B uninterpretable.")
    ap.add_argument("--dll-deep-gate-from-search", type=float, default=0.0, metavar="SNR",
                    help="task #79: AUTO-GENERATE the --dll-deep-gate set from the SEARCH. Any "
                         "PRN the search detects at snr >= this, within "
                         "--dll-deep-gate-search-hold-s, joins the deep-gated set for that "
                         "window (unioned with any hand-listed PRNs). 0 = off. "
                         "WHY THE SEARCH AND NOT A TAP STATISTIC: presence decides whether a "
                         "satellite may be CORRECTED, and every tap-derived statistic tried so "
                         "far answers it with a number that is itself suppressed when the tap "
                         "is off-peak -- q first, then prompt power, both on-peak-biased, both "
                         "latching an off-peak satellite out of its own correction (see the "
                         "--dll-deep-gate note: the same defect fixed twice with the same "
                         "class of statistic). The search is the one estimator that is "
                         "INDEPENDENT OF WHERE THE TAP SITS: it re-derives code phase over the "
                         "full period every pass, so it detects the satellite wherever the "
                         "tracker happens to be looking. It is also preferable to deep_snr, "
                         "which fires on noise (a below-horizon PRN has been measured at 11.2x "
                         "its own floor, uncorrelated with search SNR) -- the deep fold "
                         "re-searches rate AND phase, so its detection statistic keeps a "
                         "maximum over nuisance parameters that a pure-noise cell can win. "
                         "The bar is in the search's own snr units, where the acquire "
                         "threshold is 30 and the pure-noise ceiling ~17.")
    ap.add_argument("--dll-deep-gate-search-hold-s", type=float, default=90.0,
                    help="task #79: how long a search detection keeps a PRN in the "
                         "auto-generated deep-gate set (default 90 s). Must exceed the search "
                         "REVISIT (90-270 s at CHORD's cadence is the observed range, and a "
                         "PRN is not re-detected every cycle), or presence flaps on and off "
                         "with the search's luck and the trim is armed and disarmed under a "
                         "satellite that never changed. This is a HOLD, not a staleness "
                         "budget: it answers 'is this satellite up and detectable', which does "
                         "not stop being true between passes.")
    ap.add_argument("--spec-span-chips", type=float, default=2.0,
                    help="task #50: the delay fit's scan half-width, which MUST match "
                         "fit_spectrum_delay's span_chips. A fit at the edge is a saturation, "
                         "not a measurement, and is refused rather than acted on.")
    ap.add_argument("--dll-min-instances", type=int, default=2,
                    help="fleet DLL: instances that must report the same window before their "
                         "sum is used. Below this the PRN falls back to the single --combiner "
                         "discriminator, so a partially-down fleet degrades instead of "
                         "stalling.")
    # -- TASK #59: the frame-synced telemetry gather ------------------------------------------
    ap.add_argument("--telem-gather", default="",
                    help="TASK #59: read per-record telemetry from the GATHER instance at "
                         "host:port (default port 11061) instead of inferring alignment from "
                         "~60 REST polls per cycle. Every frame carries its chain, its instance "
                         "and an ABSOLUTE window index on the F-engine sample clock, so "
                         "grouping instances is an exact integer match rather than a guess "
                         "about arrival order. Empty = off; the REST path is untouched either "
                         "way, and ONE reader thread serves every chain in this process.\n"
                         "Connecting alone changes NOTHING -- it only fills the store. See "
                         "--telem-coherent for the first consumer.")
    ap.add_argument("--telem-coherent", action="store_true",
                    help="feed fleet_coherent from --telem-gather instead of from "
                         "/get_records. THE SAME ESTIMATOR, a different transport: that is "
                         "what makes the comparison a measurement. ⚠️ Run it on ONE CHAIN at a "
                         "time, with the others left on REST as a same-poll control -- a "
                         "before/after across restarts cannot resolve it, because the sky "
                         "churns faster (2026-08-13: deep_snr max swung 52-197 in four "
                         "minutes). No effect unless --telem-gather is set and the gather is "
                         "reachable; if it is not, this silently falls back to the REST poll "
                         "rather than publishing nothing.")
    ap.add_argument("--telem-windows", type=int, default=8,
                    help="windows of telemetry fed to fleet_coherent (4 records each at the "
                         "default packing, so 8 windows = 32 records ~ 0.34 s)")
    ap.add_argument("--estimator-every-s", type=float, default=15.0,
                    help="minimum seconds between the TELEMETRY-WALK estimators (cn0_prompt "
                         "+ kcoh) per chain; the last values keep being served between "
                         "runs. They are pure-Python walks over ~1500 record decodes each, "
                         "and at every-cycle cadence across five chains they ate ~75%% of "
                         "the interpreter -- the telemetry READER then missed the gather's "
                         "200 ms delivery deadline and 5 of 6 frames were dropped "
                         "(measured 2026-08-15 17:17, gaps 128k vs frames 39k). The "
                         "estimators' inputs are ~1.3 s windows; nothing downstream needs "
                         "them faster than the display refreshes.")
    ap.add_argument("--element-poll", action="store_true",
                    help="TASK #57 step 2: poll the --dll-combiners' /get_elements each cycle "
                         "and serve the per-element complex-gain table (amplitude AND phase "
                         "per antenna, per instance -- the beam/peel coefficients) via the "
                         "publisher's /get_elements. Off by default and gated on its own flag "
                         "for the same reason --spectrum-endpoints is: replay is strict-"
                         "ordered, and old transcripts do not carry these GETs. Instances "
                         "predating the endpoint (nodes not yet restarted) are skipped and "
                         "counted, never fatal.")
    ap.add_argument("--element-archive-dir", default="",
                    help="append the RAW per-element gain parts (u, p2, q per antenna) to "
                         "<dir>/elem_<chain>_<YYYYMMDD>.jsonl -- present satellites and the "
                         "noise probes only. Raw parts on purpose: the beam map and peel "
                         "solves are offline consumers and a combined number cannot be "
                         "un-combined. Empty = no archive.")
    ap.add_argument("--telem-dll", action="store_true",
                    help="TASK #63: close the CODE loop on the discriminator formed in this "
                         "broker from the un-summed comb (gnss_broker/combdll.py) instead of on "
                         "the powers the trackers formed by summing across each instance's "
                         "channels. The arithmetic is the combiner's own -- coherent across an "
                         "instance's channels, then powers added across instances -- moved to "
                         "where the whole band is visible; what it removes is the tracker-side "
                         "sum that destroys the frequency axis on the way.\n"
                         "MEASURED BEFORE SHIPPING (scripts/gnss/comb_dll_ab.py, gps_l5, 20 "
                         "cycles, 240 paired PRNs): disc agrees to +0.0015 +- 0.0223 against "
                         "the POLLED arm's own cycle-to-cycle control of +-0.0472, q to +0.008 "
                         "+- 0.044 against 0.090, prompt power to +0.13 dB, and presence "
                         "disagreed on 10 of 240 with McNemar chi2 0.10 -- symmetric, i.e. "
                         "churn rather than bias.\n"
                         "⚠️ The polled endpoints are STILL read while this is set: the deep "
                         "statistics (deep_snr / deep_floor / coherence_s) come from the "
                         "combiner's fold and are handed to the comb rows for #49's deep gate "
                         "and the publisher. The trackers' summed slots stop being DELETABLE "
                         "only once that fold also runs here. Falls back to the polled "
                         "discriminator, per cycle, whenever the gather has no windows for "
                         "this chain -- so a gather that dies costs a log line, not the loop.")
    ap.add_argument("--comb-taps-cpp", type=int, default=0, choices=(0, 1, 2),
                    help="TAKE THE COMB DLL's per-instance/per-channel taps from the gather's "
                         "C++ reduction (<fleet-trim-url>/get_taps) instead of rebuilding them "
                         "in Python from the gathered frames. "
                         "0 = off; 1 = SHADOW (fetch and compare, still USE the Python arm, log "
                         "the paired difference); 2 = ARMED (use the C++ arm).\n"
                         "\n"
                         "WHY. combdll.instance_taps walks every (window, instance, record, "
                         "PRN, channel) of the stream -- ~140k channel-tuples per chain per "
                         "cycle, ~700k across the fleet, each allocating Python complex "
                         "objects, ~18%% of chain CPU by live profile. This process is pinned "
                         "at 100%% of ONE core by the GIL and its CYCLE TIME IS the sum of the "
                         "five chains' Python CPU (measured 12.09 s cycle against 12.77 s of "
                         "CPU per cycle-set), so that 18%% is ~2 s of every cycle. The frames "
                         "are already in C++ on the gather host; the reduction is ~6k numbers.\n"
                         "\n"
                         "⚠️ GO THROUGH 1 FIRST. The equivalence is gated offline on identical "
                         "bytes (scripts/gnss/fleetdll_gate.py: e/p/l/n_chan to 1e-9, n_rec and "
                         "hop exact, per instance and per channel) -- but that gate cannot see "
                         "a LIVE mismatch of window depth, chain naming or instance tags, and "
                         "broker_equiv is blind to this path entirely (the gather is a raw "
                         "socket, not transport, so a replay carries no telemetry and falls "
                         "back to the polled arm). Shadow mode is what measures those.")
    ap.add_argument("--fleet-trim-url", default="",
                    help="TASK #51 F3: base URL of the C++ fleet loop's stage, e.g. "
                         "http://cf06:12051/fleet_trim. Each cycle the broker POSTs "
                         "<url>/set_policy with the armed PRN set, the loop constants and the "
                         "tracker endpoints; the C++ side then runs the discriminator, the "
                         "integrator and the actuation at ~23.8 Hz instead of this process's "
                         "~3.1. POLICY STAYS HERE -- presence, the floors, the deep gate and "
                         "the arming verdict are all computed above and only their RESULT "
                         "crosses. ⚠️ Do not run this together with --fast-trim-hz on the same "
                         "chain: two loops writing one trim is a race, and neither would be "
                         "measurable.")
    ap.add_argument("--fleet-trim-bandwidth", type=float, default=1.4,
                    help="closed-loop bandwidth (1/s) published to the C++ fleet loop as "
                         "gain_per_s. ⚠️ NOT a per-update gain: the controller divides by the "
                         "update rate it measures. The stability bound is bandwidth x "
                         "round-trip < ~0.5, and the loop's measurement round trip is "
                         "~0.3-0.5 s (4-window disc average + actuation + record + fold), so "
                         "~1.4 is near the ceiling. The per-update 0.25 that was stable at the "
                         "Python arm's 3.1 Hz (bandwidth 0.78) was a LIMIT CYCLE at 23.84 Hz "
                         "(bandwidth 6): measured on sky 2026-08-15, trim swinging +-1 chip at "
                         "5-10 s period, q reaching 3.3 and being thrown off the peak. Slew "
                         "authority = bandwidth x 0.25 chips = 0.35 chips/s here, ~3x the "
                         "0.121 chips/s drift and above the ~0.31 chips/s peak slew of the "
                         "+-1-chip/20 s clock breathing.")
    ap.add_argument("--fleet-trim-rebase-adjust", type=int, default=0,
                    help="#92 THE HANDOVER (default OFF). When a model-primary seed is "
                         "RE-BORN (the (slew)->(cp0) flip) its physical position steps by "
                         "the birth step while the C++ standing trim keeps carrying the "
                         "same chips -- the tap lands a chip off the sky, q craters, and "
                         "the trim is wiped and rebuilt (~25-min sawtooth, E3). With this "
                         "on, the broker POSTs the compensating -birth_step to the "
                         "gather's /adjust_trim in the SAME cycle, bounded at 2.5 chips "
                         "(a larger step -- e.g. the shared-clock births at ~140 chips -- "
                         "is not something a <=3-clamp trim was carrying; skipped loudly). "
                         "Needs the gather binary serving /adjust_trim; a 404 logs as "
                         "FAILED and the loop degrades to today's wipe-and-rebuild.")
    ap.add_argument("--fleet-trim-readback", type=int, default=0,
                    help="TASK #76: after posting /set_policy, GET <fleet-trim-url>/get_dll "
                         "and read back the controller's STANDING PER-PRN TRIMS (chips). The "
                         "broker commanded these corrections and was blind to them: the "
                         "escape referee and the innovation (task #83 2(d)) judge the seed as "
                         "if untrimmed while up to 3 chips of command stand at the trackers. "
                         "⚠️ Default OFF for the replay gate's sake, not by preference: the "
                         "harness re-invokes a fixture with the RECORDED argv, and a "
                         "recording made before this flag existed has no get_dll entries in "
                         "its get stream -- an unconditional GET would diverge every pre-#76 "
                         "fixture at the first cycle. Arm it in the chain yaml next to "
                         "fleet-trim-url; a new recording then carries it natively.")
    ap.add_argument("--fast-trim-hz", type=float, default=0.0,
                    help="TASK #51: run the CODE LOOP at this rate, on its own thread, instead "
                         "of once per policy cycle. 0 = off (the loop stays on the cycle).\n"
                         "THE DEFECT THIS FIXES: the DLL's authority is a step PER UPDATE and "
                         "the update rate was the ~12 s cycle, so the loop was 23x slower than "
                         "the drift and could never hold lock. Measured offline, noiseless, "
                         "from a GOOD (+0.373 chip) seed: the tap walks off at 0.121 chips/s "
                         "while the loop corrects at most dll_gain*|tau|max/cycle = "
                         "0.25*0.25/12.1 = 0.0052 chips/s.\n"
                         "⚠️ NO GAIN SETTING FIXES IT: tau is clamped to |tau| <= 0.25 chips by "
                         "construction, so even dll_gain 1.0 tops out at 0.021 chips/s. The E/L "
                         "discriminator is a FINE loop; update RATE is the only lever.\n"
                         "Break-even is 0.121/0.0625 = 1.94 Hz; 5-10 Hz gives 3-5x margin and "
                         "the #59 telemetry arrives at 23.84 Hz, so the data is already there. "
                         "POLICY STAYS ON THE CYCLE -- the main loop still decides who is "
                         "present and trimmable; this only runs the integrator and the POST.\n"
                         "Requires --telem-gather (it reads the gather store, never REST).")
    ap.add_argument("--fast-trim-windows", type=int, default=4,
                    help="telemetry windows averaged per fast-trim update (4 windows = 16 "
                         "records = 0.17 s). Shorter is noisier per update but the loop's "
                         "problem is SLEW, not noise -- see --fast-trim-hz. Must be short "
                         "enough that consecutive updates see fresh records.")
    ap.add_argument("--telem-dll-windows", type=int, default=0,
                    help="windows meaned for --telem-dll (0 = --telem-windows). 32 windows x 4 "
                         "records = 128, which is what the combiners' EMA integrates; a much "
                         "shorter window makes the comb arm noisier than the arm it replaces "
                         "and the A/B would read that as a regression.")
    ap.add_argument("--n2-combiners", default="",
                    help="CROSS-NODE COHERENT COMBINE FOR PATH B: comma-separated *_n2combine "
                         "endpoints ({a..b} ranges expanded), run through the SAME "
                         "fleet_coherent as --dll-combiners but kept as a separate population "
                         "so the two paths stay independently measurable. Empty = off.\n"
                         "\n"
                         "This is not an optimisation, it is how path B works at scale. A node "
                         "holds every 8th science channel and each GPU a stride-16 comb, so one "
                         "instance sees 7 of the 105 channels under the L5 lobe -- and on a "
                         "NARROW signal (L2C, E5b) at full CHORD a node may hold ONE channel. "
                         "The fleet combine is then the measurement, not an improvement to it.\n"
                         "\n"
                         "THE ESTIMATOR IS PARTITION-INVARIANT, which is the useful property "
                         "here and was verified rather than assumed. Holding the total aperture "
                         "fixed at 12 bands and 128 records and splitting it 2x6, 4x3, 6x2 and "
                         "12x1 through this same function gives fleet deep_snr 81.9/80.9/81.6/"
                         "82.0 at a fleet per-record SNR of 10.4, and 26.2/26.4/26.3/26.6 at "
                         "3.5. Node count does not enter. The reason is that the leave-one-out "
                         "reference is the sum of the OTHER instances: K instances of B/K bands "
                         "give it (K-1)/K of the total aperture, so slicing finer makes the "
                         "reference slightly BETTER, while costing K-1 alignment constants "
                         "instead of one. Those cancel. So a narrow signal spread one channel "
                         "per node over many nodes combines exactly as well as the same channels "
                         "gathered on a few -- which is what makes L2C and E5b viable at full "
                         "CHORD.\n"
                         "\n"
                         "WHAT DOES BITE is the FLEET TOTAL per-record SNR, a bands x elements "
                         "question with no node term in it. Down to ~1.7 the partitions are "
                         "still indistinguishable (11.4/11.8/11.8/11.6); at 0.87 the finest "
                         "split finally loses ground (4.3 -> 2.9), and that is already inside "
                         "the floor's shadow. Drive it with more bands or more elements "
                         "(gnssElemCal / elem_sum), not with a different node layout.\n"
                         "\n"
                         "NB the `align` figure in the log falls with instance count at fixed "
                         "total (0.917 at 2x6 down to 0.611 at 12x1, same total SNR) because "
                         "each alignment is estimated from one instance's data. It is a "
                         "per-instance diagnostic, NOT a health metric for the combine -- "
                         "deep_snr over those same four partitions is flat to 2%%.")
    ap.add_argument("--coh-min-records", type=int, default=32,
                    help="fleet coherent: common records (hops) required per PRN.")
    ap.add_argument("--cl-assist", action="store_true",
                    help="LEGACY single-chain CL mode (superseded by --cl-tracker, which keeps "
                         "CM running as the in-run control): lift each seed's code_phase_chips "
                         "IN PLACE by k*10230 with the CL segment k COMPUTED from absolute "
                         "capture time (the airspy /adcstat utc0_sample0 anchor) + almanac "
                         "range. CL's 1.5 s epoch is GPS-time-locked, so k is arithmetic, not "
                         "a 75-way search. Needs --almanac; the main trackers must despread "
                         "GPS_L2C_CL. Mutually exclusive with --cl-tracker.")
    ap.add_argument("--adc-stage", default="airspy_in",
                    help="airspy input stage name for the utc0_sample0 anchor GET (CL assist)")
    ap.add_argument("--time0-endpoint", default=None,
                    help="CHORD: REST path (relative to --rest-url, e.g. telescope/time0_ns) "
                         "serving the F-engine's GPS-disciplined absolute time of frame 0. Used "
                         "INSTEAD of the airspy /adcstat anchor. The airspy node stamps sample 0 "
                         "with host wall-clock -- good to milliseconds -- which is why it must "
                         "then SOLVE the receiver clock from measured code phases. CHORD's "
                         "frame 0 is disciplined to GPS via IRIG-B/PPS and is exact, so the "
                         "anchor is a fact rather than an estimate.")
    ap.add_argument("--dr-clock-chips", type=float, default=None,
                    help="CHORD: prime the dead-reckon receiver clock (chips) instead of "
                         "bootstrapping it from measured code phases. THIS IS WHAT LETS A NODE "
                         "WITH NO SEARCH STAGE COLD-START. The bootstrap needs --dr-min-sats "
                         "satellites already tracking to take a median of their residuals, but "
                         "nothing can track until it is seeded -- on the airspy node the search "
                         "stage breaks that circle. With a GPS-disciplined F-engine the offset "
                         "is known a priori (0 plus a fixed instrumental/cable delay), so pass "
                         "0.0 to start and calibrate the constant later from the measured "
                         "integrity residual, which the broker logs every cycle. THE PRIME IS "
                         "A SEED, NOT A MEASUREMENT: a chain with detectors REPLACES it with "
                         "its first multi-sat solve rather than EMA-ing toward one, so a wrong "
                         "prime costs one cycle rather than ~20.")
    ap.add_argument("--dr-doppler-ignores-integrity", action="store_true",
                    help="keep the BRDC (dead-reckon) Doppler for a satellite that is "
                         "MODEL-UNTRUSTED because of its CODE-PHASE integrity residual. A "
                         "stale ephemeris still demotes the Doppler -- that reason is real "
                         "for both quantities -- but an integrity residual is not: "
                         "--dr-max-integrity-chips defaults to 1.0 chip (~30 m), which is "
                         "ordinary iono plus b_sat, and it says nothing about the RANGE RATE "
                         "from the same orbit. Demoting the Doppler for it switches the seed "
                         "between two BRDC evaluations (dr and pred) every time a satellite "
                         "flips trust, and the seed is the replica's carrier phase. MEASURED "
                         "2026-08-10: gps_l5 -- the only chain with detectors, so the only "
                         "one where this fires at all -- carries 0.478 Hz of commanded-"
                         "Doppler jitter against 0.002-0.006 Hz on the four model-primary "
                         "chains, and over the 1.0486 s deep fold 0.478 Hz is 0.50 cycles of "
                         "phase ramp: |sinc| = 0.64, swinging emit to emit. Its deep fold "
                         "inflates sample-to-sample scatter by +5.55 dB where the others "
                         "manage +0.36..+0.77 (docs CHORD_JOINT_TRACKING.md 3d). NOTE this "
                         "addresses the DEMOTION-linked part; a ~0.29 Hz GPS-wide floor "
                         "remains and is NOT explained by trust flips.")
    ap.add_argument("--seed-doppler", default="auto", choices=("auto", "det"),
                    help="which Doppler the SEED carries. 'auto' (default, unchanged) prefers "
                         "the almanac/DR model + solved clock bias, which is smooth and owns "
                         "the undetected sats. 'det' uses the search's MEASURED Doppler "
                         "instead. Pick 'det' when the model is not trustworthy to well under "
                         "a Hz at the seed's age: cp0 is an ARGUMENT, so a seed's phase moves "
                         "chip_rate/carrier chips per second per Hz of Doppler error (0.0087 "
                         "chips/Hz/s) for as long as the tracker extrapolates it. Measured on "
                         "CHORD 2026-08-01: model-vs-measured +231 Hz with a 244 Hz spread, "
                         "against a measured Doppler good to ~3 Hz (proven by the residual "
                         "code rate of a real lock) -- at a 456 s anchor that is 340 chips of "
                         "seed error versus 12. 'det' also makes cp_to_seed_currency a no-op, "
                         "since cp0 was fit at exactly that Doppler.")
    ap.add_argument("--long-code-segments", type=int, default=75,
                    help="number of primary periods in the overlaid/long code the TRACKERS "
                         "despread (L2C CL = 75 x 10230; GPS L5 Q5 with NH20 baked in = 20 x "
                         "10230). The time-assist below picks which one, so this must match the "
                         "tracker's `signal`, not the search's.")
    ap.add_argument("--long-code-epoch-s", type=float, default=1.5,
                    help="the long code's GPS-time-locked repeat period, seconds (L2C CL 1.5; "
                         "L5 NH20 0.02). The assist needs unix-time mod EPOCH == GPS-time mod "
                         "EPOCH, which holds when GPS-UTC (whole seconds) and the GPS epoch "
                         "offset (315964800) are both multiples of it -- true for 1.5 and for "
                         "0.02. Absolute-time accuracy needed is ~EPOCH/2.")
    ap.add_argument("--bias-min-snr", type=float, default=0.0,
                    help="detections below this SNR do not enter the clock-freq bias median. "
                         "The bias is common-mode, so its uncertainty is (per-sat Doppler "
                         "error)/sqrt(N) -- one noise satellite is costly when N is small. "
                         "Ungated on CHORD the raw estimate scatters 10.5 Hz; the acquire's "
                         "own error predicts 0.8 Hz at N=2. 0 (default) keeps every point.")
    ap.add_argument("--nh-period-offset", type=int, default=0,
                    help="EXPERIMENT (2026-08-02): shift every seed's overlay period by N "
                         "primary code periods. The oracle measures the seeded period as a "
                         "CONSTANT 4 too high -- 4/4 strong detections, 3 satellites, seed ages "
                         "151-680 s, oracle ratios 30.1/25.8/17.4/9.8 -- and -4 == +16 mod 20, "
                         "where 16 code periods = 3125 hops = Mp = the anchor the search builds "
                         "repl0 at. This exists to TEST that, not to fix it: a constant that "
                         "works without a mechanism is how refine_span:4096 got baked in. "
                         "Applied to BOTH code_phase_at_ref_chips (which propagate_seed prefers) "
                         "and code_phase_chips. 0 = off.")
    ap.add_argument("--fit-min-snr", type=float, default=0.0,
                    help="detections below this SNR do not enter the cp-rate fit history. The "
                         "fit resolves a ~0.0148 chips/s residual; a near-threshold detection's "
                         "phase is noise, so one bad point destroys the slope rather than "
                         "degrading it. 0 (default) keeps every point -- right for the "
                         "prototype, whose detections sit well above threshold and whose "
                         "revisit is seconds. CHORD wants ~60 alongside --fit-gap-s 900.")
    ap.add_argument("--fit-hist-len", type=int, default=8,
                    help="detection snapshots kept for the cp/dop slope fits. The default 8 "
                         "predates CHORD's search cadence: at 8 detections/s, 8 points span "
                         "~1 s, so fit_maturity_span_s (30 s) is UNSATISFIABLE and "
                         "fit_trusted -- which gates BOTH hold admission and the escape "
                         "referee -- is structurally false on every strong satellite "
                         "(measured 2026-08-28: fit_trusted=False with len_h=8, snr 946; "
                         "only weak, sparsely-detected sats could ever mature, which is "
                         "why G18 got held while G20/21 never did). 256 at 8 Hz spans "
                         "~32 s: the maturity floor passes, and the slope error improves "
                         "as sigma/(T*sqrt(N)) on both fits.")
    ap.add_argument("--hold-on-present", type=int, default=0,
                    help="qualify a satellite for the cp_held seed freeze after this many "
                         "consecutive FLEET-PRESENT cycles (the population-honest presence "
                         "that admits trims), alongside the amp_snr path. The hold machinery "
                         "is the #96/#97 closure architecture -- frozen (anchor, rate, at-ref "
                         "phase) tuple, DLL owns the residual, CP_ERR at-epoch referee, "
                         "3-miss release -- but its amp_snr>=hold-snr entry rides the "
                         "coherent arc, which flickers with the deep fold (#58): measured "
                         "2026-08-28, 5 HOLD entries in a day while locked q-3.8 satellites "
                         "took per-detection seed REPLACEs (and their +-1-period flips) "
                         "un-held. fit_trusted is still required (the 2026-07-19 zombie-"
                         "cohort lesson: a birth anchor must mature before it earns "
                         "protection). A present-qualified sat also SUSTAINS on presence: "
                         "the 3-miss counter only advances when the sig path fails AND the "
                         "fleet gate reads absent. 0 (default) disables.")
    ap.add_argument("--nh-period-debounce", type=int, default=0,
                    help="require a CHANGED measured overlay period to be confirmed by this "
                         "many consecutive detections (same offset) before it may rewrite "
                         "the seed; until then the seed carries the measured fine phase "
                         "with the standing period, and ph_hist is not updated. #97: the "
                         "search's measured period toggles +-1 on strong satellites "
                         "(snr 284-413, fine phase right to ~0.5 chip), and each "
                         "single-detection flip stepped every node's seed a whole code "
                         "period -- SEEDAUDIT +-10230-chip step-back pairs, q cratering "
                         "1-2 polls, per-sat step rate ranking exactly with q churn. Not "
                         "the 2026-08-02 override this check replaced: nothing is stored "
                         "from our own correction, and a real change passes in ~2N s. "
                         "Ignored under --period-continuity correct. 0 (default) disables.")
    ap.add_argument("--nh-joint", type=str, default="off",
                    choices=["off", "vote", "apply"],
                    help="the JOINT overlay fit: the NH label is not N independent per-sat "
                         "20-way guesses -- the overlay is synchronous at the transmitters, "
                         "so at the receiver the labels differ only by the known geometric "
                         "delay plus ONE common unknown, the receiver clock mod the overlay "
                         "epoch. Every detection with a phase at its own epoch VOTES for that "
                         "common offset (continuous, chips mod the epoch -- an integer vote "
                         "splits at segment boundaries); the consensus is a circular weighted "
                         "median. 'vote' measures and logs only (NH-JOINT lines), changing "
                         "nothing. 'apply' additionally DERIVES the overlay segment from the "
                         "consensus for detections below --period-check-snr (whose measured "
                         "label is noise: G1's ran uniform over 0..19, stepping the seed a "
                         "whole code period every ~4 min, 2026-08-31) and gates the raw "
                         "det_nh fallback; strong detections keep their verified measured "
                         "label, with a disagreement alarm as the referee. Requires the "
                         "almanac (tau and the broadcast sat clock feed the prediction). "
                         "Default off.")
    ap.add_argument("--nh-joint-min-prns", type=int, default=3,
                    help="satellites that must agree (within --nh-joint-tol-chips of the "
                         "circular median) before the common offset resolves. Below this the "
                         "joint fit stays in observation and 'apply' changes nothing.")
    ap.add_argument("--nh-joint-min-snr", type=float, default=60.0,
                    help="detections below this snr do not vote (measured: snr < 60 phases "
                         "carry ~2000-chip within-period residuals -- letting them vote "
                         "dragged the on-sky consensus inlier set from 7/7 to 3/9 within "
                         "10 min on 2026-08-31); they may still be APPLIED to, which is "
                         "the point of the joint fit.")
    ap.add_argument("--nh-joint-window-s", type=float, default=600.0,
                    help="votes older than this are pruned. A resolved consensus is NOT "
                         "expired by pruning -- the receiver clock mod the epoch is a run "
                         "constant (the F-engine axis is drift-free); it is only ever "
                         "REPLACED by a newer resolution.")
    ap.add_argument("--nh-joint-tol-chips", type=float, default=1500.0,
                    help="a vote further than this from the circular median is an outlier "
                         "and does not count toward resolution. True agreement is a few "
                         "chips; the default is deliberately loose (0.15 code periods) so "
                         "the tell is the LOGGED spread, not a silent exclusion.")
    ap.add_argument("--hold-rate-source", type=str, default="none",
                    choices=["none", "dr", "dr-entry"],
                    help="#103 v2: while a seed is HELD, slew-bound its residual code rate "
                         "toward this source every cycle, anchor re-expressed at the present "
                         "(command-continuous; gate TestHoldRetagContinuity). 'dr' = the DR "
                         "clock drift (sd 6.9, cycle-to-cycle 0.4 mchips/s -- the only "
                         "measured source under the ~2 mchips/s bar a 10-min hold needs). "
                         "v1 of this idea sourced the pooled l-a estimate (+-50 mchips/s "
                         "swings) and was revert-triggered in 30 min: held sats INTEGRATE "
                         "rate noise that measurement-anchored sats never see. 'none' = "
                         "holds freeze the rate (pre-#103). 'dr-entry' (v3) = the swap at HOLD ENTRY only -- zero marginal tuple rewrites; the per-cycle 'dr' form (v2) degraded held q within minutes (suspected re-pin cost per rewrite) and stays only as the A/B arm.")
    ap.add_argument("--cp-rate-model-primary", type=int, default=0,
                    help="#103: seed the pooled-clock MODEL code rate for EVERY cp-rate fit "
                         "(the fit's rate becomes monitor-only; its position is kept). The "
                         "measured case: gps_l5's fitted rates carry ~0.006-0.03 chips/s "
                         "per-sat errors that saturate the fleet trim's 1.25-chip ceiling "
                         "in minutes and drive the escape/rebirth churn (5,773 births/"
                         "night), while model rates run ~100x cleaner post-#99. 0 = off "
                         "(the fitted rate seeds whenever the tol guard accepts it).")
    ap.add_argument("--cp-rate-model-tol", type=float, default=0.0,
                    help="reject a FITTED code rate that departs from the POOLED l-a clock by "
                         "more than this (chips/s) and seed the clock's rate instead, keeping "
                         "the fit's POSITION. The residual code rate is common-mode -- "
                         "propagate_seed feeds each satellite's own geometry forward, so what "
                         "the seed carries is the receiver clock, one number for the whole "
                         "chain (sat clock drift, af1 ~1e-12 s/s, is three orders below it). "
                         "fit_cp_rate is unbounded and its cp history is nearest-wrap "
                         "unwrapped, so one mis-wrap injects a code period into the slope: "
                         "measured 2026-08-28 on gps_l5, p90 2.16 / p99 177 / max 994 chips/s. "
                         "Only a chain with a --detectors endpoint reaches this fit at all "
                         "(17304 fits that day vs 4 on gal_e5a): all five chains run "
                         "dead-reckon, but the other four have no detector AND carry "
                         "--dr-clock-adopt, so their rate is one shared adopted clock. gps_l5 "
                         "is the clock master the others adopt from, so it cannot adopt and "
                         "its rate is whatever this fit last returned. Bounds the "
                         "DEVIATION, never the magnitude: the clock is ~0.047 chips/s "
                         "calibrated but ~3.45 uncalibrated, and an absolute ceiling would "
                         "reject the feed-forward the trim cannot live without. Stratified on "
                         "pre-fit p/noise (a fading sat makes a noisy fit AND drops out), at "
                         "matched strength 5-15 the 40 s dropout rate was 0.4%% after a clean "
                         "fit and 21.4%% after one this would reject. 0.5 is mid-plateau: "
                         "0.3-1.0 all reject 2.3%% of fits and catch 34.3%% of gps_l5 dropouts. "
                         "0 (default) disables the cross-check.")
    ap.add_argument("--cl-tracker", default=None,
                    help="L2C CL SIBLING-CHAIN mode (Mechanism A of the shared-knowledge plan; "
                         "supersedes --cl-assist's in-place lift): derive one CL pilot seed row "
                         "per CM row -- same doppler/dop-rate/carrier-trim/ref_hop (SAME carrier, "
                         "SAME 511.5 kcps chip clock), code_phase lifted by k*10230 with the "
                         "segment k pinned from absolute capture time + model range + SV clock "
                         "(t_sv = t_gpst - range/c + clk, the nh-assist convention proven to "
                         "0.01 chip by c31_convention.py) and SNAPPED to the measured CM cp -- "
                         "and POST them to THIS tracker stage's /set_seeds. The CM chain is "
                         "untouched: it stays up as the in-run control, and CL certification is "
                         "judged against it. Needs --almanac + the airspy utc0_sample0 anchor.")
    ap.add_argument("--cl-combiner", default=None,
                    help="the CL chain's combiner stage: polled each cycle so the CL-vs-CM "
                         "deep_snr comparison (the segment-pin VERIFY -- a wrong k despreads "
                         "as noise) lands in this broker's own log next to the k it verifies.")
    ap.add_argument("--xband-combiner", default=None,
                    help="S5 CROSS-BAND ASSIST (SHADOW): a SIBLING band's combiner stage "
                         "(e.g. l1_gps_combiner) whose per-sat tracked Doppler this broker "
                         "reads to predict THIS band's Doppler by the exact carrier ratio. "
                         "The satellite-motion part is geometry -- common to both bands and "
                         "scaling as f_this/f_sibling -- so `(D_sib - LO_sib)*ratio + LO_this` "
                         "predicts this band's observed Doppler; the LO terms come from each "
                         "band's own S2 fused state (the dongle LOs are INDEPENDENT -- "
                         "measured, no GPSDO common-mode -- so neither can be borrowed). "
                         "SHADOW: logs the prediction beside this band's actual acquisition "
                         "for every dual-tracked sat and accumulates the residual; nothing is "
                         "seeded from it yet. The eventual flip is RESCUE-ONLY (seed a sat "
                         "this band cannot predict itself -- cold start / stale BRDC), the "
                         "S2d lesson applied.")
    ap.add_argument("--xband-lo-dongle", default=None,
                    help="the sibling band's S2 state dongle key (e.g. gps_l1), to read its "
                         "fused LO for --xband-combiner")
    ap.add_argument("--xband-carrier-hz", type=float, default=None,
                    help="the sibling band's carrier frequency (Hz), for the Doppler ratio")
    ap.add_argument("--xband-seed", type=int, default=1,
                    help="S5b THE FLIP (default ON; a provable no-op in normal operation): "
                         "emit a SEARCH DOPPLER HINT from the cross-band prediction for a sat "
                         "the SIBLING band tracks but THIS band has NO prediction of its own "
                         "for (not in BRDC pred / no almanac). Cross-band transfers Doppler "
                         "(carrier ratio) but NOT code phase (the codes differ), so it hints "
                         "the SEARCH -- narrowing its Doppler window -- it does not seed the "
                         "tracker. RESCUE-ONLY by construction: for any sat BRDC already "
                         "predicts, the BRDC hint stands and NO cross-band hint is added, so "
                         "with fresh BRDC the cross-band hint list is EMPTY. It fires only "
                         "when BRDC is missing a sat the sibling sees (outage / deep cold "
                         "start / a band too weak to hold its own almanac lock) -- the "
                         "S2d-learned rescue-only scope, structural not just gated. 0 = pure "
                         "shadow (log the residual, emit no hints).")
    ap.add_argument("--nh-assist", action="store_true",
                    help="secondary-overlay TIME-ASSIST for a per-PRN-overlay pilot (B1C/E5a/B2a): "
                         "POST each visible sat's PREDICTED absolute overlay-chip index (from "
                         "almanac range + BeiDou/Galileo-time, one convention, the combiner "
                         "self-calibrates the constant) to the combiner's /set_nh_hint. The weak "
                         "sats that cannot win the combiner's L-way (1800 for B1C) alignment "
                         "search get the geometrically-correct alignment for free. Needs --almanac; "
                         "the combiner needs nh_assist: true. Fail-safe: a wrong hint just fails "
                         "its floor and the blind search result stands.")
    ap.add_argument("--nh-overlay-len", type=int, default=1800,
                    help="secondary-overlay length in chips (B1C pilot 1800; E5a/B2a CS100 = 100)")
    ap.add_argument("--bias-det-fresh-s", type=float, default=30.0,
                    help="clock-bias solve uses only detections FRESHER than this (seconds). "
                         "A stale detection's (meas - pred) grows at the satellite's Doppler "
                         "rate -- stale-age x dop_rate reads as a fake, GROWING clock bias "
                         "that the seeds then chase (measured: a ~90 s-stale detection walked "
                         "the bias +4 -> +68 Hz and dragged a 55-sigma tracker off the sky).")
    ap.add_argument("--almanac-epoch", type=float, default=0.0,
                    help="REPLAY BENCH ONLY: unix time of the capture's sample 0. The almanac "
                         "clock is OFFSET to this and then ADVANCES with wall time, so the "
                         "predicted sky moves as the file plays (a frozen epoch actively pulls "
                         "trackers off satellites -- measured 2026-07-27). Assumes ~realtime "
                         "replay pacing. 0 = live (use now).")
    ap.add_argument("--dead-reckon", action="store_true",
                    help="seed CODE PHASE from broadcast ephemeris (BRDC) for every visible "
                         "sat the search hasn't detected: predict the absolute transmit-time "
                         "code phase, add the receiver clock solved each cycle from the "
                         "detected sats (measured-vs-predicted circular median -- the "
                         "gnss_deadreckon_check.py bootstrap, ~100 ns), and express it in "
                         "the seed's Doppler currency. The search demotes to bootstrap "
                         "(clock solve), fallback (a detection re-anchors via the normal "
                         "seed loop) and integrity check (per-sat residuals logged); "
                         "dead-reckoning only has to land within the DLL capture range "
                         "(~0.4 chips; validated 0.10 chip rms 2026-07-13). Needs --almanac.")
    ap.add_argument("--dr-constellation", default="G", choices=("G", "E", "C"),
                    help="RINEX constellation letter for this broker's band")
    ap.add_argument("--signal-capability", default=None,
                    help="restrict ALL seeds + hints to GPS PRNs whose satellite block actually "
                         "broadcasts this signal (GPS_L1C_P -> Block III; GPS_L5_Q -> IIF+; "
                         "GPS_L2C_CM -> IIR-M+). The GENERAL form of --dr-min-prn: that numeric "
                         "cutoff only works for BDS-2's contiguous low PRNs; GPS III sats are "
                         "interspersed among IIF/IIR (4/11/14/18/20/21/23/28) so no cutoff can "
                         "express 'L1C only'. Read ONCE at startup from the live Celestrak block "
                         "names (gps_beamtrack.signal_capable_prns); on fetch failure or empty "
                         "result the filter is DISABLED with a warning (phantoms return, but the "
                         "chain lives -- better than killing L1C during a network outage). "
                         "GPS and GLONASS: for R the block marker is the Celestrak 'K' suffix on "
                         "the Uragan number, and GLO_L3OC_* -> K satellites only (the CDMA "
                         "signals are GLONASS-K's; the FDMA L1OF/L2OF are on every satellite, so "
                         "they get no filter). E/C use --tle-name-filter instead.")
    ap.add_argument("--dr-min-prn", type=int, default=None,
                    help="dead-reckon only PRNs >= this: a SIGNAL-CAPABILITY gate, not a "
                         "visibility one. Default 19 for BeiDou (B1C/B2a are BDS-3 ONLY; the "
                         "BDS-2 birds C1-C18 broadcast B1I at 1561 MHz, which is not even "
                         "inside our band), 1 otherwise. The search cannot make this mistake "
                         "-- it never detects a satellite that isn't transmitting -- but "
                         "DEAD RECKONING CAN: it seeds from the model, and the model is happy "
                         "to predict a code phase for a signal that does not exist. The "
                         "tracker then despreads noise at that phase and the cross-correlation "
                         "against real B1C satellites reports 20-60 sigma. Measured 2026-07-14: "
                         "C11/C12/C13 produced 11309 phantom rows (5.5%% of all BeiDou map "
                         "points) at a plausible-looking 25-30 dB-Hz. A model that can invent "
                         "a satellite needs a capability gate the search never needed.")
    ap.add_argument("--dr-clock-adopt-max-chips", type=float, default=0.0,
                    help="#104: refuse a SIBLING clock adoption whose step from the local "
                         "solve exceeds this many chips while the local clock is fresh "
                         "(< 300 s). The 2026-08-30 outage: gps_l5's churn ran its clock "
                         "solve to 292 chips and the unbounded adoption path relayed it "
                         "to every chain in ~2 s -- while JOINT-CLK's 5-chip bound "
                         "refused the identical values throughout. 5.0 matches that "
                         "precedent. 0 = unbounded (pre-#104).")
    ap.add_argument("--dr-min-sats", type=int, default=2,
                    help="detections needed for a receiver-clock solve (one sat is "
                         "unfalsifiable -- same reasoning as --bias-min-sats)")
    ap.add_argument("--dop-continuous", action="store_true", default=False,
                    help="DESIGN (b): update the seed Doppler every cycle (model-primary "
                         "seeding) and currency-translate cp0 on every update. VALIDATED "
                         "2026-07-19 (A/B replay legs, 17:37 capture): B1C parity-or-better "
                         "(coh duty up on every sat, C37 63->79/100, reacq 2->1) and GPS "
                         "better (RELEASE 9->0, CARRIER REACQ 6->1, coh duty up 4/6 sats). "
                         "History: the 2026-07-14 attempt measured E/C WORSE (E 42.0 -> 34.9 "
                         "dB-Hz) because the tracker's f_ref re-pin was not yet code- and "
                         "phase-continuous -- freezing the seed was double-dutying as f_ref "
                         "stabilization. The 07-14 NCO phase fold (reanchored==2) plus "
                         "max_anchor_age_s 0 completed the primitive; the fence became "
                         "moot (f_ref rate-follows the model, seed steps vanished). "
                         "run_band.sh made this the single-band default; the run_3band "
                         "transition silently dropped it (fleet ran frozen 07-18/19 -- the "
                         "release/escape churn era). Fleet default restored 2026-07-19.\n"
                         "Original rationale: update the seed Doppler EVERY cycle and "
                         "currency-translate cp0 each time, instead of freezing it and taking "
                         "a discrete step at hold_max_dop_hz. The fence was never a safety "
                         "mechanism -- it was a GRANULARITY threshold, and the "
                         "piecewise-constant-currency rule it enforced was a defence against a "
                         "NOISY, search-grid Doppler. Dead reckoning made the Doppler "
                         "model-derived and smooth, and the currency translation makes a "
                         "Doppler update cost exactly nothing (the cp0 shift cancels the "
                         "retroactive term by construction, so even jitter moves the code by "
                         "ZERO). We made the defence obsolete ourselves. Resilience now comes "
                         "from checking whether the model is RIGHT (--dr-max-eph-age-s, "
                         "--dop-max-rate-hz, the integrity residual, a railed carrier trim) "
                         "rather than whether it is MOVING. --no-dop-continuous restores the "
                         "fence.")
    ap.add_argument("--no-dop-continuous", dest="dop_continuous", action="store_false",
                    help="restore the discrete hold_max_dop_hz fence (pre-2026-07-14)")
    ap.add_argument("--nav-bits-brdc", type=int, default=0,
                    help="CONSTRUCT nav bits from BRDC for satellites that never sync "
                         "(navbit_brdc.BrdcLnavSource). DEFAULT OFF: first live trial "
                         "2026-07-25 collapsed the GPS chain to 0/14 peeling with every PRN on "
                         "`nobits` and every gain reset -- the 30 s tables make the seed POST "
                         "~14.5k numbers and the whole seed push appears to stop landing. The "
                         "bit CONTENT is validated (113820/113820 offline, hold-one-out 100%%); "
                         "it is the transport that needs sizing before this goes back on.")
    ap.add_argument("--nav-bits", type=int, default=1,
                    help="LNAV decode-and-predict from the combiner's nav_obs export "
                         "(bit_export: true), pushed as nav_bits with each seed row -- the "
                         "fused peel's sign source (P7a: ~11 -> >=26-35 dB). Self-gating: "
                         "chains whose combiner exports no nav_obs are untouched. 0 = off.")
    ap.add_argument("--nav-decoder", default="lnav", choices=("lnav", "cnav", "bcnav3"),
                    help="which decoder consumes nav_obs. lnav (default): the LNAV "
                         "decode-and-predict (GPS L1CA, the periodic-subframe future-bit source). "
                         "bcnav3: the BeiDou B2b B-CNAV3 decoder (GF(64) NB-LDPC(162,81) + CRC-24Q, "
                         "the B2b PRIMARY chain's own nav; pure decode + BRDC xcheck, no bit-pred). "
                         "cnav: the CNAV decoder (GPS L2C-CM / L5-I) -- FEC+CRC-24Q, NOT a "
                         "future-bit source (the type schedule is not fixed); it decodes live "
                         "ephemeris/messages and shadow-serves the signs of DECODED spans. Set "
                         "cnav on the L2C broker so its nav_obs (CNAV symbols) go to the right "
                         "decoder instead of churning the LNAV frame-sync forever.")
    ap.add_argument("--cnav-combiner", default=None,
                    help="AUXILIARY combiner polled purely for CNAV nav_obs, in ADDITION to "
                         "--combiner (S4). Exists because a band's CNAV can live on a chain "
                         "that is not this broker's own: at L5 the broker's combiner is the "
                         "Q PILOT (whose nav_obs are deterministic overlay predictions, which "
                         "belong to the LNAV/pilot path), while the CNAV symbols come from the "
                         "derived L5-I DATA sibling. Pointing --nav-decoder at the main "
                         "combiner cannot express that split. Symbols from here go to the CNAV "
                         "decoder regardless of --nav-decoder, and are cross-checked against "
                         "BRDC on the usual 60 s health cadence -- giving a SECOND, independent "
                         "decode of the same message set L2C already decodes, so an ephemeris "
                         "can be verified three ways (L2C vs L5 vs BRDC).")
    ap.add_argument("--inav-combiner", default=None,
                    help="AUXILIARY combiner polled for Galileo E1B I/NAV nav_obs (S5 "
                         "D-component #1), the exact analogue of --cnav-combiner: the GAL "
                         "broker's own --combiner is the E1C PILOT (deterministic overlay "
                         "signs), while the I/NAV DATA symbols come off the derived E1B "
                         "sibling chain. Symbols from here go to the InavPredictor and the "
                         "decoded ephemeris is cross-checked against BRDC on the 60 s health "
                         "cadence -- an independent E1B decode validating the Galileo "
                         "ephemeris (and the codec's ICD conventions) against the almanac.")
    ap.add_argument("--fnav-combiner", default=None,
                    help="AUXILIARY combiner polled for Galileo E5a-I F/NAV nav_obs (S5 "
                         "D-component #2), the --inav-combiner analogue on the L5 band: the "
                         "GAL/E5a broker's own --combiner is the E5a-Q PILOT (deterministic "
                         "CS100 overlay signs), while the F/NAV DATA symbols come off the "
                         "derived E5a-I sibling chain (CS20 secondary + navwipe). Symbols go "
                         "to the FnavPredictor and the decoded ephemeris is cross-checked "
                         "against BRDC on the 60 s health cadence -- an independent E5a decode "
                         "validating the Galileo ephemeris (and galileo_fnav's ICD "
                         "conventions) against the almanac, beside E1B's I/NAV decode.")
    ap.add_argument("--bcnav2-combiner", default=None,
                    help="AUXILIARY combiner polled for BeiDou B2a B-CNAV2 nav_obs (S5 "
                         "D-component #3, the FIRST non-binary FEC), the --fnav-combiner "
                         "analogue on the BDS broker: the BDS broker's own --combiner is the "
                         "B2a-P PILOT (deterministic Weil overlay signs), while the B-CNAV2 "
                         "DATA symbols come off the derived B2a-D sibling chain (CS5 secondary "
                         "+ navwipe). Symbols go to the Bcnav2Predictor (GF(64) NB-LDPC codec) "
                         "and the decoded ephemeris is cross-checked against BRDC on the 60 s "
                         "health cadence -- an independent BDS-3 decode validating the "
                         "ephemeris (and the LDPC + CNAV-eph conventions) against the almanac.")
    ap.add_argument("--bcnav1-combiner", default=None,
                    help="AUXILIARY combiner polled for BeiDou B1C B-CNAV1 nav_obs (S5 "
                         "D-component #4, the LAST), the --bcnav2-combiner analogue on the L1 "
                         "BDS broker: the broker's own --combiner is the B1C-P PILOT, while the "
                         "B-CNAV1 DATA symbols come off the derived B1C-D sibling. Symbols go to "
                         "the Bcnav1Predictor (reusing the GF(64) NB-LDPC codec, different H "
                         "matrices for SF2/SF3) and the decoded ephemeris is cross-checked "
                         "against BRDC -- completing the civil D-component set (GPS+GAL+BDS).")
    ap.add_argument("--cnav2-combiner", default=None,
                    help="AUXILIARY combiner polled for GPS L1C-D CNAV-2 nav_obs, the "
                         "--bcnav1-combiner analogue on the L1C broker: the broker's own --combiner "
                         "is the L1C-P PILOT (deterministic L1CO overlay signs), while the CNAV-2 "
                         "DATA symbols come off the derived L1C-D sibling. Symbols go to the "
                         "Cnav2Predictor (binary-LDPC systematic extract + CRC-24Q, 18 s frame) and "
                         "the decoded ephemeris is cross-checked against BRDC -- the 4th GPS-family "
                         "decode (LNAV / CNAV / CNAV-2).")
    # -- named signals (task #27 M2); see gnss_broker/signals.py --------------------------
    ap.add_argument("--signal", default=None, metavar="KEY",
                    help="name the chain instead of retyping its twelve constants: "
                         "`--signal gps_l5` sets carrier/chip rate/code length/long-code "
                         "segments+epoch/overlay length/constellation from "
                         "lib/stages/gnss/gnssSignal.hpp, which is what the trackers' "
                         "replica bank is actually built from. Every one of those flags "
                         "fails SILENTLY when mistyped (a wrong overlay length puts the "
                         "seed in a random one of N periods and errors nowhere), which is "
                         "why naming beats typing. An explicit flag that DISAGREES with "
                         "the named signal is a hard error, never a silent override. "
                         "`--signal help` lists them.")
    # -- the refactor gate (task #27 M0); see the _Transcript note at the top of this file --
    ap.add_argument("--transcript-write", default=None, metavar="FILE",
                    help="record every clock read, GET and POST to FILE (JSONL) while running "
                         "normally. Cheap and side-effect-free: safe to leave on in production "
                         "for the minute it takes to capture a golden run.")
    ap.add_argument("--transcript-read", default=None, metavar="FILE",
                    help="replay a recorded transcript instead of talking to the fleet, then "
                         "exit when it is exhausted. No network, no sleeps, deterministic "
                         "clock. The POST stream this produces is the equivalence gate -- see "
                         "scripts/gnss/broker_equiv.py.")
    return ap
