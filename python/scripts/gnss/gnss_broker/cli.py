"""The broker's command-line surface: the argument parser, the frozen tuning constants, and
the post-parse checks and resolutions of single flags (--joint-consume, --telem-gather).

None of them depends on anything in the broker, so they live here rather than in
gps_distributed_broker.main().

`--help` prints the broker's docstring, not this one: build_parser() takes its description
from the caller.

@author Keith Vanderlinde
"""

import argparse


# ── Frozen tuning ──────────────────────────────────────────────────────────────────────
# Constants, not defaults: gps_distributed_broker.main() sets each one as args.<name> after the
# parse, so use sites read it like a flag, but no command line or config can change it. Each
# entry says what the constant does and, where that matters, why it has its value.
#
# To change one, edit it here in a commit that says why; a change the replay exercises moves
# the scripts/gnss/broker_equiv.py goldens, which must then be re-blessed. A flag that nothing
# sets in production is retired by moving it here with every golden still EQUIVALENT, and
# comes back as a flag only with the A/B that needs it.
_FROZEN = dict(
    # EMA weight per solve of the clock-frequency bias that centres the search hints (and the seeds
    # under --seed-bias-source ema). Averaging ~20 solves dithers out the quantisation of the
    # search's Doppler grid.
    bias_alpha=0.05,
    # Carrier-rate residuals (fits.rate_residuals) further than this from the median across PRNs are
    # dropped before the amp_snr-weighted consensus. The median comes first because a wrong-bin
    # outlier is arbitrarily far and would drag any mean. <= 0 disables the clip.
    carrier_rate_clip_hz=25.0,
    # A rate sample more than this many emit periods (x1.5 jitter margin) after the PRN's previous
    # one is a tracker re-anchor, not a measurement: the PRN is re-baselined and the sample skipped.
    # The emit period is derived from the stream, as the smallest positive pow_hop step seen; a step
    # back (an F-engine re-base restarts the counter) is a re-anchor too.
    carrier_rate_max_gap=2.0,
    # A rate residual that moved more than this (Hz) since the PRN's previous sample is an f_ref
    # re-pin (the tracker's f_ref follows the seed's Doppler), which no window gap marks:
    # re-baseline and skip. Contiguous emits step under 1 Hz and a re-pin by several Hz, so 3 Hz
    # separates them.
    carrier_rate_max_step=3.0,
    # Served C/N0 (combdll.prompt_cn0): a PRN is served only when its mean debiased prompt power
    # over every gathered record is this many standard errors above zero (sig_inc). Under the null a
    # record's prompt power is one complex dof, so a probe's statistic is N(0,1) and 5 admits
    # nothing that is not there; at 384 records the weakest servable satellite is ~14 dB-Hz, below
    # anything the loop holds.
    cn0_prompt_min_sig=5.0,
    # Bar (Hz) for the CLOCK DRIFT ALARM log: the bias EMA departing the calibration by more than
    # this, widened by sqrt(5/n) below 5 satellites. A stale-rescue re-solve that moves the
    # calibration by more than this logs CLOCK BIAS RECALIBRATED. A GPSDO unlock or thermal event is
    # hardware news to report, not to absorb. The calibration is set only by a stale-rescue re-solve
    # (bias_stale_s, clock_bias_cal_min_sats).
    clock_bias_alarm_hz=10.0,
    # Bound (ppm) on each satellite's (l-a) code-rate sample and on their pooled median. A noisy or
    # unwrap-blown slope fit is an outlier a few-satellite median cannot reject, and the seeded code
    # rate must stay ~0.1 ppm stable for the deep integration to stay on peak.
    code_bias_max=3.0,
    # Fleet coherent (fleet.fleet_coherent): a fleet deep_snr is published as a detection only above
    # this multiple of the null floor, re-measured every cycle by shuffling each instance's records
    # (real amplitudes, no common phase). The right bar depends on fleet size and window length, so
    # the floor is measured, not fixed; real detections sit far above the null's tail, and 3x falls
    # in that gap.
    coh_floor_margin=3.0,
    # Fleet coherent: instances that must see a PRN before it is combined; below this the PRN keeps
    # its single-instance coherent numbers.
    coh_min_instances=3,
    # While no BRDC ephemeris is loaded (none fetched, cached or parseable), keep the dead-reckon
    # predict alive on this broker's own decoded ephemeris (decoded_eph.py, which reuses each
    # decoder's BRDC-validated propagator). sat_clk_s is the broadcast polynomial plus the
    # relativistic term for Galileo I/NAV and F/NAV and GPS LNAV, with no group-delay term, and None
    # elsewhere. 0 lets the predict go dark.
    decoded_eph_fallback=1,
    # Fleet DLL: an instance's measurement joins the sum only if its window is within this many
    # seconds of the newest instance's (converted once to an integer hop count). Instances free-run,
    # so their emit phases drift apart without bound: size this for that drift, not for a snapshot.
    # The code moves ~0.003 chips/s, so 5 s costs ~0.017 chips.
    dll_hop_window_s=5.0,
    # Fleet DLL: the q_floor published when fewer than 3 noise probes report and the noise q cannot
    # be measured. Such a chain admits nothing (present_gate UNANCHORED), so this only keeps the
    # field numeric; 2.2 is the single-instance lock bar.
    dll_quality_min=2.2,
    # Reject a fitted Doppler rate beyond this (Hz/s) and keep the model's. Physical: GPS Doppler
    # acceleration peaks near 0.94 Hz/s at L1 and ~0.70 at L5, and 0.8 clears every CHORD band. A
    # noise fit beyond it would add curvature error.
    dop_rate_max=0.8,
    # Detections needed before a measured Doppler rate replaces the model's; below this the model is
    # the better estimate.
    dop_rate_min_pts=4,
    # Baseline (s) the Doppler-rate fit must span. The slope error is sigma/(T*sqrt(N/12)), so at
    # ~1.5 Hz of detection scatter 4 points over 44 s give ~0.06 Hz/s; a shorter baseline fits
    # noise.
    dop_rate_min_span_s=30.0,
    # Reject a fitted Doppler rate that differs from the model's by more than this (Hz/s) and keep
    # the model. dop_rate_max bounds only the magnitude, so a wrong-signed or half-size fit passes
    # it; 0.12 admits real fit noise and rejects a fit that is simply wrong.
    dop_rate_model_tol=0.12,
    # EMA weight of each in-bound update of the solved receiver clock (chips, wrap-aware). Between
    # solves the clock propagates at the drift: the measured drift EMA, else the f_chip*(l-a) model.
    dr_clock_alpha=0.2,
    # A detected satellite whose ephemeris toe is more than this old (4 h) is demoted to untrusted
    # after 3 consecutive cycles: the dead-reckon seeder skips it and its seeds stop taking the BRDC
    # Doppler (falling back as --seed-doppler says).
    dr_max_eph_age_s=14400.0,
    # Demote a detected satellite to untrusted (as for dr_max_eph_age_s, except that
    # --dr-doppler-ignores-integrity keeps its BRDC Doppler) when its dead-reckon integrity residual
    # (search code phase against the model, solved clock removed; normally +-0.2 chips) exceeds this
    # on 3 consecutive cycles; trust returns once it is back inside half the bar. This detects a
    # model that is wrong, which no bound on Doppler motion can.
    dr_max_integrity_chips=1.0,
    # Drop a satellite from the receiver-clock solve when its offset d_i jumps by more than this
    # between consecutive cycles. d_i = clk + b_i and both are stable (the clock drifts
    # ~4e-4 chips/s, b_i is a few chips), so a real satellite moves far less than this while a noise
    # or cross-correlation track scatters over the whole code. One bad track is dropped instead of
    # the whole solve being refused (dr_max_solve_mad_chips), unless dropping would leave fewer than
    # dr_min_sats.
    dr_max_off_jump_chips=100.0,
    # Skip the dr_max_off_jump_chips test when the satellite's previous offset is older than this:
    # across a long gap the clock can really have moved.
    dr_off_jump_max_age_s=120.0,
    # After the clock solve has been refused (dr_max_solve_mad_chips) continuously for this long,
    # clear an unconfirmed clock and snap it to the refused median in the same cycle: a re-roll,
    # not a measurement. Only an accepted solve moves an unconfirmed clock, so without this a
    # persistent refusal would latch it. A confirmed clock is never re-rolled; it re-pins on
    # agreeing solves (dr_clock_repin_solves).
    dr_solve_refused_rebootstrap_s=300.0,
    # Combined |A| below which a seeded PRN counts against its coast budget; used only when no
    # status row reports a significance (sig_of), since |A| is biased by the noise floor.
    drop_amplitude=0.3,
    # Cadence (s) of the --element-archive-dir appends. The gains move on the calibration EMA
    # (~1 s), but the beam traces the archive feeds move on the transit timescale, so 60 s loses
    # nothing downstream and keeps the file to ~MB/day.
    element_archive_every_s=60.0,
    # Veto a hold escape while the held track's incoherent amp_snr exceeds this absolute value. The
    # false lobes the escape exists to catch sit ~12 dB below the prompt, so a hold despreading at
    # full amplitude cannot be on one. amp_snr only: deep_snr stays coherent on a wrong lobe.
    escape_amp_veto=100.0,
    # Span (s) of cp-fit history required before the fit is trusted, by both the escape referee and
    # hold admission. 30 s makes the code-Doppler quadratic observable on every chain.
    fit_maturity_span_s=30.0,
    # Keep a PRN armed to the C++ fleet loop for this long after it was last present. The +-1-chip
    # clock breathing (~19 s period) sweeps the tap off peak and presence with it, so without a hold
    # the loop would be disarmed by the error it exists to fix. A satellite that has set is released
    # within ~1.5 min, and while held its discriminator is zero-mean noise that the leak reverts.
    fleet_trim_hold_s=90.0,
    # Fold-independent lock: a satellite also counts as locked (for the coast/drop gate and the
    # dead-reckon re-pin) when its fleet prompt power over the noise probes' median (p_pow / p_med)
    # reaches this, whatever the deep fold says. The fold can fail to certify on a satellite the
    # despread holds continuously; sig then falls back to amp_snr, routinely below lock_snr, and an
    # unlocked reading re-pins the seed on the dr_repin_s timer, whose phase step costs the fold
    # again. This is a power ratio (1.0 = noise) and lock_snr a debiased sigma, so the two are
    # OR-ed, each against its own bar.
    lock_prompt_hold=3.0,
    # Exit gate, with hysteresis above model_primacy_p95: a flipped PRN whose 10-min MINNOV p95
    # exceeds this (chips) goes back to the search, which re-anchors it on its next detection. A
    # walking model crosses this in minutes.
    model_primacy_exit_p95=3.0,
    # Enter gate: MINNOV samples needed in the 10-min window before its p95 is judged; a p95 needs
    # a population before it means anything.
    model_primacy_min_n=30,
    # Enter gate: a PRN may flip to model-primary (--model-primacy-max) when its 10-min MINNOV p95
    # is below this (chips) over at least model_primacy_min_n samples. Healthy established
    # satellites read 1.4-1.9.
    model_primacy_p95=2.0,
    # Exit gate: a flipped PRN with no fresh detection for this long (faded, no longer detected,
    # joint row evicted) also goes back to the search: nothing may hold a seed unrefereed.
    model_primacy_starve_s=600.0,
    # Distinct detections (a new ref_hop per PRN) pooled before the nh offset is trusted and hints
    # are sent. The offset is common to all satellites, so the pool fills quickly; counting cycles
    # instead would measure uptime, not evidence.
    nh_hint_min_samples=6,
    # q at or above this counts as a locked satellite in the q stall duty (the working lock bar).
    q_stall_bar=2.2,
    # Log the q STALL notice when the trailing duty falls below this fraction of the chain's best.
    # Loose on purpose: duty varies a lot with transits, and a tighter bar would cry wolf.
    q_stall_frac=0.6,
    # Do not judge a chain whose best duty is below this: bds_b2b sits near 0.2 for structural
    # nav-bit reasons and has no headroom to fall from.
    q_stall_min_best=0.25,
    # Rate limit (s) on the q STALL notice.
    q_stall_notice_s=300.0,
    # The q stall guard (log only): trailing window (s) over which the chain's median q duty is
    # judged against its own best this session. Long enough that a transit or one bad poll cannot
    # trip it, short enough to catch a degradation in minutes.
    q_stall_window=600.0,
    # Seconds between RF-health polls (--rf-stats-endpoints), on their own clock so a diagnostic
    # never paces the control loop. The node refreshes every ~10 s, so a faster poll re-reads the
    # same pass.
    rf_stats_poll_s=30.0,
    # Per-update gain of b_sat (SatBiasFilter), the per-satellite path bias fed by the fleet
    # phase-slope tau (--spectrum-endpoints); 0.02 at the ~2 s cadence is a ~100 s time constant.
    # 0 freezes b while the fit and its publishing keep running.
    bsat_gain=0.02,
    # Hard gate on the combiner's rate-search peak/median (deep_rate_full_q) before a rate residual
    # is believed: signal ~18-22, noise ~3-6, measured on the capped search's deep_rate_q; the
    # uncapped deep_rate_full_q this gate reads has not been re-measured. A weak satellite does not
    # merely scatter, it lands on the wrong spectral bin, tens of Hz out.
    carrier_rate_min_q=10.0,
    # Fleet DLL deep gate (--dll-deep-gate): a PRN in the set is present when deep_snr >= this x
    # deep_floor. The floor is the combiner's own rectification level, so this is a detection
    # significance, not a tuned constant.
    dll_deep_gate_margin=3.0,
    # Fleet DLL presence: q must clear the noise probes' median q plus this many MAD-sigma (at least
    # +0.05), and prompt power this multiple of the probes' median (at least 1x). q = 2P/(E+L) is a
    # peak-sharpness metric: 1.0 with no peak, 4.0 for a clean lock at 0.5-chip spacing. Summing
    # instances does not raise it but shrinks its spread as 1/sqrt(K), so no constant bar suits
    # every fleet size; the floor is measured each cycle from the probes, which are signal-free by
    # construction.
    dll_quality_sigma=3.0,
    # Reject a receiver-clock drift estimate beyond this (chips/s). The estimate differences two
    # clock solves, so a node or F-engine restart enters as motion that never happened, and the 0.05
    # drift EMA would bleed it off over ~10 min while sweeping every model-primary seed off peak.
    # The true drift is ~4e-4 chips/s on this GPS-disciplined clock, so 1.0 rejects nothing real.
    dr_max_drift_chips_s=1.0,
    # Dead-reckon cadence (s): clock solve, integrity and re-pin checks.
    dr_refresh_s=2.0,
    # Re-anchor an unlocked dead-reckoned seed from the model this often: fresh code phase, Doppler
    # and rate together. Its Doppler is frozen between pins, so this also bounds the staleness: 10 s
    # at the ~0.6 Hz/s peak MEO Doppler rate is 6 Hz, inside every band's fence.
    dr_repin_s=10.0,
    # Per-event ceiling (chips) on the dead-reckon slew: each dead-reckon refresh re-anchors a
    # locked model-primary seed a quarter of the way toward the model, capped at this. 0 disables
    # slewing: the seed is still re-anchored with fresh Doppler and rate, but its phase stays where
    # the loop put it.
    dr_slew_cap=0.05,
    # Minimum seconds between /get_elements polls per chain (--element-poll; 0 = every cycle). The
    # gain table feeds only the display and the beam archive (itself throttled to
    # element_archive_every_s), and the beam traces move on the transit timescale, so polling every
    # cycle costs a GET per instance for nothing.
    element_poll_every_s=20.0,
    # Leak (1/s) published to the C++ fleet loop. Sets both the noise mean-reversion time
    # (1/leak ~ 8 s) and the reachable-correction ceiling, trim_max = 0.25 x bandwidth / leak:
    # 2.9 chips at the --fleet-trim-bandwidth default of 1.4, 5.2 at the chains' 2.5, against the
    # +-3-chip clamp trimarm posts.
    fleet_trim_leak_per_s=0.12,
    # Log when an instance serves rows whose newest pow_hop has not advanced for this many seconds
    # while most of the fleet advances; the hop comes from the fleet DLL poll, so this costs no
    # extra polling. fe_axis_stale_s watches only the --combiner's hop, so it cannot see any other
    # instance wedge. Keyed on the counter, not on reachability: a wedged instance
    # still answers 200 with well-formed rows. Healthy advance is millions of hops per 30 s and a
    # wedged one exactly 0, so 90 s only has to ride out a poll race.
    instance_stall_s=90.0,
    # 'clk' consumer (--joint-consume): refuse the joint clock if it disagrees with the legacy
    # median by more than this. The median churns by +-1-2 chips, so 5 keeps the plausible envelope
    # while refusing a wrap alias or a diverged filter. In reference chips
    # (receiver.JOINT_REF_RATE_HZ, 10.23 Mcps) on every chain, so 5 = 489 ns everywhere; a chain at
    # another chip rate converts before comparing (deadreckon.dr_joint_clk).
    joint_clk_max_chips=5.0,
    # 'clk' consumer: refuse a joint clock whose 1-sigma exceeds this (reference chips). P grows
    # while the state is unfed, so this one gate covers both estimator health and staleness; a
    # healthy state reads under 0.1.
    joint_clk_max_sigma=0.5,
    # 'clk' consumer step guard: a joint clock more than joint_clk_max_step_chips (reference
    # chips) from the one this chain last adopted is refused until it has held the new level, to the
    # same tolerance, for joint_clk_step_hold_s. A transit that blinds the feeder band walks the
    # joint clock for under a minute, inside the bounds above. The receiver clock moves under
    # 0.003 chips/s, so a slow walk never meets the guard and a real step is taken one hold late. It
    # tests the joint clock, not the delta: the legacy median's churn steps the delta while the
    # joint stands still.
    joint_clk_max_step_chips=0.3,
    joint_clk_step_hold_s=30.0,
    # A joint consumer refuses a state carrying fewer satellites than this: in a thin state one
    # satellite's bias leaks into the clock at ~1/N.
    joint_min_sats=4,
    # How long each P2c coast (--joint-p2c-rotate) runs. Must stay below the joint state's max_age_s
    # (900 s): a withheld satellite is not fed, so it would be evicted and the coast would end with
    # nothing measured.
    joint_p2c_hold_s=600.0,
    # Recently tested satellites the P2c rotation skips, so the sample spans the fleet instead of
    # re-testing the strongest satellite.
    joint_p2c_skip=4,
    # Measurement sigma (chips) of the joint solve's code-phase feed. The search's per-detection
    # noise runs 0.03-0.5 chips; 0.3 is not tuned against the answer, and the state's own covariance
    # reports whether it fits.
    joint_sigma=0.3,
    # Detection significance (sigma above noise, sig_of) at which a satellite counts as locked: the
    # primary lock metric, noise-relative where |A| is noise-biased. Noise sits at ~1, a real lock
    # well above 3.
    lock_snr=3.0,
    # Drop nh-offset samples older than this, and stop hinting when fewer than nh_hint_min_samples
    # remain. The hint must be able to expire: it narrows the search to +-nh_hint_span (a search
    # stage setting) of the 20 overlay phases, so a wrong offset points the scan away from the
    # signal and no detection can arrive to correct it. 600 s spans several per-PRN revisits, so a
    # healthy fleet always has fresh samples and only a starved one widens.
    nh_hint_max_age_s=600.0,
    # Detections below this SNR enter the period-continuity history only as a PRN's first entry, and
    # their period disagreements log as 'weak det' rather than SOURCE PERIOD DISAGREES; the
    # --nh-joint fit splits strong from weak labels here too. Above ~60 the within-period phase is
    # self-consistent to a few chips across a 400 s gap; below it the residuals are ~2000 chips,
    # i.e. noise.
    period_check_snr=60.0,
    # --rrate-command: command a satellite's carrier only while its rrate row's 1-sigma (m/s) is
    # below this; 0.5 m/s is ~2 Hz at 1176 MHz. An unmeasured row reads sigma = inf, so this also
    # gates births: a satellite is commanded only once its row has converged.
    rrate_cmd_max_sigma=0.5,
    # How long a fine (phase-feed) or KCOH acceptance keeps a satellite in the phase-governed
    # regime, where its coarse measurements are deweighted by --rrate-coarse-deweight. On expiry the
    # coarse feed returns to full weight, the drop-back an FLL/PLL handoff needs when the fine gate
    # stops passing. Well above the fine cadence (~1/min) so ordinary gaps do not flap the regime,
    # and well under the time an unsupervised row could drift far.
    rrate_fine_hold_s=240.0,
    # Minimum KCOH fold significance for --rrate-kcoh-feed: below this the fold has not caught the
    # satellite and its residual is noise (probes sit at sig ~1). Each fed measurement takes sigma =
    # min(0.3, max(0.03, 2/sqrt(sig))) Hz.
    rrate_kcoh_min_sig=20.0,
    # Base sigma (Hz) of the fine phase-step feed (--rrate-phase-feed) at a 1-poll span. Not
    # smaller: res_cycles telescopes (consecutive differences share a record), so the span noise is
    # sqrt(2)*sigma_phi/2pi ~ 0.07 cycles whatever the span, ~35-50 mHz over a 2 s poll. When the
    # combiner does not serve the applied command, the feed inflates this by the command's motion
    # over the span, sqrt(sigma^2 + (0.5*dcmd)^2); under --rrate-phase-span-s it scales it by
    # poll/span and adds the span-mean's staleness.
    rrate_phase_sigma=0.05,
    # Seconds a visible, search-seeded satellite is coasted (seed held, Doppler forecast forward)
    # through a signal dropout before it is dropped, counted in polls of --interval; a lock reading
    # resets it. Dead-reckoned seeds are model-owned and never dropped this way.
    coast_budget=30.0,
    # Hold-on-lock gate (sig_of_last): at or above this a tracked PRN's code-phase anchor is frozen
    # and the DLL owns the sub-chip residual; a new hold also needs a mature cp fit
    # (fit_maturity_span_s). The metric is amp_snr, or the larger of amp_snr and deep_snr when the
    # combiner certifies the deep (coherence_s > 0): an uncertified deep_snr sits at the nav-wipe
    # rectification floor (~7) off peak and would freeze bad anchors.
    hold_snr=8.0,
    # Hold escape: release a held seed and its DLL trim, and re-anchor on the search, when the
    # detection's code phase disagrees with the held track (propagated phase + trim) by more than
    # this (chips) on 5 consecutive sign-consistent detections whose median over the last 9 also
    # exceeds it, given a mature fit and neither veto. Also the bar of the integrity veto. This is
    # the DLL's capture half-range: a sharp-ACF (BOC) power discriminator has stable false
    # equilibria ~0.75 chips out (prompt -12 dB) that a hold would otherwise servo forever.
    hold_max_cp_err=0.4,
    # The clock-bias solve uses only detections fresher than this (s). A stale detection's
    # (meas - pred) grows at the satellite's Doppler rate and reads as a growing clock bias that the
    # seeds then chase.
    bias_det_fresh_s=30.0,
    # Detected satellites needed before a cycle's median residual may update the clock-frequency
    # bias, and so narrow the search. One satellite's residual is unfalsifiable: a bad prediction (a
    # mismapped TLE, a cross-correlation on a non-transmitting satellite) would become the bias and
    # push every other hint out of the narrow window, a deadlock no second satellite can break.
    bias_min_sats=2,
    # Multiplies the predicted Doppler; -1 if the receiver's convention is inverted.
    doppler_sign=1.0,
    # TLE file or URL for the TLE almanac (--almanac-source tle, and the fallback when BRDC is
    # unavailable at startup) and the block-name capability gate; None means Celestrak gps-ops
    # (gps_beamtrack.DEFAULT_TLE_URL).
    tle=None,
    # Stale-bias rescue: once the solved bias EMA has gone this long without a multi-satellite
    # measurement, widen the search margins and snap to the next multi-satellite median. A bias
    # latched away from truth (a GPSDO unlock walking the EMA) otherwise centres narrow hints where
    # nothing can be detected to correct it. The held value still centres the wide hints (and the
    # seeds under --seed-bias-source ema), so a chain with a sparse sky loses nothing. 0 disables.
    bias_stale_s=300.0,
    # Rate limit (s) on the CLOCK DRIFT ALARM: an alarm repeating faster than it can be investigated
    # buries the log, a real alarm included.
    clock_bias_alarm_every_s=600.0,
    # Satellites a stale-rescue re-solve needs before it may set the calibration CLOCK DRIFT ALARM
    # compares against; below this the old calibration is held, and the log says so. A median of a
    # few satellites is noise (sd ~13 Hz), and a stale reference is better than a noise-derived one.
    # 0 disables the guard.
    clock_bias_cal_min_sats=5,
    # Log when the offset between this host's wall clock and the GPS-disciplined F-engine axis jumps
    # by more than this (s). Log only, never a control input. The orbit model is evaluated on the
    # dead-reckon time base (deadreckon.dr_eph_epoch), which under --dr-fengine-axis is the
    # F-engine axis, so a wall step then moves no model seed; the jump is the first sign of a host
    # clock problem. The observable F-engine 'now' is the newest telemetry pow_hop, which trails
    # the sky by the gather/serve latency (~100 ms, tens of ms of jitter), so this sees only steps
    # well above that.
    clock_step_guard_s=0.5,
    # Must match the search stage's code_doppler_sign: the sign of the Doppler-dependent cp0
    # back-reference the search applies. Every cp0/phase conversion here uses it
    # (cp_to_seed_currency, fits.dr_cp0, fits.tracker_phase_at).
    code_doppler_sign=1.0,
    # Fleet coherent: common records (hops) required per PRN.
    coh_min_records=32,
    # Warn when the DCB product in use is older than this many days (its epoch from the filename,
    # not the file mtime). The CAS rapid product lands ~5 days late, so 5-6 days is normal and past
    # ~10 the fetch has stopped working. Age is the only early sign: when downloads fail,
    # gnss_dcb.fetch_dcb keeps serving the newest cached file for up to 14 days before it returns
    # None. 0 = never warn about age.
    dcb_max_age_days=10.0,
    # Suppress D3 (the handover-sawtooth detector) while any satellite the model carries (every
    # constellation, health ignored) is within this many degrees of boresight; 0 = off. A
    # near-boresight satellite rails the shared quantiser, so satellites drop on every chain at
    # once, which a per-satellite detector reads as many independent faults. Suppression is the
    # conservative direction for a read-only detector: it costs recall during a transit and keeps
    # the base rate from being mostly geometry.
    detector_transit_veto_deg=5.0,
    # Leak per update of the DLL integrator on the single-combiner path (0 = pure integrator): the
    # trim mean-reverts so discriminator noise cannot random-walk it to the clamp. Steady-state gain
    # is dll_gain/dll_leak, with ~1/leak windows of smoothing.
    dll_leak=0.05,
    # Leak on the fleet DLL path, where `present` already confirms signal. Kept equal to dll_leak: a
    # railed discriminator (|disc| ~ 0.9, common here) saturates tau at 0.25 chips and carries only
    # its sign, so the integrator pushes one way and the leak is what bounds the excursion
    # (gain/leak x 0.25 = 1.25 chips at --dll-gain 0.25). A smaller leak lets it drive satellites
    # off peak.
    dll_leak_present=0.05,
    # Fleet DLL: instances that must report the same window before their sum is used; below this the
    # PRN falls back to the single --combiner discriminator, so a partly-down fleet degrades instead
    # of stalling. Also the minimum sender count of the comb DLL and the telemetry C/N0 estimators.
    dll_min_instances=2,
    # Hold the receiver-clock solve and its drift EMA while the newest telemetry hop has not
    # advanced for this many seconds, and drop raw_prev so no pair of solves is differenced across
    # the gap. With the F-engine stopped the detections freeze while the orbit model moves with the
    # wall clock, so the solve would read the sky's motion as clock drift, and a drift absorbed that
    # way walks every seed off peak where nothing re-measures it. The axis advances every ~2 s
    # cycle, so 6 s is three missed advances. This is the control for what fe_axis_stale_s only
    # logs.
    dr_clock_freeze_s=6.0,
    # Consecutive solves of new detections that must agree, within dr_clock_step_max_chips of the
    # first, before the receiver clock takes a step larger than that bound, as a snap (a re-pin). A
    # confirmed clock counts only solves of dr_update_min_sats, an unconfirmed one any solve of
    # dr_min_sats. Real solves agree cycle to cycle to ~1.5 chips while noise medians land uniformly
    # over the code, so 3 leaves ~(10/10230)^2 for a chance agreement. Values below 1 count as 1.
    dr_clock_repin_solves=3,
    # Largest step (this chain's chips, solve against the propagated clock) an update of an
    # established receiver clock may take. A larger one is refused and the clock held at zero rate
    # unless dr_clock_repin_solves agreeing solves confirm it; it is then a re-pin, taken whole,
    # which is how a bad bootstrap is corrected. A confirmed clock is also held, never re-rolled,
    # through MAD refusals. Normal solves step under ~1.5 chips; 5.0 matches joint_clk_max_chips
    # (489 ns at 10.23 Mcps).
    dr_clock_step_max_chips=5.0,
    # Freeze this chain's solved receiver clock while any satellite the model carries (every
    # constellation, the D3 veto's geometry) is within this many degrees of boresight: no solve is
    # applied, no drift is differenced, the MAD re-bootstrap clock stops, and the clock is held at
    # zero rate and contributed every cycle. In a transit the search reports noise code phases just
    # over its bar, and their median is not the clock; the clock itself is a GPS-disciplined
    # constant. Only a clock confirmed by an in-bound update or a re-pin freezes (never a prime or
    # an unconfirmed bootstrap), so a chain without detections never does.
    dr_clock_transit_freeze_deg=5.0,
    # Withhold dead-reckon seeding while the receiver clock is still the --dr-clock-chips prime
    # rather than a measurement (0 = never wait). A seed born without a clock steps by the whole
    # clock at its first re-birth after the bootstrap, moving the whole fleet at once. It is a
    # timeout, not a precondition: a detector-less chain that adopts no clock never measures one
    # (the prime exists for it), so after this many seconds seeding proceeds on the prime and says
    # so.
    dr_clock_wait_s=30.0,
    # A receiver-clock drift not re-measured (or re-adopted) for this many seconds expires back to
    # the f_chip*(l-a) model. Only consecutive raw solves refresh the drift EMA, and a wrong drift
    # is what prevents them (it walks the seeds off peak), so a long-unrefreshed value is more
    # likely unmeasurable than steady.
    dr_drift_max_age_s=600.0,
    # Refuse the receiver-clock solve when the per-satellite offsets scatter by more than this (MAD,
    # chips): a circular median over dr_min_sats satellites is a measurement only if its inputs
    # agree. Real offsets cluster within ~+-10 chips, while uniform noise over a 10230-chip code has
    # a MAD of ~2557, so this is not a tuning knob. A refused solve moves nothing (a confirmed clock
    # is held), which beats a wrong clock: that acquires nothing and also poisons the search hints
    # that would recover it.
    dr_max_solve_mad_chips=100.0,
    # Detections needed for a receiver-clock solve; one satellite is unfalsifiable, as for
    # bias_min_sats.
    dr_min_sats=2,
    # Satellites a solve needs to move an established receiver clock; the first solve after a start,
    # a prime or a forced re-bootstrap needs only dr_min_sats, so a cold chain is never left
    # unseeded. Below it the clock is held at zero rate and still contributed. A 2-3 satellite
    # median follows its members' biases (+-5 chips), and is where two corrupted detections that
    # agree become a clock.
    dr_update_min_sats=4,
    # Minimum seconds between runs of the telemetry-walk estimators per chain: KCOH always, and
    # prompt C/N0 unless it reads the gather's reduced series (--comb-taps-cpp >= 2 with
    # --fleet-trim-url). The last values are served between runs. Each walk is pure Python over
    # ~1500 record decodes; every cycle on five chains it starves the telemetry reader, which then
    # misses the gather's 200 ms delivery deadline.
    estimator_every_s=15.0,
    # Log when the newest pow_hop in the --combiner status, which t_now_abs is built from, has not
    # advanced for this many seconds; 0 disables. Log only: dr_clock_freeze_s holds the clock solve
    # meanwhile. A frozen time base is an upstream instance stall, not a tracking fault.
    fe_axis_stale_s=30.0,
    # Accepted joint-state updates a satellite needs before it counts as established: before the P2c
    # rotation (--joint-p2c-rotate) may withhold it, and before its MINNOV is recorded. Withholding
    # a satellite never born into the state tests nothing. A search-anchored satellite is fed only
    # on a detection above joint_min_snr, so the count accrues slowly; 50 is enough to establish a
    # bias (sigma_b0 10 against a 0.3-chip measurement converges in tens of updates).
    joint_mask_after=50,
    # The joint 'rate' consumer (--joint-consume) refuses a |clk_rate| beyond this (ppm) and keeps
    # the l-a EMA. CHORD's reference is GPS-disciplined (code-rate offset ~4e-5 ppm), so 0.01 is
    # ~250x headroom and still refuses a runaway such as -0.028 ppm.
    joint_max_rate_ppm=0.01,
    # Minimum coherent fraction for a residual to enter the joint solve. Nearly redundant with
    # joint_min_deep_snr, since both measure the same physics; kept as a cheap independent guard.
    joint_min_coh_frac=0.3,
    # Minimum tracker significance (deep_snr, floor-cleared: coherence_s > 0) for a residual to
    # enter the joint solve. joint_min_snr gates on the search SNR carried in `best`, which latches
    # until the search next scans that PRN, so a satellite that has left the beam keeps its last
    # good SNR while its tracker output is noise. Noise rows sit at deep_snr 2-3 / coh_frac ~0.2 and
    # signal at 24+ / 0.9+; 10 is in the gap.
    joint_min_deep_snr=10.0,
    # Minimum detection SNR for a residual to enter the joint solve (and for MINNOV), equal to
    # period_check_snr: below it a detection's phase is noise with ~2000-chip residuals, and one
    # such detection can walk the state's clock rate away.
    joint_min_snr=60.0,
    # 'slew' consumer (--joint-consume): refuse the joint slew target if it disagrees with the
    # legacy target (clock EMA + SatBiasFilter) by more than this. Both describe the same receiver
    # and b_sat runs +-3-6 chips, so a larger gap means one estimator is broken. In reference
    # chips (5 = 489 ns at 10.23 Mcps), the same time on every chain, like
    # joint_clk_max_chips.
    joint_slew_max_chips=5.0,
    # Treat a satellite as locked for the dead-reckon slew/re-birth decision when its fleet
    # discriminator q is at least this (0 = ignore q). q is the metric lock is judged on everywhere
    # else, with 2.2 the working bar. amp_snr, deep_snr and the prompt ratio can all read below
    # their bars on a tracking satellite, and a re-birth then steps the seed outside the +-1-chip
    # correlation triangle, where the DLL has no gradient to recover.
    lock_q=2.2,
    # --nh-joint: satellites whose votes must agree (within nh_joint_tol_chips of the circular
    # weighted median, and carrying at least 60% of the vote weight) before the common offset
    # resolves; until then 'apply' changes nothing.
    nh_joint_min_prns=3,
    # --nh-joint: detections below this SNR do not vote (their phases carry ~2000-chip within-period
    # residuals and drag the consensus off its inliers); they may still be applied to, which is the
    # point of the joint fit.
    nh_joint_min_snr=60.0,
    # A vote further than this from the circular weighted median is an outlier and does not count
    # toward resolution. True agreement is a few chips; the bar is loose on purpose (~0.15 code
    # periods at L5) so disagreement shows in the logged spread rather than as a silent exclusion.
    nh_joint_tol_chips=1500.0,
    # Votes older than this are pruned. Pruning never expires a resolved consensus: the receiver
    # clock mod the epoch is a run constant (the F-engine axis is drift-free), so a consensus is
    # only replaced by a newer resolution.
    nh_joint_window_s=600.0,
    # --presence-admit-displaced: the deep-fold evidence bar, as a multiple of the combiner's
    # deep_floor. Prompt power is suppressed by exactly the offset being rescued. deep_snr falls
    # with it too (a coherent fold of the same prompt, carrier rate and phase searched, code not),
    # but its ~10x integration gain keeps detecting a satellite a fraction of a chip off-peak;
    # beyond ~1 chip it is blind too. Prompt power over the probe floor is accepted as an
    # alternative; the pedestal and offset tests are what refuse a centred noise realisation.
    presence_disp_deep_margin=3.0,
    # --presence-admit-displaced: max |fitted offset|, chips (fleet.epl_decompose). The fit assumes
    # taps at +-0.5 chips on a triangular correlation, where the far tap leaves the triangle at
    # |offset| = 0.5; past that the discriminator rails and carries only its sign, and admission
    # would arm a loop with no gradient to follow.
    presence_disp_off_max=0.6,
    # --presence-admit-displaced: max fitted noise pedestal, relative to the peak.
    # Displaced-but-strong rows fit ~0.03-0.08; weak rows and probes fit ~1 or more (E = P = L fits
    # an infinite pedestal).
    presence_disp_pedestal_max=0.3,
    # --prn-reconfig: refuse to schedule a swap when the F-engine axis sample is older than this (a
    # failed status poll leaves the previous one standing). The deadline advances the polled hop by
    # the sample's age on the cycle clock, and an extrapolation over a long gap is a guess: past
    # this age the swap posts unscheduled, and says so.
    prn_reconfig_axis_max_age_s=30.0,
    # Log the PRN MAP status line this often even with nothing to propose. Every other branch speaks
    # only when it has a swap in mind, and nothing to do is the normal state, so without this an
    # armed, idle stage looks the same as one that is not running.
    prn_reconfig_heartbeat_s=900.0,
    # Schedule a PRN swap this far ahead, as an absolute F-engine hop (set_prns {"at_hop": N}), so
    # every node crosses the discontinuity on the same frame; otherwise the combiner folds a window
    # whose instances disagree about which satellite a slot holds, which no downstream row shows.
    # The lead must cover the scatter of the POSTs, which go to each endpoint in turn with up to
    # prn_reconfig_timeout_s each. It must also cover the delay between this cycle's status poll and
    # the POST: the axis sample is advanced by its age on the cycle clock, which is frozen for the
    # cycle, so that delay is not measured. A node applies a deadline it can never reach (an
    # F-engine re-base moves the hop backwards) at once, with a warning.
    prn_reconfig_lead_s=5.0,
    # Time to sweep every node's live map (/get_prns), one endpoint at a time every
    # poll_s / n_endpoints, so a dead node costs one timeout per sweep step rather than a burst.
    # Read back, not remembered, so the diff is against what the nodes actually hold, including
    # after a restart reverted one to its config list.
    prn_reconfig_poll_s=60.0,
    # Per-endpoint HTTP timeout for the map GET and POST; with the one-endpoint sweep it bounds what
    # a dead node can cost a sweep step.
    prn_reconfig_timeout_s=2.0,
    # Per-poll bound (Hz) on how far the posted carrier command moves, toward its target and when a
    # released command walks back to 0. The stability term: under --rrate-feed-applied 1 the
    # measured residual reflects the command posted 1-2 polls earlier while the feed adds back the
    # latest one, so during a transient the reconstruction is off by (cmd_now - cmd_then), which
    # reinforces itself. Bounding the step bounds that error at ~slew x lag, and at the fixed point
    # the command stops moving and the reference is exact. 0.5 Hz per poll is a full 5 Hz pull-in in
    # ~20 s at a 2 s poll. Railed steps are counted in the JRR-CMD line; a rail that never clears is
    # a target out of reach.
    rrate_cmd_slew_hz=0.5,
    # Windows meaned for --telem-dll and the telemetry C/N0 estimators; 0 uses --telem-windows (32
    # in production: 32 windows x 4 records = 128, what the combiners' EMA integrates). A much
    # shorter window makes the comb arm noisier than the arm it replaces.
    telem_dll_windows=0,
)
# ── end frozen tuning ────────────────────────────────────────────────────────────────────

# The names --signal sets after the parse (gps_distributed_broker.main), after _FROZEN has been
# applied: it would overwrite a frozen value silently, so none of these may be frozen.
SIGNAL_IMPLIED = frozenset(
    (
        "carrier_hz",
        "chip_rate_hz",
        "code_length",
        "long_code_segments",
        "long_code_epoch_s",
        "nh_overlay_len",
        "constellation",
        "dr_constellation",
        "dr_min_prn",
        "signal_capability",
    )
)
assert not SIGNAL_IMPLIED & _FROZEN.keys(), SIGNAL_IMPLIED & _FROZEN.keys()

# The --joint-consume names a consumer tests for (deadreckon.py, gps_distributed_broker.main).
JOINT_CONSUMERS = ("clk", "rate", "slew")


def parse_joint_consume(s):
    """--joint-consume -> the set of consumer names; SystemExit on a name no consumer tests for.

    Any name switches the joint solve on, so a mistyped one would run the solve and consume
    nothing, silently."""
    names = {c.strip() for c in (s or "").split(",") if c.strip()}
    unknown = sorted(names - set(JOINT_CONSUMERS))
    if unknown:
        raise SystemExit(
            "--joint-consume: unknown consumer(s) %s; the consumers are %s"
            % (", ".join(map(repr, unknown)), ", ".join(JOINT_CONSUMERS))
        )
    return names


# The --telem-gather a replay (--transcript-read) connects to. A transcript does not hold the
# telemetry stream, so a replay that reached a live gather (one listens on the gnss VM) would
# fold live frames into the run. Only root can listen below port 1024 and nothing serves port
# 1, so the client exists, as in the recording, and every connect is refused.
DARK_GATHER = "127.0.0.1:1"


def telem_gather_endpoint(telem_gather, transcript_read):
    """The --telem-gather this run connects to: DARK_GATHER when a replay names one."""
    return DARK_GATHER if (telem_gather and transcript_read) else telem_gather


def build_parser(description):
    """The broker's argument parser; `description` is the --help text (the broker's docstring)."""
    ap = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--rest-url",
        default="http://localhost:12048",
        help="kotekan REST base URL that bare stage names in the endpoint lists "
        "(--detectors, --trackers, --combiner, ...) and --time0-endpoint resolve "
        "against",
    )
    ap.add_argument(
        "--detectors",
        default="",
        help="detection endpoints: a comma list of search or aggregator stage names "
        "(resolved on --rest-url), absolute URLs and {a..b} ranges, e.g. aggregate or "
        "search_{00..49}. Each cycle the broker GETs /get_detections from every one "
        "and keeps the best-SNR detection per PRN (under --detectors-monitor only for "
        "the SEARCH-MONITOR log); the --narrow-search and --nh-hint "
        "hints and the --prn-reconfig map are posted to them. Empty = no search: with "
        "--almanac --dead-reckon every visible satellite is seeded from the BRDC "
        "model, on a receiver clock adopted from another chain (--dr-clock-adopt) or "
        "primed (--dr-clock-chips).",
    )
    ap.add_argument(
        "--detectors-monitor",
        action="store_true",
        help="poll --detectors for the record only. Each detection that passes "
        "--acquire-snr becomes a receiver-clock offset as in the clock solve, and the "
        "offsets and their circular median are logged against the joint clock "
        "(SEARCH-MONITOR, every 10 s, with a 5-minute median every 60 s); then the "
        "detections are withheld, so no seed, hold, escape, deep gate, clock or "
        "clock-frequency-bias solve, or joint feed sees them. With --narrow-search the "
        "Doppler hints are still posted, and the search scans only hinted PRNs; with no "
        "bias solve the hints keep the --search-margin-wide-hz margin. Their nh "
        "alignments reach --nh-hint, but it learns only while this chain's own solve is "
        "established or unjudged, so nothing once the chain holds a clock and a drift "
        "(an adopted clock carries its donor's drift). Seeding "
        "runs as on a chain without --detectors, except that a locked satellite keeps a "
        "search-fed chain's plain hold instead of the slewed refresh. Needs "
        "--dead-reckon. For a search on this chain's own signal beside its tracker: "
        "it tells whether a standing code offset is in the signal or in the tracker "
        "path.",
    )
    ap.add_argument(
        "--trackers",
        required=True,
        help="tracker endpoints, in --detectors' list syntax, e.g. track_{00..49} or "
        "http://nodeA:12048/track_{0..24},http://nodeB:12048/track_{0..24}. Each cycle "
        "the seed set is POSTed to every one's /set_seeds, so all subbands despread "
        "the same code phase and Doppler. The C++ fleet loop's trims "
        "(--fleet-trim-url) and the --prn-reconfig map go to them too.",
    )
    ap.add_argument(
        "--combiner",
        default="combiner",
        help="combiner endpoint (name or URL) whose /get_status is this chain's "
        "per-PRN status: the amplitudes and lock metrics behind the coast, drop and "
        "hold decisions, the nav bits, and the newest F-engine hop the time base is "
        "built from. --dll-combiners adds the rest of the fleet for the code loop.",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help="control-loop period, s: each cycle polls the detectors and the combiner, "
        "runs the loops and posts the seeds, then sleeps for the rest of the period. "
        "Weights applied per update (--code-bias-alpha, --dll-gain, bsat_gain) update "
        "at most once a cycle, so their time constants grow with it.",
    )
    ap.add_argument(
        "--acquire-snr",
        type=float,
        default=12.0,
        help="minimum search SNR for a detection to be used at all; weaker detections "
        "are ignored (no seed, no fit, no clock or nh-offset sample). Hold admission "
        "and the hold escape also need a detection at twice this.",
    )
    ap.add_argument(
        "--noise-probes",
        type=int,
        default=0,
        help="keep this many signal-free PRNs seeded as noise probes: the --probe-dark "
        "satellites first, then the deepest below -15 deg elevation. Their combiner "
        "rows are the noise reference for the presence gate (its q floor needs >= 3 "
        "probes reporting and its prompt-power floor >= 2; with fewer, the chain reads "
        "UNANCHORED and no satellite is present) and for the beam map's pedestal. "
        "Probes are never coasted, dropped or carrier-commanded. Needs --almanac. 0 = "
        "none.",
    )
    ap.add_argument(
        "--post-sat-geometry",
        type=int,
        default=0,
        help="nonzero: every ~30 s, post this constellation's above-horizon "
        "satellites' az/el (with rates and epoch on the BRDC path) to each record "
        "assembler's /set_sat_geometry for the per-element steering, together with the "
        "nearest-to-boresight separation for the assemblers' model-formation freeze. "
        "The assemblers are the --dll-combiners with n2combine replaced by n2assemble. "
        "Needs --almanac. 0 = off.",
    )
    ap.add_argument(
        "--fit-flush-on-reject",
        type=int,
        default=0,
        help="after this many consecutive cp-rate rejections for a PRN "
        "(--cp-rate-model-tol), drop its cp-fit history. The seed's position is the "
        "same fit evaluated at ref_hop, so a wrap-blown slope also displaces the "
        "command (slope error x history span), and the poisoned history keeps "
        "re-poisoning it for as long as it lasts (--fit-hist-len). Flushed, the "
        "satellite is seeded from its detections while a clean fit rebuilds. 0 "
        "disables.",
    )
    ap.add_argument(
        "--lat",
        type=float,
        help="receiver latitude, deg; --almanac needs --lat and --lon",
    )
    ap.add_argument("--lon", type=float, help="receiver longitude, deg (east positive)")
    ap.add_argument(
        "--alt",
        type=float,
        default=0.0,
        help="receiver height above the WGS84 ellipsoid, m",
    )
    ap.add_argument(
        "--mask-deg",
        type=float,
        default=5.0,
        help="elevation mask, deg: a satellite below it is not up, so it is not seeded "
        "(noise probes aside), hinted or dead-reckoned",
    )
    ap.add_argument(
        "--almanac",
        action="store_true",
        help="predict each satellite's Doppler, Doppler rate and elevation from orbits "
        "(--almanac-source) every cycle. Seeds carry the predicted Doppler, plus the "
        "clock-frequency bias --seed-bias-source picks, rather than the search's grid "
        "value (see --seed-doppler), coasted seeds follow the forecast, and only "
        "visible satellites are seeded and hinted. Needs --lat/--lon. Code phase still "
        "comes from the search unless --dead-reckon.",
    )
    ap.add_argument(
        "--almanac-source",
        default="brdc",
        choices=("brdc", "tle"),
        help="orbit source for --almanac. brdc (default): the IGS broadcast-ephemeris "
        "file (gnss_ephemeris), indexed by PRN, with metre-level orbits and the "
        "satellite clock, and no skyfield needed. tle: Celestrak's GPS TLEs (gps-ops, "
        "so GPS only) through skyfield, with km-level orbits and PRNs taken from "
        "satellite names, which can be mislabelled. tle is also the fallback when BRDC "
        "is unavailable at startup.",
    )
    ap.add_argument(
        "--boresight",
        default="nominal",
        choices=("nominal", "none"),
        help="where the dishes point. nominal (default) = CHORD's fixed boresight (az "
        "180, el 81.41), which the D3 sawtooth detector's transit veto, the "
        "dead-reckon clock's transit freeze and the assemblers' model-formation freeze "
        "key on. none = the dishes are not pointed (tipped for site work): no "
        "satellite is ever near boresight, so all three stay off. The setting is "
        "process-wide, so every chain of one broker_multi.py process must agree.",
    )
    ap.add_argument(
        "--constellation",
        default=None,
        choices=("G", "E", "C", "R"),
        help="constellation the almanac predicts: G, E, C or R (one multi-system BRDC "
        "file covers all of them). Unset: --dr-constellation; --signal sets it. For C, "
        "the IGS registry's block types (--signal-capability) decide which satellites "
        "carry the signal; without a registry decision, PRNs below 19 (BDS-2) are left "
        "out.",
    )
    ap.add_argument(
        "--carrier-hz",
        type=float,
        default=1575.42e6,
        help="carrier frequency, Hz, for the predicted Doppler and every "
        "carrier-to-code conversion, and the band key (to 10 kHz) --dr-clock-adopt "
        "matches a same-band donor on. Default 1575.42e6 (GPS L1); --signal sets it.",
    )
    ap.add_argument(
        "--narrow-search",
        action="store_true",
        help="each cycle, post every up, capable PRN's predicted Doppler plus the "
        "solved clock-frequency bias to the detectors' /set_doppler_hints. The search "
        "scans only PRNs with a fresh hint, each over the hint +- a margin, so a search "
        "no chain hints scans nothing. The margin is --search-margin-hz once the bias is "
        "solved and fresh, --search-margin-wide-hz before that and while it is stale. "
        "Needs --almanac.",
    )
    ap.add_argument(
        "--search-margin-hz",
        type=float,
        default=500.0,
        help="Doppler hint half-window, Hz, once the clock-frequency bias is solved "
        "and fresh (--narrow-search)",
    )
    ap.add_argument(
        "--search-margin-wide-hz",
        type=float,
        default=3000.0,
        help="Doppler hint half-window, Hz, before the clock-frequency bias is solved "
        "and while it is stale; it must cover the receiver's unknown oscillator offset",
    )
    ap.add_argument(
        "--dr-clock-adopt",
        action="store_true",
        help="on a cycle with no code-phase offsets of its own to solve the receiver "
        "clock from (no detections, or only the ones --detectors-monitor withholds), "
        "adopt the clock another chain in this process "
        "contributes to the shared Receiver: a same-band chain's first, else any "
        "band's as a bootstrap, converted as a time through the two chip rates. "
        "Without it such a chain holds the --dr-clock-chips prime for the whole run. "
        "Steps are bounded by --dr-clock-adopt-max-chips. A donor whose code period is "
        "shorter than this chain's also needs a clock-mod-epoch record, which "
        "--nh-joint on a sibling chain contributes.",
    )
    ap.add_argument(
        "--decode-health-file",
        default=None,
        help="write this chain's nav-decode health to this path as JSON (atomically "
        "replaced): per (signal, PRN), whether it is synced, whether a full ephemeris "
        "was extracted, the BRDC-agreement residual dpos_m, and how fresh the decode "
        "is. The file's basename (without extension) names the chain. The broker only "
        "writes it; the viewer server reads the directory and merges the files. Unset "
        "= off.",
    )
    ap.add_argument(
        "--seed-bias-source",
        choices=("ema", "zero"),
        default="ema",
        help="which clock-frequency bias the seeds' Doppler carries. 'ema' (default): "
        "the solved EMA that also centres the search hints. 'zero': 0 Hz whatever the "
        "solve says, never snapping to a new solve; the hints keep the solved bias. "
        "Use 'zero' when the LO is locked to the site reference: the true offset is "
        "then ~0 and the solve (a median of a few grid-quantised detections, wandering "
        "by Hz) is estimator noise, which with the carrier loop open becomes every "
        "satellite's residual carrier and, through the code rate, walks the despread "
        "off peak. Noise probes keep the hint bias.",
    )
    ap.add_argument(
        "--code-length",
        type=float,
        default=1023.0,
        help="primary code length, chips: the period code phases are unwrapped and the "
        "receiver clock is kept in, and with --chip-rate-hz the record period. Default "
        "1023 (GPS L1 C/A); --signal sets it.",
    )
    ap.add_argument(
        "--hops-per-sec",
        type=float,
        default=125000.0,
        help="F-engine hops per second (sample rate / FFT length): converts every hop "
        "count (ref_hop, pow_hop, fit spans) to seconds. A property of the F-engine; "
        "--signal does not set it.",
    )
    ap.add_argument(
        "--chip-rate-hz",
        type=float,
        default=1.023e6,
        help="primary chip rate, chips/s: converts code phase to time (the receiver "
        "clock, code rates, the record period). Default 1.023e6 (GPS L1 C/A); --signal "
        "sets it.",
    )
    ap.add_argument(
        "--code-bias-alpha",
        type=float,
        default=0.05,
        help="EMA weight per update of the receiver code-rate clock offset (l-a), the "
        "median of the satellites' fitted code rates, which is seeded to every unheld "
        "satellite as its code-phase rate (unless --joint-consume rate supplies it) "
        "and shared with the process's chains that fit none. Small, so it averages "
        "out fit noise and follows oscillator drift.",
    )
    ap.add_argument(
        "--code-bias-min-sats",
        type=int,
        default=2,
        help="fitted satellites a cycle needs before their median (l-a) updates the "
        "code-rate clock offset",
    )
    ap.add_argument(
        "--presence-admit-displaced",
        action="store_true",
        help="admit to fleet presence a row that fails the q or prompt-power bar when "
        "the (E/P, L/P) decomposition (fleet.epl_decompose) reads its deficit as a "
        "code offset rather than weakness (the fit assumes taps at +-0.5 chips on a "
        "triangular correlation, whatever --dll-spacing is). q and prompt power both "
        "fall with the offset the DLL exists to remove, so without this a displaced "
        "satellite is never armed for its own correction. The row also needs evidence "
        "that the satellite is there (prompt power over the probe floor, or deep_snr "
        ">= presence_disp_deep_margin x deep_floor from the combiner's deep fold, "
        "whose integration gain keeps a satellite a fraction of a chip off-peak "
        "detectable), a fitted pedestal <= presence_disp_pedestal_max and |offset| <= "
        "presence_disp_off_max (constants in cli.py _FROZEN). Displacements over 0.1 "
        "chip are refused while the chain is browned out, or when 3 or more PRNs share "
        "one offset (within 0.1 chip) in a cycle: a chain event. Acts only under the "
        "probe-anchored floors. Admitted rows carry present_gate 'q+p:probes+disp' or "
        "'q+deep:probes+disp'.",
    )
    # ---- Live PRN membership (gnss_broker/prnmap.py, docs/CHORD_LIVE_PRN_RECONFIG.md) ----
    # Keeps the nodes' slot->PRN maps in step with the BRDC sky (--prn-reconfig), and can limit
    # the noise probes to PRNs the nodes hold a slot for (--probe-require-slot).
    ap.add_argument(
        "--probe-require-slot",
        action="store_true",
        help="pick noise probes only from PRNs the nodes hold a slot for, from the "
        "unanimous map that --prn-reconfig apply reads back (/get_prns): a probe "
        "without a slot is seeded but never reports, which can leave the presence gate "
        "short of the 3 probes its q floor needs. Fails open: with --prn-reconfig off, "
        "before a full sweep, or while the nodes disagree, the probes are not "
        "filtered.",
    )
    # A satellite with ephemeris that the registry says cannot broadcast this signal
    # (--signal-capability) is signal-free at any elevation: a noise probe that never sets.
    ap.add_argument(
        "--probe-dark",
        action="store_true",
        help="take noise probes first from the dark satellites: those with ephemeris "
        "that --signal-capability excludes, which carry nothing on this signal at any "
        "elevation (on L2OC, the GLONASS M and M+ satellites). They come in PRN order, "
        "any the nodes already hold first, ahead of the deepest below -15 deg. Use it "
        "on a chain with too few capable satellites to keep the 3 probes the presence "
        "gate needs below -15 deg. The nodes must hold slots for the dark PRNs; the PRN "
        "map keeps them as probes. Needs --signal-capability.",
    )
    ap.add_argument(
        "--prn-reconfig",
        choices=["off", "apply"],
        default="off",
        help="keep the nodes' slot->PRN map in step with live BRDC "
        "(gnss_broker/prnmap.py, docs/CHORD_LIVE_PRN_RECONFIG.md). off (default): the "
        "stage does nothing. apply: read each tracker's map back (/get_prns) and, when "
        "the wanted map (every satellite up, then the noise probes) differs, POST it "
        "whole to every tracker's /set_prns on one F-engine hop deadline, and without "
        "one to the detectors. A moved slot re-acquires cold (the node resets its code "
        "table, Phi cache, carrier NCO, seed, trim, power averages and element cal), "
        "so --prn-reconfig-interval-s and the admit/evict hysteresis damp the changes.",
    )
    ap.add_argument(
        "--prn-reconfig-interval-s",
        type=float,
        default=900.0,
        help="minimum seconds between map posts. One post moves every slot that needs "
        "it, on one deadline; each moved slot re-acquires cold, so this paces the "
        "re-acquisitions. It also paces re-posting the map to a node that restarted "
        "onto its config list.",
    )
    ap.add_argument(
        "--prn-reconfig-admit-deg",
        type=float,
        default=10.0,
        help="elevation, deg, at or above which a satellite without a slot is wanted "
        "and may take the slot of an unwanted satellite, judged on its current "
        "elevation rather than a forecast rise; below it, an unslotted satellite that "
        "is not a noise probe gets only a dead slot (one whose PRN has no prediction). "
        "Must be above --prn-reconfig-evict-deg: the gap is the hysteresis that keeps "
        "a satellite at the threshold from flapping in and out of a slot (with no gap "
        "the map is left alone and the fault logged).",
    )
    ap.add_argument(
        "--prn-reconfig-evict-deg",
        type=float,
        default=0.0,
        help="a slot's satellite keeps it until its elevation falls below this, deg; "
        "the slot can then go to a satellite that wants one",
    )
    ap.add_argument(
        "--fit-gap-s",
        type=float,
        default=16.0,
        help="reset a PRN's code-phase and Doppler fit histories across a gap between "
        "its detections longer than this, s (on the F-engine hop axis, via "
        "--hops-per-sec): the detection after it is a re-acquisition. The cp fit "
        "unwraps to the nearest code period, so a gap over which the clock drifts half "
        "a period puts the fit on the wrong branch.",
    )
    ap.add_argument(
        "--carrier-max-hz",
        type=float,
        default=40.0,
        help="clamp, Hz, on the carrier command --rrate-command posts "
        "(carrier_trim_hz); <= 0 = no clamp",
    )
    ap.add_argument(
        "--dll-gain",
        type=float,
        default=0.25,
        help="gain per update of the broker's code delay-lock loop (one update per new "
        "discriminator window, at most one per cycle). A locked PRN's code-phase "
        "trim, a leaky integral of gain x tau_est from the E/L discriminator (clamped "
        "to +-3 chips), is added to its seed's code phase at POST time; a PRN the C++ "
        "fleet loop has armed (--fleet-trim-url) is left to that loop. 0 turns off the "
        "whole fleet DLL stage: presence, the C++ loop's arming and every measurement "
        "taken off its poll (fleet_coherent, the --spectrum-endpoints fit, the C/N0 "
        "estimators, the RF and element polls, model primacy).",
    )
    ap.add_argument(
        "--nh-hint",
        action="store_true",
        help="hint each PRN's NH overlay alignment to the search, so it scans "
        "nh_hint_span (a search stage setting) alignments either side of the hint "
        "instead of all of them. The alignment at transmit follows from the ephemeris, "
        "round((gpst - range/c + clk_sv)/period) mod overlay_len, up to one constant "
        "every satellite shares (the receiver clock's offset), which each new "
        "detection measures; so a satellite never detected is hinted too. The offset "
        "expires when its samples go stale. Needs --narrow-search (the hints are "
        "posted with its Doppler hints), --almanac and the capture time anchor "
        "(--time0-endpoint, read under --dead-reckon).",
    )
    ap.add_argument(
        "--joint-p2c-rotate",
        action="store_true",
        help="rotating coast test of the joint receiver state: withhold the "
        "best-established satellite (>= joint_mask_after accepted updates) from the "
        "joint solve for joint_p2c_hold_s, log its residual against coast age (P2C "
        "lines), release it, and move to one not tested in the last joint_p2c_skip "
        "rotations (constants in cli.py _FROZEN). The residual's growth with age "
        "measures the b_sat random walk (the state's q_b); rotating samples many "
        "satellites and elevations, where one hand-picked PRN is one geometry and may "
        "set mid-test. The withheld satellite stays seeded. Runs where the joint solve "
        "does (--joint-consume) and the chain has detections.",
    )
    ap.add_argument(
        "--publish-port",
        type=int,
        default=0,
        help="serve the fleet-merged per-PRN state over REST on this port (0 = off; "
        "FleetPublisher in gnss_broker/publish.py). GET /get_status returns rows in "
        "GnssCoherentCombiner's schema, so the viewer reads them unchanged, built from "
        "every --dll-combiners instance rather than one instance's share of the band; "
        "/get_detections, /get_chains, /get_elements, /get_sky and /get_rf are served "
        "too. Under broker_multi.py the first --publish-port serves every chain: "
        "select one with /<chain>/get_status or ?chain=<id>. coh_src names where a "
        "row's coherent statistics came from: fleet_coherent, a quadrature of "
        "instances, or the best single instance.",
    )
    ap.add_argument(
        "--spectrum-stale-margin",
        type=int,
        default=0,
        help="before choosing the common spectrum window for --spectrum-endpoints (the "
        "newest one every instance has finished, a min over instances), drop instances "
        "whose newest window trails the fleet's newest by more than this many windows; "
        "they are listed as dropped. 0 = off: a plain min, where one frozen instance "
        "pins the common window and the fit runs on it alone. Healthy instances lag "
        "one another by a few windows.",
    )
    ap.add_argument(
        "--spectrum-endpoints",
        default="",
        help="record-assembler endpoints (GnssGpuRecordAssemble; comma list, {a..b} "
        "ranges, bare names on --rest-url) for the fleet phase-slope delay fit. Their "
        "/get_spectrum serves the per-channel prompt sums before the cross-channel "
        "combine. A delay is a phase ramp across frequency, so one aligned window from "
        "every instance is fitted for a delay tau per PRN with a free phase per "
        "instance (logged as SPEC-FIT); the fleet's union of channels resolves the "
        "grating lobes a single instance's sparse comb cannot. For a PRN at or above "
        "lock_snr, tau updates the per-satellite path bias b_sat (SatBiasFilter) that "
        "dead-reckon seeds add to the receiver clock. Empty = off. "
        "docs/CHORD_JOINT_TRACKING.md describes the measurement.",
    )
    ap.add_argument(
        "--dcb-bias",
        action="store_true",
        help="use the CAS (MGEX) measured differential code biases in place of the "
        "broadcast TGD/BGD, per satellite, where the product covers the signal. For "
        "BeiDou B2a they are the only group-delay correction: its broadcast term "
        "(TGD_B2ap) is a B-CNAV2 parameter that RINEX 3 does not carry. The product is "
        "zero-mean per constellation, so it corrects the per-satellite spread and "
        "never a constellation-common offset. Applies to the dead-reckon model "
        "(--dead-reckon). Needs an Earthdata token "
        "(~/.cache/kotekan_gps/.earthdata_token); without one it logs and uses the "
        "broadcast term.",
    )
    ap.add_argument(
        "--tropo",
        choices=("saastamoinen", "none"),
        default="saastamoinen",
        help="tropospheric delay in the dead-reckon range model: Saastamoinen zenith "
        "delays from a standard atmosphere at the site height, Chao mapping (gnss_atmos), "
        "or none. Applies to every model-evaluated code phase (seeds, clock offsets, the "
        "tracker residual).",
    )
    ap.add_argument(
        "--iono",
        choices=("none", "klobuchar"),
        default="klobuchar",
        help="ionospheric delay in the dead-reckon range model: the Klobuchar model at "
        "this chain's carrier with the GPSA/GPSB coefficients from the cached nav headers "
        "(default; the last set seen is cached and used, with its age logged, while no "
        "cached file carries them), or none.",
    )
    ap.add_argument(
        "--joint-consume",
        default="",
        help="comma list of joint receiver-state consumers to switch on; empty "
        "(default) runs none. Any name also runs the joint solve on this chain "
        "(docs/CHORD_JOINT_TRACKING.md): its detections' dead-reckon integrity "
        "residuals feed one [clk, clk_rate, b_sat[i]] state shared by every chain of "
        "the receiver. Each consumer refuses a state with fewer than joint_min_sats "
        "satellites and then keeps its own estimate, except that 'clk' may coast (the "
        "joint_* bounds are constants in cli.py _FROZEN, their chips 10.23-Mcps "
        "reference chips on every chain). Names: 'clk' (each dead-reckon pass, move the "
        "dead-reckon clock onto the joint clock as a wrapped delta; refused beyond "
        "joint_clk_max_chips or joint_clk_max_sigma, on a whole-period disagreement "
        "with another chain's clock-mod-epoch record, and while a step beyond "
        "joint_clk_max_step_chips has held for less than joint_clk_step_hold_s; a "
        "refusal within --joint-clk-coast-max-s of the last adoption coasts on that "
        "adoption; logged as JOINT-CLK); 'rate' (seed the "
        "code-rate clock from clk_rate instead of the l-a EMA, refused beyond "
        "joint_max_rate_ppm); 'slew' (dead-reckon seed offsets take the joint clk + "
        "b_sat instead of the clock plus SatBiasFilter, refused per satellite beyond "
        "joint_slew_max_chips; the SEED-OFFSET lines compare the two).",
    )
    ap.add_argument(
        "--joint-feed-warmup-s",
        type=float,
        default=240.0,
        help="withhold every measurement from the joint state (and MINNOV) for this many "
        "seconds of cycle-clock time after this chain's first cycle (context.py "
        "uptime_s). Birth is the one path with no innovation to gate on, and "
        "establishment-phase measurements are the worst the feed sees: garbage born "
        "into the state is self-consistent, so no escape fires. Seeds and holds settle "
        "in ~3 min; 240 s is that plus margin. Only the feed waits; the state object is "
        "created at once. On the cycle clock a replay counts the recorded run, and every "
        "recorded fixture is shorter than the default, so the jwarm0 variant golden "
        "(scripts/gnss/fixtures) replays the holds capture with 0 to pin the feed. "
        "Default %(default)s.",
    )
    ap.add_argument(
        "--joint-clk-coast-max-s",
        type=float,
        default=300.0,
        help="'clk' consumer (--joint-consume): seconds after this chain's last "
        "joint-clock adoption during which a refused joint clock is coasted on instead "
        "of falling back to the dead-reckon (legacy) clock. "
        "The refusal may come from the bounds, the step guard or the clock-period "
        "check, or from a state unfit to consume (fewer than joint_min_sats "
        "satellites, or deaf). The coast puts the clock on the last adopted joint clock carried "
        "forward at the clk_rate stamped at that adoption (deadreckon."
        "dr_joint_clk), and is declined "
        "when that extrapolation exceeds joint_clk_max_chips; logged as JOINT-CLK "
        "... COAST. A coast restamps nothing, so it ages: past the window a refusal "
        "is a plain REFUSED on the legacy clock, the local clock goes stale and the "
        "cross-band bootstrap can re-arm. 0 disables the coast. Default "
        "%(default)s.",
    )
    ap.add_argument(
        "--rrate-state",
        action="store_true",
        help="feed the receiver's joint rrate state from this chain: one row per "
        "satellite holding its orbit model's range-rate error (m/s), shared by every "
        "band and chain of the process, plus the receiver-wide carrier frequency "
        "offset f_carrier. The measurement is the combiner's uncapped deep-fold rate "
        "residual (deep_rate_full_hz) after rate_residuals' gates (peak q, clip about "
        "the median, continuity), from cohering satellites only (coherence_s > 0 or "
        "coh_frac >= 0.3) and never from a --probe-dark probe. Logged as the JRR "
        "line, with the fine-vs-coarse JRRP line beside it. "
        "Measurement only: seeds act on the rows only under --rrate-command, which "
        "implies this flag. --rrate-kcoh-feed and --rrate-phase-feed add feeds and "
        "need it.",
    )
    ap.add_argument(
        "--rrate-command",
        action="store_true",
        help="close the carrier loop: each seed carries carrier_trim_hz, the joint "
        "state's carrier_correction_hz() (the receiver-wide f_carrier plus this "
        "satellite's range-rate term, scaled to this band's --carrier-hz), which the "
        "tracker applies as a phase-continuous NCO offset; doppler_hz is left alone. "
        "A satellite is commanded only while its row's sigma is <= "
        "rrate_cmd_max_sigma and it passes --rrate-cmd-min-sig; the command is "
        "clamped to --carrier-max-hz and moves at most rrate_cmd_slew_hz per poll "
        "(rrate_cmd_max_sigma and rrate_cmd_slew_hz are constants in cli.py _FROZEN). "
        "A satellite that leaves the commanded set slews its command back to 0. In a "
        "poll where no status row carries deep_rate_full_*, or before f_carrier is "
        "solved, nothing is commanded and standing commands slew toward 0. Noise "
        "probes are never commanded. Implies --rrate-state. Logged as JRR-CMD.",
    )
    ap.add_argument(
        "--rrate-feed-applied",
        type=int,
        default=1,
        help="how the coarse (deep-fold) and KCOH rate feeds reference the standing "
        "carrier command. 1 (default): add the command posted last poll back to each "
        "measured rate, for records that carry the actuation, whose rate is what "
        "remains after the command. 0: add nothing, for a record assembler that folds "
        "the applied trim out of every record, so the rates already read the whole "
        "residual; adding the command back there makes each row integrate its own "
        "actuation and ramp at the slew rate. The fine feed (--rrate-phase-feed) "
        "takes its own reference either way.",
    )
    ap.add_argument(
        "--rrate-cmd-min-sig",
        type=float,
        default=0.0,
        help="a satellite may be commanded (--rrate-command) only while its latest "
        "KCOH significance (the known-rate fold's sig) is at least this; a satellite "
        "with no KCOH row is below it. The row-sigma gate (rrate_cmd_max_sigma) "
        "judges the fit, and a confident fit to a marginal satellite's noise can pass "
        "it; this bar asks for the signal itself. A satellite that fails it slews its "
        "standing command back to 0. 0 (default) = off.",
    )
    ap.add_argument(
        "--rrate-phase-feed",
        action="store_true",
        help="also feed the rrate rows from the fine ADR phase rate: the combiner's "
        "res_cycles differenced over a span (the last poll, or --rrate-phase-span-s), "
        "whose noise floor is far below the deep-fold rate's. A satellite feeds only "
        "when adr_fine_rate's structural gates pass (same arc, record counter "
        "advanced and consistent with wall time), the command moved <= 0.6 Hz over "
        "the span, it is cohering (coherence_s > 0 or coh_frac >= 0.3) and |rate| < "
        "20 Hz. The reference added back is the trim the tracker applied over the "
        "span (trim_cycles) when the combiner serves it, else the posted command's "
        "span mean, with sigma (rrate_phase_sigma, cli.py _FROZEN) widened by the "
        "command's motion. An accepted value deweights that satellite's coarse "
        "measurements (--rrate-coarse-deweight). Needs --rrate-state and a nonzero "
        "--rrate-phase-sign.",
    )
    ap.add_argument(
        "--rrate-phase-sign",
        type=float,
        default=0.0,
        help="sign (+1 or -1) that maps the combiner's res_cycles rate (its internal, "
        "r2c-flipped convention) onto the deep_rate_full_hz convention for the fine "
        "feed; it signs the applied-trim reference too. Calibrate it per chain on sky "
        "from the JRRP line's fine|coarse response to a commanded carrier step. 0 "
        "(default) = uncalibrated: the JRRP line still logs, and the fine feed stays "
        "off whatever --rrate-phase-feed says.",
    )
    ap.add_argument(
        "--rrate-phase-span-s",
        type=float,
        default=0.0,
        help="difference res_cycles for the fine rate against the newest snapshot at "
        "least this many seconds old, instead of the last poll's. The phase error "
        "sits at the two endpoints, so the rate noise falls as 1/span, while the span "
        "mean lags a drifting rate by span/2; the feed's sigma carries both terms, "
        "and the measurement stays stamped at now. In span mode a satellite feeds at "
        "most once per span, so fed values share no data; the JRRP line logs the span "
        "value every poll. On sky the combined error is smallest at 16-32 s; much "
        "longer spans lose to the lag. 0 (default) = the last poll's snapshot.",
    )
    ap.add_argument(
        "--rf-stats-endpoints",
        default="",
        help="comma list ({a..b} ranges, bare names on --rest-url) of voltage-tap "
        "endpoints, e.g. http://cx19:12048/gnss{0..1}_srch_tap, polled every "
        "rf_stats_poll_s (cli.py _FROZEN) for RF-path health: clip fraction and power "
        "per band, served on the publisher's /get_rf (--publish-port) for the "
        "viewer's stream-health panel, and logged as RF PATH; with no publisher "
        "nothing is polled. Empty = off. Set it on one chain only: the tap is per "
        "GPU and watches every chain's channels, so "
        "its readings are the receiver's, and arming it on several chains polls the "
        "same endpoints and logs each alarm once per chain. A tap whose node config "
        "has no band_power_chans reports enabled=false and is published as off, never "
        "as zeros.",
    )
    ap.add_argument(
        "--rfi-stats",
        action="store_true",
        help="also fetch each --rf-stats-endpoints instance's spectral-kurtosis RFI "
        "metrics (rfi_sk_metrics/sk_metrics_{gpu}, derived from the tap URL) and add "
        "an SK summary to its get_rf entry. Off by default.",
    )
    ap.add_argument(
        "--rf-bands",
        default="",
        help="comma list of chain (signal) names declaring the RF bands the receiver "
        "flies, e.g. gps_l5,gal_e5b,gps_l2c,bds_b3i,gal_e6; the carriers come from "
        "gnssSignal.hpp. The stream-health panel labels each channel the "
        "--rf-stats-endpoints taps monitor with the nearest declared carrier. "
        "Matching the whole band table instead would name bands CHORD does not fly, "
        "so with this unset the rows read 'lobe N'. List the fleet's bands, not only "
        "this chain's: the tap is per GPU and monitors the union of every chain on "
        "it. Set it on the chain that sets --rf-stats-endpoints. Unknown names are "
        "skipped.",
    )
    ap.add_argument(
        "--drop-stats",
        action="store_true",
        help="also read each --rf-stats-endpoints host's /metrics (once per host per "
        "poll, shared by its GPU instances) for drop counters: the gnss{gpu} search "
        "and telemetry buffer-send drops and the node's DPDK rx-missed and ring-full "
        "drops, added to each instance's get_rf entry. Off by default.",
    )
    ap.add_argument(
        "--rrate-coarse-deweight",
        type=float,
        default=8.0,
        help="the FLL-to-PLL handoff: while a satellite has an accepted fine "
        "(--rrate-phase-feed) or KCOH (--rrate-kcoh-feed) measurement within "
        "rrate_fine_hold_s (cli.py _FROZEN), its coarse deep-fold rate measurements "
        "enter the rrate row at this multiple of their sigma. They are deweighted, "
        "never dropped: the coarse feed re-acquires the row after a cycle slip or a "
        "re-seed. The default 8 (64x the variance) makes each coarse measurement count "
        "1/64 as much, so the row follows the phase and the coarse rate only "
        "supervises it. <= 1 = no handoff, both feeds at face value.",
    )
    ap.add_argument(
        "--dr-fengine-axis",
        type=int,
        default=0,
        help="nonzero: build the dead-reckon time base (t_now_abs) from the F-engine "
        "hop counter instead of wall time minus the frame-0 anchor: the newest "
        "pow_hop in --combiner's status, kept as a (hop time - wall) offset filtered "
        "against the pipeline lag's jitter, plus the wall clock. The nodes' hop->UTC "
        "mapping is PPS-disciplined, while the broker's wall clock carries NTP error "
        "that drifts (1 us is ~10 L5 chips); on this axis wall time enters only as "
        "elapsed seconds. The static part of that error is common-mode and lands in "
        "the solved receiver clock, but the drift is not re-solved on a chain whose "
        "clock is adopted, and nothing cancels it in a --seed-phase-transport phase. "
        "The orbit is evaluated on the same time base (deadreckon.dr_eph_epoch). "
        "Changing it moves every dead-reckon seed's epoch, so replay goldens that "
        "cover dead-reckon paths change. 0 (default) = wall minus anchor.",
    )
    ap.add_argument(
        "--dr-forecast-lead-s",
        type=float,
        default=0.0,
        help="build dead-reckon seeds for the whole hop nearest now + this many "
        "seconds, rather than for now. A seed is a forecast labelled with its ref_hop "
        "(phase, Doppler and rate at that hop), so the telemetry lag then only sets "
        "how large the lead must be instead of entering the arithmetic. The lead must "
        "exceed transport plus install, or a seed arrives after the epoch it "
        "describes; 1-2 s clears the few hundred ms of lag with margin. Longer leads "
        "cost accuracy: cp_predicted carries the range linearly, and a range "
        "acceleration of ~0.2 m/s^2 gives ~0.4 m (0.013 L5 chips) at 2 s. Ages, rrate "
        "updates and the joint cycle stay at now. Needs the F-engine axis sample from "
        "--combiner's status. 0 (default) = seeds for now.",
    )
    ap.add_argument(
        "--model-primacy-max",
        type=int,
        default=0,
        help="on a chain with a search, let up to this many PRNs run model-primary: "
        "their seeds come from the dead-reckon slew (the model plus the receiver "
        "clock and path bias; the joint ones under --joint-consume slew), and their "
        "detections only feed the joint filter and the MINNOV referee, never "
        "re-anchoring, holding or escaping the seed. MINNOV is a detection's code "
        "phase against the joint state's prediction made without it. A PRN is "
        "eligible when its 10-min MINNOV p95 is below model_primacy_p95 over at least "
        "model_primacy_min_n samples; the lowest-p95 PRNs fill the cap and the rest "
        "are logged as controls. A flipped PRN goes back to the search when its p95 "
        "exceeds model_primacy_exit_p95 or its referee starves (no MINNOV in the "
        "10-min window, or no detection for model_primacy_starve_s; constants in "
        "cli.py _FROZEN). Probes, untrusted PRNs and PRNs within 300 s of an exit are "
        "not eligible. Logged as MODEL-PRIMACY ENTER/EXIT. Needs --dead-reckon and the "
        "joint solve (--joint-consume). 0 (default) = off.",
    )
    ap.add_argument(
        "--rrate-kcoh-feed",
        type=int,
        default=0,
        help="nonzero: also feed the rrate rows the KCOH fold's remaining carrier "
        "rate (rate_hz + rate_resid_hz), a phase-progression estimate (lag products "
        "of each instance's prompt segments, summed over the fleet) that does not "
        "share the deep fold's spectral peak search. A satellite feeds when its KCOH "
        "sig is >= rrate_kcoh_min_sig (cli.py _FROZEN), with >= 8 lag pairs and "
        "|rate| < 20 Hz; probes never feed, and each KCOH estimate feeds once. The "
        "command is referenced as --rrate-feed-applied says. An accepted value "
        "deweights the satellite's coarse measurements like a fine one "
        "(--rrate-coarse-deweight). Logged as JRR-KCOH. Needs --rrate-state; KCOH "
        "itself needs --telem-gather and noise probes. 0 (default) = off.",
    )
    ap.add_argument(
        "--seed-phase-transport",
        action="store_true",
        help="dead-reckon births, re-pins and slews also carry "
        "code_phase_at_ref_chips, the code phase at ref_hop, beside the sample-0 "
        "argument code_phase_chips. The tracker's propagate_seed prefers the phase: "
        "the argument is referred to sample 0 through the Doppler, so a producer that "
        "edits the Doppler without re-projecting it moves the despread by t_abs x "
        "f_chip / f_carrier chips per Hz (~1700 at L5 after 2.3 days of F-engine "
        "uptime), while a phase at its own epoch has no such lever. The broker "
        "references a hop at its first sample and the tracker at its last, so the "
        "phase is converted by one hop (fits.seed_phase_at_ref) before it ships. "
        "Search-fed seeds on an overlaid code carry the phase either way.",
    )
    ap.add_argument(
        "--dll-combiners",
        default="",
        help="comma list of combiner endpoints ({a..b} ranges, bare names on "
        "--rest-url), one per instance, for the fleet code loop "
        "(docs/CHORD_GNSS_SHARED_DLL.md). An instance correlates only the channels "
        "its GPU holds, a small fraction of the band, so the broker sums the "
        "instances' raw Early/Prompt/Late powers per PRN over a common window "
        "(pow_hop within dll_hop_window_s of the newest; this and dll_min_instances "
        "are constants in cli.py _FROZEN) and forms one discriminator (E-L)/(E+L) "
        "from the sums. The discriminator is non-coherent, so the sum needs no phase "
        "alignment between instances, only the same window; it does need every "
        "instance despread at the same commanded phase, which one code loop (this "
        "broker's or the C++ fast loop's) provides. Presence (q and prompt power "
        "against the noise probes' floor) is judged on the sums. The list also feeds "
        "fleet_coherent and --element-poll, and names the record assemblers "
        "(n2combine -> n2assemble) that --post-sat-geometry posts to. A PRN seen by "
        "fewer than dll_min_instances instances, or an empty list, falls back to "
        "--combiner's own discriminator, gated on lock_snr.",
    )
    ap.add_argument(
        "--dll-spacing",
        type=float,
        default=0.5,
        metavar="CHIPS",
        help="Early/Late tap offset from Prompt, in the signal's own (component) "
        "chips. The taps are on the node; this value enters the trim law here (tau = "
        "-disc/4 x (spacing/0.5)) and the C++ fast loop's copy, posted with its "
        "policy, so it must match the node's dll_spacing for this chain "
        "(config/gen_chord_gnss_config.py dll_spacing_chips): a mismatch is a pure "
        "gain error. When changing both, cycle the nodes before the broker. The "
        "offset is set by the width of the correlation the loop closes on, not by the "
        "chip: E and L must sit inside its first null, where the discriminator is "
        "monotonic, or the loop false-locks off peak. 0.5 (the default) suits the "
        "chip-narrow correlation of a chain whose instances each despread a "
        "multi-channel comb; GPS L2C (a fleet sum of single-channel instances) and "
        "GLONASS L2OC (BOC side peaks) run the narrower offsets dll_spacing_chips "
        "gives them and says why.",
    )
    ap.add_argument(
        "--dll-deep-gate",
        default="",
        help="PRNs whose fleet presence, which arms their code trims, is judged on "
        "detection rather than on q and summed prompt power: such a PRN is present "
        "when its strongest instance's deep_snr >= dll_deep_gate_margin x that "
        "instance's deep_floor (cli.py _FROZEN), overriding the q/prompt and "
        "displaced verdicts; a row with no floor keeps the q/prompt verdict. PRNs "
        "separated by commas or spaces, or 'all'; empty "
        "(default) = none. Prompt power falls when the tap is off peak, so a "
        "prompt-gated satellite that drifts off the peak is never trimmed back. The "
        "deep fold is a coherent fold of the same prompt (carrier rate and phase "
        "searched, code not), whose ~10x integration gain keeps detecting a satellite "
        "a fraction of a chip off-peak; beyond ~1 chip it is blind too. Opt in per "
        "PRN: deep_snr can fire on noise, and this gate has no pedestal or offset test. "
        "--dll-deep-gate-from-search adds the PRNs the search detects strongly.",
    )
    ap.add_argument(
        "--dll-deep-gate-from-search",
        type=float,
        default=0.0,
        metavar="SNR",
        help="add every PRN the search detected at snr >= SNR within the last "
        "--dll-deep-gate-search-hold-s to the --dll-deep-gate set, alongside any "
        "hand-listed PRNs; 0 (default) = off. The search re-derives code phase over "
        "the full code on every pass, so its detection does not depend on where the "
        "tap sits, and unlike deep_snr it does not fire on noise: a strong detection "
        "vouches for the satellite as hand-listing the PRN does. With --almanac, only "
        "detections of satellites above --mask-deg "
        "count, so a spurious below-horizon detection cannot arm a correction. SNR is "
        "in the search's snr units: set it well above --acquire-snr and the "
        "pure-noise ceiling the search logs each pass. The set is logged (a DEEP GATE "
        "line) when it changes.",
    )
    ap.add_argument(
        "--dll-deep-gate-search-hold-s",
        type=float,
        default=90.0,
        help="seconds a search detection keeps its PRN in the "
        "--dll-deep-gate-from-search set (default %(default)s). Make it longer than "
        "the gap between the search's detections of one satellite (a PRN is not "
        "re-detected every cycle): a shorter hold flaps the PRN in and out of the "
        "deep gate with the search's revisits, and its presence and trim with it, "
        "under a satellite that never changed. It is a hold, not a staleness bound: "
        "it answers whether the satellite is up and detectable, which stays true "
        "between passes.",
    )
    # -- The frame-synced telemetry gather (gnss_broker/telem.py) ------------------------------
    ap.add_argument(
        "--telem-gather",
        default="",
        help="host:port of the frame-synced telemetry gather, read by "
        "gnss_broker/telem.py; a bare host takes port 11061 (GATHER_SERVE_PORT in "
        "config/gen_chord_gnss_config.py). Empty (default) = off. Every frame carries "
        "its chain, its instance and an absolute window index on the F-engine sample "
        "clock, so instances are grouped by exact integer match rather than by when a "
        "REST reply arrived. One reader thread serves every chain in the process. "
        "Connected, it feeds the TELEM alignment line, the fleet ADRs, the tracker "
        "code residuals and, with noise probes, the KCOH and prompt C/N0 estimators. "
        "By itself it replaces no REST poll: --telem-dll and --telem-coherent move "
        "the code loop and fleet_coherent onto it.",
    )
    ap.add_argument(
        "--telem-coherent",
        action="store_true",
        help="fleet_coherent reads this chain's records from the --telem-gather "
        "stream instead of polling each --dll-combiners instance's /get_records: the "
        "same estimator on a different transport, and the poll is skipped. Gathered "
        "records are addressed by absolute window index, so every instance "
        "contributes the same windows, where a polled set is whatever accumulated "
        "since each GET. Needs --telem-gather; reads --telem-windows windows. A cycle "
        "in which the gather has no windows for this chain polls /get_records as "
        "without the flag, and logs that it did.",
    )
    ap.add_argument(
        "--telem-windows",
        type=int,
        default=8,
        help="telemetry windows read by the gathered-stream consumers: "
        "fleet_coherent under --telem-coherent, the comb DLL (--telem-dll), and the "
        "prompt C/N0 and KCOH estimators. A window is 4 records, so the default 8 is "
        "32 records, ~0.34 s. Under --comb-taps-cpp the comb DLL and the prompt C/N0 "
        "take the gather's depth instead (FLEET_TRIM_TAPS_WINDOWS in "
        "config/gen_chord_gnss_config.py), which must equal this.",
    )
    # -- Per-element gains (gnss_broker/elemgain.py) -------------------------------------------
    ap.add_argument(
        "--element-poll",
        action="store_true",
        help="poll each --dll-combiners instance's /get_elements every "
        "element_poll_every_s (cli.py _FROZEN; every cycle under transcript replay) "
        "and serve the per-element complex-gain table (amplitude and phase per "
        "antenna, per instance) on the publisher's /get_elements. Off by default, as "
        "for --spectrum-endpoints, because replay is strict-ordered: a transcript "
        "recorded without these GETs cannot replay with them. An instance that does "
        "not answer (down, or a node without the endpoint) is skipped and counted, "
        "never fatal; one whose snapshot lags the fleet is excluded and named on the "
        "ELEM-GAIN line.",
    )
    ap.add_argument(
        "--element-archive-dir",
        default="",
        help="append the raw per-element gain parts (u, p2, q per antenna) of present "
        "satellites and noise probes to <dir>/elem_<chain>_<YYYYMMDD>.jsonl (UTC "
        "date) every element_archive_every_s (cli.py _FROZEN). Raw parts because the "
        "beam map and peel solves are offline consumers, and a combined number cannot "
        "be un-combined. Needs --element-poll. Empty (default) = no archive.",
    )
    # -- The comb DLL and the fleet (gnss_broker/combdll.py, fleetdll.py, fleet.py) ------------
    ap.add_argument(
        "--telem-dll",
        action="store_true",
        help="close the code loop on the comb discriminator formed here from the "
        "gathered per-channel Early/Prompt/Late (gnss_broker/combdll.py) instead of "
        "the polled one built from the combiners' per-instance powers. The comb is "
        "coherent over every channel of the lobe, every instance's channels first "
        "rotated onto a common phase reference, where the polled discriminator adds "
        "powers across instances. Needs --telem-gather; averages --telem-windows windows, and "
        "--comb-taps-cpp moves the reduction into the gather. The combiners are still "
        "polled: deep_snr, deep_floor, coherence_s and the quadrature fallback come "
        "from their deep fold and are carried across to the comb rows for the deep "
        "gate and the publisher. A cycle with no gathered windows for this chain, or "
        "a failed reduction, keeps the polled discriminator; the COMB-DLL line "
        "compares the two.",
    )
    ap.add_argument(
        "--comb-taps-cpp",
        type=int,
        default=0,
        choices=(0, 2),
        help="2: take the comb DLL's per-PRN taps (--telem-dll) and the prompt C/N0's "
        "per-record series from the gather's C++ reduction (<fleet-trim-url>/get_taps "
        "and /get_rec_taps) instead of walking the gathered frames in Python; 0 "
        "(default) = the Python walk. Needs --fleet-trim-url. The walk allocates "
        "Python objects for every (window, instance, record, PRN, channel) of the "
        "stream, a large share of a chain's CPU, while the gather already holds the "
        "frames and the reduction is a few numbers per PRN. The arithmetic is the "
        "same: scripts/gnss/fleetdll_gate.py requires the two arms to agree on "
        "identical bytes (e/p/l/n_chan to 1e-9, n_rec and hop exact, per PRN and per "
        "channel). The depth is the gather's (FLEET_TRIM_TAPS_WINDOWS in "
        "config/gen_chord_gnss_config.py), which must equal --telem-windows. With it "
        "the prompt C/N0 runs every cycle rather than every estimator_every_s (cli.py "
        "_FROZEN). A failed GET costs that cycle's comb DLL (the polled discriminator "
        "stands) and prompt C/N0.",
    )
    ap.add_argument(
        "--fleet-trim-url",
        default="",
        help="base URL of the C++ fast code loop (GnssFleetTrim in the telemetry "
        "gather), e.g. http://127.0.0.1:12051/fleet_trim; empty (default) = off, and "
        "the broker's own integrator trims every PRN. Each cycle the broker POSTs "
        "<url>/set_policy with the armed PRNs (present within fleet_trim_hold_s), the "
        "loop constants (--fleet-trim-bandwidth, fleet_trim_leak_per_s, "
        "--dll-spacing), the probe-anchored prompt-power floor and the trackers' "
        "/set_trim targets. The loop then forms the comb discriminator and posts each "
        "armed PRN's trim once per telemetry window (~23.8/s) instead of once per "
        "broker cycle, and the broker's integrator stands down for those PRNs. The "
        "policy stays in the broker: presence, the floors, the deep gate and the "
        "arming decision cross only as their result (fleet_trim_hold_s and "
        "fleet_trim_leak_per_s are constants in cli.py _FROZEN). Also the base for "
        "--comb-taps-cpp and --fleet-trim-readback.",
    )
    ap.add_argument(
        "--fleet-trim-bandwidth",
        type=float,
        default=1.4,
        help="closed-loop bandwidth, 1/s, posted to the C++ fleet loop "
        "(--fleet-trim-url) as gain_per_s. Not a per-update gain: the loop divides it "
        "by the update rate it measures (~23.8/s). The loop is stable while bandwidth "
        "x measurement round trip (discriminator average, actuation, record, fold) "
        "stays below ~0.5; with a 4-window gather average the round trip is "
        "~0.3-0.5 s, so the default 1.4 sits near that bound, and the gather's default "
        "2-window average (config/gen_chord_gnss_config.py --fleet-trim-windows) allows "
        "more. The slew authority, bandwidth x spacing/2 chips/s (0.35 at the defaults), "
        "must exceed how fast the code moves under the tap: the receiver clock's "
        "drift plus its +-1-chip breathing. With fleet_trim_leak_per_s (cli.py "
        "_FROZEN) it also bounds the standing trim at (spacing/2) x bandwidth / leak "
        "chips.",
    )
    ap.add_argument(
        "--fleet-trim-readback",
        type=int,
        default=0,
        help="nonzero: after each /set_policy POST, GET <fleet-trim-url>/get_dll and "
        "keep the C++ loop's standing per-PRN trims (chips). The tracker despreads at "
        "seed + standing trim; without the readback the innovation (INNOV) and the "
        "hold-escape referee (its CP_ERR line) judge the seed without that trim, the "
        "published rows' dll_trim_cpp reads 0 and the handover-sawtooth detector has "
        "nothing to watch. A failed read clears the trims "
        "rather than holding stale ones. Logged as FLEET-TRIM READBACK. 0 (default) = "
        "off, for transcript replay: a replay must make the GETs its recording made, "
        "and a recording made without the flag has no get_dll entries.",
    )
    ap.add_argument(
        "--n2-combiners",
        default="",
        help="a second population of combiner endpoints (comma list, {a..b} ranges, "
        "bare names on --rest-url), run through the same fleet_coherent as "
        "--dll-combiners and logged beside it as B on the FLEET-COH line. Kept "
        "separate so the two can be compared: both despread the same antenna "
        "voltages, so their noise is not independent and a merged sum would not gain "
        "what it appears to. Measurement only: it publishes nothing and feeds no "
        "loop. Empty (default) = off.",
    )
    # -- Seeds: dead reckoning, fits and the long code (gnss_broker/deadreckon.py, seeding.py) --
    ap.add_argument(
        "--time0-endpoint",
        default=None,
        help="REST path, relative to --rest-url (e.g. telescope/time0_ns), whose "
        "time0_ns is the absolute time of the F-engine's frame 0. Frame 0 is "
        "disciplined to GPS via IRIG-B/PPS, so this anchor, which every hop-to-UTC "
        "conversion uses (dead reckoning, the NH hint and joint fit), is a fact "
        "rather than an estimate. Read under --dead-reckon, once per process for all "
        "its chains, and latched; re-read every 60 s, and a value that changes on 3 "
        "consecutive re-reads (an F-engine restart) stops the broker so that its "
        "supervisor restarts it on the new epoch. Unset (default) = no anchor.",
    )
    ap.add_argument(
        "--dr-clock-chips",
        type=float,
        default=None,
        help="prime the dead-reckon receiver clock with this offset (chips, reduced "
        "modulo the code length) instead of waiting for a solve. A solve is a median "
        "of the code-phase residuals of dr_min_sats or more detected satellites, so a "
        "chain without a search seeds nothing until a prime or an adopted clock "
        "(--dr-clock-adopt) gives it one. With a GPS-disciplined F-engine the offset "
        "is known a priori (0 plus a fixed instrumental and cable delay), so 0.0 is "
        "the starting value. A prime is a seed, not a measurement: a chain whose "
        "detections reach the solve (not under --detectors-monitor) replaces it "
        "outright with its first multi-satellite solve; one "
        "without keeps it until it adopts a sibling chain's clock, or for the whole "
        "run without --dr-clock-adopt. Dead-reckon seeding waits up to "
        "dr_clock_wait_s (cli.py _FROZEN) for that replacement. Unset (default) = no "
        "prime.",
    )
    ap.add_argument(
        "--dr-doppler-ignores-integrity",
        action="store_true",
        help="keep the BRDC (dead-reckon) Doppler in the seed of a detected satellite "
        "whose model is untrusted only because of its code-phase integrity residual "
        "(beyond dr_max_integrity_chips in cli.py _FROZEN, 1 chip or ~30 m at 10.23 "
        "Mcps: ordinary iono plus b_sat). That residual is a code-phase statement and "
        "says nothing about the range rate from the same orbit. A stale ephemeris "
        "still demotes the Doppler, since it is doubtful for range and range rate "
        "alike. Without the flag every trust flip switches the seed Doppler between "
        "two BRDC evaluations (dr and pred), and the seed Doppler is the replica's "
        "carrier, so the switching jitters the carrier phase across the deep fold. "
        "Integrity residuals need detections, so it acts only on a chain with a "
        "search.",
    )
    ap.add_argument(
        "--seed-doppler",
        default="auto",
        choices=("auto", "det"),
        help="which Doppler a detected satellite's seed carries. 'auto' (default): "
        "the model's plus the clock-frequency bias --seed-bias-source picks (the "
        "dead-reckon BRDC Doppler under --dead-reckon while the satellite's model is "
        "trusted, see --dr-doppler-ignores-integrity, else the --almanac prediction), "
        "or the search's for a satellite with no model; smooth, and the same model "
        "that seeds the undetected satellites. 'det': the search's measured Doppler, "
        "over the model. "
        "Use 'det' when the model is not good to well under a Hz over a seed's age: "
        "the tracker extrapolates the seed's code phase at the code rate its Doppler "
        "implies, so each Hz of error walks it chip_rate/carrier chips per second "
        "(0.0087 at L5) for as long as the seed stands. 'det' also makes "
        "cp_to_seed_currency a no-op, since cp0 was fit at exactly that Doppler.",
    )
    ap.add_argument(
        "--long-code-segments",
        type=int,
        default=75,
        help="primary code periods in the long (overlaid) code the trackers despread: "
        "75 for L2C CL (75 x 10230 chips, the default), 20 for GPS L5 Q5 with its "
        "NH20 overlay. The dead-reckon model, the search's long-code phase and the NH "
        "overlay checks and joint fit work modulo this many periods, so it must match "
        "the trackers' "
        "signal, not the search's: a seed reduced to one primary period of a "
        "20-period code is right 1 time in 20. --signal sets it from "
        "lib/stages/gnss/gnssSignal.hpp; an explicit value that disagrees is an "
        "error.",
    )
    ap.add_argument(
        "--long-code-epoch-s",
        type=float,
        default=1.5,
        help="the long code's repeat period, s, locked to GPS time: 1.5 for L2C CL "
        "(the default), 0.02 for L5 with NH20. The NH joint fit (--nh-joint) predicts "
        "each satellite's phase as its GPS transmit time modulo this period, and the "
        "receiver clock it resolves, known modulo this period, is what lets a chain "
        "whose code period is longer than a donor's adopt that donor's clock "
        "(--dr-clock-adopt). --signal sets it from lib/stages/gnss/gnssSignal.hpp; an "
        "explicit value that disagrees is an error.",
    )
    ap.add_argument(
        "--bias-min-snr",
        type=float,
        default=0.0,
        help="detections below this search SNR stay out of the clock-frequency bias "
        "median and the time-base check. The bias is common mode, so its error is "
        "(per-satellite Doppler error)/sqrt(N): with few satellites, one whose Doppler "
        "is noise moves it far. 0 (default) keeps every detection.",
    )
    ap.add_argument(
        "--fit-min-snr",
        type=float,
        default=0.0,
        help="detections below this search SNR stay out of the cp-fit history "
        "(--fit-hist-len), whose slope fit seeds a detected satellite's code phase and "
        "rate. The fit resolves a slope of ~0.01 chips/s and a near-threshold "
        "detection's phase is noise, so one such point sets the slope rather than "
        "degrading it. 0 (default) keeps every detection, which suits a search whose "
        "detections sit well above threshold.",
    )
    ap.add_argument(
        "--fit-hist-len",
        type=int,
        default=8,
        help="detection snapshots kept per PRN for the cp-rate and Doppler-rate fits. "
        "A snapshot is added only for a new detection (a new ref_hop), so at most one "
        "per cycle, and a gap longer than --fit-gap-s clears the history. A cp fit is "
        "trusted (hold admission and the hold escape) only once >= 6 points span "
        "fit_maturity_span_s (cli.py _FROZEN), so the history must cover that span: N "
        "snapshots of a satellite detected every cycle span (N-1) x --interval, and "
        "the default 8 falls short at any --interval below fit_maturity_span_s/7. A "
        "longer history also shrinks the slope error, as sigma/(T*sqrt(N)) for N "
        "points spanning T.",
    )
    ap.add_argument(
        "--hold-on-present",
        type=int,
        default=0,
        help="also admit a satellite to the hold (the cp_held seed freeze: anchor, "
        "rate and at-ref phase frozen, the DLL owns the residual) after this many "
        "consecutive fleet-present cycles, the presence the fleet DLL admits trims on. "
        "The other entry, the lock metric (amp_snr, or deep_snr once the combiner "
        "certifies the deep fold) at or above hold_snr (cli.py _FROZEN), flickers with "
        "the deep fold and can leave a locked satellite unheld. A mature cp fit "
        "(fit_trusted) is still required. While set, presence also sustains a hold: a "
        "held satellite's 3-miss release count advances only on a cycle where both its "
        "lock metric and its fleet presence fail. 0 (default) = off.",
    )
    ap.add_argument(
        "--nh-period-debounce",
        type=int,
        default=0,
        help="a changed overlay period, as the search measures it, must repeat on this "
        "many consecutive detections (the same offset) before it rewrites the seed; "
        "until then the seed carries the measured fine phase with the standing period, "
        "and the period-continuity history (ph_hist) is not updated. The measured "
        "period can toggle by one on a strong satellite whose fine phase is right, and "
        "each single-detection flip would step every node's seed by a whole code "
        "period; a real change passes after this many detections. Under --nh-joint "
        "apply, a resolved consensus overrides it. 0 (default) = off.",
    )
    ap.add_argument(
        "--nh-joint",
        type=str,
        default="off",
        choices=["off", "apply"],
        help="the joint NH overlay fit. The overlay is synchronous at the "
        "transmitters, so the satellites' received overlay alignments differ only by "
        "their predicted transmit times plus one common unknown, the receiver clock "
        "modulo the overlay epoch (--long-code-epoch-s). Each detection at snr >= "
        "nh_joint_min_snr votes for that offset with its phase at its own epoch, in "
        "continuous chips (an integer segment vote would split at segment boundaries); "
        "the consensus is a circular weighted median, resolved once nh_joint_min_prns "
        "votes agree (constants in cli.py _FROZEN). 'apply': with a resolved "
        "consensus, every detection takes its overlay segment from the prediction plus "
        "the consensus and keeps only its measured fine phase, overriding "
        "--nh-period-debounce; every moved segment is logged, a strong detection's as "
        "a warning that the consensus or the prediction may be wrong. While "
        "this chain's search solve is established, the resolved offset is also "
        "published as its clock-mod-epoch record, which --dr-clock-adopt needs on a "
        "chain whose code period is longer than this one's. Needs --almanac and the "
        "capture time anchor (--time0-endpoint, read under --dead-reckon). Default "
        "off.",
    )
    ap.add_argument(
        "--cp-rate-model-primary",
        type=int,
        default=0,
        help="nonzero: every cp fit seeds the pooled code-rate clock offset (l-a, "
        "--code-bias-alpha) as its code rate instead of its own slope. The fit's "
        "position is still seeded, and its slope still feeds the l-a pool and the "
        "--cp-rate-model-tol check. The residual code rate is the receiver clock, "
        "common to every satellite (propagate_seed carries each satellite's own "
        "geometry), so a per-satellite slope adds only fit error. Until the pool has "
        "an estimate the fitted rate is seeded. 0 (default) = the fitted rate is "
        "seeded.",
    )
    ap.add_argument(
        "--cp-rate-model-tol",
        type=float,
        default=0.0,
        help="log as REJECTED, and count toward --fit-flush-on-reject, a fitted code "
        "rate that departs from the pooled code-rate clock offset (l-a) by more than "
        "this, chips/s. A rejection does not change what is seeded: the fit's position "
        "always, and its rate unless --cp-rate-model-primary seeds the clock's. The cp "
        "fit unwraps to the nearest code period, so one mis-wrap puts a whole period "
        "into the slope, and since the residual code rate is the receiver clock, "
        "common to every satellite, a fit far from the pool is wrong. It bounds the "
        "deviation, not the magnitude: an uncalibrated clock runs at several chips/s, "
        "and an absolute ceiling would reject the feed-forward the trim needs. Only a "
        "chain with --detectors fits code rates, and there is no check until the pool "
        "has an estimate. 0 (default) = off.",
    )
    ap.add_argument(
        "--nh-overlay-len",
        type=int,
        default=1800,
        help="length of the secondary (overlay) code in its own chips, one per primary "
        "code period: the modulus of the --nh-hint alignment. 1800 (default) for the "
        "B1C pilot, 100 for the E5a-Q CS100 and the B2a pilot, 20 for L5-Q NH20. "
        "--signal sets it.",
    )
    ap.add_argument(
        "--dead-reckon",
        action="store_true",
        help="seed code phase from the broadcast ephemeris (BRDC) for every visible "
        "satellite the search has not detected: predict its transmit-time code phase, "
        "add the receiver clock, and express it in the seed's Doppler currency. The "
        "clock is solved every dr_refresh_s (cli.py _FROZEN) from the detected "
        "satellites (the circular median of measured minus predicted code phase), else "
        "adopted from another chain (--dr-clock-adopt) or primed (--dr-clock-chips). A "
        "search, where there is one, solves the clock, re-anchors the satellites it "
        "detects through the normal seed path, and checks the model (per-satellite "
        "integrity residuals); the prediction only has to land inside the DLL's "
        "capture range (~0.4 chips). Needs --almanac and --time0-endpoint.",
    )
    ap.add_argument(
        "--dr-constellation",
        default="G",
        choices=("G", "E", "C", "R"),
        help="this chain's constellation as a RINEX letter: the system the dead-reckon "
        "model predicts, the key of its satellites in the receiver-wide joint state, "
        "and the almanac's when --constellation is unset. C also makes --dr-min-prn "
        "default to 19 unless --signal names B3I (signals.prn_floor). Default G; "
        "--signal sets it.",
    )
    ap.add_argument(
        "--signal-capability",
        default=None,
        help="seed, hint and dead-reckon only the satellites whose block broadcasts "
        "this signal. The value is a chain name or signal token (gps_l5, GPS_L1C_P, "
        "glo_l3oc), matched case-insensitively by constellation prefix and signal: GPS "
        "L5 is on IIF and later, L2C on IIR-M and later, L1C on III only; GLONASS L3OC "
        "on the K generations and GLO-M+, L2OC on the K generations only; BeiDou B1C, "
        "B2a and B2b on BDS-3 only. The IGS satellite-metadata registry's block types "
        "decide (prnmap.signal_incapable_prns), read at startup and re-read every 6 h. "
        "For GPS and GLONASS, Celestrak's TLE block names "
        "(gps_beamtrack.signal_capable_prns) are the fallback when the registry "
        "excludes nothing; they mark GLONASS-K but not GLO-M+, so the fallback also "
        "drops the M+ satellites. Both fail open: with no decision from either source "
        "the gate stays off (on a BeiDou chain the --dr-min-prn floor stands in). No PRN floor can express this for GPS, whose "
        "Block III satellites are interspersed among older ones. --signal sets it for "
        "B1C, B2a and B2b. Unset = off.",
    )
    ap.add_argument(
        "--dr-min-prn",
        type=int,
        default=None,
        help="dead-reckon only PRNs >= this: a signal-capability floor, not a "
        "visibility gate. Unset: the almanac's floor (signals.prn_floor), 19 when "
        "--dr-constellation is C (B1C, B2a and B2b are BDS-3 signals, and the PRNs "
        "below 19 are mostly BDS-2), except under --signal bds_b3i (B3I is on BDS-2 "
        "and BDS-3 alike), else 1. It is the "
        "fallback: when --signal-capability's registry gate is in force on a BeiDou "
        "chain the floor is 1 whatever this says, since some PRNs below 19 (C11-C14) "
        "fly BDS-3 satellites. The search never detects a satellite that is not "
        "transmitting, but the model predicts a code phase for any satellite with an "
        "ephemeris; the tracker then despreads noise there, and cross-correlation with "
        "the real satellites can make it look like a plausible satellite. --signal "
        "sets 19 for B1C, B2a and B2b.",
    )
    ap.add_argument(
        "--dr-clock-adopt-max-chips",
        type=float,
        default=0.0,
        help="bound, in chips, on adopting another chain's receiver clock "
        "(--dr-clock-adopt): while the local clock is fresh (taken from a donor, or "
        "confirmed by a JOINT-CLK adoption, within the last 300 s), a donor clock "
        "further than this from it is refused and the local clock is held at zero rate "
        "instead of extrapolated. Once the local clock is 300 s stale the donor is "
        "adopted anyway. A primed clock (--dr-clock-chips) is not defended, and a step "
        "the donor declares (a new clock epoch from its own bootstrap, re-roll or "
        "re-pin) is adopted at any size. A cross-band bootstrap is bounded as a time: "
        "this many of the donor's chips (5 = 489 ns at 10.23 Mcps), converted into "
        "this chain's (2.5 at E6, 0.25 at L2C-CM). 5 matches the JOINT-CLK bound "
        "(joint_clk_max_chips, cli.py _FROZEN). 0 (default) = unbounded.",
    )
    # -- Navigation message decode (gnss_broker/navbits.py) ------------------------------------
    ap.add_argument(
        "--nav-bits",
        type=int,
        default=1,
        help="nonzero: decode the broadcast navigation message from the nav_obs the "
        "combiner exports (bit_export: true in its config) with the --nav-decoder "
        "decoder, and cross-check each decoded ephemeris against BRDC when the BRDC "
        "almanac is loaded (--almanac, --almanac-source brdc). The auxiliary "
        "combiners (--cnav-combiner and the rest) are polled only while this is on. "
        "Self-gating: a chain whose combiner exports no nav_obs is untouched. 0 = off.",
    )
    ap.add_argument(
        "--nav-decoder",
        default="lnav",
        choices=("lnav", "cnav", "bcnav3"),
        help="which decoder takes the nav_obs of this chain's own --combiner. lnav "
        "(default): GPS L1 C/A LNAV; it also scores each newly decoded subframe "
        "against its prediction from the earlier ones (logged as predict-mismatch). "
        "cnav: GPS CNAV (L2C-CM, L5-I), FEC + CRC-24Q, decoded for the ephemeris "
        "(message types 10 and 11); a chain whose combiner exports CNAV symbols needs "
        "it, since the LNAV frame sync never finds a preamble in them. bcnav3: BeiDou "
        "B2b B-CNAV3 (GF(64) NB-LDPC(162,81) + CRC-24Q), the B2b chain's own message. "
        "Each decoded ephemeris is cross-checked against BRDC. The auxiliary "
        "combiners' symbols go to their own decoders whatever this says.",
    )
    ap.add_argument(
        "--cnav-combiner",
        default=None,
        help="auxiliary combiner (name or URL) polled only for its CNAV nav_obs, in "
        "addition to --combiner, for a band whose CNAV symbols are on a chain other "
        "than this broker's own: at L5 the broker's own combiner is the Q pilot, whose "
        "nav_obs are deterministic overlay signs, while the CNAV symbols come from the "
        "derived L5-I data sibling. Its symbols go to the CNAV decoder whatever "
        "--nav-decoder says, and the decoded ephemeris is cross-checked against BRDC "
        "on the 60 s health cadence: an independent decode of the message set L2C also "
        "carries. Needs --nav-bits. Unset = off.",
    )
    ap.add_argument(
        "--inav-combiner",
        default=None,
        help="auxiliary combiner (name or URL) polled for Galileo I/NAV nav_obs, as "
        "--cnav-combiner is for CNAV: the Galileo broker's own --combiner is a pilot "
        "(E1C or E5b-Q), whose nav_obs are deterministic overlay signs, while the "
        "I/NAV data symbols come from the derived data sibling chain (E1B or E5b-I; a "
        "combiner name containing 'e5b' labels the decode health GAL_E5BI_INAV, else "
        "GAL_E1B_INAV). "
        "Symbols go to the I/NAV decoder (InavPredictor), and the decoded ephemeris is "
        "cross-checked against BRDC on the 60 s health cadence. Needs --nav-bits. "
        "Unset = off.",
    )
    ap.add_argument(
        "--fnav-combiner",
        default=None,
        help="auxiliary combiner (name or URL) polled for Galileo E5a-I F/NAV nav_obs, "
        "as --inav-combiner is for I/NAV: the E5a broker's own --combiner is the E5a-Q "
        "pilot (deterministic CS100 overlay signs), while the F/NAV data symbols come "
        "from the derived E5a-I sibling chain (CS20 secondary + nav wipe). Symbols go "
        "to the F/NAV decoder (FnavPredictor), and the decoded ephemeris is "
        "cross-checked against BRDC on the 60 s health cadence. Needs --nav-bits. "
        "Unset = off.",
    )
    ap.add_argument(
        "--bcnav2-combiner",
        default=None,
        help="auxiliary combiner (name or URL) polled for BeiDou B2a B-CNAV2 nav_obs: "
        "the B2a broker's own --combiner is the B2a-P pilot (deterministic Weil "
        "overlay signs), while the B-CNAV2 data symbols come from the derived B2a-D "
        "sibling chain (CS5 secondary + nav wipe). Symbols go to the B-CNAV2 decoder "
        "(Bcnav2Predictor, a GF(64) NB-LDPC codec), and the decoded ephemeris is "
        "cross-checked against BRDC on the 60 s health cadence. Needs --nav-bits. "
        "Unset = off.",
    )
    ap.add_argument(
        "--bcnav1-combiner",
        default=None,
        help="auxiliary combiner (name or URL) polled for BeiDou B1C B-CNAV1 nav_obs: "
        "the B1C broker's own --combiner is the B1C-P pilot, while the B-CNAV1 data "
        "symbols come from the derived B1C-D sibling. Symbols go to the B-CNAV1 "
        "decoder (Bcnav1Predictor: the GF(64) NB-LDPC codec shared with B-CNAV2, with "
        "its own H matrix; the ephemeris is in subframe 2), and the decoded ephemeris "
        "is cross-checked against BRDC on the 60 s health cadence. Needs --nav-bits. "
        "Unset = off.",
    )
    ap.add_argument(
        "--cnav2-combiner",
        default=None,
        help="auxiliary combiner (name or URL) polled for GPS L1C-D CNAV-2 nav_obs: "
        "the L1C broker's own --combiner is the L1C-P pilot (deterministic L1CO "
        "overlay signs), while the CNAV-2 data symbols come from the derived L1C-D "
        "sibling. Symbols go to the CNAV-2 decoder (Cnav2Predictor: systematic LDPC "
        "extract + CRC-24Q, 18 s frame), whose health (CRC-passing frames, TOI) is "
        "logged on the 60 s cadence. It extracts no ephemeris, so there is no BRDC "
        "cross-check. Needs --nav-bits. Unset = off.",
    )
    # -- Named signals (gnss_broker/signals.py) -------------------------------------------
    ap.add_argument(
        "--signal",
        default=None,
        metavar="KEY",
        help="name this chain's signal (e.g. gps_l5) instead of typing its constants: "
        "it sets --carrier-hz, --chip-rate-hz, --code-length, --long-code-segments, "
        "--long-code-epoch-s, --nh-overlay-len, --constellation and --dr-constellation "
        "from lib/stages/gnss/gnssSignal.hpp, the descriptors the trackers' replica "
        "bank is built from, and for the BDS-3-only signals (B1C, B2a, B2b) "
        "--dr-min-prn 19 and --signal-capability. These constants fail silently when "
        "mistyped (a wrong overlay length seeds a random one of the overlay periods), "
        "so an explicit flag that disagrees with the named signal is an error. The "
        "name is also the chain's id on the publisher and in the shared Receiver. "
        "`--signal help` lists the known signals.",
    )
    # -- Record and replay for the broker_equiv.py gate (_Transcript, gnss_broker/transport.py) --
    ap.add_argument(
        "--transcript-write",
        default=None,
        metavar="FILE",
        help="record every clock read, GET and POST to FILE (JSONL, with this run's "
        "flags in the header) while running normally, for scripts/gnss/broker_equiv.py "
        "to replay. Recording changes nothing the broker does, so it is safe on a "
        "production run for the minute a capture takes.",
    )
    ap.add_argument(
        "--transcript-read",
        default=None,
        metavar="FILE",
        help="replay a transcript recorded with --transcript-write (FILE; .gz is read "
        "directly) instead of talking to the fleet, then exit when it runs out, "
        "printing the digest of the POST stream. No network and no sleeps; the clock "
        "is the recording's. scripts/gnss/broker_equiv.py runs it under the "
        "recording's flags and compares the digest with its blessed goldens.",
    )
    return ap
