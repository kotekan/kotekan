"""The broker's command-line surface: the argument parser and the frozen tuning constants.

Neither depends on anything in the broker, so they live here rather than in
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
    # The emit period is derived from the stream, as the smallest pow_hop step seen.
    carrier_rate_max_gap=2.0,
    # A rate residual that moved more than this (Hz) since the PRN's previous sample is the
    # tracker's f_ref fence re-anchoring mid-track, which no window gap marks: re-baseline and skip.
    # Contiguous emits step under 1 Hz and a fence re-anchor by several Hz, so 3 Hz separates them.
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
    # When the network BRDC fetch fails, keep the dead-reckon predict alive on this broker's own
    # decoded ephemeris (decoded_eph.py, which reuses each decoder's BRDC-validated propagator).
    # Geometry and Doppler are exact; the precise satellite clock is complete for Galileo, best
    # effort elsewhere. 0 lets the predict go dark.
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
    # is believed: signal reads ~18-22, noise ~3-6. A weak satellite does not merely scatter, it
    # lands on the wrong spectral bin, tens of Hz out.
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
    # while refusing a wrap alias or a diverged filter. In the joint state's chips (its feeders',
    # 10.23 Mcps) on every chain, so 5 = 489 ns everywhere; a chain at another chip rate converts
    # before comparing (deadreckon.dr_joint_clk).
    joint_clk_max_chips=5.0,
    # 'clk' consumer: refuse a joint clock whose 1-sigma exceeds this (joint-state chips). P grows
    # while the state is unfed, so this one gate covers both estimator health and staleness; a
    # healthy state reads under 0.1.
    joint_clk_max_sigma=0.5,
    # 'clk' consumer step guard: a joint clock more than joint_clk_max_step_chips (joint-state
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
    # signal and no detection can arrive to correct it. 600 s is about half the 1276 s per-PRN
    # revisit, so a healthy fleet always has fresh samples and only a starved one widens.
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
    # by more than this (s). Log only, never a control input. The orbit model is evaluated at wall
    # time, so a wall step moves every model seed on every chain at once. The observable F-engine
    # 'now' is the newest telemetry pow_hop, which trails the sky by the gather/serve latency
    # (~100 ms, tens of ms of jitter), so this sees only steps well above that. The same lag is why
    # orbit evaluation stays on wall time: the axis would trade ~1.5 ms of NTP error for ~100 ms of
    # pipeline lag.
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
    # Withhold every measurement from the joint state (and MINNOV) for this many seconds after
    # broker start. Birth is the one path with no innovation to gate on, and establishment-phase
    # measurements are the worst the feed sees: garbage born into the state is self-consistent, so
    # no escape fires. Seeds and holds settle in ~3 min; 240 s is that plus margin. Only the feed
    # waits; the state object is created at once.
    joint_feed_warmup_s=240.0,
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
    # for up to the 1276 s per-PRN revisit, so a satellite that has left the beam keeps its last
    # good SNR while its tracker output is noise. Noise rows sit at deep_snr 2-3 / coh_frac ~0.2 and
    # signal at 24+ / 0.9+; 10 is in the gap.
    joint_min_deep_snr=10.0,
    # Minimum detection SNR for a residual to enter the joint solve (and for MINNOV), equal to
    # period_check_snr: below it a detection's phase is noise with ~2000-chip residuals, and one
    # such detection can walk the state's clock rate away.
    joint_min_snr=60.0,
    # 'slew' consumer (--joint-consume): refuse the joint slew target if it disagrees with the
    # legacy target (clock EMA + SatBiasFilter) by more than this. Both describe the same receiver
    # and b_sat runs +-3-6 chips, so a larger gap means one estimator is broken. In the joint
    # state's chips (5 = 489 ns at 10.23 Mcps), the same time on every chain, like
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
    # --presence-admit-displaced: the offset-blind evidence bar, as a multiple of the combiner's
    # deep_floor. The evidence must not be an on-peak statistic, since prompt power is suppressed by
    # exactly the offset being rescued; deep_snr re-searches code phase, so it sees the satellite
    # wherever the tap sits. Prompt power over the probe floor is accepted as an alternative; the
    # pedestal and offset tests are what refuse a centred noise realisation.
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


def build_parser(description):
    """The broker's argument parser; `description` is the --help text (the broker's docstring)."""
    ap = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--rest-url",
        default="http://localhost:12048",
        help="default kotekan REST base for bare stage names",
    )
    ap.add_argument(
        "--detectors",
        default="",
        help="detection endpoints: aggregator/search stage names, absolute "
        "URLs, or {a..b} ranges (e.g. aggregate  or  search_{00..49}). "
        "OPTIONAL: leave empty to run PURELY MODEL-PRIMARY, seeding every "
        "visible satellite from the BRDC model with no acquisition search at "
        "all (--almanac --dead-reckon). That is the CHORD configuration -- the "
        "chain there has no search stage, because the station position is known "
        "to millimetres and the model already lands inside the DLL capture "
        "range, so a blind search would only re-derive what the ephemeris "
        "already says.",
    )
    ap.add_argument(
        "--trackers",
        required=True,
        help="tracker endpoints to seed (e.g. track_{00..49} or "
        "http://nodeA:12048/track_{0..24},http://nodeB:12048/track_{0..24})",
    )
    ap.add_argument(
        "--combiner",
        default="combiner",
        help="combiner endpoint for full-band |A| status (name or URL)",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help="control-loop period, s (keep below ~0.5 s drift time)",
    )
    ap.add_argument(
        "--acquire-snr",
        type=float,
        default=12.0,
        help="min detection SNR to (re)seed a PRN",
    )
    ap.add_argument(
        "--noise-probes",
        type=int,
        default=0,
        help="keep this many deepest-below-horizon PRNs seeded as NOISE PROBES: "
        "the combiner then emits genuine signal-free records for the beam "
        "map's pedestal calibration (an almanac-gated broker otherwise never "
        "tracks one and the pedestal falls back to a signal percentile). "
        "Probes fail every lock gate naturally; ~2 is plenty.",
    )
    ap.add_argument(
        "--post-sat-geometry",
        type=int,
        default=0,
        help="#102: post each satellite's az/el to the record assemblers' "
        "/set_sat_geometry every ~30 s (endpoint derived from "
        "--dll-combiners), feeding the per-element steering. 0 = off.",
    )
    ap.add_argument(
        "--fit-flush-on-reject",
        type=int,
        default=0,
        help="#100: after this many CONSECUTIVE cp-rate rejections for a PRN, "
        "FLUSH its cp-fit history. A rejected fit also poisons the seed's "
        "POSITION (the fit evaluated at ref_hop: slope error x history span "
        "= thousands of chips), and the poison self-sustains -- off-peak "
        "command, weak detections, re-poisoned history -- for as long as "
        "the history remembers (~8 min at fit-hist-len 256; G9 2026-08-28 "
        "22:36-23:00). Flushed, the sat rides the birth path while a clean "
        "fit rebuilds (~30 s). 0 disables (pre-#100 behaviour).",
    )
    ap.add_argument("--lat", type=float, help="receiver latitude (enables gating)")
    ap.add_argument("--lon", type=float, help="receiver longitude")
    ap.add_argument("--alt", type=float, default=0.0, help="receiver altitude, m")
    ap.add_argument("--mask-deg", type=float, default=5.0, help="elevation mask, deg")
    ap.add_argument(
        "--almanac",
        action="store_true",
        help="orbit-predict each PRN's Doppler (needs --lat/--lon + skyfield): "
        "seed trackers with the precise predicted Doppler (geometry + a common "
        "clock-freq bias solved from the measured sats) instead of the coarse "
        "search grid, and gate to visible sats. Code phase still from the search.",
    )
    ap.add_argument(
        "--almanac-source",
        default="brdc",
        choices=("brdc", "tle"),
        help="orbit source for the almanac assist. brdc (default) = the IGS "
        "broadcast-ephemeris file (gnss_ephemeris): PRN-INDEXED (immune to the "
        "celestrak name->PRN label rot that mislabels BeiDou IGSOs -- C31/C38/"
        "C39 ranges were 16-20k km off on 2026-07-17, corrupting nh hints by "
        "5-7 overlay chips and hiding C38's true 84-deg pass), ~m orbits vs "
        "TLE's ~km, includes the sat clock, and needs no skyfield. tle = the "
        "legacy celestrak path (also the automatic fallback when BRDC is "
        "unreachable at startup).",
    )
    ap.add_argument(
        "--boresight",
        default="nominal",
        choices=("nominal", "none"),
        help="where the dishes point. nominal (default) = CHORD's fixed boresight (az 180, "
        "el 81.41), which the D3 transit veto, the dead-reckon clock's transit freeze and the "
        "assemblers' model-formation freeze key on. none = the dishes are not pointed (tipped "
        "for site work): no satellite is ever near boresight, so all three stay off.",
    )
    ap.add_argument(
        "--constellation",
        default=None,
        choices=("G", "E", "C", "R"),
        help="constellation letter for the BRDC almanac (which file covers ALL "
        "systems, unlike the per-group TLE URLs). Default: --dr-constellation. "
        "C keeps the BDS-3 satellites per the IGS registry's block types "
        "(--signal-capability; PRN >= 19 is only the fallback): the BDS-2 "
        "birds transmit B1I at 1561 MHz, outside our band.",
    )
    ap.add_argument(
        "--carrier-hz", type=float, default=1575.42e6, help="carrier for Doppler pred"
    )
    ap.add_argument(
        "--narrow-search",
        action="store_true",
        help="push per-PRN predicted Doppler to the detectors' /set_doppler_hints so the "
        "search scans only doppler +- margin (almanac-narrowed acquisition) instead "
        "of its blind grid; needs --almanac. Far cheaper + more sensitive.",
    )
    ap.add_argument(
        "--search-margin-hz",
        type=float,
        default=500.0,
        help="narrow search half-window once the clock-freq bias is solved (Hz)",
    )
    ap.add_argument(
        "--search-margin-wide-hz",
        type=float,
        default=3000.0,
        help="wider search half-window BEFORE the bias is solved (covers the unknown "
        "TCXO offset; shrinks to --search-margin-hz once a few sats pin the clock)",
    )
    ap.add_argument(
        "--dr-clock-adopt",
        action="store_true",
        help="ADOPT the receiver clock from a co-hosted sibling chain (through "
        "the shared Receiver: the same band first, else across bands as a "
        "bootstrap) instead of holding the primed --dr-clock-chips constant "
        "forever.\n"
        "\n"
        "WHY: a chain with --detectors EMPTY cannot solve its own clock. The "
        "bootstrap takes a median of measured code-phase residuals from "
        "/get_detections, so with no detector `offs` is always empty, the EMA "
        "never runs, and the primed constant stands for the WHOLE RUN -- which "
        "the --dr-clock-chips help already says in as many words. That is "
        "every multi-constellation chain (E5a/B2a have no blind acquisition at "
        "all), so today the number is read out of the GPS broker's log by a "
        "human and pasted in, and it is wrong after any F-engine restart. "
        "A chain WITH detections ignores this flag; it solves its own.",
    )
    ap.add_argument(
        "--decode-health-file",
        default=None,
        help="publish this broker's NAV-DECODE health as JSON (atomically "
        "replaced) for the viewer + the eventual decoded-eph BRDC fallback: "
        "per (signal, PRN) whether SYNCED, whether a full ephemeris was "
        "EXTRACTED, the BRDC-agreement residual dpos_m, and decode freshness. "
        "WRITE-ONLY; the viewer reads the directory "
        "and merges. Off when unset.",
    )
    ap.add_argument(
        "--seed-bias-source",
        choices=("ema", "zero"),
        default="ema",
        help="which clock-freq bias the SEED consumers ride (#105). 'ema' = the "
        "hint EMA, the old shared-number behaviour; it commands the +-10 Hz "
        "quantization wander of the solve into every replica's code rate "
        "(measured: the wander integrates ~1 chip off-peak fleet-wide every "
        "~5 min = the #105 q-crash bursts). 'zero' = seeds ride exactly 0 Hz "
        "whatever the solve says, and never snap; the search hints keep the "
        "solved bias. "
        "For an LO locked to the site reference, where the true offset is ~0 "
        "and the solve is estimator noise: with the carrier loop open, the "
        "seeded bias IS every satellite's residual carrier. The first-seed "
        "guard does not wait for a solve under 'zero' (no seed carries one). "
        "Noise probes, which have no signal to stay coherent with, keep the "
        "hint bias.",
    )
    ap.add_argument(
        "--code-length",
        type=float,
        default=1023.0,
        help="spreading-code length (chips) for cp0 unwrap/fit (L1 C/A = 1023)",
    )
    ap.add_argument(
        "--hops-per-sec",
        type=float,
        default=125000.0,
        help="F-engine hops/s (Fs/fft_len) -- cp slope chips/s log + the code clock-bias estimate",
    )
    ap.add_argument(
        "--chip-rate-hz",
        type=float,
        default=1.023e6,
        help="spreading chip rate (L1 C/A 1.023e6) -- for the code-rate clock-bias estimate",
    )
    ap.add_argument(
        "--code-bias-alpha",
        type=float,
        default=0.05,
        help="EMA weight for the receiver code-rate clock offset (l-a); slow -> tracks TCXO/OCXO drift",
    )
    ap.add_argument(
        "--code-bias-min-sats",
        type=int,
        default=2,
        help="fitted sats needed before the pooled code-rate clock offset is trusted + seeded to weak sats",
    )
    ap.add_argument(
        "--presence-admit-displaced",
        action="store_true",
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
        "the offset-blind detector).",
    )
    # ---- Live PRN membership (gnss_broker/prnmap.py, docs/CHORD_LIVE_PRN_RECONFIG.md) ----
    # Keeps the nodes' slot->PRN maps in step with the BRDC sky (--prn-reconfig), and can limit
    # the noise probes to PRNs the nodes hold a slot for (--probe-require-slot).
    ap.add_argument(
        "--probe-require-slot",
        action="store_true",
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
        "split fleet, leaves the selector exactly as it was. ⚠️ The map comes "
        "from --prn-reconfig apply's /get_prns sweep; with --prn-reconfig off "
        "there is no map and the selector stays unfiltered.",
    )
    # A satellite with ephemeris that the registry says cannot broadcast this signal
    # (--signal-capability) is signal-free at any elevation: a noise probe that never sets.
    ap.add_argument(
        "--probe-dark",
        action="store_true",
        help="take noise probes first from the DARK satellites: predicted by BRDC but "
        "excluded by --signal-capability, so they carry nothing on this signal at any "
        "elevation (GLONASS M+ on L2OC). In PRN order, ahead of the deepest below -15 "
        "deg. For a chain with too few capable satellites to keep the >= 3 probes the "
        "presence gate needs below the horizon: glo_l2oc's six live K satellites leave "
        "fewer than 3 below -15 deg about a third of the day, and the chain would read "
        "UNANCHORED then. The nodes must hold slots for the dark PRNs (the manifest's "
        "list); the PRN map keeps them as probes. Needs --signal-capability.",
    )
    ap.add_argument(
        "--prn-reconfig",
        choices=["off", "apply"],
        default="off",
        help="keep the nodes' slot->PRN map in step with live BRDC. off (default) "
        "= this stage does nothing (the replay fixtures' argv predates the "
        "flag, so it must stay the default). apply = GET each node's map and "
        "POST /set_prns. "
        "⚠️ A SWAP COSTS A FULL RE-ACQUISITION on that slot (the node resets "
        "code table, Phi cache, carrier NCO, seed, trim, power averages and "
        "element cal COLD -- correctly, since none of it describes the new "
        "satellite), so every threshold below exists to make swaps RARE.",
    )
    ap.add_argument(
        "--prn-reconfig-interval-s",
        type=float,
        default=900.0,
        help="minimum seconds between swaps, fleet-wide. ONE slot moves per "
        "interval: re-acquisitions are the cost, and a mechanism that can "
        "churn several at once is the #92 disease one level up.",
    )
    ap.add_argument(
        "--prn-reconfig-admit-deg",
        type=float,
        default=10.0,
        help="a satellite with no slot must be ABOVE this to claim one. Not "
        "'rises eventually': the swap is only worth paying for a satellite we "
        "can use immediately, and one that is up now is up again tomorrow.",
    )
    ap.add_argument(
        "--prn-reconfig-evict-deg",
        type=float,
        default=0.0,
        help="a slot's incumbent counts as DOWN below this elevation.",
    )
    ap.add_argument(
        "--fit-gap-s",
        type=float,
        default=16.0,
        help="reset the cp-fit history across a detection gap longer than this "
        "(seconds of capture time; converted via --hops-per-sec)",
    )
    ap.add_argument(
        "--carrier-max-hz",
        type=float,
        default=40.0,
        help="clamp on the commanded carrier trim (Hz)",
    )
    ap.add_argument(
        "--dll-gain",
        type=float,
        default=0.25,
        help="code delay-lock-loop gain (0 = off): each poll, nudge a persistent "
        "per-PRN cp TRIM by gain * tau_est from the combiner's E/L discriminator. "
        "The trim rides on top of the search-fit cp, converging to the fit's "
        "grid-quantization bias -- sub-chip code tracking with no per-record "
        "decisions (R1, 3c2e98f43:docs/gnss_architecture_audit.md).",
    )
    ap.add_argument(
        "--nh-hint",
        action="store_true",
        help="hint the search's secondary-code alignment from the EPHEMERIS, so the "
        "acquire scans nh_hint_span alignments instead of all 20 (~92%% of a "
        "pass). nh at transmit is round((gpst - range/c + clk_sv)/period) mod "
        "overlay_len, leaving ONE global constant, the receiver clock reference, which "
        "any detection measures and every satellite shares. Works for sats "
        "NEVER detected, which is where a full 20-way scan hurts most. "
        "Requires --almanac and the capture time anchor.",
    )
    ap.add_argument(
        "--joint-p2c-rotate",
        action="store_true",
        help="P2c, SELF-DRIVING. Withhold the best-established satellite from the "
        "joint solve, coast it for --joint-p2c-hold-s, log the "
        "residual-vs-age curve, release, and rotate to one not tested "
        "recently. Hand-picking one PRN is a sample of ONE satellite at "
        "ONE geometry: on a transit "
        "instrument the first two runs disagreed outright (flat -0.19+-0.44 "
        "chips over 848 s, then -2.70 over 900 s on the same satellite) and "
        "a third produced nothing because the named PRN had set. What this "
        "accumulates instead is a DISTRIBUTION over satellites and "
        "elevations -- and the residual's growth with age IS the b_sat "
        "random walk, so it MEASURES q_b (currently the guessed 0.013).",
    )
    ap.add_argument(
        "--publish-port",
        type=int,
        default=0,
        help="serve the FLEET-MERGED per-PRN state on this port (0 = off): "
        "GET /get_status returns rows in GnssCoherentCombiner's schema, so "
        "the viewer consumes them unchanged, but built from ALL "
        "--dll-combiners instead of one instance's 6.7%% of the L5 lobe. The "
        "browser cannot merge 14 origins across 8 hosts itself, and the "
        "broker is already the shared-knowledge node (pooled l-a, clock-freq "
        "bias, fused LO, cross-band assist) -- this is the same kind of "
        "object. Coherent statistics are best-of-instance, not merged, and "
        "say so via coh_src; see FleetPublisher.",
    )
    ap.add_argument(
        "--spectrum-stale-margin",
        type=int,
        default=0,
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
        "margin. Excluded instances are named in the dropped list.",
    )
    ap.add_argument(
        "--spectrum-endpoints",
        default="",
        help="FLEET PHASE-SLOPE DELAY FIT (task #32, docs/CHORD_JOINT_TRACKING.md "
        "P1): comma-separated GnssGpuRecordAssemble endpoints ({a..b} ranges "
        "expanded) whose /get_spectrum -- the per-channel prompt sums BEFORE "
        "the cross-channel combine -- feed the joint (tau, phi_i) fit. A "
        "delay is a phase ramp across frequency; the fleet's UNION of combs "
        "is what suppresses the 3.27-chip grating lobes a single instance's "
        "stride-16 comb cannot resolve. MEASUREMENT ONLY today: logged as "
        "SPEC-FIT, feeding no loop. Empty = off, and every recorded transcript replays "
        "byte-identically (replay is strict-ordered; this flag not being in "
        "an old transcript's argv is what keeps the new GETs out of replay).",
    )
    ap.add_argument(
        "--dcb-bias",
        action="store_true",
        help="use the MGEX (CAS) measured differential code biases in place of "
        "the broadcast TGD/BGD, per satellite, where the product covers the "
        "signal. THE POINT IS BeiDou B2a, whose broadcast term (TGD_B2ap) is "
        "a B-CNAV2 parameter that RINEX 3 does not carry at all -- so without "
        "this it gets NO group-delay correction. ⚠️ The product is ZERO-MEAN "
        "PER CONSTELLATION by construction, so it supplies the per-satellite "
        "spread and can never explain a constellation-common offset; measured "
        "2026-08-23 the BeiDou spread is -0.006..+0.121 chips against a "
        "+0.74..+1.3 chip common trim. Needs an Earthdata token "
        "(~/.cache/kotekan_gps/.earthdata_token); without "
        "one it logs and falls back to the broadcast term.",
    )
    ap.add_argument(
        "--joint-consume",
        default="",
        help="P2b: comma-separated JOINT-state consumers to switch LIVE, one "
        "name per commit so each is A/B-able on its own. Any name runs the "
        "JOINT receiver-state solve (task #33, docs/CHORD_JOINT_TRACKING.md "
        "section 3a) -- one [clk, clk_rate, b_sat[i]] estimated together from "
        "the dead-reckon integrity residuals, receiver-scope so all three "
        "constellations feed ONE clock; empty (default) runs none. Names: "
        "'rate' (seeded code-rate clock from "
        "clk_rate instead of the l-a EMA -- which measured ~80x biased, "
        "see the call site); 'slew' (the dead-reckon SLEW TARGET's "
        "clock+bias offset from clk + b_sat instead of the clock EMA plus "
        "the SatBiasFilter -- the half of the state P2c validates, and the "
        "half that survived the 2026-08-10 gauge collapse); later 'clock'. "
        "The shadow comparison for each is logged whether or not it is "
        "consumed, so the A/B exists before the switch.",
    )
    ap.add_argument(
        "--rrate-state",
        action="store_true",
        help="P3 step 2 (task #33): FEED the receiver-wide joint state's "
        "per-satellite ORBITAL range-rate rows (m/s, band-shared) from "
        "this chain's deep_rate_hz residuals -- the validated carrier "
        "measurement (q-gated, continuity-gated, slew-gated by "
        "rate_residuals; NEVER carrier_hz_resid, which is documented "
        "signal-free at CHORD SNR). Measurement-only: cannot change a "
        "seed or a POST. This is the shadow phase, and the JRR log line "
        "is where sky validation happens. Per-CHAIN key on purpose: "
        "E5a+E5b feeding lands both bands on ONE row per Galileo sat, "
        "which is the cross-band combination the state exists for.",
    )
    ap.add_argument(
        "--rrate-command",
        action="store_true",
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
        "beats freshness, measured (deep_snr 221->17 on a 10 s re-pin).",
    )
    ap.add_argument(
        "--rrate-feed-applied",
        type=int,
        default=1,
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
        "explicitly (command-HELD gate + measured applied trim).",
    )
    ap.add_argument(
        "--rrate-cmd-min-sig",
        type=float,
        default=0.0,
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
        "below the command site.",
    )
    ap.add_argument(
        "--rrate-phase-feed",
        action="store_true",
        help="#33 PLL fine stage: feed the joint state's rrate rows from the "
        "ADR's residual half (res_cycles per adr_records -- ~5 mHz over "
        "a 20 s span) on sats that pass the fine gates: coarse loop "
        "converged (|coarse resid| < 0.3 Hz), command HELD over the span "
        "(the fine value's reference is only exact under a constant "
        "command), same arc, counter advanced. The coarse deep_rate_full "
        "feed stays as acquisition. SIGN IS CALIBRATED, NOT ASSUMED: "
        "requires --rrate-phase-sign nonzero, which is set from the "
        "JRRP shadow line's live fine-vs-coarse comparison -- closing a "
        "loop on an uncalibrated observable is how this day started.",
    )
    ap.add_argument(
        "--rrate-phase-sign",
        type=float,
        default=0.0,
        help="sign (+1/-1) mapping the combiner's internal (r2c-flipped) "
        "res_cycles rate onto the deep_rate_full_hz convention. 0 = "
        "uncalibrated: the JRRP shadow comparison still logs, the fine "
        "feed stays off regardless of --rrate-phase-feed.",
    )
    ap.add_argument(
        "--rrate-phase-span-s",
        type=float,
        default=0.0,
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
        "--rrate-phase-feed is armed, the feed -- same computed value.",
    )
    ap.add_argument(
        "--rf-stats-endpoints",
        default="",
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
        "not 'no RFI'.",
    )
    ap.add_argument(
        "--rfi-stats",
        action="store_true",
        help="Also poll each rf-stats instance's spectral-kurtosis RFI metrics "
        "(rfi_sk_metrics/sk_metrics_{gpu}) and fold an SK summary into get_rf. "
        "Default off; rides the SAME endpoints as --rf-stats-endpoints (the SK "
        "url is derived per instance), so no second endpoint list.",
    )
    ap.add_argument(
        "--rf-bands",
        default="",
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
        "and monitors the union of every chain on it.",
    )
    ap.add_argument(
        "--drop-stats",
        action="store_true",
        help="Also poll each rf-stats instance's node /metrics for drop counters "
        "(gnss{gpu} srch/telem buffer-send drops, node dpdk rx-missed and "
        "ring-full) and fold them into get_rf. Default off. One /metrics fetch "
        "per HOST per poll, shared across its two GPU instances.",
    )
    ap.add_argument(
        "--rrate-coarse-deweight",
        type=float,
        default=8.0,
        help="THE FLL->PLL HANDOFF. While a satellite holds a fresh fine lock "
        "(see --rrate-fine-hold-s), its COARSE measurements enter at "
        "sigma x this factor. Both feeds keep running -- the coarse one "
        "is what re-acquires after a cycle slip or a re-seed -- but a "
        "60 mHz observable must not out-vote a 16 mHz one at equal "
        "weight, which is exactly what pinned phi0 at 0.4-1.1 rad while "
        "the fine feed fired once a minute. 8x sigma = 64x variance, so "
        "one fine sample outweighs ~64 coarse ones; the row is then "
        "governed by phase and merely SUPERVISED by frequency. 1.0 "
        "disables the handoff (both feeds at face value).",
    )
    ap.add_argument(
        "--dr-fengine-axis",
        type=int,
        default=0,
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
        "fixtures that exercise dr paths re-bless.",
    )
    ap.add_argument(
        "--dr-forecast-lead-s",
        type=float,
        default=0.0,
        help="#46/A1 THE FORECAST EPOCH. Build dead-reckon seeds for the hop "
        "H = (newest telemetry pow_hop) + this many seconds, instead of for "
        '"now". 0 = off (the replay gate\'s requirement, and exactly the '
        "historical behaviour). A seed is a forecast WITH A LABEL "
        "(ref_hop, phase, doppler, rate) and is defined entirely on the hop "
        'axis, so it never needed a notion of "now" -- asking for one is '
        "what dragged the wall clock, NTP and the telemetry lag into the "
        "seed path, and it is why an ephemeris-epoch substitution measured "
        "65x WORSE (reverted 2db3eec1f) and re-admitting dr-owned seeds to "
        "INNOV was falsified in 20 minutes (09a8dc52e). Forecasting to a "
        "CHOSEN hop removes the "
        "question: the measured ~217 ms lag (105 ms window quantisation + "
        "~100 ms pipeline, 59 ms IQR, occasional 1 s excursions) no longer "
        "enters the arithmetic and only sets how large this lead must be. "
        "Pick it to exceed transport + install with margin -- 1-2 s is "
        "ample against a 0.2-0.3 s lag -- but not so large that the "
        "linear range-rate propagation in cp_predicted degrades (range "
        "acceleration ~0.2 m/s^2 gives ~0.4 m = 0.013 chips at 2 s).",
    )
    ap.add_argument(
        "--model-primacy-max",
        type=int,
        default=0,
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
        "chips banded over 600 s.",
    )
    ap.add_argument(
        "--rrate-kcoh-feed",
        type=int,
        default=0,
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
        "armed in the yaml globals.",
    )
    ap.add_argument(
        "--seed-phase-transport",
        action="store_true",
        help="#45 step 6: dead-reckon/slew seeds carry code_phase_at_ref_chips "
        "(the PHASE at ref_hop) in addition to the sample-0 argument. "
        "propagate_seed prefers the phase, which has no t_abs lever, so a "
        "producer that edits the Doppler without re-projecting cp0 can no "
        "longer walk the despread (~1700 chips/Hz). The search-fed path has "
        "shipped a phase all along; this closes the model-primary gap. "
        "MEASURED before enabling (scripts/gnss/e2e_phase_transport.py at "
        "31896a862, real "
        "propagate_seed + GPU): arg 0.788 chips, phase+ 0.785, and the "
        "UNCONVERTED phase 51.999 -- the broker references a hop's first "
        "sample and the C++ side its last, so the conversion is what makes "
        "this safe. OFF by default until it has flown.",
    )
    ap.add_argument(
        "--dll-combiners",
        default="",
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
        "the sum SMEAR instead of sharpen (design section 5).",
    )
    ap.add_argument(
        "--dll-spacing",
        type=float,
        default=0.5,
        metavar="CHIPS",
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
        "6.7 on peak, slope 1.10/chip, monotonic to +-2 chips.",
    )
    ap.add_argument(
        "--dll-deep-gate",
        default="",
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
        "oscillation and makes the A/B uninterpretable.",
    )
    ap.add_argument(
        "--dll-deep-gate-from-search",
        type=float,
        default=0.0,
        metavar="SNR",
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
        "threshold is 30 and the pure-noise ceiling ~17.",
    )
    ap.add_argument(
        "--dll-deep-gate-search-hold-s",
        type=float,
        default=90.0,
        help="task #79: how long a search detection keeps a PRN in the "
        "auto-generated deep-gate set (default 90 s). Must exceed the search "
        "REVISIT (90-270 s at CHORD's cadence is the observed range, and a "
        "PRN is not re-detected every cycle), or presence flaps on and off "
        "with the search's luck and the trim is armed and disarmed under a "
        "satellite that never changed. This is a HOLD, not a staleness "
        "budget: it answers 'is this satellite up and detectable', which does "
        "not stop being true between passes.",
    )
    # -- The frame-synced telemetry gather (gnss_broker/telem.py) ------------------------------
    ap.add_argument(
        "--telem-gather",
        default="",
        help="TASK #59: read per-record telemetry from the GATHER instance at "
        "host:port (default port 11061) instead of inferring alignment from "
        "~60 REST polls per cycle. Every frame carries its chain, its instance "
        "and an ABSOLUTE window index on the F-engine sample clock, so "
        "grouping instances is an exact integer match rather than a guess "
        "about arrival order. Empty = off; the REST path is untouched either "
        "way, and ONE reader thread serves every chain in this process.\n"
        "Connecting alone changes NOTHING -- it only fills the store. See "
        "--telem-coherent for the first consumer.",
    )
    ap.add_argument(
        "--telem-coherent",
        action="store_true",
        help="feed fleet_coherent from --telem-gather instead of from "
        "/get_records. THE SAME ESTIMATOR, a different transport: that is "
        "what makes the comparison a measurement. ⚠️ Run it on ONE CHAIN at a "
        "time, with the others left on REST as a same-poll control -- a "
        "before/after across restarts cannot resolve it, because the sky "
        "churns faster (2026-08-13: deep_snr max swung 52-197 in four "
        "minutes). No effect unless --telem-gather is set and the gather is "
        "reachable; if it is not, this silently falls back to the REST poll "
        "rather than publishing nothing.",
    )
    ap.add_argument(
        "--telem-windows",
        type=int,
        default=8,
        help="windows of telemetry fed to fleet_coherent (4 records each at the "
        "default packing, so 8 windows = 32 records ~ 0.34 s)",
    )
    ap.add_argument(
        "--element-poll",
        action="store_true",
        help="TASK #57 step 2: poll the --dll-combiners' /get_elements each cycle "
        "and serve the per-element complex-gain table (amplitude AND phase "
        "per antenna, per instance -- the beam/peel coefficients) via the "
        "publisher's /get_elements. Off by default and gated on its own flag "
        "for the same reason --spectrum-endpoints is: replay is strict-"
        "ordered, and old transcripts do not carry these GETs. Instances "
        "predating the endpoint (nodes not yet restarted) are skipped and "
        "counted, never fatal.",
    )
    ap.add_argument(
        "--element-archive-dir",
        default="",
        help="append the RAW per-element gain parts (u, p2, q per antenna) to "
        "<dir>/elem_<chain>_<YYYYMMDD>.jsonl -- present satellites and the "
        "noise probes only. Raw parts on purpose: the beam map and peel "
        "solves are offline consumers and a combined number cannot be "
        "un-combined. Empty = no archive.",
    )
    ap.add_argument(
        "--telem-dll",
        action="store_true",
        help="TASK #63: close the CODE loop on the discriminator formed in this "
        "broker from the un-summed comb (gnss_broker/combdll.py) instead of on "
        "the powers the trackers formed by summing across each instance's "
        "channels. The arithmetic is the combiner's own -- coherent across an "
        "instance's channels, then powers added across instances -- moved to "
        "where the whole band is visible; what it removes is the tracker-side "
        "sum that destroys the frequency axis on the way.\n"
        "MEASURED BEFORE SHIPPING (scripts/gnss/comb_dll_ab.py at 31896a862, "
        "gps_l5, 20 "
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
        "this chain -- so a gather that dies costs a log line, not the loop.",
    )
    ap.add_argument(
        "--comb-taps-cpp",
        type=int,
        default=0,
        choices=(0, 2),
        help="TAKE THE COMB DLL's per-PRN lobe / per-channel taps from the gather's "
        "C++ reduction (<fleet-trim-url>/get_taps) instead of rebuilding them "
        "in Python from the gathered frames. "
        "0 = off; 2 = ARMED (use the C++ arm).\n"
        "\n"
        "WHY. combdll.lobe_taps walks every (window, instance, record, "
        "PRN, channel) of the stream -- ~140k channel-tuples per chain per "
        "cycle, ~700k across the fleet, each allocating Python complex "
        "objects, ~18%% of chain CPU by live profile. This process is pinned "
        "at 100%% of ONE core by the GIL and its CYCLE TIME IS the sum of the "
        "five chains' Python CPU (measured 12.09 s cycle against 12.77 s of "
        "CPU per cycle-set), so that 18%% is ~2 s of every cycle. The frames "
        "are already in C++ on the gather host; the reduction is ~6k numbers. "
        "The equivalence is gated offline on identical bytes "
        "(scripts/gnss/fleetdll_gate.py: e/p/l/n_chan to 1e-9, n_rec and hop "
        "exact, per PRN and per channel).",
    )
    ap.add_argument(
        "--fleet-trim-url",
        default="",
        help="TASK #51 F3: base URL of the C++ fleet loop's stage, e.g. "
        "http://cf06:12051/fleet_trim. Each cycle the broker POSTs "
        "<url>/set_policy with the armed PRN set, the loop constants and the "
        "tracker endpoints; the C++ side then runs the discriminator, the "
        "integrator and the actuation at ~23.8 Hz instead of this process's "
        "~3.1. POLICY STAYS HERE -- presence, the floors, the deep gate and "
        "the arming verdict are all computed above and only their RESULT "
        "crosses.",
    )
    ap.add_argument(
        "--fleet-trim-bandwidth",
        type=float,
        default=1.4,
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
        "+-1-chip/20 s clock breathing.",
    )
    ap.add_argument(
        "--fleet-trim-readback",
        type=int,
        default=0,
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
        "fleet-trim-url; a new recording then carries it natively.",
    )
    ap.add_argument(
        "--n2-combiners",
        default="",
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
        "deep_snr over those same four partitions is flat to 2%%.",
    )
    ap.add_argument(
        "--time0-endpoint",
        default=None,
        help="CHORD: REST path (relative to --rest-url, e.g. telescope/time0_ns) "
        "serving the F-engine's GPS-disciplined absolute time of frame 0: the "
        "capture anchor dead reckoning needs. Frame 0 is "
        "disciplined to GPS via IRIG-B/PPS, so the anchor is a fact rather than "
        "an estimate.",
    )
    ap.add_argument(
        "--dr-clock-chips",
        type=float,
        default=None,
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
        "prime costs one cycle rather than ~20.",
    )
    ap.add_argument(
        "--dr-doppler-ignores-integrity",
        action="store_true",
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
        "manage +0.36..+0.77 (be7d46e8e:docs/CHORD_JOINT_TRACKING.md 3d). "
        "NOTE this addresses the DEMOTION-linked part; a ~0.29 Hz GPS-wide floor "
        "remains and is NOT explained by trust flips.",
    )
    ap.add_argument(
        "--seed-doppler",
        default="auto",
        choices=("auto", "det"),
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
        "since cp0 was fit at exactly that Doppler.",
    )
    ap.add_argument(
        "--long-code-segments",
        type=int,
        default=75,
        help="number of primary periods in the overlaid/long code the TRACKERS "
        "despread (L2C CL = 75 x 10230; GPS L5 Q5 with NH20 baked in = 20 x "
        "10230). The time-assist below picks which one, so this must match the "
        "tracker's `signal`, not the search's.",
    )
    ap.add_argument(
        "--long-code-epoch-s",
        type=float,
        default=1.5,
        help="the long code's GPS-time-locked repeat period, seconds (L2C CL 1.5; "
        "L5 NH20 0.02). The assist needs unix-time mod EPOCH == GPS-time mod "
        "EPOCH, which holds when GPS-UTC (whole seconds) and the GPS epoch "
        "offset (315964800) are both multiples of it -- true for 1.5 and for "
        "0.02. Absolute-time accuracy needed is ~EPOCH/2.",
    )
    ap.add_argument(
        "--bias-min-snr",
        type=float,
        default=0.0,
        help="detections below this SNR do not enter the clock-freq bias median. "
        "The bias is common-mode, so its uncertainty is (per-sat Doppler "
        "error)/sqrt(N) -- one noise satellite is costly when N is small. "
        "Ungated on CHORD the raw estimate scatters 10.5 Hz; the acquire's "
        "own error predicts 0.8 Hz at N=2. 0 (default) keeps every point.",
    )
    ap.add_argument(
        "--fit-min-snr",
        type=float,
        default=0.0,
        help="detections below this SNR do not enter the cp-rate fit history. The "
        "fit resolves a ~0.0148 chips/s residual; a near-threshold detection's "
        "phase is noise, so one bad point destroys the slope rather than "
        "degrading it. 0 (default) keeps every point -- right for the "
        "prototype, whose detections sit well above threshold and whose "
        "revisit is seconds. CHORD wants ~60 alongside --fit-gap-s 900.",
    )
    ap.add_argument(
        "--fit-hist-len",
        type=int,
        default=8,
        help="detection snapshots kept for the cp/dop slope fits. The default 8 "
        "predates CHORD's search cadence: at 8 detections/s, 8 points span "
        "~1 s, so fit_maturity_span_s (30 s) is UNSATISFIABLE and "
        "fit_trusted -- which gates BOTH hold admission and the escape "
        "referee -- is structurally false on every strong satellite "
        "(measured 2026-08-28: fit_trusted=False with len_h=8, snr 946; "
        "only weak, sparsely-detected sats could ever mature, which is "
        "why G18 got held while G20/21 never did). 256 at 8 Hz spans "
        "~32 s: the maturity floor passes, and the slope error improves "
        "as sigma/(T*sqrt(N)) on both fits.",
    )
    ap.add_argument(
        "--hold-on-present",
        type=int,
        default=0,
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
        "fleet gate reads absent. 0 (default) disables.",
    )
    ap.add_argument(
        "--nh-period-debounce",
        type=int,
        default=0,
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
        "0 (default) disables.",
    )
    ap.add_argument(
        "--nh-joint",
        type=str,
        default="off",
        choices=["off", "apply"],
        help="the JOINT overlay fit: the NH label is not N independent per-sat "
        "20-way guesses -- the overlay is synchronous at the transmitters, "
        "so at the receiver the labels differ only by the known geometric "
        "delay plus ONE common unknown, the receiver clock mod the overlay "
        "epoch. Every detection with a phase at its own epoch VOTES for that "
        "common offset (continuous, chips mod the epoch -- an integer vote "
        "splits at segment boundaries); the consensus is a circular weighted "
        "median. 'apply' DERIVES the overlay segment from the "
        "consensus for detections below --period-check-snr (whose measured "
        "label is noise: G1's ran uniform over 0..19, stepping the seed a "
        "whole code period every ~4 min, 2026-08-31); strong detections keep "
        "their verified measured label, with a disagreement alarm as the "
        "referee. Requires the "
        "almanac (tau and the broadcast sat clock feed the prediction). "
        "Default off.",
    )
    ap.add_argument(
        "--cp-rate-model-primary",
        type=int,
        default=0,
        help="#103: seed the pooled-clock MODEL code rate for EVERY cp-rate fit "
        "(the fit's rate becomes monitor-only; its position is kept). The "
        "measured case: gps_l5's fitted rates carry ~0.006-0.03 chips/s "
        "per-sat errors that saturate the fleet trim's 1.25-chip ceiling "
        "in minutes and drive the escape/rebirth churn (5,773 births/"
        "night), while model rates run ~100x cleaner post-#99. 0 = off "
        "(the fitted rate seeds whenever the tol guard accepts it).",
    )
    ap.add_argument(
        "--cp-rate-model-tol",
        type=float,
        default=0.0,
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
        "0 (default) disables the cross-check.",
    )
    ap.add_argument(
        "--nh-overlay-len",
        type=int,
        default=1800,
        help="secondary-overlay length in chips (B1C pilot 1800; E5a/B2a CS100 = 100)",
    )
    ap.add_argument(
        "--dead-reckon",
        action="store_true",
        help="seed CODE PHASE from broadcast ephemeris (BRDC) for every visible "
        "sat the search hasn't detected: predict the absolute transmit-time "
        "code phase, add the receiver clock solved each cycle from the "
        "detected sats (measured-vs-predicted circular median, "
        "~100 ns), and express it in "
        "the seed's Doppler currency. The search demotes to bootstrap "
        "(clock solve), fallback (a detection re-anchors via the normal "
        "seed loop) and integrity check (per-sat residuals logged); "
        "dead-reckoning only has to land within the DLL capture range "
        "(~0.4 chips; validated 0.10 chip rms 2026-07-13). Needs --almanac.",
    )
    ap.add_argument(
        "--dr-constellation",
        default="G",
        choices=("G", "E", "C", "R"),
        help="RINEX constellation letter for this broker's band",
    )
    ap.add_argument(
        "--signal-capability",
        default=None,
        help="restrict ALL seeds + hints to GPS PRNs whose satellite block actually "
        "broadcasts this signal (GPS_L1C_P -> Block III; GPS_L5_Q -> IIF+; "
        "GPS_L2C_CM -> IIR-M+). The GENERAL form of --dr-min-prn: that numeric "
        "cutoff only works for BDS-2's contiguous low PRNs; GPS III sats are "
        "interspersed among IIF/IIR (4/11/14/18/20/21/23/28) so no cutoff can "
        "express 'L1C only'. Read ONCE at startup from the live Celestrak block "
        "names (gps_beamtrack.signal_capable_prns); on fetch failure or empty "
        "result the filter is DISABLED with a warning (phantoms return, but the "
        "chain lives -- better than killing L1C during a network outage). "
        "GLONASS: the IGS registry's block types decide (GLO_L3OC_* / L2OC -> the "
        "K generations and the modernised GLO-M+; the FDMA L1OF/L2OF are on every "
        "satellite, so they get no filter), re-read every 6 h. The Celestrak "
        "fallback sees only the 'K' suffix on the Uragan number, so with the "
        "registry unreadable it also drops the M+ satellites.",
    )
    ap.add_argument(
        "--dr-min-prn",
        type=int,
        default=None,
        help="dead-reckon only PRNs >= this: a SIGNAL-CAPABILITY gate, not a "
        "visibility one. Default 19 for BeiDou (B1C/B2a are BDS-3 ONLY; the "
        "BDS-2 birds broadcast B1I at 1561 MHz, which is not even inside our "
        "band), 1 otherwise -- and only the FALLBACK: when the IGS registry "
        "is readable, --signal-capability decides by block type and the floor "
        "is lowered to 1, because IGS re-used C11-C14 for BDS-3 MEOs (#150). "
        "The search cannot make this mistake "
        "-- it never detects a satellite that isn't transmitting -- but "
        "DEAD RECKONING CAN: it seeds from the model, and the model is happy "
        "to predict a code phase for a signal that does not exist. The "
        "tracker then despreads noise at that phase and the cross-correlation "
        "against real B1C satellites reports 20-60 sigma. Measured 2026-07-14: "
        "C11/C12/C13 produced 11309 phantom rows (5.5%% of all BeiDou map "
        "points) at a plausible-looking 25-30 dB-Hz. A model that can invent "
        "a satellite needs a capability gate the search never needed.",
    )
    ap.add_argument(
        "--dr-clock-adopt-max-chips",
        type=float,
        default=0.0,
        help="#104: refuse a SIBLING clock adoption whose step from the local "
        "solve exceeds this many chips while the local clock is fresh "
        "(< 300 s). The 2026-08-30 outage: gps_l5's churn ran its clock "
        "solve to 292 chips and the unbounded adoption path relayed it "
        "to every chain in ~2 s -- while JOINT-CLK's 5-chip bound "
        "refused the identical values throughout. 5.0 matches that "
        "precedent. 0 = unbounded (pre-#104). #142: it bounds the "
        "CROSS-BAND bootstrap too, as a TIME -- this many of the DONOR's "
        "chips (5 = 489 ns at 10.23 Mcps), converted into each consumer's "
        "chips (2.5 at E6, 0.25 L2C-CM). 'Fresh' is 300 s since the last "
        "adoption OR the last JOINT-CLK adoption, and a refusal holds the "
        "local clock at zero rate instead of extrapolating it.",
    )
    ap.add_argument(
        "--nav-bits",
        type=int,
        default=1,
        help="decode the broadcast navigation message from the combiner's "
        "nav_obs export (bit_export: true) and cross-check the decoded "
        "ephemeris against BRDC. Self-gating: chains whose combiner exports "
        "no nav_obs are untouched. 0 = off.",
    )
    ap.add_argument(
        "--nav-decoder",
        default="lnav",
        choices=("lnav", "cnav", "bcnav3"),
        help="which decoder consumes nav_obs. lnav (default): the LNAV "
        "decode-and-predict (GPS L1CA, the periodic-subframe future-bit source). "
        "bcnav3: the BeiDou B2b B-CNAV3 decoder (GF(64) NB-LDPC(162,81) + CRC-24Q, "
        "the B2b PRIMARY chain's own nav; pure decode + BRDC xcheck, no bit-pred). "
        "cnav: the CNAV decoder (GPS L2C-CM / L5-I) -- FEC+CRC-24Q, NOT a "
        "future-bit source (the type schedule is not fixed); it decodes live "
        "ephemeris/messages and shadow-serves the signs of DECODED spans. Set "
        "cnav on the L2C broker so its nav_obs (CNAV symbols) go to the right "
        "decoder instead of churning the LNAV frame-sync forever.",
    )
    ap.add_argument(
        "--cnav-combiner",
        default=None,
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
        "can be verified three ways (L2C vs L5 vs BRDC).",
    )
    ap.add_argument(
        "--inav-combiner",
        default=None,
        help="AUXILIARY combiner polled for Galileo E1B I/NAV nav_obs (S5 "
        "D-component #1), the exact analogue of --cnav-combiner: the GAL "
        "broker's own --combiner is the E1C PILOT (deterministic overlay "
        "signs), while the I/NAV DATA symbols come off the derived E1B "
        "sibling chain. Symbols from here go to the InavPredictor and the "
        "decoded ephemeris is cross-checked against BRDC on the 60 s health "
        "cadence -- an independent E1B decode validating the Galileo "
        "ephemeris (and the codec's ICD conventions) against the almanac.",
    )
    ap.add_argument(
        "--fnav-combiner",
        default=None,
        help="AUXILIARY combiner polled for Galileo E5a-I F/NAV nav_obs (S5 "
        "D-component #2), the --inav-combiner analogue on the L5 band: the "
        "GAL/E5a broker's own --combiner is the E5a-Q PILOT (deterministic "
        "CS100 overlay signs), while the F/NAV DATA symbols come off the "
        "derived E5a-I sibling chain (CS20 secondary + navwipe). Symbols go "
        "to the FnavPredictor and the decoded ephemeris is cross-checked "
        "against BRDC on the 60 s health cadence -- an independent E5a decode "
        "validating the Galileo ephemeris (and galileo_fnav's ICD "
        "conventions) against the almanac, beside E1B's I/NAV decode.",
    )
    ap.add_argument(
        "--bcnav2-combiner",
        default=None,
        help="AUXILIARY combiner polled for BeiDou B2a B-CNAV2 nav_obs (S5 "
        "D-component #3, the FIRST non-binary FEC), the --fnav-combiner "
        "analogue on the BDS broker: the BDS broker's own --combiner is the "
        "B2a-P PILOT (deterministic Weil overlay signs), while the B-CNAV2 "
        "DATA symbols come off the derived B2a-D sibling chain (CS5 secondary "
        "+ navwipe). Symbols go to the Bcnav2Predictor (GF(64) NB-LDPC codec) "
        "and the decoded ephemeris is cross-checked against BRDC on the 60 s "
        "health cadence -- an independent BDS-3 decode validating the "
        "ephemeris (and the LDPC + CNAV-eph conventions) against the almanac.",
    )
    ap.add_argument(
        "--bcnav1-combiner",
        default=None,
        help="AUXILIARY combiner polled for BeiDou B1C B-CNAV1 nav_obs (S5 "
        "D-component #4, the LAST), the --bcnav2-combiner analogue on the L1 "
        "BDS broker: the broker's own --combiner is the B1C-P PILOT, while the "
        "B-CNAV1 DATA symbols come off the derived B1C-D sibling. Symbols go to "
        "the Bcnav1Predictor (reusing the GF(64) NB-LDPC codec, different H "
        "matrices for SF2/SF3) and the decoded ephemeris is cross-checked "
        "against BRDC -- completing the civil D-component set (GPS+GAL+BDS).",
    )
    ap.add_argument(
        "--cnav2-combiner",
        default=None,
        help="AUXILIARY combiner polled for GPS L1C-D CNAV-2 nav_obs, the "
        "--bcnav1-combiner analogue on the L1C broker: the broker's own --combiner "
        "is the L1C-P PILOT (deterministic L1CO overlay signs), while the CNAV-2 "
        "DATA symbols come off the derived L1C-D sibling. Symbols go to the "
        "Cnav2Predictor (binary-LDPC systematic extract + CRC-24Q, 18 s frame) and "
        "the decoded ephemeris is cross-checked against BRDC -- the 4th GPS-family "
        "decode (LNAV / CNAV / CNAV-2).",
    )
    # -- Named signals (gnss_broker/signals.py) -------------------------------------------
    ap.add_argument(
        "--signal",
        default=None,
        metavar="KEY",
        help="name the chain instead of retyping its twelve constants: "
        "`--signal gps_l5` sets carrier/chip rate/code length/long-code "
        "segments+epoch/overlay length/constellation from "
        "lib/stages/gnss/gnssSignal.hpp, which is what the trackers' "
        "replica bank is actually built from. Every one of those flags "
        "fails SILENTLY when mistyped (a wrong overlay length puts the "
        "seed in a random one of N periods and errors nowhere), which is "
        "why naming beats typing. An explicit flag that DISAGREES with "
        "the named signal is a hard error, never a silent override. "
        "`--signal help` lists them.",
    )
    # -- Record and replay for the broker_equiv.py gate (_Transcript, gnss_broker/transport.py) --
    ap.add_argument(
        "--transcript-write",
        default=None,
        metavar="FILE",
        help="record every clock read, GET and POST to FILE (JSONL) while running "
        "normally. Cheap and side-effect-free: safe to leave on in production "
        "for the minute it takes to capture a golden run.",
    )
    ap.add_argument(
        "--transcript-read",
        default=None,
        metavar="FILE",
        help="replay a recorded transcript instead of talking to the fleet, then "
        "exit when it is exhausted. No network, no sleeps, deterministic "
        "clock. The POST stream this produces is the equivalence gate -- see "
        "scripts/gnss/broker_equiv.py.",
    )
    return ap
