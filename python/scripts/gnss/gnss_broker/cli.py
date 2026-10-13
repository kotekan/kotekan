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
    # --bias-alpha
    #   EMA weight for the clock-freq bias (smaller = steadier seed Doppler; ~0.05 =>
    #   few-second time constant, dithers out the 500 Hz grid)
    bias_alpha=0.05,
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
    # --cn0-prompt-min-sig
    #   served C/N0 (combdll.prompt_cn0): a PRN is served only when the mean debiased prompt
    #   power over EVERY gathered record is this many standard errors above zero, the scatter
    #   measured from the same records (sig_inc, ungated). In the lobe currency a record's
    #   prompt power is one complex dof, so a below-horizon probe's statistic is N(0,1) and 5
    #   admits nothing that is not there; the weakest satellite it serves at 384 records is
    #   ~14 dB-Hz, below anything the loop holds. Measured on gps_l5: probes and untracked
    #   PRNs 0.0-1.1, real satellites 20-200.
    cn0_prompt_min_sig=5.0,
    # --clock-bias-alarm-hz
    #   CLOCK DRIFT ALARM bar: loud log if the live bias EMA departs the warm-start calibration
    #   by more than this (GPSDO unlock / thermal event -- hardware news, not something to
    #   silently absorb)
    clock_bias_alarm_hz=10.0,
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
    # --dr-clock-alpha
    #   EMA weight for the solved receiver clock (wrap-aware chips; the held value propagates
    #   at f_chip*(l-a) between solves)
    dr_clock_alpha=0.2,
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
    # --dr-solve-refused-rebootstrap-s
    #   after the clock solve has been REFUSED continuously for this long, force a one-shot
    #   re-bootstrap (clear the held clock and let the next median snap). 0 disables. THE
    #   REFUSAL GUARD IS A LATCH WITHOUT THIS: MEASURED 2026-08-10, one non-L5 PRN poisoned the
    #   median at 19:39, the clock froze at 19:47 and never re-solved, because the scatter that
    #   triggers the refusal is itself sustained BY the frozen clock (the model drifts off the
    #   sky, every sat reads as noise, MAD stays at 2045). Before the guard existed this same
    #   failure random-walked out in ~15-20 min (be7d46e8e:docs/gnss_gpu_search.md 11.33); the
    #   guard stopped the corruption and removed the escape with it. A forced re-roll is
    #   explicitly a RE-ROLL, not a measurement -- it restores the random walk rather than
    #   pretending to solve.
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
    #   against its own best this session. 600 s is ~30-45 broker cycles -- long
    #   enough that a transit or one bad poll cannot trip it, short enough to catch a
    #   degradation in minutes rather than the 3.5 h #87 ran for.
    q_stall_window=600.0,
    # --rf-stats-poll-s
    #   Seconds between RF-health polls (default 30). The node refreshes every ~10 s, so a
    #   faster poll re-reads the same pass; the point of the bound is that the broker's cycle
    #   is on cf06's single 1 GbE, which telemetry has already inflated once (#64).
    rf_stats_poll_s=30.0,
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
    # --carrier-rate-min-q
    #   hard gate on deep_rate_q before a rate residual is believed. The 8.18.5 gap:
    #   17.9-22.0 on signal, 2.8-6.1 on noise. A weak sat does not merely scatter, it
    #   lands on the WRONG spectral bin (measured: amp_snr 9.5 was 41.7 Hz out by
    #   split-half, where amp_snr 83.7 was 0.000).
    carrier_rate_min_q=10.0,
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
    #   this many seconds while the rest of the fleet does (#70). Costs no
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
    #   refusing a wrap alias or a diverged filter. In the JOINT STATE'S chips (its feeders',
    #   10.23 Mcps) on every chain, so 5 = 489 ns everywhere; a chain at another chip rate
    #   converts before comparing (deadreckon.dr_joint_clk).
    joint_clk_max_chips=5.0,
    # --joint-clk-max-sigma
    #   P2b consumer 'clk': refuse a joint clock with 1-sigma above this (chips). P
    #   grows while the state is unfed, so this one gate covers estimator health AND
    #   staleness; healthy runs measure 0.05-0.08. Joint-state chips, as above.
    joint_clk_max_sigma=0.5,
    # --joint-clk-max-step-chips / --joint-clk-step-hold-s
    #   P2b consumer 'clk', #176: a joint clock more than this many chips (joint-state
    #   chips) from the one this chain last ADOPTED is refused until it has held the new
    #   level, to the same tolerance, for this many seconds. The chain cycle is 2 s; a
    #   transit-blinded feeder set walked the joint clock 0.8 chips for ~40 s and the
    #   bounds passed it onto every chain. The receiver clock moves under 0.003 chips/s
    #   (99th percentile), so a slower walk never meets the guard and a real step is a
    #   discontinuity, delayed by the hold. Compared on the joint clock, not the delta: the
    #   legacy median's churn steps the delta ~90 times a day, the joint ~20
    #   (deadreckon.dr_joint_clk).
    joint_clk_max_step_chips=0.3,
    joint_clk_step_hold_s=30.0,
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
    #   measurement sigma (chips) for the joint solve. The search cp noise is 0.03-0.5
    #   chips per-sat-conditions; 0.3 is the middle of that and NOT tuned against the
    #   answer -- the state's own covariance reports whether it was right.
    joint_sigma=0.3,
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
    #   frontend sensitivity loss, be7d46e8e:docs/gnss_gpu_search.md 11.33). Default 600 s is ~half
    #   the 1276 s per-PRN revisit, so a healthy fleet always has fresh samples and only a
    #   genuinely starved one widens.
    nh_hint_max_age_s=600.0,
    # --period-check-snr
    #   detections below this SNR do not enter the period-continuity history and their
    #   disagreements are logged as 'weak det' rather than as source regressions.
    #   Measured on CHORD 2026-08-02: above ~60 the within-period phase is
    #   self-consistent to a few chips across a 400 s gap, below it the residuals are
    #   ~2000 chips, i.e. noise.
    period_check_snr=60.0,
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
    # --coast-budget
    #   seconds a VISIBLE sat is coasted (seed held + Doppler forecast forward) through a signal
    #   dropout before dropping it -- so a radar sweep / brief fade doesn't lose the lock. The
    #   code prediction stays good for ~tens of s on a free-running TCXO; raise it with a
    #   disciplined clock (OCXO). |A| recovering resets the coast; setting below the horizon
    #   drops immediately.
    coast_budget=30.0,
    # --hold-snr
    #   incoherent amp_snr above which a tracked PRN's cp anchor is FROZEN (hold-on-lock: DLL
    #   owns the sub-chip residual; fit re-anchors only on loss). Uses amp_snr ONLY --
    #   deep_snr's off-peak value is the nav-wipe rectification floor (~7), which would freeze
    #   bad anchors instantly.
    hold_snr=8.0,
    # --hold-max-cp-err
    #   release a HELD seed when the tracked code phase (held cp + DLL trim) disagrees with the
    #   search FIT by more than this (chips) on 3 consecutive fixes. This is the DLL's capture
    #   half-range: a sharp-ACF (BOC) power discriminator has stable FALSE equilibria ~0.75
    #   chips out (prompt -12 dB) that the hold would otherwise servo forever while the search
    #   sees the true peak.
    hold_max_cp_err=0.4,
    # --bias-det-fresh-s
    #   clock-bias solve uses only detections FRESHER than this (seconds). A stale detection's
    #   (meas - pred) grows at the satellite's Doppler rate -- stale-age x dop_rate reads as a
    #   fake, GROWING clock bias that the seeds then chase (measured: a ~90 s-stale detection
    #   walked the bias +4 -> +68 Hz and dragged a 55-sigma tracker off the sky).
    bias_det_fresh_s=30.0,
    # --bias-min-sats
    #   detected sats needed before a cycle's median residual may update the clock-freq bias
    #   (and hence NARROW the search). A single sat's residual is unfalsifiable: one bad
    #   prediction (wrong TLE mapping, non-transmitting sat cross-corr) gets swallowed as 'clock
    #   bias' and shifts every other hint out of the narrow window -- a self-locking deadlock
    #   where no second sat can ever acquire to correct it (2026-07-12: BDS-2 C14 froze the
    #   whole B1C constellation at -1550 Hz).
    bias_min_sats=2,
    # --doppler-sign
    #   multiply predicted Doppler (set -1 if the convention is inverted)
    doppler_sign=1.0,
    # --tle
    #   GPS TLE file/URL (default: Celestrak gps-ops)
    tle=None,
    # --bias-stale-s
    #   STALE-BIAS RESCUE: if the solved bias EMA has gone this long without a multi-sat
    #   measurement, widen the search margins and RE-SOLVE from the next detections (snap to the
    #   fresh median, recalibrate, persist). Closes the 2026-07-20 lockout: a GPSDO unlock
    #   walked the EMA -2 ppm, every lock died, and the EMA latched mid-walk -- hints then sat
    #   kHz off truth at narrow margins with nothing left to update them. The held value still
    #   centers the (wide) hints and seeds (hints only under --seed-bias-source=zero), so a
    #   healthy chain that merely has a sparse sky loses nothing. 0 disables.
    bias_stale_s=300.0,
    # --clock-bias-alarm-every-s
    #   rate limit for CLOCK DRIFT ALARM. Was 60 s, which against a poisoned calibration fired
    #   once a minute for hours and buried the log -- an alarm repeating faster than it can be
    #   investigated is noise, and it would have hidden a real one.
    clock_bias_alarm_every_s=600.0,
    # --clock-bias-cal-min-sats
    #   satellites a STALE-RESCUE re-solve needs before it may replace the warm-start
    #   calibration. Below this the old calibration is HELD and said so. The re-solve used to be
    #   adopted unconditionally: one median from however few sats were present became the
    #   reference and was announced as hardware news -- measured 2026-08-27, a 2-sat re-solve
    #   moved it -2.3 -> -17.9 Hz against a population whose median is +0.0 (sd 12.7 over 1222
    #   samples), and every later comparison was then against noise. An old reference is stale;
    #   a noise-derived one is WRONG, and wrong outranks stale for something everything else is
    #   measured against. 0 disables the guard (the old behaviour).
    clock_bias_cal_min_sats=5,
    # --clock-step-guard-s
    #   Log loudly when the offset between cf06's wall clock and the F-engine's GPS-disciplined
    #   axis JUMPS by more than this (s). LOG ONLY -- never a control input, and deliberately
    #   not a correction. Motivation: the orbit model is evaluated at wall time (the one
    #   consumer needing absolute UTC to be true, not merely self-consistent), and nothing
    #   bounds what that clock may do; a step moves every model seed on every chain at once,
    #   fleet-common and otherwise unattributable. ⚠️ RESOLUTION LIMIT, measured: the observable
    #   F-engine 'now' is the newest telemetry pow_hop, which trails the sky by the gather/serve
    #   latency -- -99.6 ms median with a 59 ms IQR over broker_onsky_e5a's 520 cycles. So this
    #   sees STEPS well above that jitter (the seconds-scale ones NTP actually makes) and is
    #   blind below it. That same lag is why the epoch was NOT moved onto this axis: it would
    #   trade 1.45 ms of NTP error for ~100 ms of pipeline lag (measured 0.6-2.4 chips of seed
    #   motion).
    clock_step_guard_s=0.5,
    # --code-doppler-sign
    #   must match the search stage's code_doppler_sign: the sign of the Doppler-dependent cp0
    #   back-reference the search applies. Used to re-express cp0 history in the seed's Doppler
    #   currency before the slope fit (cp_to_seed_currency).
    code_doppler_sign=1.0,
    # --coh-min-records
    #   fleet coherent: common records (hops) required per PRN.
    coh_min_records=32,
    # --dcb-max-age-days
    #   WARN when the DCB product in use is older than this (days, from the epoch in its
    #   filename, not its mtime). The CAS rapid product lands ~5 days late, so a healthy steady
    #   state is 5-6 days and anything past ~10 means the fetch has stopped working. THIS IS THE
    #   ONLY TELL: fetch_dcb checks the local cache before the network, so an expired token
    #   keeps serving the newest file on disk -- correct, then quietly staler -- and returns
    #   None only after --dcb-bias's 14-day walk runs out of cached days. 0 = never warn about
    #   age.
    dcb_max_age_days=10.0,
    # --detector-transit-veto-deg
    #   suppress D3 while ANY visible satellite (all constellations) is within this many
    #   degrees of boresight. 0 = off. A near-boresight satellite rails the 4+4b quantiser for
    #   every chain at once -- measured 2026-08-26: inside 3 deg the tracked population drops
    #   from 7 per epoch to 4 and the survivors read 2-3 dB HIGH -- so satellites vanish across
    #   all five chains together and a per-satellite detector reads that as a constellation of
    #   individual faults. ON by default because it is a SUPPRESSION (the conservative
    #   direction) and the detector is read-only; it costs recall on real events during a
    #   transit and buys a base rate that is not mostly geometry.
    detector_transit_veto_deg=5.0,
    # --dll-leak
    #   DLL integrator leak (0 = pure integrator): trim mean-reverts each update so
    #   discriminator NOISE can't random-walk it to the clamp. DC loop gain = dll_gain/dll_leak;
    #   ~1/leak windows of smoothing.
    dll_leak=0.05,
    # --dll-leak-present
    #   DLL integrator leak on the FLEET path, where `present` has already confirmed signal
    #   (--dll-combiners). The ordinary --dll-leak caps the reachable correction at
    #   (gain/leak)*0.25 = 1.25 chips, which is below the residuals actually seen and leaves the
    #   loop pushing at a railed discriminator without arriving. 0.01 lifts that ceiling to 6.25
    #   chips -- past the 3.27-chip tracker grating-lobe spacing -- while still mean-reverting
    #   over ~100 updates. The leak was a stand-in for a signal test; the fleet path now has a
    #   real one, so it does not need to be this strong. The single-combiner fallback keeps
    #   --dll-leak.
    #   MEASURED 2026-08-04 AND REVERTED: 0.01 made locks strictly WORSE -- best-q median 2.05
    #   -> 1.03 and samples above q 2.0 went 4/8 -> 0/8, every satellite driven to noise. The
    #   leak is not only a ceiling: when the discriminator is RAILED (|disc| ~ 0.9, which is
    #   most of the time here) tau saturates at 0.25 chips and carries only the SIGN, not the
    #   distance -- so the integrator pushes one way indefinitely and the leak is what bounds
    #   the excursion. Default restored to 0.05; the knob is kept so the experiment is
    #   repeatable, not because it should be moved.
    dll_leak_present=0.05,
    # --dll-min-instances
    #   fleet DLL: instances that must report the same window before their sum is used. Below
    #   this the PRN falls back to the single --combiner discriminator, so a partially-down
    #   fleet degrades instead of stalling.
    dll_min_instances=2,
    # --dr-clock-freeze-s
    #   HOLD the dead-reckon receiver clock solve and its drift EMA while the newest telemetry
    #   hop has not advanced for this many seconds. This is the CONTROL that
    #   --fe-axis-stale-s's log line always implied and never had (2026-09-03 19:37): the
    #   F-engine stopped streaming, the detections froze, the orbit model kept moving with the
    #   wall clock, and the clock solve read the sky's motion as clock drift -- -0.3543 chips/s,
    #   accepted in 0.05-EMA steps each inside the 1.0 bound, then held for 17 h because a clock
    #   walking 10 chips between solves seeds every tracker off-peak and nothing ever
    #   re-measured it. Same-band adoption spread it to gal_e5a and bds_b2a in 10 s. While
    #   frozen, raw_prev is dropped so no pair of solves is ever differenced across the gap. The
    #   axis normally advances every ~2 s cycle; 6 s is three missed advances, and a held state
    #   costs nothing on frozen data.
    dr_clock_freeze_s=6.0,
    # --dr-clock-repin-solves
    #   #142: consecutive solves of NEW detections that must agree -- within
    #   --dr-clock-step-max-chips of the first -- before the receiver clock takes a step larger
    #   than that bound, as a snap (a confirmed clock counts only solves of
    #   --dr-update-min-sats, an unconfirmed one any of --dr-min-sats). Real solves agree cycle
    #   to cycle to p99 1.5 chips while noise medians land uniformly over the code, so 3 leaves
    #   ~(10/10230)^2 for a chance agreement and costs a genuine re-pin 4 s. Values below 1
    #   count as 1.
    dr_clock_repin_solves=3,
    # --dr-clock-step-max-chips
    #   #142: the largest step (this chain's chips, solve vs the propagated clock) an update of
    #   an established receiver clock may take. A larger one is refused and the clock HELD at
    #   zero rate unless --dr-clock-repin-solves consecutive agreeing solves confirm it -- then
    #   it is a RE-PIN, taken whole (how a bad bootstrap is corrected). A CONFIRMED
    #   clock is also held, never re-rolled, through MAD refusals. Normal solves step p99 1.5
    #   chips; the solves that stepped the clock in the replayed transit passes were
    #   2-5-satellite medians 6-5000 chips off it. 5.0 is the #104 / JOINT-CLK bound (489 ns at
    #   10.23 Mcps).
    dr_clock_step_max_chips=5.0,
    # --dr-clock-transit-freeze-deg
    #   #142: FREEZE this chain's solved receiver clock while any satellite the model carries
    #   (every constellation, the D3 veto's geometry) is within this many degrees of
    #   boresight: no solve is applied, no drift is differenced, the MAD re-bootstrap clock
    #   stops, and the clock is held at zero rate and contributed every cycle. Inside a transit
    #   the search reports noise code phases just over its bar and the median of them is not the
    #   clock; the clock itself is a GPS-disciplined constant. Only a clock confirmed by an
    #   in-bound update freezes (never a prime or an unconfirmed bootstrap), and chains without
    #   detections never do.
    dr_clock_transit_freeze_deg=5.0,
    # --dr-clock-wait-s
    #   withhold dead-reckon seeding while the receiver clock is still the --dr-clock-chips
    #   PRIME rather than a measurement (0 = never wait). A seed born without a clock steps by
    #   the WHOLE clock at its first re-birth after BOOTSTRAP -- measured 2026-08-22: 40 events,
    #   all inside 57 s of broker start, step/off = 0.9958, the entire fleet moving together to
    #   0.08%. The wait is short (gps_l5 bootstraps ~1 s in, the model-primary chains adopt
    #   cross-band a few seconds later). ⚠️ IT IS A TIMEOUT, NOT A PRECONDITION: a DETECTOR-LESS
    #   chain never solves a clock at all -- that is what the prime is for -- so after this many
    #   seconds we seed on the prime and log that we did. A guard that can leave a chain dark
    #   forever is worse than the step it removes.
    dr_clock_wait_s=30.0,
    # --dr-drift-max-age-s
    #   A dead-reckon clock DRIFT that has not been re-measured (or re-adopted) for this many
    #   seconds EXPIRES back to the f_chip*(l-a) model (0 disables). The drift EMA is refreshed
    #   only by consecutive raw solves, and a WRONG drift is exactly what prevents them, so a
    #   long-unrefreshed value is not steady, it is unmeasurable -- and the longer it stands the
    #   likelier that is the reason. Breaks the self-sustaining loop regardless of how the
    #   poison got in.
    dr_drift_max_age_s=600.0,
    # --dr-max-solve-mad-chips
    #   refuse the receiver-clock solve when the per-satellite offsets scatter by more than this
    #   (MAD, chips). A circular median over --dr-min-sats satellites is a MEASUREMENT only if
    #   its inputs agree; on a starved fleet the detections are noise, scatter ~uniformly over
    #   the code, and their median is an arbitrary number that used to be snapped straight into
    #   the clock and shipped as every seed. Not a tuning knob -- real offsets cluster inside
    #   ~+-10 chips against a uniform-noise MAD of ~2557 on a 10230-chip code. Refusing leaves
    #   the clock UNSET, which is strictly better than wrong: a wrong clock also acquires
    #   nothing, and additionally poisons the search hints that would have recovered it (docs
    #   11.33).
    dr_max_solve_mad_chips=100.0,
    # --dr-min-sats
    #   detections needed for a receiver-clock solve (one sat is unfalsifiable -- same reasoning
    #   as --bias-min-sats)
    dr_min_sats=2,
    # --dr-update-min-sats
    #   #142: satellites a receiver-clock solve needs to MOVE an established clock (the first
    #   solve after a start, a prime or a forced re-bootstrap still needs only --dr-min-sats, so
    #   a cold chain is never left unseeded). Below it the clock is HELD at zero rate and still
    #   contributed. A 2-3 satellite median follows its members' biases (+-5 chips) and is where
    #   two corrupted detections that agree become a clock: outside transits n=2 / n=3 solves
    #   step p90 4300 / 0.6 chips against p99 1.3 at n=4, and they are 4.5% of cycles.
    dr_update_min_sats=4,
    # --estimator-every-s
    #   minimum seconds between the TELEMETRY-WALK estimators (cn0_prompt + kcoh) per chain; the
    #   last values keep being served between runs. They are pure-Python walks over ~1500 record
    #   decodes each, and at every-cycle cadence across five chains they ate ~75% of the
    #   interpreter -- the telemetry READER then missed the gather's 200 ms delivery deadline
    #   and 5 of 6 frames were dropped (measured 2026-08-15 17:17, gaps 128k vs frames 39k). The
    #   estimators' inputs are ~1.3 s windows; nothing downstream needs them faster than the
    #   display refreshes.
    estimator_every_s=15.0,
    # --fe-axis-stale-s
    #   Log loudly when the newest telemetry hop -- the thing t_now_abs is built from -- has not
    #   advanced for this many seconds. 0 disables. LOG ONLY. Motivation (2026-08-18):
    #   cx19/gnss0 deadlocked its DPDK capture window, its pow_hop froze, and because the broker
    #   takes the time base from that ONE combiner without a staleness test, t_now_abs froze
    #   too. det_age went negative and grew at 1 s/s, the clock solve saw every residual as
    #   enormous and latched at a frozen value, and all four clock-ADOPTING chains lost tracking
    #   together (#75) -- with nothing in the log naming the cause. Judged as a tracking failure
    #   it is unfixable; named as a stalled instance it is a node restart.
    fe_axis_stale_s=30.0,
    # --joint-feed-warmup-s
    #   withhold ALL measurements from the joint state for this many seconds after broker start.
    #   Birth is the one path with no innovation to gate on, and the establishment window is
    #   when the y-feed is at its worst: the 2026-08-12 00:08 zombie was born from
    #   establishment-phase garbage (biases +-20-46 chips), locked it in, and no escape could
    #   fire because the garbage was self-consistent. Seeds/holds settle in ~3 min; 240 is that
    #   plus margin. The filter object itself is still created (warm-started) immediately --
    #   only the feed waits.
    joint_feed_warmup_s=240.0,
    # --joint-mask-after
    #   P2c: a satellite may be withheld only once it has this many ACCEPTED updates.
    #   Withholding from startup is a vacuous test -- the sat is never born into the state, so
    #   there is no b_sat to freeze and nothing to coast on. It must be established FIRST and
    #   withheld after, which is the whole point: we are testing whether the state REMEMBERS a
    #   satellite, not whether it can invent one. ⚠️ THE TWO SEEDING DISCIPLINES FEED AT VERY
    #   DIFFERENT RATES, which is why this defaults low: a model-primary chain offers every
    #   seeded sat EVERY cycle, so the count accrues in seconds, while a search-anchored GPS sat
    #   only enters when a detection clears --joint-min-snr and can take hours. Calibrated on
    #   the wrong path a threshold either fires instantly or never -- 200 fired in 6 s on
    #   Galileo and had not fired on GPS in 14 min. 50 is reachable on both and ample to
    #   establish a bias (sigma_b0 10 against a 0.3-chip measurement converges in tens).
    joint_mask_after=50,
    # --joint-max-rate-ppm
    #   a JOINT rate consumer REFUSES a |clk_rate| beyond this (ppm) and falls back to the l-a
    #   EMA. CHORD's reference is GPS-disciplined and measures 4e-5 ppm, so 0.01 is ~250x
    #   headroom and still refuses the -0.028 ppm runaway that froze the trackers.
    joint_max_rate_ppm=0.01,
    # --joint-min-coh-frac
    #   minimum coherent fraction for a residual to enter the joint solve. Nearly redundant with
    #   --joint-min-deep-snr (both gates select the same 70.5% of rows, which is the two
    #   measuring the same physics) and kept as a cheap independent guard, not as a second
    #   opinion.
    joint_min_coh_frac=0.3,
    # --joint-min-deep-snr
    #   minimum TRACKER significance (deep_snr, floor-cleared) for a residual to enter the joint
    #   solve. --joint-min-snr gates on the SEARCH SNR carried in `best`, which is a different
    #   and much staler number: detections latch for up to the 1276 s per-PRN revisit, so a
    #   satellite that has left the beam keeps presenting the strong SNR of its last good scan
    #   while its tracker output is already noise. That is exactly how PRN 25 got in on
    #   2026-08-10 -- elevation RISING 45->65 deg (it transited out of the beam, it did not
    #   set), deep_snr 30->2.4, coh_frac 0.94->0.19, code phase uniform over the 10230-chip code
    #   -- and destroyed the state. MEASURED, not guessed: over 6827 GPS rows the population is
    #   sharply bimodal, noise at deep_snr 2-3 / coh_frac 0.2 and signal at 24+ / 0.92+; 10 sits
    #   in the empty gap and admits 70.5%.
    joint_min_deep_snr=10.0,
    # --joint-min-snr
    #   minimum detection SNR for a residual to enter the JOINT solve. The default matches
    #   --period-check-snr, whose docstring records the measurement: below it a detection's
    #   phase is noise with ~2000-chip residuals. Ungated, ONE such detection walked the state
    #   to -0.028 ppm on 2026-08-09 (be7d46e8e:docs/gnss_gpu_search.md 11.23).
    joint_min_snr=60.0,
    # --joint-slew-max-chips
    #   P2b consumer 3: refuse the joint slew target if it disagrees with the legacy (clock EMA
    #   + SatBiasFilter) target by more than this. Both describe the same receiver, so a few
    #   chips is the whole plausible range -- b_sat runs +-3-6 chips and the clock is common to
    #   both. A larger gap means one estimator is broken, and the legacy path is the one with
    #   years behind it. Same discipline as --joint-max-rate-ppm on consumer 1, which would have
    #   refused the -0.028 ppm runaway that froze the trackers no matter what went wrong
    #   upstream. In the joint state's chips (5 = 489 ns at 10.23 Mcps), the same time on every
    #   chain, like --joint-clk-max-chips.
    joint_slew_max_chips=5.0,
    # --lock-q
    #   treat a satellite as LOCKED for the dead-reckon slew/re-birth decision when its fleet
    #   discriminator q is at least this (0 = ignore q, the pre-2026-08-22 behaviour). q is the
    #   metric this project judges lock on everywhere else -- the viewer's own note says 'judge
    #   lock HERE, not on sig/C-N0' with 2.2 as the working bar -- but the seeder's lock test
    #   used amp_snr/deep_snr and a prompt-vs-fleet-median ratio instead. Measured 2026-08-22:
    #   both of those read below their bars on tracking satellites (sig_of 0.00, hold_prev
    #   0.76-1.61 against 3.0), so 24% of re-births hit sats at q >= 2.2 and five stepped the
    #   seed 1.3-4.5 chips -- outside the +-1 chip correlation triangle, where the DLL has no
    #   gradient left to recover with. That is the bimodal q collapse.
    lock_q=2.2,
    # --nh-joint-min-prns
    #   satellites that must agree (within --nh-joint-tol-chips of the circular median) before
    #   the common offset resolves. Below this the joint fit stays in observation and 'apply'
    #   changes nothing.
    nh_joint_min_prns=3,
    # --nh-joint-min-snr
    #   detections below this snr do not vote (measured: snr < 60 phases carry ~2000-chip
    #   within-period residuals -- letting them vote dragged the on-sky consensus inlier set
    #   from 7/7 to 3/9 within 10 min on 2026-08-31); they may still be APPLIED to, which is the
    #   point of the joint fit.
    nh_joint_min_snr=60.0,
    # --nh-joint-tol-chips
    #   a vote further than this from the circular median is an outlier and does not count
    #   toward resolution. True agreement is a few chips; the default is deliberately loose
    #   (0.15 code periods) so the tell is the LOGGED spread, not a silent exclusion.
    nh_joint_tol_chips=1500.0,
    # --nh-joint-window-s
    #   votes older than this are pruned. A resolved consensus is NOT expired by pruning -- the
    #   receiver clock mod the epoch is a run constant (the F-engine axis is drift-free); it is
    #   only ever REPLACED by a newer resolution.
    nh_joint_window_s=600.0,
    # --presence-disp-deep-margin
    #   admit-displaced: the OFFSET-BLIND evidence bar, as a multiple of the combiner's
    #   deep_floor. ⚠️ THE POINT OF THIS FLAG IS THAT THE EVIDENCE MUST NOT BE AN ON-PEAK
    #   STATISTIC. The admission originally required prompt power above the probe floor -- but
    #   prompt power is suppressed by exactly the offset being rescued, so it is q's disease one
    #   step along, and it was the THIRD time the same kind of statistic was swapped for another
    #   of its kind. Measured on sky 2026-08-27: 7 satellites detected at deep_snr 18-56 (C/N0
    #   up to 26.8 dB-Hz) with prompt at 0.5-2.7x noise; the prompt bar admitted 2 of them.
    #   deep_snr RE-SEARCHES code phase, so it sees the satellite wherever the tap sits, and
    #   deep_floor is the combiner's own rectification level -- not a population of peers.
    #   Prompt power stays as an ALTERNATIVE (either suffices); the pedestal and offset tests,
    #   not this one, are what still refuse a centred noise realisation.
    presence_disp_deep_margin=3.0,
    # --presence-disp-off-max
    #   admit-displaced: max |fitted offset| in chips. Inside the DLL's pull-in range (the early
    #   tap leaves the correlation triangle at 1 - spacing/2 = 0.75 chips); past it the
    #   discriminator carries only its sign and admission would arm a loop with no gradient to
    #   follow.
    presence_disp_off_max=0.6,
    # --presence-disp-pedestal-max
    #   admit-displaced: max fitted noise pedestal (relative to the peak). Measured 2026-08-26:
    #   displaced-but-strong rows sit at 0.03-0.08; genuinely weak rows and probes read >= ~1
    #   (E=P=L fits pedestal inf).
    presence_disp_pedestal_max=0.3,
    # --prn-reconfig-axis-max-age-s
    #   refuse to schedule a swap when this cycle's F-engine axis sample is older than this. The
    #   deadline is built by advancing the polled hop to the instant of the POST, and an
    #   extrapolation over a long gap is a guess dressed as a coordination point: past this age
    #   the swap posts UNSCHEDULED and says so, rather than naming a frame the fleet may already
    #   have passed.
    prn_reconfig_axis_max_age_s=30.0,
    # --prn-reconfig-heartbeat-s
    #   say ARMED AND HEALTHY this often, even with nothing to propose. ⚠️ NOT COSMETIC: every
    #   other branch of the stage speaks only when it has a swap in mind, and having nothing to
    #   do is the NORMAL state -- E36, the satellite this whole mechanism exists for, is below
    #   the admit mask ~13 h a day. Without this, armed-and-idle is indistinguishable from
    #   not-running.
    prn_reconfig_heartbeat_s=900.0,
    # --prn-reconfig-lead-s
    #   schedule a PRN swap this far ahead, as an ABSOLUTE F-engine HOP (set_prns {"at_hop":
    #   N}), so every node crosses the discontinuity on the SAME FRAME. Without it each node
    #   swaps on whatever frame it happens to be building, and the combiner folds one window
    #   whose instances disagree about which satellite slot p is -- an accumulator-identity
    #   error that is invisible downstream because every row is individually well-formed. 5 s is
    #   many frames (41.94 ms each) of margin against POST scatter and well inside the
    #   --prn-reconfig-interval-s cadence. 5 s, not the original 2: the map goes to 12 endpoints
    #   in sequence and one unresponsive node costs a whole --prn-reconfig-timeout-s, so the
    #   lead must cover the SCATTER of the posts, not just their mean. The in-cycle delay
    #   between the status poll and this post is handled separately, by advancing the axis
    #   sample to post time -- do not spend the lead on it. The lead is capped by the node: a
    #   deadline it can never reach (an F-engine re-base moves the hop BACKWARDS) is applied
    #   immediately with a warning rather than wedging the slot forever. ⚠️ HOPS, NOT SAMPLES --
    #   the first version posted hops under a key the node read as samples (hop x 16384), so
    #   every deadline was already past and every swap silently took the apply-immediately
    #   degrade.
    prn_reconfig_lead_s=5.0,
    # --prn-reconfig-poll-s
    #   how often to complete a full sweep of the nodes' live maps. Read back rather than
    #   remembered, so what this diffs against is what the nodes ACTUALLY hold -- including
    #   after a restart reverted them to the config list. ⚠️ THE SWEEP IS ONE ENDPOINT PER
    #   CYCLE, not a burst: a sweep of all 12 costs (n_dead x timeout) on a SINGLE cycle, which
    #   is a full minute of broker stall the first time site work takes the fleet down -- #81's
    #   failure exactly.
    prn_reconfig_poll_s=60.0,
    # --prn-reconfig-timeout-s
    #   per-endpoint HTTP timeout for the map GET/POST. With the one-per-cycle sweep above this
    #   is the WHOLE cost a dead node can impose on any one cycle.
    prn_reconfig_timeout_s=2.0,
    # --rrate-cmd-slew-hz
    #   per-POLL bound on how far the posted carrier command may move (Hz). THE STABILITY TERM,
    #   not a nicety: the measured residual reflects the command posted 1-2 polls earlier, while
    #   the feed adds back the LATEST command -- so during a transient the reconstruction is off
    #   by (cmd_now - cmd_then), which is self-reinforcing and walked arm 3 at full slew.
    #   Bounding the step bounds that misreference at ~slew*lag and it DECAYS: at the fixed
    #   point the command stops moving and the reference is exact. 0.5/poll = full 5 Hz pull-in
    #   in ~20 s. Railed-step count is in the JRR-CMD line -- a rail that never clears means the
    #   target is out of reach, not converging (the dr slew cap's lesson).
    rrate_cmd_slew_hz=0.5,
    # --telem-dll-windows
    #   windows meaned for --telem-dll (0 = --telem-windows). 32 windows x 4 records = 128,
    #   which is what the combiners' EMA integrates; a much shorter window makes the comb arm
    #   noisier than the arm it replaces and the A/B would read that as a regression.
    telem_dll_windows=0,
)
# ── end frozen tuning ────────────────────────────────────────────────────────────────────

# The names --signal sets after the parse (gps_distributed_broker.main). It would overwrite a
# frozen value silently, so none of these may ever be frozen.
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
    """The broker's argument parser. `description` is the BROKER module's docstring."""
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
    # ---- #91: THE BROWNOUT POLICY (default OFF; D1 supplies the trigger) -----------------
    # A band-wide carrier-coherence collapse takes PRESENCE down chain-wide (E3, 2026-08-25:
    # gal_e5a's KCOH sig_sum 43,255 -> 97 -> 14, every sat at once, ~9 min, while gal_e5b
    # held ~100k). Two things then make a 9-minute fade into a full per-sat re-pull:
    #   (b) every standing trim is RELEASED, and an unarmed trim leaks to erasure in ~5.6 s;
    #   (c) the (l-a) clock, re-fitted from the COLLAPSING population, swings (+-96 chips
    #       measured) and its garbage rate is adopted into every seed -- positive feedback
    #       that sustains the outage.
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
    # ---- LIVE PRN MEMBERSHIP (docs/CHORD_LIVE_PRN_RECONFIG.md, gnss_broker/prnmap.py) ----
    # The node's slot->PRN list was hand-written in the manifest and this broker reads live
    # BRDC; nothing reconciled them. Measured 2026-08-26: 5 dead Galileo slots while E36, which
    # transits at 83 deg, had no slot at all -- and, because the noise-probe selector picks the
    # deepest below-horizon PRN, it kept picking the one satellite the node could not represent,
    # dropping both gal chains to brightness-only presence.
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
    # -- TASK #59: the frame-synced telemetry gather ------------------------------------------
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
        choices=("G", "E", "C"),
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
        "GPS and GLONASS: for R the block marker is the Celestrak 'K' suffix on "
        "the Uragan number, and GLO_L3OC_* -> K satellites only (the CDMA "
        "signals are GLONASS-K's; the FDMA L1OF/L2OF are on every satellite, so "
        "they get no filter).",
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
    # -- named signals (task #27 M2); see gnss_broker/signals.py --------------------------
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
    # -- the refactor gate (task #27 M0); see the _Transcript note at the top of this file --
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
