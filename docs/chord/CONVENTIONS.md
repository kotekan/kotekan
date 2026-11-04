# CHORD GNSS stack: conventions

Units, axes, time and geometry conventions of the CHORD GNSS stack. Consolidated 2026-11-04 from the Claude memory files named at the end of each section. Dates are when a fact was measured or a rule set; commit hashes are on kv/chord-gnss.

## 1. Frequency and channels

**freq_MHz = freq_id × 0.1953125, counting up from 0; derive df = fs/fft_length, never from the channel count.** fs 3200 MHz, nyquist_zone 1, fft_length 16384 (not 8192): df 0.1953125 MHz, 8192 channels, freq0 = 0. Code `CHORDTelescope::to_freq_MHz` (`lib/utils/CHORDTelescope.cpp:578`). L5 1176.45 MHz = fid 6023.42; band fid 5972..6076 (2026-08-16).

**One instance holds one `freq_id mod 8` residue split over two GPUs: channels at stride 16 = 3.125 MHz, decimated across the band.** All 8 nodes' `gpu_offsets_mod16` cover {0..15}. A same-instance delay fit aliases every 320.0 ns; cross-instance pairs (1/2/4 channels apart) resolve it to 1280 ns, and the shipping combine never forms one. Telemetry carries `chan_ids` per instance.

Sources: chord-freq-id-to-hz.md

## 2. Time axes

**Quanta.** 52.3776 chips per hop (chip_rate/hops_per_sec at 10.23 Mcps). Record = 2048 hops = 10.48576 ms = 10.4857 L5 code periods. Frame = 4 records; `propagate_seed` runs per 41.9 ms window. `pow_hop` advances in 40960-hop quanta = 209.7 ms. Telemetry window = 32 windows ≈ 1.34 s. One F-engine seq step = 5120 ns.

**Two hop axes, one label: search `ref_hop` is the F-engine hop counter; DR writers stamp it from wall time, `int(round(t_now_abs × hops_per_sec))` (2026-08-17).** Differencing across axes reads the wall-vs-F-engine offset × chip rate mod the code period: integer seconds vanish, milliseconds read as thousands of chips, and single-axis round trips never notice (INNOV p95 2598 vs MINNOV 1.27 on G26; dr-owned seeds excluded, ec049e6e2). Never difference `ref_hop`s across writers; stamp DR `ref_hop` from the telemetry hop counter.

**The F-engine axis is drift-free but ~100-200 ms stale: right for labels, wrong for an epoch (2026-08-17).** `utc0_sample0` = node `time0_ns`, the GPS-disciplined time of `fpga_seq_num` 0; `pow_hop` is the newest processed record, so `(utc0 + pow_hop/hps) − wall` = median −99.6 ms, IQR 59 ms. Evaluating orbits (`predict_all`) on it was 65× worse (cp0 moved 0.6-2.4 chips/PRN); reverted. Shipped `--clock-step-guard-s` 0.5, log only (2db3eec1f). Fix-class: serve the current hop beside `pow_hop`.

**frame0 is UTC (UNIX), not GPS time (verified 2026-10-29).** `wall − age − (frame0_unix_ns + seq × 5120 ns)` = +0.100 s on the 10-29 rail file (p5-p95 0.083-0.108); GPS time would show 18 s. frame0 = 1793207549000003890 = 2026-10-28 17:12:29 UTC (F-engine re-base), attr `frame0_unix_ns` in recv1 files. `seq = (unix_ns − frame0_nano)/5120`.

**A hop's phase is referenced at its LAST sample (`window_start + fft_len − 1`) in C++ and at its FIRST in the broker: 52.3713 chips apart (52.3776 × (1 − 1/fft_len), Doppler-scaled).** The C++ reference (`ChannelizedReplicaBank::window_advance_chips`, inherited by `phase_from_arg`, `propagate_seed`, `cp_at_ref`) is the contract; the broker's `dr_cp0`/`dr_seed_phys` convert. 2026-08-12: moving `cp_at_ref` broke e2e (0.373 → −52.002 chips). Guard: `agg_up.sh` asserts `cp_at_ref − reconstruction = +52.38 ± 1`.

Sources: chord-two-hop-axes.md, chord-fengine-axis-is-stale.md, chord-hop-epoch-convention.md, chord-record-geometry.md

## 3. Code phase, chips and clocks

**`code_phase_chips` is an ARGUMENT, not a phase: `C(n) = comb_mult × code_phase_chips + n × cps(doppler)`, n the absolute sample index.** At CHORD uptime (n ~1.9e15) the lever is 5094.9 chips per Hz of Doppler. Never transport an argument: lift it to a phase at its own epoch with its own Doppler, propagate in phase, convert back with the generator's Doppler (error then 0.0087 chips/Hz/s); `phase_from_arg`/`arg_from_phase` enforce it. `cudaGnssChordTrack` broke it twice (fixed 9e12d515b, 2026-07-31).

**Chips are a UNIT, the clock is a TIME: rate-convert before a chip quantity crosses chains (2026-08-31).** The bootstrap did `donor.value % ctx.code_len`; gal_e6 (5.115 Mcps) took gps_l5's 10.23 Mcps clock as chips = 2× the time = +75 E6 chips on every seed, while bds_b3i (10.23 Mcps) locked in ~60 s: diff rates and units first. Fix 8db6b1f98: `contribute_dr_clock` carries `chip_rate_hz`, `Receiver.clock_chips_convert`, donor window in time.

**Published `doppler_hz` is the broker's last commanded seed; `doppler_applied_hz` (`REC_DOPPLER` = `PrnCtl.fcar_report`) is what ran.** `publish.py` overrides the combiner field, so the viewer's Doppler is a staircase. Seed continuity beats freshness: E4, 10 s repin median deep_snr 16.7 vs held 221.2, because each repin steps the dr-clock EMA's ±1 chip into the code phase (606648df9 reverted).

**A prime is not a measurement: the first multi-sat solve REPLACES a primed clock (#28, 01e777480).** `--dr-clock-chips 0.0` silenced the bootstrap snap on gps_l5, which EMA'd in from 0 over ~40 s (0.8ⁿ, to 151 chips); the first solve flagged every sat +114..+125 BAD and MODEL-UNTRUSTED. On sky: BOOTSTRAP 149.67 chips 0.8 s after start, settled 149.4-151.8 unchanged. No persistence file.

Sources: chord-cp-currency.md, chord-chips-are-a-unit.md, chord-published-doppler-is-the-command.md, chord-dr-clock-prime-is-not-a-measurement.md

## 4. Geometry and signs

**F-engine output is CONJUGATED relative to the nominal 4+4b decode (high nibble real, low imag, offset 8).** Sky 2026-07-30: PRN 32 acquire 10.1 as-is vs 22.5 conjugated, common +5.8 Hz GPSDO bias. `GnssChordDequantize conjugate` true on every deq leg; tracker `DespreadParams::conj_data` (8d293ba36); all four feeds agree. Invisible to the |·|² X-engine and to self-consistent synthetic tests.

**The generator already advances by `n × cps`; never add a per-hop advance. NH20 is baked into the tracker code.** `GPS_L5_Q_NH` tiles the primary 20× with NH signs (204600 chips / 20 ms) for trackers; the search keeps the short code + `nh_search` (d37064e87). The `chips_per_hop × dh` added alongside was a double count (every record but the first despread to 0), reverted 9e12d515b. GPU `n0 = window_start_sample + fft_len − 1`, same as the bank.

**The boresight is the current dish pointing, not a constant: since 2026-10-24 it is Tau A, az 180°, el 62.70° (pointing `p5_taua_20261024`; `guowang_passes.BORESIGHT`, the broker's `--boresight telescope`). From 08-03 to 10-24 it was az 180°, el 81.41°, dec +40.73° (8.59° south of zenith), and `telescope.dish_coelev_deg` (−27.3) was wrong then.** 6 m dish at L5: FWHM 2.48°, first null 2.97°, first sidelobe ~3.97° at −17.6 dB. PRN 19 passes 0.40° off boresight at 08:04 UTC (2026-08-03, ~4 min earlier daily). A sat within ~5° rails the shared 4+4b quantiser on every chain (survivors +2.7 dB, population 7 → 4): `--detector-transit-veto-deg 5.0` (e04df6da1); vetoes must be cross-chain.

**Element steering = per-(sat, channel, element) phasors in the assembler combine; the SIGN is config, measured, never derived.** Unsteered gain u carries the comb factor `C(τ) = Σ_ch exp(i2π f_ch τ)`, τ = r·ê/c: archived per-element phases are interference patterns and displaced elements lose amplitude to |C|. Layout `config/chord_dish_layout.json`: rows letters (A southernmost, 8.5 m), columns numbers (01 westernmost, 6.3 m), x EW, y NS; mapping interleave 1,5,2,6,3,7,4,8, pairs (2k,2k+1) = X,Y (sE = sN = +1). Armed 2026-08-29, e5a steered / e5b control.

**Map the beam from probe-debiased `p2 = <|A_e|²>`, never `|u|/q`.** `get_elements_callback`: `u = <A_e·conj(LOO ref)>`, `p2 = <|A_e|²>`, `q = <|LOO ref|²>`; `|u|/q` is a ratio to the array mean, flat by construction (fake 6 dB; real 14-16 dB horizon→zenith). Debias `p2 − floor` per (instance, element, 5 min), never medianed across elements. Data `fixtures/obs/elem_<chain>_<date>.jsonl`; recipe `docs/CHORD_BEAM_MAPS.md`.

Sources: chord-fengine-conjugation.md, chord-record-geometry.md, chord-boresight-and-transits.md, chord-elem-steering-and-geodesy.md, chord-beam-map-from-elem-archive.md

## 5. Record and cube rules

**Beam-cube bins are not a common unit: `incoh = Σ|G_e/E_c|²` carries 1/E_c² per channel; normalise per (bin, elem) against its own pedestal first.** BPSK(10) centre bins sit ~20 dB below edge bins at equal sky power, so a raw sum is the edge bins (2026-09-06: a 21 dB "peak" at the horizon). Builder: `(P − F)/F`, F = per-(file, 5 min, bin, elem) median; gate: peak 4-8° off axis at +15-21 dB (`fixtures/cube_look/check_export.py`). `|coh|²/(w·incoh)` = 1/w for noise, 0.006 near boresight: `coh` winds until records are rotated by the tracked carrier.

**Under the lobe combine a record's prompt power is ONE complex dof (Exp): no per-record ratio or threshold gate is usable (2026-09-08).** A med+3MAD q bar passed 9% of probe records and dropped half of a 31 dB-Hz sat's. Shipped 9ac2bdf50: presence = `sig_inc = mean(rho)/(sd/√n)` over all records ≥ 5.0 (N(0,1) null, ~14 dB-Hz floor at 384 records); lock = the window's summed P is the tallest tap (on-peak q 4.4 on L5 at 0.5 chip, 6.7 on L2C at 2.0: q constants are per-chain); probe clip 20× median.

**Carry a phase as a value at a reference plus a short increment, never (frequency) × (absolute time).** n_abs ~1.9e15 makes the ulp of a 1.18 GHz carrier (2.4e-7 Hz) a large fraction of a cycle; shorten the lever. The error re-rolls per re-propagation and every cross-record estimator integrates it. Applied: `phase_ref_chips` at `ref_hop` (#45); `DespreadJob::ang0` at `par.n0` (#52, 86349ac4d), reduced on the host in long double. Uptime scaling is necessary, not sufficient: the `C_P = cp0 + n_m × cps` twin changed 3.58e-02 → 3.48e-02; isolate the term first.

**Coherent span (revised 2026-08-10): on a strong sat the residual after the best linear ramp is 0.51 rad (coherence 0.88-0.96); the ramp is the fold-killer, not decoherence.** The standing Doppler residual is tens of Hz stepping ~10 Hz per 5 s; `deep_rate_search` finds it to 0.05 Hz. The 2.2-2.55 rad / 29 dB / ~0.25 s story is the weak-sat regime (amp_snr 3-5), where `carrier_hz_resid` is signal-free and cp slope × 115.03 Hz per chip/s amplifies noise 115×.

**No combination rule: the seed is a bare dict with ~14 writers per cycle, half REPLACE, half CORRECT, arbitrated by ~20 flags and cycle order (audit 2026-08-16, `docs/CHORD_CONTROL_AUDIT.md`).** REPLACE: Doppler source selection (B:3963-3998), the cp-rate fit overwriting phase, rate and epoch (B:4148-4152), hold-freeze (B:4483), the code-rate overwrite after the DLL (B:7998-8010); CORRECT: the DR slew (B:6364-6441), held phase as base, model as correction. The fast trim is applied once additively (`gnssSeedTransport.cpp:329`) but nothing in the broker reads `/get_dll` or `/get_trim` (#76, #77).

**`deep_rate_hz` is the fold's pick clamped to `deep_rate_max_hz` 5.0: past the cap it reports the best in-cap NOISE bin.** f44ba0b2c serves `deep_rate_full_hz/q` (uncapped ±47.7 Hz); the rrate feed prefers it and holds when absent. The measured residual lags the command 1-2 polls: `--rrate-cmd-slew-hz 0.5`. Decohered strong sats (PRN 27: amp 56, coh_frac 0.02) swing ±10 Hz per poll: feed gated on `coherence_s > 0 or coh_frac ≥ 0.3`; why is OPEN.

**OPEN: the per-record carrier phase jumps at every frame boundary (2026-08-15).** The lag-m autocorrelation `r_m = <A_{k+m} conj(A_k)>` is rate-blind and must be non-increasing for a stationary process; gps_l5 at m = 1, 2, 4, 8, 16, 32, 64 (normalised at m = 4, 4 records/frame) gave 0.682, 0.423, 1.000, 0.963, 0.835, 0.575, 0.322: |r_1| < |r_4| is a monotonicity violation. Intra-frame steps ~0.42 rad (PRN 26) vs +1.856 at the boundary; the intra-frame ramp is the bogus 2-13 Hz "residual rate" of cross-frame searches. `ang0 = 2π frac(f·n0/fs)` at absolute n0 (2.7e-7 Hz rotates it one radian, re-rolled per record) explains the fake rates, not the period-4 dip. Fix shape (not built): accumulate `phi += omega × dn`, or anchor ang0 per track; gate on |r_1|/|r_4| → 1.0.

Sources: chord-cube-bins-are-not-a-unit.md, chord-one-dof-record-gate.md, gnss-phase-from-a-reference-not-an-argument.md, chord-coherent-span-limit.md, chord-no-combination-rule.md, chord-deep-rate-alias.md, chord-frame-boundary-phase-jump.md
