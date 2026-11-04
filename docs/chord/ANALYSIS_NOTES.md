# CHORD GNSS stack: analysis notes

Estimator and loop-health knowledge for the CHORD GNSS stack: what each served statistic measures, its null, and the traps already paid for. Consolidated 2026-11-04 from the Claude memory files named at the end of each section. Commit hashes are on kv/chord-gnss.

## 1. C/N0 and significance

**`deep_snr` is a detection significance whose normalisation depends on how many instances were combined; never derive C/N0 from it.** The viewer's `cn0_coh = 20log10(deep_snr) − 10log10(T_coh)` assumes one coherent integration, but `publish.py` serves two estimators into the field: `quad:N` = `sqrt(Σ snr_i²)` reads +8.72 dB over the best instance, `fleet:N` +0.79; sqrt(12) = 10.79 dB, so a flip moves served C/N0 −7.93 dB with the sky fixed. Publish a separate per-instance `cn0_coh_db`; never rescale `deep_snr` (`--joint-min-deep-snr 10.0` depends on it).

**Served `cn0_coh` is blind to code error: `deep_snr` is a coherent sum of ~100 prompt records (10× gain), not a lag re-search.** `rate_search` (`GnssCoherentCombiner.cpp:1002`) scans a phase rate only, so a prompt at 1-2× noise reports deep 20-35. 2026-08-12 15:20-15:45: per-instance deep 1.5-2.5, s4_raw 1.00-1.05, cn0_inc 22-25 dB-Hz, served 41 dB-Hz: the "good" window had no signal. `disc ≈ 0` means on-peak OR blind (E ≈ L ≈ noise). Fix a76368bd4: `prompt_lock` = `s4_raw` (absolute) AND `fleet_present` (relative) agree; the 0.90 bar sits between blind 1.00-1.05 and tracking 0.17-0.82.

**`deep_snr` fires on noise: a probe 64° below the horizon reads 11.2× its deep floor. Check the probes before believing any deep number.** gps_l5: PRN 10 (elev −64) deep 29.9, 11.2× floor; real PRN 26 0.6×. Paired A/B of the same estimator on the same records (7912 samples): >2× on 67% of cycles, >10× on 23%, p10-p90 21.1 dB: the per-integration rate re-search is the 20-60 dB-Hz viewer swing. Fix path: fold at the KNOWN rate; do not widen #49's deep gate.

**`cn0_kcoh` is the known-rate fold: the rate is injected causally and `eta = |Abar|²·n/mean|A|²` (n coherent, ~1 noise) says whether coherence accrued.** `Abar = mean_k A_k e^{−2πi f t_k}`, `t_k = hop_k × hop_s`; fleet = mean over instances of |Abar|²; C/N0 = 10log10(rho/T_coh), the same currency as cn0_prompt, so they must agree on a strong sat. Rate = the previous cycle's record-stream fit, 0.0 for DR sats, clamped ±15 Hz. Floor σ²/n from the probes' per-record power; `floor_white` served (excess = RFI). Served `sig` = `kcoh_sig` when eta ≥ 25% of n_rec, else `sig_inc`.

**The carrier is coherent for the full 1.34 s; the injected rate was off by 1-10 Hz. Retracted: "coherence ~0.1 s, fold 13× too long".** `--coherence-scan` folds at the injected rate, so a bad rate reads as decoherence. `kcoh_rate_probe.py` (fixed 1.34 s span, rate sweep, probe null, block stability): gal_e5a PRN 4 eta/n 0.78, gps_l5 PRN 3 0.37, null 2.0-2.2; width 1/T = 0.75 Hz. gps_l5 PRN 3 sits at +12.8 Hz vs reported `carrier_hz_resid` ~0.5.

**The fold's rate is real only once the tap is on-peak. Check `fleet_q` first: q = 2P/(E+L) ≈ 1.0 is the no-peak value, and everything downstream then measures noise.** PRN 4 on-peak: coh_frac 0.034 → 0.704 at L = 128, rate +3.132 Hz, residual 0.827 rad. Fix the code latch (#49) first, then the fold. NH20 rejected by shape: a sign flip is bimodal (0, π), white is uniform; measured 0.45-0.54 against a 0.50 null on 15/16 PRNs (`scripts/gnss/phase_flip.py`).

**`rrate` (range-rate error, truth minus model) is physically sub-cm/s; the ±1-3 m/s estimated cannot be orbit error.** Budget: ephemeris velocity ~mm/s, clock drift 0.3-3 mm/s, iono mm-cm/s; receiver clock is common-mode in f_carrier. Candidate: stale commanded Doppler (`dop_rate × latency`, not established). The observable is accumulated phase differenced on ONE instance (∝ 1/T): 0.82 Hz over 60 s vs 5-8 Hz for any spectral fit. `res_cycles` has a per-instance origin: cross-instance differencing is 6× error, not misalignment.

Sources: chord-cn0-is-a-significance.md, chord-served-cn0-is-blind.md, chord-kcoh-estimator.md, chord-deep-snr-fires-on-noise.md, chord-fold-is-a-rate.md, chord-rrate-physics.md

## 2. Correlation and loop health

**Replicas must be built on the instance's GLOBAL channel ids: `GnssCudaDespread`'s `chan_offset + c` put the CHORD tracker comb at DC.** With `chan_offset = 0` every replica was bins 0-6 while the stride-16 comb sat at 5972..6068, so every tracker correlation since first light was noise. Fixed 51b1ca034 (2026-07-31): `chan_ids` overload, `channel_ids` from the generator (FATAL on size mismatch). Validate with a noiseless synthetic signal at known cp.

**Search grating lobes: a comb with common factor g repeats every `sph/gcd(g, sph)`; check the node count before blaming a loop.** Two nodes = 27 channels at stride 4 → lobes every 4096 samples = 13.09 chips, the true lobe ahead by only ~15% (42% wrong at acquire snr 711). All 8 nodes → 106 channels at stride 1, no lobes (2026-08-02). DLL pull-in is ±0.5 chips.

**Tracker grating lobes: 7 channels at stride 16 repeat every 320 ns = 3.274 chips at 90-98% of the peak; q and disc cannot tell them apart and fleet power-summing preserves them.** Measured q 3.10 at the peak, 3.08 and 2.98 at 3.30 and 6.61 chips off. The seed must land within ±0.5 chips; the search's stride-1 comb picks the lobe. Spacing: stride 16 → 3.27 chips, stride 8 → 6.55, stride 1 → 52.4.

**`channelized_peak`'s coarse code phase is correct to one hop (52.4 chips); `refine_peak` re-scans ±fft_len and absorbs the rest. Never reuse it in a new geometry.** Omitted: the replica phase at index 0 (zero only because the search anchors at `Mp × fft_len` = 51,200,000 samples = 16 periods); `Mp × sph` as whole periods (`2N × 16384` at N = 196 is 2.007 periods = 72.0 chips); `AcquisitionSurface::tau()` with coarse and fine of opposite sign and i mod sph. Unchanged in the ms-split geometry: 2568 chips.

**q = 2P/(E+L) is peak sharpness, not SNR: 1.0 with no peak, → 4.0 clean at 0.5 chip; `trim_quality_min` 2.2 sits above the noise tail. Never lower it.** Sky 2026-08-03: noise (9 PRNs) mean 0.96-1.13, tail 1.87; PRN 23 mean 2.38 (0.66-3.45 at 33.7 dB-Hz). Summing K instances does not raise q (a ratio of means); the spread falls 1/√K (tail ~1.3 at K = 8), so a constant gate fits one fleet size: `fleet_dll()` re-measures median + kσ (MAD) each cycle.

**The DLL leak bounds a railed discriminator: weakening 0.05 → 0.01 drove every satellite to noise (2026-08-04).** Steady state `trim = (gain/leak) × tau`, |tau| ≤ 0.25 → 1.25-chip ceiling at gain 0.25 (PRN 9 parked at +1.00, disc −0.984). At leak 0.01: best-q median 2.05 → 1.03; PRN 9's disc flipped to +0.981 through the peak. A railed |disc| ≈ 0.9 carries only the sign; an integrator fed a sign leaves. `--dll-leak-present` stays 0.05; fix the seed.

**E ≈ P ≈ L per channel is expected with strong signal: replica coherence <E,P> = <P,L> = 0.972, <E,L> = 0.970 at 0.5-chip spacing.** A ~195 kHz bin on a 10.23 Mcps code barely decorrelates over 49 ns; `|P|/rms(E,L)` 0.94-1.05 proves nothing. Delay lives in the phase slope across the comb (why the DLL is fleet-wide). Test signal against another node, same PRN, same time (`amplitude`, `coh_frac`), or combine channels first.

**`--fit-gap-s` must exceed the per-PRN search revisit (1276 s = 21.3 min, 2026-08-03) or `code_phase_rate` is 0 forever.** `cp_hist` resets when `ref_hop − h[-1][0] > fit_gap × hops_per_sec`; the fit needs 3 points; at 900 s (default 16) zero `cp-fit` lines in 12 min / 67 passes. The search's +1.585 Hz Doppler error (0.0138 chips/s) then drifts the code 17.6 chips between updates = 5.4 tracker lobes → q 1.0. The revisit itself is the target.

Sources: chord-sparse-comb-replica.md, chord-grating-lobes.md, chord-tracker-grating-lobes.md, acquire-coarse-cp-is-only-hop-accurate.md, chord-trim-quality-gate.md, chord-dll-leak-bounds-a-railed-disc.md, chord-epl-equal-is-not-noise.md, chord-fit-gap-vs-revisit.md

## 3. GPU cost and scheduling

**Synthesis is DRAM-byte-bound (~66% of peak: 245 MB in 536 µs = 457 GB/s), not coalescing-bound; fewer bytes RESIDENT pays, fewer MOVED does not.** Per PRN per record `7 chan × 2048 hops × 3 trials × 212 chips` = 9.1M chip steps, 3.5-4 ms of a 10.486 ms record at 100 codes. Hop-sorting cut load sectors 62% and ran 17% slower. `wavebench`: fp16 Phi 1.27-1.37× (1.55× at production shape); shared Phi 0.93× (benchmark on the A40 that ships, not the L40S); hop-sorted 0.81×; ceiling 2.13×; tiled streaming 0.60-0.70×. PRN packing is not a lever.

**Centered truncation: the sinc peaks mid-span, so one-sided `set_max_chips` cut the peak (−3 dB at 105); centered-80 is 2.6× on synthesis, ~4× with fp16.** e2e: exact to 40 chips on the 8-node comb (60 on 2-node, cliff at 52); full band has no cliff to 8 chips, floor −0.27 dB at 60, −1.1 at 40, −3.5 at 20. Production: `phi_fp16`, `despread_max_chips: 80`, `despread_chips_centered: true` (armed since the 09-05 restart). Wall model (cx52 A40, 09-08): `wall = ceil(n_prn × n_chan / 84) × T_block`, T_block ~0.15 ms at 80/fp16; 12 × 7 = 84 blocks = one wave, so fewer channels empty SMs.

**PRN swaps take `at_hop`, an absolute F-engine hop, so all nodes cross the discontinuity on one frame (5189f1496).** Applied at the first frame boundary with `seq0 ≥ at_hop`; `get_prns` reports `pending_at_hop`, `last_seq`. A map posted "now" makes one combiner window disagree about which satellite slot p is; wall deadlines fail (`AXIS INST` spread of seconds). Policy dfe511421: elevation hysteresis, no hold timers (the old ones reset on every broker restart). Inert twice: `note_frame_hop` only in `cudaGnssChordTrack` while producers are `cudaGnssInject`; `pow_hop` (hops) vs `sample_seq` (samples, 16384×). Proven 2026-08-27. `gnss{0,1}_inject` is gps_l5, not Galileo.

**The post-merge node deaths were a torn read of ring slot-0 metadata (`set_metadata` fills unlocked while `cudaCopyFromRingbuffer` reads); 9216 = 8 × 384 × 3 was the proof.** RFImask [8, 384, 128] = 393216; extent[2] still held SKtilde's 3; `ensure_frame_desc` FATALs the node. Mitigation af618106c/f6436c5cb: publish only a descriptor whose byte size equals frame_size. Source fix 7c025c5d0: fresh pool object, installed last under the buffer lock; `get_metadata` locked. Residual (#107): 09-02 autopsies show rficounts frames with hours-stale `fpga_seq_num` at 2048/4096-hop multiples (2048 = the GNSS record quantum): a cross-pool metadata pointer stamped in place.

Sources: chord-synthesis-cost-model.md, chord-gpu-optimization-queue.md, chord-prn-swap-scheduling.md, chord-ring-metadata-torn-read.md

## 4. Data supply

**BRDC supply is its own module: `gnss_brdc_supply.py` (801 lines) below `gnss_ephemeris.py` (524), with a PEP 562 read-forwarding facade. Reads forward, writes do not.** Split a15ca4911 (2026-08-27). `from x import *` copies bindings, so reads go through module `__getattr__`; no module `__setattr__` exists, so cross-module writers (`sky.py` `LOG_HOOK`, test patches) target `gnss_brdc_supply` by name. Output gate d24be516b: a frozen station-hourly fixture + `test_brdc_golden.py` (byte-identical, idempotent, order-independent; golden in git; hash decompressed bytes).

**BeiDou's predictable count sawtooths hourly (freshest 25 min → 2 h 25 min); not a fault. Check minutes past the hour first.** BDS-3 updates on the hour; IGS hourly files land ~15-25 min after close. 2026-08-27 20:57: G 55 min, E 75, C 115. `predict_all` validity is 4 h from toe; no daily backstop with BKG down; the merge rebuilt from only the hours it walked (C 28 records, toc age 115/175/295 min), so predictable C went 23 → 8. Fix: a rolling union over the 4 h window (not built that evening). Never widen max_age.

**Signal capability is per-block hardware truth: set `signal-capability` on every chain, registry-first (IGS SINEX), never Celestrak/TLE.** G5/G7/G16 (IIR/IIR-M, no L5) showed "--" and noise blips. `ctx.capable` was never populated (`--signal-capability` unset; its guard read `args.constellation`, None for signal:-configured chains); `require_hint: true` means filtered hints remove ~1/3 of GPS passes. Armed 2026-08-31: gps_l5 excludes 2, 5, 7, 12, 15, 16, 17, 19, 22, 29, 31. Galileo fails open; BDS-2 via `dr-min-prn 19`.

Sources: chord-brdc-supply-split.md, chord-brdc-beidou-sawtooth.md, chord-signal-capability-gate.md
