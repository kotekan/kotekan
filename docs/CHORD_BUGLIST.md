# CHORD GNSS — buglist / open items

Open issues, unknowns to revisit, and watch items for the **CHORD** side (branch `kv/chord-gnss`).
Newest/active at top within each group.

**Scope, and why this file exists.** `airspy_docs/buglist.md` is the PROTOTYPE's list — a
different repo (`lwlab@gx10.vdl:/home/lwlab/airspy_gps/docs`), a different instrument, and its
newest entry is 2026-07-28. It has never tracked CHORD. Before this file the CHORD open items
lived in two prose sections that only a reader of the whole document would find
(`CHORD_GNSS_STATE.md` §8.22.3/§8.22.4 and `gnss_gpu_search.md` §9.13), so "what is open" had no
single answer. This is that answer. The two sections above remain the narrative; this is the
index, and it is the one to keep current.

Last reconciled: **2026-08-06 02:0x UTC**, against session tasks #6–#20.

---

## 🔴 Blocking

### B1. Split-aperture genie anomaly — blocks `sky_deep`
On synthetic data at 1.20 rad injected phase the split rung reads **28.4 against a GENIE of
20.5**, and the genie itself moves with the injected amplitude (41.5 at 0.75 rad) when it should
be invariant. An estimator that beats perfect knowledge is not understood. Suspect the harness
first: the genie's dependence on `sky_sig` is the part that makes no sense, and `sumtrack_test`
part C shares its RNG stream between the phase draw and the noise draws. **Do NOT enable
`sky_deep` until this is closed.** Everything else about the split aperture is verified — at the
live operating point it takes 13.7 → 30.0 against a genie of 41.5, and the no-sky control
confirms the honest price is exactly √2 (40.8 → 28.2). Detail: `CHORD_GNSS_STATE.md` §8.22.3.

### B2. CUDA acquire declines a valid blind Doppler grid — blocks the blind slot
`GnssCudaAcquire::set_doppler_grid` hard-rejects grids that are not bin-aligned, and the blind
grid the search offers is being refused. Consequence: `blind_prns_per_pass` is pinned at 0, so
there is no cold-start reacquisition path and the chain depends entirely on hinted PRNs. The
rejection is deliberate (the fine-lag DFT fold is only exact on an aligned grid) — the question
is whether the blind grid should be snapped or the fold generalised.

---

## 🟠 Correctness / accuracy

### C1. The GPU chip gather carries the UNSNAPPED chip/tap boundary
`hoprate_stream` (CPU) snaps the chip/tap boundary to the exact per-sample expression
(`66ca1583a`, backported to the prototype as `ffdb60e02`); `chip_gather` in
`cudaGnssReplicaDevice.cuh` still uses the float floor, which falls to the wrong side for a whole
congruence class of `d` because `cps` is rational. Floors peel depth at about **−53 dB**. Not a
limiter today. **Do this together with P2** — same function, and touching it twice is how a
bit-exactness gate gets broken.

### C2. `test_gnss_channelized_acquire` — 2 failures, stale tolerance
`recovers_code_phase_and_doppler` and `accumulation_recovers_weak_signal_under_noise` assert
`|peak_tau_samples − true_tau| <= SP` (4) and get **24**. Byte-identical on `f789d6856` and on the
merge, so it is not from the merge: it is the deliberate "channelized_peak's coarse phase is a
SEED for the refine, not a measurement" change. 24 samples = 0.077 chips, well inside the hop
`refine_peak` re-scans, and e2e closes to +0.373 chips at 8 nodes. The TEST is what is stale —
but confirm that reading before relaxing it, and state the new bound in terms of the hop, not a
chip count.

### C3. `WITH_BOOST_TESTS` defaults OFF
Neither tree was running `tests/boost` at all. Turning it on immediately found two real defects
(a missing `glonassL2OCCode.cpp` in both source lists, so the suites did not even link; and the
FDMA carrier-anchor bug in C4's class). Should be ON by default, and in CI.

### C4. Class: a clean auto-merge is not a semantic merge
Recorded because it has now bitten once and will again. The long-double carrier ANCHOR and the
per-PRN FDMA carrier STEP were each correct on their own branch, touched no common line, merged
without conflict, and were wrong together — the offset entered the rate and not the anchor.
Caught only by a test that compared against an untouched reference. **A self-comparison (bench
vs. itself) cannot see this class.**

---

## 🟡 Instrument gaps

### I1. No 4+4b clip-fraction monitor, no band-power log
RFI 2026-08-05 15:49–16:04 UTC: broadband +8.7 dB noise with the SIGNAL falling at the same time
— the signature of the quantizer being driven toward saturation (a gain change moves both
together; a Tsys change does not suppress signal). Identical on both interleaved combs, so not
narrowband. All satellites dropped together and recovered on their own. Wanted: a clip-fraction
monitor on the 4+4b path so this is **measured rather than inferred**, and a low-cadence
per-channel band-power log so the next event is characterised in frequency while it happens.
(Checked: the nodes were on the 14:03 binary throughout, so it was not a rebuild.)

### I2. The null floor is a MAX statistic and spikes
One cycle read 16.6 against a typical 4–6 (characterised max 5.73 over 726 samples). A spike
briefly suppresses marginal detections. Consider a high percentile, or an EMA over cycles.

### I3. `log_level: WARN` hides deployment confirmations
The node configs that matter run at WARN, so INFO lines that confirm a setting took effect
(`despread_max_chips`, kernel split, fleet counts) are invisible exactly where they are needed.
Design flaw, not yet fixed — a confirmation of a deployed setting should not be INFO.

---

## 🟢 Performance / scaling

### P1. Revert `despread_max_chips` to 0
The truncation is obsolete: 212 chips (the full PFB span) at the current kernel is **cheaper than
the 140-chip kernel that the cap was introduced for**. The cap bought 1.31× by throwing away part
of the channel response; the width + fusion buy 7.6× and throw away nothing. Costs 1.42× at
scale, which at 100 codes is still comfortable. Edit `config/generated/chord_gnss_cx*.yaml`
(eight files) and restart.

### P2. `chip_gather3` in the FUSED despread kernel
`gnss_despread_kernel` still makes three separate gathers per hop; it is what the search's
`refine_peak_cuda` and the airspy path run, so the win lands on the SEARCH too. The `fmaf`
pinning is already done. Gate on `cudaGnssDespreadTest`, **not** on a bench (see C4). Do with C1.

### P3. Phi rebuild cost at 100–200 codes — UNMEASURED
`hoprate_filter` builds ~917k `std::exp` per PRN per rebuild, single-threaded, and `refresh_hz`
100 means every PRN rebuilds roughly every ~110 s at L5 Doppler rates. At 200 codes that is ~1.8
rebuilds/s. Probably fine; nobody has measured it, and it is the first CPU-side wall on the way
to a full sky.

### P4. Fleet-coherent gate has only ~2 records of margin
`deep_snr` silently alternates between the fleet value and the single-instance one when the
common-hop overlap fluctuates around `--coh-min-records`. Find the combiner window skew. The
failure mode is a number that looks plausible in both states.

---

## 🔵 Architecture / longer term

### A1. Seeded PRNs should be masked by BORESIGHT ANGLE, not elevation > 0
Boresight is dec +40.73 (8.59° S of zenith), not what `dish_coelev_deg` says. Seeding everything
above the horizon wastes GPU on satellites the dish cannot see and — worse — feeds the analysis
noise probes that look like cohort members (the el<0 trap, which has now cost time twice).

### A2. Cross-node → local migration
`fleet_coherent` scales the wrong way for full CHORD (fewer channels per node). The element axis
grows to 512 and the phase is common across elements, so split-aperture is the long-term answer;
the broker path is the reference implementation to validate it against. Blocked on B1.

### A3. Element health
~8 effective elements with a 14 dB gain spread (13 above 3σ, top 4 hold 53% of the power), so the
honest MRC gain over the reference element is **~2.1×, NOT √32**. Some feeds oscillate 18–23%
frame to frame against 2–12% typical. Not a bug in the pipeline; it bounds every sensitivity
claim made with it.

### A4. F-engine channel availability away from 1176.45 MHz — THE gate on new bands
Every GNSS band is inside CHORD's first Nyquist zone (3.2 GS/s → 1600 MHz), so the code tables are
not the constraint. The constraint is whether the F-engine will deliver comb channels at those
bins, and how much aperture each band gets (today: 7 channels × 195.3 kHz = 1.37 MHz, 6.7% of the
L5 lobe). **Answer this before choosing the second band.** Note 1176.45 MHz is simultaneously GPS
L5, Galileo E5a and BeiDou B2a, so ~100 codes are reachable with no new channels at all.

---

## ✅ Closed 2026-08-05/06 (context)

* **The search was CPU-bound** — blind 427 s/snapshot → live pass 12 ms (acquire ~3 ms, refine
  <1 ms) on the L40S. `gnss_gpu_search.md` §1–8.
* **Node GPUs pinned at 100% / 265 W** — two independent causes, both fixed. (a) The tracker
  latched seeds forever (`sd.have = true` was the only assignment in the file); fixed with
  `seed_ttl_s`. (b) Replica synthesis re-read its own 689 KB Phi slice 24× per launch; fixed by
  the block width (256 → 1024) and the E/P/L gather fusion, **7.6× total, `wave` bit-identical**.
  Now 6–10% at 9 PRNs.
* **"~2.5× of tracker saturation unexplained by the seed latch"** — CLOSED, it was the synthesis
  cost per PRN, which the old two-point fit (6.89 ms intercept) had badly wrong. Measured: at the
  then-current 256-wide kernel a fully latched 32-PRN set costs ~7.1 ms synthesis + ~0.8 ms
  correlation against a 10.486 ms record period ≈ 76% duty before the other stages. That is the
  missing factor; nothing else is needed to explain 100%.
* **The L2-residency plan for the waveform kernel** — RETRACTED, wrong on both its premise (Phi is
  bucketed per PRN, so the live set is 77 MB not 7.3 MB and no two blocks share a slice) and its
  remedy (loses at every channel-group size on parallelism). `gnss_gpu_search.md` §9.8.
* **Phi-table transpose** — RETRACTED. Aimed at coalescing; L1 was already at 78.8%.
