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

### B1. Split-aperture "beats the genie" — DIAGNOSED 2026-08-06. Not an estimator fault.
**The estimator is sound. The bound it was judged against was the wrong one, and the split itself
is degenerate.** Three separate findings, all measured (`scripts/gnss/sumtrack_test`):

**1. The genie does NOT move with the injected amplitude.** Across 8 seeds, genie at 0.75 rad vs
1.20 rad: 72.7/70.7, 75.3/71.8, 52.1/54.6, 42.0/39.3, 55.9/56.9, 23.0/21.6, 38.8/39.8 — under 7%
in seven of eight. Only seed 4242 (41.7 → 20.5) shows the effect the original entry described,
and **that is the seed the entry was written from**. A phase common to all elements factors out of
a linear combine, so invariance is what theory predicts; the harness just has enormous
realisation variance (the genie ranges 19.7–71.8 across seeds, because each seed draws a new array
GAIN realisation). One seed cannot support a conclusion here.

**2. THE SPLIT IS NOT SPLITTING.** `rebuild_split` balances the two halves by `Σ|w|`, but each
half's SNR² goes as `Σ|w|²`. At the live operating point one element carries ~5× the summed
weight of all the others, so the greedy largest-first assignment cannot balance even `Σ|w|`
(0.834), and in SNR² terms the integrating half A holds **99.3–99.8%**. Half B — the phase
REFERENCE — holds ~0.5% of the array's SNR². Consequences:
* The √2 price of the one-way split is never paid: A is effectively the whole array.
* `arg(B)` is a near-noise phase reference, so the correction is far weaker than intended, and
  the disjoint-halves protection against self-reference is nominal rather than real.
* The weak rows in the same test balance fine (0.491) — this only bites at the live gain spread,
  which is why it was never visible.

**3. The bound was the FULL-aperture genie; the split's ceiling is the HALF-aperture genie.**
Against the correct bound (`ElemCal::combine_half_genie`, added for this) the split is
**0.92–0.95 in every case** — comfortably below it, no self-reference. And the full genie is
sometimes WORSE than the half genie (20.5 vs 30.6 at 1.20 rad) because the full combine includes
half B, which at the live point holds the oscillating excess-noise feeds; the split drops them.
So the split legitimately exceeds the full genie without exceeding anything it should not.

Shuffled null behaves correctly throughout: equal to the split with no sky (29.9 vs 28.2 —
nothing to destroy), and collapsing to 12.1 / 6.5 as the sky phase grows while the split holds
~29. The estimator is recovering real common phase. Note the null cannot be driven to ~1 here:
`e^{iφ}` has a non-zero mean `e^{-σ²/2}` that is common across elements and survives any shuffle.

**⚠️ `sky_deep` is still NOT cleared** — but the reason has changed. There is no bound violation
to explain. What must be fixed first is finding 2: make `rebuild_split` balance `Σ|w|²`, and
decide what to do when a single element dominates so far that no partition balances (cap the
dominant weight, or declare the split unavailable and fall back). Until then the split rung is
running with a ~0.5%-weight phase reference and its apparent performance is coming from
integrating the whole array, not from the correction.

### B2. CUDA acquire declines a valid blind Doppler grid — blocks the blind slot
`GnssCudaAcquire::set_doppler_grid` hard-rejects grids that are not bin-aligned, and the blind
grid the search offers is being refused. Consequence: `blind_prns_per_pass` is pinned at 0, so
there is no cold-start reacquisition path and the chain depends entirely on hinted PRNs. The
rejection is deliberate (the fine-lag DFT fold is only exact on an aligned grid) — the question
is whether the blind grid should be snapped or the fold generalised.

---

## 🟠 Correctness / accuracy

### C1. GPU chip gather is UNSNAPPED — attempted 2026-08-06, MEASURED AND REVERTED (costs 5.3×)
`hoprate_stream` (CPU) snaps the chip/tap boundary to the exact per-sample expression
(`66ca1583a`); `chip_gather` still uses the float floor. Floors peel depth at about **−53 dB**.

**The port was built, proven correct, and reverted on cost.** Numbers, all at CHORD geometry:

* Ported `snap_khi` matching `hoprate_stream_into` line for line → **0 disagreements** against the
  exact assignment over a full 3125-hop replica period (662500 chip boundaries). It works.
* Cost: synthesis **0.375 → 1.973 ms, 5.3× slower**. The kernel becomes fp64-bound, and the block
  width and gather fusion collapse from 7.6× to ~1.67× because they no longer address the limit.
  At 100 codes that is ~14 ms/record against a 10.486 ms period — over real time.

**⚠️ THE CHEAP GATE DOES NOT EXIST, and the reason is worth keeping.** The obvious optimisation is
to run the fp64 check only when the float boundary is near a tie, since "the argument lands on an
exact integer" is how the CPU comment describes the failure. That is **wrong**: measured, the
correction fires on **30.1% of boundaries**, at fractional parts as far as 0.04 from a tie.
`floor((phi+d)*inv_cps)` and `floor(C - k*cps)` are INVERSE relations and inverting a floor is not
exact, so with `cps` rational the mismatch is a broad systematic congruence, not a tie. The CPU's
"~8 times per replica period" counts hops where the resulting ERROR is large — the moved tap
landing where `proto[k]` is significant — not boundaries that move.

So the trade is: −53 dB of peel depth, which is not a limiter today, against most of the 7.6×.
**Not worth it now.** Revisit if peel depth becomes the limit, and if so look for an exact
formulation that avoids re-evaluating the big-magnitude `C - k*cps` per chip (an incremental
carry is cheap but would NOT reproduce the CPU bit-for-bit, which may or may not matter).
Instrument: `snap_ab.cu` in the session scratchpad — worth re-creating, it settles this in one run.

⚠️ `cudaGnssDespreadTest` CANNOT gate this: the first affected hop is 255 and it runs 125, so
every number it prints is identical with the snap on or off. That is how it stayed hidden.

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

### I1b. ⚠️ cf06 REBOOTS ~WEEKLY AT 03:00 AND NOTHING THERE COMES BACK
Unattended kernel upgrades reboot cf06 at **03:00 UTC**, observed Sat 18 Jul, Thu 30 Jul, Thu
6 Aug (6.8.0-136 -> 6.8.0-137). **Expect the next one around Thu 13 Aug.**

The six cx nodes survive it — they are `systemd-run` units and come back. **cf06's three services
do not exist to systemd at all**: the aggregator (:12050), the broker (:12060) and the viewer are
manual `nohup setsid` processes, so a reboot silently ends the run. `/tmp` is wiped with it, so
the broker log — the only overnight telemetry there is — goes too.

It has already cost one overnight soak: nodes restarted 02:56 on 6 Aug, cf06 rebooted 03:03, and
the instrument ran for **7 minutes** and then sat idle for 7 hours. The symptom is deceptively
calm — nodes `active` with `NRestarts=0`, GPUs at 1-2% (front-end only, no despread), and
`/get_trim` still listing PRNs because the trim EMA is a HIGH-WATER MARK that persists past seed
expiry. Nothing announces it. The tell is `Connection refused` to 10.222.3.6:11040 in the node
log, and `pgrep -f gps_distributed_broker` returning nothing on cf06.

Restore (aggregator FIRST — the broker's `--detectors` points at localhost:12050):

    ssh cf06 'GNSS_BIN=$K/build_nodpdk/kotekan/kotekan $K/scripts/gnss/agg_up.sh'
    ssh cf06 'nohup setsid $K/scripts/gnss/broker_up.sh > /tmp/gnss_broker.log 2>&1 &'

⚠️ `agg_up.sh` defaults `GNSS_BIN` to `build/` — the **DPDK node build**, which is wrong on cf06.
It must be given `build_nodpdk/` explicitly or it will not start.

**Fix:** installed systemd units (not transient ones — see `node_up.sh`'s note on why transient
units vanish on stop) with `Restart=always`, plus a log path outside `/tmp`. Failing that, a
watchdog on port 12060. Any soak that spans 03:00 is otherwise unreliable.

### I2. The null floor is a MAX statistic and spikes
One cycle read 16.6 against a typical 4–6 (characterised max 5.73 over 726 samples). A spike
briefly suppresses marginal detections. Consider a high percentile, or an EMA over cycles.

### I3. `log_level: WARN` hides deployment confirmations
The node configs that matter run at WARN, so INFO lines that confirm a setting took effect
(`despread_max_chips`, kernel split, fleet counts) are invisible exactly where they are needed.
Design flaw, not yet fixed — a confirmation of a deployed setting should not be INFO.

---

## 🟢 Performance / scaling

### P1. DONE 2026-08-06 — `despread_max_chips` reverted to 0 on all eight node configs
The truncation is obsolete: 212 chips (the full PFB span) at the current kernel is **cheaper than
the 140-chip kernel that the cap was introduced for**. The cap bought 1.31× by throwing away part
of the channel response; the width + fusion buy 7.6× and throw away nothing. Costs 1.42× at
scale, which at 100 codes is still comfortable. Edit `config/generated/chord_gnss_cx*.yaml`
(eight files) and restart.

### P2. `chip_gather3` in the FUSED despread kernel — DONE, but for consistency not speed
`gnss_despread_kernel` now makes one gather per hop instead of three. Bit-exact (every gate still
`0.000e+00`).

**Measured effect: none.** 10007 vs 10303 rec/s at 12 PRN and 6841 vs 6826 at 32 PRN — inside the
noise, slightly negative at 12. The reason is the geometry AGAIN: the despread test's bench runs
airspy-scale `n_chips ≈ 8`, where the Phi slice fits in L1 and there is no re-walk to remove. The
CHORD-geometry consumer is the search's `refine_peak_cuda`, and no instrument here isolates it —
`e2e --cuda-refine` reports 7.30 s against the CPU path's 7.47 s, i.e. that stage time is host-side
setup, not the kernel. The live search's refine is already <1 ms of a 12 ms pass, so the ceiling
is small regardless.

**Kept anyway, on a different argument:** both kernels now call the SAME gather, and they are
required to produce bit-identical replicas. That is the same reason `chip_gather` is shared in a
header rather than copied. Do not quote a speedup for it.

### P0. Synthesis cost is THE scaling limiter — one lever measured (see gnss_gpu_search §10)
At 100 codes synthesis is ~3.5–4 ms of a 10.486 ms record, before correlation. Six ceiling tests,
then ncu, settled where the cycles go: **DRAM BYTES**, at ~66% of peak (245 MB in 536 µs =
457 GB/s). NOT coalescing — the gather is maximally uncoalesced at 31.5 of 32 sectors per warp
load, and hop-sorting cut that to 11.9 and ran **17% slower**.

* **fp16 Φ: 1.27–1.37×, the only lever that worked**, and it works by halving the TABLE. Accuracy
  is not the constraint (3.3e-04 per chip step). **Not yet implemented in production.**
* Rejected, all measured: Doppler-factored/shared Φ (0.93×), interleaved float4 (1.03–1.08×),
  hop-sorting (0.81×, and it eats fp16's win), lockstep locality ceiling (1.05×).
* Ceiling if memory were free: 2.13×. So ~half the runtime is memory, and fp16 takes half of that.
* **The bigger lever is fewer OUTPUTS:** E/L exist only to feed the DLL discriminator and could run
  on a subset of hops, taking the trial factor 3.0 → ~1.5, a ~2×. Physics trade (discriminator
  noise; `trim_quality_min` is calibrated on `q = 2P/(E+L)`), so it wants a decision, not a patch.
* **OPEN:** ~30 MB of hop-sorting's +55 MB DRAM is unexplained — its scattered stores account for
  only ~11–23 MB, and it has FEWER load sectors while moving MORE DRAM. `lts__t_sector_hit_rate`
  on the two variants is the next counter.
* Energy is ALREADY integrated (register accumulate + one tree reduction, 4 writes per
  job-channel = 2% of stores). Nothing to win there.

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

### A4. Band coverage — ANSWERED 2026-08-06, and it changes the plan
The whole **300–1500 MHz** band is already distributed across the 8 test nodes. So coverage
exists TODAY for everything except the L1-ish top:

    GPS L5 / Gal E5a / BDS B2a   1176.45   ✓  (running)
    GLONASS L3OC                 1202.03   ✓
    Gal E5b / BDS B2b            1207.14   ✓
    GPS L2 / L2C                 1227.60   ✓
    GLONASS L2                   ~1246     ✓
    BDS B3I                      1268.52   ✓
    Gal E6                       1278.75   ✓
    BDS B1I                      1561.10   ✗  above 1500
    GPS L1 / Gal E1 / BDS B1C    1575.42   ✗  above 1500

**⚠️ ASSUME NO PATTERN OR CONTINUITY in the frequency→node/GPU assignment.** A band's channels
may land on arbitrary nodes and arbitrary GPUs, in arbitrary order. Consequences:

* `chan_ids` must stay fully general — arbitrary, sparse, unsorted. This is why the merge kept
  our constructor over the prototype's `(n_chan, chan_offset)`; a contiguous range cannot express
  the assignment and silently builds every replica at the band centre (the original lock blocker).
* A "band" is NOT a node-local object. The per-node covering set for a given signal has to be
  DERIVED from the live assignment, not configured by hand or assumed contiguous — and a node may
  hold channels from several bands at once, or none from a band it is nominally serving.
* The fleet combine already spans nodes, so this is mostly a config-generation problem: the
  generator needs the actual assignment as input.

Open: where the authoritative frequency→node map comes from, and whether it is stable across
F-engine restarts. That is the remaining question, not the coverage.

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
