# CHORD multi-constellation / multi-band rollout

Status: PLAN + Phase 0 in progress (2026-08-06). Decision approved by KV: **constellation
first** -- GAL_E5A_Q and BDS_B2A_P on the same comb channels as GPS L5 -- then a second
band, for which the recommendation is **GAL_E5B_Q, not GPS L2C**. Task #23.

Written during the chive `get-frame0-time` outage: everything in Phase 0 is offline work,
gated by `scripts/gnss/e2e` and the boost tests, deliberately requiring no node restart.

---

## 1. What is already in the tree

Nearly everything. The airspy-era prototype was built multi-constellation from the start:

- **Descriptor registry** (`lib/stages/gnss/gnssSignal.hpp`): 28 signals across
  GPS/GAL/BDS/GLO with carrier, chip rate, code length/period, modulation, pilot/data,
  secondary shape, TDM, PRN range. Lookup by config string via `signal_by_name()`.
- **Code generators**: every constellation transcribed and check-scripted
  (`*_code_check.py`). GAL_E5A_Q and BDS_B2A_P are **sky-validated (airspy, 2026-07-15)**;
  E5b/B2b/E6/B3I/L3OC are bit-exact-verified against PocketSDR/ICD tables.
- **Secondary generators**: `galileo::e5aq_secondary(prn)` (CS100),
  `beidou::b2ap_secondary(prn)` (Weil-1021) exist and are wired into the
  `gnssOverlay.hpp` registry (`E5A_CS100`, `B2A_CS100` rows).
- **Tri-constellation ops machinery**: `gps_distributed_broker.py` takes
  `--constellation/--carrier-hz/--chip-rate-hz/--code-length`; the combiner ran the
  G26 + C33 cross-constellation confirm; per-chain buffer naming discipline is documented
  (`cudaGnssTrack.cpp:382` -- a bare `gnss_ring` collides across chains).
- **Band plan**: `gnssBandPlan.{hpp,cpp}` + `config/chord_band_plan.py` map any
  descriptor onto the per-node comb.

What does NOT exist is the CHORD-tracker form of the new signals (§4) and the config
plumbing to run more than one signal per node (§6).

## 2. Comb geometry decides the ordering

A node holds every 8th science-band channel (1.5625 MHz spacing), each GPU a stride-16
comb (3.125 MHz). Channel = carrier / 195312.5 Hz. What each candidate's main lobe puts
on the combs (occupied band = carrier ± chip rate ± 5 kHz Doppler):

COMPUTED, not estimated: `chord_band_plan.py` against the descriptors parsed out of
`gnssSignal.hpp` (`signal_table()`), cx19's comb, |Doppler| ≤ 5 kHz. "band" = channels
in the whole science band; "cx19" = that node's 1-of-8; then its two GPUs' 1-of-16 combs.

| Signal | Carrier MHz | Mcps | band | cx19 | gpu0 | gpu1 | Notes |
|---|---|---|---|---|---|---|---|
| GAL E5a-Q / BDS B2a-P | 1176.45 | 10.23 | 106 (5971..6076) | 14 | **7** | **7** | the L5 comb, byte-identical |
| GAL E5b-Q | 1207.14 | 10.23 | 106 (6128..6233) | 13 | 7 | 6 | pilot, per-PRN CS100; same sats as E5a |
| GLO L3OC-p | 1202.025 | 10.23 | 106 (6102..6207) | 13 | 6 | 7 | pilot, shared NH10; ~7 GLONASS-K sats |
| BDS B3I | 1268.52 | 10.23 | 106 (6442..6547) | 13 | 6 | 7 | DATA-only, shared NH20 + 20 ms D1 |
| GAL E6-C | 1278.75 | 5.115 | 53 (6521..6573) | 7 | 3 | 4 | pilot, per-PRN CS100 |
| **GPS L2C** | 1227.60 | 1.023 | **6** (6283..6288) | **1** | **0** | **1** | CM/CL TDM, CL 1.5 s time-assisted |
| GAL E1 / BDS B1I | 1575.42 / 1561.1 | -- | **0** | -- | -- | -- | above the science band, permanently |

Three consequences:

1. **E5a/B2a are free.** Same carrier as L5 → the identical covering channels, taps, GPU
   input bytes, and record geometry. The rollout is replica-side only.
   `chord_gnss_node.yaml` anticipated exactly this (`also_available`).
2. **L2C is structurally hostile to this instrument.** The whole L2C main lobe is SIX
   channels of the science band, of which cx19 holds one -- and that one lands on gpu1, so
   **gpu0 sees literally nothing**. One channel gives no code-phase discrimination at all
   (the E≈P≈L narrowband blindness taken to its limit: a single channel cannot form a DLL
   discriminant). It becomes meaningful only with full-fleet coherent combining, where the
   6 covering channels are spread across 6 of the 8 nodes. Add the unverified GPU TDM path
   and CM's
   20 ms nav symbols straddling our 10.49 ms records, and L2C is the most work for the
   least measurement. **Deferred indefinitely.**
3. **The whole L1 window is permanently out of reach**: L1 C/A / L1C / E1 / B1C
   (1575.42 MHz) and B1I (1561.098) all sit above the science band's 1499.805 MHz edge
   (max freq_id 7679). The lower L-band is the menu; there is no upper L-band option.

Comb-sharing bonuses for later phases: the L3OC comb overlaps the E5b comb over
6128..6207, and the E6 comb overlaps B3I's over 6521..6547 -- a widened tap serves both
of each pair.

## 3. Why E5b-Q over L2C as the second band -- and how E5a/E5b relate

**They do NOT share a code.** E5a-Q and E5b-Q are both 10230 chips at 10.23 Mcps with
per-PRN CS100 secondaries, but the primary tables are entirely different (verified by
inspection: E5a PRN 1 begins `83F6F69D…`, E5b PRN 1 `CFF914EE…`; the ICD gives E5b its own
register polynomials and X2 start values). What they share is the **satellite**: E5a and E5b
are the two sidebands of Galileo's composite E5 AltBOC(15,10), transmitted together by every
Galileo bird, 30.69 MHz apart.

Nor is the instrument seeing them through one wide passband. The F-engine digitizes
0--1600 MHz so both are *available*, but a tracker chain taps a specific stride-16 comb: E5a
at freq_id 5971..6076, E5b at 6128..6233, with 51 channels of untapped band between. They are
two combs, two taps, two chains.

**Should they be processed jointly? No -- and deliberately not.** There are two ways to
"operate together":

- *Coherent AltBOC*: treat E5 as one 51 MHz signal. This is the high-precision-ranging
  answer -- AltBOC's correlation peak is far sharper than BPSK(10)'s. It is the wrong answer
  **here**: it would require modelling the composite modulation across a 30 MHz gap with
  correct relative phase, and, decisively, it would average the two sidebands' responses
  together. Our measurand IS how the beam changes with frequency; coherent combination
  destroys that by construction.
- *Shared knowledge at the broker* (the right answer): one satellite, one orbit, one clock.
  An E5b chain can be seeded from the same ephemeris and the same receiver clock, and a
  satellite already locked on E5a tells the E5b chain exactly where to look -- the code
  phases differ only by the different code tables and a small differential instrumental
  delay. Independent chains, common sky model.

So the two frequencies stay separate on purpose: same satellite, same sightline, same second,
two frequencies 30.69 MHz apart. That is a chromatic beam measurement with the pointing held
fixed for free -- which is precisely what the roadmap's "split the frequency axis back out"
note is asking for.

### The band comparison

Identical chip rate, code length, primary period, and record geometry to E5a-Q; identical
per-PRN CS100 secondary shape, so ALL Phase-0 machinery (§4) transfers verbatim -- only
the comb (6128..6232) and the code tables differ. Decisively: the **same Galileo
satellites** broadcast E5a and E5b, so every pass measures the beam at two frequencies
71 channels apart simultaneously -- direct chromaticity on identical sightlines, which is
the roadmap's stated reason for un-collapsing the frequency axis
(`chord_gnss_node.yaml` roadmap section). B3I and L3OC stack on afterwards with only new
combs (both 10.23 Mcps, both shared -- not per-PRN -- secondaries).

## 4. The one real construction: per-PRN baked secondaries

CHORD's tracker despreads `GPS_L5_Q_NH` -- NH20 baked into a 204600-chip code -- because
a 10.49 ms record spans ~10 overlay chips whose ±1 signs cancel (measured 2026-07-31: an
snr-40 satellite invisible per record without it). E5a-Q/B2a-P have the same problem with
per-PRN, 100-chip secondaries. Same cure:

- `GAL_E5A_Q_CS` / `BDS_B2A_P_CS` descriptors: primary × that PRN's 100-chip secondary
  = **1,023,000 chips, 100 ms period**, `secondary_length = 0` (nothing left to overlay).
- `signal_code()` ladder arms in `gnssChannelizedReplica.cpp` tile the primary by
  `e5aq_secondary(prn)` / `b2ap_secondary(prn)`.
- **Sign convention is exactly `overlay_sign()`'s** (chip × secondary[period]), so a cp
  in the baked space equals (primary cp + 10230 × period-index mod 100). Pinned by a
  boost test in `tests/boost/test_gnss_channelized_replica.cpp` -- every seeding bug of
  the 2026-08-02 week lived in a convention between stages, invisible to single-stage
  tests.
- Consequences that stay true: a record straddles at most ONE code-period boundary (now
  every 100 ms), so the P_HEAD design assumption holds; the code table is 1 MB/PRN
  (trivial); the CS phase is computable exactly from constellation system time, like the
  NH phase from GPS time.

## 5. Audit: what is parameterized, what is not (checked 2026-08-06)

Clean, with one real finding.

- **No baked constants**: `204600`/`10230` appear only in comments across
  `cudaGnssChordTrack`, `GnssCudaDespread`, `cudaGnssInject`, the despread kernels, and
  both record assemblers. Code length flows from the descriptor.
- **Seed transport is generic**: `detection_phase(..., n_nh, ...)` takes the overlay
  count as a parameter; `cp_long`/`cp_at_ref` are mod n_nh·L by construction
  (`gnssSeedTransport.hpp`).
- **Broker is generic**: `--code-length`, `--chip-rate-hz`, `--carrier-hz`,
  `--constellation` are CLI arguments (`gps_distributed_broker.py:1515`).
- **The Doppler-aiding ratio** (5094.9 chips/Hz) is derived (chip_rate·t/carrier), never
  a literal -- and is numerically identical for E5a/B2a anyway (same carrier, same chip
  rate).
- **FINDING -- the deployed search cannot blind-acquire a per-PRN secondary.** Two facts
  compose. (a) The search's coherent window is a full replica period: at CHORD that is
  3125 hops = **16 primary code periods**, so it spans 16 secondary chips and decoheres
  unless the overlay alignment is known -- which is exactly why `nh_search` scans NH20's 20
  alignments for GPS L5. (b) The bank's overlay slot is single-sequence by design and its
  name ladder deliberately skips per-PRN signals (`gnssChannelizedReplica.cpp:220-248`), so
  for `GAL_E5A_Q` it falls through empty, `secondary_length()` returns 0, and `nh_search`
  silently degrades to a no-op (`GnssChannelizedSearch.cpp:222`, INFO-logged).

  Note precisely what this does and does not say. It is NOT that the primary is
  unacquirable in principle -- a **~1 ms sub-window spans one secondary chip, a constant
  sign that |D|² cannot see**, which is the whole basis of the ms-split acquire. But
  `ms_split_accumulate` lives only in `scripts/gnss/e2e.cpp`; it is NOT wired into
  `GnssChannelizedSearch`, so it is a harness experiment, not a deployable path. For the
  system as it ships, blind acquisition of E5a/B2a does not exist. Three ways out:
  1. **Dead-reckon seeding (chosen)**: GPS L5 lock has already measured the instrumental
     delay and the clock; E5a/B2a cp + CS phase are then predictable from ephemeris +
     constellation time to well inside the DLL pull-in. No bank change, no 100-alignment
     scan (which would be 5× the NH20 acquire cost even if the bank could do it).
  2. Per-PRN secondary slot in the bank (per-p `_secondary`, as `_full_code` already is):
     mechanical but touches the search-critical path. Only if dead-reckoning proves
     insufficient on sky.
  3. **Productize the ms-split acquire** into `GnssChannelizedSearch`: overlay-blind by
     construction, so it needs no per-PRN slot to FIND the peak, and the period is then
     recovered by a postfix of 100 single despreads at the one known phase -- which the
     baked `_CS` banks built here can already do. This is the route that would give
     E5a/B2a a genuine search rather than a model, and it is the same route that would
     let the fleet acquire without GPS present at all.

**How the prototype avoided all of this:** it did not bootstrap E5a from another signal --
it acquired it directly, on the bare `GAL_E5A_Q` primary, because an airspy record was
`hops_per_record: 1000` = **exactly one 1 ms code period**, so the CS100 chip was a constant
sign within every record and the overlay was wiped downstream by the combiner
(`secondary_overlay: E5A_CS100`). That is the record==period coincidence again: it silently
made the acquisition, the record and the wipe all correct at once, and CHORD's 10.49-period
record breaks all three together.

## 6. Phased plan

**Phase 0 -- offline, now (chive outage).**
1. [DONE 2026-08-06] `GAL_E5A_Q_CS` / `BDS_B2A_P_CS` descriptors + ladder arms + boost
   tests pinning the baked-cp convention against the plain-primary bank
   (`e5aq_cs_baked_convention` / `b2ap_cs_baked_convention` /
   `e5aq_cs_multiperiod_coherence`; 17/17 suite green). The coherence test confirms the
   design bound exactly: CS100(PRN 1) sums to -8, predicted overlay-blind despread
   |sum|/100 = 0.080, measured 0.080018.
2. [done -- §5] Hardcode audit.
3. [DONE 2026-08-06] Config generator `--extra-signal NAME:PRNS` (repeatable): one full
   tracker chain per signal, tagged from the signal name (`gnss0_e5a_*`). Same-carrier
   chains SHARE the primary's voltage tap (kotekan buffers are per-consumer); a new
   carrier opens its own. No search feed for extra chains (§5). Per-chain core rotation --
   without it all three chains landed their cudaProcess on one core. Verified: byte-
   identical output on four nodes × four flag sets when no extra signal is given;
   `--check-config` clean; `--dry-run` constructs every stage (an unknown signal FATALs
   there, so the new banks really were built) and the residual no-producer list is
   identical to the single-signal baseline.
4. [DONE 2026-08-06] `scripts/gnss/e2e` closure per new signal: `--signal` flag added
   (tracker name; search primary derived by stripping `_NH`/`_CS`; overlay count from the
   code-length ratio, cross-checked against the bank). The `_CS` signals REFUSE the blind
   search leg with a pointer to §5 and run `--skip-search` (the dead-reckon leg -- the only
   leg that exists for them). Measured: E5a and B2a close at 0.000 chips through the
   shipped GnssCudaDespread on the 1,023,000-chip banks, noiseless AND at --noise 30
   --quantize (deep fold snr 24.3/8 rec, 12.5/4 rec). L5 regression unchanged: 2-node comb
   12.8 chips (the documented grating lobe), 8-node comb +0.373 chips.
5. [DONE 2026-08-06] Broker: `scripts/gnss/broker_up_extra.sh {e5a|b2a}`, one instance per
   constellation beside the GPS one. NO broker code was needed -- every knob already
   existed (`--constellation/--dr-constellation {G,E,C}`, `--long-code-segments/-epoch-s`,
   `--nh-overlay-len`, empty `--detectors` = "purely model-primary ... the CHORD
   configuration"), and `gnss_ephemeris.py` already applies BDT = GPST − 14 s. Verified
   offline: both brokers start, fetch live BRDC (33 Galileo / 38 BeiDou-3 sats) and arm
   BRDC cp seeding with 0 detectors. Two defects found and fixed on the way -- see §7.

**Phase 1 -- F-engine back.** FIRST the two owed path-B measurements (freq map in situ,
combiner coh_frac -- one `--n2-debug` run, §11.13 of gnss_gpu_search.md; do not let this
queue-jump them). Then E5a-Q on cx19 against predicted Galileo transits; per-element A/B
machinery transfers unchanged.

**Phase 2.** BDS_B2A_P, same channels.

⚠️ **PATH B IS NOT MULTI-SIGNAL.** Everything in Phase 0 extends **path A** -- the voltage
tap → `cudaGnssChordTrack` chain. `build_n2dual_branch()` still reads `args.prns` and the
primary `signals.tracker`, so `--n2-dual --extra-signal` today produces three tracker chains
and ONE path-B injector, still on GPS L5. Extending it is a separate piece of work whose
shape is already known: the injector is per-signal (its own seeds, its own replica bank), and
the lane budget is the binding constraint -- 4 lanes/PRN against NSB=128 is 32 PRN slots
TOTAL across all constellations, so tri-constellation at ~25-30 visible sats fits only with
thin margin. NSB=256 (one new kernel instantiation) is the documented mechanical escape. This
matters for ordering: path B is where multi-constellation eventually wants to live, because
its marginal cost per PRN is the injector (~2.4 ms/frame indicative) rather than a whole
tracker instance (~17.9 ms).

**Phase 3.** Second band: E5b-Q comb (6128..6232), new taps + branch, machinery from
Phase 0 verbatim; two-frequency beam chromaticity from every Galileo pass. Then optionally
L3OC (overlapping comb) and B3I/E6 (overlapping pair).

**Deferred.** L2C until the full-fleet coherent era, if ever. GLONASS L2OF/L2OC (FDMA
per-satellite carriers -- per-sat combs on a stride-16 grid is its own study). All
L1-band signals: out of the science band, permanently.

## 7. Defects found while building Phase 0

Three, all latent, all of the same shape: a constant or a promise that was correct for the
signal it was written for and silently wrong for the next one.

1. **The long-code segment step was still L2C's.** `LC_EPOCH`/`LC_SEG` were parameterised
   in an earlier pass, but the segment-search correction still multiplied by a hardcoded
   `0.020` s and the search spiral still spanned ±37 of 75 segments. On any other signal a
   correction of ±1 moved the anchor by 20 segments and the spiral searched 3.7× more space
   than exists (L5 NH20: 1 ms segments, 20 of them). Dormant because the correction is 0
   unless `--cl-autoseg` engages -- "a working launch latches 0 and is untouched". Now
   derived from the two parameters; reproduces L2C's 20 ms / ±37 exactly.
2. **The clock-prime log line promised a self-correction that cannot happen.** It said
   "EMA refines from the first lock" unconditionally, but the solve draws its residuals
   from `best`, which is filled *only* from `/get_detections`. In the model-primary
   configuration (no detectors -- which is exactly what E5a/B2a run) the EMA never executes
   and the primed constant stands for the whole run. The message now states which case it
   is in. This one mattered: believing it would mean shipping a wrong receiver clock
   indefinitely and reading the result as a bad instrumental delay.
3. **Every extra chain landed its stages on the primary's cores** -- see §6 item 3. Filed as
   HYGIENE, not a fix: a cudaProcess thread mostly blocks on GPU completion, and the earlier
   core collision (gnss1_n2dual/gnss0_gpu, 2026-08-06) was investigated OUT as the cause of
   the degradation it was found near. The rotation buys determinism, nothing more.

Consequence for operations: **the extra chains' receiver clock must be primed from the GPS
broker** (`--dr-clock-chips`), and it does not survive an F-engine restart.

How far it transfers: to E5a/B2a **exactly** (same 1176.45 MHz carrier -- same cable, same
F-engine path, same PFB group delay). To **E5b, very probably too**, and an earlier draft of
this doc was wrong to say otherwise: it claimed the clock "does not transfer across a retune",
which is airspy thinking. CHORD does not downconvert or retune at all -- the RFSoC samples
0--1600 MHz directly, so 1207.14 MHz comes off the same cable and the same ADC as 1176.45.
What differs is cable dispersion over 30.7 MHz and any differential filter delay: expected
well under a chip (97.75 ns), but MEASURED-UNKNOWN. Treat a GPS-primed E5b clock as a good
starting guess to be confirmed by its own integrity residual, not as a transferred constant.

## 8. Costs and risks

- **Synthesis scales per PRN** and is the known limiter (docs/gnss_gpu_search.md §10) --
  multi-constellation multiplies PRN count, which is an argument for riding **path B**
  (injector ~2.4 ms/frame indicative) rather than extra path-A tracker instances
  (~17.9 ms). Same trajectory as task #21.
- **Search cost if blind acquisition is ever needed**: 100 alignments vs NH20's 20, on
  top of the §5 bank finding. Dead-reckoning avoids both.
- **Per-instance vs per-job constellation mixing**: cheapest correct first cut is one
  branch per signal (airspy pattern). Merging constellations into one instance's job
  table (per-job code bank) is a later optimization; measure the per-instance overhead
  before building it.
- **Sky validation without the F-engine is impossible** -- Phase 0 gates are synthetic
  (e2e + boost). The airspy sky validation of the E5a/B2a code tables (2026-07-15) is
  the confidence bridge until Phase 1.
