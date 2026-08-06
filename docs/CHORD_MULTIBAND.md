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

| Signal | Carrier MHz | Mcps | Lobe | freq_ids | chans/GPU | Notes |
|---|---|---|---|---|---|---|
| GAL E5a-Q / BDS B2a-P | 1176.45 | 10.23 | 20.5 MHz | 5971..6075 | **7 (the L5 comb, byte-identical)** | pilots, per-PRN CS100 |
| GAL E5b-Q | 1207.14 | 10.23 | 20.5 MHz | 6128..6232 | ~7 (new comb) | pilot, per-PRN CS100; same sats as E5a |
| GLO L3OC-p | 1202.025 | 10.23 | 20.5 MHz | 6101..6206 | ~7 | pilot, shared NH10; ~7 GLONASS-K sats |
| BDS B3I | 1268.52 | 10.23 | 20.5 MHz | 6442..6547 | ~7 | DATA-only, shared NH20 + 20 ms D1 symbols |
| GAL E6-C | 1278.75 | 5.115 | 10.2 MHz | 6520..6573 | ~3 | pilot, per-PRN CS100 |
| **GPS L2C** | 1227.60 | 1.023 | 2.05 MHz | 6280..6290 | **0..1** | CM/CL TDM, CL 1.5 s time-assisted |

Three consequences:

1. **E5a/B2a are free.** Same carrier as L5 → the identical covering channels, taps, GPU
   input bytes, and record geometry. The rollout is replica-side only.
   `chord_gnss_node.yaml` anticipated exactly this (`also_available`).
2. **L2C is structurally hostile to this instrument.** One GPU's comb catches at most ONE
   195 kHz channel of a 2 MHz lobe: no code-phase discrimination at all (the E≈P≈L
   narrowband blindness taken to its limit -- one channel cannot even form a DLL
   discriminant). It becomes meaningful only with full-fleet coherent combining, when all
   ~11 covering channels exist across 8 nodes. Add the unverified GPU TDM path and CM's
   20 ms nav symbols straddling our 10.49 ms records, and L2C is the most work for the
   least measurement. **Deferred indefinitely.**
3. **The whole L1 window is permanently out of reach**: L1 C/A / L1C / E1 / B1C
   (1575.42 MHz) and B1I (1561.098) all sit above the science band's 1499.805 MHz edge
   (max freq_id 7679). The lower L-band is the menu; there is no upper L-band option.

Comb-sharing bonuses for later phases: the L3OC comb overlaps the E5b comb over
6128..6206, and the E6 comb overlaps B3I's over 6520..6547 -- a widened tap serves both
of each pair.

## 3. Why E5b-Q over L2C as the second band

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
- **FINDING -- blind search cannot see per-PRN secondaries.** The bank's overlay slot is
  single-sequence by design and its name ladder deliberately skips per-PRN signals
  (`gnssChannelizedReplica.cpp:220-248`), so for `GAL_E5A_Q` the slot falls through
  empty, `secondary_length()` returns 0, and `nh_search` silently degrades to a no-op
  (`GnssChannelizedSearch.cpp:222`, it INFO-logs this). Multi-period blind acquisition of
  E5a/B2a therefore does not exist today. Two ways out:
  1. **Dead-reckon seeding (chosen)**: GPS L5 lock has already measured the instrumental
     delay and the clock; E5a/B2a cp + CS phase are then predictable from ephemeris +
     constellation time to well inside the DLL pull-in. No bank change, no 100-alignment
     scan (which would be 5× the NH20 acquire cost even if the bank could do it).
  2. Per-PRN secondary slot in the bank (per-p `_secondary`, as `_full_code` already is):
     mechanical but touches the search-critical path. Only if dead-reckoning proves
     insufficient on sky.

## 6. Phased plan

**Phase 0 -- offline, now (chive outage).**
1. [DONE 2026-08-06] `GAL_E5A_Q_CS` / `BDS_B2A_P_CS` descriptors + ladder arms + boost
   tests pinning the baked-cp convention against the plain-primary bank
   (`e5aq_cs_baked_convention` / `b2ap_cs_baked_convention` /
   `e5aq_cs_multiperiod_coherence`; 17/17 suite green). The coherence test confirms the
   design bound exactly: CS100(PRN 1) sums to -8, predicted overlay-blind despread
   |sum|/100 = 0.080, measured 0.080018.
2. [done -- §5] Hardcode audit.
3. Config generator: `signals` list in `chord_gnss_node.yaml` → one tracker/inject branch
   per signal on the shared comb; constellation-suffixed buffer names.
4. `scripts/gnss/e2e` closure per new signal (inject known cp → seed → despread, error in
   chips) -- the gate before anything touches a node.
5. Broker: instance-per-constellation (airspy pattern) with dead-reckon seeding; the
   Galileo/BeiDou ephemeris fetch already exists from the tri-constellation nights.

**Phase 1 -- F-engine back.** FIRST the two owed path-B measurements (freq map in situ,
combiner coh_frac -- one `--n2-debug` run, §11.13 of gnss_gpu_search.md; do not let this
queue-jump them). Then E5a-Q on cx19 against predicted Galileo transits; per-element A/B
machinery transfers unchanged.

**Phase 2.** BDS_B2A_P, same channels. Watch the path-B lane budget: 4 lanes/PRN × 32
slots = NSB 128; tri-constellation visible ≈ 25-30 sats fits with thin margin, NSB=256
(one new kernel instantiation) is the documented mechanical escape.

**Phase 3.** Second band: E5b-Q comb (6128..6232), new taps + branch, machinery from
Phase 0 verbatim; two-frequency beam chromaticity from every Galileo pass. Then optionally
L3OC (overlapping comb) and B3I/E6 (overlapping pair).

**Deferred.** L2C until the full-fleet coherent era, if ever. GLONASS L2OF/L2OC (FDMA
per-satellite carriers -- per-sat combs on a stride-16 grid is its own study). All
L1-band signals: out of the science band, permanently.

## 7. Costs and risks

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
