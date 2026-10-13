# CHORD multi-constellation / multi-band rollout

**Status: done.** The 2026-08-06 plan (task #23: constellation first -- GAL_E5A_Q and
BDS_B2A_P on the GPS L5 comb -- then second bands) has landed: `config/gnss_chains_chord.yaml`
runs eight chains, gps_l5, gal_e5a, bds_b2a, gal_e5b, bds_b2b, bds_b3i, gal_e6 and gps_l2c,
all on path B. What stays here is the design that still applies: the comb geometry (§2-3), the
baked per-PRN secondaries the `_CS` chains run (§4), and why those chains cannot blind-acquire
(§5). The phased plan, the defects found building Phase 0 and the cost notes are at
`be7d46e8e:docs/CHORD_MULTIBAND.md` §6-8.

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
  `beidou::b2ap_secondary(prn)` (Weil-1021) exist. The `gnssOverlay.hpp` registry holds
  only the shared overlays of data components; its per-PRN rows (`E5A_CS100`, `B2A_CS100`
  among them) were removed 2026-10-13, since the combiner refuses a per-PRN overlay.
- **Tri-constellation ops machinery**: `gps_distributed_broker.py` takes
  `--constellation/--carrier-hz/--chip-rate-hz/--code-length`; the combiner ran the
  G26 + C33 cross-constellation confirm; per-chain buffer naming matters (a bare
  `gnss_ring` collides across chains).
- **Band plan**: `gnssBandPlan.{hpp,cpp}` + `config/chord_band_plan.py` map any
  descriptor onto the per-node comb.

What did NOT exist was the CHORD-tracker form of the new signals (§4) and the config
plumbing to run more than one signal per node (the generator's `--extra-signal`).

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
   least measurement. Deferred in this plan; the gps_l2c chain came later.
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
  `ms_split_accumulate` was only ever a harness experiment (`scripts/gnss/e2e.cpp`, removed
  2026-10-13; at `be7d46e8e`), never wired into `GnssChannelizedSearch`. For the system as
  it ships, blind acquisition of E5a/B2a does not exist. Three ways out:
  1. **Dead-reckon seeding (chosen)**: GPS L5 lock has already measured the instrumental
     delay and the clock; E5a/B2a cp + CS phase are then predictable from ephemeris +
     constellation time to well inside the DLL pull-in. No bank change, no 100-alignment
     scan (which would be 5× the NH20 acquire cost even if the bank could do it).
  2. Per-PRN secondary slot in the bank (per-p `_secondary`, as `_full_code` already is):
     mechanical but touches the search-critical path. Only if dead-reckoning proves
     insufficient on sky.
  3. **Productize the ms-split acquire** (from `be7d46e8e`) into `GnssChannelizedSearch`: overlay-blind by
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
