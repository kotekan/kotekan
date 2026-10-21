# CHORD multi-constellation / multi-band rollout

**Status: done.** The 2026-08-06 plan (task #23: constellation first -- GAL_E5A_Q and
BDS_B2A_P on the GPS L5 comb -- then second bands) has landed: `config/gnss_chains_chord.yaml`
runs ten chains, gps_l5, gal_e5a, bds_b2a, gal_e5b, bds_b2b, bds_b3i, gal_e6, gps_l2c, glo_l3oc
(since 2026-10-17, §6) and glo_l2oc (since 2026-10-18, §6.2), all on path B. What stays here is the design that still
applies: the comb geometry (§2-3), the baked per-PRN secondaries the `_CS` chains run (§4),
and why those chains cannot blind-acquire (§5). The phased plan, the defects found building
Phase 0 and the cost notes are at `be7d46e8e:docs/CHORD_MULTIBAND.md` §6-8.

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

## 6. GLONASS L3OC: the fourth constellation (2026-10-17)

The chain is `glo_l3oc`: GLONASS L3OC-p, the dataless pilot of GLONASS's CDMA signal at
1202.025 MHz, 10230 chips at 10.23 Mcps with the shared NH10 secondary, so the B2b/B3I
geometry one lobe down from E5b. Everything it needed was in the tree bar five things, listed
here so the next constellation (L2OC, or the FDMA L2OF) knows where to look.

- **The tracker code.** `GLO_L3OC_P` carries `secondary_length 10`, but the bank's overlay
  ladder is for the search; the trackers despread a tiled code (§4). `GLO_L3OC_P_NH` is the
  pilot with NH10 tiled in, 102300 chips / 10 ms, built in `gnssChannelizedReplica.cpp` as
  `GPS_L5_Q_NH` and `BDS_B3I_NH` are. Without it a 10.5 ms record sums NH10, which is +2 of
  10: -14 dB. The broker pairs the two (`gnss_broker/signals.py`).
- **Capability.** L3OC rides the K generations (GLO-K1, K1B, K1+, K2) and the modernised
  GLO-M+; the plain GLO-M half of the constellation is FDMA-only. The IGS registry's block
  types decide (`prnmap.signal_incapable_prns`, now modelling R beside G and C); no PRN floor
  can, since the generations are interspersed: on 2026-10-17, slots 4, 5, 6, 8, 9, 11, 12, 16,
  19, 20, 21, 22, 23, 24 carry it, 14 of the 28 assigned. The manifest's list is those 14.
  The Celestrak fallback (`gps_beamtrack._glonass_block_filter`) sees only the K marker, so
  with the registry unreadable it would drop the six M+ slots: a degradation, not a fault.
  Because slots change hands, `prnmap.ensure_registry()` re-fetches the SINEX past 7 days
  at every process start (broker, viewer, `gen_fleet --check-prns`) and the broker re-reads
  the capability set from the file every 6 h (`almanac._refresh_capable`, logged on change).
- **The PRN.** The L3OC code index is taken to be the orbital slot, the number BRDC uses.
  The ICD does not promise it, so the sky verifies it: a slot whose code index differed would
  be seeded on its orbit and never present.
- **The time scales.** The chain adopts gps_l5's clock, in GPS time; the GLONASS code epoch
  is UTC(SU)+3h, 18 s behind and 3 h ahead, both whole multiples of the 10 ms code. Only the
  sub-microsecond GLONASS-GPS system offset survives, a chain-wide bias inside the spectrum
  fit's capture (+-1/(channel spacing) = +-1.6 chips). The orbit model is the ICD's integrated
  state vector (`gnss_ephemeris.glonass_state`, RK4 with J2 and the record's lunisolar term),
  meant for +-15 min about tb. Validated against the next record on the cached days 289-290
  (571 records, 25 satellites; the next record's own few metres included):

  | record age | line-of-sight error, median / p90 |
  |---|---|
  | 30 min | 1.2 m / 2.4 m |
  | 1 h | 4.6 m / 9.3 m |
  | 2 h | 18 m / 43 m (0.6 / 1.5 chips) |
  | 3 h | 46 m / 114 m |
  | 4 h | 117 m / 273 m |

  The hourly BRDC merge (records at :15 and :45, published ~20 min after the hour) keeps
  the newest record 35-95 min old, inside the spectrum fit's capture and at the DLL's
  (+-0.4 chip) edge at the stale end. The hourly merge is the normal supply. With it down,
  R rests on BKG's current-day daily, which carries R ~25 min behind and is re-fetched once
  it is an hour old (`gnss_brdc_supply._DAILY_TTL_S`), so a record is under ~2 h old at use:
  the hour, the lag, and up to two 15-min sky refreshes. Past 2.5 h
  (`gnss_ephemeris.GLONASS_MAX_AGE_S`: a clock-grade `predict_all` window is cut to that for
  R, where G/E/C hold for 4 h) the chain drops the satellite rather than seeding it chips
  off. With BKG down as well no daily covers today (CDDIS publishes closed days only), and
  the chain loses its satellites 2.5 h after the last hourly record.
  `_HOURLY_TARGET` now counts R (16 slots) so a merge thin in GLONASS is retried like one
  thin in BeiDou.
- **The excision.** The lobe, 1191.8-1212.3 MHz, adds freq_ids 6102-6127 below E5b's; the
  dual correlator applies the SK mask to the synthetic lanes, so those channels had to join
  the exemption (`rfi-excision-exempt-freq-ids`), and with it the N2 projection's range.

Per GPU the chain despreads 6-7 channels, 4-5 of them E5b's, at 2 lanes per slot in a 64-lane
range at base 128 beside gal_e5b and bds_b2b (the dual axis stays at 384 with 192 used).
The chain carries bds_b2b's broker key set (model-primary, clock adopted across bands, no
carrier loop); a GLONASS ray is observed on one carrier only, so it adds no tau_band equation
and is judged on lock and presence. The obs writer is `gnss-obs@glo_l3oc` (the stack target
lists it); the cube, TEC and transit-arc tables know the chain; the viewer's stream-health
rows name the lower E5b shoulder L3 now that it is flown, and its GLONASS sky layer takes
the registry's capability set rather than the Celestrak K marker.

**First sky (10-17 22:25Z broker restart, fleet on `b4ab492da` from 22:50Z):** R08, R11 and R19
present fleet-wide within minutes, q 3.6-3.8 on 11/12 instances, prompt C/N0 38-54 dB-Hz,
discriminators within +-0.3 chip of the siblings; the other eight chains unchanged (every node
within +-0.17 of the fleet median in q). So the code index is the slot for those three. Watch:
R09 (GLO-K1B) stayed absent at 13-14 deg while R19 was present at 19 deg and R12 (M+) at
5 deg; R20 has no broadcast record today. GPU 55-66 % per card against stage 2's 54-59 %.

### 6.1 What the other GLONASS bands would take

- **L1 (1602 MHz FDMA L1OF, 1600.995 MHz L1OC): nothing to do.** The science band ends at
  1499.8 MHz (freq_id 7679).
- **L2OC-p (1248.06 MHz, CDMA):** built 2026-10-18 as glo_l2oc, §6.2.
- **L2OF (1246.0 + k x 0.4375 MHz, k = -7..+6, FDMA, every satellite):** the only way to
  the 14 GLO-M slots, 586 m chips (511 kcps, the L2C scale). Four pieces. (1) Per-PRN
  carrier: the chain's covering set is the union over k, 1242.9-1248.6 MHz = freq_ids
  6364-6392 (about two per GPU), and each PRN's k x 0.4375 MHz rides the Doppler the broker
  commands (the published Doppler is the command; the waveform kernel rotates to any
  frequency), with k from `gnss_ephemeris.glonass_freq_channels()` and the capability gate
  "has a k" (`gps_beamtrack.glonass_fdma_trackable`). The chips-per-Hz aiding ratio must use
  each PRN's own carrier. (2) The data: 50 bps Manchester, +-/-+ halves of 10 ms, so a
  10.5 ms record straddles a sign flip half the time (deep periodic fades); tile a 20-chip
  overlay (ten +1, ten -1) into a 10220-chip, 20 ms code (`GLO_L2OF_MC`, the `BDS_B3I_NH`
  pattern), which leaves the 50 bps sign constant per record, as B3I's D1 is. The bit
  epoch is GLONASS time, so the clock mod 20 ms comes from gps_l5's NH joint fit as for L2C.
  (3) The code is one m-sequence for every satellite: separation is the 437.5 kHz carrier
  spacing against the 1 kHz line spectrum of a 1 ms code, as every GLONASS receiver does it.
  (4) Exemption 6364-6392 and a chains block with the L2C keys. Medium effort; the
  constellation count is the prize, TEC only against L3OC for the 14 K/M+ slots.

### 6.2 GLONASS L2OC: the second GLONASS carrier (2026-10-18)

The chain is `glo_l2oc`: GLONASS L2OC-p at 1248.06 MHz, broadcast by the K generations only
(GLO-K1B, K1+, K2: slots 6, 8, 9, 11, 19, 20, 22, 23 on 2026-10-18; the six M+ slots carry
L3OC alone, which `prnmap` already said). With glo_l3oc it gives the K satellites a carrier
pair, so tau_band and TEC (0.0202 m/TECU; `gnss_tec_chord.py --pair glo_l3oc:glo_l2oc`, out
of the defaults until both chains' observables cover the days asked for). The modulation and
the band set five things apart from L3OC.

- **The signal.** L2OC-p's base code is L3OC-p's, clocked at 511.5 kcps with each chip
  expanded to [0, 0, +c, -c]: the data component's TDM slots, then a BOC(1,1) half-cycle pair.
  The code table carries the expansion, so `GLO_L2OC_P` is plain BPSK at 2.046 Mcps, 40920
  chips per 20 ms, half of them zero, and the chain's chips are 2.046 Mcps chips (146.5 m).
  Its autocorrelation crosses zero at +-0.67 chip and bottoms at -0.5 at +-1, and its spectrum
  has two lobes at +-0.8 MHz with a null at the carrier.
- **OC2 is baked into the code.** The pilot carries a shared 50-chip overlay, one chip per
  20 ms period, 1 s. Earlier text here said a 10.5 ms record sits inside one 20 ms period, so
  a per-period wipe would do: it does not. Records are hop-aligned, so about 52% straddle a
  period boundary, and OC2 flips at 24 of its 50 boundaries. The combiner's overlay wipe cannot
  help either: it steps one overlay chip per record and needs a nav-bit wipe behind it, both
  airspy-era shapes (records were 1 ms periods). `GLO_L2OC_P_OC` therefore tiles OC2 into one
  2,046,000-chip, 1 s code (the `GLO_L3OC_P_NH` pattern, 2 MB per slot on the GPU, 28.6 MB
  for the fourteen the manifest holds with the dark probes, under gal_e5a's 32.7). The ICD starts each OC2 period on a GLONASS second
  mark. GLONASS time is UTC(SU)+3 h and GPS time UTC+18 s, both whole seconds, so the GPS
  second carries the overlay epoch and the broker places the 1 s segment from absolute time
  (good to the +-10 ms this needs), exactly as it places gal_e5a's 100 ms segment. The clock is
  still kept mod the 20 ms primary, from gps_l5's NH joint fit, as gps_l2c's is.
- **Record slot 2.** A float32 over 2,046,000 chips has a 0.125-chip ulp, a fixed bias per
  seed rather than noise (18 m). Every reader of slot 2 wraps it mod the 40920-chip primary
  (`trkresid.phys_chips`), so the assembler now reduces it there first
  (`record_cp_mod_chips`, set by the generator for this tracker code alone) and keeps
  0.004 chip.
- **One or two channels per GPU.** The lobe, 1246.0-1250.1 MHz, is freq_ids 6380-6401, so
  every GPU holds one channel or two 16 apart, and the six nodes hold 17 of the 22 (6385,
  6387, 6393, 6395 and 6401 are on cx47 and cx52), about 60% of the power. cx44 holds the two
  brightest (6386, 6394); cx43 GPU 0 holds the carrier null. The chain packs two lanes per
  slot like every other under `--epl-from-prompt`: a rotated E and L on one channel are the
  prompt times a phase, so a single instance has no discriminator of its own, but the code
  loop closes on the lobe-coherent sum across the fleet (combdll, gnssFleetDll), where the
  rotation acts on the pooled 17 channels; at the real channel geometry the rotated
  discriminator's slope is within 0.3% of the true one with no zero-crossing bias
  (fixtures/gpu_budget/epl_rotation_sim.py). The 0.24 rad intra-channel ramp of the 0.4-chip
  taps is symmetric between E and L.
- **The DLL spacing is 0.4 chip, at fleet-trim-bandwidth 1.4.** On the fleet's 17-channel sum:

  | spacing | q on peak | slope vs the trim law | linear to | sign kept to |
  |---|---|---|---|---|
  | 0.33 | 2.55 | 1.06 | +-0.30 | +-0.79 |
  | 0.40 | 4.37 | 1.81 | +-0.23 | +-0.91 |
  | 0.50 | 15.0 | 4.24 | +-0.13 | +-1.09 |

  q's ceiling decides it: the lock bars (dead-reckon's `lock_q`, the viewer's `Q_LOCK`, the
  q-stall bar) are a fixed 2.2, which a BPSK chain at 0.5 clears with a ceiling of 4, and
  0.33 would cap L2OC at 2.55, so a satellite held on peak would read unlocked unless bright.
  At 0.4 the slope is 1.81x what the trim law assumes (it scales a triangle's), so the chain
  runs the fast loop at 1.4 rather than the chains' 2.5, the same closed loop; at 2.5 it
  would run ~4.5/s, past the round-trip stability bound. Beyond +-0.91 the BOC side peaks
  (+-1.05) are stable false locks, so a seed must land within about +-0.9 chip (130 m).
  Node (`dll_spacing_chips`) and broker (`dll-spacing`) match; re-derive if the node set
  changes. No phase-slope fit: only the five two-channel instances would inform it, ambiguous
  by 0.65 chip, and its taus would feed b_sat once a sixth such instance existed, so the
  chain sets `spectrum-endpoints` to null (it would otherwise inherit gps_l5's).

- **The noise probes are dark satellites.** Presence needs 3 probes. Slots 20 and 22 have no
  broadcast record, so six K satellites are live, and in a 24 h sweep (days 289-291) fewer
  than 3 of them sit below -15 deg 35% of the time; when four are up, at most two can be
  below the horizon. The chain would read UNANCHORED then. A satellite that cannot broadcast
  the signal is signal-free at any elevation, so `--probe-dark` (on glo_l2oc only) takes the
  probes first from the capability-excluded satellites that have ephemeris, in PRN order:
  the six M+ slots (4, 5, 12, 16, 21, 24), which the manifest lists beside the eight K slots
  (14 in all, `gen_fleet --check-prns` names them "dark" on a chain that sets probe-dark).
  The plain GLO-M slots are dark too, so `prnmap.probe_order` puts the dark slots the nodes
  hold first, and orders probes for both the selector and the PRN map: the map holds the
  probes the selector picks and swaps nothing. A dark probe is a satellite glo_l3oc tracks,
  so dark probes are kept out of the receiver-wide range-rate rows (the coarse feed skips
  them; the KCOH feed skips every probe; #192 (h) asks the same for below-horizon probes). Without the registry the chain falls back to the
  below-horizon rule.

The lobe joins the excision exemption (6380-6401, which also holds the upper half of the FDMA
L2OF comb, k = 0..+6; an L2OF chain would add 6364-6379)
and with it the N2 projection's range. Lanes: base 0, 4 x 32, alone on its channels. Cost: one
more synthesis launch per record per GPU, ~0.6 ms per frame.

**First sky (fleet on `954af6047` from 02:40Z 10-18, broker 02:43Z):** R23 (GLO-K1B) present at 34 deg, fleet q 6.4,
deep SNR 296, prompt C/N0 52 dB-Hz, disc 0.09: K1B broadcasts L2OC, the code index is the slot, the 1248.06 MHz carrier
is right, and a deep SNR that high says the OC2 epoch is right too. The dark probes R04, R05 and R12 read q 0.97-1.06.
q on peak runs above the 4.4 modelled for 0.4 spacing, so the model's channel weights understate the fleet sum's width;
re-derive from the live taps before moving the spacing. Watch: R11 (K1B) absent at 13.5 deg while setting, though
present on L3OC. GPU 52-66 % per active card.

To verify on the sky: presence of the K1B slots on L2OC (they are the bulk of the list), the
code index = slot as for L3OC, the 1248.06 MHz carrier (an airspy-era note doubted it against
1248.0; 60 kHz is outside the +-5 kHz Doppler window, so a wrong carrier reads as absent
everywhere), and the OC2 epoch: a wrong one leaves q and the prompt normal but the deep,
KCOH and coherence numbers low, since only the sign sequence across periods would be wrong.
