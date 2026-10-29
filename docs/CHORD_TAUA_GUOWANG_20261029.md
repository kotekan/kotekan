# Tau A through the GNSS bands, and the uncatalogued GuoWang sets (2026-10-29)

The Tau A transit at 11:03:28 UTC on 2026-10-29 was observed with the full GNSS fleet (8 nodes, cx42/cx47 back for
the night) and recv1 recording. cx19 ran with no N2 projection (the control); the other seven nodes projected the
GNSS-exempt channel ranges. This note records what the data showed about deprojection, about Tau A under it, and
about GuoWang (bug #218): three planes carry a second, uncatalogued set of satellites, and the burst spectrum is a
GNSS-rate BPSK with a non-sinc^2 envelope.

Work directory (cf06, home is shared): `~/gnss/builds/scratch_bak_20261023_gnssmain/taua1029/` (Tau A, deprojection,
orbit fits, videos) and `.../burstspec1028/` (rail spectra). Every script named below lives there.

## 1. Setup

- Fleet: cx19 cx27 cx42 cx43 cx44 cx47 cx51 cx52 on `fleet-f90cbafc7`; cx47 and cx42 rejoined 01:03 and 01:06 UTC with
  no regeneration needed (`gen_fleet.py --check` passed, the aggregator already listed both). recv1 started via choco
  at 01:07. Array handed to the baseband-dump test at ~13:00 UTC.
- recv1 subset files: `/mnt/cs00/data/kotekan_vis_files/subset/acq_20261029_010709_064298776/` (21 GB per 6.6 min;
  40 bins of ~9.8 s; 128 inputs; 6145 channels at 0.1953 MHz; `vis` chunked 16 channels x all products).
  `time_center_ut1_ns` counts from J2000 (2000-01-01 12:00): add 946728000 s for Unix time.
- Streams: a channel's stream is `freq_id % 16`; cx19 owns phases 4 and 14. Projection acts only on the exempt
  ranges 5971-6076, 6102-6338, 6361-6401, 6442-6573 (1166.2-1186.7, 1191.8-1238.0, 1242.4-1250.2, 1258.2-1283.8 MHz).
- Satellites near boresight in the window: R07 (GLONASS) 0.53 deg at 10:56:49; C35 (BeiDou-3) 4.80 deg at 11:16:19
  (first sidelobe); GuoWang bursts listed in section 4.

## 2. Tau A under deprojection

Products: `demo_waterfall.png` (top eigenvalue / channel noise, cx19 vs the adjacent projected channel),
`raw_waterfalls.png` (autos and cross coherence on 6.3 m and 74.1 m baselines), `fringes.png` (Re rho as recorded
and phase-referenced to Tau A), `taua_deproj_20261029_v4.mp4` and `..._v4_pauses.mp4` (sky + waterfalls + spectra;
the second holds 0.5 s at each GuoWang closest approach).

**Tau A recovered / expected coherence** (`taua_flux.py`: coherence toward Tau A per channel, instrument phases from
recv1's top eigenvector at transit fitted as a delay per input over clean channels, expectation interpolated from
1100-1160 and 1290-1340 MHz; the clean channels themselves recover 0.99-1.00; peak coherence 0.067 at 1130 MHz):

| deprojected band | +-3 min of transit | +3..+8 min | during R07 (-12..-6 min) |
|---|---|---|---|
| L5 / E5a / B2a | 0.78 | 0.83 | 0.83 |
| E5b / B2b / L3OC / L2C | 0.71 | 0.86 | 0.85 |
| GLONASS L2 | 0.93 | 0.94 | 0.58 |
| B3I / E6 | 0.62 | 0.78 | 0.86 |

- **Dimensions removed.** From gain-corrected adjacent-channel autos (`imprint_test.py`; outside the projected ranges the
  ratio is 0.998), projection removes a fraction f of every input's noise power: L5/E5a/B2a 0.41 (about 52 of 127
  dimensions: three constellations share the band), B3I/E6 0.29, L3OC/L2C 0.21, E5b 0.18, GLONASS L2 0.105.
- **Tau A loss tracks f.** Tau A's loss is roughly 1 - f (GLONASS 0.93 vs 0.90, E5b 0.71-0.86 vs 0.82, B3I/E6
  0.62-0.78 vs 0.71); L5 recovers better than 1 - f (~0.8 vs 0.59).
- **A dip at transit.** Recovery drops further at the transit peak: the residual-eigen path takes part of Tau A when
  it is the strongest thing in the beam.
- **R07's lane.** R07's lane, 1-2 deg from Tau A, took ~40 % of Tau A in GLONASS L2 during its pass.

**Suppression** (eigenvalue over channel noise; clean floor 12.6 dB): quiet sky reaches the floor (control +4..+8 dB,
deprojected +1.3..+1.9 dB). Strong satellites: R07 at 0.5 deg +26 -> +8.4 dB (18 dB removed), C35 in the first
sidelobe +18 -> +3..+4 dB (14-15 dB). The leftover is the real depth limit (lane error ~12 % for 18 dB, or power
beyond one direction).

**The deprojected cross-coherence floor** is not thermal noise (thermal -64 dB per baseline per 10 s): in-range
-26.9 / -30.7 dB (6.3 / 74.1 m) against -37.8 / -41.5 dB out of range in the same streams. It does not follow the
satellites (slope of deprojected on cx19 coherence 0.0-0.2 over a ~30 dB swing) and is close to the projection's
imprint on the noise, sigma^2 P with |P_ij| ~ sqrt(f/N) / (1 - f) (0-5 dB on the short baseline, 4-8 dB high on the long
one). Residual satellite power adds at the strongest events. Per baseline, Tau A (-24 dB) sits only a few dB above it.

**What survives deprojection, and how to remove it:**
- R07's military L2 code (5.11 Mcps on 1246 + k 0.4375 MHz) spills to 1239-1242 and 1250-1257 MHz, outside the
  projected L2 range. A lane is a direction, so the delay-aware lane transfer can carry R07's lane across: widen the
  projected L2 range to ~1237-1258 MHz (meeting B3I's at 1258.2). Split the projection list from the SK exemption list
  first: they are the same list today, and 1240-1260 MHz is near aircraft radar.
- C35's wideband B2 (ACE-BOC, ~1166-1217 MHz) leaks at 1160-1166 and in the 1186.7-1191.8 gap: fill the gap and extend
  below 1166.2. Its in-range residual (~4 dB at the sidelobe pass) is not understood (dish-to-dish sidelobe differences
  break the single-direction lane is the leading guess).
- GuoWang: not a lane target. The continuous sidelobe sum shows as striated coherence across 1258-1279 MHz in the
  deprojected short baseline; the bursts are partly removed (section 5).
- The L5/E5a/B2a band spends ~52 of 127 dimensions; worth reviewing whether all ~40 lanes earn their keep.

## 3. Method notes (each cost time)

- Steering conventions: V_ab ~ a_a conj(a_b); remove exp(i (psi_a - psi_b)); geometric sign +1 (the instrument-phase
  delay fit holds coherence 0.87 with it and 0.08 with the opposite sign).
- Storage reads ~5-9 MB/s per reader on /mnt/cs00 regardless of ionice class; run readers in parallel by file.
- healpy `get_interp_val` next to UNSEEN pixels leaks values near -1e26: mask `img < -10`, not `== UNSEEN`.
- NFS between the VM and cf06 can serve a stale script for seconds after an edit: check md5 on both sides before a run.
- A wait loop `while pgrep -f "<pattern>"` inside `ssh '...'` matches its own command line and never ends: start jobs
  with `&` in the same shell and `wait`.
- `cd dir && cmd1 & cmd2 &` backgrounds the `cd` with cmd1; cmd2 runs in the old directory.

## 4. GuoWang: bursts on 10-29 and three uncatalogued sets

**Bursts in the window** (rail at 1 s; separations from the orbit model):

| burst (UTC) | satellite | closest approach | |
|---|---|---|---|
| 10:31:46 (weak) | DIGUI-H24-3 | 7.8 deg | new |
| 10:44:19 | DIGUI-H24-2 | 2.6 deg at 10:44:13 | new |
| 10:51:14 | DIGUI-106 | 0.26 deg at 10:51:15 | catalogued |
| 10:56:52 | DIGUI-H24-1 | 2.8 deg at 10:56:52 | new |
| 11:04:02 | DIGUI-21 (polar) | 1.35 deg at 11:04:01 | catalogued |
| 11:11:32 | DIGUI-138 | 3.6 deg | catalogued |
| 11:24:00 | DIGUI-140 | 1.6 deg at 11:23:59 | catalogued |

The 10:31 / 10:44 / 10:57 train is spaced exactly 12 min 38 s, GuoWang's in-plane spacing for a 50-deg plane of nine
(T = 108.3 min, T/9 = 722 s, x 1.051 for Earth rotation = 759 s). Nothing at 1.2 GHz in CelesTrak's whole active
catalogue (15,902 objects) is within 4 deg / 60 s of those bursts (`burst_any.py`).

**Orbit fit from the array** (`burst_img.py`, `orbit_fit.py`):
- **Difference images.** Per burst bin, the visibilities minus the 60-150 s neighbours (removing Tau A, steady GNSS
  and diffuse sky) are imaged over +-8 deg with the Tau A-calibrated phases.
- **The grating lattice.** The 6.3 x 8.5 m grid makes each image a lattice of replicas (~2.2 x 1.6 deg). The two
  GuoWang lobes (1202, 1268 MHz) scale the lattice by 5.5 %, so sqrt(I_1202 x I_1268) keeps the true streak and drops
  the cross-track replicas.
- **The fit.** Circular SGP4 orbits (50-deg shell: inclination 50.0, 13.2933 rev/day) are fitted to the streak over
  1-s samples, coarse stage on a blurred image.
- **Validation.** DIGUI-106 to 0.04 deg RAAN / 0.07 deg u, DIGUI-140 to 0.03 / 0.01. Polar orbits keep an ascending /
  descending mirror ambiguity (DIGUI-21's true solution came second).

**Result.** The unknown train is in the plane of DIGUI-105..113 (RAAN 24.5 deg), at u = 49.26 + 40 k deg at
10:51:18 UTC: exactly the midpoints between the nine catalogued satellites (26.8, 67.0, 108.5, ... 347.5). Propagated
back, the fitted set predicts the unmatched 10-26 16:13:03 burst to 2 s (and 11:46:30, inside an excluded window).
Midpoint sets in every catalogued 50-deg plane (`midpoint_check.py`) explain all remaining unmatched bursts with two
more planes, and no others:

| plane (RAAN) | catalogued | unmatched bursts explained |
|---|---|---|
| 24.5 deg | DIGUI-105..113 | 10-29 10:44, 10:57 (fitted, <= 4 s); 10-26 16:00, 16:13 |
| 54.1 deg | DIGUI-87..95 | 10-26 13:43:15 (0 s), 18:09:44 (-11 s) |
| 84.4 deg | DIGUI-73..80 | 10-26 15:27 / 15:40 / 15:53 train (<= 5 s) |

The other planes' midpoints predict close passes that never appear (for example the 324-deg plane at 10-29 11:17:45,
1.04 deg, inside recv1 data). The 114-deg plane already has 18 catalogued satellites (two interleaved batches): the
hidden sets are most likely a recent launch campaign whose orbital data is not public yet.

**TLEs** (`make_hidden_tle.py`): `hidden_guowang_20261029.tle` (27, named `HULIANWANG DIGUI-H<plane>-<k>`, epoch
2026-10-29 10:51:18) and `guowang_plus_hidden_20261029.tle` (catalogue + hidden), usable directly as
`guowang_passes.py ... --tle <file>`. Plane 24.5 is fitted (timing ~+-5 s, separation ~+-0.5 deg); planes 54.1 and 84.4
are midpoint estimates (~+-40 s).

**Friday 10-30 capture window** (`~/gnss/builds/guowang1028/passes_1030_with_hidden.txt`):
DIGUI-105 10:30:35 (0.93 deg); **DIGUI-H24-9 (new) 10:36:08 (1.65 deg)**; Tau A transit 10:59:32;
**DIGUI-26 11:00:03.6 (0.07 deg)**; DIGUI-145 11:03:27 (1.35 deg). A dump on H24-9 would capture an uncatalogued
satellite for the code search.

**Ground map:** https://claude.ai/artifact/EBrpVnCdLG7eK8s3WuBxEZ (source `guowang_over_chord.html`,
`map_data.py`): every plane, the catalogued and found satellites, CHORD's beam, and the bursts.

## 5. GuoWang burst spectrum

Source: the rail recorder (raw voltages, before any projection, 1 s), converted and inverted for the 4-bit
requantisation with the #218 tools (`railx.py`, `railspec.py`), fitted with `spec_fit.py`
(`spec_fit_20261029.png`); channels saturated at the rail ceiling are excluded. recv1 autos cannot be used: in
projected channels deprojection removes ~70-85 % of a burst (cx19's unprojected channels read 3-6x higher).

- **Nulls: GNSS-rate BPSK.**
  - Main lobe: nulls 1258.3 / 1278.7 MHz = 1268.52 +- 10.23: BPSK at 10.23 Mcps on the BeiDou B3I carrier.
  - 1202 lobe: nulls 1200.0 / 1204.1 MHz and first / second sidelobes at the BPSK(2) positions (1199.1, 1205.0,
    1197, 1207): BPSK at 2.046 Mcps (BeiDou B1I/B2I rate) on GLONASS L3's 1202.025 MHz. Peak 3.5x the main lobe's.
  - Centres and rates are good to ~0.05 MHz; the same on all six bursts, new and polar included.
- **Envelope: not white-code sinc^2.**
  - Flat tops (main lobe within 1-2 dB over +-6-7 MHz; sinc^2 is -3 dB at +-4.5 MHz).
  - First sidelobes 4-6 dB high (-9 dB at ~1281 MHz; -7 / -9 dB around 1202; sinc^2 -13.3).
  - Not MSK (needs 21.6 Mcps, wrong nulls); not BOC (no centre null).
- **Two candidate explanations.**
  - A chip sequence with more transitions than random: lag-1 correlation -0.17 (main) / -0.13 (1202), 57-58 %
    transitions. It gets the trend, not the whole shape.
  - Transmitter spectral shaping: a band-limited fit is best on the main lobe.
  - The baseband code search settles it (transition rate and period come straight out).
- **Oddities.** A narrow feature at 1274.0 MHz in every burst (~+1 dB); the upper first sidelobe peaks near
  1281 MHz, not 1283.

## 6. Open

- Baseband capture Friday 10-30 (DIGUI-26 and DIGUI-H24-9 windows above); code search with chip rate and centre
  pinned to 10.23 / 1268.52 and 2.046 / 1202.025.
- Refine planes 54.1 and 84.4 interferometrically when they next cross with recv1 recording; refresh the hidden-set
  TLEs as the predictions are tested.
- Deprojection: widen L2 and E5 ranges as in section 2 (after splitting the SK-exemption list), review the L5 lane
  count, model the imprint sigma^2 P for science use, and decide whether GuoWang lanes (TLE-steered, tier 2) are worth
  their cost now that hidden sets exist.
