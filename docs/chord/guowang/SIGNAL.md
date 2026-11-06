# GuoWang 1202.025 MHz: the signal

Status: 2026-11-04. What is established about the signal. The measurement record is
`docs/CHORD_GUOWANG_SIGNAL_20261103.md` (beam-dump results, the B3 lobe study, corrections to
earlier documents) and is not repeated here. Tools and the running pipeline: `PIPELINE.md`.
Message content: `PAYLOAD.md`.

## Identity

- ITU Resolution 609 filings register it as RNSS: "China SATNET LEO Augmentation Navigation
  System" (contacts at cn-satnet.com). Filings GW, GW-2, MCNT-02, MCNT-03, CENTISPACE-2 and
  CENTISPACE-3 are treated together as "SATNET LEO"; GW-A59 is not in the group. Stated purpose:
  augmentation (convergence, accuracy, integrity) and RNSS "jointly with COMPASS". The satellites
  are catalogued as HULIANWANG DIGUI (DIGUI-NN here).
- ITU-R M.1787-6 (2026-02) Annex 17 "SATNET LEO" names the signal XW-B2b: 1202.025 MHz,
  MCSK(6,2,1), 2.046 Mcps, 125 / 1500 bit/s.
- Constellation (20th consultation meeting, 2023): 504 satellites = 72 at 86.5 deg / 1175 km
  (6 planes x 12) + 432 at 50 deg / 1150 km (24 planes x 18). Our TLE check (11-03) puts every
  catalogued satellite in these shells (50.0 deg / 1149 km, 86.5 deg / 1168 km); the populated
  inclined planes are 30 deg apart (12 of the filed 24).
- Frequency plan: inclined even planes 1202.025 MHz, odd planes 1174.404 MHz; polar planes
  1178.496 or 1202.025 MHz (the 22nd meeting, 2025, settles on 1202.025). 1174.404 and 1178.496
  are L5 -/+ 2.046 MHz. Every populated plane checked transmits at 1202.025, so the 1174.404 planes
  are probably unlaunched; new inclined planes offset 15 deg should appear at 1174.404. The beam
  dump covers 1195.7-1208.0 and 1249-1288 MHz only, so it cannot see 1174/1178 (freq ids
  ~5998-6050 would).
- pfd at 40 kft, worst 1 MHz, at 1202.025: -129.3..-129.5 dB(W/m2/MHz) near 56 deg elevation,
  -137.5 at 0 deg. Filed spectrum BPSK(2.046)-like: nulls +-2 MHz, first sidelobe -14 dB at +-3 MHz.
- Not public: codes, frame and message structure, UTC epoch, and the 1268.52 MHz lobe. M.1787-6
  files an XW-B3C at 1258.29 MHz, QPSK-R(2), "even planes only"; the 10.23 Mcps QPSK we see at
  1268.52 MHz is not filed. (The 11-04 note "no even planes are launched yet" conflicts with the
  CM20 reading that the populated planes are the 1202.025 ones; unreconciled.)
- Mopra (SNIFFLES I, MNRAS 550, 2026) sees narrow DIGUI lines at 1767.74-1786.16 and
  2379.50-2418.37 MHz on exact n x 2.046 MHz: chip-clock harmonics.

## Modulation

- Carrier 1202.025 MHz, BPSK, NRZ chips at exactly BeiDou's 2.046 Mcps, natural Doppler. The chip
  clock tracks the carrier to +-9 ppb: one oscillator per satellite. Carriers match the TLE
  Doppler to -10..+20 ppb on the frame0 axis.
- Frame = 2.000000 ms = 4092 chips, two halves on one chip grid at equal power:
  - ms 1, the ranging (BPSK) period: a fixed 2046-chip Gold code, BeiDou B1I's G1 started at phase
    1206 XOR a new degree-11 m-sequence G2' (1 + x + x^3 + x^8 + x^9 + x^10 + x^11) started at a
    per-satellite phase (counted from all-ones, 0..2046). Satellites differ only by the G2' phase;
    PRN = G2' phase + 1 in our descriptor (gnss::max_prn() = 2047 for it). One sign per two
    frames (4 ms symbols, 250 sym/s): the low-rate channel.
  - ms 2, the CSK period: a second Gold code of the same family (G1 at 1206 XOR G2' at another
    phase), cyclically shifted by k = 0..63 chips: 6 bits per 2 ms, 3000 bit/s raw. Its polarity
    varies too (~54 % positive). Correlation 2045-2046 of 2046 chips on DIGUI-150 (CSK G2' 1436)
    and DIGUI-152 (1726). This is MCSK: China SatNet patent CN119788292A/B (filed 2025-03-04,
    "BPSK code periods and CSK code periods alternate") and CSNC 2024 Yan et al., who define
    MCSK(U, N, n_low) = U bits per CSK period, N code periods per TDM block, n_low ranging periods
    and work MCSK(6,2,1) at 1 ms, 2.046 Mcps, 3 kbit/s.
- No chip encryption: the patent's chip-replacement authentication would put chip errors in the
  last 200 chips of the CSK period; 0 of ~77k frames show it. The chip-error rate ramps 2 -> 7 %
  across the millisecond (reader phase drift), nothing else. Single-read chip errors 0.02-1.5 %
  on live beams; a full code is recovered from one pass.
- Companion lobe at 1268.52 MHz (BeiDou B3I's carrier, same satellites): synchronous balanced
  QPSK at 10.23 Mcps on the same clock (chip edges on the 1202 model grid to 12 ppb). No repeating
  chip pattern from 20 us to 8 s (repo doc); the whole-window sweep (DIGUI-97, ten 8 s segments
  tied by seq0) extends that to every whole-ms lag up to 80 s (unit correlation at the 0.0089
  floor; the 1202 control gives 0.41-0.47 to 40 s). Nearby B3 units correlate 0.018 at all lags
  below ~10 ms: a narrowband component, not a code. Not despreadable; only the 1202 lobe is
  tracked.
- A stationary, intermittent B3-lobe emitter (chip line 10,230,010.8 Hz, drifting -2.5 ppm in
  2 h; windows 5, 9, 12 on 11-02) is not a satellite; location open.

## Code families and sharing

- Code number: c = V - 448, V = the first 11 G2' chips as an integer (first chip = LSB). The
  codes in use run c = -1..234 (not 1..120 as first read).
- PLANE & SLOT RULE (11-05, builds/gw1202/constellation.py; zero exceptions):
  * each 50 deg plane uses 5 codes, each polar ring of 10 uses 2; the blocks ascend in ALMANAC ID
    order (PAYLOAD.md): polar rings (IDs 22-83) lowest, then the 50 deg planes in ID blocks 1-9,
    10-18, ... 100-108 (planes 2, 1, 0, 3, 4, 5, 11, 10, 9, 7, 8, 6 in the map's numbering);
  * a 50 deg plane has 18 slots 20 deg apart; almanac satellite j (= ID - block start) sits at
    s' = 2j, the inferred H satellites at odd s'; group g = (s' mod 9) mod 5 takes the plane's
    (g+1)-th lowest code: a code at u, u+80, u+180, u+260 (the 80-100-80-100 pattern), group 4
    at u and u+180 only. Nine-satellite planes: j = 0..8 -> g = 0,2,4,1,3,1,3,0,2;
  * polar ring: even slot -> the lower code, odd -> the higher.
  Tests: leave-one-out over 102 measured, 81/81 with a measured slot-mate predicted exactly, the
  other 21 inside the block-order bounds; the sweep's 28 on-sky confirmations all agree; 58 groups,
  no conflict, no order violation. Coverage of 204: measured 102, predicted 59, forced 2 (plane 2
  group 4 = 50), bounded 41 (20 in the silent polar planes). Per-plane codes (c): p2 [<41,41,47,
  49,50] p1 [51,54,64,67,70] p0 [72,74,80,81,85] p3 [86,..] p4 [100,103,104,108,..] p5 [117,119,
  120,121,123] p11 [124,131,132,134,..] p10 [137,138,141,143,147] p9 [149,154,164,168,173] p7
  [174,175,179,180,182] p8 [185,187,189,198,203] p6 [214,220,..,232,234]; rings RAAN 265 [-1,1],
  234.6 [11,14], 204 [20,30], 356 [32,..]. OPEN: which c are skipped (not the CSK codes, not
  balance); it would pin the bounded 41.
- Two satellites on one code are 80 deg or more apart, beyond the ~64 deg mutual-visibility limit
  at 1150 km, so they are never both above CHORD's horizon: cellular-style spatial reuse.
- CSK code pairing: ranging G2' -> CSK G2' is a fixed per-code table with no arithmetic offset
  (37 satellites mapped 11-04; e.g. 1144 -> 1436, 1402 -> 1726, 390 -> 1041, 1585 -> 1002;
  `builds/gw1202/csk_streams/csk_codes.json`). Satellites sharing a ranging code share the CSK
  code. No CSK code is also a ranging code in use.
- Quiet satellites cluster by plane: polar plane 300 (DIGUI-06, -07; the plane of DIGUI-01..10 was
  never detected, DIGUI-07 passed at 0.25 deg), polar plane 330 (DIGUI-69, -70), 50 deg plane 135
  (DIGUI-41, -46, -47). The two polar planes with no code are the quiet ones. The even/odd plane
  rule does not explain them; candidates are 1174.404 / 1178.496 MHz or off.

## Timing

- The frame epoch is tied to UTC. theta = t_tx(code start) - nearest UTC 2 ms boundary is constant
  through a pass (<= 0.6 chip); data symbols start on even UTC 2 ms boundaries; CSK 1 s frames on
  the UTC second; low-rate 3 s subframes on the UTC 3 s grid. GPST - UTC = 18 s is a multiple of
  2, 4 and 10 ms; BDT = UTC + 4 s.
- theta is DECLARED by the satellite itself in slow page 1, payload bits 42-51 (signed 10 bit):
  theta = -11.7 us - 2.07 us x field (49 satellites with dump theta: median |resid| 0.9 us, max
  4 us; DIGUI-55 field -28 -> +46.3 vs +46.9 us measured, so the scale holds far out). Readable
  only once tracked (a drift monitor, not a first-acquisition seed). No page carries OTHER
  satellites' theta (slow_theta_others.py: every per-cycle live window as (ID, theta), 11-05).
- theta per satellite (incl. receiver delay; ~0.1 us depends on the beam's reference input,
  116 in production): most span -10..0 us (~20 chips), median -9 us; branches at +400 us
  (DIGUI-176, -177) and +46.9 us (DIGUI-55). Across satellites rms 16 chips after the common
  offset, worst 40.
- Across days: DIGUI-158 locked on 23.5 h-old theta with mean discriminator 0.000 chip;
  DIGUI-152 dtheta = -0.72 us/day; DIGUI-94 / -112 read -9.875 / -6.375 us on the 10-30 baseband
  and -13.125 / -4.125 us in the chain on 11-04 (~0.5 us/day, or the lock-carry effect: open).
- The carrier phase steps between the five frame classes in a 10 ms pattern, UTC-locked and the
  same on every satellite, ~[0, +0.26, +0.45, -0.70, +0.11] rad (~0.7 dB over five frames); the
  frame-to-frame sign reader models per-class steps of ~+8, +16, +10, -67, +76 deg. It puts weak
  100 Hz sidelines on the order-2 line; confirm carriers by the code. 11-05: the pattern is one
  ranging half in five rotated ~-80..-90 deg at full amplitude, carrying no page symbol (12-15 %
  of frames; seen in the track beam on 80 satellites and on the 10-30 baseband).
- That rotation is locked to RECEIVER time, not to the satellite (11-06, `builds/gw1202/rot5_lock.py`
  on the track-beam syms): within a pass the rotated frame's position in the satellite's frame
  count (n mod 5) walks by the integrated range change, one slot per 2 ms of propagation delay,
  rising while the satellite approaches and reversing at closest approach. DIGUI-57 (PRN 857,
  01:15-01:30Z 11-06): slot 0 -> 2 -> 4 (= -1) against a predicted +2.1 then -3.5; PRN 1276
  (01:13-01:24Z): 3, 3, 3, 2, 1, 0, 0 against 3, 3, 3, 2, 1, 0, 0. Simultaneous satellites walk in
  opposite directions. A transmitter feature would hold its slot. So it is a 100 Hz phase effect in
  our chain upstream of both the GPU tracker and the baseband capture (analog, digitiser or
  F-engine), and the page-symbol erasure it causes (~-1.9 dB of the wipe) is ours to remove.
  Not yet checked: whether other constellations' channels show it.
- LEO geometry: Doppler rate -115..-140 Hz/s, passes 1-2 min near boresight; TLEs predict the
  chip-rate Doppler to 0.2-1.2 Hz, which identifies satellites in a far sidelobe.

## What was ruled out

- Chip rates 10.278 / 2.0556 Mcps and the "x 1.00468" factor (10-31): a mirror artefact of the
  offline stitch (conjugated samples, unchanged channel placement); retracted 11-02 21:00Z.
- OQPSK on the 1202 lobe: no quadrature chips; it is BPSK.
- "A sum of synchronous codes / multi-level composite", "Gaussian-like chips": the same artefact.
- Chip clock not coherent with the carrier, +1.5 ppm drift, mirrored Doppler and gateway
  pre-compensation (11-02 20:40Z): the same artefact; one clock, natural Doppler.
- The 2 ms repeat as a narrow "preamble" with a minor fraction of the power: it is the whole
  2046-chip ranging code.
- Transmitters -250 ppb low (11-03 00:30Z): beam-dump UTC labels 2-3 s early times the
  -125 Hz/s Doppler rate. The 10-30 dump's +13.5 ppb was a TLE range-rate bias.
- "Code starts at an arbitrary chip" (11-02): the epoch is UTC-tied (11-03).
- The data millisecond as non-repeating data chips (DIGUI-176's 8 ms repeat as its only
  structure): it is the shifted CSK code (11-04).
- A replica-despreadable B3 lobe (10,230-chip B3I-like codes): nothing repeats, 20 us to 80 s.
- 30-40 ns code-timing wander: the offline dump beamformer; live beams show 1-2 ns rms.

## Pointers

- `docs/CHORD_GUOWANG_SIGNAL_20261103.md` (corrects `CHORD_GUOWANG_BEAMDUMP_PLAN.md` and
  `CHORD_TAUA_GUOWANG_20261029.md`); `PIPELINE.md` and `PAYLOAD.md` here.
- Code generator and database (kv/gw1202): `python/scripts/gnss/guowang/guowang_codes.py`,
  `guowang_1202_db.json`, `test_guowang.py`.
- Regulatory and literature copies, with a README: `~/gnss/builds/gw1202/itu_refs/` (CM20 2023
  attachment, CM22 2025 report, SNIFFLES I, CSNC 2024 Yan et al. and full proceedings, Pocket SDR
  LDPC tables). Patents: CN119788292A/B (MCSK signal), CN115685267A (receiver), CN110244328B
  (MCSK multiplexing), CN120949268B (B3 QPSK analysis).
- Dump-era analysis: `~/gnss/bb1030/dg3_report_20261102.md`, `dg3_*.py`, `dg4_b3fold.py`.

Sources: chord-guowang-encoding-20261031.md, chord-guowang-1202-is-mcsk.md,
chord-guowang-itu-satnet-leo-filing.md, chord-guowang-payload-20261104.md,
chord-guowang-tracking-design-20261102.md, chord-guowang-handoff-20261103.md (code reuse, quiet
planes), chord-guowang-handoff-20261104.md (library counts, cross-day theta),
chord-guowang-csk-lane-plan-20261104.md (theta drift), docs/CHORD_GUOWANG_SIGNAL_20261103.md,
builds/gw1202/itu_refs/README.md.
