# GuoWang (HULIANWANG DIGUI) L-band signals: what the beam dump established

Status: 2026-11-03. Supersedes the signal description in CHORD_GUOWANG_BEAMDUMP_PLAN.md ("What
the dump told us") and the 10-31 notes: the 10.278 / 2.0556 Mcps chip rates, the offset-QPSK
reading, the chip clock "not coherent with the carrier" and the "sum of several synchronous
codes" reading were all one artefact. Conjugating each channel's samples while keeping the channel
placement mirrors every component about its own channel centre, so a wideband chip-rate line at R
appeared at 2 N x 195312.5 - R with its Doppler reversed. Stitched without conjugation
(`channels_to_baseband(..., conj=False)`, the reader's documented convention since a83cd23d9) the
signal is BeiDou-like and simple. Evidence: ~/gnss/bb1030/dg3_report_20261102.md (DIGUI-26 dump,
chunks 52/61/66, and four live satellites), tools ~/gnss/bb1030/dg3_*.py and dg4_b3fold.py,
outputs cf06:/var/tmp/bb1030/dg3 and dg4.

## 1202.025 MHz lobe (the trackable one)

- Carrier 1202.025 MHz. BPSK, NRZ chips at 2.046 Mcps (no quadrature chips; the earlier OQPSK
  offsets were noise). Frame = 4092 chips = 2.000000 ms transmitted.
- Each frame is one millisecond of a fixed 2046-chip code followed by one millisecond of data
  chips of the same power on the same chip grid. One data sign per two frames (250 sym/s)
  multiplies the whole frame.
- The code is B1I's G1 from phase 1206 XOR a new degree-11 m-sequence G2'
  (1 + x + x^3 + x^8 + x^9 + x^10 + x^11) taken from a per-satellite phase: a BeiDou-style
  assignment by G2 phase. Generator and the known phases: python/scripts/gnss/guowang/
  guowang_codes.py (`guowang_1202`, `GUOWANG_1202_G2P_PHASE`: DIGUI-26 1213, -79 2027, -18 893,
  -97 1100, -176 1518), test vectors in test_guowang.py. Every folded code agrees with its
  generated code at all 2046 chips; single-read chip errors are 0.02-1.5 % on live beams.
- One clock: the chip clock tracks the carrier to within +-9 ppb (one oscillator; the Doppler
  steers the chip rate exactly). On the GPS-disciplined frame0 axis the carriers match the TLE
  Doppler to -10..+20 ppb. (An apparent -250 ppb on the 11-02 passes came from the beam-dump
  window labels, which were 2-3 s early: 2.5 s of the -125 Hz/s Doppler rate is ~300 Hz.)
- The carrier phase steps between the five frame classes in a 10 ms pattern locked to UTC and
  the same on every satellite, about [0, +0.26, +0.45, -0.70, +0.11] rad: ~0.7 dB over five
  frames. It puts weak 100 Hz sidelines on the order-2 line, so a search started from a wrong
  prediction locks on a sideline instead of failing. Confirm carriers by the code.
- The frame epoch is tied to UTC: the code start of each frame minus the nearest UTC 2 ms
  boundary is constant through a pass, and the data symbols start on even UTC 2 ms boundaries.
  13 satellites' codes, the G2' phase rule and the epochs: python/scripts/gnss/guowang/
  guowang_1202_db.json on kv/gw1202 (the cleanup session's database run, 11-03).
- DIGUI-176's data millisecond repeated every 8 ms in one record; the others did not repeat.
- Tracking lanes for beam mapping: docs/CHORD_BUGLIST.md #227.

## 1268.52 MHz lobe (BeiDou B3I's carrier)

- Synchronous, balanced QPSK at 10.23 Mcps: equal I and Q power, one timing for both rails, a
  four-fold phase histogram, no BPSK (order-2) line. Per chip it sits ~10 dB below the 1202 lobe
  (five times the chip rate, two rails), so single chips are not readable (bimodality 0.38-0.41
  against 0.343 for noise).
- Same clock as the 1202 lobe. Driving the B3 lobe with the code-confirmed 1202 solution of the
  same 8 s window (carrier phase x 1268.52 / 1202.025, chip phase 5 x the tracked 1202 frame
  starts) puts the B3 carrier within ~1 Hz (the order-4 line at the expected residual, x550-700
  the median) and the B3 chip-rate line at +0.000 Hz from the model (x2050 the median, 0.125 Hz
  bins): the B3 chips run on the 1202 chip clock to 12 ppb, with their edges on the model grid.
- Code search (dg4_b3fold.py, DIGUI-97 window 7 and DIGUI-79 window 4, 8 s each, the 1202 lobe
  of the same windows as the control): frame-locked folds of 1 ms units with a complex gain per
  unit (rank-1, and rank-2 for two codes on I and Q) at every quarter-ms origin and sub-chip
  offset give noise (fold amplitude 1/sqrt(units), Gaussian bimodality, linear complexity ~ half
  the window); the normalised correlation between 1 ms units is at the independent level at every
  lag from 1 ms to 8 s; a block cross-correlation at chip resolution (2046-chip blocks against the
  40920 chips that follow, incoherent over 40000 blocks) shows nothing above the stitch artefacts
  (lags 52-53 chips = one channel spacing) from 200 chips to 4 ms; the coherent autocorrelation
  over the whole 8 s shows nothing beyond those artefacts either. The control recovers the 1202
  code at every stage (fold 0.98 binary, linear complexity 22, 2 ms period at 51 sigma).
- Verdict: the B3 lobe carries no repeating chip pattern with a period between 20 us and 8 s
  (a repeat of a few percent of a millisecond would have shown). Coverage: periods up to 4 ms
  with data symbols of ten chips or more (block correlation), any whole-millisecond period up to
  8 s with symbols of a millisecond or more (unit correlation), and any period up to 8 s with
  symbols of a millisecond or more (coherent autocorrelation). It is a data stream or a long
  (encrypted) code at 10.23 Mcps, locked to the 1202 clock. Periods between 8 s and the 80 s window are untested (two stitches 40 s apart,
  chip-aligned through the 1202 solution, would test them). For the instrument this means the B3
  lobe cannot be despread by a replica; GuoWang in the B3I band stays with the eigen lanes, and the
  1202 lobe is the one to track.

## Corrections to earlier documents

- CHORD_GUOWANG_BEAMDUMP_PLAN.md, "What the dump told us": chip rates are 10.23 / 2.046 Mcps
  (not 10.278 / 2.0556), the 2 ms frame is 4092 chips of the 2.046 MHz clock, the chip clock is
  coherent with the carrier, the 1202 "preamble" is the 2046-chip code, the 1202 lobe is one BPSK
  code stream (readable), and the B3 lobe is QPSK on the same clock. The instrument design and the
  rail caveats there stand.
- Memory notes chord-guowang-encoding-20261031 before 2026-11-02 22:00Z carry the same artefact;
  the CORRECTION block there points here.
- The 30-40 ns code-timing wander seen in the 10-30 dump beams is absent in the live beams
  (1-2 ns rms against the carrier-aided model), so it belongs to the offline dump beamformer.
