# T3: Quantization

T3 checks that the 4-bit outputs use their range sensibly. For noise input, the upchannelized
voltages, the baseband (BB) beams and the FRB3 beams must not be mostly zero and must not be
mostly saturated. FRB3 must encode FRB2 within half a quantization step.

## Errors this test catches

- **Wrong normalization**: a missing or extra factor of U or √U in the upchannelizer, a wrong
  window normalization, a wrong `upchan_gain` or `bb_gain`, or a wrong BB shift. The output is then
  almost all zeros or clipped at ±7.
- **Wrong int4 encoding**: offset-encoded versus two's complement, swapped nibbles. This shows
  up as -8 values, heavy saturation or a wrong rms.
- **Missing clamping**: values beyond ±7 that wrap around instead of saturating, or a
  quantizer that emits -8 (no data) for valid data.
- **FRB3 encoding**: a wrong offset or scale, offset and scale swapped, a wrong chunk layout or
  nibble order, wrong rounding, or -8 that does not match the FRB packet-loss (PL) mask.

## Noise model

The input is complex Gaussian noise with σ = 2.8 LSB per component on every input. This is the
level of RadioTelescopeFEngine. A weak tone (amplitude 1) from θ = (0.5°, 0.8°) is added.

**Rounding.** Let `x ~ N(0, s²)` be rounded to the nearest integer and clamped to [-7, 7]. Then
the value k has probability

    P(k) = Φ((k + ½)/s) − Φ((k − ½)/s),

with the tails beyond ±6.5 added to ±7. From this, rms² = Σ k² P(k), the fraction of zeros is
P(0), and the fraction saturated is P(−7) + P(7). Every stage below is modelled as Gaussian
noise with some `s` before rounding. The rounded input has rms σ_in = 2.785.

**Upchannelizer.** It computes

    Ē_u = (G/U) Σ_{s=0}^{MU−1} W(s) e^{−2πi (u − (U−1)/2) s/U} E(w + s).

For independent input noise, the output noise is `s_U = G σ_in sqrt(Σ_s W(s)²) / U`. For the
M = 4 tap sinc-Hanning window, this is `0.892 G σ_in / √U`.

- The window's 1/U keeps a tone at a fine-channel centre at unit gain, so noise shrinks with U
  unless G compensates.
- The production gain is `G = 0.8 √U`, which gives `s_U = 0.714 σ_in = 1.99` for every U: a
  model rms of 2.01 with 20% zeros.
- With G = 1 instead, `s_U` would fall from 1.76 at U = 2 to 0.31 at U = 64, which is 89% zeros.

**Baseband beams.** The beams are `J_b = bb_gain · Σ_d exp(iφ_bd) E_d`. `calcBBPhase` realizes
this as a shift `s` and a phase amplitude `|A| = 127.5 a`, with `bb_gain = 2^7 a / 2^s`.

- With `bb_gain = 1/√D`, the noise is `s_B = σ_in · 127.5/128 = 2.77`: a model rms of 2.76,
  with 1.9% saturated.
- The old `bb_scale` of 12 gave `s_B = 0.249 σ_in` for D = 64.

**FRB3.** `cudaQuantize4` splits each beam and fine channel into chunks of 256 FRB2 samples.
Each chunk gets its own offset and scale, both stored as float16:

    offset = mean,   scale = 2 c σ / 15,   q = clamp(round((x − offset)/scale), −7, 7),

where `c = stddev_cutoff = 3`. The mean and standard deviation σ are taken over the chunk's
valid samples. Samples that the PL mask marks invalid are stored as -8.

- A Gaussian chunk therefore has `s = 15/(2c) = 2.5`: a model rms of 2.51, with 0.93% of
  samples at ±7.
- FRB2 intensities are sums of a few squared voltages, so their distribution is skewed. They
  reach +7 more often than −7.
- Layout: FRB3 is `[Ttildehi256, R, Fbar, Ttildelo256/2]` int4x2 (two's complement, earlier
  sample in the low nibble). Offset and scale are `[Ttilde256, R, Fbar, 2]`.

## Checks

**Every int4 output** (each U, BB, FRB3). The check counts the components (real and imaginary
separately) over the frequency slots listed in `coarse_freq`. It measures the rms, the fraction
of zeros, the fraction at ±7 and the fraction of -8, and reports them next to the model.

The output fails if:

- its rms is below 0.5 LSB or above 3.5 LSB,
- more than 5% of its values are at ±7, or
- any value is -8 (this test has no packet loss).

These limits catch only catastrophic loading:

- At rms 0.5 about 75% of the samples are zero.
- At rms 3.5, Gaussian noise puts about 6% of the samples at ±7.
- The MSE-optimal uniform 4-bit quantizer for Gaussian noise has an rms near 3 LSB, well inside
  these limits.

**FRB3 versus FRB2.** The check dequantizes each sample as `x̂ = q·scale + offset`, using the
stored float16 offset and scale.

- **Below saturation** (|q| < 7): `|x̂ − x| ≤ scale/2 + ε`.
- **At saturation** (q = ±7): `±(x − offset) ≥ 6.5 scale − ε`. This also rules out sign flips.
- **Allowance**: `ε = 2^−9 (|x| + |offset| + scale)`. The GPU computes q in float16 as
  `x·(1/scale) + (−offset/scale)`. The rounding of 1/scale, of −offset/scale and of the fused
  multiply-add each contributes up to 2^−11 relative error. Typical offsets are about 9 steps,
  and up to about 100 steps in chunks dominated by the steady tone.
- **No-data values**: q = −8 exactly where the FRB PL mask is 0. Without packet loss, every
  sample must be valid.

## Variants

There is one variant, `noise`, which starts at FPGA sample `2^33 + 7·T`. One frame of each
output is written.

## Mutation check

To show that the check fails when it should, the check code alone was changed and the same
output re-checked with `--check-only`:

- **FRB3 scale × 1.05:** 11.9 M of 226.6 M samples were beyond half a step; the check failed.
- **FRB3 offset + 0.25 scale:** 44.1 M samples were beyond half a step; the check failed.
- **BB decoded as two's complement:** rms 6.92, 92% at ±7, 52% at -8; the check failed.

## Result (pathfinder, cx67, Release)

**PASS**, with the production gains of #1784 (`upchan_gain` = 0.8·√U, `bb_gain` = 1/√D). The
model rms is for noise only; the tone adds about 3%.

| output | rms  | model | zeros | ±7    | -8 |
|--------|------|-------|-------|-------|----|
| U=2    | 2.08 | 2.01  | 19.0% | 0.09% | 0  |
| U=4    | 2.06 | 2.01  | 19.3% | 0.13% | 0  |
| U=8    | 2.08 | 2.01  | 19.2% | 0.17% | 0  |
| U=16   | 2.05 | 2.01  | 19.5% | 0.15% | 0  |
| U=32   | 2.06 | 2.01  | 19.6% | 0.24% | 0  |
| U=64   | 2.07 | 2.01  | 19.7% | 0.41% | 0  |
| BB     | 2.86 | 2.76  | 13.9% | 2.61% | 0  |
| FRB3   | 2.48 | 2.51  | 15.7% | 1.26% | 0  |

- **FRB3:** it encodes FRB2 within `|q − (x − offset)/scale| ≤ 0.567` below saturation. All
  226.6 M samples are valid and pass. There is no -8, and no chunk has scale 0.
- **Before #1784**, with `upchan_gain` 1 and `bb_scale` 12, T3 failed. The upchannelizer rms
  fell as 0.892·σ/√U, from 1.84 LSB at U = 2 to 0.34 LSB (89% zeros) at U = 64. The BB rms was
  0.75 LSB.

Kotekan runs for about 17 s and the check for 10 s. The output is 755 MB on disk.

## Result (CHORD, cx67, Release)

**PASS**, without changes to the check. The upchannelizers match the pathfinder to the digits
shown (rms 2.05 to 2.08, 19.0% to 19.7% zeros). BB has rms 2.78 (model 2.76) with 2.05% at ±7,
FRB3 rms 2.48 (model 2.51) with 1.28% at ±7. FRB3 encodes FRB2 within
`|q − (x − offset)/scale| ≤ 0.676` below saturation; all 470.8 M samples are valid and pass.
Kotekan runs for 27 s; the test takes 2.5 minutes.

## Result (CHIME, cx68, A40, Release)

**PASS**, with the production gains of config/chime_upgrade_frb.j2. Only U = 16 exists, and CHIME
has no BB beamformer.

- **U = 16:** rms 0.70 LSB (model 0.68), 57.1% zeros, no ±7, no -8. The production
  `upchan_gain` is 1, not 0.8·√U as in config/fengine (#1784), so the output is under-loaded as
  the pathfinder was before #1784; with 0.8·√U it would be 2 LSB.
- **FRB3** is quantized to 8 bits (cudaQuantize8): codes 1 to 254 per chunk of 16 frequencies ×
  16 samples, 0 for no data. All 254 codes occur; all 67.1 M samples are valid and within half a
  step plus float16 rounding (`|q − (x − offset)/scale| ≤ 0.932`).

Kotekan runs for 46 s.
