# T3: Quantization

T3 checks that the 4-bit outputs use their range sensibly. For noise input, the upchannelized
voltages, the baseband (BB) beams and the FRB3 beams must not be mostly zero and must not be
mostly saturated. FRB3 must encode FRB2 within half a quantization step.

## Errors this test catches

- **Wrong normalization**: a missing or extra factor of U or √U in the upchannelizer, a wrong
  window normalization, a wrong `upchan_gain`, `bb_scale` or BB shift. The output is then
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
M = 4 tap sinc-Hanning window this is `0.892 G σ_in / √U`. All configs use G = 1:

| U  | s_U  | model rms | model zeros |
|----|------|-----------|-------------|
| 2  | 1.76 | 1.78      | 22%         |
| 4  | 1.24 | 1.28      | 31%         |
| 8  | 0.88 | 0.92      | 43%         |
| 16 | 0.62 | 0.68      | 58%         |
| 32 | 0.44 | 0.51      | 75%         |
| 64 | 0.31 | 0.33      | 89%         |

**Baseband beams.** The beams are `J_b = Σ_d A_bd E_d / 2^s`, with |A_bd| = 127.5 and
s = `bb_scale` = 12. With D = 64 dishes, `s_B = σ_in √D · 127.5 / 2^12 = 0.249 σ_in = 0.69`,
which gives a model rms of 0.75.

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

**FAIL**, because U = 64 is under-loaded. The model rms is for noise only. The tone adds about
3% to the rms; without the tone's fine channels, the measured rms is within 0.5% of the model.

| output | rms  | model | zeros | ±7    | -8 |
|--------|------|-------|-------|-------|----|
| U=2    | 1.84 | 1.78  | 21.4% | 0.01% | 0  |
| U=4    | 1.31 | 1.28  | 30.4% | 0     | 0  |
| U=8    | 0.96 | 0.92  | 41.8% | 0     | 0  |
| U=16   | 0.70 | 0.68  | 57.1% | 0     | 0  |
| U=32   | 0.52 | 0.51  | 73.6% | 0     | 0  |
| U=64   | 0.34 | 0.33  | 88.5% | 0     | 0  |
| BB     | 0.78 | 0.75  | 51.6% | 0     | 0  |
| FRB3   | 2.48 | 2.51  | 15.7% | 1.27% | 0  |

- **Upchannelizer:** `julia/kernels/upchan.jl` scales the window by 1/U, and every config uses
  `upchan_gain` 1. The noise rms therefore falls as 1/√U: U = 64 is below the limit, and
  U = 32 is just above it. A gain of about 1.12 √U would keep the rms near σ_in.
- **Baseband beams:** they pass, but are under-loaded by two bits. The BB beams away from the
  source have rms 0.75; the beam nearest the source has 0.99.
  - The formula `log2(√D) + 7` in the `bb_scale` comment gives 10 for the pathfinder's
    64 dishes, but the config uses 12.
  - With `bb_scale` 10 the rms would be about σ_in.
- **FRB3:** it encodes FRB2 within `|q − (x − offset)/scale| ≤ 0.570` below saturation. That
  uses at most 43% of ε, and all 226.6 M samples pass. The +7 values make up 1.20% and the −7
  values 0.07%. There is no -8, and no chunk has scale 0.

Kotekan ran for 19 s and the check for 10 s. The output is 755 MB on disk (FRB2 340 MB).
