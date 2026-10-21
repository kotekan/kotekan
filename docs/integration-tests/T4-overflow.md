# T4: Overflow

T4 checks that full-scale coherent input saturates every output cleanly. Integer outputs must
clip at ±7 with the right sign and never wrap around or produce -8. Float outputs must stay
finite and below their bounds. Where the arithmetic is integer, the output must equal the
prediction exactly.

## Errors this test catches

- **Wraparound** in the int4 packing of the upchannelizer and BB outputs: a value of 9 stored
  as -7, or clamping to [-8, 7] so that -8 (no data) appears for valid data.
- **Saturation with the wrong sign**, e.g. from clamping before negation, or from overflow in an
  intermediate integer or float16 sum.
- **Correlator errors at full scale:** int32 overflow, the -7 corner of the int4 range, the
  conjugation convention and the tile layout.
- **Wrong gains, shifts or rounding** in the upchannelizer and BB. Since the input is known
  exactly, every output value away from a rounding tie is predicted.
- **FRB float16 overflow** to Inf in FRB1 or FRB2, and NaN in the FRB3 offset and scale, e.g.
  for chunks of zero variance.

## Inputs

There is no noise, and the input repeats every 128 samples. First-stage RFI excision is
switched off (`rfi_excision_enabled: false`). A signal of constant power has spectral kurtosis
0, so the SK stage would flag every sample, and the correlator, which skips flagged samples,
would sum nothing. A probe run of `full_scale` with excision on confirms this: every
`n2k_correlation` value is 0, and each N2 integration has `n_valid_fpga_ticks` 0,
`n_rfi_fpga_ticks` 196608 (all 24 · 8192 samples) and zero visibilities.

- **`full_scale`:** `E = 7 + 7i` on every input. This is a source at zenith at the centre of
  every coarse channel. `testSkySignal` takes an optional source phase for this: amplitude
  7√2 at phase π/4 rounds to exactly 7 + 7i.
- **`full_scale_tone`:** an amplitude-7 tone from BB beam (3, 1), at 1/128 cycles per sample.
  That is a fine-channel centre for U = 64. One voltage frame is written, and the check takes
  the exact int4 input from it.

## Predictions

**Correlator** (`n2k_correlation`, before RFI and accumulation). For stations `i = p D + d`,

    V_ij = Σ_t E_i(t) conj(E_j(t)),   t over one sub-integration (8192 samples).

The data are stored as 16×16 tiles of the lower triangle, `[Tc, F, tile, ilo, jlo, re/im]`,
with `tile = ihi (ihi+1)/2 + jhi` (`external/n2k/include/n2k/Correlator.hpp`). For full scale,
every V_ij is 98 · 8192 = 802816. The int32 range would last for 2^31 / 98 = 21.9 M samples
per sub-integration.

**Upchannelizer.** Its output is

    Ē_u(L) = (G_u/U) Σ_{s=0}^{MU−1} W(s) e^{−2πi (u − (U−1)/2) s/U} E(L − (M−1)U/2 + s),

with M = 4 taps and the gain `G_u = 0.8·√U` rounded to float16. For full scale, the DC signal
falls between fine channels U/2 − 1 and U/2. Per component it reaches 5.6 (U = 2) to 23 (U = 64).
The tone reaches 4.7 to 46. The kernel computes in float16. Its output differs from the
float64 prediction by at most 0.004 LSB, so the check uses a margin of 0.02 LSB:

- where the predicted component exceeds +6.52 the output must be +7, and below −6.52 it must
  be −7;
- where |prediction| > 0.52 the output must have the predicted sign;
- away from rounding ties (more than 0.02 from k + ½) the output must be the prediction
  rounded and clamped to [−7, 7];
- there is no -8.

**BB.** The beamformer is integer arithmetic (`julia/kernels/bb.jl`). With the written phases
`A` (int8) and shifts `s`,

    J = clamp(shift(Σ_w shift(Σ_{d∈w} A_bd E_d, σ), s − σ), −7, 7),
    shift(x, n) = (x + 2^(n−1)) >> n,

per component. Each warp sums a block w of 2^(σ+4) consecutive dishes, shifts the sum by σ and
packs it into int16; the blocks are then added in int32. σ = 2 for D = 64 (one block) and σ = 3
for D = 512 (four blocks of 128). Shifting the whole sum once instead differs at rounding
boundaries: 0.1% of the CHORD components were off by one. For the pathfinder, `bb_gain = 1/8`
gives s = 10 and |A| = 127.5. The output must equal J exactly. On the source beam the unclamped
value reaches 64 · 127.5 · 7 / 2^10 ≈ 56. The intermediate `shift(Σ, σ)` reaches 14649 (CHORD:
13435 per block), inside the int16 range.

**FRB1.** The kernel (`julia/kernels/frb.jl`) forms `Ẽ_p = input_gain Σ_d W_pd Ē_pd` times
FFT phases of modulus 1, with `input_gain = 1/(2√(MN))`, and accumulates

    I = output_gain Σ_{t<Tds} Σ_p |Ẽ_p|²,   output_gain = 2/(P Tds).

Every int4 input has |Ē| ≤ 7√2, so |Ẽ_p| ≤ input_gain · 7√2 · Σ_d |W_pd|, and

    I ≤ (2/P) · 98/(4MN) · Σ_p (Σ_d |W_pd|)².

This is `lib/stages/frb1IntensityBound.hpp`, summed over the D dishes that carry voltage
(empty grid cells get none). With unit weights, D = 64, M·N = 8·12 and P = 2, the bound is
98 · 2 · 64² / 384 = 2090.7. A coherent source at the beam centre with saturated input reaches
it exactly, so `full_scale` tests the bound itself. FRB1 must be finite and at most 1% above the
bound, the float16 allowance of `frb1_intensity_limit`.

**FRB2 and FRB3.** FRB2 and the FRB3 offset and scale must be finite, and FRB3 (two's
complement int4) must contain no -8, since there is no packet loss.

## Variants and data

Both variants start at FPGA sample 2^33 + 7·T and write one frame of each output: the
upchannelized voltages (all U), `n2k_correlation`, BB beams, phases and shifts, FRB1, FRB2,
FRB3 with offset and scale, and for the tone the input voltage.

## Mutation check

The check code alone was changed and the same output re-checked with `--check-only`. Each
change failed:

- **Correlator conjugation swapped** (`conj(E_i) E_j`): 3.45 M of 7.08 M values differ
  (`full_scale_tone`).
- **Upchannelizer decoded as two's complement:** sign flips, -8 and wrong values for every U.
- **BB phases conjugated:** 175 M of 201 M components differ, 44 M with the wrong sign.
- **FRB1 bound for |E| ≤ 7 instead of 7√2:** the measured maximum is 2.01 times the bound.

## Result (pathfinder, cx67, Release)

**PASS**, with the production gains.

| output         | `full_scale`                          | `full_scale_tone`                 |
|----------------|---------------------------------------|-----------------------------------|
| n2k            | all 7.08 M values exact, max 802816   | all exact, max 413184             |
| upchan, ±7     | 0% (U=2), 25% (U=4) … 3.1% (U=64)     | 0% (U=2) … 6.8% (U=8) … 1.4% (U=64) |
| upchan errors  | none (saturation, sign, rounding, -8) | none                              |
| BB             | exact; 2.3% clamped, max 14           | exact; 14% clamped, max 57        |
| FRB1 max I     | 2104 (1.0064 × bound)                 | 1706 (0.816 × bound)              |
| FRB2 max       | 2080                                  | 1409                              |
| FRB3           | no -8; every chunk has scale 0        | no -8; every chunk has scale 0    |

- **FRB1:** at U ≥ 16 the saturated input gives 2090, the bound to float16 precision. At U = 1
  the kernel accumulates Tds = 192 float16 terms, and rounding raises a constant intensity by
  0.64%, within the 1% allowance.
- **FRB3:** both inputs have constant power, so every chunk has zero variance. The quantizer
  then stores scale 0 and q = 0, and the offset holds the value. FRB3 saturation is covered by
  T3.

Kotekan runs for 18 s (`full_scale`) and 33 s (`full_scale_tone`), the check for 16 s each.
The output is 52 MB and 440 MB on disk.

## Result (CHORD, cx67, Release)

**PASS.** All 51.1 M correlator values are exact (max 802816 and 413184). The upchannelizers
saturate as on the pathfinder, without errors. BB is exact (1.7% and 5.3% clamped, max 111 and
159). The FRB1 maximum is 22272 (0.9987 × the bound 22300 for 512 dishes) and 17584; FRB2
reaches 20496 and 16512; FRB3 has no -8. Kotekan runs for 24 s and 36 s; the test takes
2.5 minutes.

## Result (CHIME, cx68, A40, Release)

**PASS**, with the production gains. Only U = 16 exists, and CHIME has no BB beamformer; the
`full_scale_tone` source is T2's FRB2 source beam.

| output | `full_scale` | `full_scale_tone` |
|---|---|---|
| n2k | all 67.6 M values exact, max 1605632 | all exact, max 826368 |
| U = 16 | exact, max \|prediction\| 3.8, nothing at ±7 | exact, max 5.3 |
| FRB1 max I | 25632 (0.255 × bound) | 26912 (0.268 × bound) |
| FRB2 max | 25632 | 27440 |
| FRB3 (8 bits) | no 0 code; 94% at 1 or 254 | no 0 code; 93% at 1 or 254 |

With `upchan_gain` 1 the upchannelizer cannot saturate, so the saturation checks do not apply on
CHIME. The FRB1 bound uses the default feed gain 1 + 1i of loadFeedGains.

Kotekan runs for 46 s and 56 s.
