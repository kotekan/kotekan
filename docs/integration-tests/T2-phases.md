# T2: Phases

T2 checks that every subsystem treats the phase of a sky signal the same way. A point source
emits a tone. N2, the upchannelizers, the baseband beamformer and the FRB beamformers have to
show the visibility phases, fine-channel phases and beam positions that this source predicts.

## Errors this test catches

- **Complex conjugation.** A conjugated voltage reverses every phase gradient. Beams then point
  at the mirror image of the source, visibilities come out conjugated, and the tone appears in
  the mirror fine channel.
- **Sign flips along one axis.** For example, an FFT along the grid's x direction with the
  opposite sign convention from the y direction. The source then appears mirrored in x only.
- **Swapped axes** (x ↔ y, M ↔ N), and wrong dish spacing or dish positions in a beamformer.
- **Wrong fine-channel order or frequency** in the upchannelizer: reversed fine channels, the
  wrong fine-channel centre, or the wrong `freq_upchan_index`.
- **Wrong time phase** in the upchannelizer. An error in its time origin or window centre shows
  up as a phase offset that depends on U.
- **Inconsistent conventions between subsystems**, e.g. BB and FRB disagreeing about where a
  source at +x is.

## Signal model

A tone of amplitude `A` from direction `n = (n_x, n_y, n_z)` in the telescope grid frame, in
complex Gaussian noise of σ = 2.8 LSB per component. In coarse channel `f` it gives the
channelized voltage

    E_d,p(f, t) = A · exp(2πi k (t − t0)/T) · exp(−2πi ν_f n·r_d / c) + noise

- `d` is the dish, `p` the polarization, `t` the FPGA sample and `T` the frame length.
- `k` is the number of tone cycles per frame. The tone sits at `f_tone = k/T` cycles per sample
  above the coarse channel centre, at frequency `ν_f = (channel + k/T) Δν`.
- `r_d` is the dish position from `telescope.dish_inputs`.

This is the convention of RadioTelescopeFEngine.jl. It is also the one for which the
`calcBBPhase` phases `A_bd = exp(+2πi ν n_b·r_d/c)` form a beam `J_b = Σ_d A_bd E_d` that peaks
at `n_b = n`. The test does not decide whether this is the physical sky direction or its mirror,
which depends on the F-engine. It checks that all subsystems agree with this one convention.

The tone is at `f_tone = 19.5/64` cycles per sample. That is a fine-channel centre for U = 64,
and for every U the nearest fine channel and its mirror are distinct.

**Choice of amplitude.** With the production gains, the upchannelizer multiplies a tone at a
fine-channel centre by 0.81·√U (6.5 at U = 64), and BB forms `A·√D·bb_gain·√D = A·D/√D` on the
source beam. `A = 0.75` keeps the upchannelizer outputs below saturation, and the noise dithers
the int4 rounding, so phases and amplitudes come out unbiased. The BB source beam, `A·√D`, stays
below 7 only for the pathfinder (6); it saturates for CHORD (17) and CHIME (24), which the BB peak
check tolerates.

**Coherent estimates.** All checks estimate them by averaging the output against the
predicted tone over time. With about 256 (U = 64) to 8192 (U = 2) samples per average, the noise
contributes about 1.5° per input.

## Checks

**Input.** The time average of `E · conj(model/A)` per input must be 1:

- **Phase:** mean ≤ 1°, 99th percentile ≤ 30°.
- **Amplitude:** within 5%.

**Baseband beams.** The predicted beam response is

    R_b(ν) = | Σ_d exp(2πi ν (n_b − n)·r_d / c) |² ,

using the beam positions `n_b` that kotekan actually wrote.

- The predicted tone power in beam b is `(A · bb_gain · |Σ|)²`, where `Σ` is the sum inside
  `R_b`. It sits on a noise floor that is the same for all beams.
- The measured beam power (averaged over time and polarization) must peak in the same beam as
  the prediction.
- This applies to every frequency where the prediction is unambiguous: the best beam has
  ≥ 1.25 × the second best and at least 1 LSB² more. The power estimates average about 30,000
  samples, so their noise is about 0.1 LSB².
- **Tolerance: exact beam index.**

**Upchannelizer**, for each U. The kernel computes

    Ē_u(t̄) = (G/U) Σ_{s=0}^{MU−1} W(s) exp(−2πi (u − (U−1)/2) s/U) E(w + s)

- The window is the sinc-Hanning `W(s) = cos²(π x) sinc(M x)`, with `x = (2s − (MU−1))/(2MU)`.
- `w = L − (M−1)U/2` is the window start for the output labelled `L`.
- For the model tone, this is `G_u A g_d exp(2πi f_tone (w − t0)) · R_u`, with
  `R_u = (1/U) Σ_s W(s) exp(2πi (f_tone − (u − (U−1)/2)/U) s)`.
- `G_u` is the configured `upchan_gain`.

Checks and tolerances:

- **Peak fine channel:** the coherent amplitude `|Σ_t Ē_u conj(model)|` must peak at
  `u* = round(f_tone U + (U−1)/2)`. The mirror channel would indicate a conjugation or FFT sign
  error. Unlike the power, the coherent amplitude is insensitive to the noise floor's tilt across
  fine channels.
- **Phase:** at `u*`, the time-averaged ratio of measured to predicted must have phase within a
  mean of 2° and a 99th percentile (over inputs) of 10°.
- **Amplitude:** its median modulus must lie in [0.8, 1.25].

**FRB1.** FRB1 is a spatial FFT over the dish grid, with 2M × 2N beams. The FRB2 weights
(`cudaCalcFRB2WeightsKernel.cu`, after Kendrick's note) assume that beam `(p, q)` responds as

    U(p, θ_M) U(q, θ_N),   peaking at p = 2θ_M mod 2M and q = 2θ_N mod 2N,

with `θ_M = M (n·σ_M) ν/c` and `σ_M` the grid spacing along M (likewise for N).

- For every coarse channel, the FRB1 peak must be within one beam of `(2θ_M, 2θ_N)`.
- On failure, the check reports whether p or q matches the mirror `−2θ` instead.

- In each coarse channel, the tone's fine channel is the one with the strongest beam.

**FRB2.** For the `frb2` variant, the FRB2 beam whose configured position equals the source
must hold the maximum tone power in every coarse channel. Tone power is the beam power minus the
noise floor, taken as the median over beams.

- The dish spacing is many wavelengths, so grating lobes can give other beams exactly the same
  tone power. Such ties are allowed.
- **Tolerance:** the source beam must have ≥ 98% of the peak tone power, which allows for noise
  and float16 rounding.

**N2** (variant `n2`). A tone that dominates the noise has a spectral kurtosis well below 1
and is rightly flagged as RFI; a noise-free tone has spectral kurtosis 0. This variant therefore
uses a weaker tone in the same noise, which passes the RFI flagging. A tone in noise of rms
`s` has `SK = 1 − (A²/(A² + 2s²))²`, and first-stage excision flags a block of M = 256 samples
when the feed-averaged SK is more than 5 standard deviations, about `5 · 2/√(M N)` for N
inputs, from 1. `A` puts SK 1.2 standard deviations inside that threshold: `A` = 2 for the
pathfinder (N = 128) and 1.47 for CHORD (N = 1024). With `A` = 2 on CHORD, 95% of the samples
were flagged. The visibilities must satisfy

    V_ab = ⟨E_a conj(E_b)⟩ = A² g_a conj(g_b) ,   for a ≠ b

with `g` the geometric phases above and element `a = p·D + d`.

- **Phase:** mean ≤ 2°, 99th percentile ≤ 10°.
- **Amplitude:** `|V_ab| / A²` in [0.9, 1.1].

Fringe stopping is turned off for this variant (`do_fringestop: false`). Earth rotation would
otherwise add a few degrees on long baselines within one integration; fringe stopping belongs to
Level 2.

## Variants

| variant    | source direction                     | main purpose               |
|------------|--------------------------------------|----------------------------|
| `bb`       | BB beam (3, 1) of the 4 × 4 grid     | BB peak, upchannelizer     |
| `frb2`     | FRB2 beam (12, 5) of the 16 × 16 grid | FRB1, FRB2                |
| `offgrid`  | θ = (0.5°, 0.8°)                     | no symmetry to hide errors |
| `n2`       | as `offgrid`, `A` = 2 (pathfinder)   | visibilities               |

All variants start at FPGA sample `2^33 + 7·T`.

## Result (pathfinder, cx67, Release)

All four variants pass, with the production gains of #1784 and the FRB beamformer fix of #1779.

| output | result |
|---|---|
| input | tone phase within 0.13° (99th percentile 12°), amplitude 0.99 |
| BB | peak in the predicted beam at all 214–311 unambiguous frequencies |
| upchannelizers, all U | predicted fine channel; phase within 0.1° (99th percentile ≤ 4.8°); amplitude 0.97–0.99 |
| FRB1 | peak at the predicted (p, q) in all 384 channels |
| FRB2 | source beam holds the peak in all 384 channels, 79 of them tied with grating lobes |
| N2 | `V_ab = E_a conj(E_b)` within 0.00° (99th percentile 3.4°), amplitude 0.96 |

**Before #1779**, T2 found FRB1's beam index along M (grid x) mirrored, `p ≈ −2θ_M`, while
`q ≈ 2θ_N` was correct. FRB2 inherited the mirror: a source on FRB2 beam (12, 5) peaked in beam
(3, 5). The cause was the sign of the second (M) FFT in `julia/kernels/frb.jl`.

## Result (CHORD, cx67, Release)

All four variants pass.

| output | result |
|---|---|
| input | tone phase within 0.08° (99th percentile 12.5°), amplitude 0.99 |
| BB | peak in the predicted beam at all 27–37 unambiguous frequencies |
| upchannelizers, all U | predicted fine channel; phase within 0.0° (99th percentile ≤ 4.9°); amplitude 0.97–0.99 |
| FRB1 | peak at the predicted (p, q) in all 48 channels |
| FRB2 | source beam holds the peak in all 48 channels, 23 of them tied with grating lobes |
| N2 | `V_ab = E_a conj(E_b)` within 0.00° (99th percentile 4.8°), amplitude 0.97, 99.5% of the samples valid (pathfinder: 3.3°, 71%) |

Kotekan runs for 20 to 37 s per variant; the test takes about 10 minutes.

## Result (CHIME, cx68, A40, Release)

The `frb2`, `offgrid` and `n2` variants pass; `bb` does not apply (no BB beamformer). The tone is
on a U = 16 fine-channel centre (4.5/16 cycles per sample), and the FRB2 source beam is the
production beam 23 (`FRB2_BEAM` clamped to the 4 × 256 grid, at 55.6° south).

| output | result |
|---|---|
| input | tone phase within 0.04° (99th percentile 12.6°), amplitude 0.99 |
| U = 16 | fine channel 12; phase within 0.0° (99th percentile 3.3°); amplitude 0.99 |
| FRB1 | peak at the predicted (p, q) in all 16 channels (beam axes swapped, P along the cylinders) |
| FRB2 | source beam holds the peak in all 16 channels, 1 of them tied with a grating lobe |
| N2 | `V_ab = E_a conj(E_b)` within 0.00° (99th percentile 4.1°), amplitude 0.97, A = 1.33, 99.1% of the samples valid |

Kotekan runs for 58, 60 and 84 s.
