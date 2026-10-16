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

A tone of amplitude `A` from direction `n = (n_x, n_y, n_z)` in the telescope grid frame. In
coarse channel `f` it gives the channelized voltage

    E_d,p(f, t) = A · exp(2πi k (t − t0)/T) · exp(−2πi ν_f n·r_d / c)

- `d` is the dish, `p` the polarization, `t` the FPGA sample and `T` the frame length.
- `k` is the number of tone cycles per frame. The tone sits at `f_tone = k/T` cycles per sample
  above the coarse channel centre, at frequency `ν_f = (channel + k/T) Δν`.
- `r_d` is the dish position from `telescope.dish_inputs`.

This is the convention of RadioTelescopeFEngine.jl. It is also the one for which the
`calcBBPhase` phases `A_bd = exp(+2πi ν n_b·r_d/c)` form a beam `J_b = Σ_d A_bd E_d` that peaks
at `n_b = n`. The test does not decide whether this is the physical sky direction or its mirror,
which depends on the F-engine. It checks that all subsystems agree with this one convention.

The tone is at `f_tone = 19.5/64` cycles per sample. That is a fine-channel centre for U = 64.
For every U the nearest fine channel and its mirror are distinct.

## Checks

**Input.** The written voltages equal the model rounded to int4. **Tolerance: 0.5 LSB.**

**Baseband beams.** The predicted beam response is

    R_b(ν) = | Σ_d exp(2πi ν (n_b − n)·r_d / c) |² ,

using the beam positions `n_b` that kotekan actually wrote.

- The measured beam power (summed over time and polarization) must peak in the same beam as
  `R_b`.
- This applies to every frequency where the prediction is unambiguous (best beam ≥ 1.25 × second
  best) and the strongest beam has rms ≥ 2 LSB. Below that, int4 rounding noise (±1 LSB in a few
  percent of samples) decides the argmax.
- **Tolerance: exact beam index.**

**Upchannelizer**, for each U. The kernel computes

    Ē_u(t̄) = (G/U) Σ_{s=0}^{MU−1} W(s) exp(−2πi (u − (U−1)/2) s/U) E(w + s)

- The window is the sinc-Hanning `W(s) = cos²(π x) sinc(M x)`, with `x = (2s − (MU−1))/(2MU)`.
- `w = L − (M−1)U/2` is the window start for the output labelled `L`.
- For the model tone, this is `A g_d exp(2πi f_tone (w − t0)) · R_u`, with
  `R_u = (1/U) Σ_s W(s) exp(2πi (f_tone − (u − (U−1)/2)/U) s)`.

Checks and tolerances:

- **Peak fine channel:** the tone must be in `u* = round(f_tone U + (U−1)/2)`. The mirror
  channel would indicate a conjugation or FFT sign error.
- **Phase:** the measured Ē at `u*`, compared with the prediction, must agree within a mean of
  2° and a 99th percentile of 15°. Each output sample is rounded to int4 at an amplitude of
  about 5, which allows up to about 6° per sample.
- **Amplitude:** the median |measured/predicted| must lie in [0.8, 1.25]; `G` = 1.

**FRB1.** FRB1 is a spatial FFT over the dish grid, with 2M × 2N beams. The FRB2 weights
(`cudaCalcFRB2WeightsKernel.cu`, after Kendrick's note) assume that beam `(p, q)` responds as

    U(p, θ_M) U(q, θ_N),   peaking at p = 2θ_M mod 2M and q = 2θ_N mod 2N,

with `θ_M = M (n·σ_M) ν/c` and `σ_M` the grid spacing along M (likewise for N).

- For every coarse channel, the FRB1 peak must be within one beam of `(2θ_M, 2θ_N)`.
- On failure, the check reports whether p or q matches the mirror `−2θ` instead.

**FRB2.** For the `frb2` variant, the FRB2 beam whose configured position equals the source
must hold the maximum power in every channel that carries the tone.

- The dish spacing is many wavelengths, so grating lobes can give other beams exactly the same
  power. Such ties are allowed.
- **Tolerance:** the source beam must have ≥ 99.5% of the peak power, which allows for float16
  rounding.

**N2** (variant `n2`). A noise-free tone has spectral kurtosis 0 and is rightly flagged as RFI,
so this variant uses a weaker tone (`A` = 2) in complex Gaussian noise of 2.8 LSB per
component. The visibilities must satisfy

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
| `n2`       | as `offgrid`, with noise             | visibilities               |

All variants start at FPGA sample `2^33 + 7·T`.

## Result (pathfinder, cx67, Release)

**All four variants pass**, with the FRB beamformer fix of kotekan/kotekan#1779:

- **Input, BB and all upchannelizers:** upchannelizer phase within 0.3° on average, amplitude
  within 10%.
- **FRB1:** peak at the predicted (p, q) in all 384 channels.
- **FRB2:** the source beam holds the peak in all 384 channels with signal, 41 of them tied with
  grating lobes.
- **N2:** `V_ab = E_a conj(E_b)` within 0.00° (99th percentile 3.4°), amplitude 0.96.

Without that fix, T2 found FRB1's beam index along M (grid x) mirrored, `p ≈ −2θ_M`, while
`q ≈ 2θ_N` was correct. FRB2 inherited the mirror: a source on FRB2 beam (12, 5) peaked in beam
(3, 5). The cause was the sign of the second (M) FFT in `julia/kernels/frb.jl`.
