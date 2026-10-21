# T6: Dish locations

T6 checks that every subsystem puts each dish where `telescope.dish_inputs` says: N2 at the
right element index, BB with the phase of the dish's position, FRB1 at the dish's grid cell. It
also checks a half-filled array.

## Errors this test catches

- **Wrong element order** (e.g. dish-major `d·P + p` instead of `p·D + d`), wrong product
  indexing in `N2Accumulate`, or polarizations swapped.
- **A dish at the wrong position** in a beamformer: x and y swapped, a grid index mirrored or
  shifted, the wrong separation, or a dish permutation between `calcBBPhase` and the kernel.
- **Wrong FRB1 dish layout**: a dish in the wrong cell, M and N swapped, one axis mirrored.
- **Missing dishes shifting the others**, e.g. when Missing dishes fill the empty FRB1 cells.
- **Wrong telescope metadata** in the written files.

## Conventions

- Element `a = p·D + d` (`CHORDBeamformer` order). N2 stores the upper triangle `a ≤ b`
  row-major; with `N = 2D = 128` elements, product `(a, b)` is at `a·N − a(a−1)/2 + (b − a)`.
- `r_d = (σ_x x_d + δ_x, σ_y y_d + δ_y, δ_z)` from the grid indices `(x_d, y_d)`, the dish
  separations `σ = (6.3, 8.5)` m and `feed_pos_disp_m` δ. North is +y: the grid axes are given
  in the topocentric frame (east, north, up; `geoUtil.hpp`), `grid_y_axis` is (0, 1, 0), and
  `grid_y_idx` 0 is the southmost row (`CHORDTelescope.hpp`).
- Each written dataset carries `dish_grid_indices` [D, 2] (`(x_d, y_d)` for an `ArrayDish`,
  (−1, −1) otherwise) and `feed_positions_m` [D, 3] (`r_d`, also for Missing dishes).

## Variants

**`onehot`.** Eight inputs (d, p) carry a tone in turn: the corners of the grid (the first and
last dish of row 0, the last dish of the last complete row, and the last dish) and the inner
cells (3, 4), (5, 2), (1, 6), (4, 7). For the pathfinder these are (0, 0), (6, 1), (63, 0),
(62, 1), (31, 0), (19, 1), (43, 1), (53, 0), with the corners (0, 0), (6, 0), (6, 8) and the
single dish at (0, 9). For CHORD the corners are dishes 0, 23, 503 and 511 (cell (7, 21)). Each
tone (amplitude 6.1, 13 cycles) lasts for its own
window of T/8 = 1024 samples, all in the first frame of the first complete N2 bin. At any
time at most one input is non-zero. There is no noise, first-stage RFI excision is off (a tone
has spectral kurtosis 0), and so is fringe stopping (see the result).

- **N2:** `V_ab = Σ_t E_a conj(E_b) / n_valid`. Only the eight autocorrelations `(a, a)` may
  be non-zero, and they must equal `Σ|E_a|² / n_valid` from the written input. All other
  products must be exactly 0. The `dish_grid_indices` and `feed_positions_m` must equal the
  config.
- **BB:** `J_b = bb_gain Σ_d A_bd E_d` with `A_bd ∝ exp(+2πi ν_f n_b·r_d / c)`, where `ν_f` is
  the coarse channel centre and `n_b` the written beam position. In the window of (d, p),
  `R_bf = Σ_t J_b conj(E_d) / Σ_t |E_d|²` estimates `bb_gain · exp(2πi ν_f n_b·r_d / c)`.
  The check computes this phase from the config independently, for 8 inputs × 16 beams ×
  384 frequencies (pathfinder). The beams of the other polarization must be exactly 0.
- `bb_gain` is 0.6. Int4 rounding sets the error floor. With a gain of 1 or 1/2, a small
  rotation of a small lattice point rounds back onto the same point, and the phase can be off by
  5° to 10°. At 0.6, simulations of the integer beamformer give at most 2.2°, and |J| stays ≤ 4.

**`pairs`.** Six pairs of grid cells carry equal tones from the zenith; pair j uses
(j − 2.5)/8 + 1/128 cycles per sample, a U = 64 fine-channel centre. FRB1 places dish d in
cell (m, n) = (x_d, y_d) of the M × N grid (8 × 12 for the pathfinder, 24 × 24 for CHORD) and
forms 2M × 2N beams. Beam p peaks at
`p = 2θ_M` (T2), so it weights cell m by `exp(+iπ p m / M)`. For two equal signals,

    I(p, q) ∝ |e^{iπ(p m₁/M + q n₁/N)} + e^{iπ(p m₂/M + q n₂/N)}|² ∝ 1 + cos π(p Δm/M + q Δn/N).

The fringe has period 2M/Δm beams along p and 2N/Δn along q, and its orientation follows the
sign of Δm·Δn. It peaks where `p Δm N + q Δn M ≡ 0 (mod 2MN)`.

| pair (cells) | dishes (pathfinder) | dishes (CHORD) | (Δm, Δn) | peak bins (pathfinder, CHORD) |
|---|---|---|---|---|
| (2, 3), (3, 3) | 23, 24 | 74, 75 | (1, 0) | 24, 240 |
| (5, 1), (5, 2) | 12, 19 | 29, 53 | (0, 1) | 48, 240 |
| (0, 5), (2, 8) | 35, 58 | 120, 194 | (2, 3) | 48, 240 |
| (1, 7), (3, 4) | 50, 31 | 169, 99 | (2, −3) | 48, 240 |
| (0, 2), (6, 2) | 14, 20 | 48, 54 | (6, 0) | 48, 288 |
| (0, 0), (0, 9) | 0, 63 | 0, 216 | (0, 9) | 48, 144 |

A zenith source makes the fringe independent of frequency. A one-axis mirror only flips the sign
of one of Δm, Δn, so only the diagonal pairs detect it. The two diagonal pairs differ in that
sign. In the fine channel of each pair's tone, the check compares `I / max I` with
`(1 + cos)/2` over all beams. The peak bins must be exactly the predicted ones: those above the
middle of the highest gap of at least 0.02 between predicted values. On a fine beam grid the
values next to the peak are close to it (0.9957 for CHORD's 48 × 48 beams), closer than the
measurement error of 0.003, so the peak bins include them. The fine-channel gains are 1, so the
upchannelized tones (about 5 LSB) do not saturate. The tones are 1/8 cycles per sample apart,
so only U = 16, 32 and 64 resolve them: 86 coarse channels for the pathfinder (11 for CHORD),
which cover three FRB1 kernels.

**`half`.** With `half_filled: true`, dishes 32 to 63 have `type: Missing`: row y = 4 from
x = 4 on, and rows 5 to 9. This is the northern half of an array filled from the south.
`bf_mode: auto` also marks them bad. The generator gives Missing dishes no signal, and
`kit.dish_positions` leaves them out of every prediction. The input and checks are those of
T2's `frb2` variant: a tone in noise from FRB2 beam (12, 5). BB must peak in the beam predicted
from the 32 present dishes, every upchannelizer must show the predicted fine channel and phase,
FRB1 must peak at the predicted (p, q), and FRB2 at the source beam. The written
`dish_grid_indices` must be (−1, −1) for the Missing dishes.

## Tolerances

- N2: exact zeros; autocorrelations within float32 rounding (1e−5).
- BB: phase residual ≤ 5° (rounding floor 2.2°, see above); |R| / bb_gain in [0.9, 1.1].
- FRB1 fringe: peak bins exact; |I / max I − fringe| ≤ 0.01. Float16 and tones leaking from
  other pairs give at most 0.003. A wrong separation gives errors of order 1.
- `half`: the T2 tolerances.

## Mutation check

Only the expectation was changed in Python, and the same output re-checked with `--check-only`:

- **x and y of the dish positions swapped** in `kit.dish_positions`: BB fails, residual up to
  180° (99th percentile 178°).
- **Grid x index mirrored** (x → 6 − x): the N2 `dish_grid_indices` check fails, and so do the
  FRB1 fringes of the diagonal pairs (35, 58) and (50, 31), with deviations up to 1.0. The
  axis-aligned pairs still pass, as expected.

## Result (pathfinder, cx67, Release)

**PASS**, all three variants.

| variant | result |
|---|---|
| `onehot` N2 | in all 384 frequencies only the 8 autocorrelations are non-zero, equal to Σ\|E\|²/n within 2.5e−8; the other 8248 products are 0; grid indices and positions match |
| `onehot` BB | phase residual ≤ 2.16° (99th percentile 1.68°); \|R\| / bb_gain 0.951 to 1.054; other polarization 0 |
| `pairs` FRB1 | all 6 pairs, 86 channels each: peak bins exact, \|I/max − fringe\| ≤ 0.0028 |
| `half` | BB peak in the predicted beam at all 189 unambiguous frequencies; upchannelizers within 0.0° (99th percentile ≤ 4.9°); FRB1 peak at (p, q) in all 384 channels; FRB2 source beam in all 384 channels (104 ties) |

**Fringe-stopped N2 visibilities are truncated** (found here). With fringe stopping on, some
autocorrelations came out one count low (38431 instead of 38432). `N2Accumulate` adds the
complex float result of the fringe-stop multiplication into its `int32` accumulator, which
truncates towards zero (`lib/stages/N2Accumulate.cpp:989`). This loses up to one count per
component per sub-integration. That is negligible for noise-dominated visibilities, but it is a
bias, not rounding. The `onehot` variant therefore switches fringe stopping off.

Kotekan runs for 14 s (`onehot`), 18 s (`pairs`) and 55 s (`half`); with the checks the test
takes 2 minutes. The output is 25 MB, 64 MB and 1.5 GB on disk.

## Result (CHORD, cx67, Release)

**PASS**, all three variants.

| variant | result |
|---|---|
| `onehot` N2 | in all 48 frequencies only the 8 autocorrelations are non-zero, equal to Σ\|E\|²/n within 2.5e−8; the other 524792 products are 0; grid indices and positions match |
| `onehot` BB | phase residual ≤ 1.97° (99th percentile 1.52°) over 96 beams; \|R\| / bb_gain 0.961 to 1.051 |
| `pairs` FRB1 | all 6 pairs, 11 channels each: peak bins exact, \|I/max − fringe\| ≤ 0.0034 |
| `half` | BB peak in the predicted beam at all 33 unambiguous frequencies; upchannelizers within 0.0° (99th percentile ≤ 4.6°); FRB1 peak at (p, q) in all 48 channels; FRB2 source beam in all 48 channels (29 ties) |

With the peak-bin threshold halfway between 1 and the next predicted value (0.9957), the two
diagonal pairs failed in the four U = 32 channels, where the measured fringe deviates by up to
0.003. Kotekan runs for 17 s, 20 s and 53 s; the test takes 2 minutes.

## Result (CHIME, cx68, A40, Release)

All three variants pass. Grid cells beyond the 4 × 256 grid are wrapped into it. CHIME has no BB
beamformer, so `onehot` checks only N2.

- **`onehot`:** only the 8 one-hot autocorrelations are non-zero, equal to Σ|E|²/n within
  2.5·10^−8. The written feed positions are those of the config offset by (−33.0, −38.862) m:
  CHIMETelescope centres the array.
- **`pairs`:** the FRB1 fringes match within 0.0044. The peak bins are checked only for the two
  pairs separated along the 4 cylinders. Along the 512 beams of the cylinder axis the fringe
  levels have no gap of 2%, so the peak bins are not checkable there.
- **`half`:** T2's `frb2` variant passes with the 512 present dishes.
  CHIMETelescope has no dish types, so inventBFMask `auto` marks all dishes good; the
  Missing dishes carry no input.

Kotekan runs for 134, 44 and 95 s.
