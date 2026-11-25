# T7: Beam pointings

T7 checks that the beams `setBBBeams` and `setFRBBeams` are configured with are the beams that
BB and FRB2 form, in `grid_degrees` and `manual` mode, and that this holds across every lifetime
of the BB phases, the FRB1 weights and the FRB2 weights. Beams can only be set at startup
(re-pointing at runtime is a missing feature), so all pointings are static.

## Errors this test catches

- **Wrong grid positions**: degrees taken as radians or as direction cosines (`grid` instead of
  `grid_degrees`), x and y swapped, the wrong beam order, wrong end points, or a single-beam axis
  at an end point or NaN (fixed in #1785).
- **Wrong manual beams**: the list reordered, truncated or shifted, or ids that lose bits (the
  ids are random 64-bit values, almost all above 2^53 and about half above 2^63).
- **Positions and beams out of step**: `calcBBPhase` or `cudaCalcFRB2Weights` forming beam b
  from another beam's position, or BB and FRB2 disagreeing about where a direction lies.
- **Wrong stream timing**: positions frames that do not start at the stream origin, FRB2
  positions without (or with twice) the upchannelizer offset `T*`, or the wrong
  `time_downsampling_fpga`.
- **Errors at lifetime boundaries**: a phase or weights matrix that is read before it is
  written (zero or stale memory), the wrong ring buffer element, or samples dropped or repeated
  when a kernel invocation stops at the end of a lifetime.

## Beam positions

`grid_degrees` (both setters) places beam `b = b_x + n_x b_y` at

    θ_x = grid_point(x_min, x_max, n_x, b_x) = (x_min (n_x − 1 − b_x) + x_max b_x) / (n_x − 1),
    n_b = (sin θ_x, sin θ_y),   id_b = b,

with θ in degrees and a single beam (`n_x = 1`) at `(x_min + x_max) / 2`. In `manual` mode the
positions are `(x_dir_grid, y_dir_grid)` and the ids as listed, in list order. The check
evaluates these in double precision with the same operations as `lib/utils/beamUtil.cpp` and
the setters, and rounds them to `float32` as written.

## Variants

Both variants use T2's weak tone (amplitude 0.75 LSB, 19.5/64 cycles per sample, 2.8 LSB noise)
from one direction that is a configured BB beam and a configured FRB2 beam at the same time.

| variant | BB beams | FRB2 beams | source |
|---|---|---|---|
| `grid` | `grid_degrees`, 1 × 16: x in [0.90625°, 1.90625°] (midpoint 1.40625°), y in [−2.96875°, 1.71875°] (the FRB2 spacing, centred on the source) | production 16 × 16 `grid_degrees`, 0.3125° spacing | BB beam 7 = FRB2 beam (12, 5) = (1.40625°, −0.78125°) |
| `manual` | 16 beams at random positions within ±2.5°, at least 0.4° apart | the production grid jittered by up to 0.1° per axis, shuffled | (0.5°, 0.8°), BB beam 11, FRB2 beam 173 |

The single-beam axis has an asymmetric range, so a midpoint at 0 or at an end point fails. The
BB column is centred on the source: on CHORD (96 beams 0.078° apart), a column that starts at
the source has a grating lobe of the source on another BB beam at every frequency, and no BB
peak is unambiguous. In `manual`, the source replaces the nearest jittered FRB2 beam.

Lifetimes are short, so the matrices are recomputed often: `bf_mask_lifetime_frames` 3 gives a
BB phase and FRB1 weights lifetime of L = 3 T = 24576 samples (the shortest that is a multiple of
both T = 8192 and the FRB processing quantum 192 U = 12288), and
`frb2_weights_lifetime_in_samples` is one FRB2 frame, L₂ = 256 · 192 = 49152 samples. The run
writes 10 BB frames (4 lifetimes) and 3 FRB2 frames (3 FRB2 and 6 FRB1 lifetimes).

## Checks

**Positions and ids.** Every written frame of `bb_beam_positions`, `bb_beam_ids`,
`frb2_beam_positions` and `frb2_beam_ids` must equal the configured beams exactly. Frame k must
have `fpga_seq_num` `seq0 + k L` (BB) and `seq0 + T* + k L₂` (FRB2), with `T* = 3 · 64 / 2 = 96`
the upchannelizer output offset (`upchan_output_offset`), and `time_downsampling_fpga` L or L₂.

**Peaks.** With T2's criteria and the configured positions:

- **BB:** the measured power must peak in the beam with the largest predicted tone power
  `(A · bb_gain · |Σ_d exp(2πi ν (n_b − n)·r_d / c)|)²`, at every frequency where the
  prediction is unambiguous (≥ 1.25 times and ≥ 1 LSB² above the second beam). Exact index.
- **FRB2:** in every coarse channel, the source beam must hold ≥ 98% of the peak tone power
  (beam power in the tone's fine channel minus the median over beams); grating lobes tie.
- Every BB beam must be non-zero. (`calcBBPhase` stops kotekan on a non-finite position.)

**Continuity.** The tone in the source beam is estimated in consecutive windows:

- **BB**, windows of 512 samples: `c_w(f, p) = Σ_{t∈w} J_src(f, p, t) exp(−2πi f_tone (t − seq0))`
  and `r_w = Σ_{f,p} c_w conj(c̄) / Σ_{f,p} |c̄|²`, with `c̄` the mean over windows. A window
  with wrong phases changes |r_w|; a dropped or repeated sample turns its phase by
  `2π f_tone = 110°` per sample.
- **FRB2**, windows of 16 samples (3072 FPGA samples): the source beam's power summed over the
  tone's fine channel of every coarse channel, minus the median over beams, normalized by its
  median over windows.

Windows start at the lifetime boundaries. Every window must lie within 6σ of the median, with σ
the robust standard deviation (1.4826 × median absolute deviation) over all windows; the steps at
the boundaries are reported.

With static pointings, consecutive matrices are identical, so a stale element that holds the
previous lifetime's matrix cannot be seen. Re-pointing at runtime would expose it.

## Mutation check

Only expectations were changed in Python, and the kept output re-checked with `--check-only`:

- FRB2 positions expected at `seq0 + k L₂` (without `T*`): fails, `fpga_seq_num` 96 samples
  later than expected.
- The single BB beam axis expected at `x_min` instead of the midpoint: fails, the source is no
  configured BB beam.
- A 1% step in the BB tone after the first boundary: fails, a window at 9.1σ.

## Result (pathfinder, cx67, Release)

**PASS**, both variants.

| check | `grid` | `manual` |
|---|---|---|
| positions, ids | 4 BB and 3 FRB2 frames exact, at `seq0 + k L` and `seq0 + T* + k L₂` | same |
| BB peak | source beam at all 275 unambiguous frequencies | source beam at all 366 |
| BB beams | all 16 non-zero, rms 3.9 to 6.4 LSB | same |
| BB continuity | 160 windows within 2.9σ, σ = 0.08%; steps ≤ 0.19%; phase range 0.19° | within 2.3σ, σ = 0.08%; steps ≤ 0.15%; 0.20° |
| FRB2 peak | source beam in all 384 channels (79 grating-lobe ties) | all 384 (8 ties) |
| FRB2 continuity | 48 windows within 2.6σ, σ = 0.08%; steps ≤ 0.19% | within 2.3σ, σ = 0.14%; steps ≤ 0.44% |

Kotekan runs for 27 s and 25 s; the test takes 81 s. Each variant writes 1.9 GB, mostly the
10 BB frames and 3 FRB2 frames.

## Result (CHIME, cx67, A40, Release)

**PASS**, both variants. The production FRB2 beams are `manual`. `grid` keeps them, with the
source on beam 23 at (0.8°, −55.6°), and centres the BB column (1 × 32) on it with the FRB2 y
spacing at the source; `manual` jitters both.

| check | `grid` | `manual` |
|---|---|---|
| BB positions, ids | 4 frames of 32 beams exact, at `seq0 + k · 49152` | same |
| BB peak | source beam 15 in all 16 frequencies | beam 11 in all 15 unambiguous frequencies |
| BB beams | all 32 non-zero, rms 3.91 to 9.25 LSB | rms 3.90 to 9.25 LSB |
| BB continuity | 320 windows within 2.7σ, σ = 0.14%; steps ≤ 0.22%, phase range 0.41° | within 3.5σ, σ = 0.12%; steps ≤ 0.26%, phase range 0.46° |
| FRB2 positions, ids | 3 frames of 1024 beams exact, at `seq0 + T* + k L₂` | same |
| FRB2 peak | source beam in all 16 channels (1 grating-lobe tie) | beam 173 in all 16 (no ties) |
| FRB2 continuity | 48 windows within 1.7σ, σ = 0.05%; steps ≤ 0.10% | within 3.9σ, σ = 0.03%; steps ≤ 0.10% |

The BB and FRB1 lifetimes are 3 frames (49152 samples), the FRB2 lifetime one FRB2 frame (98304
samples). setBBBeams computes the grid positions from degrees, so the BB beam on the source is
the closest one (within 10^−6). Kotekan runs for 17 s and 16 s.
