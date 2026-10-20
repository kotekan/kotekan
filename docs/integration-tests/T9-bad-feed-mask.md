# T9: Bad feed mask

T9 checks that every subsystem excludes bad feeds exactly while the bad feed mask marks them bad,
so that whatever a bad feed carries does not matter. T9r checks the production path, a POST to
`/updatable_config/bad_inputs`.

## Errors this test catches

- **Leakage**: a bad input that still reaches a BB, FRB1, FRB2, FRB3 or RFI output, or a good
  N2 product, e.g. because a phase or weight is not zeroed, a grid cell maps to the wrong dish,
  or a kernel treats -8 or other garbage specially.
- **Timing**: a mask applied a frame, a lifetime or an FRB sample late or early, or two consumers
  of one mask that disagree on its time.
- **Wrong set**: the wrong inputs masked, polarizations swapped, or masks not combined with the
  Missing dishes.
- **N2 flags**: flags that are not the AND of the masks of the integrated frames.

## Schedule and streams

`testBadFeeds` (lib/testing/testSkySignal.hpp) drives `inventBFMask` and `inventVoltage`. Time
is cut into periods of `bf_change_frames` frames from `seq0`; period k has
`n = 3 + h(k, 0) mod 3` random bad inputs, the first n distinct values of `h(k, i) mod 128`,
with `h(k, i) = splitmix64(((k << 16) | i) ^ splitmix64(seed))`. Input `p·D + d` is
polarization p of dish d, the order of the mask. The check reproduces this in Python. With
`bf_mode: auto` the Missing dishes are bad as well.

There are two mask streams with the same content (see the lifetime finding in
dev_integration_tests.rst): `host_bf_mask_buffer` with a lifetime of L = 3 frames, for RFI, BB
and FRB1, and `host_n2_bf_mask_buffer` with one frame per correlation frame, for N2Accumulate.
Each mask frame takes the mask at its first sample; periods are multiples of both lifetimes, so
both streams carry the same mask at every sample. The runs start at
`seq0 = 2^33 + 5·T`, three frames before an N2 bin boundary, so that both N2 bins written begin
and end at a mask change.

## Poison

A bad feed may carry anything: no data (-8-8i), a stuck value, or an unconnected input. With
`bad_voltage: random`, `inventVoltage` replaces every sample of a bad input and of a Missing dish
by a random byte, so all 256 int4 pairs, -8 included, are equally likely. This changes 255/256 of
the values and makes every product with a bad input differ from the reference. Unlike lost
samples (T8), bad inputs must not be zeroed upstream: excluding them is the job of the mask.

## How each subsystem applies the mask

- **BB:** `calcBBPhase` makes phase frame k from mask frame k and sets `A = 0` for bad inputs
  (lib/stages/calcBBPhase.cpp:255). The beamformer is integer arithmetic, so a zero phase removes
  the input exactly. A BB sample at time t uses phase frame `(t − seq0) / L`.
- **FRB1:** `setFRB1Phase` sets `W = 0` at the grid cells of bad inputs, placing dishes as the
  kernels do (`frb_dish_index_at_location`: ArrayDishes at their grid indices, other dishes in the
  empty cells in order). Weight frame k is stamped `seq0 + T* + k·L` (setFRB1Phase.cpp:258). A
  float16 `0 · E` is ±0 for any int4 E, so a zeroed input adds nothing.
- **FRB2, FRB3:** linear in FRB1 at the same time and fine channel; FRB3 quantizes 256 samples
  with one offset and scale.
- **RFI:** `S012tilde` sums S0, S1, S2 over the good inputs, with the mask element found by ring
  position (lib/cuda/cudaRFIS012tilde.cpp:172); SKtilde and the RFI mask follow from it.
- **N2:** `cudaCorrelator` correlates all inputs. `N2Accumulate` writes per-element `flags`
  (1 good, 0 bad), the AND of the masks of the correlation frames of the bin; visibilities and
  weights of flagged products are not changed, they hold the bad data. A consumer must honour
  the flags.

## Predictions and tolerances

- **Masks:** both streams equal the schedule frame by frame, with `fpga_seq_num = seq0 + k·L`
  and `seq0 + k·T`, and agree at every sample. Exact.
- **A and W:** zero exactly for the bad inputs of their frame, non-zero (A) or
  `frb1_input_scale + 0i` (W) otherwise, for all U. Exact.
- **BB:** bit-identical to the reference everywhere.
- **FRB1, FRB2:** the sample labelled `ℓ = seq0 + T* + 192 j` reads the inputs
  `[ℓ − (M−1)U/2, ℓ + 192 + (M−1)U/2)` (U = 1: `[ℓ, ℓ + 192)`) and weight frame
  `j / 128`. It is *exposed* if a poisoned input in that window is good in its weight frame;
  all other samples must be bit-identical to the reference. Since `T* = (M−1)·64/2 = 96`, mask
  changes fall 96 samples into an FRB sample and every window spans them alike for all U:
  exactly the last FRB sample of each weight frame is exposed.
- **FRB3:** every 256-sample chunk contains an exposed sample, so its scale changes; clean
  values below saturation must be within one step plus T3's float16 allowance of the reference.
- **RFI:** `S012tilde` equals the sum of the single-feed `S012` over the good inputs of each
  256-sample block, so `S0tilde = 256 · n_good`; `S012tilde`, SKtilde, the RFI mask, and the
  single-feed S012 of good inputs are bit-identical to the reference.
- **N2:** the flags of each integration are the AND of the masks of its 24 frames. Products of
  two inputs good throughout the bin, their weights and the sample counts are bit-identical to
  the reference; n_valid must be non-zero.

Bit-exact comparison works because the good inputs carry the same voltages in both runs, every
kernel is deterministic, and a zeroed input contributes an exact zero. The N2 comparison also
needs a fixed telescope time origin (`gps_time`): without it the telescope takes the system time
at startup (lib/utils/CHORDTelescope.cpp:270), fringe stopping differs at float rounding level
between runs, and about 60 visibilities and weights of an integration change.

## T9r: POST to `/updatable_config/bad_inputs`

Two `bufferBadInputs` stages produce the streams, as in config/chord/pathfinder.j2 (one per
lifetime). All inputs are good until 3 s after the REST server answers; then inputs
5, 17, 70, 100 are posted with a start time one second later. `bufferBadInputs` applies an
update to the frames it produces once the wall clock passes the start time
(lib/stages/bufferBadInputs.cpp:257), so the change lands where each stage's lead over its
consumers puts it. The check requires one change per stream to the posted set, the two streams
to agree at every sample, and A, W, `S0tilde` and the N2 flags to follow the written masks.

## Runs

| run | array | bad inputs per period | voltage of bad inputs |
|---|---|---|---|
| `full_reference`, `full` | 64 dishes | 3 to 5 | signal / random |
| `half_reference`, `half` | 32 dishes, 32 Missing | 3 to 5 plus 64 Missing | signal / random |

`half` sets `rfi_feed_averaged_min_good_frac` to 0.1, as config/chord/pathfinder.j2 does.
Written: masks for 51 frames, A and W for 3 and 17 lifetimes, one voltage frame, BB and RFI 7
frames, FRB1 one and FRB2/FRB3 two FRB frames, two N2 integrations.

## Mutation check

Expectations only, re-checked with `--check-only` on kept output:

- Schedule one frame late: fails in both mask streams, A, W, the voltage, S012tilde and S012.
- One bad input of period 1 dropped: fails in the masks, A, W, S012tilde, the N2 flags and the
  good N2 products, and in FRB1/FRB2/FRB3 for `half`.
- FRB window placed before its label instead of after it: fails in FRB1, FRB2 and FRB3.

## Result (pathfinder, cx67, Release)

**PASS** with the N2Accumulate fix of #1790; every check is exact. Without that fix T9 failed in
the N2 flags only (last row).

| check | `full` | `half` |
|---|---|---|
| masks | 17 + 51 frames, 16 changes, equal to the schedule, streams agree | 65 to 67 bad |
| A, W (all U) | zero exactly for the bad inputs | same |
| voltage | 0 good values differ; 99.61% of bad values poisoned | same |
| BB | 704 643 072 values identical | identical |
| FRB1 / FRB2 | 0 clean differ; 99.5% of 2 654 976 / 3 539 968 exposed differ | same |
| FRB3 | ≤ 1.067 steps | ≤ 1.062 steps |
| RFI | S0tilde = 256 × 123…125, S012tilde, SKtilde, RFI mask identical | 256 × 62…63 |
| N2 good products | 5253 and 4753 per frequency identical | 1431 and 1176 |
| N2 flags | the AND over each bin; before #1790 extra: 33, 53 (bin 1), 3, 6, 31, 41, 98 (bin 2) | before #1790 extra: 3, 6, 31 (bin 2) |

The extra flags were the inputs bad in the frames before each bin: N2Accumulate folded the masks
of the frames skipped while waiting for the first bin, and the mask of the frame that completes a
bin into the next bin. #1790 folds only the masks of accumulated samples.

T9r: in three runs the beamformer stream changed at frames 54, 42 and 108 after `seq0` and the
N2 stream at 65, 58 and 131, so they disagree for 11 to 23 frames (**FAIL**); A, W, S0tilde
and the N2 flags followed their streams.

Kotekan runs for 38 to 57 s per run; the test takes 4.5 min and writes 2.3 to 2.5 GB per run.
T9r runs for 26 s and writes 0.4 GB.
