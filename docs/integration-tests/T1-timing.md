# T1: Timing

T1 checks that each output sample carries the correct time label. A short signal goes into
the X-engine at known FPGA sample numbers. In every output it has to appear at the
FPGA sequence numbers that kotekan's timestamp convention predicts.

## Errors this test catches

- **Off-by-one and off-by-frame errors in `fpga_seq_num`.** A stage might stamp the wrong
  frame, count from the wrong origin, or advance by the wrong amount from frame to frame.
- **Misaligned upchannelizers.** Every upchannelizer, for every factor U, skips
  `T* = (M−1)·Umax/2` input samples once at startup, so that all outputs start at the same time
  (with M = 4 taps and Umax = 64 for CHORD, T* = 96). If one U skips the wrong amount, or labels
  its output by the start of its PFB window rather than its centre, its signal moves by up to
  `(M−1)U/2` samples.
- **Wrong time direction or reordered samples** within a frame, e.g. swapped time tiles in a
  kernel. The signal centroid then moves, or the signal spreads out.
- **FRB time-downsampling errors.** An FRB sample might be labelled by its end instead of its
  beginning, or the downsampling factor might differ from the metadata.
- **Truncation in sequence-number arithmetic.** The `seqlarge` variant starts at
  `2^33 + 7·num_times`. Any 32-bit intermediate, or any arithmetic that assumes the stream
  starts at 0, shifts or corrupts the labels.
- **Missing or non-contiguous frames.** Consecutive frames must advance `fpga_seq_num` by
  `dim0 · time_downsampling_fpga`.
- **Frequency metadata errors.** `coarse_freq` must be a subset of the configured channels.
  `freq_upchan_factor` must equal U, and `freq_upchan_index` must lie in `[0, U)`.

## Input

A "box": every dish, polarization and coarse channel carries the constant voltage `1 + 0i` for
FPGA samples `t ∈ [b0, b1)`, and zero elsewhere. This is a tone at the centre of each coarse
channel, from zenith, so every beamformer sees it.

- `b0 = t0 + 3·T + 1000`, with `T` the frame length (8192 samples) and `t0` the first sample. The
  box therefore starts in the middle of the fourth frame, at a sample that is not a multiple of
  16, 64 or any downsampling factor.
- The box is 3000 samples long.
- The amplitude is small enough that no output saturates with the production gains. Saturation
  would flatten the box and could move the upchannelized power centroid.

## Timestamp convention

kotekan labels a sample with the FPGA sequence number of its *beginning*. For a sample that
spans `S` FPGA samples, the centre is `label + (S−1)/2`. The checks compare power-weighted
centres of the signal with the centre of the box,

    c = (b0 + b1 − 1) / 2 .

## Checks

**Input voltage** (`T` axis, one FPGA sample per element). The non-zero samples must be exactly
`[b0, b1)`. This validates the generator and the labelling of single-file output.

**Baseband beams** (`[Thi, B, F, P, T]`, one FPGA sample per element). The beamformer combines
dishes within a sample and has no memory, so the first and last non-zero samples must be exactly
`b0` and `b1 − 1`. **Tolerance: exact.**

**Upchannelized voltages**, for each U ∈ {2, 4, 8, 16, 32, 64} (`[Tbar, Fbar, P, D]`):

- Output sample `t̄` reads input samples `[U t̄, U t̄ + M U)` and is labelled with the beginning
  of its centred sample, `L = U t̄ + (M−1)U/2` (plus the stream origin).
- Its centre time is `L + (U−1)/2`.
- With the power `P(t̄) = Σ |Ē|²` over the fine channels in use, dishes and polarizations, the
  centroid `Σ P (L + (U−1)/2) / Σ P` must equal `c`.
- The PFB window is symmetric, so leakage does not move the centroid. Rounding of the int4
  output does, slightly.
- **Tolerance: ±U/4 samples**, the same as the upchannelizer alignment test in `gpu_batch`.
- In addition, no power may lie further than `M U` samples outside the box.
- All U must start at `t0 + T*`.

**FRB1 and FRB2 intensities** (time downsampled by `Tds` = 192 FPGA samples):

- The centre of FRB sample `i` is `label + i·Tds + (Tds−1)/2`.
- The power-weighted centroid must equal `c`.
- **Tolerance: ±Tds/4.** The box covers about 16 FRB samples, so the partly filled edge samples
  shift the centroid by much less than this.
- The first sample must be at `t0 + T*`.

**HFB** (CHIME). The U = 128 upchannelizer is checked as the others, starting at `t0 + T*` of
the HFB chain. HFB1 sums 16384 samples per output sample: exactly the HFB1 samples that overlap
`[box_begin − M U, box_end + M U)` must hold power. HFB2 is checked only for the frame that
holds the box.

**Metadata audit**, on every output:

- consecutive frames are contiguous;
- no NaN or Inf in float outputs;
- frequency metadata is consistent with the config.

Only frequency slots listed in `coarse_freq` count. Buffers are sized for the largest channel
count, and the remaining slots hold the no-data value `−8−8i`.

## Variants

| variant    | first FPGA sample  | purpose                                             |
|------------|--------------------|-----------------------------------------------------|
| `seq0`     | 0                  | the simplest origin                                 |
| `seqlarge` | 2^33 + 7·T         | 64-bit arithmetic; origin not aligned to lifetimes  |

`N2Accumulate` requires correlation frames to start at multiples of the frame length, so the
large origin is a whole number of frames.

## What this test does not cover

- N2 time bins. These are aligned to Earth rotation angle bins rather than to the box.
- Lifetime boundaries of slowly varying inputs (see T7, T9).
- Time labels of the packet-loss masks (see T8).

## Mutation check

Shifting the expected box by 16 samples in the check (not in the input) has to make the test
fail. It does, at the first exact comparison.

## Result (pathfinder, cx67, Release)

Both variants pass:

- **BB:** signal edges exact.
- **All upchannelizers:** power centroids within ±U/4.
- **FRB1:** centroid 0.5 samples from the box centre.
- **FRB2:** centroid 0.3 samples from the box centre.

## Result (CHIME, cx67, A40, Release)

Both variants pass, with the production gains of config/fengine.

- **Input:** the transposed voltage holds the signal exactly in [box_begin, box_end).
- **BB:** signal edges exact.
- **U = 16:** centroid 2.0 samples from the box centre (±4).
- **U = 128 (HFB):** centroid 1.4 samples from the box centre (±32).
- **FRB1, FRB2:** centroids 2.1 samples from the box centre (±96).
- **HFB1:** power only in the sample at `t0 + T*` (16384 samples), which holds the box.
- **HFB2:** power only in the frame at `t0 + T*`.

Kotekan runs for 37 s per variant.
