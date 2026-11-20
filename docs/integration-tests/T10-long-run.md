# T10: Long run

T10 runs the pipeline for about 10 minutes with everything that changes over time changing at
once: packet loss, bad feeds, and the BB, FRB1 and FRB2 lifetimes. T7 to T9 check these for a
few seconds; T10 checks that they still hold after thousands of repetitions, that no frame is
dropped, and that kotekan's memory stays flat.

## Errors this test catches

What accumulates over a long run, and is invisible in a short one:

- **Ring buffer wraps.** Host buffers have 4 frames, the FRB2 weights ring 2 matrices. A cursor
  that slips after many wraps, or a slot reused before its last reader is done, shifts a mask or
  a matrix against its data.
- **Lifetime boundaries.** The bad feed mask, BB phases and FRB1 weights change every 3 frames,
  the FRB2 weights every FRB frame, the bad feeds every 9 frames, and N2 bins every 24. The
  pattern of their boundaries repeats every 72 frames; a switch that drifts, is skipped, or
  deadlocks for one rare alignment shows up only after many periods.
- **Sequence arithmetic.** The run starts at `2^33 + 7·T` and lasts 7·10^7 samples. Lifetime
  indices, FRB frame labels (`seq0 + T* + k·49152`) and N2 bin starts must stay exact; a float
  or a rounding in any of them drifts. (Sample counts since the start stay below 2^31: an int32
  overflow would need about 3 h.)
- **Stalls and drops.** A consumer that falls behind, or a stage that stops at a lifetime
  boundary: kotekan times out, or writes fewer or non-contiguous frames.
- **Leaks.** Anything allocated per frame, lifetime or written file and not freed.

## Input

T2's weak tone (0.75 LSB, 19.5/64 cycles per sample) on FRB2 beam (12, 5), in 2.8 LSB Gaussian
noise, with the production gains, from `seq0 = 2^33 + 7·T` (T = 8192):

- random packet loss at 0.01% (`loss_seed` 10), lost voltages 0+0i so that N2 stays exact (T8);
- 3 to 5 random bad inputs changing every 9 frames (`bf_seed` 10), poisoned with random bytes
  where their packets are not lost (T9);
- `bf_mask_lifetime_frames` 3 (BB phases and FRB1 weights, L = 24576 samples) and an FRB2
  weights lifetime of one FRB frame (L₂ = 49152 samples, T7).

The run covers 8640 frames (362 s simulated). Every writer stops after the frames that end
within this span, so all finish at about the same simulated time:

| output | frames | per frame |
|---|---|---|
| `n2_bf_mask`, `n2k_plcounts` | 8640 | one correlation frame |
| `bf_mask`, `bb_beam_positions`, `bb_beam_ids` | 2880 | one lifetime |
| `frb_pl_mask`, `frb2_beam_positions`, `frb2_beam_ids` | 1439 | one FRB frame |
| `n2_vis` | 359 integrations | the first 16 of 384 frequencies |
| `frb2_beams` | the first of every 359 FRB frames | 453 MB |

`n2_vis` and `frb2_beams` use `hdf5FileWrite` frame decimation (`write_x_frames` /
`per_y_frames`, jinja variable `write_decimation`); all frames would be 14 GB of N2 and 470 GB
of FRB2. Writing one FRB2 frame takes about 5 s and stalls the pipeline meanwhile.

## Checks

- **Frames.** Every writer received its number of frames with contiguous `fpga_seq_num`; the N2
  integrations are contiguous from the first bin boundary (`seq0 + T`); the decimated writers
  wrote the first X of every Y frames, as documented.
- **Switches.** Both mask streams equal the schedule in every frame and agree at every sample
  (T9). Beam positions and ids equal the configuration, one frame per lifetime at `seq0 + k L`
  (BB) and `seq0 + T* + k L₂` (FRB2) (T7). The N2 flags of every written integration are the
  AND of the masks of its 24 frames.
- **Loss.** `n2k_plcounts` equals the lost samples per frame and frequency, and every FRB PL
  mask frame the propagated loss (T8), exactly. Every written N2 integration has
  `n_pl` = the lost samples, `n_valid` = 24 T − `n_pl` and `n_rfi` = 0.
- **Values.** No NaN or Inf in N2 visibilities and weights or in FRB2; in every written FRB2
  frame the source beam holds the peak (T2), so the FRB2 weights are right in late lifetimes.

## Loss rate

`testPacketLoss` drops each packet (16 samples, 4 frequencies) independently with probability
p = 10^−4. Over N = 8640 · 512 · 96 = 4.25·10^8 packets the number lost is binomial, with mean
Np = 42 467 and σ = √(Np(1−p)) = 206, 0.49% of the mean. The check counts the lost packets as
Σ `n2k_plcounts` / 64 (16 samples in each of 4 frequencies) and requires |K − Np| ≤ 3σ. The hash
is deterministic, so this confirms that the generator applies the configured rate, and the exact
checks confirm that the pipeline counts every packet. A rate error of 1.5% (3/√(Np)) fails.

## Memory

`run.py` samples kotekan's `VmRSS` (`/proc/<pid>/status`) and its GPU memory (`nvidia-smi`)
every 30 s. Kotekan allocates its buffers, including the pinned host memory, at startup; RSS
reaches its plateau of 7.2 GB within 30 s. The check skips a warm-up of 60 s and compares the
median of the last third of the remaining samples with the median of the first third; the
difference may be at most 32 MiB. A leak raises every later sample; a transient allocation
raises single samples and does not move the medians. (`hdf5FileWrite` holds about one extra
frame while it compresses, 475 MiB for FRB2; one sample fell into such a write.) The medians are
about 340 s apart, 5000 frames and 15 000 written files, so a leak of 6.5 KiB per frame or 2 KiB
per file fails. Allocator and HDF5 caches grow by a few MiB.

## Mutation check

Expectations only, re-checked on kept output:

- Bad feed schedule one lifetime late: fails in both mask streams (7446 and 22 338 entries) and
  the N2 flags (4063 of 6103 files).
- One lost packet (frequencies 156 to 159) dropped from the prediction: fails in `n2k_plcounts`
  (4 entries) and the FRB PL mask (16 samples).
- Expected loss rate 2% higher: fails at −3.7 σ.
- Memory tolerance 1 MiB: fails.

## Result (pathfinder, cx67, Release)

**PASS** with the hdf5FileWrite fix of #1797; every check is exact. Without that fix T10 failed
in the frame decimation only: `hdf5FileWrite` wrote the first X + 1 of every Y frames instead of
the first X as documented (N2 17 frequencies, FRB2 frames 0, 1, 359, 360, …).

| check | result |
|---|---|
| run | 8640 frames, 362 s simulated; kotekan about 585 s wall-clock, 13 to 19 frames/s per tenth of the run, no trend |
| frames | every writer got all its frames, contiguous `fpga_seq_num` |
| masks | 2880 + 8640 frames, 959 changes, equal to the schedule, streams agree |
| beams | BB 2880 and FRB2 1439 frames hold the configured beams at `seq0 + k L`, `seq0 + T* + k L₂` |
| `n2k_plcounts` | 2 722 944 lost samples in 8640 frames, exact |
| loss rate | 42 546 packets of 4.25·10^8 lost, expected 42 467 ± 206: +0.38 σ |
| FRB PL mask | 1439 frames, 2 352 584 invalid samples, exact |
| N2 | 359 contiguous integrations; counts and flags (8 to 20 elements flagged) exact; no NaN or Inf |
| FRB2 | beam 92 holds the peak in all 384 channels in all 10 frames; no NaN or Inf |
| memory | 7160 MiB, growth +1.5 MiB; single samples up to +475 MiB during an FRB2 write; GPU 11 092 MiB throughout |
| decimation | the first 16 of every 384 N2 frequencies; FRB2 frames 0, 359, 718, 1077, 1436 |

The test takes about 15 minutes (5 of them for the check) and writes 4.2 GB in 26 274 files:
FRB2 3.3 GB, N2 0.6 GB, the rest 0.3 GB.

## Result (CHIME, cx68, A40, Release)

**PASS**; every check is exact. As in T9, the poison is `bad_voltage: random_valid` and the mask
lifetime 3 frames, with changes every 9 frames. The N2 integrations are 24 frames (1 s), as in
config/fengine.

| check | result |
|---|---|
| run | 8640 frames, 362 s simulated; kotekan 1670 s wall-clock, 4.9 to 5.8 frames/s per tenth of the run |
| frames | every writer got all its frames, contiguous `fpga_seq_num` |
| masks | 2880 + 8640 frames, 959 changes, equal to the schedule, streams agree |
| beams | BB 2880 frames hold the 32 beams at `seq0 + k · 49152`; FRB2 1439 frames hold the 1024 production beams at `seq0 + T* + k L₂` |
| `n2k_plcounts` | 234 112 lost samples in 8640 frames, exact |
| loss rate | 3658 packets of 3.54·10^7 lost, expected 3539 ± 59: +2.00 σ |
| FRB PL mask | 1439 frames, 274 176 invalid samples, exact |
| N2 | 359 contiguous integrations of all 16 frequencies; counts and flags (9 to 20 elements flagged) exact; no NaN or Inf |
| FRB2 | beam 23 holds the peak in all 16 channels in all 5 written frames; no NaN or Inf |
| memory | 17 425 MiB, growth +0.5 MiB; GPU 34 618 MiB throughout |

Kotekan uses about 18 CPU cores. The earlier CHIME setup, without BB and HFB and with the input in
element order, ran at 15 frames/s; the cause of the slowdown is not known.
