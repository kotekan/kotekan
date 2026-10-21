# T8: Packet loss

T8 checks that lost F-engine packets reach every packet-loss (PL) mask exactly as predicted, and
that lost samples do not change any valid output sample.

## Errors this test catches

- **Wrong mask windows**: an upchannelized or FRB mask that is off by a sample, ignores the
  `T*` skip, uses the wrong number of taps M or the wrong U or Td, or maps frequencies to the
  wrong fine channels.
- **Wrong bit handling**: bit order within a byte, the 2-sample bits of the input mask, an OR
  instead of an AND.
- **Leakage**: a valid upchannelized, BB, FRB1 or FRB2 sample that depends on a lost sample,
  e.g. because a mask is shifted against its data by a ring buffer offset.
- **Quantizer**: -8 that does not match the FRB mask, or an offset or scale that includes
  invalid samples.
- **N2 normalization**: wrong valid or lost sample counts, or visibilities not normalized by the
  valid samples.

## Loss model

`testPacketLoss` (lib/testing/testSkySignal.hpp) drives both `inventPLMask` and `inventVoltage`.
A packet holds 16 FPGA samples of 4 frequencies for all dishes and polarizations. Packet
`(p, q) = (seq / 16, f / 4)` is lost at random if

    splitmix64(((p << 16) | q) ^ splitmix64(seed)) < rate · 2^64.

The check reproduces this hash in numpy. Loss boxes `[begin, end)` remove all frequencies, at
the PL mask resolution of 2 samples. Relative to `seq0 = 2^33 + 7·T` (T = 8192), the boxes are:

- `[80, 96)`, inside the first `T* = (M−1) Umax/2 = 96` samples;
- `[T, T + 32)` and `[2T − 48, 2T)`, at the start and the end of a frame;
- `[3T − 32, 3T + 48)`, across a frame boundary;
- `[T* + 49152 − 16, T* + 49152 + 16)`, across the boundary of the first two FRB frames;
- seven single sample pairs at `1.5 T + 514 k`, k = 1…7, at every even offset within a packet.

The single pairs matter: the upchannelizer windows start at `seq0 + 96 − 3U/2 + U t`. For U = 4,
8 and 16 every window edge lies at least 2, 4 or 8 samples from a packet boundary, so with
packet-aligned loss alone a window off by one sample would go unnoticed.

`inventVoltage` overwrites lost samples (`lost_voltage`): with 0+0i (`zero`, byte 0x88), as
`TransposeBasebandArray` does for lost blocks (lib/stages/TransposeBasebandArray.cpp:370), or
with -8-8i (`nodata`, byte 0x00).

## Propagation rules

Every mask is predicted per sample and frequency from the injected loss, and must equal the
kotekan mask exactly, for all dishes and polarizations:

- **Input and expanded mask:** sample s is valid iff it is not lost. The input mask has one bit
  per sample pair, the expanded mask one bit per sample, least significant bit first.
- **BB mask:** the AND of the expanded mask over dishes, per time, frequency and polarization.
- **Upchannelized mask (each U):** the output labelled L (`fpga_seq_num + U t̄`, with
  `fpga_seq_num = seq0 + T*`) is valid iff no input sample in
  `[L − (M−1)U/2, L − (M−1)U/2 + MU)` is lost.
- **FRB mask:** an FRB1 sample sums Td = 192/U upchannelized samples, so the sample labelled L
  (`seq0 + T* + 192 k`) depends on `[L − (M−1)U/2, L + 192 + (M−1)U/2)`, (Td + M − 1) U input
  samples. At U = 1 the beamformer reads the voltages directly: `[L, L + 192)`. The U fine
  channels of a coarse channel share its mask.

## Valid data

A valid output sample depends only on valid input samples, which are the same in every run, and
the GPU kernels are deterministic. Valid upchannelized, BB, FRB1 and FRB2 samples must therefore
equal the reference run bit for bit. As a control, the check reports how many invalid samples
differ, to show that the lost voltages reach the outputs; where the PFB window gives them little
weight, the int4 output often rounds to the reference value.

**FRB3.** Invalid samples must be -8 and valid ones must not. `cudaQuantize4` takes each chunk's
offset and scale over its valid samples, so they differ from the reference. A valid sample below
saturation in both runs encodes the same FRB2 value x, so the dequantized values
`x̂ = offset + scale·q` of the two runs differ by at most `(scale + scale_ref)/2` plus T3's
float16 allowance `2^−9 (2|x| + |offset| + |offset_ref| + scale + scale_ref)`: one step.

## N2

How lost samples are excluded:

- `cudaCorrelator` sums `RFI(t) E_a conj(E_b)` over each sub-integration; it does not read the PL
  mask. The RFI statistics S1 and S2 also come from the voltages alone. Both rely on lost samples
  being 0+0i (external/n2k/include/n2k/rfi_kernels.hpp:142).
- `cudaPL1bitCorrelator` counts the samples with `PL_a PL_b RFI = 1`; `CountLostPLSamplesScalar`
  counts the lost samples per frame and frequency (2 per mask bit).
- `N2Accumulate` sums the first count entry into `n_valid_fpga_ticks` and the lost counts into
  `n_pl_fpga_ticks` (lib/stages/N2Accumulate.cpp:676, 694), and writes
  `V = fl(1/n_valid) · Σ corr` in float32 (line 1243).

The `sparse` run therefore writes 0+0i, and the check predicts, per frequency and product,

    V n_valid = V_ref n_ref − Σ_lost E_a conj(E_b),

with `Σ_lost` from one frame of reference voltages (the input repeats every frame). The sums are
exact integers below 2^24; `fl(1/n)·v` has a relative error of at most 2·2^−24, so the check
allows `|Δ| ≤ 10^−6 (|V n| + |V_ref n_ref|)`. Per frame, every `n2k_counts` entry must be T minus
the lost samples and `n2k_plcounts` the lost samples. The integration must count every lost
sample as lost, all others as valid, and no RFI flags in either run. Fringe stopping is off,
since `N2Accumulate` truncates fringe-stopped values (T6).

With -8-8i in the lost samples instead (tried once), the poisoned samples raise the spectral
kurtosis, RFI excision flags 1.29 M samples of the integration, and V n deviates by up to 19%.

## Runs

| run | random loss | lost voltage | outputs |
|---|---|---|---|
| `reference` | none | — | masks, data, N2, one voltage frame |
| `sparse` | 0.01% (about 5 packets per frame), boxes | 0+0i | masks, data, N2 |
| `dense` | 1%, boxes | -8-8i | masks, data |

Written: masks and N2 counts 8 frames, upchannelized voltages and BB 4, the FRB mask 2 FRB
frames, FRB1 to FRB3 one FRB frame, and the first full N2 integration (frames 1 to 24).

## Mutation check

Only expectations were changed in Python, and the kept output re-checked with `--check-only`:

- One randomly lost packet dropped from the prediction (`sparse`): fails, in every mask that
  covers it, in FRB1/FRB2/FRB3 data, the N2 counts and Σ V n.
- Upchannelizer windows shifted by +1 or −1 sample: fails for every U. Before the single-pair
  boxes were added, +1 was caught for U = 2, 32 and 64 only.
- N2 prediction without `Σ_lost`: fails, |Δ| up to about 5000 counts.

## Result (pathfinder, cx67, Release)

**PASS**, all three runs.

| check | `sparse` | `dense` |
|---|---|---|
| samples lost (32 frames) | 0.093% | 1.08% |
| expanded mask | 1 398 656 invalid bits, exact | 5 345 408, exact |
| upchan masks | U=2: 203 328 … U=64: 10 656 invalid, exact | U=2: 844 608 … U=64: 39 104, exact |
| FRB mask | 65 004 of 1 769 984 invalid, exact | 335 878, exact |
| valid upchan, BB, FRB1, FRB2 | 0 differ | 0 differ |
| invalid that differ | 25% (U=64) to 98% (BB) | 42% (U=64) to 99% (BB) |
| FRB3 | -8 exactly where invalid; ≤ 1.089 steps | ≤ 1.076 steps |
| N2 counts | 87 416 lost samples in 8 frames, exact | — |
| N2 integration | 85 944 lost, n_valid 196 324 to 196 402, no RFI flags | — |
| Σ V n | max relative |Δ| 5.1·10^−8 (0.32 counts); Σ_lost up to 5656 | — |

The reference run has every mask valid and every count at T. No kotekan error was found.

Kotekan runs for 40, 40 and 36 s; the whole test takes 3 min. The runs write 2.6, 2.4 and
2.4 GB, mostly FRB1, the upchannelized voltages and the voltage frame.

## Result (CHIME, cx68, A40, Release)

**PASS**, all three runs. As in production, the loss enters as lost samples per group of 4
frequencies, which zeroSamples applies to the voltages (also in `dense`) and lostSamplesToPLMask
turns into the PL mask. CHIME has no BB beamformer, only U = 16, and its N2 counts come from
lostSamplesToN2Counts.

| check | `sparse` | `dense` |
|---|---|---|
| samples lost | 0.014% | 1.00% |
| PL mask | 95 232 invalid bits, exact | 754 688, exact |
| expanded mask | 761 856, exact | 6 037 504, exact |
| U = 16 mask | 70 656, exact | 1 668 096, exact |
| FRB mask | 1088 of 131 072 invalid, exact | 32 512, exact |
| valid U = 16, FRB1, FRB2 | 0 differ | 0 differ |
| FRB3 (8 bits) | 0 exactly where invalid; ≤ 1.53 steps | ≤ 1.57 steps |
| N2 counts | 2976 lost samples in 8 frames, exact | — |
| N2 integration | 6400 lost, n_valid exact, no RFI flags | — |
| Σ V n | max relative \|Δ\| 6.0·10^−7; Σ_lost up to 8655 | — |

Kotekan runs for 93, 90 and 67 s.
