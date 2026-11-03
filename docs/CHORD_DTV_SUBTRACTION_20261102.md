# DTV band (470-600 MHz): projection vs blind channel identification, 2026-11-02

KV's question: the 480-580 MHz band carries digital TV, likely one transmitter per channel plus
reflections. Can a mode or two projected out of those channels recover Tau A, as the N2
projection does for the GNSS lobes? Data: chunks 0, 48 and 95 (0, 40 and 80 s) of the
2026-10-30 10:59:20Z baseband dump, all 16 instances, 470-800 MHz kept (freq_id 2407-4096),
Tau A at the boresight throughout. Tools `~/gnss/bb1030/dtv_*.py`, data
`cf06:/var/tmp/bb1030/dtv` (chunk 0) and `cf05:/var/tmp/bb1030/dtv` (48, 95).

Units throughout: whitened eigenvalues of the 128-input covariance in units of the noise block
(mean of eigenvalues 20..120). On a clean channel Tau A is the pair 6.1 / 5.5 (one mode per sky
polarisation) and the third mode is 1.5. "Tau A kept" is the excess of a template beam over the
mean per-input residual power, relative to the clean channels; the template is the top-2
subspace of the clean channels extrapolated across the DTV channel by per-input delay, which
reproduces clean channels to 0.99 even across a 6 MHz gap.

## What is in the band

Seven ATSC 8-VSB channels, each identified by its pilot at the lower edge + 0.31 MHz:

| ATSC ch | MHz | lambda1 / noise | modes above Tau A | DTV share of the input power |
|---|---|---|---|---|
| 17, 19 | 488-494, 500-506 | 8-9 | 2 | 10 % |
| 24 | 530-536 | 57 | 5 | 44 % |
| 30 | 566-572 | 143 | 11 | 81 % |
| 31, 33, 35 | 572-602 | 7.7 | 2-3 | 9 % |

The strong channels are one dominant mode plus a multipath tail (channel 24: 57, 18, 14, 11,
6.3, 4.8, 4.6, 3.7 ...) that runs through Tau A's level. The mode subspaces are the same at
42 ms and 0.84 s to 1.00.

Side finding: the F-engine's per-channel gain is lower inside the DTV channels (total power
only 14-19 % above the neighbours while the noise is 52 % / 17 % of it in channels 24 / 30, i.e.
2 and 7 dB down). The gain tables were levelled with the DTV in; channel 30's noise is
quantised at about 0.8 LSB rms. Only 2 % of samples are on the rail.

## Eigen-projection does not work here

Tau A kept after projecting the top K modes of the channel's own covariance:

| channel | K = 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 17, 19 | 0.94 | 0.88 | 0.75 | 0.25 | 0 | 0 |
| 31, 33, 35 | 1.0 | 0.55-0.86 | < 0.2 | 0 | 0 | 0 |
| 24 | 0.73 | 0.78 | 0.72 | 0.71 | 0.69 | 0.16 |
| 30 | < 0.3 at any K | | | | | |

The DTV tail sits at Tau A's eigenvalue, eigenvectors of near-equal eigenvalues mix, and a
fixed K removes Tau A with the tail. In channel 24, four modes out leave Tau A at 70 % with four
residual DTV modes of as much power as Tau A. Channel 30 is not recoverable this way.

## Blind single-source channel identification and subtraction works

Model per 195 kHz channel: x_i(t) = sum_tau g_i(tau) s(t - tau) + n_i(t), one transmitter
timestream and a lagged impulse response per input (49 lags, -41 .. +205 us). Solve: s from the
top eigen-beam of the covariance deflated by the Tau A template (so the timestream carries no
Tau A), g_i by least squares (Toeplitz normal equations via FFT), subtract, then peel the next
source from the residual the same way. `dtv_blind.py`, `dtv_gstab.py`.

| channel | sources | residual after peeling | Tau A kept |
|---|---|---|---|
| 17 | 1 | 6.0, 5.3, 1.7, 1.6 | 0.96-0.98 |
| 24 (16 sub-channels) | 2 | 5.5, 4.9, 1.6-2.0 | 0.86-0.92 |
| 30 | 2 | 4.8, 4.1, 2.0 (refined) or 5.1, 4.5, 3.8-4.2 (single fit) | 0.66-0.72 |

After peeling, every channel is the Tau A pair over the clean floor. The remaining Tau A
deficit (4 / 14 / 34 %) tracks the DTV's share of the 4-bit input power: quantisation and rail
damage on the loaded inputs in the recorded bits, not the method.

Findings on the way:

- Channels 24 and 30 each carry two transmitters. After the main one is removed the leftover
  mode (8 and 32 noise units) is uncorrelated with the main timestream at any lag (0.06) and
  with its cubic distortion (0.04), and peels as a single-path source: co-channel stations.
- The sky must be protected when the source is weak: peeling from the plain top eigenvector
  mixes with Tau A at near-equal eigenvalue (0.88 to 0.57 in channel 24). Starting each
  timestream orthogonal to the Tau A template fixes it. The faint sky needs no protection.
- MMSE refinement of the timestream (combining all inputs and lags) costs a few percent of
  Tau A per pass on weak sources (0.90, 0.85, 0.72 for channel 24's second source) and buys
  little; on strong sources it is harmless and tightens the residual (channel 30's second
  source: floor 4 without, 2.5 with). Refine only sources above about 20 noise units. The
  mechanism of the loss was not found.
- Stop peeling when the residual's top mode is at the sky level or matches the template; a
  third peel on channel 30 and a second on channel 17 start eating Tau A.
- The Green's function of channel 24's main transmitter: the dominant arrival is preceded by
  paths at -5, -10, -15 and -20 us (1.5-6 km shorter, together 60 % of the dominant power)
  and followed by weaker ones to +80 us, the same across the 6 MHz. The strongest path into
  the valley is not the shortest (diffracted direct path over the rim, hillside reflection).

## Stability of the Green's functions

Fitted on chunk 0 and applied frozen at 40 and 80 s (own templates at each epoch): the main
transmitters' g change by 0.4-0.8 % in power, the strongest taps 1.00 +- 0.02 in amplitude
and within +-2 deg, and the frozen subtraction equals the refitted one (channel 24 at 80 s:
7.1 / 5.6 / 4.7 frozen vs 7.2 / 5.6 / 4.8 tracked). Tau A kept is the same at 0 / 40 / 80 s:
0.91 / 0.90 / 0.89 (ch 24), 0.98 / 0.97 / 0.98 (ch 17), 0.71 / 0.69 / 0.69 (ch 30). The weak
second sources' g estimates vary 15-45 % on their +-1 lag taps (estimation noise), without
effect on the residual.

## Proposed live pathway

The expensive step, the eigen-decomposition, is only the initialisation, once per channel and
off-transit; afterwards nothing needs a sky template.

1. **Slow state per DTV sub-channel (per GPU: ~12 sub-channels of the 2-3 strong ATSC
   channels, more if the weak ones are wanted):** g_i(tau), 128 x 49 complex, and the combining
   weights derived from it. Initialised from the eigen-beam + least squares, then tracked by
   the cross-correlation of each input against the current timestream at the 49 lags (51 MMAC
   per channel per frame, ~15 GMAC/s per GPU) with an EMA over seconds and the 49 x 49
   Toeplitz solve on the host once a second. Static over 80 s to < 1 %; the tracker covers
   refraction and the odd vehicle.
2. **Per frame (8192 hops):** timestream s = sum_i sum_tau w_i(tau) x_i(t + tau), residual
   r_i = x_i - g_i * s, requantised to 4+4 bits in place in the voltage ring before
   cudaCorrelatorDual: ~100 MMAC per channel per frame, ~30 GMAC/s per GPU, well under 1 ms on
   an L40S. It is the beam-form kernel of cudaGnssBeamDump with lags, a lagged subtract and a
   requantiser. The second source is the same two kernels on the residual.
3. **A cudaCommand in the gnss<g>_n2dual process** before the correlator (one stage: lagged
   beam, lagged subtract, requantise, lag correlator for the tracker), a host thread for the
   initial eigen start and the per-second solve, generator plumbing like
   `--beam-dump-freq-ids`, a REST hook to freeze / re-initialise, then A/B with the beam dump
   and recv1 subset files (the N2 Tau A beam in the DTV channels with the stage on and off).
4. Limits: the 4-bit damage in channel 30 (noise at 0.8 LSB, 34 % of Tau A lost in the
   recorded bits) is not recoverable downstream; the F-engine gain in those channels is the
   lever. Re-quantising the residual adds 1/12 LSB^2, negligible except there, where the
   residual could be rescaled since the DTV is gone.
5. The same stage covers the 600 MHz (Band 71) and 700 MHz (Bands 12/13/17) LTE downlinks:
   fixed towers with stable g per cell and bursty scheduled power, which the per-frame
   timestream handles. Phone uplinks (699-716 MHz) stay out.

A visibility-domain variant (subtract g g^H times a per-frame power estimate from the N2
matrix) is cheaper but removes only the expectation, leaving the DTV's self-noise (about 2x
the thermal variance in channel 24); the voltage version removes the realisation and costs
little more.

Open items before building: the lag cross-correlation between the two peeled sources'
timestreams (confirms distinct stations), the direction fit of each arrival with the dish
positions, the cause of the MMSE loss on weak sources, and whether the g tracker's EMA time
constant can be minutes (test on a longer capture or two dumps hours apart).

## Test version (2026-11-03)

KV: "dig in on implementing a test version ... try this on a couple of channels". Jim agreed to
exempting the test channels from first-stage RFI excision.

**What it is.** `cudaDtvSubtract` (lib/cuda/cudaDtvSubtract.{hpp,cpp}, kernels in
cudaDtvSubtractKernel.{hpp,cu}) is `cudaCopyToRingbuffer` with the subtraction added: it replaces
the copy-in of `run_send_voltage`, so the N2 correlators, the RFI statistics and the GNSS
stages all read the subtracted voltages. Per frame and selected channel it decodes the channel,
forms each source's timestream s = sum_i w_i x_i (lag 0), accumulates R_xs,i(lag) and R_ss(d)
on the GPU, subtracts sum_lag g_i(lag) s(t - lag) and re-quantises the residual in place. The
host fit is not in the node: `scripts/gnss/dtv_state.py fit` solves the Toeplitz normal
equations from the recorded correlations and writes a state file; the node loads it by REST.
It is the single-fit recipe of the study above (eigen-beam start, one least-squares fit of g, no
MMSE refinement), with up to four sources peeled in turn.

**Controls** (REST, on the command, e.g. `/run_send_voltage/gpu_0/commands/0`):
`POST start {"source": k, "frames": n}` (eigen start of source k on the residual of sources
< k: covariance over n frames, whitened top eigenvector, phase-referenced to the strongest
input; the source then observes), `POST load {"file": path}` (weights and impulse responses
from dtv_state.py; a record subtracts all its sources), `POST clear {"source": k}` (sources k
and above off: pass-through), `GET status`. Its per-frame output (weights, per-input powers
before and after, |s|^2, the correlations) goes through `gnss<g>_dtvdump_gate` to
`<record_dir>/dtv`, armed like the beam dump's gate. Generator: `--dtv-subtract-freq-ids`,
`--dtv-sources`, `--dtv-state-file`, `--dtv-lag-min`, `--dtv-n-lag`; stock_parity.py declares
the replaced copy-in.

**Validation.**
- `cuda_dtv_subtract_test` (synthetic frame, known impulse response): decode, covariance and
  powers exact; beam, R_xs, R_ss to 1e-4 of a CPU sum; the eigen start finds the direct path
  (coherence 0.94-0.96); the host fit and GPU subtraction leave 1.07 x the noise (the excess is
  the re-quantisation); the re-quantised bytes match a CPU subtraction; other channels untouched.
- `run_dtv_subtract.yaml` (gpu_batch, through the kotekan binary): pass-through without state; a
  unit state (s = input 0, g = 1 at lag 0 on input 0) zeroes input 0 exactly and leaves the other
  127 exactly unchanged, and the ring consumer behind it sees input 0 at zero power; start,
  load, a bad load (400), clear and status all behave during a run.
- Replay on dump chunk 0 through the same kernels (`~/gnss/bb1030/dtv_replay.{cu,py}`,
  `dtv_replay_run.sh`, `dtv_replay_b3src.sh`): eigen start or protected weights on frames 0-9,
  g fitted there, scored on frames 10-19 (out of sample). Eigenvalues in noise units, Tau A
  kept as above:

| channel | raw | stage's own starts, 2 sources | protected starts, 2 sources | protected, 3 sources |
|---|---|---|---|---|
| 2512 (ATSC 17) | 8.5 6.2 5.5 4.9, kept 0.94 | 1 source: 5.7 5.0 2.4, 0.78 | 5.9 5.3 1.8, 0.96 | - |
| 2720 (ATSC 24) | 46 20 14 11, kept 0.74 | 5.2 4.4 1.7, 0.75 | 5.5 4.9 1.7, 0.87 | 5.5 4.9 1.6, 0.87 |
| 2912 (ATSC 30) | 147 126 81 66, kept -0.28 | 4.4 3.9 3.4 2.8, 0.55 | 4.5 3.9 3.4 2.7, 0.56 | 4.4 3.5 2.2 1.9, 0.57 |

  The protected column reproduces the offline study (0.96 / 0.82-0.89 / 0.66), so the GPU stage
  does what dtv_blind.py did. Channel 30 needs a third source; channel 17 has one station. Tau A
  in channel 30 is limited by the 4-bit damage already in the recorded bits (above).
- GPU time: 0.28 ms per channel per frame with two sources (3 channels x 2 sources: 0.84 ms of
  a 41.9 ms frame on an L40S).

**Rules the replay set.**
- Run every eigen start with no bright source in the beam. With Tau A in the beam an
  unprotected start mixes it into s: channel 17's single source kept Tau A at 0.78 instead of
  0.96, and a second start there took Tau A itself (0.24). Live, start at night and away from
  the Tau A transit (and other bright transits); the weights and g are static (above), so a
  night-time solution serves the day.
- Choose the source count per channel (`dtv_state.py fit --freq`): an eigen start covers every
  channel on the GPU, but a source fitted where there is no station projects a sky mode out.
- The last 8 samples of each frame miss the negative-lag terms (s after the frame's end is not
  yet known): about 0.1 % of the samples keep the precursor paths' share of the DTV.

**Live test (cx51, needs KV's go and the node cycle; after the 11-03 02:22Z C57 test).**
cx51 holds residues 7 and 13, so the test channels are 2727 and 2733 (ATSC 24) and 2903 and 2909
(ATSC 30), currently excised 8-34 % of the time by the SK flagger. Manifest, `nodes: cx51:`:
`dtv-subtract-freq-ids: "2727,2733,2903,2909"`, `dtv-sources: 3`,
`rfi-excision-exempt-freq-ids: "5939-6626,2727,2733,2903,2909"`,
`n2-project-live-freq-ids: "5939-6626"` (it follows the exemption set otherwise and would put
the GNSS projection live in the DTV channels) and the four ids added to `beam-dump-freq-ids`
(the beam dump's whitened lambda1 of the ring is the per-frame judge of what remains). Then, off
transit: start source 0 on both GPUs (24 frames), arm the dtv gates for a minute, `dtv_state.py
fit --source 0`, copy the state to the node, load; source 1 the same with `--state`; source 2
with `--freq 2903,2909`. Judge in minutes: residual / raw power (dtv_state.py info), the beam
dump's lambda1 in the four channels (raw 46-147, clean ~1.5 off transit), the SK flag fraction.
At the Tau A transit compare the four channels' N2 with clean neighbours. `clear` reverts to
pass-through without a restart.

**Live canary (cx51, 2026-11-03 01:10-01:45Z, off transit).** Stage on channels 2727 and 2733 (ATSC 24) and
2903 and 2909 (ATSC 30), driven over REST from cf06 (`~/gnss/bb1030/dtv_live.py`, `dtv_fleet.sh`). Top whitened
eigenvalue of the ring per 42 ms frame (the beam dump in the same channels) and residual / raw power:

| channel | raw | source 0 | sources 0 + 1 | power left |
|---|---|---|---|---|
| 2727 | 27.4 | 8.0 | 4.2 | 0.53 |
| 2733 | 37.6 | 6.9 | 3.9-4.5 | 0.50 |
| 2903 | 27.4 | 17.4 | 5.2-5.6 | 0.21 |
| 2909 | 23.9 | 14.3 | 5.6-5.9 | 0.21 |

The power left is about each channel's non-DTV share; the live fits explain the same fractions as the replay
(0.45-0.50 and 0.75 for source 0, 0.03-0.06 for source 1) with the same arrival lags. A third source explains 1 %
live and does not help. The ATSC 30 response drifts: with frozen g the top eigenvalue rose from 5.6 to 7.2 in about
ten minutes and a refit of source 0 restored it, so the test runs a host refit loop (source 0 every 5 minutes,
source 1 every 20, none of source 1 near the Tau A transit). Source 0's fit-to-fit change is 2-4 %, near its noise;
source 1's is 20-90 %. The per-second lag-correlation tracker of the pathway above is what replaces the loop.
