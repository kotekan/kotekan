# GuoWang beam dump: a plan sketch

Status: 2026-10-31. KV: build it. Tier 1 is implemented on branch kv/guowang-beam (this document's
branch): `cudaGnssBeamDump` (lib/cuda), its kernel test `cuda_gnss_beam_dump_test`, the generator
option `--beam-dump-freq-ids` (fleet key `beam-dump-freq-ids`) with the gate and writer, and
`scripts/gnss/beam_dump_read.py` (info, stitch), and `scripts/gnss/beam_dump_sched.py` (plan,
run: arms every gate on one seq window per pass from the TLE planner, as viscap.py does). Not
yet: the fleet key in config/gnss_fleet_chord.yaml and a node canary (both need KV and the
array back).
Analysis so far: `~/gnss/bb1030/dg_*.py`, data `cf06:/var/tmp/bb1030/dg2`, notes in the gnss-main
session memory (`chord-guowang-encoding-20261031`).

2026-11-06: the eigen beam is gated. `cudaGnssBeamDump` forms it only for the frames its gate
(`beam_eig_gate`, the `gnss<g>_beamdump_gate` or `gnss<g>_trackbeam_gate`) would pass, with
`beam_eig_resume_iters` power iterations on the first frame of a window; the other frames carry
zeros in the eigen sections and `eig_iters` 0 in the header. `--beam-dump-eig-mode off|gated|always`
sets it at generation, `POST <instance>/eig_mode {"mode": ...}` changes it live, and
`GET <instance>/eig_status` reports it. The target beams are formed every frame.

## Why

The 10-30 baseband dump gave us the GuoWang (HULIANWANG DIGUI) signal parameters from one satellite
and 80 s. To learn the code family, its length, the data rate and whether satellites differ, we want
coherent baseband from many passes, without a 129 GB dump per 0.84 s each time.

## What the dump told us (chunk 52 and 61, DIGUI-26)

Superseded 2026-11-03: the chip rates, the offset-QPSK reading and the non-coherent chip clock
below were artefacts of conjugating the channels in the stitch. The signal as established with the
correct stitch is in CHORD_GUOWANG_SIGNAL_20261103.md (10.23 / 2.046 Mcps, one clock, a readable
2046-chip code per satellite on the 1202 lobe). The instrument description and the rail caveats
stand.

- Two lobes, carriers 1268.5483 and 1201.9735 MHz (after the TLE Doppler; the TLE predicts the
  carrier to 1-2 Hz over 7.5 s).
- Chip rates 10,278,000 and 2,055,600 Hz transmitted (10.23 and 2.046 x 1.00469, ratio 5), both
  offset-QPSK-like. The 2 ms frame is 20556 chips of the 10.278 MHz clock. The chip clock is not
  coherent with the carrier: +1.5 ppm against the carrier between 10:59:64 and 11:00:11Z.
- 1202 lobe: 2.000 ms frames; one 1 ms half carries a frame-locked, narrow (+-1.25 MHz) preamble-like
  pattern with one sign per frame; the 2.0556 Mcps chips themselves do not repeat within 0.84 s.
- B3 lobe: no repeat at 2 ms, no chip autocorrelation out to 420 ms (long code or data).
- Both lobes look Gaussian rather than like one PSK stream, at every rail level (chunk 66, 2 % rail on
  the 1202 lobe: chip bimodality 0.39 where a single OQPSK through the same chain gives 0.76 and a
  Gaussian signal 0.33; envelope kurtosis 1.9-2.2). Each lobe is most likely a sum of several
  synchronous codes on one chip clock. Chip-by-chip reading will not recover the codes even unclipped;
  the code study needs correlation methods over long spans, which is what the beam dump feeds.
- At closest approach 80 % of the samples sit on the F-engine's 4-bit rail. Beamforming 128 clipped
  inputs does not restore the constellation (simulated and measured). Chips are readable only where
  the inputs are off the rail: pass edges, or satellites 2-4 deg off boresight, where the 128-input
  sum still gives +15 to +25 dB per chip.

## What the instrument has to produce

Per target satellite: the coherent sum over the 128 inputs, per channel, at the channel rate
(195.3125 kHz, 5.12 us), for the B3 lobe (coarse 6395-6594, ~200 channels) and the 1202 lobe
(6122-6185, 64 channels). Nothing finer is needed on the nodes: each GPU holds every 16th channel,
so the wideband stitch, the carrier wipe, the chip clock and the chip reads happen offline on cf06
with the existing scripts. Output 8+8 bit complex: 4.9 MB/s per target per GPU, ~6 GB per 80 s pass
over the 16 GPUs.

## Tier 1: a beam dump on the nodes (about a week)

1. **Kernel.** A cudaCommand in the GNSS cudaProcess, where the voltages already are:
   `y_c(t) = sum_i w_ic x_ic(t)` for the configured targets over that GPU's lobe channels, written
   8+8 bit to a host buffer. ~200 lines of CUDA; the pathfinder config runs no baseband beamformer,
   so nothing to reuse. Load is negligible (12.5 channels x 128 inputs x 195 kHz per target per GPU).
2. **Weights, two candidate sources.**
   - Tier-2 geometric lanes: the fleet-lane machinery already carries a per-satellite line of sight
     and rate to every node (ProjEntry los/los_rate, ElemSteer::track); add the per-pol shared gains.
     Designed, not built. Works when the satellite is weak.
   - The projection stage's eigen path: the top eigenvector per channel per frame is the satellite's
     steering vector when it dominates, phase-referenced to one input (what the offline analysis
     used). No TLE needed, self-calibrating, but only while the satellite dominates the channel.
3. **Track feed.** The pass planner (`guowang_passes.py`) decides the targets and windows; the lane
   relay carries the los/los_rate to the receivers as it does for fleet lanes.
4. **Writer and transport.** A per-GPU file per pass on local disk, shipped to cf06, stitched and
   read with `dg_npz2.py`, `dg_lines.py`, `dg_carrier.py`, `dg_oqpsk.py`, `dg_code1202.py`.
5. **Plumbing.** Generator entries (target list, channel ranges), a stage gauge, one node as the
   canary, judged against an offline beam of the same pass.

## Validate first, offline, half a day

Form a TLE-steered tier-2 beam from the existing dump and compare its chip stream with the eigen
beam: that measures the coherence loss of geometric weights on this satellite and picks the weight
source before any node code is written. Also test the eigen-path weights' phase continuity across
frames (the offline beam used 42 ms windows with a reference-input phase).

## Caveats

- The 4-bit rail: useful data comes from pass edges and off-boresight satellites, or from a
  campaign-time ~10 dB F-engine gain cut on the lobe channels (costs our B3I and E6 chains).
- The chip clock wanders against the carrier; any chip reader needs its own clock recovery (the
  offline tools have it); a future tracker chain needs an independent code-rate loop.
- This produces coherent baseband per pass; reading the code stays offline until the code is known.
  A GuoWang chain in the tracker (10.278 Mcps, long code, non-coherent chip clock) is a later tier.
- Data volume: ~6 GB per pass per target; ~10 passes per hour are visible, so passes are selected,
  not recorded blindly.

## Decision points

- Weight source (after the offline validation).
- Whether to ask for the F-engine gain cut during a campaign.
- How many passes and satellites before a code-level answer is likely (depends on whether the B3
  code is a long code or data; the first unclipped pass tells).
