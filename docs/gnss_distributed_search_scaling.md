# Distributed GNSS Search — Data Volume & What Crosses the Wire

A note on *why* the distributed channelized search merges centrally, how much
data that actually is, and how it degrades when part of the band is missing.
Companion to `gnss_chord_framework.md`; grounded in the current stages
`GnssChannelGather` → `GnssChannelizedSearch` (central acquire over gathered covering
voltage), `GnssChannelizedTracker` / `GnssCoherentCombiner`, and the broker. (The
earlier comb-the-`P_c`-surface variant was retired — see below.)

## Two different "merges" — don't conflate them

- **Across antennas/dishes.** Acquisition does **not** need this: one feed already
  has ample SNR from the code gain (it's what a phone does). Only *measurement*
  uses all antennas, and it folds into the N² correlator (see the framework note).
- **Across frequency channels.** A GNSS carrier is wider than one PFB channel
  (a ~2 MHz code ≈ 10 channels; an L5 code ≈ 100), so its power is spread over
  many channels. Despreading is a **coherent sum over those channels**,
  `G = Σ_c ⟨X_c, R_c⟩`. This merge is unavoidable for *both* acquisition and
  measurement, because the per-channel signal is **below the noise floor** — no
  single channel (or node) can see the satellite alone. You don't get to skip it;
  you only get to choose what crosses the wire to do it.

## The catch: a correlation *surface* is bigger than the data

It is tempting to push the correlation to the edge (each node correlates its own
channels) and ship the result. For **measurement** that is great — the despread
contribution per channel is one complex `G_c` + one real `E_c`. For **search** it
is the opposite, and this is the non-obvious part:

The acquisition product is a matched-filter **surface** over hypotheses
(PRN × Doppler × code-lag), not a single number. Per channel, per snapshot:

| object | size (per channel, per snapshot) |
|---|---|
| raw channelized voltage (the *input*) | `nwin·hpr` complex |
| `P_c` search surface (the *output*) | `n_prn · nwin · nd · Mp` complex |

The surface is **`n_prn · nd` times larger than the raw data it came from**
(`Mp ≈ hpr`). For the prototype (2 PRN, 25 Doppler) that is **50×**; for a real
32-PRN search it is ~**800×**. Concretely the prototype ships ~800 kB per 195 kHz
channel per 8 ms snapshot ≈ **0.8 Gbps/channel** — ~16 Gbps across a 20-channel
band, ×constellations into the 100s of Gbps. That does not scale, and the reason
is simply that **`P_c` is an expansion of the data, not a compression.** The raw
channelized voltage is the minimal sufficient statistic; anything derived from it
to search is larger.

So: shipping the full `P_c` surface cross-node is *strictly worse* than shipping
the raw voltage. The prototype does it anyway because in one kotekan it is all
local RAM (~free); it let us validate the stages, the central refine, and the
closed-loop lock without a network. It is a **validation scaffold, not the
cross-node transport.**

## What actually crosses the wire

| mode | when | per channel | who pays |
|---|---|---|---|
| **Track** | continuous (dominant) | `(G_c, E_c)` per record per *tracked* PRN, ~12 B/ms | combiner gather |
| **Acquire** | rare, duty-cycled | **raw** covering-channel voltage (the 50–800× *smaller* number) | gather to search node |
| ~~Surface~~ | — | ~~full `P_c`~~ — never cross-node | — |

This matches the framework note's choice: **remap the ~5 MHz GNSS sub-band onto
one node and search locally**, rather than comb-distribute the search. It also
sharpens that note's "ship the *tiny* partial correlations" line — true for the
measurement `G_c`, but the *search* partial correlations are the expansion above,
which is exactly why acquisition is remapped (or single-antenna), not combed.

Tracking is the steady state and is genuinely tiny: ~10 sats × ~20 channels ×
12 B/ms ≈ **3 MB/s** total into the combiner. Acquisition ships raw covering
voltage only while actively searching: at 195 kHz × (4-bit complex) ≈
0.2 MB/s/channel, a GNSS band is sub-Gbps and duty-cycled.

## Missing channels degrade gracefully (a node down for maintenance)

Because every step is `Σ over the *available* covering channels`, each tagged with
its global frequency index, a missing channel is simply a missing term — no
special handling, and the prototype already does this (the aggregator combines
whichever channels report `covers=1` each snapshot; a node dropping just removes
its frames from the sum).

- **Measurement / refine** (exact despread, `A = ΣG_c / ΣE_c`): missing channels
  lower SNR in proportion to the code power they held, but the estimate stays
  **unbiased and auto-renormalized** (the `ΣE_c` denominator only counts present
  channels). No bias, no breakage — just reduced sensitivity.
- **Acquisition coarse fine-lag** (`D(s) = Σ_c P_c(q) e^{i2πcs/sph}`): a gapped
  channel set is a sparse aperture, so the fine-lag reconstruction raises
  sidelobes / blurs, much like missing antennas in an array. The *peak* survives
  while enough channels remain; the central **refine then cleans it up** (the
  exact despread has no aperture-sidelobe issue — gaps only cost it SNR). The
  two-stage coarse→refine design is what makes acquisition gap-tolerant.

Caveat worth tracking: at low SNR with *large* gaps the coarse sidelobes could
seat the coarse peak wrong; lean on the refine, and report sensitivity over the
*present* channels (the amplitude calibration is already correct, the SNR budget
just shrinks). Most of the band present → negligible impact.

## Almanac collapses the search dimension

The cost above is for a *blind* 2-D (Doppler × code-lag) scan. In practice the orbit
removes the expensive axis: the broker predicts each sat's Doppler (geometry + a common
clock-frequency bias) and pushes it to the search (`--narrow-search`), which then scans a
**tight Doppler window** instead of the blind grid — wide until the first sat solves the
common bias, then narrow. Acquisition becomes a code-phase pin at the known Doppler, the
same cheap targeted despread as tracking; on a disciplined clock (known offset) even the
code phase is predicted and the blind search disappears. So the search isn't just *remapped*
cheaply — with the almanac it is *small*, and shrinks further toward CHORD's no-blind-search
reality. (The acquisition surface is still bigger than the voltage, so it still gathers raw
voltage centrally rather than combing `P_c` — the volume argument above is unchanged.)

## Status

- **Search compute + gather:** done & validated. `channelized_acquire` operates over an
  *arbitrary, possibly gapped* covering-channel subset (channels carry explicit
  `chan_freq`); the two-stage coarse→refine locks on real sky data; and the raw-voltage
  remap — `GnssSubbandSplit` → `bufferSend`/`bufferRecv` → `GnssChannelGather` →
  `GnssChannelizedSearch` — is the live `live.yaml` path. The snapshot is drop-tolerant
  (zero-fills gaps at the true `sample_seq` offset).
- **Almanac-narrowed acquisition:** built (`set_doppler_hints` + broker `--narrow-search`),
  the robust + forward-compatible path (above).
- **Not the plan:** shipping the `P_c` surface cross-node. Prototype scaffold only.
