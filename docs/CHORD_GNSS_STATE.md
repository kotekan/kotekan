# CHORD GNSS — state of play, 2026-07-30

Working state of the CHORD-side GNSS instrument on branch `kv/chord-gnss`.

**Where we are:** the pipeline runs end to end on real sky with zero frame loss, the broker seeds
nine GPS L5 satellites from BRDC, and the acquisition search is correctly configured. The two costs
that made the search unusable are **fixed** (§5): a pass over 9 hinted PRNs was ~14 min and ~15 GB
and is now **~1.4 min and ~150 MB**. A real numerics bug in the shared replica turned up on the
way, and an 8-hour hang was waiting on the first detection. **No detection yet: the fixed search
has not been run against sky.** That is the next thing to do, and the first number to read off it
is the instrumental delay constant (§6).

Read `config/chord_gnss_node.yaml` first — it is the source of truth and every measured number
lives there. This file is the narrative: what works, what broke, and what is left.

---

## 1. What is built and validated

| Piece | State |
|---|---|
| `GnssChordVoltageTap` | tap `host_voltage_buffer` → `[hop][chan][elem]`; owns the valve/drop behaviour |
| `launch_waveform` + `launch_correlate_nm` | split replica generation from N×M correlation; **exact-equality gated** vs the fused kernel |
| `cudaGnssChordTrack` | new cudaProcess command; model-primary, no frozen-seed machinery, no peel |
| element axis | through the record schema, assembler and combiner; `n_elements: 0` reproduces the airspy layout byte-for-byte |
| `GnssChordDequantize` | 4+4b → cfloat32 for the search; optional zero-fill (off) |
| `GnssChannelizedSearch` | now accepts a **sparse** comb via `channel_ids` |
| broker | CHORD model-primary path: `--time0-endpoint`, `--dr-clock-chips`, optional `--detectors` |
| config generator | production config as BASE + injected GNSS branch; separate search instance |
| `--check-config` / `--dry-run` | config and pipeline-graph validation, incl. the stall check |

Two kernel gates, both **exact** (`0.000e+00`), re-run as regression after every change:
split-vs-fused at N=1, and the element axis at N=5/stride=7 with poison padding.

## 2. Proven on sky

* **Ingest is lossless at production rate.** 391.4 kpkt/s per port (production runs 388.5),
  `rx_missed` 0 in steady state, `ring_full_dropped` 0, `gnss_tap_dropped_frames_total` **0**.
* **The tap reads real sky.** Peeked 1,835,008 bytes = exactly 8192 hops × 7 chan × 32 elem;
  real/imag means ≈ 0, rms 1.81 per component (the 4+4b contract's intended band), range
  [−7,+7], and **zero occurrences of nibble 0** — matching the paper's clip to ±7 and proving
  byte alignment.
* **Records are correct.** 52,228 B = 32 PRN × 408 floats (24 header + 32×12 element blocks),
  finite, at the predicted rate (4 records/frame ÷ `output_every` × 2 GPUs = 19/s).
* **The broker seeds.** Nine GPS L5 PRNs from BRDC every cycle, code phases in [0,10230),
  Dopplers within ±2.8 kHz, and the tracker's reported Doppler tracks the seed (PRN 32:
  2025 Hz vs seeded 2133).

## 3. Bugs found and fixed (all mine unless noted)

1. **`f_offset` must be the SKY CARRIER, not 0** — the big one. `ChannelizedReplicaBank` places
   the replica at `f_offset`, and `covering_bins()` sets `local.carrier_hz = f_offset`. With 0
   the replica sat at DC (bins −51..52) while the data is at 5971..6076: **zero overlap**, every
   correlation noise. On the airspy node `f_offset` is the post-mixer IF because that receiver
   downconverts; CHORD does not downconvert at all. Fixed on both the search and the tracker.
   **It failed silently in two disguises** — the search reported "carrier not in this subband"
   (indistinguishable from an empty sky) and the tracker reported `amp_snr 0` with a railed
   `dll_disc` (indistinguishable from an unknown instrumental delay, which cost a whole cycle
   chasing cable lengths).
2. **Pruned an ingest stage.** `process_packet_mask` was dropped as "N²-only"; it produces the
   packet-loss mask `TransposeBasebandArray` REQUIRES. The transposes blocked forever with DPDK
   happily filling its input buffer. `run_send_voltage` was the inverse error — kept when it
   should have been dropped, and would have stalled from the other side.
   → `--dry-run` now fails on any buffer left with consumers and no producer.
3. **Wire-format mismatch on the search link.** `bufferSend`/`bufferRecv` default
   `use_config_tracker` to whether the *instance* has a `/config_tracker` block, and the two
   instances differed. Header fields shifted; surfaced as "Frame size does not match between
   server: 57344 and client: 12" where both sides were really 57344 and 12 is
   `sizeof(GnssChanMetadata)`. Now pinned explicitly on both ends.
4. **`hint_ttl_s` 8 s vs a 10 s broker refresh.** With `require_hint` that skips every PRN
   *before* its SNR is computed, so the stage goes silent for a reason nothing in its output
   explains. The airspy chain sets 30 s; now so do we.
5. **Broker variable shadowing** — my `t0` clobbered the loop's cycle timestamp, so
   `time.sleep()` got 1.8e18 s and it died after one (successful) cycle.
6. **Nibble convention** — I stated low-nibble-real from `s12_kernel.cu`'s naming; that kernel
   only forms `re²+im²` so its labels are unobservable. Authoritative is `gpuSimulate.cpp`:
   **HIGH nibble = REAL**. Getting it wrong conjugates the data — magnitudes and acquisition SNR
   unchanged, Doppler sign inverted.

**Pre-existing, not ours:** `cuda_gnss_despread_test` exits 1 on a trial-reorder invariant
(1.914e-07), confirmed present before our changes. `restServer::set_server_affinity` segfaulted
on an unstarted thread (fixed, worth upstreaming).

## 4. Things I got wrong about the architecture, corrected

* **The search does NOT need contiguous channels.** `channelized_accumulate` FFTs along the
  **hop** axis within each channel and sums per-channel surfaces — it is "the distributable half
  of the search", built for scattered channels. I conflated the snapshot's existing gap
  tolerance in TIME with a supposed requirement in FREQUENCY. The zero-fill I added was
  unnecessary (kept, off by default). The 640 ns comb ambiguity is a property of the
  measurement, resolved by the BRDC model with ≥16× margin.
* **Search products are ~38× the raw voltage** (26.4 MB of surfaces from 700 kB of data), which
  is why the design ships **raw voltage** to a central aggregator and runs ONE search over the
  gathered union — not a search per node. At eight nodes that union is contiguous (all eight
  mod-8 offsets present), so full sensitivity and no aliasing.

## 5. RESOLVED 2026-07-30 — the search was ~600x too expensive (was THE blocker)

Two independent costs, in the order they were found rather than the order they mattered.

### The replica: banded and precomputed

Measured at CHORD scale (Mp = 3125,
fft_len = 16384, N = 8192, the node's 14 covering channels):

| | per PRN | per pass (32 PRN x 2 instances) |
|---|---|---|
| was: `channels()`, full spectrum, every pass | 0.87 s, 204.8 MB | **55.7 s, 13.1 GB, every pass** |
| now: `channels_hoprate()`, banded, cached | 0.22 s, 0.35 MB | **14.3 s, 22 MB, ONCE** |

Three separate changes, in increasing order of how much they mattered:

1. **Banded.** `channels_hoprate()` already existed and already took a `want` channel list —
   no new interface, which keeps the merge story clean. `channelized_accumulate` reads only
   `repl0_ch[covering[ci]]`, so the non-covering rows are left as empty vectors; that is now
   gated (`accumulate_reads_only_covering_rows`, poison + empty, bit-identical surfaces).
2. **Cached.** `repl0` is the code at Doppler 0 and code phase 0, so it depends only on
   (PRN, covering set, Mp) — none of which move between snapshots. The covering set comes from
   the fixed blind grid, not the per-PRN hint, so the broker's 10 s hint refresh does not
   invalidate it. Keyed on the covering set anyway rather than asserting that from a distance.
3. **The refine loop was worse than the acquire.** It called full-spectrum `channels()` once per
   1-sample step over +-1 hop: at fft_len 16384 that is **32769 builds ~ 8 hours per detected
   PRN**, and it would have fired only on the FIRST DETECTION — i.e. it would have looked like a
   hang at exactly the moment the thing finally worked. Now banded, with the Doppler-only filter
   hoisted out of the scan (it is constant across it) and the data columns hoisted too, and
   `refine_step` config-driven: **225 steps, 40 s**. The step is set from the physics — only the
   covering channels enter the despread, so it is band-limited to `n_chan * 195.3 kHz` and cannot
   resolve better than `fft_len/n_chan` ~ 2341 samples ~ 7.5 chips; the generator emits
   `refine_step: 292`, one eighth of that. Stage default stays 1 (correct for airspy's fft_len 20).

### The bug: one filter tap on the wrong side of a chip boundary

`channels_hoprate()` was documented as equal to `channels()` "to ~machine precision". At CHORD
scale it was not: **1.8e-3 rms over a replica period, and up to 1.99 absolute on isolated hops
against a typical |R| of 912 (-53 dB)**. Against a double-precision reference, `channels()` is
exact to 3.7e-8 and the hop-rate path was not — so `channels()` was right.

Cause: chip `d` owns the taps `k` in `((phi+d-1)/cps, (phi+d)/cps]`, and evaluating that bound in
floating point puts the boundary tap on the wrong side whenever `(phi+d)/cps` lands on an exact
integer. That is **systematic, not luck**: `cps = chip_rate/sample_rate` is rational (1023/320000
on CHORD L5), so `phi` is a multiple of 1/320000 and the tie is a congruence in `d` that hits
about 8 times per replica period. The partition stays exact — `klo(d+1)` is literally
`khi(d)+1`, so no tap is lost or double-counted — the tap just gets its **neighbour's code sign**,
costing `2*proto[k]`, which is why every bad hop was off by very nearly the same ~1.9.

Fixed by snapping the boundary to the direct per-sample expression `floor(cp0 + (n_m - k)*cps)`,
which is what `channels()` (and therefore the F-engine) uses. Now 3.4e-8 vs the double reference —
marginally better than `channels()`, since the hop-rate path accumulates in double instead of
through a float32 FFT. The per-chip tap range and code value also moved out of the channel loop
(they never depended on the channel), which is where most of the 3.9x came from.

**How it hid**, both ways worth remembering:
* At airspy scale it is ~1e-5 rms, right under the existing 1e-5 tolerance.
* The first affected hop at CHORD scale is **hop 255**. My first version of the CHORD-scale test
  used 12 hops — because `channels()` costs O(n_hops * 16384) and 12 felt sufficient — and passed
  at 6e-8. The test now runs a full replica period at zero Doppler and checks the **worst single
  hop**, not just the rms: 1.99 absolute out of 3125 hops is only 1.8e-3 in rms, so an rms-only
  bound over a long run hides exactly what a long run was added to catch.

**Still divergent, deliberately: the GPU path.** `chip_gather` in `cudaGnssReplicaDevice.cuh` is a
heavily optimized form of the SAME unsnapped formula (its own comments record 9 boundary
disagreements over 16e6 (phi,d) pairs, judged benign at airspy scale). It has NOT been snapped —
that is optimized device code whose float32 integer/fraction split exists precisely to avoid a
double floor per chip, and the effect is -53 dB on isolated hops, i.e. invisible to acquisition
and tracking. So the CHORD **tracker** (GPU) still carries the one-tap error while the **search**
(CPU) no longer does. `cuda_gnss_despread_test` still passes GPU-vs-CPU at 5.3e-07 and the CHORD
split gates are still exact, so nothing regressed. **This matters for peeling, not for us yet** —
a -53 dB replica error floors the peel depth, which is the one place the "machine precision"
claim would have been leaned on. Flag it to the core dev with the merge note.

### The bigger half: the acquisition surface was 16x oversampled

The replica turned out to be about **1% of pass time** — it was the memory churn (204.8 MB
allocated and freed per PRN, 19.4 GB RSS). Measured, the dominant cost was
`channelized_accumulate`: at 5 Doppler bins, **91 s per PRN per pass and a 2.05 GB surface**, so
**13.7 min** for 9 hinted PRNs. That matches the observed 8-minute passes far better than the
replica ever did.

`D[s] = sum_ci P_ci e^{+i2pi f_ci s/sph}`, and the surface keeps only `|D[s]|^2`. Factor out the
first channel's ramp — unit modulus, so it drops out of the magnitude — and what is left depends
only on the channel **differences**. If those share a factor `g` with `sph`, the fine-lag axis is
exactly periodic with period `sph/g` and the other `g-1` copies are bit-for-bit identical.

For a combed band `g` is large. The node's comb is stride 8, split across two GPUs, so **each
search instance sees stride 16 and g = 16**: fifteen sixteenths of both the compute and the 2 GB
surface was recomputing numbers it already had. (Airspy's covering set is stride 2, so g = 2 —
a free doubling there too.)

This is the SAME redundancy as the code-phase ambiguity already documented in §4: period `sph/g`
samples is `1/(g * channel_width)` in time, 320 ns at g=16, resolved by the BRDC model with ~16x
margin. Nothing is lost that was ever independent information.

**Exact, not approximate**, which is why it is safe: `channelized_peak` takes the first *strict*
maximum, so among `g` identical copies it already returned the `s` in the first period — the peak
cell is unchanged; and its noise floor is a mean over the surface, which is unmoved by dropping
`g` uniform copies of every value. `AcquisitionSurface` now carries `s_stored` alongside `sph`:
index by `s_stored`, but form the absolute delay with `sph` (tau is still `q*sph + s`). Gated by
`fine_lag_period_is_exact`, which checks the stored surface against the *formula* in double
precision and then checks that `channelized_peak` reports an identical peak cell, Doppler, SNR and
code phase off the reduced surface and off a hand-tiled full-width one.

| per PRN per pass (nd=5) | time | surface |
|---|---|---|
| before | 91.4 s | 2.05 GB |
| after | 9.4 s | 0.13 GB |

**A pass over 9 hinted PRNs: 13.7 min → 1.4 min.** Combined with the cached replica, the search
is affordable to run continuously.

### Why the replica was 1170x, kept for the arithmetic

```
N = 8192 (CHORD's PFB spans 0-1600 MHz), Mp = 3125
  per PRN              204.8 MB   (8192 channels x 3125 hops)
  x 32 PRNs              6.55 GB
  x 2 search instances  13.11 GB   (observed RSS 19.4 GB)
we used 7 of 8192 rows = 0.09%   ->  1170x waste, in memory AND compute
```

On the airspy node `N = 10`, so the full spectrum *is* the useful set and there was nothing to
notice. This was the single reason the search could not keep up here: it explains the 8-minute
passes, the "336 passes then nothing" burst, and the final starvation (1 pass in 5 min with
workers idle). It would have mattered MORE at full scale, not less -- the eight-node aggregator
wants 104 of 8192 bins, still 79x.

`gnssChannelizedReplica` is shared GNSS-side code. The banding needed no interface change (the
`want` overload already existed); the boundary snap changes shared numerics and must be flagged
on merge -- see `cudaGnssChordTrack.hpp`'s note for the shared/duplicated split.

## 5b. FIRST LIVE RUN OF THE FIXED SEARCH, 2026-07-30 12:3x-12:45

The speedups hold on sky: `precomputed banded repl0 for 32 PRNs x 7 channels x 3125 hops in
3.0 s (5.6 MB; the full-spectrum form would be 6553.6 MB)`, logged once per instance, and passes
now cycle in seconds instead of minutes.

**THE TRACKER IS DOWNSTREAM OF DETECTION GATING — a trap I walked into.** The broker builds its
tracker seeds by iterating `best`, which is populated only from detections passing
`--acquire-snr` (broker line ~1247, seed loop ~1546). So with the search threshold at the 12.0
default, nothing detected, `active=[]` every cycle, and the trackers were seeded with an EMPTY
list. Records were still produced at the right size and rate, with correct PRN numbers and
capture-UTC — and every measurement slot zero. 96 nonzero floats out of 13057, all of them PRN
indices and the UTC double. **A record file that looks structurally perfect and is empty.**

Yesterday's config carried `acquire_snr: 0.2`, which made every PRN "detect" and hid this
entirely; regenerating the config restored the 12.0 default and switched the measurement chain
off. I chose 12.0 deliberately that morning, reasoning only about not paying the refine cost on
noise peaks, and missed the gating dependency completely.

**The right fix is `--dr-clock-chips 0.0`, not a lowered threshold.** That flag exists for
precisely this circle -- its own help says "THIS IS WHAT LETS A NODE WITH NO SEARCH STAGE
COLD-START. The bootstrap needs --dr-min-sats satellites already tracking to take a median of
their residuals, but nothing can track until it is seeded." With a GPS-disciplined F-engine the
clock offset is known a priori, so priming it lets dead-reckoning seed every visible sat from
BRDC with no detection at all. It was written yesterday FOR CHORD and then left out of both the
command line and the documented run recipe in §7. It is now in §7 -- do not omit it again.

With it: **`active=[5, 6, 9, 11, 12, 18, 21, 25, 26, 28, 29, 31]` (12 sats)** and the trackers
produce real per-PRN correlations at model-predicted phases. Reported Dopplers follow the seeds
(PRN 18: 2467 vs seeded 2472; PRN 31: -188 vs -164), `nchan` 7 as configured.

### No signal yet, and the numbers say why we should not have expected one

* `amp_coh` is 10-25x BELOW `amp_incoh` for every one of the 12 (e.g. PRN 31: 2.07 vs 33.6).
  That is the noise signature: a coherent average decays as 1/sqrt(N) while the incoherent one
  sits at the noise floor. `deep` is 0.0000 for all 12.
* `dll_disc` rails past +-0.9 on half of them, with E/L power ratios up to 230x (PRN 21:
  17.5 vs 3996). Not noise-like (noise gives balanced E/L) and not lock-like.
* **We applied ZERO instrumental delay.** `--dr-clock-chips 0.0` means the seed carries no
  delay term, while the cable alone is 4.26 +- 0.18 chips and the F-engine framing/PFB offset is
  unknown on top (one frame = 52.4 chips, PFB group delay = 104.8). The DLL captures +-0.5 chip.
  So the seeds are certainly outside capture, and a railed DLL at an uncorrected multi-chip
  offset is what that looks like.
* Search-side, the threshold is miscalibrated for this surface, independently of the above. With
  32-window incoherent integration the surface is Gamma(32), so sigma/mean = 1/sqrt(32) = 0.177,
  and the max over ~80M cells lands at about mean*(1 + 5.5 sigma) ~ 2.0. Observed best-SNR is
  **2.1-2.3 and the winning PRN wanders (5, 18, 12, 12)** -- textbook noise. A threshold of 12
  cannot fire at this integration depth no matter how strong the signal; it is calibrated for
  airspy's far smaller single-window surface. A real bar here is ~2.8-3.

### Next step, and why it is the search rather than a sweep

The delay constant could be found by sweeping `--dr-clock-chips` and watching `amp_incoh`, but
the plausible range is 0-160 chips (cable + framing + group delay) and each setting needs seconds
of integration. The acquisition search covers **all 10230 chips at once** -- that is what it is
for, and it is now cheap enough to run continuously. So: set `acquire_snr` to ~2.8, raise
`acquire_windows` for sensitivity (the accumulate is 16x cheaper, so this is now affordable), and
let the search report the code phase. Delay constant = reported code phase minus model
prediction. Expect to need real integration: only 7 of 106 covering channels are used, which is
-11.8 dB before any consideration of where a parked dish's sidelobe points.

## 5c. THE COHERENT WINDOW WAS MISCONFIGURED THREE WAYS, 2026-07-30 13:00-13:20

Chasing sensitivity found three independent losses, all from the same root: **the coherent window
length is not a free parameter.** `hops_per_record` defaults to the replica period = 3125 hops =
**16 ms**, and everything else has to be consistent with that.

1. **NH20 overlay off: -12.7 dB rms, with exact nulls.** L5 Q5 is a dataless PILOT, which does
   not mean unmodulated -- it carries the NH20 secondary overlay, one +-1 chip per 1 ms code
   period. A 16 ms window spans SIXTEEN code periods, and the replica was built with the overlay
   off (`nh_phase` defaults to -1). Summing 16 chips of a near-balanced sequence, computed over
   all 20 alignments: **-12.7 dB rms, best case -8.5 dB, and EXACTLY ZERO for three of twenty.**
   Worse, it is not a fixed penalty -- consecutive 16 ms windows step the alignment by 16 mod 20,
   so a snapshot only ever visits phases {0,4,8,12,16}, and phase 4 is one of the nulls. A fixed
   fraction of every snapshot contributed nothing.
2. **Doppler step 250 Hz against a 62.5 Hz requirement.** A coherent window of length T cannot
   tolerate a Doppler error much past 1/(2T). At 16 ms that is 31 Hz; the grid step was 250 Hz,
   so a half-bin miss is 125 Hz = **two full cycles of phase rotation across the window**, i.e.
   a sinc null. The Doppler grid resolution and the coherent window length are ONE parameter.
3. **`--search-margin-wide-hz 3000` is airspy conservatism.** Its help says it "covers the
   unknown TCXO offset" -- but CHORD's F-engine is GPS-disciplined and the `gpsdo` clock profile
   is 0.06 ppm = **+-71 Hz at L5**. There is no unknown TCXO. This was the dominant cost driver,
   and narrowing it to +-150 Hz is what made a 31.25 Hz grid affordable at all (11 bins, vs 25 at
   the old 3000/250 and vs 194 if we had kept the wide margin).

### `nh_search` (new, default OFF)

`GnssChannelizedSearch` can now search every secondary-code alignment and keep the best peak
(`nh_search: true`; `_n_nh = secondary_length()`). Alignments are processed SEQUENTIALLY so peak
memory is one surface, not twenty. The precompute is 20x the banded repl0: **112 MB in 57 s,
once** -- worth stating that the full-spectrum form of the same thing would be **131 GB**. The
post-detection refine uses the winning alignment (`hoprate_stream(..., best_nh)`), or it would
despread an overlay-blind replica against an overlay-aligned peak. Default off keeps airspy and
any signal without a secondary code bit-identical.

12.7 dB won COHERENTLY would need ~350x more incoherent windows to match, so 20 acquires is a
strongly favourable trade -- but it IS 20x, and the pass cost is real: 20 alignments x 8 windows
x 11 PRNs ran >2 min/pass. Going deep on the three highest-elevation sats (PRN 29 at 65 deg,
5 at 53, 25 at 52) with 16 windows is the better iteration loop AND the more sensitive
measurement.

### Result: still no detection, and the threshold arithmetic says why it looked like one

Best-SNR came in at **3.0-3.4 against a 2.90 bar** -- above threshold, and NOT a detection:

* **The winning PRN wanders** (29, 5, 5 on one instance; 25, 5 on the other), and the two
  instances name DIFFERENT PRNs for the same sky at the same time. A real satellite is in both.
* **The winning NH alignment jumps randomly for the same PRN** -- PRN 5 came up at nh 0, then 7,
  then 14. The alignment is a deterministic function of GPS time; it cannot hop like that. This
  is the sharpest single discriminator the alignment search bought us, and it says noise.
* **My threshold was wrong, in the direction that manufactures detections.** The bar
  1 + c/sqrt(k) is for ONE surface. `nh_search` takes the max over TWENTY, which raises the
  effective noise ceiling (c goes ~6.0 -> ~6.5 through the sqrt(2 ln N) growth): the real bar at
  k=16 is ~2.6-3.0, not 2.75. The observed 3.0-3.4 sits exactly on it. **A detection threshold
  must be recomputed whenever the number of surfaces searched changes** -- otherwise adding a
  search dimension silently converts a null result into a positive one.

### What is left, in order

The three losses above are fixed. The remaining structural one is not a bug: **only 7 of the 106
covering channels are on this node**, which is -11.8 dB, and no amount of local integration
recovers it. That is precisely what the eight-node aggregator is for (all eight mod-8 offsets
present -> the union is contiguous and complete). Beyond that, the honest unknown is where a
PARKED dish's sidelobe points for any given satellite; the user's goal is the full hemisphere
response including sidelobes, so low levels are expected rather than surprising.

So: (a) recompute the detection bar for n_nh surfaces and make the stage log the bar it is
actually using, (b) integrate deeper now that a pass is affordable, (c) scale to a second node and
use the aggregator -- the -11.8 dB is the biggest single term left and it is architectural.

## 5d. THE AGGREGATOR IS BUILT AND LIVE, 2026-07-30 14:30-15:30

The -11.8 dB structural term is now addressed the way the design intended: ONE search over the
gathered union of nodes' combs, not a search per node.

* **`GnssChanAlignMerge`** (new stage): N cfloat32 feeds -> one [hop][sum chan] stream, aligned
  on `GnssChanMetadata::sample_seq`. Alignment is the entire point: the send legs run
  `drop_frames: true` (acquisition must never back-pressure a node's science chain), so inputs
  lose frames independently, and the existing `GnssChannelGather` -- which is INDEX-LOCKSTEP and
  never reads sample_seq -- would silently combine different epochs from the first drop onward.
  sample_seq inherits from the F-engine's GLOBAL fpga_seq_num, so equality IS simultaneity and
  the union search is coherent across nodes. Emits only epochs present on ALL inputs;
  `gnss_merge_skipped_frames_total{input=}` counts what alignment discarded. A feed without
  GnssChanMetadata is a FATAL error, because guessing is the failure mode the stage exists to
  prevent.
* **`--aggregator-instance cx19 cx27`** in the generator: per-feed recv+dequantize, the merge,
  and one search over the union, with the search block factored into `search_stage()` shared
  with `--search-instance` so the two modes cannot drift. Feed i listens on search_port_base+i,
  node-major GPU-minor -- exactly where the node configs already send, so the NODE configs did
  not change at all.
* **Union geometry (cx19+cx27):** 27 channels, stride-4 comb, 5972..6076. +5.9 dB over one
  node's 7; -5.9 dB vs the full 106 cover (all eight nodes closes it, and makes the union
  CONTIGUOUS). The fine-lag reduction still applies at g=4: s_stored 4096.
* **The aggregate is now threaded** (`acquire_threads`, default 1 = the exact serial path,
  bit-identical -- gated). Necessary, not nice-to-have: the union surface is ~16x a single
  node's (4x channels x 4x stored lags), ~10 s per window on one core, which x windows x NH
  alignments x PRNs is HOURS per pass. Parallel over Doppler bins (disjoint surface slices);
  the aggregator config grants 6 cores. Pass time at 3 PRN x 20 nh x 16 win: **~7 min**.

**Proven live:** `first aligned frame at sample_seq 1542426640515072 across 4 inputs (27
channels merged)` -- cx19 local + cx27 over the network, one search, 27 covering channels,
precompute 40.5 MB/27 s (the full-spectrum form would be 12.3 GB).

**Result: still noise.** Best-SNR 3.11 vs a 3.40 bar, winning PRN and NH alignment wander pass
to pass. That is a clean null at 256 ms integration, -5.9 dB of full cover, on whatever gain a
parked dish's sidelobe offers at 1176 MHz. Levers, in order of expected value: more nodes (the
remaining -5.9 dB, plus contiguity), deeper integration (windows scale sqrt), and if pass cost
ever binds again, the NH factorization (per-code-period partial correlations combined with 20
sign patterns would amortize the 20x alignment cost to ~1x -- an algorithmic project, noted not
planned).

## 5e. THE BEAM-CROSSING NULL AND THE GAMMA CEILING, 2026-07-30 evening

The dishes are parked on the meridian for the Cyg A transit: boresight (az 180, el 81.4), beam
FWHM ~2.9 deg at L5. Computed from BRDC: **PRN 32 crossed 2.1 deg from that boresight at 19:41
UTC** -- a +40 dB event if the pointing assumption and the analog budget hold. We watched it
with 4-window (~3 min) passes from 25 deg off-axis all the way through the crossing.

**Result: nothing.** The pass whose snapshot started 19:39:41 (dead inside the FWHM) reported
best snr 6.74, indistinguishable from every pass before and after.

**And the "detection plateau" leading up to it was my own miscalibration.** Twenty consecutive
passes at snr 5.9-7.3 over a 5.0 bar looked like a persistent weak detection. The bar was set
with the Gaussian-ish heuristic 1 + c/sqrt(k); the honest ceiling is the max of ~N Gamma(k)
cells, solving  k*x - (k-1)*ln(k*x) + ln((k-1)!) = ln(N):

    k = windows    heuristic bar    Gamma ceiling    measured noise maxima
        16             ~2.7             3.4              3.11 - 3.40
        64             ~1.8             1.94             1.86 - 1.92
         4             ~4.7             ~7.5             "plateau" 5.9 - 7.3

The Gamma model reproduces BOTH deep-run measurements to a few percent; at k=4 the heavy tail
sits 50% above the heuristic, and the plateau was BELOW the true ceiling the whole time. Flat
snr from 25 deg to 2 deg off-axis, wandering nh, Doppler bouncing anywhere in the drifting hint
window -- noise, end to end. The stage now computes and logs the ceiling every pass and flags a
threshold set below it ("every 'detection' is meaningless"), so this cannot be silently done a
third time (the acquire_snr 0.2 era was the first).

**What the null MEANS:** with the software chain validated by synthetic injection (5a) and the
channel map verified to the packet copy, a no-show at 2 deg off assumed boresight leaves two
explanations: the park pointing is not (180, 81.4) -- the user's "I think ~10 deg off zenith"
was hedged -- or the analog path toward those sats delivers far less than the sidelobe budget
(feed/filter/pointing unknowns at 1176 MHz). Next discriminators: PRN 3 passes 5.3 deg off the
assumed boresight at 23:23 UTC (watched at 32 windows with the honest bar 2.8 vs ceiling
~2.5); the element-0 node cycle; and a boresight FIT -- run crossing watches on several sats
and fit where the snr actually peaks, which measures the true pointing instead of assuming it.

## 5f. FIRST LIGHT, 2026-07-30 evening -- and the conjugation that hid it

**Two, then five-plus GPS satellites detected on CHORD dishes through the L5 despread.** The
finding chain, because each link was necessary: the aggregate GNSS glow was in the GAIN TABLE
(+2.4 dB, sinc^2-shaped, centered on the L5 carrier bin -- not DME); cross-element correlation
showed the same glow SKY-COHERENT (|r| ~ 0.1) in exactly the channels the search consumes,
proving the channel map with the sky itself; the user then argued the brightest satellite HAD to
be despreadable; offline acquisition on captured data with convention variants found it in two
runs: **the F-engine output is CONJUGATED relative to the nominal decode** (equivalently
imag-high nibbles; indistinguishable, and both invisible to the |.|^2-only X-engine, so
gpuSimulate was never load-bearing evidence, and invisible to any self-consistent synthetic
test). PRN 32: 10.1 (noise) as-is -> 22.5 conjugated; Dopplers then match BRDC to ~6 Hz on two
satellites with a common +5.8 Hz = the GPSDO's frequency bias, measured for the first time
(5e-9, well inside spec). A firmware-derived float translation of the FFT says standard forward
convention; the deployed bitstream disagrees -- flagged to the F-engine team. All four feeds
share the one convention.

Deployed: `conjugate` on GnssChordDequantize (search path) and DespreadParams::conj_data through
the NxM kernel (tracker path), both default-off so airspy and every exact-equality gate are
untouched; the generator sets them on all CHORD legs.

**Post-fix live behaviour:** passes detect 5-6 satellites at SNR up to 108 against honest
Gamma ceilings. Two operational lessons the first hours taught: (1) the post-detection REFINE
runs per DETECTION, so cost appears exactly when the search starts working -- at refine_step
resolution/8 it was ~5 min per detected sat and pushed "8-minute" passes to 30+; now step =
resolution/2 (~1 min/sat, ~10 min full passes). (2) Detection STALENESS: the broker compares
frozen measurements against a drifting model, so every residual consumer needs pass cadence <<
Doppler/code drift times; the ±3000-chip "integrity failures" and the -80 chips/s "clock drift"
were stale-input artifacts, not physics.

**The delay measurement (direct, bypassing the broker):** undo the sample-0 currency (validated:
reproduces the stage's raw coarse+refine EXACTLY), compare against BRDC t_sv = t_gpst - range/c
+ clk at the shared snapshot epoch. Within one pass, satellites agree to 0.02-0.03 chips (!) in
clusters; between passes the common mod-residual shifts by the RECEIVER CLOCK code-rate
(+5.8 Hz carrier bias = 0.05 chips/s = ~70 chips per 23-min pass, aliased mod the 13.0944-chip
comb ambiguity -- this aliasing also explains the within-pass "bimodality" of the first pass).
So the observable is delay + clock_rate*t: a two-parameter solve the broker's machinery was
built for, needing only fresh detections (now available). The raw residuals box the absolute
delay at ~100-155 chips; the tooth-quantized center ~140 chips ~ 2.7 F-engine frames + the
4.3-chip cable -- physically sensible pipeline depth.

**CHORD-scale caveat for the currency layer:** the search's sample-0 cp referencing multiplies
Doppler error by the FULL F-engine uptime -- 4412 chips/Hz at 5.75 days (airspy: hours, and 13x
smaller chip/carrier ratio). The undo is exact when the same dop double is used, but any
consumer that mixes currencies pays the full lever. Prefer ref_hop-anchored cp end to end at
CHORD scale.

**Next session:** joint delay+clock solve from consecutive passes (or teach the broker); feed
--dr-clock-chips; watch the trackers lock (deep > 0) and the per-element beam-map values become
physical. Then: remaining six nodes (contiguous cover kills the comb ambiguity entirely),
element ranking by cross-element sky coherence (measured: element 0's coupling is ~half of
elements 3/9/10's), Galileo E5a + BeiDou B2a on the same carrier.

## 5g. THE LOCK CAMPAIGN, 2026-07-31 00:00-01:00 -- capture PROVEN, broker freshness next

**The DLL captures on command.** A direct sweep (bypassing the broker: /set_seeds driven at
cp_model(now) + clk_try in 0.4-chip steps) hit hard: PRN 9 at clk ~192.9 despread at amp_incoh
1483 against a ~30 floor (50x), PRN 26 at ~202.1 (418) -- classic capture-and-drag as the sweep
walked away. The tracker locks the instant the seed is right. Lock is not a sensitivity problem
anywhere in this system.

**What still blocks fleet-wide lock, in order:**
1. **Broker estimator freshness.** The dead-reckon clock, drift, and clock-freq-bias estimators
   consume the CURRENT detection table without an age gate (only the seeding path checks
   ref_hop advance). With minutes-long passes, 20-40-min-stale detections poison every solve
   within seconds: the "+518 Hz clock-freq bias" and the clock EMA walking off the primed value
   were both this. Fix is mechanical: age-gate (ref_hop epoch vs now) the inputs to the offs
   loop and the bias residual loop -- a --dr-max-det-age-s. The seeding freshness guard
   (det_fresh) shows exactly the pattern to follow.
2. **Per-sat cp model bias ~9 chips in MY quick python model** (P9 vs P26 sweep centers) --
   likely its missing relativistic SV-clock / TGD terms; the broker's own cp_predicted (proven
   0.01 chip on airspy) should not share it, but VERIFY against the sweep-measured clks before
   trusting fleet seeding.
3. Search cp precision: the fast refine (step 303) scatters cp +-2-3 chips; the calibration
   refine (span 4096 = +-1 tooth, step 75) gives +-0.3 at the same 109-step cost and should
   simply become the default for the aggregator.

**Primed-broker plumbing that now exists:** --dr-clock-chips <clk> --dr-clock-drift <chips/s>
(drift priming added 2026-07-31; without it a slow-cadence search pins drift to zero on stale
repeats and the clock freezes between passes).

**Session artifacts** (logs with every detection line = the calibration dataset, fit/sweep
scripts, captured voltage frames, gain snapshot): `~/gnss/session_artifacts_20260730/`.

**Overnight state:** aggregator searching (precision-refine config), plain hint broker (no lock
attempts), viewer live on 8080/8539 with detections. Tomorrow: the age-gate patch, re-verify
per-sat cp under the broker's model, prime, lock, then wire the viewer's combiner columns to
12049.

## 6. Also outstanding

* **Instrumental delay is still unmeasured.** The cable term is now well determined —
  100 m LMR-400 (vf 0.85) + 6 m LMR-195 (vf 0.83) = **4.26 ± 0.18 chips**, comfortably inside
  the ±0.5-chip DLL capture. What is NOT known is the F-engine's internal pipeline/framing
  offset (one frame = 52.4 chips; PFB group delay = 104.8 chips, and whether it cancels because
  the replica goes through the same PFB is an argument, not a measurement). The search is meant
  to measure this: its reported code phase minus the model prediction IS the constant. Once
  measured it goes in `chord_gnss_node.yaml` and dead-reckon cold-start works permanently.
* **`rawFileWrite` makes one file per frame** — 19/s ≈ 1.6 M files and 86 GB/day. Fine for a
  test, wrong for a soak. Raise `output_every` or use an appending writer.
* **Frequency axis collapse** is an early priority once there is proof of life — see the
  `roadmap` block in `chord_gnss_node.yaml`. BOC signals make it critical, not cosmetic.
* **A grinding search worker ignores SIGTERM** (it cannot check `stop_thread` mid-FFT), so it
  holds its REST port and the replacement instance fails to bind and exits silently. Use
  `kill -9` when restarting the search.

## 7. How to run it

```bash
# search instance -- ordinary user, no DPDK/GPU/hugepages, fully dry-runnable
./build/kotekan/kotekan --config config/generated/chord_gnss_search_cx19.yaml \
    --bind-address 0.0.0.0:12050

# node instance -- needs sudo for DPDK hugepages (/dev/hugepages is root:root, and
# /dev/vfio/{19,38} too, so the group route needs BOTH; sudo is simpler for debugging)
sudo ./build/kotekan/kotekan --config config/generated/chord_gnss_cx19.yaml \
    --bind-address 0.0.0.0:12049

# broker -- note -u (python buffers stdout when piped and you lose everything on kill), and
# note --dr-clock-chips: WITHOUT IT the trackers are seeded with an empty list and every record
# comes out structurally perfect and numerically zero. See 5b.
PYTHONUNBUFFERED=1 /home/kvand/gnss/venv/bin/python -u \
  python/scripts/gnss/gps_distributed_broker.py \
  --rest-url http://localhost:12049 \
  --detectors http://localhost:12050/srch0_search,http://localhost:12050/srch1_search \
  --trackers gnss0_track,gnss1_track --combiner gnss0_combine \
  --almanac --almanac-source brdc --dead-reckon --narrow-search \
  --time0-endpoint telescope/time0_ns --dr-clock-chips 0.0 \
  --constellation G --carrier-hz 1176.45e6 --code-length 10230 --hops-per-sec 195312.5 \
  --lat 49.32075144444 --lon -119.62081125 --alt 545 --mask-deg 0 --interval 2
```

Ports: **12048** production (choco owns it; the generator REFUSES to emit it), **12049** node,
**12050** search, **11040/11041** the raw-voltage feed. Always validate with
`sudo ... --dry-run` before a real run — it is the only way to exercise the full graph including
DPDK, and it is what would have caught the pruned-ingest-stage stall.

**Regenerate, never hand-edit:**
```bash
python3 config/gen_chord_gnss_config.py --base <base.json> --node cx19 \
    --search-element 23 [--search-instance] --out config/generated/...
```
where `<base.json>` is `curl -s http://cx19:12048/config` from a node running production.

## 8. Operational notes

* Cores: DPDK owns 5-7 and 21-23 (0% idle) plus their isolcpus siblings — never touch. Our
  stages use NUMA1: 19, 24, 31, 57-63.
* `/home/kvand` is NFS shared across all eight nodes at the same path (code, config, venv);
  records must go to local disk (`/data/gnss`, 2.6 TB free).
* venv at `/home/kvand/gnss/venv` (skyfield 1.54 + numpy/scipy/h5py/yaml/requests).
* `kvand` has sudo (password), and is in `docker`. Passwordless is NOT configured.
* choco has a per-node **maintenance mode** ("choco will not push"). It must be ON while we run,
  or choco may push a production config over us mid-run.
* **Cyg A transits ~07:2x UTC**, 81.5° elevation, drifting 3m56s earlier daily. Several dishes
  use it for beam maps and noise temperature; keep the X-engine healthy ±4 h around it.
* Eight nodes: cx19(4) cx27(0) cx42(5) cx43(6) cx44(2) cx47(3) cx51(7) cx52(1), where the number
  is the mod-8 comb offset. Together they tile the science band exactly.
