# CHORD GNSS — state of play, 2026-07-31 (evening)

Working state of the CHORD-side GNSS instrument on branch `kv/chord-gnss`.

**READ THIS FIRST — the picture changed completely on 2026-07-31.** For weeks the tracker could
not lock on anything. The cause was found this evening and it was not tuning, calibration or
the clock: `GnssCudaDespread` synthesized every replica at `chan_offset + local`, i.e. a
CONTIGUOUS subband, and CHORD's tap delivers a STRIDE-16 comb. With `chan_offset = 0` every
CHORD replica was built at global bins 0..6 — DC — against data at 1176 MHz. **Every tracker
correlation since first light was noise at every code phase** (§5k). Fixed in `51b1ca034`; the
channel-id list is now the only API (`cae8cad2d`), so no front end can inherit that assumption.

**Where we actually are:**
* The tracker DESPREADS REAL SATELLITES. PRN 32's transit gives ratio 49.8 through the
  tracker's own path, with the 3.27-chip grating lobes of the single-GPU comb resolved (§5l).
* The search has been healthy throughout — snr up to 1084 on a transit — and was never
  affected by the DC bug (it passes global channel ids).
* **NO LOCK YET.** One thing blocks it: the code phase the search reports and the code phase
  the tracker wants are in different currencies. An anchor bug in the search is proven (§5m),
  and a further 391.26-chip acquire-vs-despread gap on IDENTICAL data is unexplained.
* **START HERE NEXT SESSION:** extend `scripts/acq_conv.cpp` to push ONE injected synthetic
  signal through both the acquire and `GnssCudaDespread`, and fix whichever misplaces the
  known phase. Pure software, no sky, ~10 minutes. The despread side is already proven exact
  (+0.000), so the acquire is the prime suspect. Do NOT run more live sweeps first — two
  returned exact noise tonight and cost hours.
* **HOLD the F-engine clock escalation** (§5h/§5j): the "±98 ns / 20 s breathing" was measured
  through a correlator that was despreading noise, and the record-geometry bugs explain the
  same pattern. Re-measure with the trim once tracking works.

Six other real defects were found and fixed on 2026-07-31, each of which the DC bug had been
masking: the missing nominal code advance per record and NH20 intra-record cancellation
(`d37064e87`), the assembler's element-count energy offset (`2527e0298`), a lock statistic that
provably could not see a lock (`db4c0fa45`), host-clock record stamps (`f2d2e9dd3`), and a
silent no-op in the hint-gated search (`83e5110b0`). A 52-chip hop-truncation bug in every
seeding script I wrote is fixed in `scripts/trim_lock2.py` (§5j).

Read `config/chord_gnss_node.yaml` for measured constants. §§5i–5m are this day's narrative,
newest last; §6 lists what remains.

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

## 5h. MORNING 2026-07-31: the clock BREATHES, and the trim port is the fix

**Chronology of the morning's findings, each one real:**

1. **Overnight hint poisoning (+347 Hz), fixed and verified.** The unpatched broker's bias EMA
   ingested stale detections when the nodes resumed (det_fresh stamped wall-clock novelty, not
   data epoch); its hint windows sat ~+350 Hz off truth and every morning "detection" echoed
   the poison back (the cp scatter came free via the 4412-chips/Hz sample-0 lever). Patched:
   det_fresh now stamps ref_hop epoch (single point, all consumers honest, restart-proof);
   drift EMA frozen when primed. First honest pass: SNRs straight back to 63-128.
2. **THE CLOCK-CHAIN OSCILLATION -- the morning's big find.** Full-rate record streams (19/s,
   not the 1-in-45 subsample that faked "decay") show despread bursts: CONSECUTIVE strong
   records (median gap 0.11 s) in ~0.5-1 s bursts every 15-25 s, cyclically, with NO
   re-seeding. That is a PERIODIC PHASE OSCILLATION, amplitude ~+-1 chip (+-98 ns), period
   ~20 s -- a steering/synthesis-loop signature (GPSDO 1PPS discipline or the RFSoC's
   10 MHz -> 3.2 GHz PLL), NOT oscillator jitter. The TM-4-class OCXO should sit sub-ns; the
   user is escalating to the F-engine team ("got to be a bug"). Burst-timing extract for them:
   `~/gnss/session_artifacts_20260730/clock_oscillation_bursts.txt`. NOTE: if the breathing is
   real at the ADC clock it matters for VISIBILITIES too, not just GNSS.
   It retro-explains: calibration-pass cp scatter (+-2-3 chips = oscillation sampled at random
   phase), the hold-test flicker, one-shot ~10 s holds, and the external-DLL failure (REST
   latency 1.5-3 s can't follow +-0.2 chips/s slew).
3. **The clock does NOT accumulate.** The same absolute clk values captured on BOTH nights
   (sweep centers ~191-201 last night AND this morning): the GPSDO steers phase to GPS =
   bounded, so last night's "+0.044 chips/s drift" was loop wander over a short baseline, not
   a rate to extrapolate. Overnight extrapolation is unnecessary AND wrong.
4. **Per-sat split ~9.5 chips, stable across days** (P9 192.9 / P26 202.1 night; P25 191.1 /
   P24 200.7 morning -- same two clusters!). A systematic in MY quick python model (suspects:
   SV-clock relativistic term, TGD), not noise. The broker's own cp_predicted disagrees with
   my model by ~3000 chips AND churns cycle-to-cycle (-0.3..-0.9 c/s apparent) -- reverse
   engineering its currency from log echoes was abandoned; reconcile by READING cp_predicted
   when needed. My +5.8 Hz "clock bias" may be partly common-mode model error too.
5. **Capture is routine** (sweep bites at 441-2919 inc vs ~30 floor, on command, both nights).
   Only HOLD is missing, and the reason is now precise:

**THE FIX: port the in-tracker DLL trim into cudaGnssChordTrack.** The airspy tracker holds
locks because the trim closes the code loop per record ON the tracker (19 Hz laughs at
+-0.2 c/s breathing); cudaGnssChordTrack computes the E/P/L discriminator but NEVER APPLIES
it -- the model-primary design deliberately dropped the frozen-seed machinery ("if CHORD ever
needs the fence, port it knowingly" -- cudaGnssChordTrack.hpp header). CHORD needs the trim
(not the whole fence), knowingly.

### Trim port: DONE 2026-07-31 (post-compaction)

Implemented in `cudaGnssChordTrack.{cpp,hpp}`, config key `code_trim` (default OFF = the old
behaviour, bit-identical; the generator turns it on for CHORD). Both node configs regenerated
-- diff vs deployed was trim-keys-only, so nothing else moved.

* **What it is:** the broker's 3c DLL math (disc = (E^2-L^2)/(E^2+L^2), tau = -disc/4 *
  (spacing/0.5), leaky integrator, clamp +-3 chips), run PER GPU FRAME (~24 Hz) inside the
  command instead of ~1 Hz in the broker. Observable = coherent channel sum of the tracker's
  own E/P/L rows at `trim_ref_elem` (default 0) -- exactly the assembler's header rows, so the
  sign convention is inherited from the airspy-proven broker loop, not re-derived.
* **Plumbing:** a CUDA command never sees its results on the host, so each execute() enqueues
  a ~20 kB strided D2H of its E/P/L rows (ref element only) + an event; a later execute()
  consumes completed slots non-blockingly before building specs. One frame (~42 ms) of loop
  latency against the 20 s clock breathing.
* **Application:** `cp = model + trim[p]` at the single Spec-construction point; the trim
  rides on top of whatever the broker/scripts seed, and reported cp_seed includes it (records
  and broker see the true commanded phase). Quality gate q = 2|P|^2/(|E|^2+|L|^2) (~1 noise,
  ~4 locked; `trim_quality_min` 1.8) keeps unlocked PRNs from walking.
* **Watching it:** GET `/gnss{0,1}_track/get_trim` -> {prn, trim_chips, disc, quality,
  updates}; plus a "code trim:" INFO heartbeat every ~10 s in the node log.
* **Knobs:** trim_gain 0.15, trim_leak 0.002, trim_clamp 3.0 per update at ~24 Hz
  (steady-state residual = leak/gain * trim ~ 0.013 chips/chip of static bias).
* **Cautions:** (1) run the broker with `--dll-gain 0` against a trimming CHORD node -- the
  in-tracker loop nulls the disc so the broker loop mostly idles, but during transients both
  integrate the same discriminator; (2) the trim does NOT reset on broker re-anchor (the
  broker pops its dll_trim on re-anchor; ours persists) -- fine for hold_seeds/one-shot
  driving, revisit if broker-primary seeding jumps anchors; (3) despread exactness gates
  verified green post-port (CHORD split + element axis 0.000e+00; the 1.914e-07 trial-reorder
  failure is pre-existing).
* **Next live step:** sudo dry-run both node configs, restart trackers, sweep-capture one PRN
  (sweep_clk.py; P24 200.72 / P25 191.12 in my-model currency), then watch get_trim: the trim
  should breathe +-1 chip with the ~20 s clock oscillation while inc/coh stay high and deep
  goes > 0 -- that is LOCK HELD THROUGH THE BREATHING, the thing this port exists for.

### Original trim-port worklist (for reference)
* REFERENCE: lib/stages/gnss/cudaGnssTrack.cpp (airspy; ~12 'trim' sites). UNDERSTAND FIRST:
  where the airspy tracker computes its code trim (from its own E/P/L within the command? or
  host-side between records?) and where it applies it (cp extrapolation per record).
* TARGET: lib/stages/gnss/cudaGnssChordTrack.cpp -- Spec construction ~lines 230-260 (the
  per-record seed -> Spec extrapolation, sp.spacing_chips already there); state on
  cudaGnssChordTrackState (per-PRN trim accumulator beside Seed).
* The discriminator today is computed downstream (GnssCoherentCombiner, CMB_DLL_DISC) -- the
  port likely wants the E/L powers already in the tracker's own output rows (corr rows E/P/L
  exist per record in the epl frame) so the trim can close inside the command WITHOUT waiting
  on the combiner. Airspy's answer to this is the thing to copy.
* Config-gate it (default off = current behaviour; CHORD generator turns it on) and keep the
  exact-equality despread gates green (trim off must be bit-identical).
* After trim works: single sweep-capture per sat (proven) -> trim holds through the breathing
  -> broker's lock-driven EMA takes the fleet; then viewer combiner columns (12049), then the
  per-sat model reconciliation, then multi-node scaling.

**Current running state:** aggregator searching (agg_cal config: precision refine span 4096 /
step 75, stage renamed gps_search), plain broker (hints only), viewer on 8080/8539. All
scripts + logs synced to ~/gnss/session_artifacts_20260730/. Sweep-proven per-sat clk values
(MY python-model currency, compute_clk.py/sweep_clk.py): P24 200.72, P25 191.12 (morning),
P9 ~192.9, P26 ~202.1 (night).

## 5i. AFTERNOON/EVENING 2026-07-31: the trim was necessary but NOT sufficient -- two structural despread killers found and fixed (d37064e87)

Chronology: trim port deployed and mechanically verified (loop runs, gate holds on noise)
-> records structurally dead -> assembler energy-offset bug fixed (2527e0298) -> still no
capture at any clk -> search revealed silently idle since 12:16 (require_hint + dead broker;
83e5110b0 makes it say so) -> hints-only broker revived it (passes to snr 615) -> P10 sweep
null -> element probe null -> OFFLINE AUDITS on a captured live frame (buffer REST
/buffer/gnss0_volt_buf/frame) found everything:

* **P10 at snr 40 in a single 42 ms frame, elements 0 AND 8** (cp consistent to 0.2 chips,
  dop matches model+5.4 bias to ~1 Hz). Data, conjugation, element 0, doppler: all fine.
* **KILLER 1 -- missing nominal advance**: records are 10.4857 code periods (airspy: exactly
  1); residual-only extrapolation dropped 4969.3 chips/record mod L. Every record despread at
  a pseudo-random offset; the isolated q-spikes in sweeps were REAL single-record alignments.
  Fixed: absolute extrapolation (52.3776 c/hop nominal + residual + quadratic code-Doppler).
* **KILLER 2 -- NH20 intra-record cancellation**: overlay flips every 1 ms (20 ms is the
  sequence period, not the transition spacing); ~10 overlay chips per record, partial sums
  mostly cancel. Fixed: GPS_L5_Q_NH (primary tiled 20x, NH baked, 204600 chips / 20 ms) --
  bit-identical to bare+nh_phase per alignment; trackers switch, search keeps the short code.
* **Currency map (verified)**: bank cp args are SAMPLE-0-referenced with internal FF; the
  search anchors repl0 at Mp*fft_len = exactly 16 periods (hiding the convention); the GPU
  despread is window-anchored. Seeding chain: cp_phys(t_det) = cp0 + off + drift (exact
  inverse of GnssChannelizedSearch.cpp:429-446), cp204(t) = cp204_model(t) +
  (cp_phys - cp_model(t_det)) + clk_try, NH period from GPS time, clk_try ~ 0 expected.
  A ~10-chip cross-audit residual remains unexplained -- the live sweep covers it.
* **CLOCK OSCILLATION RETRACT-LIKELY**: the burst pattern (consecutive strong records every
  15-25 s) is the alignment-walk + NH-beat signature of the two bugs, not a clock. HOLD the
  F-engine escalation. The fixed tracker's trim measures the actual clock phase directly.

**TO RUN (after restarting node kotekans on the d37064e87 binary + regenerated configs):**
`python trim_lock.py` (scratchpad; copy to session_artifacts) -- seeds every fresh search
detection through the verified currency chain, sweeps clk_try +-15, holds and logs the trim
on a multi-sat bite. Also: my python model needs TODAY's BRDC (stale-cache poisoning cost an
afternoon: P26 550 Hz off at 13:52, fine after the 14:05 refetch).

## 5j. EVENING 2026-07-31: the measurement that was fooling us, and what the null means

**THE METHODOLOGICAL ERROR (mine, and it cost the afternoon).** I steered every sweep by the
in-tracker trim's per-frame quality q = 2|P|^2/(|E|^2+|L|^2). That statistic CANNOT see a lock:
at 0.5-chip spacing E and L each carry R(0.5)^2 = 1/4 of the peak, so q saturates at 4 for ANY
signal strength -- 3.60 for our strongest satellite, 3.97 even at search snr 600 -- while its
own noise tail reaches 7. Every "bite" (clk 0.0, clk -0.25, NH k=7) and every "null" (the P10
clk sweep, the element probe, the NH scan) was noise read as signal or the reverse. Compute the
expected value of a statistic BEFORE running an experiment on it.

Fixed in db4c0fa45: the E/P/L powers are EMA-averaged (~1 s, trim_pow_alpha 0.05) before the
gate and the discriminator touch them -- q_locked stays ~3.6 while q_noise falls toward 1, and
the disc the trim integrates stops being a per-frame random number. This is exactly what the
airspy broker always did with the combiner's WINDOW-AVERAGED E/L; porting the formula per-frame
had dropped the averaging that made it work. Gate 2.2, warm-up 1/alpha frames, EMA reset on
re-seed. get_trim now reports ema_frames -- never trust a q whose average has not filled.

**THE SEEDING BUG (mine, in every seeder written this week).**
`hop = int((time.time() - FRAME0) * HPS)` truncates to an integer hop. A hop is 5.12 us =
**52.4 chips**, so the seed's claimed epoch and its cp's epoch differed by up to 52 chips,
RANDOM on every repost, against a +-0.5-chip capture. Measured live: 49 chips at that instant,
common-mode across satellites. This contaminated every clk calibration (Wednesday's 191/200
"winners" included) and is why sweep winners never reproduced. Fixed by evaluating the model at
EXACTLY FRAME0 + hop/HPS. Note also that float64 cannot hold GPS seconds better than ~4 chips
(ulp 2.4e-7 s at 1.8e9), so the code phase must come from the FRACTIONAL second only (carried
as an exact Fraction) -- legitimate because 1 s = 50 x 20 ms and GPS-UTC offsets are whole
seconds. Reference implementation: scratchpad/trim_lock2.py (model_at/hop_epoch).

**THE FIRST TRUSTWORTHY NULL.** With the EMA statistic and exact-epoch seeding, a +-4 chip
sweep in 0.5-chip steps over 6 satellites returned q = 1.0-1.3 EVERYWHERE -- exactly the
theoretical noise value, no excursions. So the commanded code phase is wrong by much more than
4 chips. Note a single GPU's 7 channels are 3.125 MHz apart, so its correlation has GRATING
LOBES every 3.27 chips under an envelope tens of chips wide (the 27-channel union: 13.09
chips). The sweep therefore crossed several lobes and still saw nothing -- the error is
hundreds or thousands of chips, i.e. a currency/convention error, not a calibration offset.
(Grating lobes are also a standing hazard for the DLL: locking one lobe off is full amplitude
at a wrong delay. More nodes -> denser comb -> the ambiguity opens out.)

**WHAT IS SOLID.** The delta between the search's measured cp and my model, per satellite:
P8 +136..140, P10 +151..155, P24 +134..138, P23 +149, P27 +155, P32 +148 -- a common
**~148 chip** term (the instrumental delay constant of section 6, first measurement) plus the
same +-9 chip per-satellite spread seen Wednesday. Stable to a few chips over 30+ minutes, so
the receiver clock's frequency offset is small (~0.002 chips/s). Weak (snr ~10) detections give
garbage deltas (P27 swung +155 -> -4381) -- only trust deltas from snr >~ 20.

**ALSO FIXED (f2d2e9dd3):** records were stamped with HOST wall clock (hdr->utc0 was 0, so the
assembler fell back to system_clock::now() at assembly time). Now frame0_utc from
telescope/time0_ns, in chord_gnss_node.yaml -- re-read after any F-engine restart. Chip-level
work still uses sample_seq; a double at 1.8e9 s resolves only 2.4 chips.

**IN FLIGHT:** scratchpad/cp_oracle.cpp brute-forces all 204600 chips of the NH code space on a
CAPTURED frame using the tracker's own GnssCudaDespread (stacking records at their correctly
advanced cp). Its answer, minus what the seeder would command for that same frame
(P10: 11335.40 at hop 113561591808, dop -51.6), IS the remaining bug. Structure of the residual
tells the story: ~0 = seeding right and the fault is live-side; multiple of 10230 = NH phase;
multiple of 4969.3 = record advance; else = the model/delta chain.

## 5k. THE LOCK BLOCKER, FOUND 2026-07-31 EVENING: tracker replicas were built at DC

`GnssCudaDespread` synthesized channelized replicas at `all_chans[c] = chan_offset + c` -- a
CONTIGUOUS subband, which is what the airspy node has. CHORD's tap delivers a **stride-16
comb** (cx19 gpu0: 5972, 5988, ... 6068), and `cudaGnssChordTrack` passed `chan_offset = 0`.
So every replica was built for global bins **0..6 -- DC** -- while the data sat at 1176 MHz.
**Every CHORD tracker correlation since first light was noise, at every code phase.** No seed,
clk, NH alignment, trim or statistic could ever have locked. The SEARCH never had this bug: it
passes global channel ids to `channels_hoprate`.

Fixed (51b1ca034): a `chan_ids` overload taking each local channel's GLOBAL bin;
`cudaGnssChordTrack` reads config `channel_ids`, generator-emitted per GPU, FATAL on size
mismatch. Airspy's offset constructor untouched.

**Why it survived every test.** It is invisible to self-consistent checks: the exactness gates
compare GPU against GPU, or against a CPU replica built from the SAME wrong channel list, so
they read 0.000e+00 all day. Only an INDEPENDENTLY generated signal exposes it.

**THE TEST TO KEEP** (`scripts/conv_test.cpp`): synthesize data with the CPU bank at a known
cp, scan the full code on the GPU, require the peak at offset +0.000 with a large ratio.
    before: no peak anywhere, ratio 4.0,  |P| 9e5
    after:  peak at cp_true exactly,      ratio 30.5, |P| 6.2e9   (x7000)
It also settles the cp convention permanently: the bank's `code_phase_chips` and the GPU's
`cp_seed` agree exactly at a common window (both window-referenced in the live usage). Run
this BEFORE chasing sky, clk calibration, or lock statistics.

**What this retroactively explains:** every q at its theoretical noise value 1.0-1.3 in the
first trustworthy sweep; the flat NH scan; both oracle nulls (they used the same broken
harness); and why the earlier fixes -- record advance, NH-baked code, EMA statistic,
exact-epoch seeding -- were each necessary but individually untestable while this masked them.

**NEXT (after restarting the node kotekans on 51b1ca034 + regenerated configs):** seed from a
fresh strong search detection via scripts/trim_lock2.py (exact-epoch, per-sat delta ~148
chips), then watch `get_trim`: q should climb toward ~3.6 with ema_frames filling, and the
combiner amplitudes should lift off the 0.014 floor for the first time. Only after a real lock
does the clock-oscillation question (5h/5j) become answerable.

## 5l. FIRST SKY CONFIRMATION of the fix, and why the SEARCH's cp cannot bridge to the tracker

With `51b1ca034` applied offline (the live nodes were still on the old binary), a frame captured
during **PRN 32's transit (search snr 936)** was scanned with the tracker's own despread path
(scripts/oracle2.cpp, all 32 elements, 4 records):

    ratio 49.8 at cp204 162833, top cells spaced 3.25-3.50 chips  <- the stride-16 GRATING LOBES

That is the first real correlation the CHORD tracker path has ever produced. The same scan
before the fix, and on every other configuration tried, returned flat noise. The lobe comb at
3.27 chips is exactly the predicted single-GPU comb ambiguity, visible for the first time.

**The remaining gap is a CURRENCY problem, not physics.** The measured cp is +4477 chips from
the model, while the search-derived delta says +140. It is NOT Doppler (model vs search agree
to 7 Hz) and NOT drift (search-derived deltas were stable to a few chips over an hour). The
reason the search's cp0 cannot be bridged to the tracker:

    cp0 = (best_cp - off - drift) mod L,  drift = hop*fft_len*cps*(sgn*dop/carrier) mod L
    d(drift)/d(dop) = hop*fft_len*cps/carrier = ~5085 CHIPS PER HZ at 6.8 days of uptime

Undoing that term requires reproducing the search's Doppler AND its exact convention to
sub-Hz; any mismatch is thousands of chips. (This is the same 4412-chips/Hz lever noted in
5b, now larger with uptime.) Weak detections' deltas swinging by thousands of chips (P27 +155
-> -4381) were this, not noise in the fit.

**BOOTSTRAP INSTEAD (scripts/lock_bootstrap.py, ready to run):**
 1. capture a frame; measure the strongest satellite's cp DIRECTLY with oracle2 (the tracker's
    own convention, no reconstruction);
 2. delta_common = cp_true - cp_model(hop) -- instrumental delay + receiver clock, common to
    all satellites (the per-sat term is only ~+-9 chips);
 3. seed every strong satellite at cp_model(now) + delta_common;
 4. sweep a shared +-12 chip offset with the EMA'd q and hold at the peak.
Only a transit-strength satellite (snr >~ 300) is findable in a single 42 ms frame with 7
channels -- P1/P28/P27 at snr 44-60 gave ratios 2.6-3.0 (noise) in the same frame. Wait for a
bright one, or stack more frames.

**STILL REQUIRED: restart both node kotekans on 51b1ca034** (running instances at 18:20 predate
it -- `curl -s localhost:12049/config | grep channel_ids` returns nothing, so they are still
correlating at DC).

### The one calibration datum so far (use this to seed the first lock)

    frame seq0 1874906635567104  (hop 114435219456, 2026-07-31 ~19:41:51 UTC)
    PRN 32, model dop +189.7 Hz (model vs search agree to 7 Hz)
    model  cp204 158356.27
    MEASURED cp204 162833.09   ratio 49.8, grating lobes at 3.25-3.50 chip spacing
    => delta_common = +4476.82 chips  (instrumental delay + receiver clock, common-mode)

A second attempt 25 min later (seq0 1878764187287552) found nothing (ratio 2.5) because the
satellite had left beam centre: **search detections are LATCHED and can be ~18 minutes stale**,
so the snr shown in get_detections is NOT the snr at capture time. Capture immediately after a
pass line appears in the log, and prefer a satellite whose snr is still RISING pass over pass.

### Channel-set generality (audited 2026-07-31, after the DC-replica bug)

The upstream is more general than a comb, so nothing downstream may assume one. Audit result:

* **Already general, takes the list verbatim:** `GnssChordVoltageTap` (`chan_ids`), the replica
  bank (`hoprate_filter(want, ...)`, `channels_hoprate(..., want, ...)`),
  `GnssChannelizedSearch::local_of_global` (arbitrary id list; falls back to offset+count only
  when no list is configured), `gnssBandPlan` (enumerates, no stride logic), the NxM kernel
  (`chan_ids` is a data-layout map).
* **Was the exception, now removed:** `GnssCudaDespread` had an `(n_chan, chan_offset)`
  constructor -- `global = chan_offset + local`. It is DELETED. The channel-id list is the only
  form, so no caller can silently assume contiguity; the two airspy sites, whose subbands
  genuinely are contiguous, now say so in one `std::iota` line. The constructor also validates
  ids against `spectrum_length()` and rejects duplicates (a repeated bin would double that
  channel's weight in the coherent sum and bias every correlation).
* **Guarded by a test** (`scripts/irregular_test.cpp`): an UNSORTED, IRREGULARLY SPACED set
  {6020, 5972, 6068, 5975, 6023, 5988, 6119} -- gaps of 16, 3, 45, 1, 96, 7 -- synthesizes and
  despreads at exactly the injected code phase (offset +0.000, ratio 22.1). Run it after any
  change to channel handling; it fails loudly if structure creeps back in.

No spacing, ordering, contiguity or comb assumption remains in the synthesis path. What DOES
depend on the channel geometry is the *interpretation* of results, not their correctness: the
correlation's grating-lobe spacing is 1/(channel spacing) (3.27 chips for CHORD's stride-16
single-GPU comb, 13.09 for the 27-channel union), and an irregular set simply gives a messier
ambiguity pattern rather than a wrong answer.

## 5m. THE SEARCH'S REPORTED CODE PHASE -- PARTLY RETRACTED, see 5n

> **RETRACTION (2026-07-31, later).** The measurements in this section are all sound and
> reproduce; two of the CONCLUSIONS drawn from them are wrong, and 5n replaces them:
>   * "`cp0 = best_cp - off - drift` double-counts" -- it does NOT. Subtracting `off` and
>     `drift` is the correct inversion of the fixed anchor; the search's cp0 is right.
>   * "the 391.26-chip acquire-vs-despread gap is unexplained" -- it is the Doppler lever.
>     The two tools were handed Dopplers 0.077 Hz apart, and 0.077 x 5094.9 = 392 chips.
> The anchor effect described below is real and is exactly what `off` removes. Read 5n first;
> this section is kept because its numbers are the evidence 5n is built on.

### 5m (as written). An anchor effect, proven offline (2026-07-31 night)

Two live seeding attempts on the FIXED binary (both nodes confirmed carrying `channel_ids`)
returned clean nulls -- q = 1.0-1.2 everywhere, the exact noise value:
  * model + delta_common (+4476.82, from the P32 transit) swept +-12 chips: nothing;
  * the search's cp_phys extrapolated by the tracker's own formula, + the -1961.26 correction
    measured against the oracle, swept +-16 chips: nothing.
So delta_common is not constant and the -1961 correction is epoch-dependent: the search's code
phase and the tracker's are in different currencies by a term that MOVES.

**Root cause, proven with a synthetic injection** (`scripts/acq_conv.cpp` -- inject a known cp,
ask the SEARCH's own acquire where it is):

    repl0 anchored at Mp*fft_len  (WHAT THE LIVE SEARCH DOES)  -> error -3942.61 chips
    repl0 anchored at the data window                          -> error    -1.20 chips (= its
                                                                  own refine resolution)

The offset is exactly the data window's absolute code advance:
`off = (window_start_sample * chip_rate/sample_rate) mod L` = 6288.59 for that window, and
-(L - 6288.59) = -3941.41 vs the -3942.61 observed. The cause is that the bank's
`code_phase_chips` is SAMPLE-0-REFERENCED (conv_test settled this), so a repl0 anchored
anywhere but the data window carries that window's advance into the reported lag.

Consequence for `GnssChannelizedSearch::cp0 = best_cp - off - drift`: since `best_cp` ALREADY
contains `+off` via the anchor, cp0 = true_cp_at_window - drift, NOT the intended
true - off - drift. Intent and behaviour disagree, which is why no consumer-side inversion has
ever worked. NOTE the caching tension: repl0 is cached across snapshots precisely BECAUSE its
anchor is fixed (Mp*fft_len); re-anchoring per snapshot would cost the 14 s/pass the banded+
cached optimization bought (S5). The fix is therefore analytic (correct the reported cp by the
window advance, or define cp0 consistently and document it), NOT re-anchoring.

**REFERENCE DATASET, preserved:** `session_artifacts_20260730/data/`
`p32_transit_frame_seq1874906635567104.raw` -- the 42 ms frame captured during PRN 32's
transit (search snr 936), 8192 hops x 7 chan x 32 elem, 4+4b, gnss0's comb
(5972 + 16k). Everything below reproduces from it offline, with no sky and no live node:
    ./oracle2 <frame> 32 189.7 1874906635567104 158356.27 5115 0.5 0   -> ratio 49.8 @ 162833.09
    ./audit_any <frame> 0 32 1 7 5972 1874906635567104                 -> acquire cp 8991.83
(model cp204 at that hop = 158356.27, dop +189.7; primary of the oracle answer = 9383.09.)

**STILL UNEXPLAINED, and the first thing to chase next:** on the SAME frame (f3.raw, P32
transit), with the anchor handled correctly, the two paths still disagree:

    search's acquire, repl0 anchored at the data window : cp 8991.83  (snr 164.5)
    tracker's despread (oracle2)                        : cp 9383.09  (ratio 49.8)
    difference                                          : 391.26 chips

Same data, same window. 391.26 is not a multiple of the 3.2735-chip grating-lobe spacing, so it
is not a lobe ambiguity. [RESOLVED in 5n: the two tools were handed Dopplers 0.077 Hz apart,
and the code phase moves 5094.9 chips per Hz. Neither tool was wrong.]

## 5n. THE ACTUAL LOCK BLOCKER: an ARGUMENT is not a PHASE (2026-07-31, fixed)

Everything in 5m was measured by asking one tool where a signal was. The question that settled
it was different: **inject a signal at a phase we CHOSE, and push it through both paths.**
`scripts/mkframe.cpp` writes a byte-for-byte synthetic voltage frame (CHORD's native
[hop][chan][elem] 4+4b, conjugated like the F-engine) at a known code phase, so the existing
tools run on it unmodified. What that showed, in order:

**1. Both engines are individually correct.** Over injected phases across the code, Doppler
-250..+500 Hz, both anchors, NH and no-NH, float and 4+4b byte paths, the acquire's reported cp
matches

    reported = injected + (n_window - n_anchor)*cps + n_window*cps*(dop/carrier)   (mod L)
    n_x = x + fft_len - 1     cps = chip_rate/sample_rate

to within +-3 chips (its own refine resolution, set by n_cover*bin_width). The GPU despread
lands on the injected phase at +0.00. The search's `cp0 = best_cp - off - drift` inverts that
expression exactly -- `off` is the anchor term, `drift` is the Doppler term. **The search is
right, the despread is right, and they agree.**

**2. The currency they share is the problem.** Every generator here, CPU and GPU, defines

    C(n) = comb_mult*code_phase_chips + n*cps(doppler),   n ABSOLUTE

so `code_phase_chips` is not a physical phase: it is a phase back-referenced to sample 0 along
a Doppler-scaled rate. At 6.8 days of uptime (n ~ 1.9e15) that back-reference has a lever of

    d(argument)/d(doppler) = n*chip_rate/(sample_rate*carrier) = **5094.9 chips per Hz**

Measured directly on the tracker's own GPU path: despreading a known injection with the Doppler
wrong by 0.1 Hz moves the peak 509.50 chips (predicted 509.49); by 0.05 Hz, 258.00. This is the
same number the broker has always known as the "0.65 chips per Hz per 1000 s" projection noise
and the memory as the ~5085 chips/Hz drift lever. It is not new physics; what was new is
realising the TRACKER was violating it.

**3. Two violations, both in `cudaGnssChordTrack`, both fixed in 9e12d515b.**

  * *The nominal advance was added twice.* The generator already advances the code by n*cps.
    d37064e87 added `chips_per_hop*dh` on top. Measured over four records of one seed:
    power 1.0000 / 0.0000 / 0.0000 / 0.0002 -- the record at the seed epoch despread perfectly
    and every later one despread noise. d37064e87 read the symptom right and the cause
    backwards; the airspy record being exactly one code period is what makes the extra term
    ~0 mod L there, so it was harmless in the prototype and fatal here.
  * *The argument was re-used against a moving Doppler.* `dop = doppler_hz + dop_rate*dt`
    changes every record, and an argument is only valid at the Doppler it was derived at.
    Isolated (dop_rate -0.4 Hz/s, seed otherwise exact): 1.0000 / 0.0004 / 0.1003 / 0.0010 --
    note the 0.1003, a partial grating-lobe re-alignment, which is precisely what a "flash"
    looks like from the outside.

The fix is a rule, now enforced by two functions on the bank (`phase_from_arg` /
`arg_from_phase`, with `window_advance_chips` doing the absolute reduction in long double
because double is only good to ~1e-3 chips at this uptime): **never transport or extrapolate an
argument.** Lift it to a physical phase at its own epoch with its OWN Doppler (the lever cancels
exactly), propagate in the phase domain, convert back with the Doppler actually being handed to
the generator. A Doppler error then only accrues over the propagation interval -- 0.0087 chips
per Hz per second -- which is what the DLL trim exists to close.

Same four records, same seed, Doppler errors out to 1.5 Hz: **0.988-1.000**. The residual
decline is the genuine code-rate error over dt.

`scripts/seed_chain.cpp` is that whole demonstration and runs offline in under a minute. The
broker needs no change: `cp_to_seed_currency` already re-expresses cp0 in the seed's Doppler.
The airspy tracker is deliberately untouched (its extra term is ~0 mod L, its lever ~30
chips/Hz).

### 5n.1 CONFIRMED ON SKY, and the FIRST LOCK (2026-07-31, ~23:30-00:40)

Both nodes restarted on 9e12d515b. Measured with `scripts/seed_audit.py` /
`scripts/audit_series.py` -- capture a live voltage frame, compute EXACTLY what the tracker
would despread it at (the same phase_from_arg -> propagate -> arg_from_phase chain, in exact
arithmetic), then ask the oracle where the satellite actually is. The difference IS the seed
error, in chips, with no DLL and no q in the loop:

    PRN 3 (search snr 883), seed extrapolated from one search pass:
      age  651 s -> oracle ratio 121.4, NH period 15, seed error  +4.25 chips
      age 1028 s -> oracle ratio  40.0, NH period  7, seed error  +2.75 chips
      age 1314 s -> oracle ratio  24.0, NH period 16, seed error  +2.50 chips

A few chips after twenty minutes of extrapolation, where before the fix it was thousands of
chips and moving. Fitted: a CONSTANT offset of ~+6.8 chips plus a drift of -0.004 chips/s
(= 0.46 Hz of Doppler error). Note the constant is a clean number now and is the honest
replacement for the ~148-chip "instrumental delay" of section 6, which was measured through the
anchor-contaminated cp -- do not reconcile them, re-measure.

**FIRST LOCK.** Seeding PRN 3 from that direct measurement (`scripts/lock_measured.py`: seed
the argument the oracle actually found, anchored at that frame's own hop):

    t+150s   q 3.8 / 3.7 / 3.2 / 3.7      (q_noise ~1.0, q_locked ~3.6)
    t+240s   q 3.7 / 3.6 / 3.1 / 3.6

All FOUR tracker stages (both GPUs on both nodes) at once, held 90+ s, with the DLL trim moving
in lockstep across all four to +-0.02 chips -- which is itself the proof it is one physical
signal and not four noise excursions.

**What is NOT yet solved: the residual rate.** The trim ramped at +0.0257 chips/s through that
lock -- a ~3 Hz per-satellite Doppler residual (the search's grid is 31.25 Hz and its parabolic
refine is only good to ~1/20 bin) -- so it walks into the +-3 chip `trim_clamp` in ~2 minutes.
Folding the measured rate back in by hand OVERSHOT on the next attempt (trim ramped negative
into the clamp, q never settled), so this needs a closed loop, not a hand-tuned constant. The
broker already has the machinery (rate fitting, clock-bias solve, `carrier_trim_hz`); it is
currently running with `--trackers ""`, i.e. hinting the search but not seeding. **Next step:
point the broker at the trackers and let its loop close the rate**, rather than more hand
seeding.

## 6. Also outstanding

* **Instrumental delay: partially measured, but the number below is NOT yet trustworthy.**
  A common ~148-chip term appeared in the search-vs-model deltas across satellites (P8 +136..140,
  P10 +151..155, P24 +134..138, P23 +149, P27 +155, P32 +148), with a reproducible ±9-chip
  per-satellite spread seen on two separate days. TREAT IT AS PROVISIONAL: those deltas are
  built on the search's reported code phase, which §5m proves carries an anchor artefact, and
  the direct oracle measurement of the same quantity for P32 disagreed by thousands of chips.
  Re-derive the constant once the acquire-vs-despread question is settled — the oracle
  (`scripts/oracle2.cpp`) measures it directly and needs only one transit-strength satellite.
  The cable term is well determined —
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
* **The per-satellite ±9-chip spread is real and unexplained** (seen 2026-07-30 and again
  2026-07-31, in two different currencies). Suspects unchanged: SV-clock relativistic term,
  TGD, or a per-PRN code-generation offset. Only worth chasing after a lock, when the trim
  measures each satellite's residual directly.
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

## 5o. WHY THE SEARCH IS SLOW, and what is actually redundant (2026-08-01)

A cyclic FFT correlation needs a window that is a whole number of hops AND a whole number of
code periods. One L5 code period is 3.2e6 samples, a hop is 16384, and gcd is only 1024 -- so
the shortest legal window is lcm/16384 = **3125 hops = 16 ms = SIXTEEN code periods**. On the
airspy prototype the hop divided the code period, so the window was exactly ONE. That factor of
16 is the whole difference, and it lands on `aggregate_accumulate`, whose
`n_dop * Mp * s_stored * n_chan` loop is the cost of a pass. Measured on the archived P32
frame: 73 s unfolded vs 9.4 s with the coarse axis capped at one code period -- so the surface
aggregate IS the dominant cost, and it does scale with Mp.

**But the obvious fold is WRONG, and the trap is worth recording.** The argument "repl0 is
periodic in the window, so 15/16 of the lag axis is a copy" holds only for the BARE primary
code. The search's repl0 carries the NH overlay (`nh_phase >= 0`), whose period is 20 ms
against a 16 ms window -- not periodic at all. The (tau + 1 period, nh - 1) equivalence that
looks like it should rescue this breaks at the cyclic wrap, because the window truncates at 16
periods and not a multiple of 20. Measured, same frame, folded vs unfolded:

    unfolded : snr 164.54  nh 10  cp 8991.83
    folded   : snr 106.29  nh 12  cp 9060.60      <- 68.8 chips off, peak degraded

Implemented, measured, reverted the same night. Do not re-attempt it in that form; the 7.8x is
real but it is not free.

**The direction that IS exact:** take the overlay OUT of the correlation. Correlate against the
bare primary code (periodic in the window, so the coarse fold becomes exact), keep the per-code-
period COMPLEX correlations, and recover the NH alignment afterwards as a sign-weighted sum over
those 16 periods. That attacks both multipliers at once -- the 16x redundant lag axis and the
20x whole-surface rescan (the surface is currently computed 20 times, once per alignment, and
the alignments differ only by a per-period sign pattern). Needs care: the current surface is
incoherent |D|^2, so the per-period correlations must be kept complex.

Second lever, independent and easier: the fine axis is stored at full `s_stored` = 4096 columns
= 0.0032-chip resolution, when the refine that follows spans +-4096 samples (+-13 chips) and
only needs the peak localized well inside that. The cross-channel peak is ~157 samples wide
(1/20.3 MHz of comb span), so a stride of ~64 samples samples it fine -- another ~64x sitting
in the same loop.

## 5p. SEEDING: what the broker was doing wrong (2026-08-01)

* `--cl-assist` computed the long-code segment with L2C CL's 75 segments / 1.5 s epoch
  HARDCODED, so every other overlaid signal was pinned to period 0. Parameterised
  (`--long-code-segments`, `--long-code-epoch-s`); defaults unchanged, L2C bit-identical.
* The tracker's phase propagation dropped the code-Doppler feed-forward (fixed, e9892d445 --
  see 5n). Hand seeding hid it because it had been supplying that exact term as `code_phase_rate`.
* The seed carried the MODEL Doppler (`src=pred`), never the measured one, whenever `--almanac`
  was on. Model-vs-measured is +231 Hz with a 244 Hz spread here, against a measured Doppler
  good to ~3 Hz (proven by the residual code rate of the 5n.1 lock). At a 456 s anchor that is
  340 chips of seed error vs 12. New `--seed-doppler det`; measured seed error on PRN 4 went
  from >300 chips to **-14.5**.
* **The NH period the assist picks is WRONG by 7 ms** (seeded period 8, truth 1, measured
  against the oracle at ratio 165.8). And the `fine +-0.5 ms` figure the assist logs CANNOT
  reveal it: k absorbs whole milliseconds, so `fine` is confined to +-0.5 ms however wrong the
  absolute time is. On L2C CL a segment is 20 ms and that log's +-10 ms budget is real; our NH20
  segment is 1 ms, needing 20x better absolute time. `--cl-time-adjust` is the escape hatch and
  the constant is measurable from one strong satellite -- but VERIFY it per satellite before
  trusting it, since a range-model error would not be common-mode while a clock error would.

**Anchor age is the multiplier behind all of this.** A detection's `ref_hop` stamps its
snapshot's START, but the result only appears when the whole pass finishes, so a seed is already
~13 minutes old the first time it is posted. Doppler error x anchor age is the seed error:
0.0087 chips per Hz per second. Nothing here locks reliably until the pass is seconds, which is
why 5o matters.

## 5q. SEARCH COST, ROUND 2 (2026-08-01): what worked, and why the NH restructure cannot

**Landed, verified, 8.8x end to end** on the archived P32 frame (`scripts/audit_step.cpp`,
commit 088394619). The surface is the entire cost of a pass -- 2.8 GB and 15 Tflop per pass,
both of which match the ~880 s observed -- so all three changes attack it:

1. **Fine-lag decimation, `acquire_fine_step` (7.4x).** The fine axis is the beam of the
   covering comb: main lobe `sph/(comb span in bins)` = 157 samples, stored at 1-sample
   resolution, i.e. ~157x oversampled. Step 1 vs 32: snr 164.54 -> 163.77, cp 8991.83 ->
   8991.86, same NH alignment, same Doppler. 0.03 chips and 0.5% for 7.4x. Unlike 5o's fold
   this assumes NO periodicity, so the secondary code cannot invalidate it.
2. **Aggregate splits the (Doppler x coarse-lag) product**, not the Doppler axis alone -- the
   d axis is short exactly when the broker's hints are working, so the old split idled cores
   in the case that matters.
3. **Per-channel correlation runs in parallel** (it was serial only because all channels
   shared the one caller-owned FFTW workspace). 6.9 -> 4.3 s, and it dominates once (1) lands.

At `fine_step 1` the new code reproduces the original serial answer EXACTLY.

### The NH restructure: attempted, measured, abandoned -- and the reason is worth keeping

The idea: the 20 NH alignments are not independent problems. The overlay multiplies each CHIP
by a sign, so `repl^(nh) = sum_k NH[(k+nh) mod L] B_k` with `B_k` the replica restricted to code
period k; correlation is linear; and since the primary code repeats, `B_k` is `B_0` delayed by
exactly one code period T. If `D^(k)(tau) = D^(0)(tau + kT)` then ONE correlate + ONE aggregate
yields all 20 alignments as sign-weighted sums of 16 shifted copies -- ~17x, and it delivers 5o's
coarse-axis reduction for free because the periods get summed explicitly.

The algebra is right (note `+kT`, not `-kT`: the surface's lag axis is the negative of physical
delay, and the wrong sign sums one real peak against 15 empty cells -- a ~sqrt(n_per) SNR loss
that reads as "nearly working"). **The representation is what fails.** Measured on NOISELESS
synthetic injection, the per-period magnitudes at `tau + kT`, which must be identical, came out

    10, 27, 73, 92, 113, 135, 186, 200, 211, 228, 239   (should all be 239)

and the assembled result was snr 971 vs 4470, 16 chips off, with the wrong NH alignment.

The cause is stated in gnssChannelizedAcquire.hpp's own header: the cross-channel reconstruction
`D(q,s)` is *exact for a critically-sampled rectangular bank and APPROXIMATE for the windowed
PFB* -- "sufficient because acquisition only has to localize the peak to ~1 chip". Summing 16
shifted copies COHERENTLY needs `D` accurate in amplitude and phase at 16 arbitrary sub-hop
offsets (the k-th shift lands at `k*0.3125` hop, spanning the full range), which is a far
stronger requirement than localizing a peak. The spread above IS that reconstruction error.

**Do not retry this by re-deriving the indexing.** Anything that coherently combines `D` at
sub-hop offsets inherits the same limit. The routes that remain honest are: generate 16
separately PFB-correct masked replicas (only 20/16 = 1.25x, not worth it), or accept the 20-way
scan and attack the surface, which is what (1)-(3) do.

**Remaining, untried:** the Doppler grid is 31.25 Hz, exactly half the 62.5 Hz bin spacing of
the Mp-point transform. At 62.5 Hz the wipe becomes an integer bin rotation, so `FFT(wiped)` is
a cyclic shift of ONE `FFT(data)` -- roughly halving the correlate cost, which is now the
dominant term. The parabolic refine already recovers sub-grid Doppler.

## 5r. THE PERIOD, AND THE WINDOW/OVERLAY MISMATCH (2026-08-01 afternoon)

Two real defects found, both from the same root: **CHORD's natural integration window is 16
primary code periods, and L5 Q5's overlay is 20.** Nothing divides anything.

### 5r.1 acquire_windows was smearing across overlay bins

The search accumulates |D|^2 over `acquire_windows` windows of Mp = 3125 hops. Mp is a whole
number of PRIMARY periods (16), so the code phase is stationary window to window -- but 16 is
NOT a whole number of NH periods, so each window's overlay alignment advances by 16 = +4 mod 20
and lands in a DIFFERENT nh bin. Measured on noiseless synthetic, every injection:

    1 window : snr 9121  9124  8738  8744  8908  8905
    2 windows: snr 4579  4635  4577  4375  4470  4529     <- exactly HALF, every time

The second window's signal goes to another bin; its noise raises the mean everywhere. So
`acquire_windows` was costing sensitivity, not buying it, AND scrambling the reported `nh`
(with 8 windows the true alignment gets at most 1 in 8, and 5 aliases sit at equal height).
Live confirmation: PRN 23 at snr 342 with one window, against 56 with eight.

`acquire_windows: 1` is deployed as the immediate fix. NOTE the side effect: the pure-noise
ceiling rises 4.45 -> 19.1 (a Gamma(1) tail, not Gamma(8)), so `acquire_snr` MUST rise with it
-- 5.5 would have let pure noise through as detections. It is now 30 in both the stage and the
broker. **The better fix, not yet done:** route window w into bin (a + 4w) mod 20 -- pure
bookkeeping, no extra compute, recovering full multi-window integration AND a clean nh, and
letting the ceiling and the threshold come back down.

### 5r.2 the code phase was reduced at the wrong length

The tracker despreads the 204600-chip overlaid code; the search reduced cp0 mod the 10230-chip
PRIMARY, which cannot carry which of the 20 periods it was in. That is why NOTHING supplied the
period: `--cl-assist` needs clock+range under half a period (0.5 ms here, vs L2C CL's 20 ms),
and reconstructing from the measured `nh` needs a convention that turned out to depend on the
replica anchor's own period index -- measured across three anchors, offset = (4 - k0) mod 20.

Fixed by having the SEARCH report the phase at the overlaid length (`code_phase_long_chips`),
which needs BOTH the nh lift AND dropping the `_snap_start_hop % Mp` shortcut (valid mod L,
since one replica period is 16 L; invalid mod 20 L). Broker prefers it over any reconstruction.

**Result on a snr-5801 PRN 32 transit, freshest seed:** within-period error **-0.50 chips**,
against -11 to -18 before. The phase is right.

**STILL WRONG, and where to start next:** the period is off by 4 even at 77 s, and BOTH errors
grow with seed age -- (-0.50, -6.00, +36.50) chips and (-4, -5, -6) periods at (77, 214, 266) s,
i.e. ~1 period per 95 s of residual rate. That is ~10 ppm, three orders of magnitude too large
for a Doppler error and flatly contradicting the 0.0257 chips/s measured on the 2026-07-31 lock.
So the -0.50 is ONE fresh point, not a fixed improvement -- do not build on it. Something in the
propagation is rate-wrong at the long length while being right at the primary length; that
asymmetry is the lead.

**Method note.** Three times now a mapping derived on paper (the +kT shift, cp0 + nh*10230, and
the -7 ms cl-time-adjust fitted to ONE satellite) survived to deployment and was wrong, while a
noiseless injection sweep settled it in minutes. Injection is cheap here (mkframe + audit_nwin,
seconds per point). Calibrate first, deploy second.

# ============================================================================
# 6. STATE AT COMPACTION (2026-08-02). READ THIS SECTION FIRST.
# ============================================================================

## 6.1 Known FOR SURE (measured, reproducible, with the number)

**Geometry -- the root of most of it.** CHORD's record is 2048 hops = 10.4857 code periods.
The replica period Mp = 3125 hops = **16** code periods exactly. NH20 is **20** periods. 16 and
20 do not divide each other and neither is a whole number of hops. airspy had record = period =
1, so every one of these was invisible there.

**The code phase is an ARGUMENT, not a phase.** Every generator forms
`C(n) = arg + n*cps(doppler)` over the ABSOLUTE sample index. At 6.8 days of uptime that gives
`d(arg)/d(doppler) = 5095 chips/Hz`, measured by injection (509.50 chips at 0.1 Hz, predicted
509.49). Referenced to the search's FIXED anchor instead: **1e-4 chips/Hz**, 4e7x smaller.
NEVER transport an argument; transport a phase at its own epoch. (Fixed: db02bebfc.)

**Sub-period phase is now CORRECT.** Verified live across 4 satellites over consecutive passes:
the reported phase advances exactly as the code rate requires, residual +-10 chips (the
acquire's own resolution). Every remaining error is an EXACT integer number of overlay periods.

**The overlay period randomised every pass** (+1/+3/-4/+3/+3/-1/-2), because `best_nh` is the
alignment of a replica at a FIXED anchor while the correlation lag slides it by a varying number
of periods. Pinned by continuity across passes (52bb7e6f1): margin is 1500x (a 270 s revisit at
1.5 Hz Doppler error predicts to ~3.5 chips against a 5115-chip half-period tolerance).

**acquire_windows > 1 was destroying SNR.** Each accumulated window advances the overlay
alignment by 16 = +4 mod 20 and lands in a DIFFERENT nh bin. Noiseless synthetic: a second
window HALVES the SNR, every injection (9121 -> 4579). Live: PRN 23 snr 342 with one window vs
56 with eight. Deployed `acquire_windows: 1`; the noise ceiling rises 4.45 -> 19.1 (Gamma(1),
not Gamma(8)) so `acquire_snr` MUST rise with it -- it is 30 now, in both stage and broker.

**Search cost.** fine-lag decimation (`acquire_fine_step: 32`, 7.4x -- the fine axis stored a
157-sample-wide lobe at 1-sample resolution), (Doppler x coarse-lag) parallelism, and parallel
per-channel correlation: 64.6 -> 7.3 s offline. With `prns_per_pass: 1`, ~9 s per PRN-pass live,
from 790-1270 s three days ago. Seed epoch at emit: <= ~27 s.

**Where it stands right now:** 4-6 satellites at q 2.4-3.2 simultaneously on ALL FOUR tracker
stages, holding for minutes, trims agreeing across stages to ~0.05 chips. q_locked is ~3.6.

## 6.2 The ONE thing now blocking, and it is a single number

Code trims walk at 0.034 chips/s and saturate the +-3 chip `trim_clamp` in ~90 s. That is
**3.9 Hz of residual Doppler** -- a RATE error, not a phase error. The lever that cancels it is
`code_phase_rate`, and the broker is emitting **0.000e+00** where it needs **1.736e-07
chips/hop**. The broker's cp-slope fit is meant to produce this from `cp_hist`. Either it is not
accumulating points (round-robin gives each PRN one only every ~90-270 s) or its output is being
discarded. Both are checkable in the broker alone, no sky.

## 6.3 RETRACTED -- do not re-attempt these

* **The NH restructure** (all 20 alignments from one correlation via `D^(k)(tau) = D^(0)(tau+kT)`).
  Algebra is right, including the `+kT` sign. The REPRESENTATION fails: the cross-channel
  reconstruction is approximate for a windowed PFB (its own header says so), fine for localizing
  a peak, not for coherently summing 16 sub-hop-shifted copies. Noiseless per-period magnitudes
  that must be identical came out 10..239. See 5q.
* **The coarse-lag fold.** Only valid for the BARE code; repl0 carries the overlay, whose 20 ms
  period does not divide the 16 ms window. Measured 68.8 chips off. See 5o.
* **Reconstructing the period from absolute time** (`--cl-assist`). Needs clock+range good to
  under half a period -- 0.5 ms for NH20, against L2C CL's 20 ms. Off by 7-11 ms in practice.
* **`--cl-time-adjust` as a fitted constant.** Measured 7 periods on PRN 4 and 11 on PRN 23: NOT
  common across satellites, so no single constant exists.
* **The instrumental-delay figures (~148 chips, then ~16 chips).** Contaminated. -9 chips of the
  latter is an ACQUIRE artefact, measured on NOISELESS synthetic. Re-measure from scratch once
  the period is settled.
* **`--carrier-gain` before a lock exists.** Its gate only engages for sats already in TRACK, so
  with nothing locked it integrates +-20 Hz of noise into the NCOs. Tried twice, wrong twice.

## 6.4 Proposed next moves, in order

1. **BUILD THE END-TO-END HARNESS FIRST.** Inject a known cp204 (`scripts/mkframe.cpp`), run it
   through the search stage, the broker's seed arithmetic, the tracker's propagation, and the
   GPU despread, and print the final error in chips. Every piece already exists and is already
   used separately -- they have never been closed into one loop, and that is precisely why this
   week found bugs one at a time in deployment order rather than all at once. Nothing else on
   this list should be attempted before it exists.
2. The `code_phase_rate` lever (6.2) -- the current blocker.
3. A residual gate on the continuity correction: one live correction came back +18 periods with
   a -173.6 chip residual against ~10 for the others. That is resolution against a bad prior;
   reject corrections whose residual exceeds ~50 chips, or one bad pass poisons a whole sequence.
4. One oracle calibration on a strong satellite to pin the ABSOLUTE period. Continuity gives
   self-consistency, not absolute truth -- a whole sequence can sit a constant integer off.
5. The window/overlay bookkeeping fix: route window w into bin (a + 4w) mod 20, recovering full
   multi-window integration AND a clean nh at no compute cost, and letting the noise ceiling and
   `acquire_snr` come back down from 19.1/30 to 4.5/5.5.

## 6.5 Method note, addressed to whoever picks this up (including me)

Three separate mappings derived on paper reached deployment wrong this week: the `+kT` shift
direction, `cp204 = cp0 + nh*10230`, and a `-7 ms` cl-time-adjust fitted to ONE satellite and
generalised. A noiseless injection sweep settled each in minutes -- `mkframe` + `audit_nwin` is
seconds per point. The tooling was there the whole time and was used to CONFIRM after deploying
rather than to DECIDE before. Calibrate first, deploy second, and close the loop end to end
before trusting any component in isolation.

# ============================================================================
# 7. THE HARNESS, AND WHAT IT FOUND IN ITS FIRST DAY (2026-08-02)
# ============================================================================

## 7.1 It exists: scripts/gnss/e2e

Item 6.4/1 is done. `scripts/gnss/e2e.cpp` injects a known cp204 into a noiseless synthetic
CHORD sky and runs it through the ACTUAL SHIPPED code of every stage -- acquire, refine,
`detection_phase`, seed, `propagate_seed`, GPU despread -- printing the error in chips at each
hand-off. `--skip-search` isolates the tracker leg in ~5 s; the full chain is ~28 s per PRN.
Build with `scripts/gnss/build_tool.sh e2e`.

The rule that makes it worth anything: it CALLS the stage arithmetic rather than restating it.
The three cross-stage mappings now live in `lib/stages/gnss/gnssSeedTransport.{hpp,cpp}` --
`detection_phase`, `propagate_seed`, `refine_peak` -- and the stages and the harness call the
same functions. A harness that re-derives a mapping tests the author's understanding of it,
which is the thing already known to be unreliable.

The KEY TRICK that makes arbitrary seed ages free: an argument is referenced to absolute sample
0, so ONE argument describes the satellite for all time. A record at window W synthesized with
argument cp204 is exactly the continuation of the same signal, so a 300 s seed age costs one
extra 2048-hop synthesis, not 300 s of samples.

## 7.2 What it found immediately

**The tracker leg was already exact** -- 0.000 chips, P/P_true 1.0000. Good; that half was right.

**6.2 confirmed and its fix verified without sky.** The search's phase transport is essentially
exact (+0.07 chips); the whole failure is a +1.70 Hz Doppler error acting over the seed age.
Measured drift 0.01476 chips/s against 0.01476 predicted from dop_err * chip_rate/carrier. With
`code_phase_rate = -7.7e-8` chips/hop the error over 107 s goes 1.58 -> 0.038 chips and q holds
at 3.06-3.11 instead of collapsing to 0.004.

**THE OVERLAY PERIOD (5r's "STILL OPEN", now closed).** The coarse lag spans 16 primary periods
but `best_cp` is that lag reduced mod ONE, so the lag's whole-period count never crossed the
stage boundary; 16 is not 0 mod 20, so `best_nh` cannot carry it. Reported period came out
(best_nh - 4) mod 20 every time and was wrong by -2..+3 in 8 of 15 runs. **That is the
"+1/+3/-4/+3/+3/-1/-2" randomisation from 6.1, which was recorded as unexplained. It was never
random.** Fixed by lifting with `peak_tau_samples` (present in AcquisitionResult all along).
The fold of [0,16) into (-8,+8] is measured: over 25 injections |folded| never exceeded 3.

**THE REFINE STOPPED TOO SOON.** `refine_span: 4096` on the aggregator against true offsets that
reach +17.19 chips -> settles on the neighbouring grating lobe, exactly 13.09 chips out, 4 runs
in 15. Restored to the fft_len default and parallelised: 4x the work, 2x FASTER (59 -> 27-29 s).

Final acceptance on the fixed code: **16/16, every period correct, every residual <= 0.113
chips**, over tau 0..15 and true periods 0..18, including held-out PRNs and snapshot epochs.

## 7.3 A wrong turn worth remembering

I first reported the 13.09-chip error as a fine-lag ALIAS the acquire could not resolve, "fixed"
by span 8192, with 17/17 acceptance behind it. That was wrong. 13.09 chips is BOTH
`s_stored*cps` AND the old `refine_span*cps`, because the aggregator's span had been set to
4096 = s_stored -- two unrelated quantities, one number. Span 8192 helped because it reached the
peak, not because it resolved an alias, and the 17/17 confirmed a coincidence. Two "clever"
fixes built on the wrong mechanism made it worse (13.198 chips out, then 26.292).

What settled it was dumping the objective (`e2e --dump-refine`) instead of theorising about it:
lobes 4096 samples apart, the true one 15% higher, nothing to resolve. **Testing that a change
HELPS is not testing that the explanation is RIGHT.** Look at the shape before naming the
mechanism.

## 7.4 Next

1. `code_phase_rate` from the broker (6.2) -- now the only open item in the chain, and the
   harness measures the fix directly (`--seed-cp-rate`).
2. Put the REAL broker in the loop: `e2e --emit-detection` writes the /get_detections wire
   format; serve it, run gps_distributed_broker.py against it, capture /set_seeds, feed back
   with `--seed-file`. That closes the last stage that is still tested by proxy.
3. The broker's continuity resolver (52bb7e6f1) was compensating for the period bug at source.
   With 7.2 fixed it should now always agree -- verify, then consider whether the residual gate
   (old 6.4/3) is still needed or whether continuity should simply become a consistency check.
4. Re-measure the instrumental delay (6.3) now that the period and the refine are honest.

# ============================================================================
# 8. STATE AT COMPACTION #2 (2026-08-02 evening). READ AFTER SECTION 6/7.
# ============================================================================

## 8.1 The one-line summary

The ±60 chip seed scatter that was blocking every lock was **grating lobes from a 2-node comb**,
and running all 8 nodes removes them structurally. That is now done and the nodes are stable --
but the 8-node search does not complete a pass, so we have no detections yet.

## 8.2 Settled today (measured, reproducible)

* **The overlay period fix is correct.** 16/16 on injection, and **4/4 on sky against an
  independent oracle** (PRN 20, snr 477, four different periods: 17, 16, 1, 14). Robust to
  noise: 0/12 wrong at every noise level tested. Section 7's fix stands.
* **`refine_span: 4096` was a hand-added override** in a generated config, inside a commit about
  broker clock drift. The generator emits `refine_step` from geometry and no span at all; the
  stage default (`fft_len`) was always right. Reverted (58dd2e553). The override traded reach
  for resolution at CONSTANT cost -- 109 vs 110 evaluations -- and cost a week.
* **THE ±60 CHIP SCATTER = GRATING LOBES.** The 2-node comb is 27 channels at stride 4, so
  gcd = 4, `s_stored` = 4096 samples = 13.09 chips, and the refine's objective carries grating
  lobes at that spacing with the true one leading by only ~15%. Reproduced on the bench once the
  harness gained additive noise:

      noise  acquire snr   rms err   worst   WRONG LOBE
         10        711     20.3 ch   56.4      42%
         20        185     41.5 ch   60.9      92%
         40         48     39.9 ch   61.0     100%

  Sky agreed: PRN 20 probed at four seed ages gave -1.5 / +19.0 / +62.5 / +64.0 chips,
  uncorrelated with age. **Raising `acquire_snr` does NOT fix it** -- 42% wrong at snr 711.
* **8 nodes removes it structurally.** All eight `gpu_offsets_mod16` cover {0..15}, so the union
  comb is 106 channels at **stride 1**: gcd 1, `s_stored` = 16384 = a whole hop, no periodicity,
  no lobes. Merge confirmed live: "first aligned frame across 16 inputs (106 channels merged)".
* **THE MERGE BROKE `GnssCoherentCombiner`.** All 8 nodes SIGSEGV at 42-46 s on the post-merge
  binary; all 8 stable on `pre-merge-backup` (58dd2e553). That file took +342 lines in a
  CONFLICT-FREE auto-merge (prototype added snr_q / NaN fail-open / overlay carrier squaring;
  we added the per-antenna beam map), so nobody reviewed the combination. `_st_snr_q` and the
  merged `_navbuf[p]` block are both cleared as suspects.
* **Three SNR gates, one defect.** Continuity, the cp-rate fit, and the clock-freq bias median
  all averaged per-satellite quantities with no quality gate. All three now gated
  (`--period-check-snr`, `--fit-min-snr`, `--bias-min-snr`), all defaulting to 0 = prototype
  behaviour. The bias median's raw scatter was 10.5 Hz where the acquire's own error predicts
  0.8 Hz at N=2.
* **`--fit-gap-s` defaults to 16 s** and resets the cp-fit history on every detection at CHORD's
  90-270 s revisit, so `code_phase_rate` could NEVER be fitted. With `--fit-gap-s 900
  --fit-min-snr 60` it fires and the common-mode clock path came alive for the first time.

## 8.3 Open, in the order I would take them

1. ~~**The ms-split lag mapping**~~ **CLOSED 2026-08-02**, `41d5fe80a` / `gnss::ms_split_peak`.
   Three terms, each isolated by a geometry sweep rather than argued: the replica's own anchor
   phase, the cyclic wrap `Ns*cps`, and the fine axis's opposite sign. See
   `CHORD_GNSS_MS_SPLIT_SEARCH.md` §4b. Exact to <0.05 chips at stride 1; still Phase A only,
   so still no overlay period.
2. ~~**The 8-node search does not complete a pass.**~~ **WRONG -- retracted 2026-08-02.** It
   completes passes continuously and has all along. Checked at 2h21m uptime: `pass best snr`
   lines every few seconds, PRN 10 at 2827, PRN 24 at 1072, PRN 23 at 722 against threshold 30
   and a ceiling of 18.4; nine PRNs seeded, the newest 2 s old, all carrying a fitted
   `code_phase_rate` and `doppler_rate_hz_s`. The process sits at 33% of ONE core *because it
   is idle between passes*, not because the acquire is serial -- all 17 `/gps_search` threads
   exist and no thread is above 3%.

   What I mistook for a stalled pass was the round-robin: `1 searched, 9-11 deferred to later
   passes`. One PRN per pass is by design, so a full PRN sweep takes ~10 passes; watching for
   "a pass" to cover everything sees nothing finish. **The lesson is the one in 8.6 again --
   I read a rate off a process-average `%CPU` and a partial log tail instead of asking the
   stage what it had done.** The 1.7 GB / ~13 min serial `repl0` precompute is real and still
   worth attacking, but it is a startup cost, not a stall.
3. **Bisect the combiner segfault** across the merge's 24 changed source files. Offline work.
4. **Phase B of the ms-split** -- Phase A carries NO overlay period (phase mod 10230, not
   204600), which is LESS than the current search reports.
5. **Are the trackers locking?** The seeds are good and current; what happens downstream of
   them on the nodes is now the open end of the chain, and is not yet measured.

## 8.4 The ms-split, honestly

`gnss::ms_split_accumulate` is implemented and builds; the harness drives it
(`--ms-split K --sub-hops N --dump-mssplit`). Economics confirmed: over 100 ms it is
**0.53 G ops vs 137 G** for today's full-length -- 258x -- because a ~1 ms sub-window spans one
NH chip (no 20x alignment axis) and one code period of lag (not 16). Sensitivity cost is real:
**-11.9 dB** vs a full-length search on the same 106 channels (-5.9 dB vs the 2-node baseline we
were actually running).

**It does not work yet, and two of the three bugs found were mine, not the algorithm's:**

* `-(-a/b)` is NOT `ceil(a/b)` in C++ (truncation is toward zero). Gave `nwin = 0` (instant
  crash) or a read PAST THE END of the data buffer. **Every ms-split number reported before
  d05729d37 came off out-of-bounds memory** -- and reproduced perfectly, because that memory was
  stable. A deterministic result is not a correct one.
* The two-period replica (needed for full overlap at every lag) makes the lag axis 2.0070 code
  periods, so a periodic code appears on it TWICE and `channelized_peak` cannot prefer either.
  Surface index maps to physical lag as `(q + N) mod 2N`. **Fix: restrict the peak search to
  physical lag [0, N].**
* The two copies are **6.55 chips too close** -- they must be exactly one code period apart. The
  196-hop sub-window is 1.0035 code periods and that 0.35% is not handled anywhere. Same
  non-integer-hops-per-period root as everything else, surfacing a third time.

## 8.5 Running state at compaction

* 8 nodes on `/home/kvand/gnss/kotekan_premerge` via `systemd-run --unit=gnss-node`.
  **Its `--version` banner misleadingly reports HEAD** -- trust the path, not the banner.
* Aggregator: `chord_gnss_agg8.yaml`, 106 channels, merging, NOT completing search passes.
* Broker running with all three SNR gates plus `--fit-gap-s 900`.
* `/tmp/gnss` created on all 8 nodes (records go there; `/data` exists only on cx19 and was
  deliberately NOT created elsewhere).

## 8.6 Method note

Three wrong stories today, all the same shape: a number was explained before its mechanism was
looked at. The 13.09-chip coincidence (`s_stored*cps` == the old `refine_span*cps`), the record
directory (falsified by cx27 all along), and the ms-split offset fit (a peak nobody had dumped).
Each was settled in one run by looking at the actual shape -- `--dump-refine`, then
`--dump-mssplit`. **Dump the objective before naming the mechanism.** The harness now has the
instrument for both; use it first, not after.

## 8.7 Why nothing locks -- measured 2026-08-02, NOT yet closed

The viewer is right and it is not an interface fault. `GnssCoherentCombiner` reports
`coherence_s = 0` deliberately: a wiped rung must clear `FLOOR_MARGIN` x its noise floor to
count, "when none does, coherence_s = 0 (the honest 'no coherent detection')" -- a guard added
2026-07-12 after a convincing false lock. Every PRN on every stream reports 0.

What IS healthy, checked rather than assumed:

* **The search.** Passes every few seconds; PRN 10 at snr 2827, 24 at 1072, 23 at 722 against
  threshold 30 and ceiling 18.4.
* **The detections are physically real.** The broker fits a code-rate clock offset of +0.337 ppm
  from **five satellites independently**, each within ~0.1 ppm. Noise cannot agree like that.
* **The record data.** `gnss1_srch_tap` (feeds the aggregator that detects at 2827) and
  `gnss1_tap` (feeds the tracker that sees nothing) read the SAME `host_voltage_buffer_1`. The
  data is identical; the fault is in the despread parameters, not the samples.
* **Rate seeding.** Most PRNs carry the common (l-a) rate, matching the broker's fitted ppm.

What is broken:

1. **Half the seeds dead-reckon from reference epochs ~1 hour old.** Measured ref_hop ages in one
   seed set: PRN 1/3/28/31 = 0 s, but 2 = 3414 s, 24 = 3505 s, 10 = 3799 s, 8 = 3890 s,
   32 = 3981 s, 27 = 4633 s. With the loops open (`--dll-gain 0.0 --carrier-gain 0.0`,
   `--dead-reckon`) a rate error of only **0.1 ppm** -- well inside the +-0.3 ppm scatter of the
   per-satellite fits -- accumulates 3500-4700 chips over that. The code period is 10230. Those
   PRNs cannot track, and the per-sat fits that deviate most (PRN 10 at +0.065 ppm vs the common
   +0.34) are the worst off.
2. **Something else, still unidentified.** A seed only **13.5 s** old (PRN 28) despreads to
   ratio 2.79 -- noise. So staleness is necessary but not sufficient. Calibration for that
   number: the same tool on the same kind of frame returned 19.5 / 17.3 / 12.3 on 2026-07-31.
   Whatever changed since, this is the open end.
3. **Only cx19 and cx27 are in the broker's `--trackers`.** The 8-node bring-up updated the
   aggregator's feeds and never the tracker list; six nodes are not tracking at all.
4. **cx19 gnss0 stopped** ~40 min before this was written (last record 18:42) while gnss1 kept
   running. Its combiner status is frozen at that time. Unexplained.

Next: (2) is the one that matters. Every earlier successful probe was on **gnss0**; all of
today's were on gnss1, because gnss0 had stopped. Establish whether a fresh seed despreads on
gnss0 before assuming the two branches behave alike -- they are structurally identical in the
generated config, but they carry different channel combs.

### 8.7.1 The cx19 wedge -- diagnosed, restarted 2026-08-02 20:05

Not a GNSS bug and not a crash. For ~50 minutes cx19 dropped **100% of one of its two
391k pkt/s ports**, so the 8-node aggregator ran on 15 of 16 streams (including for the
detections quoted above as evidence the search is healthy -- they stand, but the SNRs are from
a degraded comb).

Measured before the restart:

| | port 0 | port 1 |
|---|---|---|
| NIC receiving | 391,730/s | 391,730/s |
| NIC rx errors / missed | 0 | 0 |
| distributor forwarded | **0/s** | 391,731/s |
| distributor dropped, ring full | **391,731/s** | 0/s |

GPU 0 sat at 0% and 30 C, GPU 1 at 100% and 69 C; no Xid. Ruled out: network (all 16 feeds
ESTABLISHED, every socket queue zero), disk (23% used, 1% inodes), the aggregator connection.

The backtrace (`/home/kvand/gnss/cx19_wedge.bt`) says what it was NOT: every stage in the port-0
chain -- `GnssChordVoltageTap`, `gpuProcess`/`cudaInputData::wait_on_precondition`,
`TransposeBasebandArray`, `ProcessPacketMask` -- sat in `Buffer::wait_for_full_frame`, i.e.
**starved for input**. Nothing was blocked on output, so nothing was holding frames, which rules
out the ordinary downstream-deadlock. Meanwhile `dpdkCore::main_thread` was in its normal
`sleep(0)` supervisory loop and both RX lcores were spinning in `ice_recv_pkts_vec_avx2_offload`
/ `dpdkCore::lcore_rx`. So the packets were being received and never turned into frames: the
stall is inside the DPDK distributor/handler for port 0, upstream of every kotekan GNSS stage.

`sudo systemctl restart gnss-node` cleared it -- both ports now forward 391,488/s with zero
drops and both branches write records. Cause still unknown; `journalctl -u gnss-node` around
18:42 needs sudo and was not read. **If it recurs, that journal and the backtrace are the
artifacts.** This is shared CHORD DPDK infrastructure, so it is not ours to change unilaterally.

### 8.7.2 What the restart ruled out, and the live suspect

A fresh 16.8 s seed probed on **gnss0** after the restart still despreads to ratio 2.43 -- noise,
same as gnss1's 2.79. So the second fault is NOT branch-specific, and "I was probing a starved
branch" is dead as an explanation.

What did change between the probes that worked (2026-07-31: ratios 19.5 / 17.3 / 12.3) and now
is the search comb: 2-node stride 4 -> 8-node stride 1. That interacts with today's ms-split
finding. The shipped coarse mapping adds the fine lag with the WRONG SIGN (see 8.7 / the
ms-split doc §4b); it is survivable only because `refine_peak` re-scans +-`refine_span` =
+-fft_len = +-52.38 chips. The size of that error is `s_stored`:

* stride 4: `s_stored` = 4096 samples = **13.1 chips** -- comfortably inside the refine.
* stride 1: `s_stored` = 16384 samples = **52.4 chips** -- exactly at its edge.

Measured on the bench, noiseless, both recover: stride 4 lands -0.225 chips with the refine
moving +10.58; stride 1 lands +0.206 with the refine moving **+50.30**, under 2 chips of margin
against the span limit. Under noise 20, stride 4 is 0/12 wrong lobe, worst 0.225 chips. The
stride-1 noise run is the open measurement.

**If stride 1 shows wrong-lobe failures under noise, that is the live bug** -- the 8-node switch
would then be putting every seed a full hop (52.4 chips) off, which is exactly enough to make
the tracker despread noise while the search still reports a huge SNR. Fix would be to widen
`refine_span` beyond one hop, or better, correct the fine sign so the coarse number is right and
the refine only has to polish.

### 8.7.3 It is the OVERLAY PERIOD, not the code phase -- measured 2026-08-02 21:5x

Two broker changes (relaunch script `run_broker_all8.sh`, previous line saved beside it):

* `--fit-min-snr 60 -> 0`, the documented default. At 60 the cp-rate fit was starved (PRN 10
  skipping snr-54 points with 3 in history, 18 skipping snr-32 with 1, 32 skipping snr-30 with
  2) -- and a starved fit still WINS, because :2369-2373 overwrites `seed["ref_hop"]` AND
  `seed["code_phase_chips"]` with the fit's own anchor. The fresh detection was being discarded
  in favour of an hour-old extrapolation. **This gate was added earlier the same day and is a
  self-inflicted regression**; it postdates the 2026-07-31 probes that worked.
* `--trackers` cx19+cx27 -> all 8 nodes (17 endpoints). The other six had NEVER been seeded --
  no broker log this session mentions them. They were healthy throughout; nobody was talking to
  them. Nothing to restart on the nodes.

Result: ref_hop ages collapsed from ~5500 s to 0-12 s, seeded PRNs 9 -> 13, zero posting errors
across 16 endpoints, zero fit-min-snr skips.

**And the probe found the satellite.** Every probe before this returned ratio 2.4-2.8 (noise).
With a fresh seed:

| PRN | seed age | ratio | within-period err | NH period |
|---|---|---|---|---|
| 28 | 2.2 s | **22.98** | +151.00 chips | 7 vs seeded 1 |
| 1 | 151 s | **9.8** | **+0.50 chips** | 2 vs seeded 6 |
| 26 | 6.6 s | 1.6 (noise) | -63.50 | 2 vs seeded 0 |

PRN 1 is the one that matters: **the within-period code phase is right to half a chip.** The
seed arithmetic, the propagation, `phase_from_arg`/`arg_from_phase`, the argument currency --
all of it transports correctly to a live frame. What is wrong is the **NH overlay period**: off
by 4, 6 and 2 across the three.

That is the signature we have been staring at all session. Despreading at the right code phase
but the wrong overlay period yields incoherent power (the `amp_snr` 3-8) and no coherent
detection (`coherence_s = 0` everywhere) -- exactly what the combiner reports.

It also puts the blame back on the 16-vs-20 collision: the search resolves the overlay alignment
mod 20 (`best_nh`) and the coarse lag mod 16 (`lag_span_periods`), and `gnss::detection_phase`
has to combine them into one period label. That lift is correct ON THE BENCH (the shipped path
at stride 1 reports `-0.000 periods`), so the defect is something the bench does not reproduce
-- the obvious candidate being that the bench detects and despreads in the SAME window, while on
sky the period label has to survive the ref_hop -> frame_hop gap.

**Next:** instrument the period specifically. Compare, for one PRN over several passes, the
period the search reports, the period the seed carries, and the period the oracle finds. The
within-period residual is already good, so the whole question is which of the 20 the label lands
on and where it changes.

### 8.7.4 The cx19 wedge is SELF-INFLICTED -- unpinned benchmarks, 2026-08-02

It recurs, and the interval matches when I was running benchmarks on cx19:

| | node started | gnss0 died | ran for |
|---|---|---|---|
| first | ~16:33 | 18:42 | ~2h09m |
| second | 20:04 | 21:46 | ~1h41m |

`chord_gnss_cx19.yaml` pins DPDK to `lcore_cpu_map [5, 21, 7, 6, 22, 23]`, `main_lcore_cpu 4`
-- **the cores we were told never to touch** -- and `lcore_port_map [[0],[1]]`, so port 0's RX
lcore is CPU 5. Port 0's downstream chain sits on very low cores: `process_packet_mask_0` = CPU
**1**, `transpose_voltage_0` = CPU **2**. Port 1's equivalents are CPU 16 and 3.

Every `scripts/gnss/e2e` run this session used `--threads 16` with **no affinity mask**, on
cx19, which also hosts the aggregator and the broker. An unpinned OpenMP job fills CPUs 0-15
first -- directly on top of port 0's entire chain, and next to its DPDK lcore. Port 1 is out of
the blast radius, which is exactly the asymmetry observed (port 0 dropping 100%, port 1
untouched, every time).

That also explains why TRANSIENT contention becomes a PERMANENT wedge, and it is what the
backtrace shows: starvation -> distributor ring overflows -> packets dropped -> the frame
assembler waits for a frame whose packets no longer exist -> every stage parked in
`Buffer::wait_for_full_frame`, none blocked on output. It never recovers because the missing
packets are gone.

**Rule from here: nothing heavy runs unpinned on cx19.** CPUs claimed by the node + aggregator
configs are {0-8, 11-19, 21-24, 31, 57-63}; unclaimed are {9, 10, 20, 25-30, 32-47}. Run
benchmarks as `taskset -c 36-47 ./scripts/gnss/e2e ... --threads 12`, or better, off this node
entirely. The two "mystery" DPDK stalls in 8.7.1 were us.

### 8.7.5 NH20 verified against the ICD

`gps::L5_NH20` = {1,1,1,1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,-1,1} -> bits
**00000100110101001110**, exactly the published GPS L5 Q5 Neuman-Hofman sequence, in phase.
`GPS_L5_Q`/`GPS_L5_Q_NH` were untouched by the merge. So the shared-wrong-template hypothesis --
the one class of defect the bench is structurally blind to, since e2e SYNTHESIZES its test
signal with the same template it validates against -- is dead at the sequence level.

Also settled this round: the broker preserves the search's period EXACTLY (seed argument ==
detection `code_phase_long_chips`, +0.00 chips on all six PRNs, same ref_hop), so the period
defect is not in the hand-off; and `detection_phase`'s `lag_periods` term IS correct for
nonzero lag -- the default bench cp204 happens to give "0 periods", but sweeping cp204 to force
lag values 15, 14, 1 gives the right period every time.

### 8.7.6 The lock blocker is SEARCH LATENCY -- measured 2026-08-02 23:0x

Measured directly (pass-count deltas, not inferred): **1 pass in 90 s**; 248 passes over the
aggregator's 6h08m uptime = **89 s per pass**. `prns_per_pass: 1` with 13 hinted PRNs gives a
**~19 minute revisit per PRN**.

That interacts fatally with two settings:

* `--fit-gap-s 900` (15 min) < the 19 min revisit, so `cp_hist` is RESET on every detection
  (`if h and (ref_hop - h[-1][0]) > MAX_GAP_HOPS: h = []`). The history never reaches two
  points, `fit_cp_rate` never returns, and `code_phase_rate` is **0.0000 for every PRN** --
  observed directly after the broker relaunch. The same failure as the old `--fit-gap-s 16`,
  just at a bigger number.
* `--dll-gain 0.0 --carrier-gain 0.0 --dead-reckon`: the loops are open, so nothing corrects
  what the missing rate leaves behind.

The broker's own five-satellite fit says the local clock runs +0.337 ppm = **3.5 chips/s**. With
no `code_phase_rate` to cancel it and a 19-minute gap, that is ~4000 chips of drift between
updates, against a correlation lobe ~1 chip wide. The tracker cannot hold, which is exactly
`amp_snr` 3-8 and `coherence_s = 0` on every stream.

**So the search-cost work is not an optimisation -- it IS the lock blocker.** The ms-split
(8.4 / the ms-split doc) is 258x cheaper and would take the revisit from ~19 min to seconds.

NOT explained by this, and still open: PRN 28's probe had a seed age of **2.2 s** and was still
+151 chips / 6 periods off. Drift at 3.5 chips/s over 2.2 s is 7.7 chips, so staleness does not
account for it. Either the probe is wrong (it is a Python re-implementation -- VALIDATE IT
AGAINST THE SHIPPED `propagate_seed` VIA `e2e --seed-file` BEFORE BUILDING ON IT), or there is a
second, phase-quantised defect. Do not conflate the two: the latency finding above is solid and
actionable on its own.

Corrections to earlier sections, both from measuring the wrong quantity:
* 8.7 called the seeds "stale by an hour". The seed's `ref_hop` is the FIT ANCHOR for fitted
  PRNs (:2373), not the detection epoch, so that number never meant what I read it as.
* The "frozen PRN" split (1/4/17/25/28/32 vs 2/6/9/16/26) was the fitted/unfitted split, and the
  "smooth Doppler tracks" of the second group are BRDC dead-reckoning output, not measurements.

### 8.7.7 A CONSTANT -4 OVERLAY PERIODS -- 4/4, measured 2026-08-02 23:5x

The probe is validated first (8.7.6 / `seedchk`, worst 2.2e-3 chips vs the shipped
`propagate_seed`), so these numbers stand on their own.

| PRN | oracle ratio | period found vs seeded | offset |
|---|---|---|---|
| 3 | 30.1 | 8 vs 12 | **-4** |
| 28 | 17.4 | 19 vs 3 | **-4** (+16) |
| 1 | 25.8 | 5 vs 9 | **-4** |
| 1 (earlier) | 9.8 | 2 vs 6 | **-4** |
| 32 | 1.6 -- NOISE, discarded | 5 vs 0 | +5 |

Four strong detections, three satellites, seed ages 151-680 s: **exactly -4 every time**, and
-4 == +16 mod 20. Sixteen code periods = 3125 hops = `Mp` = `anchor`, the sample the search
generates repl0 at (`anchor = Mp * fft_len` = 51,200,000 samples = 16 code periods EXACTLY).
The within-period residuals (+54.5, +35.5, -220.5) are the separate latency problem of 8.7.6 --
scattered, age-dependent, and not quantised. The period error is constant and quantised. **Two
independent defects, cleanly separated.**

Ruled out this round:
* The acquire window is NOT shorter than the replica period: `_hops_per_record` defaults to
  `repl_period_hops()` and the stage logs "3125-hop (16.0 ms) coherent window", same as e2e.
* `_snap_start_hop` is the F-engine `sample_seq / fft_len` with frames placed at their true
  offset, and `anchor` is `Mp * _fft_len` in both stage and harness -- the conventions agree.
* Node and aggregator share one absolute sample counter (checked live; the only difference was
  the detection's own age).

**Why the bench cannot see it:** e2e SYNTHESISES its truth with the same absolutely-referenced
overlay convention it then validates, so any constant offset between our overlay-period origin
and real GPS's is invisible offline -- it cancels on both sides. Sky has an independent NH
phase; the bench does not. This is the same blind spot flagged in 8.7.5, and it is the leading
candidate: a replica generated at `anchor` presents overlay period `(16 + nh) mod 20`, not
`nh`, and `detection_phase`'s lift is `(best_nh + lag_periods) * L`.

**Next, and it is cheap:** the offset is constant and measured 4/4, so apply +16 (== -4) periods
to the seed and see whether the tracker locks. That is minutes of work and settles empirically
whether this is THE defect or only one of them -- then find the principled fix. Do not ship a
constant; use it as an experiment.

### 8.7.8 The -4 experiment -- INCONCLUSIVE, and why

Added `--nh-period-offset N` to the broker (default 0, so nothing changes unless asked). It
shifts every seed's overlay period by N primary code periods, applied to **both**
`code_phase_at_ref_chips` and `code_phase_chips` -- the former matters, because
`propagate_seed` prefers `phase_ref_chips` whenever it is >= 0, so offsetting only the argument
would change nothing the tracker reads. `ph_hist` keeps the UNSHIFTED phase so the period
continuity check still compares like with like.

Ran it at -4. Result: **no lock** (`coherent: 0/32` on all six streams, `amp_snr` 2.8-4.7), and
the follow-up probe was **inconclusive** -- the oracle returned ratio 1.6 on both PRNs, i.e. it
found nothing, so no period could be read at all.

The null is NOT evidence against the offset. Checked immediately after: PRN 1 had **no current
detection**, and PRN 3 had fallen from snr 2527 to **294** -- the sky moved on in the hour
between measuring the -4 and testing it. 8.6x less power puts the expected oracle ratio near 4;
we saw 1.6. The test was badly timed, not informative.

The no-lock result is also confounded by 8.7.6 and was expected: with a ~19 min revisit and no
`code_phase_rate`, the within-period phase drifts tens to hundreds of chips between updates, so
a correct period alone cannot lock.

**Live system reverted to `--nh-period-offset 0`.** The -4 measurement (4/4, ratios 30.1-9.8)
stands; it just has not been tested. Re-run when a satellite is strong: pick one with search snr
> 1000, probe it, apply -4, probe again within minutes. Do not leave the constant running in
between -- an unvalidated shift in the live seeds is exactly the kind of thing that gets read as
established later.

Order of work from here: **8.7.6 (latency) first, then re-test 8.7.7.** The latency defect
blocks locking on its own AND confounds every period test, because it is what makes strong
satellites rare in any given probe window.

## 8.8 Deployed 2026-08-03 -- the latency defect is FIXED, and it was not the lock blocker

`channelized_peak`'s fine-lag sign fix (1dffd625b) is live on the aggregator.

| | before | after |
|---|---|---|
| pass | 50.0 s | **6.0 s** |
| revisit (12 hinted PRNs) | 10.4 min | **1.2 min** |
| seed `ref_hop` age | ~4000 s | **0-138 s** |
| seeded `code_phase_rate` | 0.0000 (never fitted) | **-0.0971 ppm** |

8.7.6 is closed: the revisit is now far inside `--fit-gap-s 900`, so `cp_hist` stops resetting on
every detection, the fit forms, and the seeds carry a real rate. Everything that section blamed
for the lock failure is gone.

**And the trackers still do not lock:** `coherent 0/32` on all four streams, `amp_snr` 3.2-4.4.
So search latency was a real defect and a real 8x win, but it was NOT what stands between us and
a lock. What remains is 8.7.7 -- the constant **-4 overlay periods**, measured 4/4 -- which is
now finally *testable*, because seeds refresh every ~1 min instead of every ~20.

Deployment notes, both mistakes worth not repeating:
* Regenerating the aggregator config, I dropped `--acquire-windows 1` and the generator's
  default of **32** applied. Multi-window accumulation smears across NH bins (the 16-vs-20
  defect), and detections collapsed to snr 2.3 against a 2.23 ceiling. Any regeneration must
  carry ALL the original flags: `--acquire-fine-step 128 --prns-per-pass 1 --acquire-threads 16
  --acquire-windows 1`.
* The 8-node config had been running on 2 nodes all night: a 106-channel comb of which **79
  channels were pure noise**. `chord_gnss_agg2.yaml` is the honest config for cx19+cx27.

**Standing recommendation: cx19+cx27 is the WORST live pair.** Offsets 4 and 0 differ by 4, so
g = 4, `s_stored` = 4096, and the 13.09-chip grating lobes -- the original lock blocker -- are
back. cx19 paired with cx42, cx47, cx51 or cx52 gives **g = 1** at identical thermal cost: same
two nodes, same channel count, no lobes, and `refine_span` 512 instead of 16384. Swapping cx27
for any of those is free and strictly better.

## 8.9 The -4 is REAL and VALIDATED -- 2026-08-03

Conditions were finally clean enough to test it: cx19+cx51 (offsets 4,7 -> **g=1**, no grating
lobes), pass 1.05 s, revisit 12 s, seed `ref_hop` ages 0-17 s, `code_phase_rate` fitted.

The probe reproduced **-4 overlay periods** again on both PRNs tested (26: period 6 vs seeded
10; 28: 10 vs 14, within-period only -1.50 chips). That is **6/6** across two node pairs, two
combs (g=4 and g=1), two days and four satellites.

Then `--nh-period-offset -4`, verified reaching the seeds (PRNs 11/21/26/28 all showing exactly
-4 between the detection's period and the seed's -- checked, because an earlier run of this
experiment silently tested a no-op). Re-probed:

```
tracker would despread at cp204 25115.53 (NH period 2)
oracle peak:              cp204 25115.53  ratio 6.87  (period 2)
```

**Exact.** Period error 0, phase error 0, and the oracle ratio ROSE from 3.5 to 6.87 -- the
signal is stronger at the corrected position, which is what a true correction does and what a
coincidence does not. The seed now points at the satellite.

**And the trackers still report `coherent: 0/128`.** So the -4 was a genuine defect and is now
fixed, but it is still not sufficient. `amp_snr` sits at 3-6 across all 32 PRNs on every stream,
which is a noise distribution, and the strongest per-stream entry (P26 at 6.1 on cx51 gnss1) is
the satellite the oracle can only see at ratio 6.87 -- weak.

The live suspicion is therefore SENSITIVITY, not bookkeeping: the tracker despreads on 7
channels (t_stride 16, one GPU's comb) regardless of how many nodes are up, so unlike the search
it gains nothing from more nodes. Yesterday the same oracle saw ratios 17-30 on the same code
path; today's satellites are simply weaker. Before concluding anything, re-run when a satellite
with search snr > 1000 is up -- the probe's own ratio is the honest gauge of whether there is
enough signal for a lock to be possible at all.

**Do not leave `--nh-period-offset -4` running as a magic constant.** It is now validated as
CORRECT, which changes its status from guess to finding, but the principled fix is to locate why
`detection_phase`'s lift is 4 periods (== +16 == Mp, the anchor's own 16 code periods) short and
fix it there. The flag is the experiment; the fix belongs in the code.

## 8.10 `coherence_s = 0` DOES NOT MEAN "NOT LOCKED" -- 2026-08-03

I have been reading `coherence_s` as the lock indicator all session (8.7 onwards). **On CHORD it
cannot be anything but zero**, and that is a property of the configuration, not of the sky.

`GnssCoherentCombiner` computes `coh_s[p]` in exactly three places (1062, 1142, 1309), and all
three are rungs INSIDE a wipe:

```
_wipe_buffer = (_navwipe_bit_records > 0 || !_secondary.empty() || !_l1co.empty());
```

`chord_gnss_cx19.yaml`'s `gnss0_combine` sets **none** of those three: no `navwipe_bit_records`,
no `secondary_overlay`, no per-PRN L1CO. So `_wipe_buffer` is false, the deep block at :850 is
skipped entirely, and `coherence_s` stays at its initialised 0.0 forever. Every "coherent 0/128"
in 8.7-8.9 is that, not a failed detection. The combiner's own "honest no coherent detection"
comment applies when a wipe RUNS and clears nothing -- not when no wipe is configured.

**And the reason it is not configured is a cadence assumption -- the fifth instance.**
`overlay_apply` (gnssChannelizedDespread.cpp:145) computes the overlay chip index as

```
k = round((utc[r] - utc[0]) / rec_dt) + phase   (mod L)
```

i.e. **one overlay chip per RECORD**, stated in the header as "one record = one primary period,
so one overlay chip". True on airspy. On CHORD a record is 2048 hops = **10.4857 code periods**,
so the overlay advances ~10.49 chips per record and flips ~10 times WITHIN each record -- which
the head/tail split (built for records straddling ONE boundary, the 2026-07-15 bistable fix)
cannot represent. Enabling `secondary_overlay` here would run that path and get garbage.

**What CHORD actually needs is simpler than any of the wipes.** Its tracker despreads with
`GPS_L5_Q_NH` -- the 204600-chip code with the overlay BAKED IN (verified in the live
cudaGnssChordTrack command) -- so each record's amplitude already has the overlay removed,
correctly, chip by chip, straddling or not. The right deep integration is therefore a **plain
coherent sum of per-record amplitudes with no wipe at all**, and the combiner has no such rung:
every path to `coh_s` goes through a wipe.

So the deep-integration gain (~15 dB per the combiner's own comment) has never been available on
CHORD, which also reframes 8.9's "sensitivity" suspicion: the tracker is running incoherently at
one record (10.5 ms) when the pilot supports far longer.

**Next:** add a no-wipe coherent rung -- sum `_navbuf[p]` directly over the ladder lengths, with
the same `FLOOR_MARGIN` discipline and a Rayleigh floor that has no alignment search in it
(a straight sum has no max-over-trials bias, so the floor is lower than the wipe rungs' one).
`_navbuf` is only populated under `_wipe_buffer`, so that gate needs widening too. Then
`amp_snr` stops being the only thing we can see.

## 8.11 The merge segfault, the plain coherent rung, and what now blocks the lock -- 2026-08-03

**The merge segfault is fixed** (c5e9e754a). Reproduced deterministically on cx51 under gdb --
`vmovdqu (%rax),%ymm1` with `rax = 0` in `GnssCoherentCombiner::main_thread()` -- and it was one
unguarded loop:

```
for (int p = 0; p < _n_prn; ++p) if (!_navbuf[p].empty()) ++active;
```

`_navbuf` is allocated ONLY under `_wipe_buffer`. CHORD configures no wipe, so it is EMPTY and
the loop runs off the end; the compiler vectorises the `.empty()` checks into a null load. A
conflict-free auto-merge (metrics block from one branch, `_wipe_buffer` gating from the other),
and CHORD-specific for exactly the reason airspy never saw it -- the prototype sets
`secondary_overlay: L5_NH20`, so `_navbuf` is always allocated there. **cx51 has now run
post-merge code for the first time since the merge, with no crash.**

**The plain coherent rung works.** cx51 vs cx19 (premerge control):

| | cx51 | cx19 |
|---|---|---|
| `deep_snr` | **2.81** | 0.0 |
| `deep_records` | **100** | 0 |
| `deep_floor` | **2.177** (= sqrt(2 ln 2)+1, this rung's) | 0.0 |

So `coherence_s = 0` is now an HONEST verdict instead of a structural impossibility:
`deep_snr 2.81 < FLOOR_MARGIN 2.0 x 2.177 = 4.355`. Pure noise gives snr ~1.41, so there is
signal -- it just is not building.

**And it is not building because the carrier is spinning.** `carrier_hz_resid` = -6.6 Hz, which
over the ladder's rungs is 6.96 / 3.48 / 1.74 / 0.84 / 0.42 cycles at 100 / 50 / 25 / 12 / 6
records. A coherent sum needs well under half a cycle. EVERY rung is decohered.

**Closing the carrier loop did not help, and the reason matters.** With `--carrier-gain 0.25`
the per-emit residual swings +-20 Hz between 2 s polls and the trim wanders (PRN 10: +10.7,
+9.9, +13.8, +11.5, +5.0, +3.3, +1.2, +4.8) -- it chases noise. `deep_snr` went 2.81 -> 2.31.
At `amp_snr` ~5 the per-emit carrier discriminator is noise-dominated: you need coherence to
measure the carrier and the carrier to get coherence. Reverted to `--carrier-gain 0.0`.

**Also found: `--dll-gain` was 0.0 against its own default of 0.25** -- the code loop was off
too. Now on.

**Next, and the code already names the way out:** the carrier offset is COMMON-MODE across
satellites ("the converged trim is the chain's deterministic frac-N LO offset, common-mode
across sats" -- the `--carrier-fleet-seed` comment). A per-PRN loop at this SNR cannot see it,
but a FLEET estimate averages ~10 satellites and pulls the noise down by ~3x. The knobs exist:
`--carrier-fleet-seed`, `--carrier-min-sig` (hold the trim when significance is low), and a much
smaller `--carrier-gain` so the integrator averages over many polls instead of tracking each
one. That is the principled configuration, and it should be tried before any more code.

## 8.12 There is NO common-mode carrier offset -- 8.11's diagnosis was wrong

Ran the fleet measurement with both nodes on the deep path (cx19 regenerated with
`deep_coherent`, so its carrier residual comes from the buffered-stream fit rather than the
per-emit squared fallback -- median |resid| 12.57 -> ~3.4 Hz).

46 signed samples across 12 PRNs and 4 streams:

```
fleet mean   -0.436 Hz
fleet median +0.535 Hz
fleet stdev   4.127 Hz
mean is 0.7 sigma from zero
```

**No common-mode offset.** The per-PRN values scatter within a single PRN across streams
(PRN 27: +6.67, -8.03, +2.73, -6.36), which is noise, not a satellite-dependent Doppler error
and not a chain LO offset. So `--carrier-fleet-seed` has nothing to average toward, and the
-6.6 Hz in 8.11 was ONE DRAW from a +-4 Hz distribution, not a measured offset. **8.11's "the
carrier is spinning at -6.6 Hz" is retracted.**

What the 4 Hz scatter actually is: `carrier_resid_hz` fits phase evolution across the buffered
records, so when those records are not coherent the fit returns noise. The residual is a
SYMPTOM of no coherence, not its cause. The loop is circular -- no coherence -> noisy phase fit
-> no carrier correction -> no coherence -- and closing the carrier loop on it (8.11) could only
ever chase noise, which is exactly what it did.

**Breaking the circle needs per-record SNR, and there is a free 3 dB sitting in the config.**
`GnssCoherentCombiner`'s contract is "Per-subband tracker record streams" (plural) combined
coherently -- "one loop at full-band SNR instead of N noise-driven per-channel FLLs". CHORD runs
it as TWO combiners with ONE input each:

```
gnss0_combine  in_bufs=['gnss0_rec_buf']   # GPU 0: 7 channels 5972, 5988, 6004, ...
gnss1_combine  in_bufs=['gnss1_rec_buf']   # GPU 1: 7 channels 5980, 5996, 6012, ...
```

The two GPUs' combs are interleaved (stride 16, offset 8): together they are the node's full
stride-8 comb, 14 channels. One combiner over both `in_bufs` is +3 dB per record, is what the
stage was designed for, and needs no code.

+3 dB alone may not clear the bar (`deep_snr` 2.8 needs x1.56 = +3.9 dB), but the gain is not
just linear: a better per-record SNR also sharpens the carrier phase fit, which is the thing
currently returning noise. That is the lever worth pulling next, and 14 channels is the per-node
ceiling -- combining ACROSS nodes would mean shipping records to one place, which the aggregator
does for the search and nothing does for tracking.

## 8.13 First coherent detections on CHORD -- marginal, but real (2026-08-03)

One combiner over both GPUs (`--combine-gpus`, 010e8f487): 7 channels -> 14, the node's full
stride-8 comb.

| | 7-channel (before) | 14-channel (after) |
|---|---|---|
| best `deep_snr` | 2.81 | **4.77** (peak 7.46 over 2 min) |
| `carrier_hz_resid` median | 3.4-4.4 Hz | **1.32 Hz** (cx19), 2.13 (cx51) |
| `coherence_s > 0` | never, on any node, ever | **5 PRNs cleared the bar** |

**The nonlinear feedback predicted in 8.12 is real and is the bigger half of the gain.** +3 dB
of channels cannot by itself take 2.81 to 4.77 (that is +4.6 dB). What it also did was sharpen
the carrier phase fit -- residual scatter fell ~3x -- and a sharper fit means less decoherence
across the buffered records, which raises `deep_snr`, which sharpens the fit further. The
circle of 8.12 turns the other way once there is enough per-record SNR to start it.

**But it is marginal, and the honest number is a duty cycle, not a lock.** Sampling both nodes
every 5 s for 2 minutes: best `deep_snr` **max 7.46, median 2.58, over the 4.355 threshold 10%
of the time**. Five distinct PRNs cleared it (cx19 16/27, cx51 20/23/27) -- each in 1 of 24
samples. So the chain now reaches coherent detection intermittently on several satellites
rather than never on any. That is a threshold crossing, not a tracking lock, and it should not
be reported as one.

**What this says about the remaining gap:** ~3 dB more would move the median over the bar rather
than the peak. The per-node ceiling is now reached -- 14 channels IS the node's full comb -- so
the next 3 dB has to come from somewhere else:
* **Across nodes.** Tracker records from cx19 and cx51 combined would be 28 channels. The
  combiner already takes N subband streams; nothing currently ships records between nodes (the
  aggregator does that for the SEARCH only). This is the obvious next lever and it is a
  plumbing change, not a physics one.
* **Longer coherent integration**, which is now finally worth something: with the residual at
  1.3 Hz, 0.5 s is 0.65 cycles -- still lossy, but the ladder's shorter rungs are no longer
  hopeless the way they were at 6.6 Hz.
* **The DLL trim**, now that it is on (`--dll-gain 0.25`, restored from an erroneous 0.0): a
  sub-chip code error costs amplitude directly, and the trim had never been allowed to converge.

## 8.14 The lock is weak because the satellites are in the SIDELOBES (2026-08-03)

The arithmetic that should have come first. Link budget for GPS L5Q on one CHORD feed,
on boresight: 6 m dish, eta 0.5 -> 34.4 dBi; Tsys 50 K -> N0 -211.6 dBW/Hz; Prx -158 dBW gives
**C/N0 ~88 dB-Hz**. Over one 10.5 ms record, on the 7 channels the tracker uses (6.7% of the
L5 lobe), the expected per-record SNR is ~4.4e5. **Measured `amp_snr` is ~5.** That is a
**~49 dB shortfall**, which no amount of error bar on Tsys or efficiency explains -- it is not a
sensitivity limit, it is a gross loss.

It is the beam. lambda/D at 1176 MHz on a 6 m dish is **2.44 degrees**, and CHORD is a transit
instrument. Every satellite being tracked right now:

| PRN | elev | off-zenith | amp_snr |
|---|---|---|---|
| 18 | 70 | 20 deg | 1.99 |
| 29 | 59 | 31 deg | 1.39 |
| 15 | 32 | 58 deg | 3.08 |
| 27 |  1 | 89 deg | 4.21 |

All of them tens of degrees out, i.e. deep in the sidelobes -- and `amp_snr` correlates with
elevation not at all, which is what a sidelobe forest looks like. Nothing has been near
boresight all day.

**This reframes the whole afternoon.** Buying +3 dB by merging combiners against a 49 dB deficit
was never going to produce a lock; the "first coherent detections at 10% duty cycle" (8.13) are
just sidelobe-level signal occasionally clearing a threshold. Chasing SNR was the wrong axis.

**And it reframes what success means.** For beam mapping the satellite is SUPPOSED to be off
boresight most of the time -- measuring the sidelobes is the point. So the operating mode is:
point the code phase and carrier from the BRDC model plus broker trims, COAST through the
pattern (`--dead-reckon`), and record the amplitude per subband per feed. A hard lock is neither
available nor required away from boresight. `coherence_s > 0` is a validation signal, not the
product.

What actually matters, then:
1. **Pointing accuracy** -- the code phase must be right so the measured amplitude is the true
   beam response and not a despread miss. That is what the -4 overlay-period fix (8.9) bought,
   and it is worth more than every dB chased since.
2. **A near-zenith transit to validate against.** Within ~1-2 degrees of boresight the signal
   should be ~40 dB stronger and lock trivially. THAT is the test that closes this out, and it
   is a scheduling question, not a code one: find when a GPS satellite passes within a degree or
   so of zenith at 49.32 N, and look then.
3. Per-subband tracking at sidelobe SNR needs the loop closed centrally (broker trims) with the
   model carrying the phase -- not bigger coherent fits, which is where I was heading.

## 8.15 CONSOLIDATED STATE AT COMPACTION #3 (2026-08-03) — read this first

### 8.15.1 Correction to 8.14: it is NOT a signal deficit

8.14 concluded "the lock is weak because the satellites are in the sidelobes". Half right, and
the conclusion drawn from it was wrong. Inverting the measurement instead of predicting it:
`amp_snr` ~5 over a 10.5 ms record on 7 channels is **45.5 dB-Hz full-band-equivalent**.
Receivers track to 25–30 dB-Hz, and a commercial receiver has already done this in CHORD's deep
sidelobes. **The signal is comfortably trackable; the deficit is not the blocker.**

The real limiter is **bandwidth, and it is structural**: the F-engine comb spreads L5 across all
eight nodes, so one tracker instance correlates 7 × 195.3 kHz = 1.37 MHz of a 20.46 MHz lobe —
**6.7%, −11.8 dB**. What the loop sees is **33.7 dB-Hz**. No single instance can ever do better;
the bandwidth is not on the machine. That is the gap, and `docs/CHORD_GNSS_SHARED_DLL.md` is the
design for closing it.

### 8.15.2 The trim gate is CORRECT — do not lower it

`q = 2P/(E+L)` from EMA'd Early/Prompt/Late powers (`trim_pow_alpha` 0.05, N_eff ≈ 39). It is a
peak-sharpness metric, not an SNR: **1.0 with no peak** (all three taps see equal noise power),
**→4.0** for a clean lock at 0.5-chip spacing. `trim_quality_min` 2.2 gates the trim update.

Measured over 2 minutes across all PRNs on sky:

| population | mean q | sigma | max seen |
|---|---|---|---|
| noise (PRNs 5,15,16,18,24,26,27,29,30) | 0.96–1.13 | 0.07–0.31 | **1.87** |
| PRN 20 | 1.584 | 0.244 | 2.186 |
| PRN 23 | 2.381 | 1.461 | 3.449 |

**The noise tail reaches 1.87, so the 2.2 gate sits just above it and is correctly placed.**
Lowering it to "let weak satellites through" walks the trim onto noise peaks — I proposed 1.3
and it would have been 1.5 sigma into the noise. The gate is not the failure; the correlation
peak is genuinely intermittent (PRN 23 swings q from 0.66 to 3.45), which is what 33.7 dB-Hz
looks like.

### 8.15.3 What is deployed right now

* **Search**: 1.05 s/pass, ~12 s revisit (was 50 s / 10.4 min). From the `channelized_peak`
  fine-lag SIGN fix (1dffd625b) — `refine_span` 16384 → 512, 426 despreads → 4.
* **cx19 + cx51**, comb offsets 4 and 7 → **g = 1**, no grating lobes, `refine_span 512`.
  cx19 hosts the aggregator + broker. Both on `build/kotekan/kotekan` (the fixed binary);
  `node_up.sh` defaults to it, `up-premerge` is the fallback.
* **Merge segfault FIXED** (c5e9e754a): an emit-metrics loop indexed `_navbuf` past the end
  whenever no wipe was configured — i.e. every CHORD config. CHORD-specific because airspy sets
  `secondary_overlay` and always allocates it.
* **Plain coherent rung** (`deep_coherent: true`): CHORD's first route to `coherence_s` at all.
  Before it, `coherence_s` was **structurally 0** — every path ran through a wipe, and
  `overlay_apply` advances the overlay one chip per RECORD (true on airspy, where a record is
  exactly 1 code period; false here at 10.4857).
* **`--combine-gpus`**: one combiner over both GPUs, 7 → 14 channels. Produced the first
  coherent detections ever seen on CHORD — but **marginal**: best `deep_snr` 4.77 against a
  4.355 bar, median 2.58, over threshold ~10% of samples.
* **Broker**: `--nh-period-offset -4` (validated, see below), `--dll-gain 0.25`,
  `--carrier-gain 0.0`, all 8 nodes in `--trackers`. Launch scripts in the scratchpad;
  `run_broker_dll.sh` is the current one.

### 8.15.4 The -4 is real and validated — keep it

Six independent measurements across two node pairs, two combs, two days, four satellites: the
seeded overlay period is a constant **4 too high**. Applying `--nh-period-offset -4` puts the
oracle's peak EXACTLY on the seeded position (cp204 25115.53 vs 25115.53, period 2 vs 2) and
RAISES the oracle ratio 3.5 → 6.87. It is a finding, not a guess — but it is still a constant in
a launch script. **The principled fix is to find why `detection_phase`'s lift is 4 periods
(== +16 == Mp, the anchor's own 16 code periods) short.**

This matters far more on CHORD than on airspy, and that asymmetry is the clue: airspy despreads
`GPS_L5_Q` (primary only) and WIPES the overlay afterwards, searching all 20 alignments — so a
wrong seeded period self-corrects every emit. CHORD despreads `GPS_L5_Q_NH` with the overlay
baked in, because a record spans 10.49 overlay chips and a primary-only despread would cancel.
The period must therefore be right up front, with no recovery path.

### 8.15.5 Measurement discipline that cost real time today

* **Time nothing on cx19 unpinned.** The node kotekan runs ~1000% CPU. Identical work measured
  88 s and 693 s. Use `taskset -c 40-51` (verified free) and confirm reproducibility before
  believing any number. Three separate "findings" today were my own interference.
* **`nproc` says 48; the machine has 64.** The shell is masked to
  `0-3,8-19,24-35,40-51,56-63`, so `taskset -c 36-47` silently gives 40-47.
* **Diff every regenerated config against the running one.** Caught three silent breakages:
  `--search-host` rewriting cx19's loopback feed to an external IP, a dropped
  `--acquire-windows 1` letting the default 32 smear NH bins (detections collapsed to snr 2.3),
  and the cx51 port-base mismatch.
* **Rebuild AND RELINK.** `ninja kotekan_stages` does not relink `kotekan`; two crash runs
  tested a stale binary.
* **The node unit is TRANSIENT** — stopping it deletes it, so `systemctl start` can never bring
  a node back. Use `scripts/gnss/node_up.sh`.

### 8.15.6 Next, in order

1. Implement `docs/CHORD_GNSS_SHARED_DLL.md` — §9 has the four steps. Acceptance test is **q
   before/after**, not `coherence_s`.
2. Find the principled fix for the -4 (8.15.4).
3. A near-zenith transit to validate against: within ~1–2 degrees of boresight the signal is
   ~40 dB stronger and everything should lock trivially. If it does not, THAT is a real bug.

# ============================================================================
# 8.16 FLEET-COMBINED DLL, BUILT (2026-08-03 afternoon)
# ============================================================================

`docs/CHORD_GNSS_SHARED_DLL.md` §9 steps 1-4 are implemented and pushed (369ae9d3f, aae275790).
Nothing is measured on sky yet, and §8.16.2 is why.

## 8.16.1 What shipped

* **Combiner** publishes `e_pow` / `p_pow` / `l_pow` / `n_chan` / `pow_hop` / `pow_fft_len`.
  Raw powers, because ratios do not sum. `pow_hop` is the absolute F-engine hop index, taken
  from `GnssChanMetadata::sample_seq` on the input frames -- already stamped by the record
  producers, so no new plumbing; `fft_len` on the combiner is the only new config key. Also
  written to the combined record at `CMB_HOP_SLOT` (int64, free slots 19-20), so a raw capture
  says exactly which window each row is.
* **`p_pow` is NOT `acc_pow`.** `acc_pow` takes whichever straddle-immune form is available
  (`|v_cur|^2`, or `|head|^2 + |tail|^2`), correct for `|A|` and wrong as the P in `2P/(E+L)`:
  the segment split alone costs `f^2 + (1-f)^2`, 0.5 at a mid-record boundary, and would read as
  a collapsing peak. E and L are unsegmented, so P must be too.
* **Broker `fleet_dll()`** behind `--dll-combiners`: group by (PRN, `pow_hop`), sum, one
  discriminator. Fewer than `--dll-min-instances` agreeing instances -> fall back to the single
  combiner, so a partly-down fleet degrades instead of stalling. Trim HOLDS with no live
  `code_phase_rate` (design §7).
* **`--local-trim-gain 0`.** Measured first: every tracker already reported `trim +0.00`, since
  no PRN clears its own 2.2 gate. Nothing was given up.

## 8.16.2 THE PREREQUISITE, and it is not code

`scripts/gnss/e2e`, noiseless, run both ways to be sure:

| search comb | refine | seed handed to the trackers |
|---|---|---|
| stride 4 (**2 nodes** -- today) | -13.63 | **-13.188 chips**, a grating lobe |
| stride 1 (**8 nodes**) | -0.07 | **+0.373 chips** |
| stride 1 + `--seed-cp-rate -7.06e-8` | | **0.373 over 4 records -- CHAIN CLOSES** |

A DLL pulls in +-0.5 chips. Thirteen is 26x outside it, so at 2 nodes no code loop of any kind
can recover, and every lock-quality measurement is measuring the wrong thing. **Check the node
count before concluding anything about tracking.**

## 8.16.3 The gate, and a prediction of mine that was wrong

The design predicted that summing instances would lift `q = 2P/(E+L)` past 2.2. It will not.
`q` is a ratio of means and summing K instances scales every tap's mean by K alike -- a fleet of
eight reports the same 1.5 one node does. What collapses is the **variance**, as 1/sqrt(K): a
signal-free PRN's `q` tightens from a measured tail of 1.87 (K=1) toward ~1.3 at K=8.

The gain is therefore a **lower bar, not a higher statistic**, so any constant gate is right for
exactly one fleet size -- and one chosen for K=1 rejects every real signal once the sum has
tightened the distribution around 1.0. `fleet_dll()` re-measures the floor each cycle as
median + `--dll-quality-sigma` x MAD-sigma over the live `q` population (most tracked PRNs are
signal-free at any moment, so the median IS the no-peak value) and logs it every time.

This also settles what looked like a contradiction with §8.15: 1.3 is 1.5 sigma INTO one
instance's noise and ~3 sigma clear of eight instances'. Same number, opposite verdict, because
the population changed. The rule survives -- never move a threshold to buy sensitivity, measure
the population you are thresholding.

## 8.16.4 Config debt

All eight node configs were patched in place (fft_len + deep_coherent on every combiner,
trim_gain 0 on every tracker) because the generator needs a live production `/config` on 12048
as its base and production is not running. Six of them still predate `--combine-gpus` and run
two 7-channel combiners instead of one merged 14-channel one -- a graph change, not a scalar, so
it wants a real regeneration. It costs 3 dB per instance and doubles the endpoint count; it
costs the fleet DLL no bandwidth, since the sum is over instances either way.

**Regenerate all eight the next time production is up on 12048.**

# ============================================================================
# 8.17 STATE AT END OF 2026-08-03/04 -- READ THIS FIRST IN THE MORNING
# ============================================================================

## 8.17.1 The one real explanation: -4, +16 and the +-2 jitter are ONE thing

```
Mp spans 16 code periods        overlay length 20
16 mod 20 = 16                  -4 mod 20 = 16          <- the same number
measured offset (predicted nh - reported nh) = 16
```

The empirical `--nh-period-offset -4`, the ephemeris-vs-reported offset of +16, and the acquire
window's length in code periods are **one quantity**, reached three independent ways. The
"global clock constant" (8.17.4) is NOT a clock reference: it is the window-length artefact.

The search picks the alignment that best matches over a window spanning 16 periods, and the
reported `nh` is anchored at one end of that window while the phase it is paired with refers to
the other -- off by exactly 16 == -4 (mod 20).

That also explains the **+-2 jitter** and why propagating our own `nh` forward failed (4 of 26
correct). The window start lands at a different FRACTIONAL position inside a code period every
snapshot -- 195.3125 hops per period never repeats -- so "which period index does the window
start in" shifts by +-1 on rounding, and +-2 once the 16-period span and Doppler drift are in.
It is quantisation we were reading as noise.

**Consequence:** it does NOT re-randomise across restarts (it is geometry, not timing) --
correcting what was said earlier in the session. **Fix:** do not add another offset constant.
Make the search report `nh` anchored to a STATED end of the window and carry the period count
explicitly instead of letting it fall out of a modulus. This is the FOURTH defect from the same
root (16-period window vs 20-chip overlay, neither dividing the other).

## 8.17.2 What was fixed, measured, on 2026-08-03/04

| | start of day | end |
|---|---|---|
| replica build | 1223 s | **0.11 s** (factorised + disk cache) |
| search pass | 106 s | 2.05 s measured, then 12 s -- see 8.17.3 |
| revisit | 1276 s | ~25 s at best |
| code walk per seed update | 765 chips | 0.29 chips |
| fleet DLL | did not exist | 14 instances, 106 channels, self-calibrating gate |
| satellites at a real peak | 0 (all pinned at q = 1.0) | several at q 1.9-3.1 |

Also fixed: `refine_span` missing from the deployed aggregator (426 evals -> 14, 30x);
cx51 writing cx27's aggregator feed slot; `dish_coelev_deg` 18.7 deg wrong; `--nh-overlay-len`
defaulting to B1C's 1800 on an L5 chain; the fleet hop-window sized from one sample of a
free-running quantity (fleet silently flapping 14 -> 8 instances).

## 8.17.3 PICK UP HERE, in this order

1. **The search bottleneck, and the 2.05 s -> 12 s regression.** Same instance, no code change
   between: 44 passes/90 s became 10 passes/120 s. Measured at the slow rate: 36% CPU on a
   16-thread stage while DROPPING 17.7% of input frames. Dropping means the consumer is behind,
   so it is not starvation; 36% means it is not compute-bound either. Something serialises the
   pass. **Instrument the pass with stage timers** (the `GNSS_MSSPLIT_PROFILE` pattern) rather
   than inferring from CPU percentages.
2. **Fit `dop_rate` ourselves instead of taking BRDC's.** BRDC seeded PRN 3 at -0.4699 Hz/s
   where the Doppler track measured -0.578; the 0.108 Hz/s residual is the quadratic walk. The
   search reports `doppler_hz` every pass, so its slope is a direct, well-conditioned fit -- at
   a 25 s revisit, four points span 100 s over which Doppler moves ~58 Hz against ~1.5 Hz
   per-detection noise. **Fit the DOPPLER, not a quadratic on cp**: cp0 is an argument with the
   5094.9 chips/Hz lever and re-opens the currency problem; Doppler has no such lever.
3. **Burst the round-robin for BOOTSTRAP only.** A sat with no fit needs 3 points before
   `code_phase_rate` exists -- today 3 x revisit = 75 s, but 3 x pass ~ 6 s if bursted. Uniform
   round-robin once fitted: bursting a fitted sat makes its revisit k times WORSE, and clustered
   points are a poorly-conditioned slope fit. The cursor already exists; it needs to prefer
   unfitted PRNs.
4. **Hoist the data FFT out of the nh loop.** In `channel_correlate_into` the data transform
   depends only on the snapshot and channel -- not on PRN or `nh` -- yet sits inside the
   alignment loop: 20x redundant (5x with a hint). The REPLICA FFT genuinely cannot be factored
   (R_nh = s0[m]*head[m] + s1[m]*tail[m] with signs varying per hop, so the FFT does not
   distribute). `prns_per_pass` only becomes worthwhile AFTER this hoist -- that transform is
   the sole work shared across PRNs.
5. **Verify the nh hint actually narrows.** The offset now calibrates legally (15/16) but the
   pass time did not obviously drop. Check `nh_scan.size()` is 5, not 20.
6. **The discriminator is RAILED** (|disc| ~ 0.9 most of the time), which is why the leak
   experiment failed (8.17.4). A railed disc carries only the SIGN of the error. The honest fix
   is a seed inside the linear region, i.e. 8.17.1 -- not a bigger integrator.
7. **PRN 19 passes 0.40 deg off boresight at 08:04 UTC, daily.** Full dish gain, ~10^4x the
   power of a deep-sidelobe sat. The validation pass: if it does not lock hard, sensitivity is
   exonerated and it is a bug. PRN 9 at 1.31 deg / 02:06 UTC is the half-power warm-up.

## 8.17.4 Measured and REJECTED -- do not re-try without new evidence

* **Weakening the DLL leak on the fleet path** (`--dll-leak-present` 0.01, ceiling 1.25 -> 6.25
  chips). Strictly WORSE: best-q median 2.05 -> 1.03, samples above q 2.0 went 4/8 -> 0/8, every
  sat driven to noise, PRN 9's disc flipping -0.984 -> +0.981 as its trim ran a chip through the
  peak. The leak is not only a ceiling: with a RAILED discriminator `tau` saturates and carries
  only the sign, so the integrator pushes one way indefinitely and the leak is what bounds the
  excursion. Reverted to 0.05.
* **Echoing our own reported `nh` back as a hint.** 4 of 26 propagations correct, with or
  without the argument-vs-phase correction -- explained by 8.17.1.
* **Gating the fleet DLL on q.** q is peak-SHARPNESS: high only when already ON the peak, so it
  vetoes exactly the shoulders a DLL exists to pull in. Replaced by a signal-present test on
  summed prompt power.
* **Lazy replica building.** Deferred the 1223 s rather than removing it, and made passes 7x
  slower while the cache filled. Superseded by the exact factorisation.
* **ms-split.** Acquire 77x cheaper (0.55 s vs 42.18 s) but it was only ~30% of the bill then.
  Now that `refine_span` is fixed the acquire IS ~92%, so this is worth REVISITING -- against a
  32.55 s NH postfix, which the nh work may make unnecessary.

## 8.17.5 FFT lengths -- asked and answered

`Mp = 3125 = 5^5`: smooth, single small prime, FFTW has radix-5 codelets. Not a `2^N-1` or
large-prime pathology. And padding is NOT available: the correlation is CYCLIC over the code
period, so zero-padding to 4096 would make it linear and destroy the wrap-around. 3125 is also
not free to choose -- it is `lcm(code period, hop)/fft_len`, the shortest window that is a whole
number of both, and every legal alternative is a multiple of 5^5.

# ============================================================================
# 8.18 STATE AT COMPACTION #4 (2026-08-04 15:00 UTC) -- READ FIRST
# ============================================================================

## 8.18.1 SUSTAINED CODE LOCK, at last

Three to six satellites hold `q` = 2.5-3.4 continuously (theoretical max 4.0, floor ~1.09),
14 instances, 106 channels. Yesterday morning every PRN was pinned at exactly 1.0. Nothing was
wrong with the loops -- the seed went stale faster than it could be used.

| | 08-03 morning | now |
|---|---|---|
| replica build | 1223 s | **0.11 s** |
| search pass | 106 s | **0.90 s** |
| revisit | 1276 s | **11 s** |
| code walk per seed update | 765 chips | **0.055 chips** |
| sats at a real peak | 0 | **3-6 at q 2.5-3.4** |

The chain, each measured: `refine_span` 512 (426 evals -> 14, and the deployed aggregator was
simply MISSING the key); nh factorisation (20 replica builds -> 2, exact via head/tail through
the nav_bit hook); disk cache (121 s -> 0.11 s); `refine_hops` 391 (16 code periods -> 2,
bit-identical answer under 20x noise).

## 8.18.2 THE OPEN PROBLEM: coherent detection, and it is per-record SNR

`coherence_s` is still 0 almost everywhere; `deep_snr` sits at 1.2-3.6 against a 2.18 floor,
with occasional excursions to 11.4 -- so coherent detection IS achievable here, intermittently.

The cause is measured: **sigma_phi ~2.2-2.55 rad over the 1.05 s deep window**, i.e. coherent
retention exp(-s^2/2) = **0.039, a 29 dB loss**. That is why PRN 3 shows search snr 1400 and
deep significance 2-3 sigma: the search integrates 16 ms where drift does not matter, the deep
integrates 1.05 s where it dominates.

**BOTH routes to fixing it are noise-dominated at CHORD's per-record SNR (amp_snr 3-5):**

* `carrier_hz_resid` (direct phase-slope fit) is SIGNAL-FREE: |resid| median 0.519 Hz on sats
  with signal, 0.492 Hz on sats without. The carrier loop was integrating noise -- warm-vs-warm
  A/B showed it made things WORSE (deep_snr mean 1.265 vs 1.447 off, best 6.9 vs 11.4). OFF.
* Code-derived carrier (fitted cp slope x f_carrier/f_chip = 115.03 Hz per chip/s; ratio itself
  validated in 7.2) swings tens of Hz between consecutive samples. The ratio is also a 115x
  NOISE amplifier: 0.05 chips/s of slope scatter -> +-5.75 Hz.

**So the constraint is per-record SNR, not the estimator.** At 2.2 rad / 1.05 s the phase
wanders ~2.1 rad/s, capping the usable coherent span at ~**0.25 s** (0.5 rad).

## 8.18.3 PICK UP HERE

1. **Dump `deep_snr` PER RUNG, not just the winner.** The ladder already walks short rungs
   (`min_rung(ub, 4)` = 4 records = 42 ms). Over 0.25 s the phase DOES cohere, so the question
   is whether ~23 records give enough gain to clear the floor. Short rungs failing on COHERENCE
   and failing on SNR want opposite fixes, and right now we cannot tell which it is. One
   measurement, and it gates everything else here.
2. **Cross-node coherent combining, for 8.8 dB of per-record SNR** (14 of 106 channels today).
   Better than I framed it in the shared-DLL design: coherent summing ACROSS FREQUENCY already
   works inside a node (`gr += rec[3]`), time alignment is exact and free from `sample_seq`, and
   the data volume is the same handful of numbers the fleet DLL already ships. What remains is
   a per-node instrumental phase offset -- constant per node per channel, calibratable exactly
   like the nh constant. And it HELPS the phase problem rather than depending on it: more SNR
   per record -> less phase noise -> longer coherent span.
3. Remaining search items, now smaller: the nh anchor fix (span 2 -> 0, ~27% of the mean pass,
   and it fixes a real correctness wart -- see 8.17.1); the data-FFT hoist is only ~3% and was
   over-sold in 8.17.3 item 5.
4. Bootstrap bursts for unfitted sats (3 x pass instead of 3 x revisit to first fit).
5. **Beam anomaly, deprioritised but unexplained:** PRN 19 passed 0.40 deg off the assumed
   boresight and peaked at snr 74, while PRN 28 at 4-22 deg ELEVATION reaches thousands.
   Separation from boresight does not predict strength (Spearman +0.20). `beamlog.py` is
   accumulating az/el vs strength to settle it. Does not block locks, which is why it was
   deprioritised -- strong detections are strong regardless of why.

## 8.18.4 MEASURED AND REJECTED TODAY (do not re-try without new evidence)

* **Weakening the DLL leak** (`--dll-leak-present` 0.01): best-q median 2.05 -> 1.03, 4/8 -> 0/8
  above q 2.0. With a RAILED discriminator `tau` carries only the SIGN, so the leak is what
  bounds the excursion, not merely a ceiling.
* **Carrier loop on** (`--carrier-gain 0.25`): worse on every metric, sigma_phi unchanged.
* **Code-derived carrier**: unstable by tens of Hz (8.18.2).
* **Allocation hoist in `refine_peak`**: predicted to be the cost on the ms-split precedent;
  measured 3.59 s vs 3.62 s, i.e. nothing. Kept (correct, removes a hazard) but it was not the
  win, and `refine_hops` was.
* **Lazy replica building**: deferred the cost instead of removing it; superseded by the exact
  factorisation.

## 8.18.5 THE STANDING CHECK (four instances in one day)

Four separate loops were found consuming a statistic that does not measure what its name says:
`q` (peak sharpness, so it vetoed exactly the shoulders a DLL exists to pull in), the reported
`nh` (anchor moves, so it does not propagate), `carrier_hz_resid` (signal-free), and the
code-derived carrier (noise-amplified 115x).

**Before wiring any statistic into a loop: measure it ON SIGNAL and ON NOISE, and measure
whether it is STABLE.** Two cheap tests, and each of the four would have been caught by one of
them. A corollary learned the hard way today: never validate a new statistic by comparing it
against one already known to be bad.

Related discipline, also re-earned: characterise a DISTRIBUTION before quoting it. The "2 -> 12 s
search regression" and "the search is 82% idle" were both artefacts of reading a handful of log
lines instead of the whole population.

## 8.18.6 Running state

All 8 nodes active. Aggregator on agg8 (`GNSS_SEARCH_PROFILE=1`), broker via
`scripts/gnss/broker_up.sh` (carrier-gain 0, dll-gain 0.25, nh-hint span 2, publish-port 12060),
viewer on 12060 through the broker's publisher (one origin for both `get_status` and
`get_detections`; it cannot straddle two ports -- `host: location.hostname`, so tunnel 8080 and
12060 or proxy through the viewer). `beamlog.py` accumulating. Tree clean, all pushed.
