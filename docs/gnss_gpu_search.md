# Moving the acquisition search onto the GPU

*Scoped 2026-08-05. Baselines in here are measured on cf06 (2x L40S, 64 cores), not estimated.*

## 1. Why now, and what the real problem is

Every search speedup to date has been a *parameter* that buys latency by not searching:

| knob | live value | blind value | what it costs |
|---|---|---|---|
| `require_hint` | true | false | only PRNs the broker already found are searched at all |
| `nh_hint_span` | 2 (-> 5 alignments) | 20 alignments | 4x |
| `prns_per_pass` | 1 | 32 | 32x |
| `doppler` window | broker hint, a few bins | 321 bins | ~30x |
| `acquire_windows` | 1 | 4-64 | integration length |
| `acquire_fine_step` | 128 | 1 | 128x (this one is honest -- see 5q) |

That is why a live pass is 0.6 s: it is not a search, it is a *re-acquire of things already known*.
The instrument has no cold-start path that finishes in useful time, and every one of those
multipliers gets worse with a second constellation.

**Measured blind cost of the surface aggregate** (`scripts/gnss/aggbench`, cf06, 16 threads,
nc=79, n_dop=321, Mp=3125, sph=16384, fine_step=128):

    CPU direct : 0.667 s   121.6 GFlop/s   (81.1 GFlop)   <- per NH alignment, per window

x20 NH alignments = 13.3 s per PRN; x32 PRNs = **427 s per snapshot for GPS L5 alone**. That is
the historical 13-minute pass, and it is ~100% of it (5q: the surface is the entire cost).

The scaling that motivates this thread is worse than 32x, because the NH multiplier is
per-signal and L5 has the *short* secondary code:

| signal | sats | secondary length (the NH multiplier) |
|---|---|---|
| GPS L5 Q5 | 32 | 20 |
| Galileo E5a Q | 36 | 100 |
| BeiDou B2a Q | ~46 | 100 |
| GPS L1 C/A | 32 | 1 (but 1 ms code -> different Mp geometry) |

A blind all-constellation pass on CPU is hours. This is the wall.

## 2. What does NOT change

Two settled results constrain the design, and neither is being re-litigated:

* **The 20-way NH scan stands.** 5q derived an exact-looking restructure (one correlate ->
  all 20 alignments as sign-weighted sums of shifted copies) and it failed *on the
  representation*: the cross-channel reconstruction `D` is only approximate for the windowed
  PFB, and coherently summing copies at sub-hop offsets needs far more accuracy than localizing
  a peak does. Do not re-derive it. The GPU's answer to a 20x multiplier is to *be 200x faster*.
* **`acquire_fine_step` is the one honest reduction** and it is already taken (7.4x, 0.03 chips).

## 3. The one new algorithmic result: the fine axis is an exact DFT

    D(q,i) = sum_c P_c(q) e^{+i 2 pi f_c (i * s_step) / sph}

When `s_cols * s_step == s_stored` -- which is the live configuration exactly (128 * 128 =
16384) -- the ramp is `e^{+i 2 pi (f_c - f_0) i / s_cols}` up to a common unit-modulus factor
that `|D|^2` discards. It therefore depends on the channel **only through `(f_c - f_0)/g mod
s_cols`**. So: fold the `nc` channel values into an `s_cols`-length bin array, take one
`s_cols`-point inverse DFT. Not an approximation of the direct sum -- the direct sum,
reassociated.

    direct : nc * s_cols complex MAC = 79 * 128 * 8   = 80896 flop/row
    folded : nc adds + one 128-pt FFT ~ 5*128*log2(128) = 4480 flop/row      (18x fewer)

**This is not the reverted coarse fold of 5o.** That one assumed the *replica* was periodic in
the window and died on the NH overlay. This is an identity about the *ramp*; it assumes nothing
about the signal. `aggregate_folded` in `scripts/gnss/aggbench.cpp` refuses to run when its one
arithmetic precondition fails, rather than falling back to something close.

Validated at blind dimensions against the shipped `gnss::aggregate_accumulate`:

    max|diff| 2 vs surface max 1.0e7  -> rel 2.0e-07     (float round-off)
    argmax ref == fold == planted cell
    snr ref 63440.057  fold 63440.071

**But it is worth ~2.2x, not 18x, on either processor** -- because it stops being a flop problem
and becomes a bandwidth problem. Measured (see §5): CPU 0.507 -> 0.202 s, GPU 5.3 -> 2.5 ms. The
folded GPU kernel runs at **673 GB/s of the L40S's 864 GB/s peak**, i.e. it is within 22% of the
memory system and no further flop reduction can help it.

The fold's real value is therefore *not* the 18x. It is two things:

1. **It removes an 81 KB table.** The direct path needs the `[nc][s_cols]` ramp, which does not
   fit in shared memory at CHORD scale and so streams from L2 -- that is why the direct GPU
   kernel stalls at 312 GB/s. The folded path needs `nc` ints plus an `s_cols/2` twiddle table,
   both of which live in shared memory.
2. **It scales with `nc`.** Direct cost is linear in the channel count; folded is `nc` adds plus
   a fixed `s_cols`-point transform. At `nc = 79` that is 2.2x; a wider band at `nc = 200` makes
   it ~5x. Since the whole point of this thread is *other bands and constellations*, that
   asymptote is the reason to keep it.

## 4. Where the time actually goes, and what has to move

Per (PRN, NH alignment, window):

| step | cost | note |
|---|---|---|
| replica materialise | 0.002 s / 5 align | elementwise, already negligible |
| `FFT(data)` | recomputed per (PRN, NH) | **depends on neither** -- hoist, free 640x |
| correlate: `nc x n_dop` IFFTs of len 3125 | ~4.6 GFlop, 634 MB out | memory-bound |
| **aggregate** | **81 GFlop, 0.667 s** | **the whole cost** |
| peak (max/mean over 128M cells) | host scan of 1.03 GB | never needs to leave the device |

The live config also misses a free 2x: `doppler_step: 31.25` is *half* the transform's own bin
spacing `Fs/(Mp*sph)` = 62.5 Hz, so `channel_correlate_into`'s bin-aligned path (one forward FFT
serving every Doppler trial) is **not** being taken. On the GPU always use the bin-aligned grid;
the parabolic refine in `channelized_peak` already recovers sub-grid Doppler.

## 5. The GPU decomposition

The batch axes are enormous and independent: **PRN (32) x NH (20) x Doppler (321) x channel
(79)**. The Doppler x channel product alone is 25359 independent length-3125 transforms, which
saturates an L40S without touching the other two.

```
ONCE per snapshot                     ONCE at startup
  upload snapshot   2 MB                upload replica bank head/tail   126 MB
  A[c][k] = FFT(data_c)                 (32 PRN x 79 ch x 3125 x 2 x cf)
    nc transforms, len Mp

PER (PRN, NH)                                              device-resident, no PCIe
  1. materialise  R[c][m] = head*sgn0[m] + tail*sgn1[m]     elementwise 247k
  2. conjB[c][k]  = conj(FFT(R_c))                          nc transforms len Mp
  3. correlate    P[c][d][q] = IFFT_q( A[c][k+b_d] * conjB[c][k] )
                                                            batch nc*n_dop = 25359
  4. aggregate    fold nc -> s_cols bins, 128-pt IFFT, accumulate |.|^2
                                                            batch n_dop*Mp = 1.00M rows
  5. reduce       max + mean over the surface               on-device
PER SNAPSHOT
  download        peak cell + value + mean, per (PRN, NH)   ~16 bytes
```

**PCIe traffic per snapshot: 2 MB up, kilobytes down.** The 1.03 GB surface never crosses the
bus and is never scanned by the host -- which is a structural change, not just a speedup: today
the surface must be materialized in host RAM because that is where the peak search lives.

### MEASURED, A1 as built (cf06, one L40S, blind dimensions)

`aggbench --nd 321 --mp 3125 --threads 16 --reps 5 --gpu`:

| | time | rate | vs CPU |
|---|---|---|---|
| CPU direct (shipped `aggregate_accumulate`, 16 threads) | 0.507 s | 160 GFlop/s | 1x |
| CPU folded | 0.202 s | | 2.5x |
| GPU direct | 5.3 ms | 312 GB/s (compute-bound: the 81 KB ramp) | **84x** |
| **GPU folded** | **2.5 ms** | **673 GB/s of 864 peak** | **180x** |
| GPU peak+mean over 128M cells | 0.9 ms | 567 GB/s | (was a 1 GB host scan) |

Correctness at blind dimensions, against the double-precision CPU surface:

    gpu-direct : rel 0         argmax == planted cell   snr 63440.057  (CPU: 63440.057)
    gpu-folded : rel 2.99e-07  argmax == planted cell   snr 63440.077

So per (PRN, NH, window) the surface work is **3.4 ms against 507 ms**, and the peak amortizes
further when `acquire_windows > 1` (it runs once after all windows, not per window).

    blind GPS L5 aggregate+peak, 1 GPU : 20 NH x 32 PRN x 3.4 ms = 2.2 s   (vs ~324 s CPU)

Add the projected device correlate (~1.9 ms, bandwidth-bound on 1.3 GB) and a full blind
GPS L5 snapshot lands near **3.4 s on one GPU, ~1.7 s on both** -- against 427 s today. That is
the headroom to turn `require_hint` off and add constellations.

### Still to come: fusion (A4)

Steps 3-5 tile over Doppler so `P` never round-trips to HBM. A Doppler tile of 16 gives
`P_tile = 79 * 16 * 3125 * 8 B = 31.6 MB`, resident in the L40S's 48 MB L2. Since both kernels
are already bandwidth-bound, removing a 0.63 GB round trip is where the next factor lives -- not
in any further flop reduction.

## 6. Staging

Correctness first, throughput second -- the same G1a/G1b/G1c discipline the despread migration
used, and for the same reason: a search that is fast and subtly wrong is indistinguishable from
one that is slow, until weeks later.

* **A1 -- device aggregate + on-device peak. DONE 2026-08-05**, numbers above.
  `lib/cuda/cudaGnssAcquireKernel.{hpp,cu}`, driven by `scripts/gnss/aggbench --gpu`.
  *Not deployable alone*: `P` is 634 MB per NH, so shipping it over PCIe (~25 ms) would cost
  more than the 505 ms the kernel saves. A1 is a correctness milestone, not a release -- which
  is exactly why it was built against a planted peak and the double CPU surface rather than
  wired straight into the stage.
* **A2 -- device correlate. DONE 2026-08-05.** `lib/stages/gnss/GnssCudaAcquire.{hpp,cpp}` driving
  the kernels, validated end to end by `scripts/gnss/acqbench` against
  `gnss::channelized_accumulate` on identical inputs with a planted delay + Doppler.

  | n_dop | CPU (16 thr) | GPU chain | speedup |
  |---|---|---|---|
  | 11 | 0.046 s | 0.55 ms | 84x |
  | 19 (typical hinted) | 0.070 s | 0.59 ms | 118x |
  | 41 | 0.145 s | 1.2 ms | 124x |
  | 161 | 0.52 s | 4.7 ms | 111x |
  | **321 (blind)** | **1.15 s** | **8.3 ms** | **113x** |

  Surface agrees to `rel 5.9e-08` at every size, same argmax cell, same SNR to 7 digits
  (153982.541 vs 153982.537 at blind dims). 1.16 GB of device memory per engine.

  Blind GPS L5 is then `20 NH x 32 PRN x 8.3 ms` = **5.3 s per snapshot on one GPU**, against
  427 s. Hinted, all 32 PRNs, 5 alignments: **95 ms** -- ~10% duty at a 1 s cadence.
* **A3 -- batch over (PRN x NH)** in one launch set, replica materialise on device. KV settled
  two policy questions here on 2026-08-05, and they change what A3 builds:

  **Drop the round-robin; search every PRN every pass.** The cost argument is easy -- all 32
  PRNs hinted is ~95 ms, ~10% of one GPU at a 1 s cadence, and one rotating *blind*
  reacquisition slot (1 PRN, 321 bins x 20 NH) adds ~166 ms. But the better reason is not
  speed: `prns_per_pass: 1` gives a ~1276 s per-PRN revisit, which exceeds the broker's
  `--fit-gap-s 900`, so `cp_hist` resets on every detection and **`code_phase_rate` is never
  fit at all**. Searching every PRN every pass closes that as a side effect.

  **Throttle to ~1 s, deliberately leaving the GPU idle.** Faster buys nothing real: the
  snapshot is 16 ms of data, and seed staleness costs 0.0087 chips/Hz/s, so a 1 s-old seed
  carries ~2 chips against the ~13 minutes we have today. The headroom is not waste -- it is
  what the other signals will spend. Galileo E5a alone (36 sats x a **100**-length secondary
  x ~19 bins) is ~1.1 s of GPU per pass and already exceeds a 1 s cadence on one device. The
  NH multiplier is per-signal, and L5's 20 is the *small* one.

  So A3's shape: all eligible PRNs per pass at their hinted grids, plus a rotating blind slot,
  with a configured GPU duty-cycle budget rather than "go as fast as the hardware allows".
* **A4 -- fuse 3-5** with Doppler tiling for L2 residency.
* **A5 -- refine on GPU. THE WHOLE REMAINING COST.** Measured live 2026-08-05 with A2/A3 on:
  `pass 0.473 = acquire 0.0031 (0.7%) + refine 0.465 (98.2%)`. Everything below is now noise
  next to this.

  **What it is.** `gnss::refine_peak` scans `n_eval = 2*span/step + 1` trial code phases (live:
  span 512, step 103 -> 10 evals), and at each one generates a full channelized replica
  (`hoprate_stream_into`) over `refine_hops` (391) x 79 channels and takes `|<data, replica>|^2`.
  Synthesis + MAC, exactly what `GnssCudaDespread` does -- and its replica semantics ARE
  `hoprate_stream`, so this is a reuse, not a rewrite.

  **The clean mapping.** The scan is UNIFORM (`cp0 + k*step`), and `DespreadJob` already emits a
  triple at `cp-ds, cp, cp+ds`. So three consecutive trial phases = ONE spec with
  `spacing_chips = step * cps`, and 10 evals become 4 specs. No new kernel.

  ⚠️ **THE BLOCKER: the GPU despread has no secondary overlay.** `DespreadJob` carries only
  `m_head` -- a split of the prompt at ONE code-period boundary, so the record assembler can
  apply the NH sign afterwards (gnssRecord slots 16-18). But the refine window is
  391 hops x 16384 = 6.406e6 samples = **2.00 code periods**, so it spans TWO boundaries and a
  one-way split cannot reconstruct it. The CPU path has no such problem: it passes `nh_phase`
  straight into the generator and the replica carries the overlay per chip.

  Three ways out, in preference order:

  1. **Emit PER-CODE-PERIOD partial correlations** instead of one overlay-aware sum, and let the
     host apply the signs. Strictly more general than `m_head`, and it hands the SEARCH all 20
     NH alignments as sign-weighted sums of the same partials -- the "direction that IS exact"
     from 5o, which failed there only because it was attempted on the approximate cross-channel
     reconstruction `D`, not on exact despreads. Here there is no such approximation.
  2. **Add `nh_phase` to `DespreadJob`** and apply the sign per chip in the kernel. Smallest
     change, matches CPU semantics exactly; but it touches the kernel the TRACKER depends on, so
     it needs the `cuda_gnss_despread_test` suite green before deployment.
  3. **Shorten `refine_hops` below one code period** (< 196) so `m_head` suffices. Cheapest, but
     it trades integration length for implementation convenience -- acceptable only because the
     refine merely LOCALISES a peak that already cleared detection. Measure the peak margin
     before accepting it; 5o's note about the true lobe beating the grating lobe by only ~15% is
     the number at risk.

  **RESOLVED 2026-08-05 -- there was no blocker.** The analysis above is kept because the
  MISTAKE in it is the useful part. `GPS_L5_Q_NH` declares `code_length 204600` (= 10230 x 20,
  the NH overlay BAKED INTO THE CODE TABLE) and `secondary_length 0`, so:
  * `nh_phase` is INERT for this signal -- `hoprate_stream`'s overlay branch never fires;
  * the GPU's code lookup carries the overlay for free, with no kernel change;
  * the effective code period is **20 ms, not 1 ms**, so the 391-hop (2.0 ms) refine window
    holds NO code-period boundary at all, and `m_head`'s single split is exactly right for a
    10.49 ms record.

  Computing the window in PRIMARY periods says "2.00 periods, two boundaries" and sends you off
  to generalize the kernel for nothing. Check `code_length` against `eff_code_length` and
  `secondary_length` BEFORE reasoning about overlay boundaries.

  **A5 IS THEREFORE A PURE REUSE, and it is DONE**: `gnss::refine_peak_cuda`
  (lib/stages/gnss/gnssSeedTransport.cpp), driven by `use_cuda_refine` (defaults to follow
  `use_cuda_acquire`). Validated by `scripts/gnss/e2e --cuda-refine`, which runs BOTH refines on
  the same data and prints the delta:

      live geometry  (span 512, step 103): cp 1622.465213 both, delta 0.000e+00,  8.9x
      default        (span fft_len, 303) : cp 1609.153425 both, delta 0.000e+00, 52.2x
      SEARCH LEG ERROR unchanged at -13.188 chips (the 2-node grating-lobe figure)

  Both speedups are with the harness chunking ONE spec per launch (`n_prn = 1`); the live stage
  has 32 PRN slots, so its 4 specs go in a single launch.

  ⚠️ `despread_batch`'s job arena is `n_prn`, NOT `gnss_gpu::max_specs(n_prn) = n_prn * MAX_REC`
  -- that larger bound belongs to the DEVICE path (`enqueue_batch_device`), whose per-frame slice
  holds every record in flight. Confusing them over-batches by MAX_REC and the only symptom is
  `jobs upload: invalid argument` from the H2D, a long way from the cause. `max_batch_specs()`
  now exposes the right bound and `refine_peak_cuda` chunks against it.
* **A6 -- other signals.** Only after A2 is on sky, because a 100-length secondary code is a 5x
  larger NH multiplier and will find any remaining cost that scales with it.

## 6b. FIRST LIVE RUN, 2026-08-05 17:44-17:52 -- works, and exposed something else

Deployed on cf06 via `chord_gnss_agg6_cuda.yaml` (patched from `chord_gnss_agg6.yaml` by exactly
two keys: `use_cuda_acquire: true`, `doppler_step` 31.25 -> 62.5). Rolled back after ~8 min.

**The acquire itself: confirmed.** Over 200 live passes each:

    CPU acquire  0.1375 s      GPU acquire  0.0030 s      46x
    pass composition after: acquire 0.9%, refine 97.6%   <- A5 is now the whole cost

Same 7 PRNs detected, comparable SNRs, ZERO `declined this Doppler grid` fallbacks, 907 MiB
resident. Doppler differed between the runs by -0.07 to -0.28 Hz/s, all NEGATIVE -- correct sky
motion (GPS Doppler falls monotonically through a pass), not a bias.

**What it exposed: the fleet-coherent gate has ~2 records of margin.**
`fleet_coherent` needs `--coh-min-records` (default **32**) COMMON record hops across >= 3
instances. Measured on PRN 10:

| config | common hops | instance window spread |
|---|---|---|
| CPU | **34** (vs 32 required) | 34 records |
| GPU trial | 0-16 | 205-263 records |

and sampling repeatedly on one config gives 0, 4, 12, 16, 34 -- it FLUCTUATES across the
threshold, so the combine fires intermittently and `deep_snr` alternates between the fleet value
(hundreds) and the single-instance value (~14) with nothing in the row saying which regime it is
in. That is why a 1160 in the morning and a 14 in the afternoon can both be "normal".

⚠️ **UNPROVEN**: the GPU config demonstrably had a 6x larger window spread, but the only
mechanism identified is a 1.44x faster pass (per-PRN revisit 16.3 s -> 11.3 s), which is a weak
lever for a 6x effect. Do NOT record "the GPU search broke the deep fold" as fact. What IS
established: the gate is marginal on BOTH configs, and after rollback the published rows still
showed no `fleet:N` even though a direct `fleet_coherent(min_rec=32)` call returned 17 PRNs.

**The real question this raises**, and it is not a GPU question: why do ten combiners' 128-record
windows sit up to 34 records (~36 s) apart at all? Fix that and the gate stops being marginal.

## 7. Traps specific to this port

1. **Ramp sign.** The aggregate sums `e^{+i...}`; FFTW `FFTW_BACKWARD` and cuFFT `INVERSE` both
   match it. A planted test peak needs the *conjugate* ramp, or it lands at `s_cols - s0` --
   which is a correct answer to a different question, and looks like an indexing bug. (Hit this
   while writing `aggbench`.)
2. **The lag axis is the negative of physical delay** (5q). Any new indexing must be validated
   against a planted peak at a *known* cell, not against self-consistency.
3. **`s_stored` vs `sph`.** `s_stored` bounds the *range* of the fine lag; the ramp argument
   still divides by the full `sph`, and `tau = q*sph + i*s_step` uses `sph`. The live geometry
   has `g = 1` so `s_stored == sph` and the distinction is invisible -- it will not be for a
   comb with a common factor (airspy: g=2).
4. **Precision.** The CPU surface accumulates in `double`, the GPU will accumulate `float`. Over
   `acquire_windows` accumulations of `|D|^2` with `nc=79` this is fine, but it is a real
   difference and the validation must compare against the double reference at blind dimensions,
   not at toy ones.
5. **`CMAKE_CUDA_ARCHITECTURES` defaults to 86** (A40, the cx nodes). cf06's L40S is sm_89.
   An sm_86 cubin *runs* there (same major version) but is not Ada-optimal; set 89 for cf06.
6. **Reduction shape, not precision, was the peak kernel's problem.** It first ran at 195 GB/s
   and the obvious suspect was the FP64 accumulator (1/64 rate on an L40S). Converting the
   grid-stride partial to float changed nothing. The actual cause was 32 cells per thread
   followed by 8 `__syncthreads()` stages -- reducing cost as much as reading. 256 cells per
   thread: 195 -> 567 GB/s. Measure before attributing.
7. **`cudaDeviceProp::memoryClockRate` no longer exists in CUDA 13** (cf06 runs 13.3). Use
   `cudaDeviceGetAttribute(cudaDevAttrMemoryClockRate)`.
8. **Asking cuFFT for the layout you want is not free.** The aggregate wants channel-fastest
   `[d][q][c]`, and cuFFT will write it via `ostride = nc`. That is 8-byte writes at 632-byte
   stride: the correlate took **41 ms against a 2.5 ms aggregate**. Writing cuFFT's natural
   contiguous layout and transposing through a shared-memory tile on the aggregate's READ side
   took the whole chain 45 ms -> 8.3 ms and dropped 0.63 GB (the inverse now runs in place).
   Layout conversions belong where they are a *pattern*, not where they are *traffic*.
9. **A NULL `inembed` with a non-NULL `onembed` is not a valid mixture** -- cuFFT follows FFTW's
   rule that NULL inembed means "basic layout, ignore every other stride argument", and returns
   `CUFFT_INVALID_VALUE` rather than doing what you meant.
10. **Size-dependent alignment faults.** `launch_peak`'s scratch originally laid out
    `float[nb]` before `double[nb]`, which leaves the `double*` 4-byte aligned whenever `nb` is
    ODD. `n_dop` 321 gives `nb` 1960 and passes; 41 gives 251 and faults with "misaligned
    address" -- reported at the *next* sync point, so it presents as a failure in an unrelated
    memcpy. Sweep several sizes, not one; a single blind-dimension test would have shipped this.

## 8. Reproducing the baseline

```
KBUILD=build_nodpdk scripts/gnss/build_tool.sh aggbench      # on cf06
scripts/gnss/aggbench --nd 321 --mp 3125 --threads 16        # blind dims, direct vs folded
scripts/gnss/aggbench --nd 5 --mp 400                        # fast correctness check
```

`build_tool.sh` now takes `KBUILD` (cf06 has no DPDK or testing lib) and writes `<tool>.<host>`
with a `<tool>` symlink, because `/home/kvand` is NFS-shared and two hosts building the same
tool clobbered each other -- which presents as "the tool suddenly segfaults".

---

## 9. THE TRACKER kernel, 2026-08-05 -- where the GPU actually goes

Not the search. This is `cudaGnssChordTrack`, and it is what saturated the node GPUs.

### 9.1 The split, measured in situ

`log_kernel_split` on the tracker command (pairs with the cudaProcess stage's `log_profiling`:
one gives the command total, the other says where it goes). cx19, real geometry:

    7 spec x 7 chan x 32 elem x 2048 hops
      synthesis   (launch_waveform)      2.632 ms   89.4%
      correlation (launch_correlate_nm)  0.313 ms   10.6%

**Folding the correlation into CHORD's N^2 recovers at most ~11%.** The replica synthesis, which
N^2 cannot absorb because it knows nothing about GNSS codes, is nine tenths of the cost.

This contradicts BOTH earlier positions, so record why:
* the kernel header's 2026-07-16 "the despread MAC measured at 0% of the kernel" was a
  ONE-ELEMENT measurement (airspy). It does not transfer to 32 elements.
* the synthetic sweep that corrected it (54% correlation at n_elem 32) was measured at the unit
  test's geometry, where the gather is ~8 chips per hop against CHORD's **210**. Synthesis per
  sample is far more expensive here, and the element axis does not rescue the correlation
  because the replica is generated ONCE and broadcast across element lanes.

The lesson is the same one twice: `n_chips` = ceil((Lf-1)*cps)+2 depends on the geometry
(65536-sample PFB span / 312.8 samples per chip = 210 at CHORD, ~13 at airspy L5, ~8 at the unit
test). Nothing measured at one geometry transfers to another.

### 9.2 The gather IS the cost -- confirmed, not inferred

`despread_max_chips` caps the gather depth (BENCH ONLY: it truncates the filter, so the despread
output and every detection downstream are invalid; the stage WARNs).

    n_chips 210: synthesis 2.632 ms / 7 spec = 0.376 ms/spec | correlation 0.313 ms
    n_chips 105: synthesis 1.432 ms / 8 spec = 0.179 ms/spec | correlation 0.313 ms
    ratio 0.476, against 0.500 for a perfectly linear gather

The control is what makes this trustworthy: correlation is 0.313 ms in BOTH runs. It does no
gathering, so it must not move, and it does not.

### 9.3 The two optimizations, now justified

1. **Transpose the Phi tables.** `khi_base` advances by `ks` = 313 entries per chip, so
   consecutive chips read phiA/phiB **2502 bytes apart** -- every 8-byte load claims its own
   128 B line, and across a warp adjacent hops sit ~16400 entries apart, so it is fully
   scattered. A `[t mod ks][t / ks]` layout makes the walk contiguous.
   ⚠️ The gain is NOT the naive 16x: the stride is non-integer (312.8), so the row index drifts
   and only ~5 chips stay contiguous before it steps. Estimate ~5x on the gather. It touches
   `hoprate_filter`, which the CPU despread, the search and the e2e harness all share -- so it
   is exact but has real blast radius.
2. **Truncate `n_chips` to the filter's real support.** Now a MEASURED multiplier. But the PFB
   prototype is a 4-tap windowed sinc and its span IS the channel response, so this trades
   accuracy directly. Validate against `cuda_gnss_despread_test` and the e2e SEARCH LEG figure.

**Do (2) first**: the knob exists, its cost is measurable in tests that already run, and it needs
no change to shared code. Judge (1) after (2) shows how much depth is actually needed.

### 9.4 Also settled today

* **The tracker latched seeds forever** -- `sd.have` was only ever set true, so a satellite that
  had SET was still despread and the spec count grew with everything that had ever risen. Fixed
  with `seed_ttl_s` (60 s). Confirmed live: expiry log lines firing, kernel spec count tracking
  the broker's live set. NOTE `/get_trim`'s "active" count is a HIGH-WATER MARK (trim state
  persists past expiry by design) -- the spec count in the split line is the live number.
* **It does not explain saturation on its own.** Fitting the soak gives ~1.12 ms/PRN over a
  6.89 ms intercept, so even a fully latched 32-PRN tracker predicts ~39% duty against the
  observed 100%. ⚠️ That intercept is a TWO-POINT fit -- no residual, so it cannot be
  distinguished from a curve. It needs a PRN sweep before it is worth chasing.

### 9.5 Gather truncation: measured, and it is 1.31x -- not the big win

`ChannelizedReplicaBank::set_max_chips`, swept with `scripts/gnss/e2e --max-chips` at CHORD
geometry (GPS_L5_Q_NH, 3.2 GSPS, N=8192) through the shipped code:

| max_chips | search SNR | SEARCH LEG ERROR | nh |
|---|---|---|---|
| 210 (full) | 26812 | **+0.123 chips** | 5 correct |
| 160 | 26524 | **+0.123 chips** | 5 correct |
| 105 | 18641 | -13.073 chips | 4 WRONG |
| 64 | 6742 | -78.545 chips | 3 WRONG |
| 32 | 12800 | -91.537 chips | 6 WRONG |

**160 is free (error identical to full depth, SNR down 1%); 105 already breaks it.** The failure
is the familiar one: the truncated replica no longer makes the true lobe win, so the search
settles on a grating lobe -- the same shape as the old `refine_span: 4096` bug, and the same
13-chip signature.

So truncation is worth **1.31x** on the dominant cost (210 -> 160): synthesis 2.632 -> ~2.0 ms,
tracker kernel ~2.95 -> ~2.3 ms, about a 22% cut in tracker GPU load for a measured-zero change
in the answer. Real and free -- but NOT the several-fold win truncation looked like it might be.
The PFB span genuinely IS the channel response; the filter's tails matter down to ~160/210 and
then fall off a cliff.

**The cliff, bracketed** (second sweep): 150 / 140 / 130 / 120 all give the IDENTICAL
+0.123 chips. So the edge sits between 120 and 105 -- everything at or above 120 is exact, and
105 is 13 chips out. That is a genuine discontinuity, not a gradual degradation, which is what
makes picking a value without bracketing it dangerous.

**Recommended: max_chips 140.** 210 -> 140 is **1.50x** on the dominant cost (synthesis
2.632 -> ~1.75 ms, tracker kernel ~2.95 -> ~2.06 ms, a ~30% cut in tracker GPU load), with 17%
margin above the measured 120 floor. 160 would be more conservative at 1.31x; below 120 the
search lands on a grating lobe.

**How to measure this honestly** (the trap is easy to fall into): e2e's synthetic sky comes from
`tbank` and the search analyses with `sbank`. `--max-chips` caps **sbank only**. Capping both
would truncate the sky and the replica identically, the error would come out ~0, and it would
measure nothing at all.

⚠️ SCOPE: this is the SEARCH leg. The tracker despreads with `tbank`, which also generates the
sky, so e2e cannot isolate the tracker's truncation error the same way. Same generator and same
geometry, so 160 should transfer -- but that is an inference, not a measurement.

Note also that SNR is NOT the metric here: max_chips 32 scores a HIGHER SNR than 64 (12800 vs
6742) while being more wrong (-91.5 vs -78.5 chips). Score the code phase.

### 9.6 What is left, in order

1. **max_chips ~160** once the cliff edge is bracketed. Measured, free, 1.31x.
2. **Transpose the Phi tables** -- now the larger remaining lever (~5x on the gather, EXACT, no
   accuracy cost), and the one with real blast radius: `hoprate_filter` is shared by the CPU
   despread, the search refine and the e2e harness.
3. The ~2.5x of tracker saturation still unexplained by the seed latch (see 9.4), which needs a
   PRN sweep rather than the two-point fit it currently rests on.
