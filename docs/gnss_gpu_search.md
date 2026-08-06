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

### 9.6 What is left, in order  (SUPERSEDED -- see 9.11)

Kept as written for the record; both leads in it turned out badly and it is worth seeing how.

1. **max_chips ~160** once the cliff edge is bracketed. Measured, free, 1.31x.
   -- SUPERSEDED by 9.9: the block width makes the FULL span cheaper than the truncated one, so
   the cap should be reverted rather than tuned.
2. **Transpose the Phi tables** -- now the larger remaining lever (~5x on the gather, EXACT, no
   accuracy cost), and the one with real blast radius: `hoprate_filter` is shared by the CPU
   despread, the search refine and the e2e harness.
   -- RETRACTED, see 9.9. It was aimed at coalescing, and coalescing was never the problem.
3. The ~2.5x of tracker saturation still unexplained by the seed latch (see 9.4), which needs a
   PRN sweep rather than the two-point fit it currently rests on.  -- still open.

### 9.7 ncu: the waveform kernel is DRAM-bound at 96%, and L2 hit rate is 3%

Profiled on cx19 (A40, sm_86), `--section SpeedOfLight,MemoryWorkloadAnalysis,WarpStateStats`,
report at `/tmp/gnss_ncu.ncu-rep`. Grid (11, 7) x 256 = 11 jobs x 7 channels.

    DRAM throughput        96.26%   (669 GB/s of an A40's ~696 peak)
    Compute (SM)            5.58%
    L1/TEX hit rate        78.84%
    L2 hit rate             3.14%   <-- THE FINDING
    warp stall             30.1 cycles/warp on an L1TEX scoreboard dependency
    duration                2.40 ms

**Memory-bound, not compute-bound.** That settles what the n_chips linearity could not: the
linearity was consistent with either, since both scale with the loop count.

**Why L2 misses.** The Phi tables are 1.05 MB per channel (65537 entries x 8 B x 2 images) and
the grid runs all 7 channels concurrently, so the live working set is **7.3 MB against the A40's
6 MB L2**. It does not fit, the access is scattered, nothing is reused, and every load reaches
DRAM.

### 9.8 The channel-serialization plan was WRONG, on both its premise and its remedy

The plan §9.7 pointed at was: launch the waveform kernel once per channel so each launch's Phi
working set (1.05 MB) is L2-resident instead of seven channels' 7.3 MB against a 6 MB L2. Its own
"check before building" list killed it. Both checks failed.

**The premise is false: Phi is bucketed PER PRN, not per channel.** `GnssCudaDespread::Impl` holds
`std::vector<PhiCache> phi` -- one `[n_chan][Lf+1]` pair of tables per PRN SLOT, rebuilt whenever
that PRN's Doppler moves by more than `refresh_hz`, because the carrier rate `wc` sits inside the
table's exponent. So the live Phi is `n_prn x n_chan x 1.05 MB` = **77 MB at 11 PRNs**, not 7.3,
and no two blocks share a slice. Serializing over channels reaches 11.5 MB -- still twice the L2.
It could never have been resident.

**The remedy loses anyway, at every group size.** Measured on cx19 (`scripts/gnss/wavebench.cpp`),
against the 2.42 ms all-channel launch:

    chan group 1: 0.946 ms x 7 launches = 6.62 ms     2.74x SLOWER
    chan group 2: 1.032 ms x 3 + 0.946  = 4.04 ms     1.67x SLOWER
    chan group 4: 1.537 + 1.220 (4+3)   = 2.76 ms     1.14x SLOWER

Exactly the parallelism failure the check anticipated: 11 blocks against 84 SMs.

**`L2 Persisting Size 1.18 MB` was a red herring** -- it is the configured persisting-access
window (`cudaLimitPersistingL2CacheSize`), a device default that nothing in this tree sets. It is
not a working-set measurement and says nothing about the diagnosis either way.

### 9.9 THE FIX: the block width IS the traffic knob -- 3.67x, bit-identical

The right question was never "does the working set fit in L2". It is **how many times does one
block re-read its own slice**, and the answer was 24.

A block owns one (PRN, channel) Phi slice. Within `chip_gather`, thread lane `m`'s tap index is
`t(d) = d*ks + floor(base + d*kf) + 1` with `base = phi*inv_cps`, and `phi` is a FRACTIONAL part,
so `base` is in `[0, inv_cps)` for every hop and every E/P/L trial. At chip `d` the whole block
therefore lands in the same ~313-entry window -- which is why L1 hit rate was already 78.84%. But
the block walks that slice once per **hop pass** and once per **trial**, and between two visits to
chip `d` it has streamed the entire 689 KB slice through a 128 KB L1. Nothing survives:

    hop passes = n_hops / blockDim = 2048 / 256 = 8,  x 3 trials  = 24 re-walks
    24 x 689 KB x 77 blocks = 1.24 GB per launch, against 51.8 MB of distinct data

So DRAM traffic is inversely proportional to the block width, and the width was 256 for no reason
beyond it being the fused kernel's. Raising it to 1024 cuts the passes to 2. Measured, cx19 A40:

     256 threads   8 hop passes   2.519 ms   1.00x    <- as profiled
     512 threads   4 hop passes   1.234 ms   2.03x
    1024 threads   2 hop passes   0.684 ms   3.67x

**`wave` is bit-identical at every width** (checked over all 4.6M replica samples at CHORD
geometry): each hop's replica sample depends only on that hop, so the width moves which thread
computes it and nothing else. Only the ENERGY reduction's lane count changes -- and at this
bench's inputs even that came out bit-identical, though only the `wave` guarantee is structural.
`CHORD split vs fused despread` still reports `corr 0.000e+00 energy 0.000e+00`.

Two things had to be got right:

* **`__launch_bounds__` is load-bearing.** Unbounded, the gather compiles to ~72 registers and the
  driver refuses 1024 threads outright ("too many resources requested") -- the first 1024-wide run
  failed to launch. The kernel is now templated on its width so ptxas is told the bound: at
  `MAXT=1024` it fits in 64 registers with **0 bytes spilled**. Without the template the width
  silently clamps to 512 and you get 2.03x instead of 3.67x.
* **The width is a REQUEST, not a promise.** `launch_waveform_tuned` queries
  `cudaFuncGetAttributes().maxThreadsPerBlock` and falls back to the 512 instantiation, because
  the ceiling moves with the architecture and with any edit to the gather, and exceeding it does
  not degrade -- it fails the launch.

Holds across the envelope (chips x PRNs), never below 2.74x:

              4 PRN    11 PRN    20 PRN
    140 chips  2.74x     3.68x     3.43x
    212 chips  2.81x     3.76x     3.72x

**This retires the `max_chips` truncation.** 212 chips (the full PFB span) at 1024 threads is
1.001 ms -- 2.5x FASTER than the 140-chip truncated kernel that is deployed today at 2.517 ms. The
cap bought 1.31x by throwing away part of the channel response; the width buys 3.67x by throwing
away nothing. Recommend reverting `despread_max_chips` to 0 on all eight node configs.

**RETRACTED, do not attempt:** transposing the Phi tables to `[t mod ks][t / ks]`. It needs
consecutive chips to share a row index, but dt = ks + (0 or 1) with kf ~ 0.8, so the row changes
~80% of steps; and `base` varies per hop, so no FIXED permutation linearises more than one hop's
walk. The ~5x quoted earlier assumed slow drift; it drifts fast. And it was aimed at coalescing,
which was never the problem -- L1 was already at 78.84%.

### 9.10 The bench, and why a synthetic one is allowed here

`scripts/gnss/wavebench.cpp` (build: `KBUILD=build scripts/gnss/build_tool.sh wavebench`, run on a
cx node). Two previous attempts to reason about this kernel died on numbers carried across a
GEOMETRY -- `n_chips` is 212 at CHORD, ~13 at airspy L5, ~8 at the unit test. The trap was never
"synthetic", it was the geometry, so the bench hard-codes CHORD's and its FIRST OUTPUT is a
faithfulness gate: the 256-wide launch must reproduce the 2.402 ms ncu measured in situ. It comes
out at 2.51 ms, +4.5%, and the run says so in as many words. Phi contents do not affect timing --
only addresses do, and those are fixed by (inv_cps, n_chips, Lf, n_hops, frac(cp0)) -- so the
tables are filled with junk.

Note the unit test does NOT gate the width: at its 125 hops the block is 64 threads wide and the
launch is byte-for-byte what it always was. The bit-identity check that matters is the bench's,
over all 4.6M samples at 2048 hops.

### 9.11 Trial fusion: the last factor of 3, and it holds -- 7.6x total

`chip_gather3` walks the chip loop once for all three E/P/L trials. It works because the three
trials' taps at chip `d` are NEIGHBOURS: each `t(u,d) = d*ks + floor(base_u + d*kf) + 1` with
`base_u` in `[0, inv_cps)` -- a fractional part, so the trial offset moves the tap inside a ~625
entry window rather than to a different part of the slice. Issued together they share L1 lines;
issued a whole 689 KB slice apart, as three separate calls are, they cannot.

Measured on cx19 (contended -- see the caveat below), 11 PRNs x 7 chan x 140 chips:

                       re-walks       ms      vs 256/3-gather
    3-gather   256        24        2.985        1.00x
    3-gather   512        12        1.482        1.92x
    3-gather  1024         6        0.781        3.65x
    fused      256         8        1.036        2.75x
    fused      512         4        0.618        4.61x
    fused     1024         2        0.375        7.61x     <- shipped

The two knobs do NOT simply multiply: fused-1024 spills (104 B stores, 72 B loads) where
3-gather-1024 does not, and it wins anyway. Fused-512 (96 registers, no spill) beats
3-gather-1024, so if a future edit pushes the spill further, dropping to 512 is the fallback and
costs 1.6x rather than everything. Holds across the envelope: at 50-200 PRNs it is 14-16x over
the original.

**⚠️ THE `cmulf` TRAP -- this is the part worth remembering.** Switching the fusion on made the
split-vs-fused test report 3.078e-07 where it had always reported exactly 0. The obvious suspect
was the new gather, and it was innocent: an isolated A/B of `chip_gather` x3 against
`chip_gather3` on identical inputs, over 4096 code phases x 4 geometries (including a small-Lf
one where the clamps actually fire), found ZERO mismatches. The culprit was `cmulf`, untouched,
several lines downstream -- written as `a.x*b.x - a.y*b.y` it is ptxas's choice whether to
contract to fmaf, and it took that choice differently once the surrounding code changed shape.
Both `cmulf` and `chip_gather`'s accumulate are now pinned to explicit `fmaf`, which removes the
freedom. Pinning `cmulf` is not quite a no-op -- it moves the replica ~1 ulp and the GPU-vs-CPU
agreement in the test IMPROVES (5.400e-07 -> 5.276e-07, peel residual 2.461e-07 -> 2.454e-07, and
the peel cross-term perturbation 1.914e-07 -> exactly 0). Every exactness gate is back at 0.

**And note which instrument caught it.** Not the bench -- the bench compares this kernel against
itself, so a change that moves BOTH paths is invisible to it by construction. The first version
of this work also hoisted the carrier/write/energy block into a shared loop, which broke the
UNFUSED path's bit-identity, and the bench called that bit-identical too. Only the unit test,
which compares against the untouched `gnss_despread_kernel`, could see either. The else branch in
`gnss_waveform_kernel` now duplicates eight lines on purpose, with a comment saying why.

### 9.12 Scaling to the full GNSS sky

Synthesis, one A40, 7 channels, 140 chips, fused/1024. Contended numbers corrected by the
11-PRN calibration (0.930 contended vs 0.684 idle = 0.735); duty is against the 10.486 ms record
period:

    codes     idle-equiv ms    duty      vs the original 256/3-gather
      11          0.375        3.6%          7.6x
      50          1.46        13.9%         14.3x
     100          2.73        26.0%         14.2x
     200          5.44        51.9%         14.2x

Below ~12 codes the GPU is not full: at 1024 threads and 64 registers a block owns a whole SM, so
84 SMs / 7 channels = 12 PRNs before it saturates, and everything under that is nearly free.
Above it the cost is linear at ~0.027 ms per code.

Adding the correlation kernel (0.313 ms at 7 spec, saturating past ~12 then ~0.026 ms/code --
ESTIMATED, wavebench does not measure it): **100 codes ~5.3 ms/record (~51% duty), 200 codes
~10.6 ms (~101%)**. Before this work, 200 codes would have been ~134 ms -- 13x over real time.

**What "100 codes" means physically.** `f_offset_hz` is 1176.45 MHz, which is simultaneously GPS
L5, Galileo E5a and BeiDou B2a, all 10.23 Mcps, all the same Lf/n_chips geometry. So ~100 codes
(≈32 GPS + 36 Galileo + 35 BeiDou) is a pure `n_prn` scale-up **in the 7 channels that already
exist** -- no new bandwidth, no new PFB, just more entries in the replica bank. 200 means a second
band (L1/E1/B1), which is a second channel set and a second tracker instance: it doubles
everything rather than extending the PRN loop.

### 9.13 What is left, in order

> Mirrored, with the rest of the CHORD open items, in `docs/CHORD_BUGLIST.md` (P1-P3, C1).

1. **Revert `despread_max_chips` to 0** -- 9.9/9.11 make the full 212-chip span cheaper than
   today's truncated one. Removes a known-lossy approximation. Costs 1.42x at scale (212 vs 140
   chips), which at 100 codes is 2.73 -> 3.9 ms and still comfortable.
2. **`chip_gather3` in the FUSED despread kernel too.** `gnss_despread_kernel` still makes three
   separate gathers, and it is what the search's `refine_peak_cuda` and the airspy path run. Same
   argument, same win, and the pinning work is already done. NOT attempted here.
3. **Phi rebuild cost at 100-200 codes.** `hoprate_filter` builds 917k `std::exp` per PRN per
   rebuild, single-threaded, and `refresh_hz` 100 means every PRN rebuilds every ~110 s at L5
   Doppler rates. At 200 codes that is ~1.8 rebuilds/s. Probably fine; UNMEASURED.
4. The ~2.5x of tracker saturation still unexplained by the seed latch (see 9.4), which needs a
   PRN sweep rather than the two-point fit it currently rests on.

## 10. The synthesis kernel: structure, cost model, and where the levers are

Written 2026-08-06 because synthesis is now the dominant GPU cost and the question "can this
scale to 100+ codes" needs a shared model rather than a per-thread argument.

### 10.1 Notation -- TWO things are called phi, and they are unrelated

* **`phi` (lowercase, in the code)** is the FRACTIONAL code phase, `C - floor(C)`, in `[0,1)`.
  It only places the chip/tap boundary within a hop.
* **`Phi` / `PhiA` / `PhiB`** is the cumulative filter TABLE, `[n_chan][Lf+1]` complex, built on
  the host by `hoprate_filter`. This is the 1.06 MB per (PRN, channel) object the kernel streams.

Everything below about "the Doppler is in the exponent" is about the TABLE.

### 10.2 What the kernel computes

One complex sample per (PRN, channel, hop, trial). Because the code is piecewise constant over a
chip, the PFB's response over its whole filter span collapses to a sum over CHIPS:

    sample = sum_d  code[chip0 - d] * ( Phi[t_d] - Phi[t_{d-1}] )

`Phi` is a prefix sum of the channel-modulated PFB prototype, so a chip's contribution is one
DIFFERENCE of it. That is the whole algorithm: 212 chip steps instead of 65536 sample steps --
a 300x algorithmic win that is already banked, and the reason a dense/FFT formulation loses
(10.6).

    Phi_A,c[K] = sum_{j<K} proto[j] * exp(-i (off_c + wc) j),   off_c = 2*pi*c/fft_len
    Phi_B,c[K] = same with (off_c - wc)          wc = 2*pi*(f_offset + doppler)/fs

**THE DOPPLER SITS INSIDE THAT EXPONENTIAL, MULTIPLIED BY THE TAP INDEX j.** That is why every
PRN needs its own table: two PRNs at different Doppler differ by a factor `exp(-i d(wc) j)` that
varies ALONG the sum, so one table cannot be scaled into another -- it would have to be re-summed.
Two tables (A and B) because the replica is the real-signal pair
`r = 0.5*(pa*sA + pb*sB)`, `pa/pb` being the per-hop carrier phasor and its conjugate.

### 10.3 The loop nest, at deployed CHORD geometry

    per record                        10.4858 ms = 2048 hops x 16384 samples @ 3.2 GS/s
      per PRN                         n_prn seeded
        per channel                   7 per instance
          per hop                     2048
            per trial (E/P/L)         3        <- fused into one walk since f789d6856
              per chip                212      <- THE INNER LOOP
                2 loads from Phi (16 B) + ~14 flops

`n_chips = ceil((Lf-1)*cps)+2 = 212` is NOT a tuning knob: `Lf = num_taps * fft_len = 4*16384 =
65536` samples of PFB span at 312.8 samples/chip. It is set by the F-engine.

Per PRN per record: `7 * 2048 * 3 * 212 = 9.1M` chip steps. At 11 PRNs, 100M chip steps,
~1.6 GB of loads, ~1.4 Gflop.

### 10.4 The cost model: memory, by ~60x

Arithmetic intensity is **0.9 flop/byte** (14 flops per 16 B) against the A40's balance point of
~54 (37.4 Tflop/s fp32 / 696 GB/s). Nothing about cheaper arithmetic matters. Only bytes.

And the bytes are awkward structurally: consecutive `t_d` are ~312 entries apart (2.5 kB), so
every chip touches a fresh line, and **no two blocks share a table** (10.2). Each block streams
its own 1.06 MB slice.

    measured, 11 PRN x 7 chan x 212 chips, fused/1024   0.546 ms   (5.2% of a record)
    pure-traffic floor (each slice read once)           0.12-0.24 ms
    linear regime above ~12 PRN                         ~0.038 ms/PRN

Below ~12 PRNs the GPU is not full: 1024-thread blocks are one per SM, and 84 SMs / 7 channels =
12 PRNs. **At 100 codes: ~3.8 ms synthesis + ~2.6 ms correlation = ~61% of a record period, for
ONE band on ONE instance.** That is the scaling problem.

### 10.5 Precision -- MEASURED, and it corrects an earlier claim

`scripts/gnss/phibits.cpp` builds `Phi` at deployed geometry and reports what the gather actually
needs. **The "Phi is a prefix sum so it needs the dynamic range" argument is WRONG**, and I made
it before checking:

    max|Phi|                                    0.674
    |dPhi| over one chip step   median          0.0825      max|Phi|/median = 8.2  (3.0 bits)
                                min             2.56e-05    max|Phi|/min    = 26325 (14.7 bits)

The integrand oscillates at 2.31 rad/sample, so the prefix sum does NOT grow secularly -- it is a
bounded oscillation, and the differences are within ~8x of the values. Rounding `Phi` to N
mantissa bits (true per-value float exponents) costs, on the chip-step difference:

    23 bits (fp32)  median 1.5e-08   worst 4.2e-08
    15 bits         median 4.0e-06   worst 1.2e-05
    11 bits (fp16)  median 6.2e-05   worst 3.3e-04
     8 bits (bf16)  median 5.2e-04   worst 1.8e-03

**fp16 Phi is viable.** A 3e-4 worst-case relative error per chip step is ~-70 dB on the replica,
far under every other error in the chain. That halves the table, halves the traffic, and the
kernel is memory-bound by 60x -- so it is a near-direct **2x**, and it is the cheapest lever on
this list. What it costs is the bit-exactness contract between the waveform and fused kernels
(both would move together, so the gates stay at 0, but the replica changes by ~1e-4 -- a REAL
change, unlike the fmaf pinning).

**Where the doubles are, and why they stay.** The inner loop is already entirely float. Double
appears in exactly two places, both OUTSIDE it:

1. **Once per hop, for absolute time.** `C = cp0 + n_m*cps` with `n_m ~ 2.95e15` after ten days
   of F-engine uptime, and the carrier angle `wc*n_m ~ 6.8e15` -- which lands in the binade where
   a double's ULP is EXACTLY ONE RADIAN (hence the fma two-product in the kernel). float has 24
   bits; this needs 52. Non-negotiable, and it is 1/212 of the work.
2. **The reductions.** `acc[]` and `e[]` accumulate over 2048 hops in double. That is about the
   SUM, not the sample.

So "why full floats" has a clean answer: the samples are float already, the doubles are absolute
time and final accumulation, and the table could plausibly be HALF a float.

### 10.6 BENCHED 2026-08-06 (idle GPU, min-of-5, all through ONE harness path)

    jobs      fp32      interleaved      fp16        addr-collapsed
      11    0.5372 ms  0.4979 (1.08x)  0.3927 (1.37x)  0.2309 (2.33x)
      50    2.3994 ms  0.2309 --       1.8592 (1.29x)  1.1419 (2.10x)
     100    4.3954 ms  4.2472 (1.03x)  3.4530 (1.27x)  2.0556 (2.14x)

* **`addr-collapsed` is the ceiling**: loads kept, every instruction kept, only the ADDRESS forced
  to 0. Memory is therefore **~55% of the runtime** -- the kernel IS memory-bound, and the most
  any layout work can buy is ~2.1-2.3x.
* **fp16 Phi: 1.27-1.37x.** Reproducible, roughly half the available headroom.
* **Interleaved float4 (PhiA/PhiB in one table, one 16 B load instead of two 8 B): 1.03-1.08x.**
  Essentially nothing.

**A prediction that failed, recorded because the reasoning looked sound.** I argued the cost was
the per-warp TRANSACTION count: 32 lanes scatter over a 313-entry window, ~26 distinct sectors per
array, two arrays = ~52 transactions, so interleaving to one array should give ~1.8x on the
memory half. Measured 1.05x. fp16 -- which does NOT reduce the load count, only narrows the
window so lanes pack into fewer sectors -- is the one that works. So "sectors touched" explains
fp16 and fails on the interleave, and I do not have a validated model of what this kernel pays
for. What exists is the measurements above.

⚠️ **METHOD, learned the hard way today.** Four separate times a Python string-replace silently
matched nothing, the build reused a stale binary, and an unparsed flag fell through to the default
path -- producing "no change" results that looked like clean negatives. One of them (`--noload`)
was reported as "the kernel is not I/O bound", which is the OPPOSITE of the truth. Every bench
flag now prints a MODE banner, and every edit script asserts before writing. **If a ceiling test
reports exactly the baseline, suspect the harness before the hardware.**

### 10.6b Hop-sorting: NOT DEMONSTRATED (0.81-0.86x as implemented)

Sort each job's hops by fractional code phase and give thread lane `m` hop `perm[m]`. The premise
is sound and still unrefuted: `base = frac(C_P)*inv_cps` is FIXED per hop for the whole chip loop
while the `d*kf` term is common to every lane, so the lanes' order in the window never changes --
sorting should put a warp's 32 lanes inside ~313*32/2048 ~ 5 entries instead of all 313.

    jobs      fp32     hop-sorted   hop-sorted+fp16   fp16      ceiling
      11    0.5366 ms  0.6216 (0.86x)  0.4795 (1.12x)  0.3917 (1.37x)  0.2311 (2.32x)
     100    4.3708 ms  5.4194 (0.81x)  4.1945 (1.04x)  3.4329 (1.27x)  2.0535 (2.13x)

**Slower, consistently -- and it also removes most of fp16's benefit** (1.04x combined against
1.27x for fp16 alone), which says the loss is something the sorting ADDS rather than a failure of
the clustering. Two candidates, neither eliminated:

1. **A dependent global load at the head of every hop.** `p.hop_perm[b*n_hops + mh_i]` must
   complete before `n_m`, `C_P`, `base` and therefore the entire 212-step chain can start. With
   only 2 hops per thread there is almost nothing to overlap that latency against. Fixable:
   preload the permutation into registers or shared memory before the hop loop.
2. **The `wave` writes become scattered.** `wave[...][mh]` with permuted `mh` puts a warp's 32
   lanes at 32 scattered offsets in a 16 KB row, 3 trials per hop. Estimated at ~1% of the read
   traffic, so probably not the whole story, but it is real.

So this is NOT a refutation of the idea -- it is one implementation of it, measured, and it lost.
Anyone retrying should fix (1) first and re-measure before touching anything else.

### 10.6c ncu COUNTERS: it is DRAM BYTES, not coalescing -- and that overturns 10.6/10.6b

Six black-box A/B tests produced a contradictory picture. The counters settle it (cx19 idle,
11 jobs x 7 chan x 212 chips, one launch):

    variant     dur us   LD req      LD sectors    sec/req    ST sec/req   DRAM
    fp32        535.97   9,486,400   298,601,238     31.5         7.9      245 MB
    fp16        393.06   9,486,400   271,464,521     28.6          --       --
    ilv         495.46   6,334,944   199,358,636     31.5          --       --
    hopsort     629.63   9,491,328   113,385,918     11.9        31.4      300 MB

**The gather is maximally uncoalesced -- 31.5 of a possible 32 sectors per warp load, i.e. every
lane in its own sector -- AND THAT IS NOT WHAT COSTS.** Hop-sorting did exactly what it was
designed to do, cutting load sectors 62% (31.5 -> 11.9 sec/req), and came out 17% SLOWER. The
interleave cut requests and sectors 33% for an 8% gain. fp16 cut sectors only 9% and gained 27%.
Load sectors do not order the results; they anti-correlate with them.

**DRAM bytes do.** fp32 moves 245 MB in 536 us = 457 GB/s, ~66% of the A40's 696 GB/s. fp16 wins
because it halves the TABLE, not because it coalesces.

**Where the stores are** (asked, and the count identifies itself): they are the `wave` writes in
the kernel body, plus the energy rows. At 11 jobs x 7 chan x 2048 hops,
`3*11*7*2048 = 473,088` wave stores / 32 lanes = 14,784 warp requests, plus `4*11*7 = 308` energy
requests = **15,092, exactly the measured ST req**. Nothing else stores, and the Phi tables are
built on the host and uploaded OUTSIDE the timed region. Hop-sorting changes no output structure
-- every element lands at the same address and the contents are bit-identical -- it changes which
LANE writes which element, so a warp's 32 stores go from contiguous (256 B = 8 sectors) to 32
scattered offsets in the 2048-element row (31.4 sectors).

⚠️ **PARTIALLY ATTRIBUTED, do not quote this as settled.** The scattered stores are real but do
NOT account for the whole DRAM increase: 473,396 - 118,580 = 354,816 extra store sectors is ~11 MB,
or ~23 MB if every one costs a write-allocate read-modify-write, against an observed 245 -> 300 MB
(+55 MB). And hop-sorting has FEWER load sectors (113M vs 298M) while moving MORE DRAM, which
means the reordering hurt reuse somewhere between L1 and DRAM. The missing ~30 MB is unexplained;
`lts__t_sector_hit_rate` on the two variants is the next counter to pull.

**So the lever is DRAM FOOTPRINT.** fp16 (1.27-1.37x) is right for the right reason. The
interleave changes no bytes and duly did nothing. Anything that shrinks the resident table helps;
anything that only rearranges access does not.

**Hop-sorting is still not dead, but it needs its stores fixed.** The read permutation and the
write index are the same `mh`, so they cannot be decoupled directly -- but the writes could be
staged in shared memory and flushed in hop order, keeping the coalesced store AND the coalesced
gather. Worth one attempt, now with the counter that will tell you immediately whether it worked
(`l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum / ..._requests_...`, must stay near 8).

⚠️ **METHOD.** Six A/B experiments produced a model that was wrong three times (bandwidth-bound,
then transaction-bound, then sector-bound). Two ncu runs settled it. **Profile before theorising
about a kernel; the black-box ladder cost most of a day and got the mechanism backwards.**

### 10.7 Levers, ranked, with what is measured vs speculative


1. **fp16 `Phi` -- BENCHED 1.23-1.36x** (10.6), not the 2x predicted. Still the best single
   available win, and accuracy is not the constraint.
2. **Doppler-factored `Phi` -- REJECTED, benched (10.6).** The accuracy works (quadratic over a
   +-1000 Hz bucket is 1.3e-04, better than fp16's own error), but the sharing CEILING is slower
   than the private-table baseline, so the scheme cannot win. Kept below for the record. Share one table set
   per channel across all PRNs in a Doppler bucket, plus a low-order correction. Measured tilt
   across the span: **0.643 rad at +-5 kHz** (chip-step `dPhi` changes by 63%), **0.064 rad at
   +-500 Hz** (6.4%). So a shared table needs either genuine correction terms or buckets far
   tighter than the current 100 Hz `refresh_hz` implies -- but even ~20 buckets beats 100
   per-PRN tables at 100 codes, and it stops live `Phi` growing with `n_prn` at all. Needs an
   accuracy study; the tilt numbers above are the input to it.
3. **Shorter PFB span.** `n_chips` is exactly proportional to `num_taps * fft_len`. A 2-tap PFB
   halves synthesis. F-engine parameter, changes the channel response, not ours -- but it is
   linear and worth knowing.
4. **Code-transition sparsity -- LOOKS good, IS not.** By summation by parts,
   `sum_d c_d (Phi_d - Phi_{d-1}) = sum_d Phi_d (c_d - c_{d+1}) + boundary`, and with `c = +-1`
   half those weights are exactly zero. But threads in a warp sit at different HOPS, and `chip0`
   advances 52.4 chips per hop, so the skip mask is uncorrelated across lanes: the branch is
   fully divergent, the warp still issues the loads, and we are memory-bound. Real flop saving,
   worthless.
5. **Tensor cores -- NO, and it is worth recording why.** The binding constraint is a
   data-dependent GATHER, not a matrix multiply; tensor cores raise a flop ceiling that is 60x
   away from binding. For a fixed hop the `t_d` are shared across channels, so `s = M v` with M
   7x212 -- but building M *is* the gather, and the multiply that follows is trivial. The only
   formulation that is a genuine dense GEMM is the pre-collapse one (`n_chan x Lf` per hop,
   459k MACs against the gather's 1484): **300x more arithmetic**, against a tensor advantage of
   8-16x. Loses by more than an order of magnitude. The chip gather already IS the sparse
   algorithm.
6. **Traffic engineering -- mostly spent.** 24 re-walks -> 2 (block width + gather fusion, 7.6x).
   We sit 2-4x above the pure-traffic floor; that is all that is left there.

## 11. NEXT BIG JOB: fold the correlation into CHORD's N^2 kernel

Scoped 2026-08-06. This is the framework doc's **"path B"** (`gnss_chord_framework.md`, referenced
from `cudaGnssChordDespread.hpp`'s header); everything shipped so far is "path A", the standalone
`launch_correlate_nm`. **Start this with a full context window -- it is a bigger job than anything
in section 10.**

### 11.1 The shape

Extend CHORD's existing N^2 correlator to take TWO inputs and split its output:

    in:   antennas  [N, t]        the real voltages, as today
          synthetic [M, t]        our replicas, injected as extra "antennas"
    out:  (N+M)^2  ->  split kernel  ->  N^2   pass to the STANDARD pipeline, untouched
                                        N x M  ours: the despread correlations
                                        M^2    ours: the replica energies / cross-terms

Why it is worth it: the N^2 kernel already loads every antenna voltage once and is heavily
optimised. Path A re-loads the voltage in `launch_correlate_nm`; path B does not load it at all.
The M^2 block also gives the replica cross-terms `<R_P,R_E>`/`<R_P,R_L>` for free -- the peel's
analytic add-back currently computes those separately (`XC` in `gnss_despread_kernel`).

### 11.2 What this does and does NOT save

⚠️ Measured, so nobody re-derives it: **synthesis is 73-89% of the tracker kernel and N^2 CANNOT
absorb it.** N^2 knows nothing about GNSS codes; the replicas must still be synthesised by
`launch_waveform` and handed over. Folding the correlation in recovers the CORRELATION only --
0.313 ms of a 1.15 ms kernel at 7 spec (27%), or ~11% at the older 11-spec measurement. The
synthesis cost problem (section 10) is untouched by this and stays the scaling limiter.

So path B is about (a) not loading the voltage twice, (b) riding the optimised (N+M)^2 kernel
instead of maintaining our own correlator, and (c) getting M^2 free. It is an ARCHITECTURE win,
not a synthesis-cost win.

### 11.3 ⚠️ THE HOP-ORDER CONTRACT (surfaced 2026-08-06, and it now has teeth)

`wave` is `[3*n_job][n_chan][n_hops]` and there has always been an IMPLICIT contract that hop `mh`
lives at index `mh`. Nothing states it; both kernels just assume it.

Section 10.6b wanted to break that contract (store in sorted-hop order to coalesce the scattered
store, 7.9 -> 31.4 sectors/request). That is safe between OUR two kernels -- the correlator's
`wave` read is a broadcast and its data read only row-selects, so a permuted `mh` is free on both
sides. **It is NOT safe once the consumer is CHORD's N^2 kernel**, which has its own time-axis
conventions and is not ours to permute.

**Decide the contract BEFORE building path B**, and write it down in `gnssRecord.hpp` next to the
row definitions. Options: (a) keep hop order as an invariant and fix the scattered store with a
shared-memory exchange inside the synthesis kernel (24 KB, self-contained, no downstream change);
(b) permit a permutation and carry it explicitly as part of the interface. (a) is the safe default
and is why 10.6b lists SMEM first.

### 11.4 Starting points

* `lib/cuda/cudaGnssChordDespread.{hpp,cu}` -- the header explains why the split exists and what
  path B changes ("only the CONSUMER changes -- from launch_correlate_nm to synthetic lanes").
* `gnssRecord.hpp` -- the record schema does NOT change under path B: N x M is the element blocks,
  M^2 is the header energies. That is the division path B gets for free.
* `lib/stages/gnss/GnssGpuRecordAssemble.cpp` -- consumes the correlations today.
* The N^2 kernel itself lives in the CHORD/kotekan correlator tree (`lib/cuda/`, n2k), NOT in the
  GNSS stages. Read it first; the two-input extension is its constraint, not ours.

### 11.5 Open questions to settle first -- ALL ANSWERED 2026-08-06 (see 11.6)

1. Does the N^2 kernel's input layout admit extra "antennas" without a copy? **YES, better than
   hoped**: n2k's e_in is dereferenced in exactly one place (`prefetch_chunk`), per-warp,
   per-128-station panel, with the base offset host-precomputed (the "ptable"). A second input
   is a warp-uniform pointer select + a runtime per-warp time stride -- zero cost in the MMA
   loop, no interleaving copy. Constraint: NSA and NSB each a multiple of 128.
2. Skip the M^2 block? **YES, host-side**: blockIdx.x -> (atile,btile) is decided when the
   ptable is built, so block classes (AxA / mixed / BxB) can be dropped per launch with zero
   device-code change (`block_class_mask`).
3. Precision: int32 accumulators, EXACT (4-bit products over nt_inner <= 8192 cannot overflow);
   samples must be in [-7,7] -- the value -8 SILENTLY corrupts (`negate_4bit`). Measured cost of
   quantizing the REPLICA side to 4+4b: see 11.6.
4. Cadence: one GNSS hop == one correlator time sample, exactly (5.12 us). Production frame =
   8192 hops = 4 tracker records; `sub_integration_ntime` = 8192 -> one visibility per 41.94 ms.
   DECIDED (user, 2026-08-06): keep 8192. Path-B records are frame-length; the re-blocker is the
   consumer stage (M5).

### 11.6 STATUS 2026-08-06: n2k_dual built; M1-M3 gates PASS

Decisions in force: straight to the two-input kernel (the 96 dead elements 32-127 are being
populated with real dishes over ~a month -- not available); **clone, don't clobber** (others
develop n2k: `external/n2k` and `cudaCorrelator` are untouched; the clone is
`external/n2k_dual/`, namespace `n2k_dual`, divergences listed in its README); keep
`sub_integration_ntime` 8192.

The module: `DualCorrelator(nstations_a, nstations_b, nfreq, block_class_mask)`. The kernel
template depends only on the TOTAL station count (the A/B split is host-side ptable data), so
one instantiation serves every 128-multiple split; compiled: (128,8), (256,8) tests,
(256,384) pathfinder deploy. **P_HEAD survives `nt_outer=1` as a 4th TIME-GATED lane** (the
prompt with zeros beyond m_head), so lanes are 4 per job: E/P/L/P_HEAD, 32 jobs at NSB=128.

Gates (all on cx19 A40):

* **M1/M2** `scripts/gnss/n2dualtest` (seeds 12345/2/777, int32 = exact equality):
  dual(128+0) == n2k(128) **bitwise**; dual(128+128) AA block is a **verbatim prefix** of each
  (t,f) slice, bitwise == n2k(128) (so the N^2 pass-through split is one strided copy /
  `cudaMemcpy2DAsync`); full dual == CPU reference over 256 stations incl. random RFI mask;
  class-mask run bitwise == full run on its tiles.
* **M3** `scripts/gnss/n2dualxval` (real replicas via the shipped `enqueue_batch_nm`, identical
  frame bytes to both paths so the difference isolates replica quantization):
  **orientation: path A == conj(V_BA)/s** (mixed tiles put the synthetic station on the row
  side; the consumer conjugates); clean-channel quantization cost **+0.04 dB, 2.3 mrad**;
  row-level amplitude agreement -0.08 dB; per-entry phase ~17 mrad rms; M^2 replica
  cross-terms vs float reference 3e-3; P_HEAD gating verified with a combined-period boundary
  mid-window (m_head 852).
* The hop-order contract (11.3) is decided -- option (a), time order, now written into
  `gnssRecord.hpp` and on `launch_waveform`'s wave param. 10.6b-style sorted-hop storage is
  disallowed.

### 11.7 M4 BUILT AND SMOKE-PROVEN ON cx19 (2026-08-06, fake-input; live start pending)

The kotekan integration exists and ran end to end on cx19's GPUs at production shape:

* `lib/cuda/cudaCorrelatorDual.{hpp,cpp}` -- clone-wrapper (cudaCorrelator untouched): voltage
  ring + a per-frame-slot synth array (named gpu mem `gnss_synth`, 0x88-filled at
  construction, NO ring semantics -- see below), n2k_dual launch, GPU-side split: N^2 prefix
  -> standard `correlation_buffer` via one cudaMemcpy2DAsync (downstream shape-identical),
  mixed+M^2 tiles of the comb channels -> ~0.75 MB `gnss_tiles` output
  (`cudaGnssN2Gather.cu`); the 107 MB extended triangle never leaves the GPU.
  `rfi_all_pass: true` = constant all-ones mask, behavior-identical to production (which runs
  first-stage excision OFF).
* `lib/stages/gnss/cudaGnssInject.{hpp,cpp}` -- ordered BEFORE the correlator in the same
  cudaProcess/stream; that ordering is the whole synchronization story. Consumes the voltage
  ring for its TIME REFERENCE only (FATALs if `time_downsampling_fpga != 1` rather than
  inject at a wrong code phase). Seeds via the tracker's own state class
  (`cudaGnssInjectState`, endpoints `/gnssN_inject/set_seeds`; `snapshot_seeds` factored so
  TTL semantics cannot fork; `propagate_seed` already shared). Synthesis per record (4x2048,
  `GnssCudaDespread::enqueue_waveform`) -- replicas bit-identical to the tracker's for the
  same seeds. Pack: `launch_pack44`, lane = 4*prn_slot + E/P/L/P_HEAD (STABLE identity),
  head lane zeroed at m_head, scale 7/(3*rms) from the energy rows, clamp [-7,7]; consumers
  normalize by the M^2 diagonal so the scale cancels.
* Config: generator `--n2-dual` (+`--n2-dump`) -- keeps `run_send_voltage`, adds the per-GPU
  `gnssN_n2dual` process; `config/generated/chord_gnss_cx19_n2dual.yaml` written (tracker
  blocks verified identical to the live config). Broker: cx19's `gnss{0..1}_inject` added to
  `broker_up.sh`'s tracker list -- same payload, zero broker code change.

**Fake-input smoke (testDataGen at [8192][384][2][64], isolated: REST 12055, no aggregator
sends, records to scratch):** constructed and ran on BOTH GPUs. Pre-seed the gathered tiles
were ALL-ZERO (0x88 background + addressing clean); after POSTing a PRN-3 seed to
`/gnss0_inject/set_seeds`: lanes 0-3 live in the mixed tile, BB diagonal = lane energies
(purely real, as V_ii must be), P_HEAD at 55% of P (the per-record head fraction over 4
records), unseeded lanes exactly zero, frame exactly as sparse as it should be.

### 11.8 SOLVED (2026-08-06): the CHORD tracker never waited for its H2D copy

Measured on sky, controlled (same node, same PRN, minutes apart, peers as reference):

    cx19 + path B   PRN10 amplitude 1.2e-05  coh_frac 0.09  deep_snr 2.1-2.8
    cx19 plain      PRN10 amplitude 9.2e-05  coh_frac 0.76  deep_snr 13.05
    cx27 / cx43     PRN10 amplitude 1.2e-04  coh_frac 0.76-0.78  deep_snr 13.4-14.0

cx19 on the plain config reaches deep_snr 13.05 at **adr_lock_s = 23 SECONDS**, so this is not
warm-up: the relevant constants are elem_sum_tau_s 5 s (warm ~15 s) and the deep window 1.05 s.
With path B it sat at 2.x for 25 minutes. Not the tile dump either -- the non-dump n2dual run
was equally degraded.

SYMPTOM SHAPE, which is the lead: amplitude down 7.7x AND coh_frac collapsed. 1/sqrt(32) = 5.7x
is exactly the elem_sum MRC falling back to reference-element-only, and BOTH that calibration
and the coherent integration need a CONTINUOUS record stream -- so record discontinuity explains
both, where mere GPU slowness explains neither. But `kotekan_gnss_tap_dropped_frames_total` was
10 over 25 min (static), so the discontinuity is NOT the tap valve. Unexplained.

Candidate mechanisms, none yet measured:
  * `--n2-dual` restores `run_send_voltage` -- the 402 MB/frame (9.6 GB/s PCIe per GPU) upload
    the GNSS configs deliberately drop. Most suspicious single difference.
  * the injector duplicates synthesis (tracker + injector both run launch_waveform).
  * the dual correlator's 3.1 ms kernel serialising against the tracker's on one GPU.

NEXT DIAGNOSTIC (do this before any more sky time): bring cx19 up on the n2dual config with
`log_kernel_split: true` and the cudaProcess `log_profiling`, and watch the RECORD SEQUENCE for
gaps (GnssGpuRecordAssemble / the combiner's record export) rather than the tap counter. If
records are contiguous, the mechanism is not discontinuity and the amplitude/coherence
collapse needs a different explanation.

**ROOT CAUSE, and it was never path B.** `cudaInputData` is COPY_IN -> stream 0;
`cudaGnssChordTrack` is KERNEL -> stream 2 (`cudaCommand::set_command_type`). The only thing
ordering the despread after the frame's upload is `cudaStreamWaitEvent(stream, pre_events[0])`,
and this stage threw the events away with `(void)pre_events`. Its airspy sibling
`cudaGnssTrack` has always done the wait -- exactly the drift `cudaGnssChordTrack`'s header
warns about. Latent for months because stream 0 is otherwise idle in the GNSS-only configs, so
the 1.8 MB tap copy always beat the kernel; `run_send_voltage`'s 402 MB/frame on the same copy
engine delayed it past the launch and the tracker despread the PREVIOUS frame's voltage against
THIS frame's replicas.

CONFIRMED ON SKY, same n2volt config that reproduced the failure:

    cx19 before  top deep_snr 1.8   max amplitude 1.69e-05  coh 0.09
    cx19 after   top deep_snr 11.8  max amplitude 2.19e-04  coh 0.72
    cx27/43/44   top deep_snr 12.9-13.1  max amplitude 2.0-2.1e-04  coh 0.76-0.77

(cx19's amplitude is the HIGHEST of the four; its deep_snr/coh trail only because it was minutes
into lock against the others' hours.)

**The diagnostic lesson.** Every input checked out -- records 100% contiguous, hops correct,
replica energy stable to 0.2%, absolute hop within one frame of the fleet, and the voltage bytes
statistically identical to a healthy node (rms 1.797 vs 1.787 lsb, dumped via the `peek_hold`
added for this). Data right + replica right + time right + result noise => the two never met.
When every input is verified and the output is uncorrelated, stop testing inputs and look at the
ORDERING between them.

Any new cudaCommand consuming a COPY_IN buffer must honour `pre_events[0]`. `(void)pre_events`
in a KERNEL command is a bug unless everything it reads is written on its own stream.

### 11.9 ON-SKY VALIDATION (2026-08-06): path B detects the sky

First clean measurement, cx19 on chord_gnss_cx19_n2dump.yaml with the 11.8 fix in place (path A
healthy ALONGSIDE path B: P18 14.2, max amplitude 1.07e-04).

Signal test: CROSS-FRAME ELEMENT-VECTOR COHERENCE (scripts/gnss/n2tiles_signal.py). Do NOT use
|P|/rms(E,L) -- the E/P/L replicas are ~0.97 coherent within one 195 kHz channel, so that
statistic cannot tell signal from noise here (11.10). Instead: for a real source the 32-antenna
complex vector is stable frame to frame and a common phase rotation cancels in a normalized
inner product, so signal -> ~1 and noise -> 1/sqrt(32) = 0.177.

    PRN   path-B coh   path-A deep_snr
     27      0.534         12.31
     32      0.446         11.97
     18      0.394         11.28
     23      0.387         11.74
     24      0.321         13.21
     10      0.298          1.23
      8      0.202          0.66
      2      0.193          2.16

    Spearman = 0.667 (n=8); path-A-strong mean coh 0.416 vs weak 0.231; PERFECT SEPARATION
    (min strong 0.321 > max weak 0.298), weak group sitting on the 0.177 noise floor.

WHAT THIS PROVES: the whole path-B chain works on real sky -- broker seeds -> shared
propagate_seed -> per-record synthesis -> conj_replica pack into 4+4b lanes -> the dual N^2 ->
the GPU tile gather -> host. It detects the same satellites the shipped tracker does.

PER-ELEMENT A/B ON REAL SKY -- DONE (scripts/gnss/n2sky_fetch.py + n2skyab.cpp). Path A
exports only the elem_sum-calibrated header value, so instead of a live export we pull the
actual voltage bytes (peek_hold on the tap buffer) plus the tracked (cp, doppler), and run BOTH
paths over those identical bytes offline: path A through the shipped enqueue_batch_nm, path B
through the production launch_pack44 + n2k_dual. Measured, 8 PRNs x E/P/L, all 4 records:

    |corr| over the 32-element vector : median 0.994-0.996, min 0.979, max 0.999
    phase of the fitted scale k       : within +-22 mrad
    |k| spread                        : 0.0037-0.0042 (the pack's per-lane 1/s)
    unrelated 32-vector would give    : 1/sqrt(32) = 0.177

So path B reproduces the shipped despread per element on real sky, at the level the offline M3
gate predicted (+0.036 dB / 2.29 mrad quantization cost).

⚠️ MAKE THE COMPARISON SCALE-FREE. V/M^2 and corr/energy differ by the pack's 1/s (V ~ s,
M^2 ~ s^2), so a raw ratio measures the QUANTIZER, not the agreement -- the first version of
n2skyab did that and reported 5-90x. Comparing the 32-element VECTORS removes k entirely, and
as a bonus needs no correct seed: identical inputs must agree even on noise (PRN 2, stale seed
cp=0, still gives 0.985).

### 11.10 Two statistics that are BLIND on CHORD (do not reuse them)

1. |P|/rms(E,L) per channel. Measured from the M^2 block: <E,P> = <P,L> = <E,L> = 0.97 at
   0.5-chip spacing, because each PFB bin is ~195 kHz against a 10.23 Mcps code. E ~ P ~ L is
   what STRONG SIGNAL looks like too. This statistic reported "path B is despreading noise" on
   a chain that was working.
2. amp_snr from a single get_status sample. It swings 0.4 -> 45 record to record. Use
   `amplitude` and `coh_frac`, and compare against another NODE at the same instant -- the sky
   moves, so a single PRN is not a stable probe across a long session.

### 11.7 (continued) Remaining for M4 sign-off

Remaining for M4 sign-off (needs a sudo node start): live run on sky under
`chord_gnss_cx19_n2dual.yaml` (`GNSS_CFG=... node_up.sh cx19`), broker restart with the
inject endpoints, then the A/B: mixed tiles vs the path-A tracker's EPL on the same seeds
(conjugate + normalize by M^2 diag; run with `code_trim: false` on the tracker -- the
injector applies no trim), and the sky-sign check of the conjugation orientation.
Startup datapoint: tracker + injector each build a replica bank -- construction is ~2x
(minutes at 32 PRNs); consider sharing the bank between the two states later.
**M5** -- consumer stage -> frame-length records -> assembler/broker (survey
hops_per_record assumptions first).

