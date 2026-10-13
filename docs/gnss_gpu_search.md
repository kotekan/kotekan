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

## 6. Staging -- done (2026-08-05)

A1 (device aggregate and on-device peak, `lib/cuda/cudaGnssAcquireKernel.{hpp,cu}`), A2 (device
correlate, `lib/stages/gnss/GnssCudaAcquire.{hpp,cpp}`, 113x at blind dimensions), A3 (every
eligible PRN every pass at its hinted grid, throttled to a GPU duty budget rather than run flat
out) and A5 (the refine on the GPU, `gnss::refine_peak_cuda` behind `use_cuda_refine`, a pure
reuse of the despread) are built and run in the aggregator (`use_cuda_acquire: true`). A4
(fusing steps 3-5) was not built. A6 (other signals) went the dead-reckon way instead: only
gps_l5 runs a search (`docs/CHORD_MULTIBAND.md` §5). The staging journal and the first live
run (old §6b) are at `be7d46e8e:docs/gnss_gpu_search.md` §6.

## 7. Traps specific to this port, and to path B's bring-up

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
11. **Check `code_length` against `secondary_length` before reasoning about overlay
    boundaries.** `GPS_L5_Q_NH` bakes NH20 into a 204600-chip code (`secondary_length 0`), so
    its code period is 20 ms and a 2 ms refine window holds no boundary at all. Counting in
    PRIMARY periods says "two boundaries" and sends you off to generalise a kernel for nothing.
12. **`despread_batch`'s job arena is `n_prn`, not `max_specs(n_prn) = n_prn * MAX_REC`** (that
    bound is the device path's). Over-batching shows only as `jobs upload: invalid argument`
    at the H2D, far from the cause; chunk against `max_batch_specs()`.
13. **A KERNEL cudaCommand that reads a COPY_IN buffer must wait on `pre_events[0]`.** Path A's
    tracker discarded it (`(void)pre_events`) and, once another upload shared the copy engine,
    despread the previous frame's voltage against this frame's replicas (2026-08-06). When
    every input checks out and the output is uncorrelated, look at the ORDERING between them.
14. **Two statistics are blind on CHORD.** `|P|/rms(E,L)` per channel: the E/P/L replicas are
    ~0.97 coherent within one 195 kHz channel, so E ~ P ~ L is what STRONG signal looks like
    too. And `amp_snr` from one `get_status` sample swings 0.4 -> 45 record to record; use
    `amplitude` and `coh_frac`, against another node at the same instant.

## 8. Reproducing the baseline

```
KBUILD=build_nodpdk scripts/gnss/site/build_tool.sh aggbench      # on cf06
scripts/gnss/aggbench --nd 321 --mp 3125 --threads 16        # blind dims, direct vs folded
scripts/gnss/aggbench --nd 5 --mp 400                        # fast correctness check
```

`build_tool.sh` now takes `KBUILD` (cf06 has no DPDK or testing lib) and writes `<tool>.<host>`
with a `<tool>` symlink, because `/home/kvand` is NFS-shared and two hosts building the same
tool clobbered each other -- which presents as "the tool suddenly segfaults".

---

## 9. What the path-A tracker work left in the synthesis kernel (2026-08-05)

The journal of `cudaGnssChordTrack`'s kernel (the path-A tracker, removed in f91b6b15e) is at
`be7d46e8e:docs/gnss_gpu_search.md` §9. Three of its results ship in `launch_waveform`
(`lib/cuda/cudaGnssChordDespread.cu`), which path B's injector runs, and §10 builds on them:

* **The block width is the DRAM-traffic knob.** A block re-walks its (PRN, channel) Phi slice
  once per hop pass per trial; 1024-wide blocks cut 24 re-walks to 2 (3.67x, `wave`
  bit-identical at every width). `__launch_bounds__` is load-bearing (unbounded, the gather
  needs ~72 registers and a 1024-thread launch fails), and the width is a request:
  `launch_waveform_tuned` clamps to each instantiation's reported `maxThreadsPerBlock`.
* **Trial fusion** (`chip_gather3`): one chip walk serves all three E/P/L trials, whose taps sit
  inside a ~625-entry window and share L1 lines. 7.6x over the original 256-wide three-gather
  kernel.
* **`fmaf` pinning.** `cmulf` and the gather's accumulate use explicit `fmaf`: ptxas's freedom
  to contract `a.x*b.x - a.y*b.y` changed with the surrounding code and broke a bit-exactness
  gate the new gather was innocent of. Only a test against an untouched reference kernel saw
  it; a bench that compares a kernel with itself cannot.

That work's gather-truncation sweep (a one-sided cap that cliffed between 105 and 120 chips) is
superseded by the centered 80-chip window the manifest runs (`despread-max-chips`,
`config/gnss_fleet_chord.yaml`).

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

## 11. Path B: the correlation folded into CHORD's N^2 kernel

Path B is the live path: the replicas are injected as extra stations into a two-input clone of
CHORD's N^2 kernel (`external/n2k_dual`, `cudaCorrelatorDual`). Path A, the standalone
`launch_correlate_nm` tracker, was removed in f91b6b15e. 11.1-11.3 are path B's shape and
contract, 11.6 its gates, 11.11 the consumer and 11.15 the coexistence with production's N^2.
The bring-up journal (11.4-11.5, 11.7-11.10, 11.12-11.14 and 11.16-11.33) is at
`be7d46e8e:docs/gnss_gpu_search.md`.

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

### 11.6 STATUS 2026-08-06: n2k_dual built; M1-M3 gates PASS

Decisions in force: straight to the two-input kernel (the 96 dead elements 32-127 are being
populated with real dishes over ~a month -- not available); **clone, don't clobber** (others
develop n2k: `external/n2k` and `cudaCorrelator` are untouched; the clone is
`external/n2k_dual/`, namespace `n2k_dual`, divergences listed in its README); keep
`sub_integration_ntime` 8192 (the dual pass later moved to 2048, one record: see 11.11).

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

### 11.11 M5 DESIGN: the consumer, and why it needs the replica energy

GOAL: path B produces a frame in the EXISTING epl layout (gnssGpuChain.hpp: FrameHdr +
window_start[MAX_REC] + PrnCtl[MAX_REC][n_prn] + corr[jobs][chan][elem] + energy[jobs][chan]),
so the SHIPPED GnssGpuRecordAssemble consumes it unchanged. The assembler is header-driven --
it never reads hops_per_record -- which is what makes this a drop-in.

THE SCALE, and it is not optional. Measured in the on-sky A/B
(be7d46e8e:docs/gnss_gpu_search.md 11.9): V = s*corr_A and M^2 = s^2*E_R, so
V/M^2 = amplitude_A / s. The pack normalizes every lane to a CONSTANT
energy (measured: M^2 diagonals agree within 1% across all PRNs), i.e. s ~ 1/sqrt(E_R), so
V/M^2 carries a spurious sqrt(E_R) per (lane, channel). Feeding that to the assembler would
WEIGHT THE CHANNEL COMBINE BY THE PFB RESPONSE instead of by SNR. The fix needs the true
replica energy E_R, which launch_waveform already computes and the injector already has:

    corr_out   = V                (the mixed tile, orientation: path A == V with conj_replica)
    energy_out = M^2 / s          (then corr/energy = s*V/M^2 = amplitude_A exactly)

SHAPE: cudaGnssInject writes the control block ITSELF -- FrameHdr, window_start, PrnCtl and the
energy rows are all things it already knows (it builds the specs and gets the energy array back
from launch_waveform). The consumer then only has to gather corr from the tiles. Division of
labour:

    cudaGnssInject      -> gnssN_n2ctl_buf   (hdr + winstart + PrnCtl + energy, ~10 kB/frame)
    cudaCorrelatorDual  -> gnssN_n2tiles_buf (the mixed + M^2 tiles, 0.75 MB/frame)
    GnssN2RecordAssemble (new host stage) -> epl-format frame -> the EXISTING assembler

CADENCE: the dual correlator integrates one 2048-hop record (`sub_integration_ntime` 2048 on
the n2dual pass), so a 41.94 ms frame carries n_rec = 4 records, the same records the assembler
always consumed. (The first design integrated the whole 8192-hop frame; see the journal.)

## 11.15  THE N^2 ARCHITECTURE: three options, measured (2026-08-08)

KV's question -- "are we producing an extended (N+M)^2 across all 768 freqs/node, or
full-efficiency N^2 on 761 plus (N+M)^2 on 7?" -- is the one that decides this, and it turns
on a property of the kernel rather than of the config. **`freq_map` and `block_class_mask` are
both GLOBAL to a launch** in n2k_dual. "AA on every frequency, MIXED|BB on the comb" is
therefore not expressible as one launch, however the flags are set.

The three block classes partition the extended triangle exactly -- AA (antenna x antenna),
MIXED (antenna x replica), BB (replica x replica) -- so any correct scheme computes each
exactly once. The options differ only in how they cover the frequency axis.

| | what runs | GNSS cost/frame | science N^2 |
|---|---|---|---|
| **A. path A today** | production N^2, all freqs + `cudaGnssChordTrack` | **17.712 ms** | production's, untouched |
| **B. coexistence** | production N^2, all freqs + dual in FREQ-MAP mode (MIXED\|BB, 7 chan) | **2.512 ms** | production's, untouched |
| **C. replacement** | dual in FULL mode (all 384 local freqs, NS=256), run_n2k dropped | **6.141 ms** | dual's prefix, needs validation |

All measured on cx19, 200-800 frame averages: `cudaGnssChordTrack` 17.712, `cudaGnssInject`
2.435, `cudaCorrelatorDual` 0.077 mapped / 3.698 full.

**Coexistence (B) wins, and not narrowly.** It is 7x cheaper than path A on the GNSS branch and
2.4x cheaper than replacement, and it leaves the science correlator BIT-IDENTICAL by
construction -- it is the same stage, untouched, not a reimplementation whose output has to be
proven equivalent. Option C's whole extra cost is waste: full mode computes the extended
triangle on ALL 384 local frequencies, and on 377 of them the M half is entirely zeros.

**No double counting in B.** Freq-map mode passes `BLOCK_MASK_MIXED | BLOCK_MASK_BB` -- AA is
explicitly excluded, with the source comment "the antenna (AA) block is production's N^2". So
AA comes once from `cudaCorrelator` on all 384 channels, MIXED and BB once from
`cudaCorrelatorDual` on 7. The only redundancy is a DATA LOAD: the mixed blocks need the
antenna panels for those 7 channels, so 7/384 = 1.8% of the antenna voltages are read twice.
That is most of why the mapped launch costs 0.077 ms rather than nothing.

The property that blocks C is exactly the property that makes B safe: freq-map mode cannot
feed the science pipeline BECAUSE it never computes AA, which is also why it cannot collide
with the correlator that does.

**Measured working, cx19 2026-08-08.** Both correlators on one GPU, N2Accumulate live and
accumulating, path B matching path A on the same sky: PRN 21 A 25.6/0.925 vs B 25.2/0.924,
PRN 8 A 24.9/0.922 vs B 25.6/0.928, PRN 9 A 23.7/0.917 vs B 24.0/0.919. Tap drops 7-8 frames
total, 0 on the search taps.

**Config:** the generator keeps production's N^2 stages beside the dual unconditionally (the
flags that once selected this are retired). Two things are load-bearing and were each found by
a failure:
  * the dual needs its OWN gpu array name. Both correlators allocate
    `<n2k_correlation_name>_buffer` sized from their own station count -- 28,311,552 B at
    NS=128 against 113,246,208 B at NS=256 -- and sharing "correlation" is a hard startup
    failure that presents as an ACTIVE unit with zero packets moving.
  * `host_correlation_buffer` must survive beside the dual; an early generator dropped it for
    every dual config, which left run_n2k pointing at a deleted buffer.

Since 2026-10-06 the generator folds every path-B chain on a GPU into ONE dual pass over the
union of their combs (`docs/CHORD_VIS_CAPTURE.md`, "One pass per GPU"); the coexistence above
is unchanged by it.
