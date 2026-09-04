# The ms-split search — design plan (2026-08-02)

## 1. Why

The acquire integrates coherently over **one replica period**, `Mp = code_samples /
gcd(fft_len, code_samples)` hops. Nobody chose that as a window length; it is what the cyclic
correlation needs in order to be valid, and on airspy it happened to equal exactly one code
period:

| | airspy | CHORD |
|---|---|---|
| code period | 5000 samples | 3,200,000 samples |
| `fft_len` | 20 | 16384 |
| hops per code period | **250 (integer)** | **195.3125** |
| `Mp` | 250 hops = **1 ms** | 3125 hops = **16 ms** |

`fft_len` divides `code_samples` on airspy and does not on CHORD, so the same line of code
gives a window 16× longer than the lag search actually needs — and that one number then drags
two more axes with it:

* **coarse lag 3125 instead of ~196.** The lag axis is that long *because* the NH20 overlay is
  applied across the 16 ms window, which makes the replica non-periodic over the window so all
  3125 lags are genuinely distinct. For the bare code, lag is ambiguous modulo one code period
  — 196 hops covers the whole space. (This is also exactly why the coarse-lag fold in
  CHORD_GNSS_STATE.md §6.3 measured wrong: it assumed a periodicity the overlay destroys.)
* **Doppler 13 bins instead of 1.** A window of length T resolves Doppler to 1/T. 16 ms → 62.5
  Hz, so the grid is 31.25 Hz and the hinted ±200 Hz needs 13 bins. 1 ms → 1000 Hz, and ±200 Hz
  is a single bin.
* **20 NH alignments instead of 1.** 16 ms straddles 16 of the 20 overlay chips at an unknown
  starting position, so all 20 are searched. 1 ms spans exactly one overlay chip: a constant
  ±1 on the whole window, invisible to |D|².

Measured cost today, per PRN per pass: **2.81 G cross-channel ops** (13 × 3125 × 128 × 27 × 20).

## 2. The geometry, stated exactly

```
code period      3,200,000 samples  =  195.3125 hops  =  195 hops + 5120 samples
slip per period  0.3125 hop = 5120 samples = 16.36 chips
16 code periods  51,200,000 samples =  3125 hops exactly      <- why Mp = 3125
```

The **sample** grid is already aligned (3.2 GHz × 1 ms is an integer). Only the **hop** grid is
not, and it cannot be re-referenced: the F-engine applied the PFB in hardware, so by the time we
see the data the hop boundaries are fixed. There is no "pad by a fractional sample" available —
the slip is 5120 whole samples per ms, and it is a *sub-hop* offset, which is precisely the
quantity the fine-lag axis already represents.

## 3. Algorithm

### Phase A — incoherent (do this first)

Split the snapshot into `K` sub-windows of `N ≈ 196` hops (one code period of lag ambiguity).

1. **Per sub-window k**, build the replica at that sub-window's own start phase and correlate.
   Use a **zero-padded (linear) correlation**, not cyclic: over 195 hops the replica advances
   0.9984 code periods, so a cyclic wrap is 16.4 chips wrong. FFT length 512 (≥ 2N) removes the
   artefact for ~2× the transform cost, which is not where the money is.
2. **Cross-channel combine** to recover fine lag, exactly as now:
   `D(s) = Σ_c P_c e^{i2πf_c s/sph}`.
3. ~~**Roll each sub-surface** by the known slip before summing.~~ **Not needed — measured
   2026-08-02.** Every generator here forms `C(n) = arg + n·cps` over the *absolute* sample
   index, so a sub-window's replica, generated at its own start with argument 0, advances in
   lockstep with the data. The peak lands at the same lag in every sub-window. Confirmed by
   K = 1, 2, 4, 16 all peaking in the same cell with power growing ~linearly in K.

   What *does* drift is the code Doppler the replica omits: `N·sph·cps·ε` = **0.0165 chips per
   sub-window**, so K = 100 smears the peak by 1.65 chips. That is a sensitivity question for
   Phase A at full length, not a bookkeeping one, and it is 2000× smaller than the 35.9 chip
   nominal slip the roll was written for.
4. **Sum |D|²** across the K sub-windows.

### Phase B — coherent recombination (only if Phase A's sensitivity is not enough)

Keep the sub-surfaces **complex** instead of squaring, then combine afterwards:

```
for each NH hypothesis a in 0..19:
    for each fine-Doppler bin δf:
        S_a,δf(cell) = Σ_k  nh_sign[(k + a) mod 20] · e^{i2π δf k T_sub} · D_k(cell)
```

This recovers full 20 ms coherent gain. It is **exact** — the sub-correlations are genuinely
independent complex quantities and are combined by addition. It is NOT the NH restructure that
failed in §6.3: that one tried to synthesise 20 shifted copies from a *single* correlation via
the cross-channel reconstruction, which is only approximate for a windowed PFB.

The cost is that coherent recombination needs Doppler good enough not to smear across 20 ms
(δf ≲ 8 Hz), so the fine Doppler axis returns — in the combine stage, where it is a phase
rotation on precomputed values, not a re-correlation.

## 4. Cost and sensitivity

Per PRN per pass, cross-channel ops (the dominant term):

| | Doppler | lags | fine | chan | align | sub-win | ops | vs now |
|---|---|---|---|---|---|---|---|---|
| now | 13 | 3125 | 128 | 27 | 20 | 1 | 2.81 G | — |
| Phase A | 1 | 196 | 128 | 27 | 1 | 100 | 67.7 M | **41×** cheaper |
| Phase B | 1 | 196 | 128 | 27 | 1 | 100 | 67.7 M + combine | ~4× cheaper |

Sensitivity, SNR ∝ T_coh·√N_incoh over a 100 ms snapshot:

| | coherent | incoherent | SNR figure |
|---|---|---|---|
| now | 16 ms | ×√6 | 39.2 |
| Phase A | 1 ms | ×√100 | 10.0 — **3.9× worse (≈6 dB)** |
| Phase B | 20 ms | ×√5 | 44.7 — slightly *better* than now |

Phase A's 6 dB matters: the current noise ceiling is 19.1 with `acquire_snr` 30, and ~51% of
live detections already sit in that margin. Phase A alone would make that worse. **It should be
deployed together with, or after, the window/overlay bookkeeping fix** (route window w into bin
`(a + 4w) mod 20`), which restores multi-window integration and brings the ceiling to 4.45.

## 4b. The peak mapping — measured 2026-08-02, `gnss::ms_split_peak`

`channelized_peak` cannot be used on an ms-split surface. Its tau → code-phase mapping assumes
two things the shipped geometry arranges and this one cannot, and both were worth thousands of
chips:

| term | why | size at the bench point |
|---|---|---|
| `phi_r0` | the replica's own code phase at its index 0. The search anchors repl0 at `Mp·fft_len` = 16 exact code periods, so it is ~0 there. The ms-split anchors N hops before *its* data, at an arbitrary absolute sample. | **7611 chips**, and it moves −52.3776 per hop of uptime |
| `Ns·cps` | at the peak the cyclic correlation reads replica indices `(m−q) mod Mp`, a whole `Mp` hops on. Zero for the shipped `3125·16384` (16 periods); `2N·16384` is 2.007 periods. | **72.0 chips** |
| fine sign | the coarse and fine halves of the lag carry **opposite** signs, so `AcquisitionSurface::tau()`'s `q·sph + i·s_step` is not the delay. Injecting +5 chips moves the fine index +49 columns at `s_step` 32 = 9.8/chip = exactly `1/cps`, while `q` moves the other way. | up to **52.4 chips** (one hop) |

The fine index must also be folded signed into `(−sph/2, +sph/2]` first: the cross-channel DFT
reports it modulo `sph`, so a small negative offset returns near the top of the axis and reads
as a whole extra hop. Every +52.35 chip outlier in the bench sweeps was this.

The shipped path never had to get the last two right, because `refine_peak` re-scans a full hop
either way and absorbs them. That is worth remembering before trusting any *other* coarse
number it produces.

**The lag window.** The peak search is restricted to `q ∈ [N, 2N)`. Those are the lags whose
replica indices are contiguous; for `q ∈ [1, N)` the sum straddles the replica's own wrap and
correlates over only `N−q` hops. Those partial lags are not small — the strongest stood at
**26% of the true peak's power, second place on the whole surface**, and it tracks the satellite
(19.09 hops per 1000 chips, exactly `1/cph`), so no statistic distinguishes it from a detection.
Accepting one seeds the tracker nearly a full code period off. `[N, 2N)` still covers the entire
ambiguity: one code period is 195.3125 hops ≤ N = 196.

**It only works at stride 1.** The mapping needs `s_stored == sph`, i.e. `g = 1`. At the 2-node
stride-4 comb the fine axis spans 4096 of 16384 samples and the recovered phase is quantized to
that: measured errors 0, ±13.33, ±26.42 chips — integer multiples of 13.09. Same structural
requirement as the grating lobes, arrived at independently.

## 5. Validation — bench before sky, no exceptions

`scripts/gnss/e2e` already injects a known code phase and reports the recovered phase and
period against ground truth. The ms-split path must be validated the same way, and the specific
things to test are the ones that have gone wrong before:

1. **Recovered cp and period vs injection**, across a cp sweep and a hop0 sweep — same
   acceptance bar as the current path (period 16/16, residual ≲ 0.2 chips).
   **Done 2026-08-02, stride 1** (`e2e --ms-split K --s-stride 1 --s-nchan 106`). Coarse phase
   tracks the injection to <0.05 chips over cp +0…+2500, N = 196…300, hop0 +0…+11, K = 1…16;
   end-to-end within-period residual −0.235 chips, and −0.337 at K ≥ 4 (the code-Doppler smear
   of §3.3). Under noise 10 and 20: worst 0.337 chips, **0/8 wrong lobe**. Period is still
   0/8 — Phase A carries no overlay period, as designed. That is the remaining gap.
2. **The sub-window roll (step 3)** — this is the exact bookkeeping that `acquire_windows` got
   wrong, where each window landed in a different NH bin and a second window *halved* SNR. Test
   it by summing K sub-windows on a noiseless injection and confirming the peak grows as K, not
   as √K and not at all.
3. **The zero-padding** — confirm a cyclic (unpadded) implementation reproduces the predicted
   16.4-chip wrap error, so we know the padding is doing what it claims.
4. **Measured SNR vs the analytic table above**, on identical injected data, both paths.

## 6. Risks

* **The 6 dB.** Phase A is a straight sensitivity loss. If the overlay bookkeeping fix does not
  land first, this trades a cost problem for a detection problem.
* **Phase B's Doppler coupling.** 20 ms coherent needs δf ≲ 8 Hz; the acquire's own
  interpolation error is a −2.0…+2.1 Hz sawtooth, so this is feasible but not with margin to
  spare. Measure before relying on it.
* **The lag ambiguity is only mod one code period.** The overlay period must then come from the
  NH combine (Phase B) or from continuity/GPS-time assist. Phase A on its own reports a phase
  mod 10230, which is *less* than the search reports today (mod 204600) — a regression in what
  the seed can carry, unless Phase B lands with it.
* It does not address the ±60 chip per-detection seed scatter, which is a separate open item.

## 7. Not yet answered

Whether the 100 sub-correlations can share work. Sub-window k's replica differs from k−1's only
by a known phase advance, so the per-channel filter (the slow half of the hop-rate generator)
should hoist across all K — the same trick the refine already uses. If it does, Phase A's
correlation cost drops further and Phase B becomes clearly the better deal.

## 8. Measured cost, 2026-08-03 -- the acquire was never the bill

Timed at live parameters (stride 1, 106 channels, fine_step 128, refine_step 77, dop +-200 Hz,
12 threads, one PRN). `GNSS_MSSPLIT_PROFILE=1` and e2e's `[stages]` line:

| | acquire | nh-postfix | refine | total |
|---|---|---|---|---|
| shipped | 42.18 s | -- | 88.12 s | 135 s |
| ms-split K=1 | **0.55 s** | 32.55 s | 89.51 s | ~123 s |
| ms-split K=16 | 12.36 s | 41.41 s | **874.39 s** | 933 s |

Three things, and only the first was in the plan:

1. **The acquire claim holds.** 0.55 s vs 42.18 s is **77x**. Inside the K=16 acquire: replica
   11.77 s, correlate 0.13 s, aggregate 0.44 s -- it is now ALL replica generation, which is
   still not hoisted across sub-windows (§7).
2. **The acquire is only ~30% of the bill.** The REFINE is 88 s and is identical in both paths
   (`refine_peak` ignores `dims`; 426 evals either way; 0.21-0.33 s/eval measured at 9, 33 and
   129 evals in both). A free acquire takes the shipped path 135 -> 93 s and no further. **The
   ms-split cannot fix the ~10 min revisit on its own.** The refine -- 426 full replica builds
   per detection -- is the target nobody has looked at.
3. **K=16's 874 s refine is an ALLOCATION artefact, not algorithmic.** Same 426 evals, same
   inputs; K=1 refines in 89.51 s (== shipped 88.12) and K=16 in 874 s. Cost per eval is flat
   to 129 evals (0.216-0.272 s) then jumps 9.5x -- a cliff, not a slope. `ms_split_accumulate`
   allocates a fresh `repl` and a fresh `P` per sub-window (16 x 106 channel vectors), and the
   refine's 12 OpenMP workers then pay ~10x on their own 2.65 MB per-iteration allocations.
   Fix: hoist `repl`/`P`/`d` out of the k-loop and reuse them. That also removes most of the
   remaining replica cost, since the hop-rate FILTER is Doppler-dependent, not
   sub-window-dependent.

**Revised plan.** Reuse the buffers (fixes 3, cheap). Then attack the refine, not the acquire --
that is where the revisit actually lives. The "258x cheaper -> revisit in seconds" claim in §4
was an op count for the acquire alone and does NOT translate to pass time; §4 stands as an op
count and nothing more.
