# GNSS GPU pipeline — optimization queue (opened 2026-08-28)

Ranked by measured lever. §10.6c of docs/gnss_gpu_search.md is the governing measurement:
**this kernel is DRAM-FOOTPRINT-bound, not coalescing-bound** (fp16 won 1.27-1.37x by halving
the table; hop-sorting cut load sectors 62% and ran 17% SLOWER). Anything that shrinks the
resident table helps; anything that only rearranges access does not.

⚠️ EVERY item here is a kernel-contract change. #71 (carrier NCO: tried, worse, reverted)
is the precedent: *the gate tested the FORMULA, not the kernel contract*. Each item is
validated against the SHIPPED kernel's output at CHORD geometry (scripts/gnss/wavebench.cpp,
scripts/gnss/e2e), bit-compared where it should be bit-exact, before it is armed.

---

## 1a. Stop GATHERING the BB block  [DONE 2026-08-28]

**Simpler than planned: the BB block had NO consumer at all.** Its only read,
`m2`, was immediately `(void)`-cast -- so no M²-from-pack44 was needed, just deletion.
Tiles/channel 52 -> 16; tiles buffer 2,981,888 -> 917,504 B (7-chan GPU) and
2,555,904 -> 786,432 B (6-chan), a **69% cut**, times 10 chain-instances per node.
Gate: n2dualtest 6/6 (incl. [3] AA-prefix bitwise), e2e unchanged at 17.612 chips.

## 1b. Disable the BB block CLASS in the correlator  [MEASURED: NO WIN -- NOT ARMED]

**The projected ~33-50% correlator saving does not exist at the deployed geometry.**
Measured on an L40S with n2timing case [5] (added for this), three runs: **1.00-1.01x**,
0.150 ms -> 0.148 ms.

Why, and it is not subtle once measured: under `gnss_freq_map: true` -- which all 10
chain-instances set -- the launch already covers only the 7-channel comb, and AA is ALREADY
masked off (the existing code picks MIXED|BB when the freq map is on; my earlier claim that
all three classes were computed was wrong). So the comparison is **14 thread blocks vs 7**,
on a GPU with **142 SMs**. Both launches occupy under 10% of the device and are
launch-latency-bound, not throughput-bound. Halving an already-empty launch buys nothing.

The correlator is simply not where the time is: its marginal cost is 0.150 ms against
synthesis at ~536 us for a single 11-job x 7-chan launch (§10.6c). **That reranks the queue:
item 2 (shared Phi) is the top item, and 1b is closed.**

KEPT from this work, because both are worth having:
  * n2dualtest [5b] `class-mask AA|MIXED (SHIPPED) vs full, bitwise` and [5c] `BB region
    left UNCOMPUTED` -- these prove the mask path in the direction we would ship (gate [5]
    only ever tested the mirror image) and that a masked launch writes survivors at their
    FULL-TRIANGLE offsets. If BB is ever masked off for a non-performance reason, this is
    the gate that makes it safe.
  * n2timing [5], which records the null result so nobody re-derives the projection.

⚠️ ONE THING THIS BENCH CANNOT SEE: it times a launch in ISOLATION. In production 10
chain-instances share 2 GPUs, so fewer blocks might still help under contention. That is a
hypothesis, not a measurement, and it is not a reason to arm.

### superseded plan (kept for the argument, not the numbers)

n2dualtest gate **[5] `class-mask MIXED|BB vs full, bitwise` passes with 0 mismatches**,
and it compares the masked output against the full one AT THE SAME NS STRIDE -- so a masked
launch writes surviving blocks at their FULL-TRIANGLE offsets and simply leaves the disabled
ones untouched. Mixed-tile offsets therefore do not move, and the gather built in 1a needs no
change. What remains is plumbing `block_class_mask = BLOCK_MASK_AA | BLOCK_MASK_MIXED` from
kotekan config into DualCorrelatorParams (cudaCorrelatorDual never exposes it).
T=2 -> 3 blocks, so dropping BB is ~33% of the correlator.
⚠️ Extend n2dualtest with the AA|MIXED case (gate [5] masks off AA, the symmetric case)
before arming -- test what ships, not its mirror image.

### original note (kept: it is the argument for why BB is dead)

`GnssN2RecordAssemble` reads exactly **128 numbers per channel** out of the BB block: the
diagonal element of each diagonal tile (`slice[bb_k*512 + 34*ilo]`), one per synthetic lane.
It gathers 36 tiles x 512 int32 = 18,432 int32 per channel to do it -- **144x overshipping**,
and BB is **69% of all gathered tile bytes** (36 of 52 tiles/channel).

That value is the QUANTIZED replica's frame-integrated energy, used to undo the pack's
per-lane scale. `gnss_pack44_kernel` computes every quantized sample itself, so it can
accumulate sum|q|^2 per (lane, channel) as a by-product: 896 doubles/record. The TRUE
(pre-quantization) energy already ships this way via the ctl block (`d_energy0`), so the
transport exists.

With no consumer left, the BB block class can be switched off entirely --
`n2k_dual` already implements `block_class_mask` (see external/n2k_dual/README.md); the
kotekan wrapper simply never plumbs it. T=2 -> blocks AA/AB/BB = 3, so dropping BB is
**~33% off the correlator** on top of 69% off the tile gather and the EPL buffer.

GATE: compute M² BOTH ways for a run and assert bit-equality (or bound the difference)
BEFORE disabling BB. They should agree exactly; if they do not, that is a finding.

## 2. TWO Doppler-free tables per channel, shared by every PRN
##    [CORRECT AND GATED -- but SLOWER ON THE NODE GPU. NOT ARMED.]

### THE VERDICT (measured 2026-08-28, scripts/gnss/phisharegpu)

|  n_prn | per-PRN tables | L40S (cf06, 96 MB L2) | **A40 (NODE, 6 MB L2)** |
|--------|----------------|-----------------------|-------------------------|
|   8    |  58.7 MB       | 0.62x  SLOWER         | **0.88x SLOWER**        |
|  16    | 117.4 MB       | 1.22x  faster         | --                      |
|  24    | 176.2 MB       | 1.04x  (no change)    | **0.92x SLOWER**        |
|  32    | 234.9 MB       | 1.24x  faster         | **0.89x SLOWER**        |

**The mechanism is real and the hardware does not have it.** On the L40S there is a clean
crossover: below ~96 MB the per-PRN tables already fit L2, so the shared path only pays the
reconstruction (a per-chip sincos + 2 loads) and loses; above it, the shared 14.7 MB is
resident where 176-235 MB is not, and it wins. **The nodes are A40s with 6 MB of L2, where
14.7 MB does not fit either** -- so there is no residency to buy, only the cost to pay, and it
is slower at every PRN count.

That is the second projected win this session (after 1b) to be killed by measurement on the
target, both from architectural arguments that were sound in the abstract -- block counts
there, cache footprint here. The pattern is worth naming: *an argument about the memory
hierarchy is a hypothesis about a SPECIFIC part number.* cf06's L40S is not the node.

### WHAT IS NEVERTHELESS TRUE, and why the code stays (default off, gates green)

  * It is CORRECT: phisharegpu ALL PASS on both GPUs, worst 4.3e-6 at -5 kHz (77x under
    fp16), ddw == 0 exact. e2e 17.612 unchanged; n2dualtest 6/6; phishare ALL PASS.
  * It frees **1.8-2.4 GB of GPU allocation per node** (176-235 MB x 10 chain-instances),
    which is not the bottleneck today at 48 GB but is the kind of headroom the dual
    correlator's 1.61 GB synth ring eats.
  * It would be a ~1.2x win on L40S-class hardware at the deployed PRN counts. If the node
    GPUs are ever refreshed, this is already built and gated.

### THE OBVIOUS NEXT LEVER IF ANYONE REVISITS IT

The reconstruction costs a `__sincosf` PER CHIP, chosen for clamp-safety over advancing the
rotor multiplicatively (see chip_gather3's note). The measured penalty is only 8-12%, so
removing the transcendental could plausibly close it -- and on the L40S would widen the win.
That is a measurement, not a certainty, and it needs the clamp cases handled honestly.

### the design and its validation (unchanged, all of it still holds)

STATUS 2026-08-28:
  * [x] reconstruction validated on the real tables (phibits [2c])
  * [x] ANCHOR CHOSEN: t_prev, not the window midpoint. The midpoint is the natural anchor
        for the algebra but steps by ks, ks+1/2 or ks+1, so the kernel could not ADVANCE its
        rotor; t_prev steps by exactly ks or ks+1 -- two precomputed rotors and an exact
        select, no transcendental in the loop. Costs 6x accuracy (7.5e-6 vs 1.2e-6 at
        +-5 kHz) and still sits 44x below fp16.
  * [x] Psi built in ChannelizedReplicaBank::hoprate_filter behind `want_psi` (off by
        default, so every existing caller is untouched)
  * [x] reconstruction wired into the CPU generator (hoprate_stream_into), keyed on
        ddw = wc_stream - f.wc_built. ddw == 0 skips every branch, so a per-PRN filter is
        bit-identical to before BY CONSTRUCTION -- that is the safe fallback.
  * [x] VALIDATED IN THE SHIPPED GENERATOR, not just the algebra (scripts/gnss/phishare):
        one shared pair built at doppler 0, streamed against per-PRN truth --
        0 Hz 0.000e+00 (exact), +-5 kHz 2.8e-6, +-10 kHz 8.8e-6 relative. e2e VERDICT
        unchanged at 17.612 chips; broker gate EQUIVALENT.
  * [x] GPU: DespreadJob/PeelJob carry psiA/psiB/ddw (AT THE END -- both are built with
        positional aggregate initializers at three sites); chip_gather3 gains a SHARED
        template parameter; GnssCudaDespread grows a single shared (Phi,Psi) set and
        set_shared_phi(bool), which RETURNS whether it took (FDMA refuses it).
  * [x] GPU GATE PASSES (scripts/gnss/phisharegpu): drives the shipped despread through its
        public API with real tables, per-PRN vs shared. Worst 4.3e-6 at -5 kHz (77x under
        fp16); ddw == 0 EXACT. e2e 17.612 unchanged, phishare ALL PASS, n2dualtest 6/6.
        ⚠️ IT FAILED FIRST, at 3e-2..1.0 relative, and that is the entry worth reading:
        chip_gather3's psi arguments DEFAULT to null, so the despread kernel's call site --
        which I had not updated -- silently took SHARED=false and read the Doppler-free table
        RAW. The error was linear in Doppler and matched phibits' "no rotor" case, which is
        what identified it. Defaulted arguments make a missed call site look like working
        code. My first hypothesis (fp32 cancellation in dPsi - t0*dPhi) was WRONG and a
        five-line CPU probe killed it before it cost an afternoon.
  * [ ] NOT YET ENABLED ANYWHERE: no config sets shared_phi. Needs the stage-level flag,
        then wavebench timing + ncu lts__t_sector_hit_rate to confirm the L2 residency the
        whole win rests on (1b is the standing reminder that a projected win can be 1.00x).
  * [ ] the shared build currently reports 29.4 MB for 7 channels where the design says 14.7 --
        worth reconciling before arming; it may be that Lf is 2x what I assumed here, which
        would also mean the per-PRN footprint is 2x the 176-235 MB quoted (i.e. the case is
        stronger, not weaker) -- but it is UNVERIFIED arithmetic either way.

### the measured case for it

**Measured 2026-08-28 by phibits [2b]/[2c] against the REAL tables (hoprate_filter at CHORD
geometry), not from algebra.** Reconstruction error of a chip-window difference, worst case:

| Doppler offset | ships (2c) | vs fp16 (3.3e-4) |
|---|---|---|
| +-250 Hz  | 3.2e-09 | 103,753x better |
| +-1000 Hz | 4.8e-08 |   6,877x better |
| +-5000 Hz | 1.2e-06 |     275x better |
| +-10 kHz  | 4.8e-06 |      68x better |

So ONE pair of tables covers the whole sky -- no Doppler buckets at all -- with 275x margin
at the full GPS Doppler range.

### What the measurement corrected

My first form (rotor about the |w|-weighted centroid) gave 4.7e-3 at 5 kHz -- 14x WORSE than
fp16 -- and the error scaled LINEARLY with Doppler, which is the signature of a surviving
first-order term. dPhi is a sum of ROTATING phasors, so the centroid that cancels first order
is the COMPLEX one, c = sum_j j*w_j / sum_j w_j. With that, error goes quadratic and drops
four orders (2b). phibits' pre-existing [2] figure of 6.3e-1 is not a table property at all:
it is |exp(i*0.643)-1|, the pure endpoint tilt -- exactly what the rotor removes.

### The form to build

    dPhi(w0+ddw) ~ exp(-i*ddw*mid) * [ dPhi_0 - i*ddw*( dPsi - mid*dPhi_0 ) ]

with Psi[k] = sum_{j<k} j*proto[j]*e^{-i(off+w0)j} the second shared prefix table and
mid = (t0+t1)/2 the window midpoint, known from the INDICES. Splitting c into (mid) +
(c - mid) is what makes this cheap: the large part is a CONSTANT ROTOR advancing ~ks per chip
(one complex multiply, no transcendental), and the small remainder linearizes. Per chip:
+2 loads (Psi telescopes exactly like Phi), +1 complex FMA, +1 rotor multiply. No divide, no
exp in the loop. It measures BETTER than 2b's exp-of-complex-centroid ceiling, because the
linear form does not carry the exp's own O((ddw*c)^2) residual.

### Why it wins even though it doubles the loads

Footprint, which §10.6c proved is the axis: 2 shared tables x 1.05 MB x 7 chan = **14.7 MB
TOTAL, shared by every PRN**, against 24-32 PRN x 7 chan x 1.05 MB = **176-235 MB** today.
The L40S has 96 MB of L2 -- so the shared pair is L2-RESIDENT and reused by every job, while
today's per-PRN tables are streamed from DRAM once per job and evicted. This converts DRAM
traffic into L2 hits; the extra loads are cheap in exactly the way the DRAM ones were not.
With fp16 (item 3) on top: 7.3 MB, and the 275x margin is what pays for the fp16 error.

Also deletes ensure_phi's rebuild path entirely (cudaMalloc + 2 MB H2D per PRN whenever a
Doppler drifts past refresh_hz) -- the tables no longer depend on Doppler.

⚠️ GATE ON prn_df: valid only while every prn_df == 0 (CDMA). CHORD is; FDMA is already
refused in ChannelizedReplicaBank::swap_prn. Fall back to per-PRN tables otherwise.
⚠️ VALIDATE AGAINST THE KERNEL, not the algebra (#71 -- "the gate tested the FORMULA not the
kernel contract"). phibits proves the reconstruction; wavebench + e2e must prove the KERNEL.
⚠️ MEASURE, do not assume, that L2 residency materialises -- 1b is this session's reminder
that a projected win can measure at 1.00x. lts__t_sector_hit_rate is the counter.

### superseded first-pass note

Phi[c][k] = sum_{j<k} proto[j]*exp(-i(off_c +- wc)j), Lf+1 = 65537 entries x float2 x 2
images = **1.05 MB per channel PER PRN** -> 176 MB (24 PRN) to 235 MB (32 PRN) resident per
chain-instance. But wc splits: wc = w_common + dw, w_common = 2*pi*f_offset/fs = 2.31
rad/sample (the band carrier, identical for every PRN) and dw = 2*pi*doppler/fs <= 9.8e-6
(prn_df == 0: CHORD is CDMA-only, FDMA refused in swap_prn).

The kernel uses only the DIFFERENCE over one chip's tap range, and the two scales separate:
dw*Lf = 0.64 rad (significant -- and exactly the part that factors out as the endpoint
phase), dw*ks = 0.0031 rad over one chip (negligible). So
    Delta(dw) = exp(-i*dw*kbar) * Delta_0,   kbar = prototype-weighted window centroid
(the centroid kills the first-order term exactly; residual ~ (dw)^2*Var_w(u)/2 ~ 8e-7
relative, five orders below the 4-bit quantization floor).

k_lo advances by exactly ks per chip, so the correction is a CONSTANT ROTOR per chip: one
complex multiply, no transcendental, no lookup.

=> 7 x 1.05 MB = **7.3 MB instead of 176-235 MB, a 24-32x footprint cut** on the measured
lever. Also deletes ensure_phi's rebuild path (cudaMalloc + 2 MB H2D per PRN per Doppler
drift past refresh_hz). Composes with item 3 (-> 3.7 MB).

⚠️ GATE ON prn_df: valid only while every prn_df == 0. Fall back to per-PRN tables
otherwise, matching the FDMA refusal already in ChannelizedReplicaBank::swap_prn.
⚠️ VALIDATE AGAINST THE KERNEL, not the algebra (#71). wavebench A/B, bit-compare the
replica at CHORD geometry; e2e closure must not move.

## 3. fp16 Phi tables (task #22)  [BUILT + GATED 2026-08-31; config `phi_fp16`, default OFF -- ARM = one yaml line + node restart]

Already measured 1.27-1.37x, and §10.6c explains WHY it is the right lever (halves the
resident table). Composes with item 2: 7.3 MB -> 3.7 MB.

### production path (2026-08-31)

`DespreadParams::phi_half` -> launch_waveform takes the __half2 gather instantiation;
`GnssCudaDespread::set_phi_fp16` stores __half2 tables in the PhiCache (host-converts via
`gnss_cuda::phi_to_half`, half-size H2D); `cudaGnssChordTrackState` reads config `phi_fp16`
(default false) and READS THE RETURN (#96/#97: armed != in effect).

SCOPE GUARD: launch_despread/launch_peel read Phi as RAW float2, so despread_batch,
enqueue_batch_device and the peel THROW while fp16 is armed -- none is in the shipped inject
graph (gnss_chain.j2 runs cudaGnssInject -> enqueue_waveform only). fp16 and shared tables
(item 2) refuse each other, both directions.

GATE (scripts/gnss/phi16gpu, cx52 A40 2026-08-31, ALL PASS): drives enqueue_waveform -- THE
production call -- fp32 vs fp16 at the production record shape (11 jobs x 7 chan x 2048 hops).
Worst wave rel 1.055e-3 (~3x the 3.3e-4 storage floor = the ~52-term gather accumulation),
energy 2.3e-4, host-side correlation vs shared random voltage 6.3e-3 worst (noise-noise
quotient); fp32 flag-off DETERMINISTIC and arm/disarm ROUND-TRIPS byte-exact; both refusal
guards fire. TIMING through the shipped call: **1.55x** (0.623 -> 0.402 ms) -- measured with
~45% background GPU load, which biases the ratio TOWARD 1, so >= the wavebench 1.36x.
n2dualtest 7/7. e2e VERDICT unchanged at the CURRENT baseline (see below).

⚠️ e2e's DEFAULT-RUN BASELINE MOVED 17.612 -> 19.496 chips ON 2026-08-30/31, and it was NOT
this item: verified by building clean HEAD 51ff44836 (no fp16 edits) -- also 19.496. The
mover is #105 fix 3 (the local-tau Doppler vertex): e2e's acquire refines dop +1894.99 ->
+1891.19 Hz on the synthetic (err +1.6 -> -2.2 Hz, both fine for a u=1 acquire that is NOT
this estimator's gate -- acqbench is, at |err| <= 0.09 Hz), and every downstream cp shifts by
exactly the 5094.9 chips/Hz argument lever. Anyone diffing e2e against banked 17.612: compare
against 19.496 from 51ff44836 onward.

## 5. Tiled shared-memory streaming -- BUILT, MEASURED NO-WIN ON THE A40, CLOSED (2026-08-30)

The direct attack on 10.6c's verdict: stream each block's Phi slice through shared-memory
tiles (every table byte crosses DRAM once, coalesced), gather from shared, keep the walk order
per (hop, trial) untouched so `wave` is BIT-IDENTICAL. Built as `launch_waveform_tiled` +
`wavebench --tiled/--tiled16/--tsort/--tile N`, benched on cx52 (IDLE node A40 -- both idle
GPUs, no production contention). Correct at every step: wave bit-identical in all variants,
energy bit-identical unsorted / 2e-16 sorted.

**It loses at every geometry, both precisions** (vs the shipped fused/1024 gather):

    11 jobs x 140 chips   0.375 ms gather   0.583 ms tiled   0.64x
    32 jobs               1.053             1.692            0.62x
    100 jobs              3.069             5.126            0.60x
    100 jobs x 212 chips  4.455             6.376            0.70x
    fp16, 11 jobs         0.276             0.740            0.37x  (fp16 WIDENS the gather's lead)

The mechanism chain, each step measured (ncu, cx52), is the record worth keeping:

1. v1 (per-chip tile-boundary `break`): 1.14 ms, issue_active 21%, long-scoreboard 31/issue.
   A data-dependent break gates every load behind a compare; NOTHING pipelines. The shipped
   gather's fixed-trip chip loop is what lets the compiler hide its scattered DRAM latency --
   the "dumb" kernel is dumb like a fox.
2. v2 (interleave the 6 walks, predicated): 1.13 ms. ILP was not the binding constraint.
3. v3 (two-phase: fixed-count safe run + checked tail): 1.07 ms, issue 35%. Next: shared-pipe
   costs -- bank conflicts 23M (4 wavefronts/request on random-address LDS.64), and the
   per-chip code[] byte-gather (66M global sectors) through an L1 the 69 KB shmem carveout
   starved to ~28 KB. Tile 3072 vs 4096 is a CLIFF (0.64 vs 1.07 ms): the carveout steps
   64->100 KB there. Tile size is an L1-carveout knob, not a barrier-count knob.
4. v4 (+ hop-sorted lane mapping, 10.6b's idea -- viable here because the perm load amortizes
   into the prologue and the reordered loads hit shared): conflicts -> ZERO, shared wavefronts
   /3.4 -- and 0.64 -> 1.00 ms SLOWER, because the sorted lanes scatter the code[] gather
   across the whole table and thrash the starved L1.
5. v5 (+ code as a 25.6 KB bit table in shared, ballot-packed once per block; the walk then
   touches NO global memory): 0.583 ms, issue 42% (= the gather's 45%), all stalls low. The
   kernel is now HEALTHY -- and still 1.5x slower, because it executes 1.44x the instructions
   (122.6M vs 85.0M: staging, bit extracts, two-phase bookkeeping, barriers). That overhead is
   the design's floor, not a defect in it.

**And the DRAM prize evaporated as the kernel got faster**: v1 moved 98 MB (near the 81 MB
footprint), v5 moves 222 MB -- rising issue rate piles outstanding misses onto L2 until the
staged streams thrash each other. Low-DRAM and high-issue actively traded against each other
in this design. Meanwhile the gather at 100 jobs runs ~490 GB/s -- 70% of the A40's peak, so
its "wasted" re-reads fit inside headroom the part actually has at every deployed job count.

**The law this adds to 10.6c**: the gather kernel's scattered streaming at 2/3 of DRAM peak
with a fixed-trip loop is a local optimum that orchestration (tiles, barriers, carveouts,
sorted lanes) cannot beat on this part -- it can only spend instructions to move the
bottleneck somewhere less forgiving. Fewer BYTES RESIDENT (fp16, item 3) remains the one
lever that pays; fewer bytes MOVED does not, because moved bytes are not what binds.

KEPT: `launch_waveform_tiled` (no production callsites) + the four wavebench modes, as the
reproducible null result with its bit-exactness gates. If the nodes are ever refreshed to a
part where DRAM headroom is scarce relative to SM throughput, re-run `--tiled --tsort --tile
3072` before re-deriving any of this.

ALSO 2026-08-30: item 2's A40 verdict RE-VERIFIED ON AN IDLE NODE (cx52): 0.88x at 8 PRNs,
identical to the 08-28 cx43 number -- which was taken with production at ~85% GPU and could
have been contamination. It was not. The shared-Phi A40 verdict stands on a clean measurement.

## 6. CENTERED chip-window truncation  [BUILT + GATED 2026-08-31; arm = --despread-max-chips 80 --despread-chips-centered + node restart]

Item 9.5's cliff ("exact at 120, 13 chips out at 105") was the WINDOW GEOMETRY, not the
physics: the prototype is a windowed sinc peaking mid-span (chip ~105 of 210) and
set_max_chips cuts one-sided from chip 0 -- at 105 it cuts AT the peak (50.6% of filter
energy, -3.0 dB). The same 105 chips CENTERED keep 99.58% (-0.02 dB). Energy curve
(sinc*Hamming, centered): 80 chips -0.05 dB, 70 -0.12, 60 -0.27, 52 (the sinc main lobe,
one tap) -0.50, 40 -1.1 dB. Adjacent-channel leakage: -35 dB at 105, -28 at 80, -18 at 52.

MEASURED through the shipped e2e on cx52 (4f3050d05: ChannelizedReplicaBank::
set_chips_centered + e2e --chips-centered), scoring CODE PHASE (9.5's own rule -- at the
flip the error LOOKS smaller):
    2-node comb (harsh):    centered 105/90/80/70/60 all EXACT (VERDICT 17.588-17.615 vs
                            17.612 baseline); 52 FLIPS to the opposite grating lobe.
    8-node comb (deployed): centered 80/60/52/40 all EXACT (+0.371 chips = baseline).

So the degradation law: SNR falls as the kept filter energy (gentle), and failure is a
CLIFF -- alias capture, winner-take-all -- at ~1 tap in the harsh regime, below 40 chips
(0.76 taps) at the deployed comb. Recommended: **centered 80** (margin in both regimes)
= **1.75x on synthesis** vs the shipped one-sided 140; composes with fp16 -> ~2.4x total;
at 100 codes ~3.1 -> ~1.3 ms.

Also settles the "cheaper channel model" family: for any span the amplitude-optimal model
filter is the TRUE prototype restricted to the best (centered) window -- a designed 2-tap
PFB can only ever tie truncated-true above the cliff, and top-hat SPECTRAL bins are the
WRONG direction entirely (brick-wall response = longer time support; cost is linear in
time span).

OCCUPANCY MAPPING (2026-08-30, KV's reframe: nodes are arbitrary channel groupings --
think "X% of band missing", ~1% at full CHORD vs 25% today at 6/8). Truncation error is
COMMON-MODE across channels (one prototype serves all), so it cannot re-weight the
cross-channel delay beam; the only thing it can erode is the occupancy pattern's own
alias contrast. Measured (e2e, centered windows, score code phase):

    occupancy (s-stride)              worst alias      centered-window floor
    stride 4  = 75% missing periodic  ~unity @ 13.1 ch  cliff 52-60 (flip) -> use 80
    stride 2  = 50% missing periodic  ~unity @ 26.2 ch  >= 40 exact (no cliff found)
    stride 1  = full band             band-edge only    NO CLIFF DOWN TO 8 CHIPS --
                                                        sub-chip at 30/20/12/8; the floor
                                                        becomes the SNR budget, not aliasing

  Alias amplitude for arbitrary patterns: periodic KEPT comb stride s -> unity aliases at
  52.4/s chips (stride 16 = 3.27 = the tracker instance comb; 4 = 13.1; 2 = 26.2);
  m of 8 nodes MISSING -> m/(8-m) at multiples of 6.55 chips (2/8 -> 33%, robust);
  random p missing -> ~sqrt(N_miss)/N_kept (1% of 105 ch -> -40 dB, negligible).

  So at full CHORD the window is SNR-limited, not alias-limited: -0.27 dB @ 60 chips,
  -1.1 @ 40, -3.5 @ 20, ~-7 @ 8. Window 40-60 = 2.3-3.5x on synthesis vs the shipped 140,
  x fp16 -> ~3-5x total (100 codes: ~3.1 -> 0.6-0.9 ms). Degraded ops (whole nodes down)
  stay robust to >= 33%-amplitude aliases at window 40; only periodic sparse KEPT combs
  (today's stride-4 search, the per-instance stride-16 tracker comb) need >= 80. The
  tracker leg is not winner-take-all (locked DLL + fleet_dll combines instances), but
  needs its own check before arming small windows there.

ARMED-READY 2026-08-31 (config `despread_max_chips` + `despread_chips_centered`, default
off = the full-span walk BIT-FOR-BIT): DespreadJob/PeelJob gained d_first (appended last);
chip_gather AND chip_gather3 take a walk-start whose seeds all have closed forms in d0 and
reduce bit-exactly to the historical prologue at d0 == 0 (fmaf(0,kf,base) == base) -- one
change covers the waveform, fused-despread, peel and refine paths, which all share the two
gathers (only the tiled bench kernel keeps its own walk and does not honour d_first).
PhiCache/SharedPhi carry d_first from the bank (e2e/search arming) and the engine's own
set_chips_centered applies the same centered cap at set_max_chips level (production arming).

⚠️ THE "SHIPPED ONE-SIDED 140" NEVER SHIPPED: no config anywhere sets despread_max_chips
(verified against the generator, the generated yamls and the running node config, 08-31).
Production has always walked the full ~210 chips -- so centered 80 is **2.6x** on synthesis
against what actually runs, not 1.75x, and composes with fp16 (1.55x) to ~4x.

GATES (2026-08-31): e2e default VERDICT unchanged (19.496, the post-fix-3 baseline) with the
flag off; **e2e --max-chips 80 --chips-centered = 19.496 EXACTLY** -- and that run now
exercises the GPU TRACKER leg's own truncated walk (despread_batch -> chip_gather3 honours
d_first), closing part of 9.5's "e2e cannot isolate the tracker's truncation" caveat; the
pre-registered multi-PRN/multi-Doppler sweep (PRN 3/14/27 x dop -4200/+150/+3100, centered
80 vs untruncated baseline) agrees pairwise to <= 0.002 chips in all 9 cells; n2dualtest
7/7; phi16gpu ALL PASS on top of it. Arm = --despread-max-chips 80
--despread-chips-centered on the yaml regeneration + node restart; the tracker state warns
below the centered floor of 60.

## 4. Fuse synthesis + pack

`wave` is [3*n_job][n_chan][n_hops] float2 = **11 MB per record** at 32 jobs x 7 chan x 2048
hops, written by the synthesis kernel and read straight back by pack44 to be quantized to a
1.84 MB 4-bit array: ~22 MB/record of DRAM traffic purely to hand data between two kernels
(2.1 GB/s per chain-instance).

Tractable because the quantizer scale is ALREADY frame-frozen: `launch_pack44` takes
`d_energy0` = record 0's energy, snapshotted as the frame reference. So for records 1-3 the
scale is known before synthesis runs and a fused kernel can emit 4-bit directly. Only
record 0 needs the current two-pass (or the previous frame's scale -- that changes bits and
would need its own A/B).

---

## Record-format cleanups (small bytes, real clarity) -- from the 2026-08-28 audit

- **UTC (slots 9-10) is duplicated across every PRN row.** `const double utc` is computed
  once per record in GnssGpuRecordAssemble and written into all 32 rows: 1024 B/frame/chain
  of pure replication. Its two real consumers (the #46 ingestion clock, nav-bit timing) are
  per-RECORD. It belongs in the frame header. The combiner's own comment already says the
  alignment key is the hop, not the UTC ("equal hop IS the same sky with no tolerance").
- **REC_DOPPLER (slot 1) has ZERO consumers**; REC_CP (slot 2) has exactly one, in
  cudaGnssTrack.cpp -- the old path-A tracker, not the shipped inject path. Both are echoes
  of what the broker seeded. They are not free-standing (the slot-2 currency contract pairs
  cp with the same record's carrier), so removing them means moving the currency tag to the
  header, not just deleting fields.
- **The combiner reads the prompt by RAW INDEX** -- `rec[3]`, `rec[4]`, `rec[5]`, `rec[6]` --
  while using named REC_* constants for E/L/PH/SKY/RES two lines below. That is why
  REC_P_RE and REC_NCHAN audit as zero-consumer. Normalize before anyone reorders a slot.
