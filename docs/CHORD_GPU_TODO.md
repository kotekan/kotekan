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

## 2. TWO Doppler-free tables per channel, shared by every PRN  [CPU PATH DONE; GPU NEXT]

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

## 3. fp16 Phi tables (task #22)

Already measured 1.27-1.37x, and §10.6c explains WHY it is the right lever (halves the
resident table). Composes with item 2: 7.3 MB -> 3.7 MB.

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
