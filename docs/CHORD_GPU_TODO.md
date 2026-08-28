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

## 1. M² from pack44 -> disable the BB block class  [IN PROGRESS]

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

## 2. ONE Doppler-free Phi table per channel, shared by every PRN

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
