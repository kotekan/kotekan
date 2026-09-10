# GNSS GPU pipeline — closed record

Opened 2026-08-28 as an optimization queue. **Closed 2026-09-10: GPU cost is not the binding
constraint, and the two items that mattered shipped.** This is the record of what was measured,
so nobody re-derives it. Open GPU work lives in `CHORD_BUGLIST.md` (#112, #113, #114).

---

## The headline number

**Synthesis costs 10.20 ms (gpu0) / 10.80 ms (gpu1) per 41.94 ms frame = 24.3% / 25.8% of each
A40.** Derived from `T_block ≈ 0.15 ms` at 80 chips / fp16 with `waves = ceil(n_prn·n_chan / 84)`
on the 84-SM A40 (measured 2026-09-08 on cx52), evaluated against the live per-instance PRN and
channel counts read off the running nodes on 2026-09-10 (68 waves/frame on gpu0, 72 on gpu1).

Nothing is backpressured: input and voltage buffers 1/24, every n2rec buffer 1/4 (the 2026-09-08
fault read 22–24/24 and 4/4). The one-CPU-core limiter is also gone — the 15 combiners sit on 15
distinct cores at 12–21% each.

**Had items 3 and 6 not been armed**, the same shape at 212 chips / fp32 would cost 25.16 /
26.64 ms = **60.0% / 63.5%** of each GPU. The two arms bought exactly the headroom the band
expansion then consumed: instances went 10 → 15 when B3I, E6 and L2C came on. That is the whole
value of this queue, and it is already banked.

**Two measurement traps for whoever re-runs this.** `nvidia-smi` reads 57–92%, but a synthesis
launch is 84 blocks of 1024 threads — exactly one SM wave — so it reads ~100% busy for its whole
0.15 ms: that metric is time-occupancy, not allocation. And `/gpu_profile` sums to 736% of one
GPU-equivalent across two GPUs, which is arithmetically impossible; it reports ~14 ms for every
gnss0 instance and ~7 ms for every gnss1 instance regardless of shape — the signature of an
elapsed-time measurement dominated by queue waiting.

---

## What shipped and is armed

| item | status | evidence |
|---|---|---|
| **1a** stop gathering the BB block | **armed**, unflagged, 2026-08-28 | tile buffers on the running nodes are the post-1a sizes |
| **3** fp16 Φ tables (#22) | **armed** by `477c06eca`, live since 09-05 and still live after the 09-10 cycle | `phi_fp16: true` ×15 per node; `fp16 Phi ARMED` ×15 in the node log, zero refusals |
| **6** CENTERED-80 chip-window truncation | **armed** by the same commit | `despread_max_chips: 80` + `despread_chips_centered: true` ×15 on all six nodes |
| **7** per-kernel breakdown | done | `scripts/gnss/gpu_breakdown.sh` — it already existed |

⚠️ **This file said "default OFF" for items 3 and 6 for ten days after they were armed.** The doc
text landed at 02:07 on 08-31 and the arming config at 11:02 the same morning, and the text was
never revised. Anyone who planned around "arm = one yaml line + a node restart" was planning a
no-op. That is the failure mode this record exists to prevent.

## What was measured and rejected — do not re-try without new hardware

| item | verdict | number |
|---|---|---|
| **1b** disable the BB block class | no win | 1.00–1.01×. The `block_class_mask` plumbing it asked for was built anyway (09-07); BB still is not dropped because dropping it buys nothing |
| **2** two Doppler-free Φ tables per channel | **slower** | 0.88× at 8 PRNs, 0.92× at 24, 0.89× at 32 on the A40 (08-28, re-verified on idle cx52 08-30). Keep the code: it frees 1.8–2.4 GB/node and is a ~1.2× win on L40S-class parts |
| **5** tiled shared-memory streaming | **slower** | 0.60–0.70× at every geometry. Kept as bench-only; it is the reproducible null result. Its law is the useful part: **bytes moved are not what binds on this part, bytes resident are** |
| **8** one (N+M)² correlator per GPU | **worse at today's geometry** | replacement 6.141 ms vs coexistence 2.512 ms (measured 08-08) — 2.4× worse. The cause is a kernel property, not a config choice: `freq_map` and `block_class_mask` are global to a launch, so a merged pass must compute AA+MIXED+BB on all 384 local frequencies, 377 of them with a zero M half |

## Still open, and both keyed to element count rather than PRN count

- **Item 8 at full CHORD.** At 1024 elements AA is 2080 of 2628 tiles, so running both passes
  doubles the correlator's voltage-ring traffic and the merge becomes worth doing. The single
  change that unblocks it: make `freq_map` and `block_class_mask` **per-block-class rather than
  per-launch** in `n2k_dual`. Not actionable until the element count grows.
- **The hop-split grid.** Per-GPU SM-time scales with channels-per-GPU while wall-time per
  instance does not — the lever if channel count grows before element count does.
- **Item 4, fuse synthesis + pack** — never started, and demoted: it saves ~22 MB/record of DRAM
  traffic, and item 5's measured law says traffic is not what binds here. Do not spend the cycle.

## Path B — the dual-input N² correlator, as built

Path B is the production tracker on all six nodes (`cudaGnssInject` ×15, `cudaGnssTrack` ×0;
`--no-path-a`, so path A is not instantiated: 17.712 → 2.512 ms/frame).

| milestone | status |
|---|---|
| M1 clone regression | done — `dual(128+0)` bitwise vs `n2k(128)`, and the AA prefix of `dual(128+128)` bitwise vs a pure-128 run |
| M2 dual correctness | done — vs a CPU reference at (256), including a random RFI mask, plus class-mask and freq-map cases added later |
| M3 offline path-A cross-validation | done — orientation confirmed, quantization loss **+0.04 dB / 2.3 mrad** against a predicted <~0.2 dB |
| M4 node deployment + A/B | done and superseded — it is production on all six nodes, and the live A/B closed. ⚠️ Two clauses ("SK stats identical with injection on/off", "N² prefix bit-identical") were never *measured*: under coexistence they hold **by construction**, because freq-map mode never computes AA. That is a sound argument, not a measurement — and it is exactly the measurement a real merge would need |
| M5 chain closure | done except the survey — records ship and are validated on sky. **Not done:** the frame-cadence (8192-hop) `hops_per_record` survey; production kept 2048-hop records, 4 per frame. 174 references across 39 files is its scope |

**A partial delivery of item 8 is already live**: `gnss_gather_aa` (09-07) added the AA block class
to the dual correlator's mask and gathers the live antennas' own N² tiles, fleet-wide on the
gps_l5 instance of each GPU. The merge's core mechanism — the antenna N² coming off the dual
pass — is running in production; only its consumer is the capture sink rather than the science
correlation buffer.

## One geometry the gates never covered

L2C runs `n_chan = 1` with CENTERED-80 armed on five nodes. The item-6 gate matrix used 2-node
and 8-node combs at 6–7 channels; L2C is also the chain with the one-channel Dirichlet pathology
that forced DLL spacing 2.0 → 0.4. Empirically it is the tightest readback of all eight chains
(±0.037 chips), so this is fine — but the gate does not vouch for it, and it is the one place
CENTERED-80 sits at its recommended floor with no margin.
