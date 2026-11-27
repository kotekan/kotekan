# The N^2 lane projection on the GPU (design, phase 1)

Status 2026-11-27: built and tested on `kv/gpu-projection` (off kv/chord-gnss `b1d66d225`),
not deployed; section 7 has what was built, what departs from the design below, the test results
and the canary steps. Sections 1-6 are the design of 2026-11-26. The stage being moved is `GnssN2Project`
(lib/stages/gnss/GnssN2Project.{hpp,cpp}); what a lane is and how lanes travel between nodes is
in [CHORD_FLEET_LANES.md](CHORD_FLEET_LANES.md); what the projection does to the science data is
in docs/chord/PROJECTION.md (branch kv/docs-1125).

**Summary.** The projection products, their write-back and the 34 MB frame copy (72% of the
stage's time) move into the science correlator's own GPU pipeline (`run_n2k`), in place on the
correlation frame, right after `cudaCorrelator`. The host keeps everything that decides the
lanes (board, carry, re-steer, #229 ranking, Gram-Schmidt) and the diagnostics. The lane basis
built from frame t is applied from frame t+1 on (the newest basis the GPU holds; the correlator
never waits for it), and the GPU writes the basis record of what it applied, so N2Accumulate's
mean projector stays exact and N2Accumulate no longer depends on the CPU stage at all. Measured:
the GPU work is 0.74 ms per frame in FP64 for 43 channels and 40 lanes on an L40S, bit-identical
to the CPU path; the stage's host time falls from 24.4 to about 6.8 ms per frame single-threaded.
A generator flag per GPU selects the GPU or today's CPU path.

## 1. Today's data path

Per GPU (the two GPUs of a node are symmetric; GPU 0's names below, GPU 1's carry `_1` or
`gnss1`). Shapes from the generated cx43 config of the 11-27 cycle
(`~/gnss/builds/cycle-20261127/cfg/chord_gnss_cx43_multi.yaml`).

```
 voltage ring (GPU)                       stock run_n2k (cudaProcess, stream base 0, cpus 12-14)
   |                                      +---------------------------------------------------+
   +--> cudaCorrelator (n2k, kernel stream 2) --> correlation_buffer[gpu_frame] (device, x4)  |
   |    cudaSyncOutput                                                                        |
   |    cudaOutputData (copy-out stream 1) ----------------------------------+                |
   |                                      +-----------------------------------|---------------+
   |                                                                          v
   |                                  host_correlation_buffer_0  (34.2 MB, 36 frames)
   |                                                                          |
   |                       GnssN2Project proj_0 (cpus 15+47, num_workers 2)   |  ProjBoard (lanes
   |                       copies all 464 channels, projects the 43 live <----+  from the chains),
   |                       ones, writes:                                          set_lanes (relay)
   |                         gnss_n2_proj_buffer (34.2 MB, 36)  gnss0_proj_basis (1.50 MB, 36)
   |                         gnss0_lanedump_raw (6.33 MB every 2 s -> gate -> /tmp/gnss/lanedump)
   |                                      |                        |
   |                                      v                        v
   |                       N2Accumulate accum_0 (cpu 9): in_buf, in_proj_basis_buf (1:1, seq
   |                       checked), counts, RFI counts, PL counts, frame mask, bad feeds, gains
   |                                      |                        |
   |                                      v                        v
   |                         n2_buffer_0 (per freq, ~21 s bins)    gnss0_proj_mean (2.06 MB/bin)
   |                           -> N2Subset -> bufferSend (recv1)     -> rawFileWrite /tmp/gnss/projmean
```

| buffer | frame | depth | rate per GPU |
|---|---|---|---|
| `host_correlation_buffer_0` | int32 [Tc 1][F 464][36 blocks][16][16][2] = 34,209,792 B | 36 (`host_side_buffer_depth`) | 23.84 frames/s, 816 MB/s |
| `gnss_n2_proj_buffer` | same | 36 | 816 MB/s |
| `gnss0_proj_basis` | 64 + 4n + 43 (16 + 8 k_slot n) = 1,501,028 B (n 109, k_slot 40) | 36 | 36 MB/s |
| `gnss0_proj_mean` | 64 + 4n + 43 (16 + 4n(n+1)) = 2,063,468 B | 4 | one per 21.04 s bin |
| `gnss0_lanedump_raw` | 64 + 43 (64 + 4n(n+1) + 96 (160 + 8n)) = 6,325,192 B | 4 | one per 2 s |

The frame is one 8192-sample integration (41.94 ms; `samples_per_data_set` =
`sub_integration_ntime`). The n2k layout holds, per channel, the 36 lower-triangle 16x16 blocks
of the 128 inputs; entry (i, j) is data where j <= i, and the other half of a diagonal block is
never written (gnssN2ProjectMath.hpp `Layout`). Nothing but GnssN2Project reads
`host_correlation_buffer_0`, and nothing but N2Accumulate reads `gnss_n2_proj_buffer`.

Per frame the stage copies the whole frame to its output, then for each live channel (43 of
464 on this comb phase; 109 stations, 19 other inputs): extracts the station block, steps the
#229 subspace iteration (`rank_step`), builds the lane basis (board entries, instrumental carry,
LEO re-steer, rank keys, fleet lanes, the Rayleigh-quotient diagnostic every 4th frame,
Gram-Schmidt up to `k_lanes_max` 40), records the dump slot (every 2 s), projects the station
block, writes the basis slot, updates `null_db` (the projected block's EMA every frame, a solve
every 4th, on the 31 channels of the eigen list), writes the block back and projects the
station-other products on the station side.

Production at 2026-11-26 23:05Z (GET /metrics): `kotekan_gnss_n2project_frame_ms` 26.0-28.2 ms
on cx27, cx42, cx51, cx52 (2 workers on a hyperthread pair); 26-29 lanes per live channel on
average (median 31-34, maximum 40), about 2 duplicates per channel per frame, 2-4 fleet lanes in
the basis, the #229 ranking active on 9-12 channels, `null_db` on 31-33.

## 2. Measured CPU profile

Method: the stage at `b1d66d225` built Release with `-march=native` and per-phase timers (a
scratch patch, not committed: `build-prof/timers.patch` in the worktree), run on cf05 (Xeon Gold
5416S, the nodes' CPU) on a production-shaped bench: cx43 gnss0's own GnssN2Project block (its
stations, tracker elements, live and eigen lists, `k_lanes_max` 40, `lane_rank_min_rel` 10,
`lane_resteer`, `fleet_lanes`, `lane_inst_phase`, the lane dump and the basis record) on 464
channels of one comb phase, so 43 are live and 31 on the eigen list; testN2kGen input;
GnssProjBoardTestPub publishing 29 directions per live channel; 28 fleet lanes and an
instrumental phase table posted to the bench's own `set_lanes` every 2 s; 32 lanes per channel
in the basis; 960 frames back to back. Bench scripts: `build-prof/{mkbench_prod.py,
post_fleet.py,run_prof.sh}`.

ms per frame (thread time summed over workers; wall time on the first line):

| phase | 1 worker | 2 workers, HT pair (production) | 2 workers, 2 cores | 1 worker, 40 lanes |
|---|---|---|---|---|
| **wall** | **24.40** | **21.37** | **12.00** | **27.76** |
| frame copy (464 ch, 34 MB) | 4.22 | 7.61 | 4.26 | 4.15 |
| extract + mean auto | 0.60 | 1.16 | 0.59 | 0.58 |
| rank_step (#229 modes) | 0.77 | 1.36 | 0.78 | 0.76 |
| board collect | 0.41 | 0.70 | 0.46 | 0.52 |
| carry + re-steer + rank keys + fleet append | 0.18 | 0.29 | 0.17 | 0.21 |
| Rayleigh-quotient diagnostic | 0.20 | 0.34 | 0.20 | 0.20 |
| Gram-Schmidt | 2.76 | 4.88 | 2.67 | 4.15 |
| **project (station block)** | **9.29** | **15.83** | **9.09** | **10.76** |
| basis slot | 0.17 | 0.17 | 0.16 | 0.20 |
| null_db (push 0.59, solve 0.54) | 1.12 | 1.61 | 1.00 | 0.94 |
| **write-back** | **1.00** | **1.69** | **0.99** | **0.97** |
| **cross products (station side)** | **3.11** | **6.00** | **3.04** | **3.70** |
| frame head/tail, dump, auto | 0.15 | 0.17 | 0.13 | 0.15 |

- The bench runs about 20% faster than the nodes (24.4 ms single-threaded against 30-37 ms in
  production on 11-25; 21.4 ms on the HT pair against 26-28 ms): the nodes share memory
  bandwidth and caches with the rest of the pipeline.
- What moves to the GPU (bold, plus the copy): 17.6 of 24.4 ms single-threaded (72%), 31.1 of
  41.7 thread-ms on the HT pair.
- What stays: about 6.4 ms single-threaded, plus about 0.4 ms to extract the projected blocks
  for `null_db` (section 3.2): **~6.8 ms**, 3.5x less. Gram-Schmidt (2.8 ms, 4.2 ms at 40
  lanes) is the largest part, then `null_db` and `rank_step`.
- With every board lane a fast mover (`los_rate` 5e-3 rad/s), the re-steer costs 2.44 ms per
  frame (1247 lanes x 128 phasors); on sky only the GuoWang lanes are, a fraction of that.
- The HT pair gains little over one thread (24.4 -> 21.4 ms); two real cores give 12.0 ms. The
  projection is the part that scales worst on a hyperthread pair.

## 3. Proposed architecture

### 3.1 What moves, what stays

**GPU:** V <- (I - Q Q^H) V (I - Q Q^H) on every live channel, the station-other products
included, in place on the correlation frame; the gather of the raw and projected live blocks
for the host; the basis record of what was applied. The 34 MB frame copy disappears: N2Accumulate
reads `host_correlation_buffer_0` directly.

**Host (GnssN2Project):** the board read, instrumental carry, LEO re-steer, fleet lanes, #229
ranking (`rank_step` on the raw block, rank keys), Gram-Schmidt, the Rayleigh-quotient
diagnostic, the lane dump, `null_db` and every gauge, `get_lanes`/`set_lanes`. Gram-Schmidt is
sequential per channel with a data-dependent accept/reject per candidate, 2.8 ms per frame; it
is not worth a kernel. `rank_step` (0.8 ms) can follow later as one more batched GEMM on the
dense blocks the GPU already holds (the iterate goes up with the basis, V U comes down with the
live blocks); not in this change.

**Embedding.** With Q~ the basis on the 128-input axis, zero on the inputs outside the stations,
I - Q~ Q~^H is I - Q Q^H on the stations and the identity elsewhere, so one projection of the
whole 128 x 128 matrix gives the projected station block, (I - Q Q^H) X for the station-other
products and leaves the other inputs' own products unchanged: exactly what `project`,
`project_cross` and the write-backs do today, with no station extraction and no separate cross
pass. The micro-benchmark (section 3.6) checks this word for word against the stage's own CPU
functions.

### 3.2 Where it runs in the GPU pipeline

A new command `cudaGnssN2Project` in `run_n2k`, between `cudaCorrelator` and `cudaSyncOutput`,
on the pipeline's kernel stream (2), as is the correlator:

```
run_n2k gpu_0 commands:
  cudaCorrelator          correlation_buffer[f]                               (stock, unchanged)
  cudaGnssN2Project       1. upload the newest basis record (H2D, 1.5 MB, from pinned staging)
                          2. gather the raw live blocks       -> gnss_n2live_buffer[f] (raw half)
                          3. project in place                 on correlation_buffer[f]
                          4. gather the projected live blocks -> gnss_n2live_buffer[f] (2nd half)
                          5. the applied record, seq patched  -> gnss_n2proj_basis_buffer[f]
  cudaSyncOutput                                                               (stock, unchanged)
  (live blocks)           gnss_n2live_buffer       -> gnss0_n2live              (GnssN2Project)
  cudaOutputData          gnss_n2proj_basis_buffer -> gnss0_proj_basis          (N2Accumulate)
  cudaOutputData          correlation_buffer       -> host_correlation_buffer_0 (now projected)
```

It has to be this stage: the correlation frame is a per-frame device array that `run_n2k` owns,
named GPU memory is refused to a second stage, and only GPU ring buffers cross stages, so a
separate projection stage would need `cudaCorrelator` (stock code) to write a ring. No new
streams: the H2D upload is on the command's own kernel stream, ahead of its kernels, and the
copy-outs ride the existing copy-out stream after `cudaSyncOutput`. The command registers its
device regions (`register_gpu_memory_name`) and creates one cuBLAS handle on its stream, as
`cudaFRBBeamReformer` does; cuBLAS is already linked.

**The raw V for the host.** The raw live blocks are gathered before the projection and the
projected ones after it, into one frame `gnss0_n2live`: a 64-byte header (seq, the applied
record's source seq, slot count) and [2][43][36][16][16][2] int32, raw then projected, each half
in the n2k layout with F = the live channels; 6.34 MB per frame. The frame carries the
correlator's metadata with `coarse_freq` set to the live channels' freq_ids. GnssN2Project in GPU
mode then runs its per-channel code on this frame as if the live channels were all its channels:
`extract` on the raw half for `mean_auto`, `rank_step`, the Rayleigh-quotient diagnostic and the
dump; on the projected half for `auto_proj` and `null_db`.

### 3.3 How Q reaches the GPU in time: one frame ahead, never waited for

GnssN2Project, having processed live frame t, writes the basis it built into its `basis_buf`
(the existing record, version 1, with seq = frame t's: the source), now consumed by
`cudaGnssN2Project` instead of N2Accumulate. The command, in `wait_on_precondition` (after
`cudaCorrelator`'s wait for the voltage, so as late as possible), takes every full frame of that
buffer without waiting, keeps the newest, releases the rest, and copies it into the frame's
pinned staging; `execute` enqueues the upload. Before the first record it applies k = 0 on every
slot and records that, with its configured stations (the generator renders the stage's
`stations` and `live_freq_ids` into both). A record whose station list or slot freq_ids disagree
with the command's own is fatal.

Timeline at steady state: frame t's correlation, projection and live-block copy-out end ~2-3 ms
after its voltage is complete (the live blocks go out before the 34 MB correlation frame); the
host builds the next basis in ~6-7 ms (section 3.6); frame t+1's preconditions come ~42 ms after
frame t's. So the basis built from frame t is applied to frame t+1, with ~30 ms to spare. A late host makes the lag 2 for that frame; the
correlator never stalls on it. The lag is recorded per frame (3.4) and exported by the stage as
a gauge (`kotekan_gnss_n2project_basis_lag_frames`).

Rejected: a blocking 1:1 pairing (the command waits for the basis of frame t-1). It makes the
lag exactly 1 but couples the science correlator to the lane builder's tail latency with one
frame of slack (on 11-25 five projecting nodes restarted on "stream is out of step" while the
single-threaded stage took 86-93% of the frame, which is why #225 threaded it). Also rejected:
the same-frame round trip (correlate, copy the raw live blocks out, build, upload, project, copy
out), which makes the GPU frame wait on the host mid-frame; and keeping the stage on the data
path with the GPU as an accelerator for its products (upload the live blocks and Q, project,
download), which keeps the 34 MB copy and the stage between the correlator and N2Accumulate.

What one frame of lag changes:
- **#229 ranking**: `rank_step` runs on frame t's raw block and ranks the basis applied to t+1,
  i.e. the previous frame's modes. The modes are tracked by one subspace-iteration step per frame
  already, so they lag by more than that.
- **lane_resteer**: build_lanes turns fast movers to the mid-time of frame t+1 (t_frame + dt +
  dt_int/2) instead of frame t; one more frame of lag (a late host) is 42 ms of GuoWang motion,
  ~2e-4 rad, against the ~0.3 deg a lane is used over today (`lane_max_age_s`).
- **fleet lanes**: none; they are re-evaluated every 0.25 s along their tracks.
- **lane dump**: unchanged layout; a dump records frame t's raw block and the candidates and
  verdicts of the basis built there, which the GPU applies from the next frame (the basis
  record says exactly which frames).
- **null_db, auto_proj_db**: from the GPU's projected block of frame t, i.e. the basis actually
  applied (built at t-1). The reference eigenvalue (first lane's Rayleigh quotient) is frame t's.
- **lane_k, fleet_lane_k, lane_held_k, lane_ranked**: describe the basis built at t.

### 3.4 The mean-projector record still equals what was applied

The GPU writes `gnss0_proj_basis` itself: in step 5 it copies the device-resident record the
kernels just used, with seq set to the frame's own (from the correlator's metadata) and the
source frame's seq in a spare header field (bytes 40-47; N2Accumulate reads only bytes 0-35 of
the header). `cudaOutputData` copies it out 1:1 with the correlation frame, from the same
pipeline and the same frame slot, and N2Accumulate keeps checking basis seq = frame seq. So the
record is the applied bytes, by construction, with no host bookkeeping, and N2Accumulate no
longer waits on GnssN2Project at all.

**Q precision (open decision 1).** The record holds Q as complex<float>; today the CPU applies
the double Q that Gram-Schmidt produced, so record and applied differ by float rounding (~1e-7).
The GPU applies the record's own values (promoted to double), so the record is exact there. I
propose the CPU path do the same (round the basis to complex<float> before `project`): then the
CPU and GPU paths are bit-identical given the same basis (measured, 3.6), and the record is exact
in both. Rounding Q to float moves it by ~6e-8 relative, so today's CPU output should move by at
most a count, on the largest values only (not measured on sky).

### 3.5 CPU fallback switch

Generator `--n2-project-gpu GPUS` (fleet yaml `n2-project-gpu: "0,1"`, per-node override in the
manifest; default empty = today's wiring for every GPU). For a GPU in the list it renders:
the command and the two copy-outs in `run_n2k.gpu_<g>`; `n2_accumulate.accum_<g>.in_buf` =
`host_correlation_buffer_<g>` and `in_proj_basis_buf` = the GPU's record; GnssN2Project with
`gpu: true`, `in_buf: gnss<g>_n2live`, `basis_buf` to the command, no `out_buf`, and no
`gnss_n2_proj_buffer`; the stage's `num_local_freq` is that GPU's live channel count, which the
generator already derives from the comb phase to size the dump and the record. The stage refuses `gpu: true` with an `out_buf`, without `basis_buf`,
with source eigen or lanes+residual (the eigen path stays CPU-only; production runs lanes), or
with more than one integration per frame. The stage sees only the live channels, so `null_db`
covers the eigen-list channels among them (all of production's eigen list is live). Off,
the stage is today's code path. Switching is a regeneration and a node cycle; the per-GPU form
gives a same-node, same-time A/B (GPU 0 on the GPU, GPU 1 on the CPU).

### 3.6 GPU cost

**Kernels** (variant A below): a gather kernel builds, per live channel, the dense Hermitian V
(128 x 128, complex double) from the int32 blocks and Q~ (128 x k_slot) from the record; cuBLAS
`cublasZgemmStridedBatched` forms W = V Q~, M = Q~^H W and Z = W - Q~ M / 2 (three calls over all
channels; k padded to `k_slot` 40); one fused kernel per (16x16 block, channel) applies
V' = V - Q~ Z^H - Z Q~^H to the lower-triangle entries only, reading and writing the int32 frame
in place with the stage's rounding (`nearbyint`, saturating, diagonal imaginary 0), never
touching the upper half of a diagonal block. Z is the form of the stage's
V - Q W^H - W Q^H + Q M Q^H that makes the update one Hermitian rank-2k term. A dense
four-GEMM update (variant C) is slower; cuBLAS has no batched ZHER2K.

**Micro-benchmark** (`build-gpubench/projbench.cu` in the worktree; cf05 GPU 1, L40S, idle;
kernels only, 95 repetitions; checked on one channel, all 16,512 words, against GnssN2Project's
own `extract`/`project`/`writeback`/`project_cross`/`writeback_cross` with the same float Q):

| live channels | k | A: FP64 cuBLAS + fused update | B: FP32 throughout | C: FP64 dense, 4 ZGEMM |
|---|---|---|---|---|
| 43 | 27 | 0.564 ms | 0.062 ms | 0.724 ms |
| 43 | 40 | **0.737 ms** | 0.077 ms | 0.967 ms |
| 43 | 64 | 1.229 ms | 0.089 ms | 1.449 ms |
| 100 | 27 | 1.043 ms | 0.109 ms | 1.424 ms |
| 100 | 40 | 1.845 ms | 0.144 ms | 2.397 ms |
| 100 | 64 | 2.482 ms | 0.174 ms | 3.013 ms |
| words differing from the CPU path | | 0 | 2-4%, by 1 count | 0 |

FP64 runs at 1/64 of the FP32 rate on the L40S but the problem is small: FP64 is 10x slower here,
not 64x. **Proposed: FP64 (variant A)**, bit-identical to the CPU path, so the A/B and the CI
check are exact. FP32 is the lever if the projected range grows by ~10x (open decision 2).

**PCIe** (pinned, same benchmark): the basis record up, 1.50 MB, 0.064 ms; the live blocks down
(raw and projected), 6.34 MB, 0.245 ms; the applied record down, 1.50 MB, ~0.06 ms. 100
channels: 3.5 MB up 0.14 ms, 14.8 MB down 0.57 ms. Together +9.3 MB per frame, +223 MB/s per
GPU, against the 404 MB per frame voltage copy (CHORD_GPU_BUDGET.md): +2.3% of the PCIe traffic.
The correlation frame's 34 MB copy-out is unchanged.

**Device memory** per GPU (43 channels, k_slot 40): dense V 11 MB, Q~ and W/Z 7 MB, the live
block frames 4 x 6.34 MB, the record in and out 4 x 3 MB, cuBLAS workspace a few MB: under
100 MB of the 46 GB. **Host memory** per GPU: -1.23 GB (`gnss_n2_proj_buffer`, 36 x 34 MB,
goes), +228 MB (`gnss0_n2live`, 36 frames).

**GPU budget.** +0.74 ms of kernels per 41.94 ms frame measured idle (+1.8%); in situ beside n2k
and the input copy expect up to ~1.5 ms (the budget doc saw 10-15% of GNSS launches slow down
4x there). +0.3 ms on the copy-out stream. The budget doc's numbers are from A40s (2026-10-13):
kernels then busy 4-10 ms per frame. The correlation frame reaches the host ~0.7 ms later than
today, ~25 ms earlier than through today's CPU stage.

**Host after the move** (bench): ~6.8 ms per frame on one thread instead of 24.4; on the HT
pair ~11 thread-ms over two workers, ~5.6 ms (about 7 ms on a node, at the bench-to-node ratio
of section 2, before the bandwidth the 34 MB copy no longer takes). 100 live channels at 40
lanes: ~16 ms on one thread, 1.8 ms on the GPU.

## 4. Change list, tests, rollout, risks

### 4.1 Per file

| file | change |
|---|---|
| lib/cuda/cudaGnssN2ProjectKernel.{cu,hpp} (new) | the gather (dense V, Q~), the fused in-place update, the live-block gathers, the record copy with seq; host wrappers taking a stream |
| lib/cuda/cudaGnssN2Project.{cpp,hpp} (new) | the command: config (`stations`, `live_freq_ids`, `in_basis_buf`, the GPU memory names), shared state (newest record, slot table from the first frame's `coarse_freq`), non-blocking take of the record, pinned staging, cuBLAS handle, enqueue, metadata on the live-block frame, a dropped-frame counter if open decision 3 goes that way |
| lib/cuda/cudaGnssN2ProjectTest.cpp (new) | standalone check like `cuda_gnss_beam_dump_test`: the kernels against gnssN2ProjectMath.hpp, word for word (from `projbench.cu`) |
| lib/cuda/CMakeLists.txt | the sources and the test executable |
| lib/stages/gnss/GnssN2Project.{hpp,cpp} | `gpu` mode: input is the live-block frame (raw half for the lane work, projected half for `auto_proj`/`null_db`), no output copy, no projection; the basis record's seq is the source frame's; re-steer to the next frame's time; the lag gauge; the CPU path rounds the basis to complex<float> (open decision 1); config checks; doxygen |
| lib/stages/gnss/gnssN2ProjectMath.hpp | the live-block frame header and the record header field offsets, shared by the stage, the command and the test |
| lib/stages/N2Accumulate.hpp | doc only: the record may come from cudaGnssN2Project; bytes 40-47 hold the source seq |
| config/gen_chord_gnss_config.py, config/test_gen_chord_gnss_config.py | `--n2-project-gpu GPUS`, the wiring of 3.5, buffer sizes from the live channel count; tests for both wirings |
| config/gnss_fleet_chord.yaml | `n2-project-gpu` (empty), per-node override at the canary |
| scripts/gnss/stock_parity.py | declare `run_n2k.gpu_*.commands` (+ the command and its copy-outs) and the changed `n2_accumulate.accum_*.in_buf` |
| config/ci-tests/gpu_batch/run_n2accum_gnssproject_gpu.yaml (new), a checker in scripts/gnss | see 4.2 |
| docs/CHORD_FLEET_LANES.md, this doc | the GPU mode, the lag, the record's source seq |

No change to cudaCorrelator, cudaOutputData, cudaSyncOutput or N2Accumulate's code.
tools/ci_select_jobs.py already sends lib/cuda changes to the GPU jobs.

### 4.2 Tests

1. **Kernel check** (`cuda_gnss_n2_project_test`, needs a GPU): random Hermitian frames in the
   n2k layout with poison in the upper halves of the diagonal blocks, a random station subset
   with one station whose rows are all zero, k = 0, 1, 27, 40, random float bases: every word of
   every live channel equals the stage's CPU functions; non-live channels, the other inputs' own
   products and the upper halves untouched. The benchmark already shows 0 differing words.
2. **Boost** (CPU): the record and live-block header helpers; the stage's slot mapping on a
   live-block frame; with the basis rounded to float, `project` output unchanged by repeating it
   (the rounding is applied once).
3. **GPU CI config** `run_n2accum_gnssproject_gpu.yaml`: testN2kGen into `correlation_buffer`
   via cudaInputData, `cudaGnssN2Project`, GnssN2Project in `gpu` mode fed by
   GnssProjBoardTestPub, N2Accumulate with the record, rawFileWrite of the live-block frames and
   the records. A checker re-projects every frame's raw live blocks with the record that says
   what was applied (numpy, double) and compares with the projected half: |d| <= 1 count (numpy
   sums in another order; the exact comparison is test 1), the records' seq equal the frames',
   the lag 1 or more.
4. **Recorded sky**: the lane dumps of 11-25 (G03's pass, `fixtures/deproj_1125`) hold raw
   station blocks and every candidate with its verdict; rebuild each basis by Gram-Schmidt and
   run the kernel against `project` on them: exact on real bright-pass data.
5. **The existing CPU configs** (`cpu_batch/run_n2accum_gnssproject_{live,lanes,fleet,projmean}`)
   keep passing in CPU mode; with open decision 1 their projected outputs change by a count here
   and there, so their references (if any are pinned) are regenerated once.

### 4.3 Rollout

1. Build the fleet binary (CUDA 13.5, as the nodes), all defaults off; run tests 1-5 on cf05.
2. At a node cycle, one GPU of a canary node: `n2-project-gpu: "0"` for that node. Judge in
   minutes: no FATAL (record or slot mismatch), `NRestarts` 0; the lag gauge 1;
   `kotekan_N2accumulate_proj_mean_dropped_total` flat; GnssN2Project `frame_ms` ~6-8 ms; GPU 0's
   `null_db`, `auto_proj_db` and `lane_k` against GPU 1's (CPU) and the night before; the
   canary's projmean record through `pmcheck_1125.py` (C's eigenvalues in [0, 1], trace = mean
   lane count).
3. Both GPUs of the canary, then the fleet at the next cycle. The science check (coherent power
   per band against cx19's control phases, fixtures/deproj_1124) on the first full day.
4. Then widen: `n2-project-live-freq-ids` to 1140-1299 MHz and a higher `k_lanes_max` as KV
   decides; the GPU cost grows ~linearly in channels and k.
5. Rollback at any step: the flag off, regenerate, node cycle; the CPU path is unchanged.

### 4.4 Risks

- **Our command inside the stock science correlator's pipeline.** A bad slot table would
  corrupt or crash the science N^2 for every channel. The kernels write only the slots' channels
  and only lower-triangle entries; the slot table is checked against the frame metadata and the
  record; test 1 covers the layout edges. stock_parity.py declares the change.
- **The lane builder can still stall the correlator** if the live-block copy-out is stock
  (blocking) cudaOutputData: a GnssN2Project stalled for 36 frames (1.5 s) stalls `run_n2k`, as
  today. Open decision 3 removes it.
- **Lag**: the basis is one frame older (42 ms); the re-steer covers LEO motion; a late host
  is recorded, not hidden.
- **FP64 on the L40S** is the cost driver: fine at 43-100 channels (0.7-2.5 ms), FP32 needed
  beyond that.
- **One integration per frame** is required in GPU mode (production has one); the record already
  supports several, the lag logic does not.
- **cuBLAS in run_n2k**: a handle per command instance, workspace on the device, first-call
  latency at startup; the library is already loaded for cudaFRBBeamReformer.
- **A40 budget numbers**: the GPU budget was measured on A40s; re-measure the node in situ (nsys,
  CHORD_GPU_BUDGET.md "How to measure again") at the canary.

## 5. Open decisions

1. Apply the recorded complex<float> basis in the CPU path too (record exact in both modes, CPU
   and GPU bit-identical; today's output moves by about a count at the brightest channels).
   Proposed: yes.
2. FP64 (0.74 ms at 43 x 40, exact) or FP32 (0.08 ms, 1 count on 2-4% of the words). Proposed:
   FP64 now.
3. The live-block copy-out: stock blocking cudaOutputData (today's exposure: the lane builder
   can stall the correlator after 1.5 s) or the command copies it out itself and drops a frame
   when GnssN2Project has no empty one (the science N^2 never waits on the lane builder; the
   stage then sees a gap in seq, which it already handles; ~40 more lines). Proposed: drop, given
   the 11-25 restarts.
4. The lag policy: newest record, never wait (proposed), lag 1 at steady state, recorded per
   frame.
5. `rank_step` on the GPU now or later (0.8 ms of host per frame). Proposed: later.
6. The canary node and GPU.

## 6. Size (estimate before the build)

About 1,600 lines: CUDA kernels ~250, the command ~400, the kernel check ~250, the stage ~150
added and ~40 removed, the generator and its tests ~200, the CI config and checker ~250,
stock_parity and docs ~100. Two to three working sessions to a tested branch, then the canary.

## 7. As built (2026-11-27)

KV's decisions (via coord): FP64; both paths apply the recorded complex<float> basis; the
command copies the live blocks out itself and drops a frame when GnssN2Project has no empty one;
the newest basis, never waited for; `rank_step` stays on the host; the canary layout is cx43
GPU 0 on the GPU and GPU 1 on the CPU; the command lives in the stock `run_n2k`.

**What was built.**
- `lib/cuda/cudaGnssN2Project.{hpp,cpp}`, `lib/cuda/cudaGnssN2ProjectKernel.{hpp,cu}`: the command
  and its kernels, as in 3.2-3.6 (`REGISTER_CUDA_COMMAND_WITH_STATE`; config `stations`,
  `live_freq_ids`, `k_lanes_max`, `in_basis_buf`, `live_buf`, `gpu_mem_basis`,
  `n2k_correlation_name`).
- `GnssN2Project` `gpu: true` (GnssN2Project.hpp "GPU"); both paths round the basis to
  complex<float> before applying it; the record's bytes 40-47 hold the source frame's seq on both
  paths.
- Generator `--n2-project-gpu GPUS`; `stock_parity.py` declares the `run_n2k` changes; the fleet
  manifest leaves the flag off and names the canary beside cx43.
- Tests: `cuda_gnss_n2_project_test`, two Boost cases in `test_gnss_n2project_math`, the GPU CI
  config `gpu_batch/run_n2accum_gnssproject_gpu.yaml` with `scripts/gnss/n2proj_gpu_check.py`,
  a generator test.

**Departures from sections 3-4.**
- The live-block header also carries the words per slot and the k applied per slot (padded to
  64 bytes), so the stage knows which channels the GPU projected
  (`gnssN2ProjectMath.hpp` `live_head_bytes`).
- The bases go to the GPU in their own buffer, `gnss<g>_lane_basis`; `gnss<g>_proj_basis`
  keeps its name and size and now carries the GPU's copy to N2Accumulate.
- The gauges: the command's `kotekan_gnss_n2project_gpu_basis_lag_frames` and
  `..._gpu_live_dropped_total` (labelled by the command), and the stage's
  `..._basis_dropped_total` (a basis with no empty `gnss<g>_lane_basis` frame).
- The GEMMs run over the frame's largest applied k, not `k_slot`, so the cost follows the lanes in
  use.
- Word-for-word equality needs real autos, which the correlator's are: with a nonzero imaginary
  part on the diagonal the block is not Hermitian and the two paths' (equivalent for Hermitian V)
  formulas differ by a count on a few percent of the words. testN2kGen's correlation types all
  have one, so the GPU CI config runs the real correlator on random voltages and
  `gpuSimulateN2kCorr` for the CPU path instead of testN2kGen through `cudaInputData` (4.2).
- Test 4 (recorded lane dumps) was not run: no raw dump frames are on disk off the nodes.

**Results.**
- Kernel check (cf05 GPU 1, L40S): every word of every projected channel equals the CPU path,
  k 0-40 with k_slot 40 and k 2-64 with k_slot 64 (shared memory above 48 KB), a station with no
  lane component, the channels outside the slots, the other inputs' own products and the poisoned
  upper halves untouched; the live blocks equal the raw and projected channels. 0.740 ms per
  frame at 43 slots x 40 directions (gather, project, gather).
- GPU CI check (cf05 GPU 0; 8 channels, 6 live, 119 stations, 6 lanes, 24 frames): 23 of 24
  frames had the same basis on both paths and are equal word for word; the first, with none
  applied, is the raw N^2; the other channels raw on both paths; the live blocks, records (seq,
  source seq, lag 1) and gauges agree (lane_k and auto_raw_db equal, auto_proj_db within
  8e-6 dB; null_db -4.87 against -4.71 dB, the GPU stage's EMA starting a frame later); no
  frame dropped. `run_tests.sh` passes the config in 15 s.
- The float basis on the CPU path against today's (same inputs): on synthetic frames with values
  up to ~1e7, 7-8% of the projected words move by 1 count, none by more; on the four CPU CI
  configs, 0.04-0.07% of N2Accumulate's visibilities differ, by at most 1.4e-3 (the mean
  projector records are identical: they were float already).
- In situ (cf05 GPU 0, the CI pipeline at production shape: 43 live channels of 48, 109
  stations, 40 lanes from the test publisher, #229 ranking and the lane dump on, 2 workers on the
  hyperthread pair 6+38; the voltage generator limits it to ~2.7 frames/s): `cudaGnssN2Project`
  1.14 ms per frame on the GPU (the kernels plus the record's upload and the live blocks'
  copy-out), `cudaCorrelator` 0.10 ms, the record's copy-out 0.06 ms; GnssN2Project's
  `frame_ms` 10.1 ms (against 21-28 ms for the CPU path at 32 lanes on the bench and 26-28 ms on
  the nodes at 26-29 lanes); lag 1, nothing dropped. The frames arrive ~0.4 s apart there, so
  the stage starts each with cold caches; on a node expect less.
- Generator: `config/test_gen_chord_gnss_config.py` and `scripts/gnss/test_gen_fleet.py` pass;
  a cx43 config generated from the 11-27 cycle's recipe with `--n2-project-gpu 0` passes
  `kotekan --dry-run` on cf05 after the offline conversion (`fixtures/gpu_budget/make_prof_config.py`:
  DPDK input replaced, senders dropped; the telemetry senders dropped too and the DTV bundle read
  from the repo, as cf05 has neither). A full offline run of that config is input-bound (4
  frames in 35 s) and gives no timing.

**Canary (KV and coord).**
1. A fleet binary from `kv/gpu-projection` (CUDA 13.5, Release, DPDK, as
   `builds/build_fleet_6a59985ee_cu135.sh`), staged under a new name; the python half (generator,
   `stock_parity.py`) in the checkout the stack regenerates from.
2. Fleet manifest: `cx43: {n2-project-gpu: "0"}`. `gen_fleet.py --node cx43 --preview-dir DIR`
   first, then `--check`; the preview's `run_n2k.gpu_0` holds `cudaGnssN2Project`, its
   `gnss_n2_project.proj_0` has `gpu: true` and `num_local_freq` 43.
3. A node_up cycle of cx43 alone onto the binary.
4. Judge in minutes on cx43 (`curl -s cx43:12048/metrics`): `NRestarts` 0 and no FATAL in the
   log (record or slot mismatch, live-block header); `kotekan_gnss_n2project_gpu_basis_lag_frames`
   1; `..._gpu_live_dropped_total` and `..._basis_dropped_total` flat;
   `kotekan_N2accumulate_proj_mean_dropped_total` flat; proj_0's `frame_ms` well under proj_1's;
   proj_0's `lane_k`, `auto_proj_db` and `null_db` per freq_id against proj_1's and the night
   before; `GET /gpu_profile/run_n2k/gpu_0` for `cudaGnssN2Project` (~1-1.5 ms) and the
   correlator's copy-out unchanged; the projmean record through `pmcheck_1125.py`.
5. Then GPU 1 (`n2-project-gpu: "0,1"`), then the fleet in the common block, each at a cycle.
6. Rollback: remove the key, regenerate, cycle: the stage's CPU path, as today but for the float
   basis.

