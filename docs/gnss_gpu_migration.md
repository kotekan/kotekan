# GNSS GPU migration plan (GX10 → CHORD x-engine)

*2026-07-10. Companion to `gnss_architecture_audit.md` §3 (GPU placement), which this makes
concrete. Target: CHORD x-engine nodes — 6144 × ~195 kHz PFB subbands, a comb of 48 per node,
each band processed independently; DPDK → host ring buffer → DMA → A40 → kernels → DMA out →
50 GbE. The GX10 (GB10, sm_121, CUDA 13, 121 GB unified) mimics this with its own fftw PFB
F-engine. Design rule: write for the A40 (explicit copies, `cudaInputData`/`cudaOutputData`);
the GB10's unified memory just makes staging cheap — never rely on it.*

## 0. What moves, what doesn't

| Op | Shape (per node) | GPU? | Phase |
|---|---|---|---|
| Tracker despread (E/P/L) + replica | PRN × subband × record × {E,P,L}: batched complex dot products | **Yes — the CHORD bulk compute** | G1 |
| Acquisition / search | PRN × Doppler × lag × window: batched FFT correlate + \|·\|² accumulate | **Yes — biggest single-op FLOPs** | G2 |
| Voltage peel | PRN × subband at full voltage rate, in-stream reconstruct−subtract | **Yes — required at scale** | G3 |
| Combiner / deep wipe | records only (~44 B/rec) | No — CPU | — |
| Broker / DLL / carrier loop | ~Hz control plane | No — CPU | — |
| F-engine | — | CHORD's upstream FPGA/GPU F-engine; fftw on GX10 | — |

Both despread and peel are *necessarily subband-local* (they touch full-rate voltage) — which
is exactly the correlate-at-data design restated for GPUs: each node's GPU sees precisely the
subbands it owns, no cross-band data ever needed (records combine downstream in CPU space).
Acquisition is flexible; run it on the node GPU's spare capacity against local subbands
(matches require_hint: only ~visible-set PRNs scanned).

## 1. Kotekan CUDA framework (as read, lib/cuda + lib/gpu)

- **`cudaProcess`** stage: owns a `cudaDeviceInterface` (N streams, default 3: COPY_IN /
  COPY_OUT / KERNEL), instantiates a config-declared `commands:` list; per GPU frame it walks
  the chain, each command's end-event feeding the next (`execute_base(pipestate, pre_events)`),
  `_gpu_buffer_depth` instances per command for frames in flight.
- **`cudaCommand`**: subclass implements `execute()`; helpers `record_start_event()` /
  `record_end_event()`; registered via `REGISTER_CUDA_COMMAND` (or `_WITH_STATE` for shared
  state across instances). Config keys per command: `cuda_stream`, `required_flag`, plus
  whatever the command reads.
- **I/O**: `cudaInputData` (host kotekan buffer → named GPU array, `cudaHostRegister`ed,
  COPY_IN stream), `cudaOutputData` (reverse), `cudaCopy{To,From}Ringbuffer` for the CHORD
  ring-buffer pattern. GPU arrays are named (`gpu_mem_*` config keys) and frame-indexed via
  `device.get_gpu_memory_array(name, frame_id, len)`.
- Reference chain example: `chordMVPSetup.cpp`; per-command kernels live in `.cu` files
  compiled by CMake when `USE_CUDA=ON` (`CMAKE_CUDA_ARCHITECTURES=121` for GB10).
- Build: separate tree `build_cuda/` (same flags as `build/` + `-DUSE_CUDA=ON`), so the CPU
  production binary is untouched.

## 2. Phase G1 — batched despread (the first port)

*Why first (updated from the audit): R1 made the tracker a decision-free batched complex dot
product — the simplest correct kernel — and it is the op that scales with CHORD (dish × PRN ×
subband × 3 correlators). The audit's original "search first" motivation (CPU pain from blind
grids) was retired by require_hint (visible-only narrow scans).*

**Unit of work** (one GPU frame = one hop-window of one subband group):
for each active (PRN, {E,P,L}) pair: `corr = Σ_ch Σ_hop data[ch][hop] · conj(repl[ch][hop])`,
`energy = Σ |repl|²` — i.e. `gnss::channelized_despread` batched over PRN×3.

**Replica source — the design decision:** the CPU per-sample path (`channels()`: PFB fold +
FFT per hop) is the expensive part and must NOT be ported naively. Use the hop-rate closed
form (`hoprate_stream`): per (channel, hop), the replica value is a short gather-and-sum over
the ~n_chips spanned chips using the precomputed cumulative filter tables `PhiA/PhiB[ch][tap]`
(Doppler-bucketed; rebuilt on CPU only when the Doppler moves > refresh_hz, uploaded once).
GPU kernel = fused replica-gen + dot product:

```
grid:  (n_prn_active × 3 correlators, n_covering_channels)   block: hops (125 @ L1 5 MSPS)
thread(prn3, ch, hop):
    C    = cp0[prn3] + n_hop·cps            // absolute-anchored code phase (double)
    sA,sB = Σ_d code[chip0−d] · (Phi[ch][khi+1] − Phi[ch][klo])   // ~2-25 chips
    r    = 0.5·(pa·sA + conj(pa)·sB)        // the replica channel sample
    partial += data[ch][hop] · conj(r); energy += |r|²
block-reduce → corr[prn3][ch], energy[prn3][ch]; tiny second kernel (or CPU) sums channels.
```

All state the kernel needs per (PRN,corr): `cp0, cp_rate, f_ref (fixed!), code table,
Phi tables, nav/overlay signs` — uploaded by the command from the seed snapshot (the R1 seeds
are already a flat REST-updated array). NOTE the carrier NCO/ff_hz/trim stay on the CPU side
(they act on the *record*, post-despread — records return to host anyway).

**Stage plumbing on the GX10** (mirrors CHORD): `chan_buf2 → cudaProcess[cudaInputData →
cudaGnssDespread → cudaOutputData] → rec_buf` — drop-in replacement for
`GnssChannelizedTracker`'s hot loop; the tracker keeps its seed/REST/NCO logic and consumes
the GPU correlations (or, simpler first cut: a standalone `cudaGnssDespread` command validated
against the CPU despread offline, then integrate).

**Validation gate:** GPU corr/energy vs `gnss::channelized_despread` on the same window ≤1e-5
relative (float); then replay `l1_5msps_2026-07-04.bin` end-to-end and match deep_snr.

## 3. Phase G2 — acquisition

`channelized_accumulate` decomposes as:
1. `channel_correlate(data_ch, repl0_ch, doppler_grid)` per covering channel: FFT-based
   circular correlation per Doppler → **batched cuFFT** (plan: [n_chan × n_dop] batch of
   length-Mp c2c FFTs; the replica FFT is cached per PRN).
2. `aggregate_accumulate`: for each (dop d, hop-lag q, sample-lag s):
   `D = Σ_ch P[ch][d][q] · ramp[ch][s]`, `surf[d][q][s] += |D|²` — pure MAC, one thread per
   (d,q,s), ramp in shared memory. (The CPU SoA loop is this exact kernel.)
3. `channelized_peak`: max-reduce over the surface (thrust/cub or a two-stage reduce), peak +
   sub-grid parabola on CPU (tiny).
Windows accumulate in place on-device (~20 windows) → only the peak crosses back. With
require_hint the batch is (visible PRNs × narrow grids) — small on GX10, and the same kernel
scales to CHORD's per-node comb.

## 4. Phase G3 — peel

Same fused replica kernel as G1, output flipped: `resid[ch][hop] = data − a·r` (per-PRN
successive, or joint multi-PRN with a small per-PRN gain vector). In-stream: input ring →
kernel → output ring (the `cudaCopy{From,To}Ringbuffer` pattern), records of the despread
by-product emitted for free. This is the CHORD RFI-mitigation payload; port after G1 proves
the replica kernel (it *is* the same kernel + a subtract).

## 5. Sequencing & gates

| Step | Deliverable | Gate |
|---|---|---|
| G0 | `build_cuda/` builds; framework smoke (cudaProcess in+out chain runs on GB10) | pipeline runs, no CUDA errors |
| G1a | `cudaGnssDespread` kernel + standalone CPU-vs-GPU test | ≤1e-5 rel. on random + replay data |
| G1b | wired stage (tracker consumes GPU correlations), replay A/B | deep_snr matches CPU path |
| G1c | live soak; measure headroom (PRNs × signals simultaneously) | ≥10× CPU throughput |
| G2 | cuFFT acquisition + accumulate/peak kernels, replay A/B | detections match CPU |
| G3 | peel in-stream on ring buffers | residual suppression matches CPU peel |

Risks: (a) GB10 sm_121 is new — pin nvcc 13 codegen, keep kernels plain (no arch-specific
intrinsics); (b) FRAGILE-DRIVER rule on GX10 — userspace only, never touch the driver;
(c) double-precision code-phase arithmetic in the kernel (absolute sample indices ~1e10 —
float is NOT enough for `cp0 + n·cps`; keep phase accumulation in double, math is cheap
relative to memory traffic); (d) the E/P/L triple shares the data window — compute all three
in one kernel launch (3× replica, 1× data read).
