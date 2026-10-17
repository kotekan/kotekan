# CHORD node GPU budget (measured 2026-10-13)

Where the GPU time goes on a GNSS node, measured with Nsight Systems on a spare node (cx52)
running the production cx44 config against a real-time replay, with the F-engine off. Tools,
captures and the regeneration recipe live in `fixtures/gpu_budget/` (README there). Numbers are
per GPU (A40) per 41.94 ms frame (8192 hops = four GNSS records), kernel time from the CUPTI
trace, frames counted by `pl_mask_expand_kernel` launches.

## The budget

| per GPU, per 41.94 ms frame | stock production | + GNSS, 12 PRN x 7 bands | GNSS delta |
|---|---|---|---|
| gnss_waveform (replica synthesis, 28 launches) | 0 | 6.4-6.8 ms (3.2-3.5 with `epl_from_prompt`) | +6.4-6.8 |
| gnss_pack44 (lanes into the voltage ring) | 0 | 1.05 | +1.05 |
| n2k_dual (MIXED+BB over the comb union) | 0 | 0.62-0.68 | +0.65 |
| n2k (science N^2, 384 ch) | 1.03 | 1.18 | +0.14 (contention) |
| RFI: s12 / sk / s012 / s0 | 0.94 | 1.00-1.06 | +0.06 |
| PL mask expand + correlate, tile gather | 0.03 | 0.08-0.10 | +0.05 |
| **kernels, union busy** | **2.0 ms (4.8%)** | **9.6-10.2 ms (23-24%)** | **+7.6-8.2 ms (+18-19%)** |
| host->device voltage copy (copy engine) | 16.1 ms (404 MB, 25 GB/s) | 16.3 | +0.1 |
| device->host copies | 1.14 | 1.43 | +0.29 (GNSS records) |

Captures: `nsys_prof_cx52_prod_20261013_1454Z.*` (stock `pathfinder.j2`), `nsys_prof_cx52_20261013_1428Z.*`
(GNSS, 12 PRN seeded on every chain), `nsys_prof_cx52_20261013_1417Z.*` (GNSS config, seeds expired:
no synthesis), `delta_gnss12_vs_prod_20261013.txt`, Gantt PNGs beside each.

What the numbers say:

* **nvidia-smi's utilization.gpu is mostly the input copy.** It read 40/46% on the production-only
  node and 55-70% with GNSS while the SMs were busy 5% and 24% of the time. The counter includes
  copy-engine time, and the 404 MB per frame host->device voltage copy occupies 16 ms of every
  frame. SM headroom today is ~75% per GPU with GNSS running.
* **GNSS costs ~8 ms per frame per GPU, 80% of it synthesis.** Each waveform launch takes a median
  0.18 ms whether it has 36 or 84 blocks (one wave on 84 SMs; the bench said 0.15 ms on an idle GPU),
  so a chain's record costs one wave regardless of its channel count, and the per-frame cost is
  7 chains x 4 records x 0.18 ms. About 10-15% of launches take 0.8-0.9 ms when they coincide with
  n2k or the input copy (both DRAM-heavy). Scaling: flat in PRN count up to ~12 per band (one
  wave), then linear; flat in dish count (dishes enter only the N^2).
* **Nothing overlaps.** Overlap factor 1.06-1.09: the RFI and science N^2 run first, then the seven
  chains serially on one stream, then the dual; an 11 ms burst mid-frame and ~20 ms idle. The
  input copy of the next frame does not overlap any of it.
* **GLONASS L3OC** (1202 MHz, on 5 of the 7 E5b comb channels) is one more small launch per record:
  ~0.9 ms per frame, ~2% of a GPU. Its blockers are ephemerides (#129) and the code/NH, not GPU.

## Replica synthesis: the levers

Measured per-record cost = (chains) x (one 0.18 ms wave each, at fp16 Phi and 80 centred chips).

1. **E and L lanes from P: verified on sky and implemented (`epl_from_prompt`, 10-13).** A
   code offset of ds chips is a per-channel phase rotation of the prompt replica,
   exp(+i 2 pi (f_c - f_car) ds / chip_rate) for the prompt - ds trial, up to the offset's phase
   ramp across one channel (0.06 rad at 195 kHz and 0.5 chips of 10.23 Mcps). On the 10-12
   all-band telemetry capture (`fixtures/tec_wander/telem/telem_all8_20261012_1643.bin`, 578k
   frames, per-channel E/P/L; `fixtures/gpu_budget/epl_identity_sky.py`) the fitted rotation is
   -1.000 k0 (E) / +1.000 k0 (L) on every 10.23 Mcps band and E6, the identity holds to -18..-26 dB
   over all rows and -21..-37 dB on the top SNR decile, and the lobe-coherent DLL discriminator
   (gnssFleetDll) rebuilt from the rotated prompt matches the true one with zero median bias and
   0.004-0.016 chips rms on strong rows (0.03-0.04 on all rows = noise). L2C is the exception:
   one channel per instance, so its rotated discriminator is identically zero; it keeps three
   trials. The DLL is lobe-coherent everywhere (C++ gnssFleetDll, python combdll, the combiner's
   header sums); the incoherent per-channel taps are diagnostics only.
   Implementation (branch kv/epl-from-prompt, worktree ~/gnss/elp-wt): the waveform kernel walks
   the prompt once and writes the E and L rows as the rotated prompt with the prompt's energy;
   lanes, pack, N^2, assembler, records and broker are unchanged. Config key `epl_from_prompt`
   on the track block (generator `--epl-from-prompt`, every chain but L2C), default off.
   `cuda_gnss_despread_test` keeps P and P_HEAD bit-exact against the fused kernel and pins the
   rotation's sign against the fused E row (the analytic sign was wrong by a conjugation).
   Measured on cx52, same binary, 12 PRN x 7 chains per GPU (`run_epl_ab.sh`,
   `delta_elpon_vs_elpoff_20261013.txt`): gnss_waveform 6.40 -> 3.24 ms per frame on GPU 0
   (6.77 -> 3.47 on GPU 1 against the 14:28Z baseline), n2k contention -0.15, kernels union busy
   9.57 -> 6.57 ms (22.8% -> 15.7% of the frame): the GNSS delta over stock falls from ~7.6 to
   ~4.6 ms per frame per GPU. The remaining half of the launch is the per-hop fixed cost
   (carrier phasor, the fp64 code phase, the Phi prologue), not the gather.
   On sky 10-17 (#189 wire test, cx51 alone armed): nvidia-smi utilisation 71/67 -> 60/62 % on
   cx51's two GPUs with cx43 unchanged at 70/67, tracking indistinguishable from the sibling;
   fleet-wide from 18:36Z: 58-64 % on every GPU with data (#189 closed).
   Stage 2 (`b309a150a`, 10-17): with `epl_from_prompt` the comb chains pack two lanes per PRN
   slot (P, P_HEAD; `lanes_per_slot`) and GnssN2RecordAssemble forms the E and L rows from P and
   the kernel's rotation carried in PrnCtl::rot, so the pack writes half the lanes, each chain's
   gather carries half the tiles (917504 -> 458752 bytes per frame) and the dual's used lanes
   fall from 384 to 192 (it still runs its one 128+384 instantiation). Canary cx51 19:47Z:
   tracking indistinguishable from the fleet (q within -0.10..0 of the median, discriminator
   0.02-0.07 rms apart), the telemetry E/L rows the rotated P to -99..-107 dB, nvidia-smi
   56/56 % against a four-lane sibling's 59/59 %. The lane headroom is the other half of it: a
   ninth chain (GLONASS L3OC on the E5b comb) fits the axis with no wider dual.
2. **Fewer chips (`despread_max_chips`, centred).** Loss is exactly the dropped window energy:
   212 chips 0 dB, 105 -0.018, 80 -0.045 (production), 60 -0.27, 40 -1.1, 20 -3.4. Cross-PRN
   leakage does not depend on it (set by the comb's channel count: -32 dB on 6 channels, -38 on
   27) and the aliases shrink rather than grow. The per-block time scales with chips (bench: 0.10
   ms at 40 vs 0.15 at 80), so 40 chips saves ~2 ms per frame. Nothing but the tracking SNR
   sees the dropped energy, see below.
3. **Centralising synthesis** moves the same GPU-seconds (cost is linear in channels, nothing is
   shared across channels or PRNs because the Doppler sits in the Phi table exponent). Fleet-wide
   today ~1.2-2.4 A40s, full CHORD channel count 3-6, 1-2 with lever 1. The replica is already
   quantised to the ring's 4+4 bit, so shipping it is 1 B per (lane, channel, hop), 7-12 Gb/s per
   node. Worth it only at full CHORD's ~1 channel per lobe per node, or to free correlator GPUs.

## Chips kept: what else sees it

The truncated replica R_n is the kept part of the true channelised signal S, and the kept and
dropped parts are orthogonal to the precision of the loss curve (loss = kept energy exactly).
Consequences for everything downstream of the correlation:

* **Element correlations keep their ratios.** Every element correlates against the same replica,
  so the truncation is a common-mode scale (0.88 in amplitude at 40 chips) on all per-element,
  per-channel correlations: element cal and steering solutions are unaffected; absolute C/N0
  gates (deep_snr, present, trim quality) see the loss.
* **The DLL is unaffected** beyond SNR: the discriminator slope at 80 chips equals the full one
  (-4.55 vs -4.56 per chip) because on a comb the code phase is the cross-channel phase.
* **Deprojection does not see the replica.** Both projections are spatial. GnssN2Project takes
  its subspace from the science N^2 block's own top-k eigenvectors and removes the whole mode in
  each channel: the tracked code, the pilot and encrypted components, and the clipping products
  that carry the satellite's signature (G19, P(Y) only, was projected on 10-03). The record-row
  projection in GnssGpuRecordAssemble uses the bright satellite's P row for its direction only,
  which truncation scales and does not turn. Depth is set by the F-engine clipping floor
  (-20 dB rank 1, -25..-30 at rank 2-3) and by the in-channel delay spread of the array
  (rank-2 leak ~ (2 pi tau B)^2/12: -27 dB at the pathfinder's 0.12 us, -21 dB at 0.24 us),
  not by chips kept. The chips knob trades tracking SNR against GPU time and nothing else.

## How to measure again

* **Offline on a spare node, no F-engine:** `fixtures/gpu_budget/README.md`. The stock production
  config comes from the generator's `render_stock_template(config/chord/pathfinder.j2)`; a node
  config from `config/generated/`; both go through `make_prof_config.py` (replay input on tmpfs,
  senders dropped, no FPGA controller, REST 12049), then `prof_run.sh`, and `prof_delta.py` diffs
  two reports. About six minutes per run.
* **Live node, no restart:** `nsys profile --gpu-metrics-devices=all -t none -d 30 sleep 30` gives
  SM-active %, warp occupancy and DRAM bandwidth system-wide at 10 kHz.
* **Live node, kernel timeline:** launch that node's kotekan under `nsys profile -t cuda -y 120 -d 60
  -o F ...` at a node cycle, `nsys export --type sqlite`, `nsys_gantt.py F.sqlite`.
* nsys 2026.3.2 and ncu are on every node with `RmProfilingAdminOnly: 0` (no root). DCGM is not
  installed: it is a per-node daemon (`nv-hostengine`) with `dcgmi` or `dcgm-exporter` clients;
  its profiling fields SM_ACTIVE (1002), SM_OCCUPANCY (1003), DRAM_ACTIVE (1005), PCIe (1009/1010)
  are the continuous version of these numbers, and must be paused for nsys/ncu runs.

## Open items (for the buglist)

* Arm `epl_from_prompt` on the fleet at a node cycle (generator `--epl-from-prompt` in the
  manifest's common flags; the worktree branch kv/epl-from-prompt merges first) and judge the
  DLL by the readback decay and the trim-quality gates against the previous night.
* Stage 2 of lever 1: 2 lanes per slot, E/L rows from P in `GnssN2RecordAssemble`.
* `despread_max_chips` 80 -> 40 is -1.1 dB of tracking SNR for ~2 ms per frame; take it only
  after lever 1, which costs nothing.
* The synthesis tail (p90 0.8-0.9 ms when overlapping n2k or the copy): order the chains' stream
  away from the n2k window, or accept ~2 ms per frame.
* `testDataGen` cannot source 9.6 GB/s (per-byte type-string compare) and multiplies its start
  seq in 32 bits; `rawFileRead` has no loop; both worth a note upstream if anyone simulates input.
