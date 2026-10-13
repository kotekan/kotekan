# CHORD node GPU budget (measured 2026-10-13)

Where the GPU time goes on a GNSS node, measured with Nsight Systems on a spare node (cx52)
running the production cx44 config against a real-time replay, with the F-engine off. Tools,
captures and the regeneration recipe live in `fixtures/gpu_budget/` (README there). Numbers are
per GPU (A40) per 41.94 ms frame (8192 hops = four GNSS records), kernel time from the CUPTI
trace, frames counted by `pl_mask_expand_kernel` launches.

## The budget

| per GPU, per 41.94 ms frame | stock production | + GNSS, 12 PRN x 7 bands | GNSS delta |
|---|---|---|---|
| gnss_waveform (replica synthesis, 28 launches) | 0 | 6.4-6.8 ms | +6.4-6.8 |
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

1. **E and L lanes from P (untested on sky, the biggest lever).** A code delay is a per-channel
   phase rotation of the prompt replica, exp(-i 2 pi (f_c - f_car) delta). In the direct PFB model
   the lobe-coherent DLL discriminator formed from the rotated prompt differs from the exact E/L
   one by < 1e-4 chips at production geometry; the incoherent per-channel E/L powers carry no
   code phase at all (slope 0.001/chip vs 4.56 coherent). The kernel walks three trials that each
   load their own Phi lines and the N^2 carries four lanes per PRN (E, P, L, P_HEAD): dropping E
   and L is ~2-3x on synthesis (~4 ms of the 6.5 per frame) and halves the MIXED lanes, with the
   record format unchanged if the assembler emits E/L rows as P x rotation. VERIFY FIRST on the
   sky capture `fixtures/tec_wander/telem/telem_all8_20261012_1643.bin` (rows carry per-channel E,
   P, L): check E_c = P_c e^{i theta_c} per record before any code moves.
2. **Fewer chips (`despread_max_chips`, centred).** Loss is exactly the dropped window energy:
   212 chips 0 dB, 105 -0.018, 80 -0.045 (production), 60 -0.27, 40 -1.1, 20 -3.4. Cross-PRN
   leakage does not depend on it (set by the comb's channel count: -32 dB on 6 channels, -38 on
   27) and the aliases shrink rather than grow. The per-block time scales with chips (bench: 0.10
   ms at 40 vs 0.15 at 80), so 40 chips saves ~2 ms per frame. BUT the dropped energy is also
   what deprojection cannot remove, see below.
3. **Centralising synthesis** moves the same GPU-seconds (cost is linear in channels, nothing is
   shared across channels or PRNs because the Doppler sits in the Phi table exponent). Fleet-wide
   today ~1.2-2.4 A40s, full CHORD channel count 3-6, 1-2 with lever 1. The replica is already
   quantised to the ring's 4+4 bit, so shipping it is 1 B per (lane, channel, hop), 7-12 Gb/s per
   node. Worth it only at full CHORD's ~1 channel per lobe per node, or to free correlator GPUs.

## Chips kept and deprojection

The truncated replica R_n is the kept part of the true channelised signal S, and the kept and
dropped parts are orthogonal to the precision of the loss curve (loss = kept energy exactly).
Three consequences for everything downstream of the correlation:

* **Element correlations keep their ratios.** Every element correlates against the same replica,
  so the truncation is a common-mode scale (0.88 in amplitude at 40 chips) on all per-element,
  per-channel correlations: element cal and steering solutions are unaffected; absolute C/N0
  gates (deep_snr, present, trim quality) see the loss.
* **The DLL is unaffected** beyond SNR: the discriminator slope at 80 chips equals the full one
  (-4.55 vs -4.56 per chip) because on a comb the code phase is the cross-channel phase.
* **Deprojection depth is capped by the dropped energy.** The rank-1 projection removes the part of
  the satellite the replica models, a a* |R_n|^2, and leaves a a* (|S|^2 - |R_n|^2): the residual
  satellite power in the science visibilities is (1 - kept) of the satellite, whatever the array
  gain. That is -19.7 dB at 80 chips, -12 dB at 60, -6.5 dB at 40, -24 dB at 105, -33 dB at 160.
  More dishes raise tracking SNR but do not touch this fraction. So 80 -> 40 trades 2 ms of GPU
  per frame for a 13 dB shallower null; if the measured projection floor (proj_b in the rows;
  E6 is set by F-engine clipping, other bands unknown) is near -20 dB, the lever points the
  other way: 105-120 chips buys 4 dB of depth for ~1 ms per frame. Check proj_b per band against
  -19.7 dB before moving this knob in either direction.

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

* Verify lever 1 (E/L from P) on the 10-12 sky capture; then the kernel (one trial + P_HEAD gate),
  the N^2 lane count, and `GnssN2RecordAssemble` emitting E/L rows from P.
* Compare proj_b per band with the -19.7 dB replica cap before touching `despread_max_chips`.
* The synthesis tail (p90 0.8-0.9 ms when overlapping n2k or the copy): order the chains' stream
  away from the n2k window, or accept ~2 ms per frame.
* `testDataGen` cannot source 9.6 GB/s (per-byte type-string compare) and multiplies its start
  seq in 32 bits; `rawFileRead` has no loop; both worth a note upstream if anyone simulates input.
