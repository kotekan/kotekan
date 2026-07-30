# CHORD GNSS — state of play, 2026-07-30

Working state of the CHORD-side GNSS instrument on branch `kv/chord-gnss`. Written at the end of
the first day of live work, for whoever picks it up (including me tomorrow).

**Where we are:** the pipeline runs end to end on real sky with zero frame loss, the broker seeds
nine GPS L5 satellites from BRDC, and the acquisition search is correctly configured. **No
detection yet.** One blocker remains and it is well characterised (§5).

Read `config/chord_gnss_node.yaml` first — it is the source of truth and every measured number
lives there. This file is the narrative: what works, what broke, and what is left.

---

## 1. What is built and validated

| Piece | State |
|---|---|
| `GnssChordVoltageTap` | tap `host_voltage_buffer` → `[hop][chan][elem]`; owns the valve/drop behaviour |
| `launch_waveform` + `launch_correlate_nm` | split replica generation from N×M correlation; **exact-equality gated** vs the fused kernel |
| `cudaGnssChordTrack` | new cudaProcess command; model-primary, no frozen-seed machinery, no peel |
| element axis | through the record schema, assembler and combiner; `n_elements: 0` reproduces the airspy layout byte-for-byte |
| `GnssChordDequantize` | 4+4b → cfloat32 for the search; optional zero-fill (off) |
| `GnssChannelizedSearch` | now accepts a **sparse** comb via `channel_ids` |
| broker | CHORD model-primary path: `--time0-endpoint`, `--dr-clock-chips`, optional `--detectors` |
| config generator | production config as BASE + injected GNSS branch; separate search instance |
| `--check-config` / `--dry-run` | config and pipeline-graph validation, incl. the stall check |

Two kernel gates, both **exact** (`0.000e+00`), re-run as regression after every change:
split-vs-fused at N=1, and the element axis at N=5/stride=7 with poison padding.

## 2. Proven on sky

* **Ingest is lossless at production rate.** 391.4 kpkt/s per port (production runs 388.5),
  `rx_missed` 0 in steady state, `ring_full_dropped` 0, `gnss_tap_dropped_frames_total` **0**.
* **The tap reads real sky.** Peeked 1,835,008 bytes = exactly 8192 hops × 7 chan × 32 elem;
  real/imag means ≈ 0, rms 1.81 per component (the 4+4b contract's intended band), range
  [−7,+7], and **zero occurrences of nibble 0** — matching the paper's clip to ±7 and proving
  byte alignment.
* **Records are correct.** 52,228 B = 32 PRN × 408 floats (24 header + 32×12 element blocks),
  finite, at the predicted rate (4 records/frame ÷ `output_every` × 2 GPUs = 19/s).
* **The broker seeds.** Nine GPS L5 PRNs from BRDC every cycle, code phases in [0,10230),
  Dopplers within ±2.8 kHz, and the tracker's reported Doppler tracks the seed (PRN 32:
  2025 Hz vs seeded 2133).

## 3. Bugs found and fixed (all mine unless noted)

1. **`f_offset` must be the SKY CARRIER, not 0** — the big one. `ChannelizedReplicaBank` places
   the replica at `f_offset`, and `covering_bins()` sets `local.carrier_hz = f_offset`. With 0
   the replica sat at DC (bins −51..52) while the data is at 5971..6076: **zero overlap**, every
   correlation noise. On the airspy node `f_offset` is the post-mixer IF because that receiver
   downconverts; CHORD does not downconvert at all. Fixed on both the search and the tracker.
   **It failed silently in two disguises** — the search reported "carrier not in this subband"
   (indistinguishable from an empty sky) and the tracker reported `amp_snr 0` with a railed
   `dll_disc` (indistinguishable from an unknown instrumental delay, which cost a whole cycle
   chasing cable lengths).
2. **Pruned an ingest stage.** `process_packet_mask` was dropped as "N²-only"; it produces the
   packet-loss mask `TransposeBasebandArray` REQUIRES. The transposes blocked forever with DPDK
   happily filling its input buffer. `run_send_voltage` was the inverse error — kept when it
   should have been dropped, and would have stalled from the other side.
   → `--dry-run` now fails on any buffer left with consumers and no producer.
3. **Wire-format mismatch on the search link.** `bufferSend`/`bufferRecv` default
   `use_config_tracker` to whether the *instance* has a `/config_tracker` block, and the two
   instances differed. Header fields shifted; surfaced as "Frame size does not match between
   server: 57344 and client: 12" where both sides were really 57344 and 12 is
   `sizeof(GnssChanMetadata)`. Now pinned explicitly on both ends.
4. **`hint_ttl_s` 8 s vs a 10 s broker refresh.** With `require_hint` that skips every PRN
   *before* its SNR is computed, so the stage goes silent for a reason nothing in its output
   explains. The airspy chain sets 30 s; now so do we.
5. **Broker variable shadowing** — my `t0` clobbered the loop's cycle timestamp, so
   `time.sleep()` got 1.8e18 s and it died after one (successful) cycle.
6. **Nibble convention** — I stated low-nibble-real from `s12_kernel.cu`'s naming; that kernel
   only forms `re²+im²` so its labels are unobservable. Authoritative is `gpuSimulate.cpp`:
   **HIGH nibble = REAL**. Getting it wrong conjugates the data — magnitudes and acquisition SNR
   unchanged, Doppler sign inverted.

**Pre-existing, not ours:** `cuda_gnss_despread_test` exits 1 on a trial-reorder invariant
(1.914e-07), confirmed present before our changes. `restServer::set_server_affinity` segfaulted
on an unstarted thread (fixed, worth upstreaming).

## 4. Things I got wrong about the architecture, corrected

* **The search does NOT need contiguous channels.** `channelized_accumulate` FFTs along the
  **hop** axis within each channel and sums per-channel surfaces — it is "the distributable half
  of the search", built for scattered channels. I conflated the snapshot's existing gap
  tolerance in TIME with a supposed requirement in FREQUENCY. The zero-fill I added was
  unnecessary (kept, off by default). The 640 ns comb ambiguity is a property of the
  measurement, resolved by the BRDC model with ≥16× margin.
* **Search products are ~38× the raw voltage** (26.4 MB of surfaces from 700 kB of data), which
  is why the design ships **raw voltage** to a central aggregator and runs ONE search over the
  gathered union — not a search per node. At eight nodes that union is contiguous (all eight
  mod-8 offsets present), so full sensitivity and no aliasing.

## 5. THE REMAINING BLOCKER — replica generation cost

`GnssChannelizedSearch` calls `_replica->channels(p, anchor, 0, 0, Mp)`, which returns the
**entire** spectrum `[N][Mp]`, then slices out the channels it wants:

```
N = 8192 (CHORD's PFB spans 0-1600 MHz), Mp = 3125
  per PRN              204.8 MB   (8192 channels x 3125 hops)
  x 32 PRNs              6.55 GB
  x 2 search instances  13.11 GB   (observed RSS 19.4 GB)
we use 7 of 8192 rows = 0.09%   ->  1170x waste, in memory AND compute
```

On the airspy node `N = 10`, so the full spectrum *is* the useful set and there is no waste.
This is the single reason the search is unusable here: it explains the 8-minute passes, the
"336 passes then nothing" burst, and the final starvation (1 pass in 5 min with workers idle).

**Fix direction (agreed to start here tomorrow):** a *banded* replica call that generates only
the requested channel set. Two notes:

* It matters far more at full scale, not less: the eight-node aggregator wants 104 of 8192 bins,
  still 79× waste.
* **It should be PRECOMPUTED, not regenerated per pass.** The `repl0` used by the search is the
  code at Doppler 0 and code phase 0 — it depends only on (PRN, channel set, Mp), none of which
  change between passes. The airspy node gets away with regenerating because its spectrum is 10
  channels wide. Caching it per PRN, banded, is both the correctness-neutral and the cheap fix.

This is shared GNSS-side code (`gnssChannelizedReplica`), so coordinate before changing the
interface — see `cudaGnssChordTrack.hpp`'s merge note for the shared/duplicated split.

## 6. Also outstanding

* **Instrumental delay is still unmeasured.** The cable term is now well determined —
  100 m LMR-400 (vf 0.85) + 6 m LMR-195 (vf 0.83) = **4.26 ± 0.18 chips**, comfortably inside
  the ±0.5-chip DLL capture. What is NOT known is the F-engine's internal pipeline/framing
  offset (one frame = 52.4 chips; PFB group delay = 104.8 chips, and whether it cancels because
  the replica goes through the same PFB is an argument, not a measurement). The search is meant
  to measure this: its reported code phase minus the model prediction IS the constant. Once
  measured it goes in `chord_gnss_node.yaml` and dead-reckon cold-start works permanently.
* **`rawFileWrite` makes one file per frame** — 19/s ≈ 1.6 M files and 86 GB/day. Fine for a
  test, wrong for a soak. Raise `output_every` or use an appending writer.
* **Frequency axis collapse** is an early priority once there is proof of life — see the
  `roadmap` block in `chord_gnss_node.yaml`. BOC signals make it critical, not cosmetic.
* **A grinding search worker ignores SIGTERM** (it cannot check `stop_thread` mid-FFT), so it
  holds its REST port and the replacement instance fails to bind and exits silently. Use
  `kill -9` when restarting the search.

## 7. How to run it

```bash
# search instance -- ordinary user, no DPDK/GPU/hugepages, fully dry-runnable
./build/kotekan/kotekan --config config/generated/chord_gnss_search_cx19.yaml \
    --bind-address 0.0.0.0:12050

# node instance -- needs sudo for DPDK hugepages (/dev/hugepages is root:root, and
# /dev/vfio/{19,38} too, so the group route needs BOTH; sudo is simpler for debugging)
sudo ./build/kotekan/kotekan --config config/generated/chord_gnss_cx19.yaml \
    --bind-address 0.0.0.0:12049

# broker -- note -u, python buffers stdout when piped and you lose everything on kill
PYTHONUNBUFFERED=1 /home/kvand/gnss/venv/bin/python -u \
  python/scripts/gnss/gps_distributed_broker.py \
  --rest-url http://localhost:12049 \
  --detectors http://localhost:12050/srch0_search,http://localhost:12050/srch1_search \
  --trackers gnss0_track,gnss1_track --combiner gnss0_combine \
  --almanac --almanac-source brdc --dead-reckon --narrow-search \
  --time0-endpoint telescope/time0_ns \
  --constellation G --carrier-hz 1176.45e6 --code-length 10230 --hops-per-sec 195312.5 \
  --lat 49.32075144444 --lon -119.62081125 --alt 545 --mask-deg 0 --interval 2
```

Ports: **12048** production (choco owns it; the generator REFUSES to emit it), **12049** node,
**12050** search, **11040/11041** the raw-voltage feed. Always validate with
`sudo ... --dry-run` before a real run — it is the only way to exercise the full graph including
DPDK, and it is what would have caught the pruned-ingest-stage stall.

**Regenerate, never hand-edit:**
```bash
python3 config/gen_chord_gnss_config.py --base <base.json> --node cx19 \
    --search-element 23 [--search-instance] --out config/generated/...
```
where `<base.json>` is `curl -s http://cx19:12048/config` from a node running production.

## 8. Operational notes

* Cores: DPDK owns 5-7 and 21-23 (0% idle) plus their isolcpus siblings — never touch. Our
  stages use NUMA1: 19, 24, 31, 57-63.
* `/home/kvand` is NFS shared across all eight nodes at the same path (code, config, venv);
  records must go to local disk (`/data/gnss`, 2.6 TB free).
* venv at `/home/kvand/gnss/venv` (skyfield 1.54 + numpy/scipy/h5py/yaml/requests).
* `kvand` has sudo (password), and is in `docker`. Passwordless is NOT configured.
* choco has a per-node **maintenance mode** ("choco will not push"). It must be ON while we run,
  or choco may push a production config over us mid-run.
* **Cyg A transits ~07:2x UTC**, 81.5° elevation, drifting 3m56s earlier daily. Several dishes
  use it for beam maps and noise temperature; keep the X-engine healthy ±4 h around it.
* Eight nodes: cx19(4) cx27(0) cx42(5) cx43(6) cx44(2) cx47(3) cx51(7) cx52(1), where the number
  is the mod-8 comb offset. Together they tile the science band exactly.
