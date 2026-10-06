# CHORD GNSS visibility capture (N² + N×M at full cadence)

A REST-armed capture of the Path-B dual-correlator output — the antenna–antenna (N²) and
antenna–replica (N×M) visibilities over the L5 channels the fleet despreads — for a few
minutes, at sub-integration cadence, on every node at once, started only after the tracker
is locked and happy. Written for the wider team's (N+M)² request.

## What the request maps onto

| asked | what the fleet has |
|---|---|
| (N+M)² triangle | N² block (`gnss_gather_aa`) + N×M mixed block + M² block (`gnss_gather_bb`, 2026-10-06; the correlator always computes it, the tracker never reads it). One triangle per chain: the replica × replica blocks BETWEEN chains (L5 × E5a, ...) are not computed anywhere until the chains share one correlator pass. |
| full cadence — 40 ms? 10 ms? | **10.49 ms is available now**: each 41.94 ms frame holds `n_rec = 4` sub-integrations of 2048 hops (hop = 5.12 µs). A 2.62 ms cadence would be a chain-wide `hops_per_record` change (tracker records, broker, everything) — not part of this. |
| ~100 L5 subbands | the fleet despreads **79 absolute channels** (6–7 per GPU instance × 12 instances; `freq_MHz = freq_id × 0.1953125`). Only those channels are correlated in freq-map mode; nothing outside the comb exists to capture. |
| N | the **32 live elements** (ids 0–15, 64–79 = `array.live_element_ranges`), so N² is 32×32 per channel per record. The remaining 96 element slots carry nothing. |
| M | 128 synth lanes = 32 PRN slots × 4 rows (E, P, L, P_HEAD) per chain (B2a: 24 slots, 96 lanes used). Which slot holds which PRN, its NCO frequency and seed phase, is in the ctl block record by record. The running slots are every capable satellite above the horizon that has a slot plus the broker's 3 noise probes (`mask-deg: 0`, `noise-probes: 3`). Since 2026-10-06 the manifest captures all three 1176 MHz chains (`vis-capture: gps_l5,gal_e5a,bds_b2a`); the extra chains' files carry the chain tag (`<node>_gnss<g>_e5a_vistiles_...`). |

## Data layout on disk

Per GPU instance, two `rawFileWrite` series in `<record_dir>/viscap/` (node nvme, `/tmp/gnss`):

    <node>_gnss<g>_vistiles_NNNNNNN.raw   int32 [n_rec][n_chan][n_tile][16][16][2]
    <node>_gnss<g>_visctl_NNNNNNN.raw     FrameHdr | winstart | PrnCtl[16][n_prn] | energy

24 frames (1.0 s) per file, `continue_numbering`. rawFileWrite framing per frame is
`[u32 metadata_size][metadata][frame]`; the gate deep-copies metadata when a pool exists.

`n_tile` per channel = `8 × nlive16` mixed + `nlive16(nlive16+1)/2` AA (+ `8·9/2 = 36` BB with
`gnss_gather_bb`), with `nlive16 = 2` live 16-element column groups → 16 + 3 = 19 tiles =
38912 B per (rec, chan), or 55 tiles = 112 640 B with BB. Frame sizes: 1 089 536 B (7-channel
instance) / 933 888 B (6-channel), or 3 153 920 / 2 703 360 B with BB. ctl frame 155 824 B at
32 slots (116 912 B for B2a's 24).

The ctl block's `FrameHdr.seq0` is the absolute F-engine **ADC sample** (3.2 GS/s) of the
frame's first hop — **one clock fleet-wide**, and the clock the gate uses. One hop = 16384
samples = 5.12 µs; a record (`winstart` step) = 2048 hops = 33 554 432 samples; a frame
(`seq0` step, the gate's `seq_per_frame`) = 8192 hops = 134 217 728 samples. `utc0` maps
sample 0 to UTC: `utc = utc0 + seq / 3.2e9`.

Decoder: `scripts/gnss/viscap_read.py` (`/home/kvand/gnss/venv/bin/python`). It reads the
pair, attaches the ABSOLUTE `channel_ids` from the generated node config and the element ids
from `chord_gnss_node.yaml`, undoes the 4-bit replica quantizer scale (`s = 7/(3·rms)` from
record 0's energy), and writes an h5 with `vis_aa[rec, chan, 32, 32]`,
`vis_mixed[rec, chan, slot, row, 32]`, `vis_bb[rec, chan, 128, 128]` (lane = 4·slot + row, when
gathered), `winstart`, `prn`, `run`, `fcar_report`, `f_nco`, `fcar`, `cp_seed`, `energy`,
`scale`, `freq_id`, `element_id`. Orientation: `V_mixed = synth·conj(antenna)`,
`V_aa[i,j] = E_i·conj(E_j)`, `V_bb[a,b] = synth_a·conj(synth_b)`. `viscap_read.load(...)` gives
the same arrays in memory for a slice of files (what the tutorial uses); `--tag _e5a` selects a
tagged chain's series. `scripts/gnss/viscap_selftest.py` checks the decoder cell by cell on
synthetic frames (mixed, AA, BB, the ctl join).

Sidecar metadata: `scripts/gnss/viscap_metadata.py <capture dir>` writes `<dir>/metadata/`
— `timing.json` (clock geometry, both windows in seq and UTC), `channels.csv` ((node, gpu) →
absolute freq_ids), `elements.csv` (element → dish/pol/grid position), `station.json`
(receiver point and array-origin frame), `sat_azel.csv` (az/el/range-rate of every satellite
at 10 s cadence from the broker's ephemeris code) and the BRDC files used. It reads only the
config snapshot in `<dir>/configs/` and the ctl files, so it can be rerun on any capture.

Tutorial: `scripts/gnss/viscap_tutorial.ipynb` — framing by hand, ctl header/PRN table,
time, tiles decode, N² and N×M plots, metadata join. Ships with outputs; re-execute with
`scripts/gnss/nb_run.py scripts/gnss/viscap_tutorial.ipynb` (no Jupyter needed).

## Volumes

~1.09 MB tiles + 0.16 MB ctl per 41.94 ms frame per instance ≈ **30 MB/s per instance per
chain**, 60 MB/s per node for gps_l5 alone. Five minutes ≈ 9 GB per instance, **18 GB per node,
~110 GB fleet** (gps_l5 chain only, no BB). With the three 1176 MHz chains and BB: 9.9 MB per
frame per 7-channel instance ≈ **236 MB/s per instance, ~2.7 GB/s fleet, ~800 GB per 5 min**
(300 GB without BB). Node nvme has ~3 TB free; the post-capture home is NFS
`/home/kvand/gnss/fixtures/viscap_<date>/` (8 TB free). The nvme write rate is well inside
what the beam-cube archiver already sustains; the gate drops (and counts) rather than
back-pressuring the GPU chain if a sink ever falls behind.

## Mechanism

```
n2dual (GPU) ─► gnss<g>_n2tiles_buf ─┬─► GnssN2RecordAssemble (tracker, unchanged)
                                     └─► gnss<g>_viscap_gate ─► viscap_tiles_buf ─► rawFileWrite
                gnss<g>_n2ctl_buf   ─┬─► GnssN2RecordAssemble
                                     └─► (same gate, lockstep) ─► viscap_ctl_buf ─► rawFileWrite
```

`FrameWindowGate` (`lib/stages/frameWindowGate.{hpp,cpp}`) consumes its input buffers in
lockstep and passes a frame tuple only while `start_seq ≤ seq < end_seq`, where seq is read
from the ctl frame (`clock_source: frame`, `clock_offset: 16` = `FrameHdr.seq0`). Outside
the window it releases frames immediately; inside it copies if every output has an empty
frame and otherwise counts a `window_dropped` frame (WARN, rate-limited). It auto-disarms
past `end_seq`. REST on the node's port (12048):

    POST /gnss<g>_viscap_gate/arm      {"start_seq": S, "end_seq": E}   (E ≤ 0 = open-ended)
    POST /gnss<g>_viscap_gate/disarm
    GET  /gnss<g>_viscap_gate/status   → armed, start/end/last seq, seq_per_frame, passed, dropped, window_dropped

Metrics: `kotekan_framewindowgate_{passed,dropped,window_dropped}_frames_total`,
`kotekan_framewindowgate_armed`.

The sinks are `rawFileWrite` with the new opt-in `create_base_dir` (the stage otherwise
`exit(errno)`s on a missing directory at its FIRST write — which here would be minutes after
arming, hours after startup).

Generator: `--vis-capture <chains>` (armed in `config/gnss_fleet_chord.yaml` as
`vis-capture: gps_l5,gal_e5a,bds_b2a`) adds, per captured chain, the gate, the two staging
buffers (`--vis-capture-depth 8`), the two sinks (`--vis-capture-frames-per-file 24`), and sets
`gnss_gather_aa: true` on the correlator AND the assembler; `--vis-capture-bb` (manifest
`vis-capture-bb: true`) sets `gnss_gather_bb: true` on both as well. Those two plus the
generator's `n2_tiles_per_chan()` form one contract: the assembler FATALs on a tiles frame-size
mismatch rather than mis-decode.

## Procedure (after a node cycle)

The gate is inert until armed. Nothing is written by a restart alone.

1. Confirm the tracker is locked and happy (viewer; `q`, not `sig`).
2. `scripts/gnss/viscap.py status --tags ,_e5a,_b2a` — every gate answers (36 with the three
   chains), `last_seq` advancing on all, `seq_per_frame` identical (134217728 samples = 8192
   hops). Without `--tags` the tool addresses the primary (gps_l5) gates only.
3. `scripts/gnss/viscap.py arm --tags ,_e5a,_b2a --start-in 30 --duration 300` — frame-aligns
   one absolute window and POSTs it to every gate; prints the window in seq and wall time.
4. `scripts/gnss/viscap.py watch --tags ,_e5a,_b2a` — `passed` climbs ~24/s per gate,
   `window_dropped` stays 0. Judge in the first minute; do not soak. `viscap.py disarm` aborts
   everywhere.
5. After `end_seq` the gates disarm themselves. Files: `ls -la /tmp/gnss/viscap/` on each node
   (owned root — the node service's user). Copy to
   `/home/kvand/gnss/fixtures/viscap_<date>/<node>/`, snapshot the six generated configs plus
   `chord_gnss_node.yaml` into `<dir>/configs/`, write the sidecar, decode:

       viscap_metadata.py /home/kvand/gnss/fixtures/viscap_<date>
       viscap_read.py /home/kvand/gnss/fixtures/viscap_<date>/cx43 --node cx43 --gpu 0 \
           --config /home/kvand/gnss/fixtures/viscap_<date>/configs/chord_gnss_cx43_multi.yaml \
           --node-yaml /home/kvand/gnss/fixtures/viscap_<date>/configs/chord_gnss_node.yaml \
           --to-h5 cx43_gnss0.h5

6. Sanity on the product: `vis_aa` diagonal real and positive on the 32 live elements;
   record `winstart` steps of 2048 hops inside a frame, 8192 across; the same `seq0` span
   on all six nodes.

## One pass per GPU (`--n2-dual-merged`, 2026-10-06)

With the manifest key `n2-dual-merged: true` the generator folds every path-B chain on a GPU
into one `gnss<g>_n2dual` process: the injectors write one compact synth array
(`gnss<g>_synth`, one slice per comb channel, 128 lanes per chain at its own offset: L5 0,
E5a 128, B2a 256 on 1176 MHz; E5b 0, B2b 128; B3I 0, E6 128; L2C 0), one `cudaCorrelatorDual`
(`num_synth` 384, `gnss_synth_compact`) correlates the (N+M)² over the union of the combs, and a
gather per chain (`gnss_gathers`) hands each assembler the frame it always had. The capture is
then ONE gather per GPU, `gnss<g>_captiles`: for the captured chains' channels (the 1176 comb),
every synth row they occupy (384 lanes), the AA block once and, with `--vis-capture-bb`, the BB
triangle over all of them -- including the replica × replica blocks BETWEEN chains. Per record
and channel that is 48 mixed + 3 AA + 300 BB = 351 tiles (20.1 MB per frame per GPU).

On disk: `<node>_gnss<g>_capvistiles_NNNNNNN.raw` plus one ctl series per captured chain with
the usual names (`<node>_gnss<g>_visctl`, `..._e5a_visctl`, `..._b2a_visctl`), gated in lockstep
by one `gnss<g>_viscap_gate` (`viscap.py --gates gnss0_viscap_gate,gnss1_viscap_gate`, the
default). `viscap_read.py --tag <chain>` decodes the capture frame against that chain's ctl:
lane = 384-lane index, this chain's slots at `lane_base + 4·slot + row`; `vis_bb` is the full
384 × 384 block, `vis_aa` the N² once. Gates: `n2dualtest` [7]-[9] (compact addressing, the
512-station launch, each chain's rows bitwise equal to a one-chain launch) and the cf06 smoke
in `fixtures/viscap_plan_20261005/` (`make_smoke.py`, `smoke_check.py`: seeded lanes light up
only in their own chain's rows).

## Not done / decisions left open

- The replica × replica blocks between chains (L5 × E5a, L5 × B2a, E5a × B2a) are not computed:
  each chain has its own synth buffer and correlator pass. They arrive when the chains share
  one (N+M)² pass per GPU (the "stage 1" merge of 2026-10-06), which also gives one N² per GPU
  instead of one per captured chain.
- Cadence below 10.49 ms: chain-wide change, not offered.
- `scripts/gnss/j2_chain_equiv.py` (manual j2 bridge gate) fails on the committed configs
  already (`n2sink` lost `cpu_affinity` when it became `dropAllFrames`); the viscap vars are
  wired through it (`viscap` in the vars fragment and template) but the gate itself needs
  repair before it can prove anything.
