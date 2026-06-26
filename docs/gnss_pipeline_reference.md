# Channelized GNSS Pipeline — Stage & Buffer Reference

The pipeline as built on the airspy prototype, mirroring the CHORD F/X layout.
Three signal planes (front end → search → track) plus a Python control plane,
all hanging off one F-engine. Companion to `gnss_chord_framework.md` (the CHORD
integration) and `gnss_distributed_search_scaling.md` (why search merges
centrally and what crosses the wire).

```
                         ┌─────────── FRONT END ───────────┐
   rawFileRead/airspyInput ──► input_buf ──► fftwEngine ──► chan_buf
   (raw int16 samples)        (real volts)   (r2c PFB)      ([hop][N] cf, fpga_seq)
                                                  │
                                       GnssSubbandSplit
                                                  │
                          ┌───────────────────────┼────────────────────────┐
                   ch_00 … ch_(N-1)   (one channel each, [hop] cf, fpga_seq)
                          │                                                  │
          ┌───────── SEARCH PLANE ──────────┐              ┌──── TRACK PLANE ────┐
          ▼ (variant A: combed P_c)         ▼ (variant B)  ▼                     │
   GnssChannelCorrelator ×N        bufferSend ×Ncov   GnssChannelizedTracker ×N   │
        │ pc_NN                        │ (TCP)            │ rec_NN                 │
   GnssSearchAggregator          bufferRecv ×Ncov    GnssCoherentCombiner         │
        │                             │ rc_NN             │ out_buf                │
        │                        GnssChannelGather        ▼                       │
        │                             │ gather_buf    (ΣG)/(ΣE) = |A|             │
        │                        GnssChannelizedSearch     │                      │
        ▼                             ▼                     ▼                      │
   REST /get_detections ◄────────────┘              REST /get_status              │
        │ (PRN, dop, refined cp)                          │ (|A| per PRN)         │
        └───────────► broker.py ──── POST /set_seeds ─────┴──────────────────────┘
```

## Front end (shared)

| stage | in → out | role |
|---|---|---|
| `rawFileRead` / `airspyInput` | → `input_buf` | source: raw real samples (int16) |
| `fftwEngine` | `input_buf` → `chan_buf` | F-engine: r2c polyphase filterbank, 2N real samples/hop → N complex channels. **Stamps `GnssChanMetadata.fpga_seq`** = absolute index of the frame's first sample |
| `GnssSubbandSplit` | `chan_buf` → `ch_00…` | demux the `[hop][N]` block into one buffer per channel (the CHORD "comb"); copies `fpga_seq` into each |

`fpga_seq` is the linchpin: the absolute "sample 0" reference that makes every
reported code phase node-independent, so a seed found on one node is valid on
another. Stamped once at the F-engine, propagated through split / gather.

## Search plane — two realizations of the *same* search

Both find `(PRN, Doppler, code phase)` with a coarse acquire then an exact ±1-hop
despread **refine** (the refine is mandatory — the coarse cp is 1–3 chips off the
true peak, too coarse to lock a tracker). They differ only in *what crosses the
wire*.

**Variant A — combed P_c** (`search_dist.yaml`; local validation scaffold):

| stage | in → out | role |
|---|---|---|
| `GnssChannelCorrelator` ×N | `ch_NN` → `pc_NN` | per-channel coarse FFT correlation → the `P_c` surface; covering channels also append their window-0 raw (for the refine) |
| `GnssSearchAggregator` | `pc_00…` → REST | gather all `P_c`, cross-channel fine-lag combine + peak (coarse), then central refine on the appended raws |

This ships the **`P_c` surface**, which is `n_prn·nd` *larger* than the raw data
— fine in one kotekan (local RAM), prohibitive cross-node (see the scaling note).

**Variant B — unified raw gather** (`search_unified.yaml`; the cross-node keeper):

| stage | in → out | role |
|---|---|---|
| `bufferSend` ×Ncov | `ch_NN` → TCP | ship covering channels' raw voltage (the *small* data) |
| `bufferRecv` ×Ncov | TCP → `rc_NN` | receive on the search node (deserializes `GnssChanMetadata`) |
| `GnssChannelGather` | `rc_…` → `gather_buf` | lock-step interleave the covering channels → one `[hop][Ncov]` buffer |
| `GnssChannelizedSearch` | `gather_buf` → REST | the monolithic coarse+refine search, unchanged, over the gathered channels |

Covering set = channels whose passband overlaps `carrier ± (chip_rate + Doppler
margin)`. For the L1/20 MSPS validation that's 17 channels (12–28).

## Track plane (measurement → coherent combine)

| stage | in → out | role |
|---|---|---|
| `GnssChannelizedTracker` ×N | `ch_NN` → `rec_NN` | despread this channel against the seeded replica at `(cp, dop)`; emit a per-PRN record |
| `GnssCoherentCombiner` | `rec_00…` → `out_buf` | sum `G_c`, `E_c` across channels → `A = (ΣG)/(ΣE)`, full-band amplitude; publish per-PRN `|A|` over REST |

`A = ΣG/ΣE` is the coherent reassembly; it sums only the channels that show up,
so it degrades gracefully if a node drops (unbiased, auto-renormalized).

## Control plane

`python/scripts/gps_distributed_broker.py` — the only non-kotekan piece, touches
no sample buffers:
- polls `…/get_detections` → best-SNR consensus per PRN;
- polls `combiner/get_status` for `|A|` → drops a PRN on sustained low amplitude;
- `POST …/track_NN/set_seeds` with the refined `(cp, dop)` to every tracker.

Re-seeds every ~0.2 s to stay ahead of code-Doppler drift. The loop *locks*
(combiner `|A|` 1.0 → 3.5–5.1) only once the **refined** cp is flowing.

## Buffer / metadata summary

| buffer | layout | metadata |
|---|---|---|
| `input_buf` | raw real int16 | none |
| `chan_buf` | `[hop][N]` cfloat | `GnssChanMetadata` (fpga_seq) |
| `ch_NN` / `rc_NN` | `[hop]` cfloat (one channel) | `GnssChanMetadata` |
| `pc_NN` | `CorrFrameHeader` + `P_c[prn][win][d][q]` + raw block | none |
| `gather_buf` | `[hop][Ncov]` cfloat | `GnssChanMetadata` |
| `rec_NN` / `out_buf` | 11 floats/PRN: `[0]`PRN `[1]`dop `[2]`cp `[3]`G.re `[4]`G.im `[5]`E `[6]`n_chan `[9,10]`UTC | none |

Everything channelized rides the `gnss_pool` (`GnssChanMetadata`, an 8-byte
`fpga_seq`); `input_buf` and the record buffers carry no metadata.

## Configs

| config | arrangement |
|---|---|
| `comb_single.yaml` | front end + track plane (config-seeded); coherent reassembly == full band |
| `search_dist.yaml` | front end + search variant A |
| `search_unified.yaml` | front end + search variant B (cross-node keeper) |
| `loop_single.yaml` | front end + both planes, broker-driven: the full closed loop |

## Practical notes

- **Gather size is set by signal bandwidth, not the array.** CHORD channels are
  fixed at 1600 MHz / 8192 ≈ 195.3 kHz. A carrier spans `2·chip_rate / 195 kHz`
  channels: L2C ≈ 5–10, **L5 ≈ 100** (10.23 Mcps). Constellations on the 1176
  cluster (GPS L5 / Galileo E5a / BeiDou B2a) **share** channels, so they add PRNs
  (a search dimension), not channels — the channel count is bounded (~100), but
  L5 is ~6× the 17-channel demo, which is what motivates a multiplexed receive.
- **Gather bandwidth is negligible.** Per channel at 195.3 kHz: ~1.5 Mbps (4-bit
  complex) to ~12.5 Mbps (cfloat). ~100 channels for L5 → 0.15–1.25 Gbps *peak*,
  and search is duty-cycled (snapshot, then search slower than real time), so the
  average is far lower. The expensive thing was never the raw — it's the `P_c`
  surface, which is why we don't ship it.
- **`GnssSubbandSplit`/`GnssChannelGather` vs generic `BufferSplit`/`bufferMerge`:**
  the generic stages route *whole frames* (round-robin in the stream dimension);
  the GNSS pair reshapes the *channel axis within* each `[hop][N]` frame (strided
  column extract / interleave). Different operation — not subclassable from the
  frame-routing stages.
- **`end_interrupt: false`** when a send/recv transport is in the pipeline, or the
  instance shuts down the instant the input file ends and in-flight data is lost.
