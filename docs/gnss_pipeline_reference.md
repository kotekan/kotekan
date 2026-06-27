# Channelized GNSS Pipeline — Stage & Buffer Reference

Two arrangements share one F-engine and one DSP core (`lib/stages/Gnss*` /
`gnss*`):

- **Live receiver chain** (what runs on the airspy prototype, `config/live_intgn.yaml`):
  the full single-node GNSS receiver — acquire → code+carrier track → coherent
  combine → record, broker-driven. This is the proven, on-sky path: search → code
  lock → carrier FLL → nav-message decode → *seconds* of coherent integration.
- **Distributed (CHORD-scale) realization**: the same DSP with `GnssSubbandSplit`
  / `bufferSend`+`bufferRecv` / `GnssChannelGather` so a carrier's covering
  channels can live on different X-engine nodes. Validated as a scaffold; the data
  volumes and what crosses the wire are in `gnss_distributed_search_scaling.md`,
  the CHORD F-X integration in `gnss_chord_framework.md`.

```
   airspyInput ─► input_buf ─► fftwEngine ─► chan_buf ([hop][N] cf, fpga_seq)
   (real int16)   (volts)     (r2c PFB)        │
                                   ┌────────────┴───────────────┐
                                   │ (async, lossy)             │ (lossless)
                          GnssChannelizedSearch          Valve ─► chan_buf2
                            coarse + refine                      │
                                   │ REST get_detections   GnssChannelizedTracker
                                   │                        cp slope+pull-in (code)
                                   │                        carrier FLL/NCO (phase)
                                   │                              │ rec_buf
                                   │                        GnssCoherentCombiner
                                   ▼                              │ out_buf (|A|, <A>)
                          gps_distributed_broker.py ──► rawFileWrite ─► /tmp/gpsintgn
                          almanac Doppler + EMA clock-bias              │
                          cp-slope fit, set_seeds ◄────────────────────┘ REST get_status
                                                       gps_intgn_check.py (offline):
                                                       coherence, FLL-oracle, nav-bit
                                                       wipe; gps_nav_decode.py (IS-GPS-200)
```

## Front end (shared)

| stage | in → out | role |
|---|---|---|
| `airspyInput` / `rawFileRead` | → `input_buf` | source: raw real samples (int16). Logs a WARN when the USB FIFO blocks (pipeline behind realtime) |
| `fftwEngine` | `input_buf` → `chan_buf` | F-engine: r2c polyphase filterbank, 2N real samples/hop → N complex channels. **Stamps `GnssChanMetadata.fpga_seq`** = absolute index of the frame's first sample |
| `GnssSubbandSplit` | `chan_buf` → `ch_00…` | (distributed only) demux `[hop][N]` into one buffer per channel |

`fpga_seq` is the linchpin: the absolute "sample 0" reference that makes every
reported code phase node-independent (a seed found on one node is valid on
another) **and** gives the tracker a true capture clock. Stamped once at the
F-engine, propagated through split / gather / valve.

## Live receiver chain (the prototype we run)

A complete GNSS receiver on one airspy. The **search** reads `chan_buf` directly
(async, never backpressures); a **valve** feeds the lossless tracker so the airspy
capture stays drop-free (drops would diverge `fpga_seq` → random code phase).

| stage | in → out | role |
|---|---|---|
| `GnssChannelizedSearch` | `chan_buf` → REST | coarse acquire + exact ±1-hop **refine** over the covering channels; async worker, holds the latest detection. Publishes `(PRN, dop, refined cp, ref_hop, snr)` |
| `Valve` | `chan_buf` → `chan_buf2` | drops a frame if the tracker is behind — decouples the lossless track→record chain from the realtime airspy (keeps capture drop-free) |
| `GnssChannelizedTracker` | `chan_buf2` → `rec_buf` | despread each active PRN. **Code:** seed cp extrapolated by the broker's slope, then a ±`pullin_chips` DLL search locks the peak. **Carrier:** a per-PRN **FLL** (NCO derotation, `fll_gain`) drives the residual to ~Hz so coherence holds for seconds. Emits per-PRN `(G, E)`, derotated |
| `GnssCoherentCombiner` | `rec_buf` → `out_buf` | sum `G_c`,`E_c` across covering channels → `A = ΣG/ΣE`; accumulate `integration_length` records (incoherent `√⟨|A|²⟩` + coherent `|⟨A⟩|`). Publishes per-PRN `|A|` over REST |
| `rawFileWrite` | `out_buf` → disk | per-PRN records to `base_dir` (run_live.sh creates it) |

**Code tracking = slope + pull-in.** The search's `cp0` drifts ~linearly with
capture hop (residual code-rate error: sample-clock + Doppler-grid). The broker
least-squares-fits `cp0` vs `ref_hop` and sends a slope; the tracker extrapolates
to its own hop, then the pull-in mops up the sub-chip residual. Latency/drift-robust.

**Carrier tracking = FLL/NCO.** *Design point:* the replica carrier is anchored to
the absolute sample index, so you must **not** retune it mid-stream (a 0.1 Hz step
jumps the phase by `2π·df·t_abs` ≈ cycles). Instead hold the replica at a fixed
reference Doppler `f_ref` (re-acquired from the broker seed only on loss) and run a
phase NCO: accumulate `phi = ∫2π·f_track·dt` and **derotate the despread output**.
The bit-robust discriminator `arg((A_k conj A_{k-1})²)/2 / 2π·dt` (squaring cancels
the ±1 nav-bit π flip) drives `f_track`. Closed the live ~15 Hz seed wander to ~5 Hz
(SNR-limited on a faint indoor sat; → <1 Hz on a bright sat / CHORD dishes).

## Control plane (broker)

`python/scripts/gps_distributed_broker.py` — the only non-kotekan piece, touches no
sample buffers:
- polls `…/get_detections` → best-SNR consensus per PRN;
- **almanac assist** (`--almanac` + `--lat/--lon`): `gps_beamtrack.predict_dopplers`
  gates to visible sats and seeds the precise **predicted Doppler** plus a common
  clock-frequency bias, **EMA-smoothed** (`--bias-alpha`) so the coarse-grid jitter
  doesn't wander the seed;
- fits the per-PRN **cp slope** (`cp0` vs `ref_hop`) and sends `code_phase_rate` + anchor;
- polls `combiner/get_status` for `|A|` → drops a PRN on sustained low amplitude;
- `POST …/set_seeds` to the tracker(s). Re-seeds every ~0.2 s.

`run_live.sh` patches a temp config with the **currently-visible PRNs**
(`gps_visible_prns.py`) each launch — the search list is otherwise static and the
constellation rotates ~half an orbit in ~8 h.

## Offline analysis tools (the receiver's back end, pre-port)

| tool | role |
|---|---|
| `python/scripts/gps_intgn_check.py` | reconstructs coherent vs incoherent integration vs K from the recorded per-record `A`; noise-floor (1/√K vs flat), carrier-stability fit, perfect-FLL oracle, **nav-bit wipe** (estimate bits → wipe → integrate past 20 ms), and the on-sky **nav decode**. Auto-detects `n_prn` |
| `python/scripts/gps_nav_decode.py` | IS-GPS-200 L1 C/A decoder: (1+2) Hamming parity, TLM-preamble frame sync, HOW (TOW + subframe ID). Validated on-sky (5 subframes, marching TOW) |

These run on `rawFileWrite` output. Moving the nav-bit wipe + long coherent
accumulate **into the combiner** (live) is the next pipeline step.

## Distributed (CHORD-scale) realization

The same DSP, but covering channels are split across nodes and the search remaps to
one node. Ships *correlations* for measurement, *raw covering voltage* for the
(duty-cycled) acquire — never the `P_c` surface (it is an expansion of the data; see
the scaling note).

| stage | in → out | role |
|---|---|---|
| `bufferSend`/`bufferRecv` ×Ncov | `ch_NN` ↔ TCP | ship covering channels' raw voltage to the search node |
| `GnssChannelGather` | `rc_…` → `gather_buf` | lock-step interleave covering channels → `[hop][Ncov]` |
| `GnssChannelizedSearch` | `gather_buf` → REST | the monolithic coarse+refine over the gathered channels |
| `GnssChannelizedTracker` ×N | `ch_NN` → `rec_NN` | per-subband despread; `GnssCoherentCombiner` reduces `ΣG/ΣE` across nodes |

`A = ΣG/ΣE` is the coherent reassembly: it sums only the channels that show up, so a
node dropping is just a missing term — unbiased, auto-renormalized.

> **Retired:** the *combed-P_c* variant (`GnssChannelCorrelator`→`GnssSearchAggregator`)
> shipped the `P_c` surface, which is `n_prn·nd` *larger* than the raw voltage — only
> sensible in one kotekan (local RAM). Superseded by the raw gather. See git history.

## Record / buffer layout

| buffer | layout | metadata |
|---|---|---|
| `input_buf` | raw real int16 | none |
| `chan_buf` / `chan_buf2` | `[hop][N]` cfloat | `GnssChanMetadata` (fpga_seq) |
| `ch_NN` / `rc_NN` / `gather_buf` | `[hop](×Ncov)` cfloat | `GnssChanMetadata` |
| `rec_NN` / `out_buf` | 11 floats/PRN | none |

Record floats: `[0]`PRN `[1]`Doppler (tracker: **f_ref+f_track**, the FLL-tracked
carrier) `[2]`code phase (pulled-in) `[3]`G.re `[4]`G.im `[5]`E `[6]`n_chan_used
`[9,10]`UTC (double; **capture time** `capture_utc0 + window_start/Fs` when
`capture_utc0>0`, else wall-clock at emit). Combiner reuses the slots: `[3]`=`√⟨|A|²⟩`
incoherent, `[4,5]`=`⟨A⟩`, `[6]`=`|⟨A⟩|` coherent.

## Configs

| config | arrangement |
|---|---|
| `live_intgn.yaml` | **live receiver chain**, 5 MSPS, valve, FLL on, `integration_length:1` (record every raw `A` for the offline analyzer). The deep-integration test bed |
| `live_direct.yaml` | live chain with `integration_length:10` — the lock/track config |
| `live_full.yaml` / `live_lowrate.yaml` | comb/full-rate variants + the browser power viewer |
| `comb_single` / `search_unified` / `loop_single` | distributed scaffolds (split/gather/transport) |

## Practical notes

- **Valve, not backpressure.** The lossless track→combine→disk chain would stall the
  F-engine → airspy USB drops → `fpga_seq` diverges → random code phase. The valve
  drops `chan_buf2` frames instead, keeping capture exact. Heavier per-PRN work (large
  pull-in, many PRNs) just raises the drop rate; capture stays clean.
- **Capture time, not emit time.** Set `capture_utc0>0` so records carry
  `capture_utc0 + window_start/Fs` — exact 1 ms/record with clean gaps. Emit
  wall-clock is bursty (the pipeline drains buffered frames then stalls) and breaks any
  phase-vs-time fit (the FLL discriminator, the analyzer's carrier fit).
- **Replica generation is the hot path** (regenerated per PRN per pull-in trial per
  record). The carrier is a phasor recurrence (one complex mult/sample, no per-sample
  `cos`). Code-Doppler is feed-forward in `chip_per_sample`. A cache-and-rotate scheme
  was rejected: it can't follow code-Doppler drift (worst on L5). This is the first
  thing to profile for the realtime port.
- **Gather size is set by signal bandwidth, not the array.** A carrier spans
  `2·chip_rate / channel_BW` channels: L2C ≈ 5–10, **L5 ≈ 100** (10.23 Mcps). The 1176
  cluster (GPS L5 / Galileo E5a / BeiDou B2a) shares channels, so constellations add
  PRNs (a search dimension), not channels.
- **`end_interrupt: false`** with a send/recv transport, or in-flight data is lost at
  end-of-file. **Realtime replay pacing** (`rawFileRead frame_period_us`) so a wall-clock
  broker can keep pace; a live airspy paces itself.
