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
   airspyInput ─► input_buf ─► fftwEngine ─► chan_buf ([hop][N] cf, sample_seq)
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
| `fftwEngine` | `input_buf` → `chan_buf` | F-engine: r2c polyphase filterbank, 2N real samples/hop → N complex channels. Propagates `GnssChanMetadata.sample_seq` if the input carries it (else derives its own) |
| `GnssSubbandSplit` | `chan_buf` → `ch_00…` | (distributed only) demux `[hop][N]` into one buffer per channel |

`sample_seq` is the linchpin: the absolute "sample 0" reference that makes every
reported code phase node-independent (a seed found on one node is valid on
another) **and** gives the tracker a true capture clock. **`airspyInput` stamps it**
(when its output buffer has a GNSS pool) and folds in `dropped_samples`, so a USB
drop becomes a *known gap* in `sample_seq`, not silent counter divergence — the
absolute-hop referencing in the search/tracker (and the search's zero-filled snapshot)
then handle the gap cleanly. Propagated through split / gather / valve. (Generic name:
the airspy has no FPGA; CHORD's separate `chordMetadata.fpga_seq` is untouched — a thin
entry adapter would copy it into `sample_seq`, dropped UDP frames showing as the same
kind of gap.)

## Live receiver chain (the prototype we run)

A complete GNSS receiver on one airspy. The **search** reads `chan_buf` directly
(async, never backpressures); a **valve** feeds the lossless tracker so the airspy
capture stays drop-free. (Drops are no longer catastrophic — `airspyInput` folds them
into `sample_seq` as known gaps, and the search zero-fills its snapshot at the true
`sample_seq` offset — but the valve still keeps the common case clean.) The default
`live_l1.yaml` runs the *distributed* arrangement of this same chain (split → per-channel
send/recv → gather/search; per-channel track → combiner); the single-node form here is
`live_intgn.yaml`.

| stage | in → out | role |
|---|---|---|
| `GnssChannelizedSearch` | `chan_buf` → REST | coarse acquire + exact ±1-hop **refine** over the covering channels; async worker, holds the latest detection. Snapshot is **drop-tolerant** (places each frame at its `sample_seq` offset, zero-fills gaps). **Almanac-narrowed** via `POST set_doppler_hints`: per-PRN, scan only `doppler ± margin` instead of the blind grid (blind fallback when no hint). Publishes `(PRN, dop, refined cp, ref_hop, snr)` |
| `Valve` | `chan_buf` → `chan_buf2` | drops a frame if the tracker is behind — decouples the lossless track→record chain from the realtime airspy (keeps capture drop-free) |
| `GnssChannelizedTracker` | `chan_buf2` → `rec_buf` | despread each active PRN. **Code:** seed cp extrapolated by the broker's slope, then a ±`pullin_chips` DLL search locks the peak. **Carrier:** a per-PRN **FLL** (NCO derotation, `fll_gain`) drives the residual to ~Hz so coherence holds for seconds. Emits per-PRN `(G, E)`, derotated |
| `GnssCoherentCombiner` | `rec_NN` → `out_buf` | sum `G_c`,`E_c` across covering channels → `A = ΣG/ΣE`; accumulate `integration_length` records (incoherent `√⟨|A|²⟩` + coherent `|⟨A⟩|` + nav-wiped deep `|A|`). `integration_mode: rolling` = an EMA (no nav-bit cap on the incoherent `|A|`), so a weak sat climbs out continuously. Publishes per-PRN `|A|` over REST |
| `GnssBeamCube` | `rec_NN` → `beam_buf` | the combiner *without* the cross-channel sum: per-channel deep `|A_c|` = **beam(time, frequency)**. Reuses the same tracker records (no extra correlation); matters across L5's ~10 MHz. Reader: `gps_beam_watch.py` |
| `rawFileWrite` | `out_buf` / `beam_buf` → disk | per-PRN records (and the beam cube) to `base_dir` (run scripts create it) |

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

`python/scripts/gnss/gps_distributed_broker.py` — the only non-kotekan piece, touches no
sample buffers:
- polls `…/get_detections` → best-SNR consensus per PRN;
- **almanac assist** (`--almanac` + `--lat/--lon`): `gps_beamtrack.predict_dopplers`
  gates to visible sats and seeds the precise **predicted Doppler** plus a common
  clock-frequency bias, **EMA-smoothed** (`--bias-alpha`) so the coarse-grid jitter
  doesn't wander the seed;
- **narrowed search** (`--narrow-search`): pushes each predicted Doppler to the search
  (`set_doppler_hints`) so it scans a tight window instead of the blind grid — **wide**
  until the common clock-freq bias is solved off the first sat, then **narrow** (the
  forward-compatible CHORD pattern: a disciplined clock collapses the search to a
  code-phase pin);
- fits the per-PRN **cp slope** (`cp0` vs `ref_hop`) and sends `code_phase_rate` + anchor;
- polls `combiner/get_status` for `|A|` → drops a PRN on sustained low amplitude;
- `POST …/set_seeds` to the tracker(s). Re-seeds every ~0.2 s.

`run_live.sh` patches a temp config with the **currently-visible PRNs**
(`gps_visible_prns.py`) each launch — the search list is otherwise static and the
constellation rotates ~half an orbit in ~8 h.

## Offline analysis tools (the receiver's back end, pre-port)

| tool | role |
|---|---|
| `python/scripts/gnss/gps_intgn_check.py` | reconstructs coherent vs incoherent integration vs K from the recorded per-record `A`; noise-floor (1/√K vs flat), carrier-stability fit, perfect-FLL oracle, **nav-bit wipe** (estimate bits → wipe → integrate past 20 ms), and the on-sky **nav decode**. Auto-detects `n_prn` |
| `python/scripts/gnss/gps_nav_decode.py` | IS-GPS-200 L1 C/A decoder: (1+2) Hamming parity, TLM-preamble frame sync, HOW (TOW + subframe ID). Validated on-sky (5 subframes, marching TOW) |

Live readouts + site diagnostics:

| tool | role |
|---|---|
| `gps_mono_watch.py` | live per-PRN SNR from the monolithic ground truth (`run_mono.sh`) |
| `gps_beam_watch.py` | live **beam(time, frequency)** from the beam cube — ASCII `\|A\|`-vs-channel per PRN |
| `gps_gain_sweep.py` | sweep the airspy LO vs ADC rms — tuner gain-slope vs in-band RFI |
| `gps_pulse_check.py` | time-domain RFI triage of a raw `airspy_rx` capture: pulse/CW/broadband, PRF, ADC-level/quantization, in-band spectrum shape, rms-vs-time |

The nav-bit wipe + long coherent accumulate now live **in the combiner**
(`navwipe_bit_records`, `integration_mode: rolling`); the next pipeline step is the
**voltage peel** (framework note, step 4).

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
| `chan_buf` / `chan_buf2` | `[hop][N]` cfloat | `GnssChanMetadata` (sample_seq) |
| `ch_NN` / `rc_NN` / `gather_buf` | `[hop](×Ncov)` cfloat | `GnssChanMetadata` |
| `rec_NN` / `out_buf` | 11 floats/PRN | none |
| `beam_buf` | `4 + Ncov + 2` floats/PRN | none |

Record floats: `[0]`PRN `[1]`Doppler (tracker: **f_ref+f_track**, the FLL-tracked
carrier) `[2]`code phase (pulled-in) `[3]`G.re `[4]`G.im `[5]`E `[6]`n_chan_used
`[9,10]`UTC (double; **capture time** `capture_utc0 + window_start/Fs` when
`capture_utc0>0`, else wall-clock at emit). Combiner reuses the slots: `[3]`=`√⟨|A|²⟩`
incoherent, `[4,5]`=`⟨A⟩`, `[6]`=`|⟨A⟩|` coherent, `[7]`=n_chan, `[8]`=nav-wiped deep `|A|`.
Beam cube: `[0]`PRN `[1]`dop `[2]`cp `[3]`channel0, then `Ncov` per-channel `|A_c|`, then
UTC (double) at slot `4+Ncov`.

## Configs

All three live bands come from one signal-parameterized generator
(`GNSS_SIGNAL=L1CA|L2C_CM|L5_Q python3 config/gen_live_config.py`); `N` is per-signal so
`fft_len = 2N` divides one code period (the integer-period invariant — see below).

| config | arrangement |
|---|---|
| `live_l1.yaml` | **the live config** (run_live.sh default), L1 C/A, 5 MSPS, N=12 (~208 kHz channels), **distributed CHORD-mirror** (per-channel track + bufferSend/Recv → gather/search + combiner + beam cube), **valve** after the F-engine (airspy-safe), FLL on, GPS-only browser viewer on :8080 |
| `live_l2c.yaml` | retuned to **L2C** (1227.6 MHz, `GPS_L2C_CM`). N=10 (7 covering channels), fine `doppler_step:50`. **N=10 (not 12) is required**: at N=12 one 20 ms code period is 4166.67 hops, so the incoherent window wasn't an integer number of periods → the peak smeared and L2C detected nothing. CM is half the L2C power + time-multiplexed with CL (the replica zero-stuffs the combined code) |
| `live_l5.yaml` | retuned to **L5** (1176.45 MHz, `GPS_L5_Q` pilot). **Wide front end**: `sample_bw:10` → 20 MSPS (captures ~half the ~20 MHz main lobe). N=10, all 10 channels covering. Dataless pilot → long coherent integration; ~10 MHz spans real beam variation (use the beam cube) |
| `live_mono*.yaml` | **monolithic ground truth** (`gen_mono_config.py`, `run_mono.sh`): airspy → valve → `GpsReplicaCorrelator` (full-band FFT search/track) → records. The "what SNR should we get" reference, independent of the channelized path. L1 / L2C / L5 variants |
| `live_l1_wipe.yaml` | live chain + combiner nav-bit wipe (`navwipe_bit_records`, deep `\|A\|` past the 20 ms bit) — the navwipe / deep-integration demo |
| `live_intgn.yaml` | single-node live chain, `integration_length:1` (record every raw `A` for `gps_intgn_check.py`). The deep-integration test bed |

## Practical notes

- **Valve, not backpressure.** The lossless track→combine→disk chain would stall the
  F-engine → airspy USB drops. The valve drops `chan_buf2` frames instead, keeping the
  capture clean. Drops are also now *survivable* (folded into `sample_seq` as known gaps,
  not silent counter divergence), so the valve is about staying clean in the common case,
  not avoiding catastrophe. Heavier per-PRN work just raises the drop rate; capture holds.
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
