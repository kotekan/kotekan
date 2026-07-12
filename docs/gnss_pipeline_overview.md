# GNSS tri-constellation pipeline — block-level overview (2026-07-11)

One airspy tune (1575.42 MHz, 20 MSPS real, IF = Fs/4 = 5 MHz) carries **GPS L1 C/A,
Galileo E1-C and BeiDou-3 B1C** simultaneously. Reference config: `config/live_l1_dual20.yaml`
(launch: `KOTEKAN=./build_cuda/kotekan/kotekan TRK=track CFG=config/live_l1_dual20.yaml
LAT=.. LON=.. ALT=.. ./config/run_live.sh`).

## Dataflow

```
 airspy (USB, 20 MSPS real, bias-T)                                  [CPU]
    │ input_buf (i16, GnssChanMetadata sample_seq stamped at source)
    ▼
 fengine: fftwEngine r2c PFB, N=10 chans x 1 MHz                     [CPU, ~10% core]
    │ chan_buf ──────────────┬──────────────┬─────────────┐   PROTOTYPE-ONLY stage:
    │ (channelized voltage,  │              │             │   CHORD's FPGA delivers
    ▼                        ▼              ▼             │   channelized voltage here
 valve (drop-tolerant     search        gal_search    bds_search     [CPU, ~1 core each]
 decoupler: airspy can    GPS L1CA      GAL_E1C       BDS_B1C_P
 never block)             1 ms windows  4 ms          10 ms, step 50 Hz
    │ chan_buf2            blind/hinted channelized acquire (FFT corr + incoherent stack
    │                      + refine); require_hint = scan only broker-hinted visible PRNs
    ├──────────────┬──────────────┐
    ▼              ▼              ▼
 track          gal_track      bds_track      three cudaProcess GPU chains       [GPU]
 (cudaProcess)  (cudaProcess)  (cudaProcess)  each: cudaInputData → cudaGnssTrack
    │              │              │           → cudaOutputData. cudaGnssTrack:
    │              │              │           * internal chan-major device RING
    │              │              │             (k x hops_per_record → windows never
    │              │              │             straddle; gaps zero-filled; allocs
    │              │              │             NAMESPACED per stage)
    │              │              │           * host pass-1 control per record: seed
    │              │              │             extrapolation + QUADRATIC code-Doppler FF
    │              │              │             + code-currency translation into pinned
    │              │              │             f_ref + fence (fll_reacq scales w/ record)
    │              │              │           * ONE batched E/P/L despread launch/record
    │              │              │             (mixed-precision fused replica+MAC kernel,
    │              │              │             grid-stride hops: 1000/4000/10000-hop recs)
    │ epl_buf      │ gal_epl_buf  │ bds_epl_buf   (control block + raw per-chan corr)
    ▼              ▼              ▼
 assemble       gal_assemble   bds_assemble   GnssGpuRecordAssemble               [CPU]
    │              │              │           pass-2: covering-mask channel sum, carrier
    │ rec_buf      │ gal_rec_buf  │ bds_rec_buf   NCO integrate/derotate, gnssRecord floats
    ▼              ▼              ▼
 combiner       gal_combiner   bds_combiner   GnssCoherentCombiner                [CPU]
    │ K=1000        K=250          K=100      incoherent moments (amp/unbiased/amp_snr),
    │ navwipe 20    CS25 wipe      B1C wipe   deep coherent integration via overlay wipe
    │ (GPS nav      (shared 25-    (per-PRN   (GPS: estimated nav bits; E1C/B1C: KNOWN
    │  bits est.)    chip)          1800-chip) secondary codes, alignment searched),
    │              │              │           auto-coherence ladder, carrier-resid slope fit
    ▼              ▼              ▼
 record         gal_record     bds_record     rawFileWrite → /tmp/gpswipe/*_level_*.raw
 + REST get_status per combiner → viewer, status loggers (status_log[_gal|_bds].jsonl)
```

## Control plane (all CPU, pure REST)

Three independent `gps_distributed_broker.py` instances (one per constellation; run_live
auto-launches off the `gal_track:`/`bds_track:` stage names):
* almanac (skyfield + Celestrak TLE per GROUP; per-group cache files) → visible PRNs +
  predicted Doppler/rate → `set_doppler_hints` (narrowed search) + seeds
* per-PRN cp-vs-hop slope fit **in the current seed's Doppler currency** (the code-currency
  family fix) → seed (cp0, rate, ref_hop, doppler, doppler_rate)
* HOLD-ON-LOCK: freezes the whole seed tuple once amp_snr ≥ 8; DLL (E/L disc) trims sub-chip;
  release on amp collapse or Doppler staleness > 0.1 cycle/record (auto-scaled per signal)
* shared carrier loop: combiner's phase-slope residual → integrator → `carrier_trim_hz`
* l−a (LO-vs-ADC clock) pooled per constellation → persisted ppm files (cross-checkable)

Seeds POST to the stage-named endpoint (`/track/set_seeds` etc.) served by the
`cudaGnssTrack` command state. The **aggregator** role (cross-channel consensus) lives in
the channelized search itself at this scale; on CHORD the same brokers front per-node combs.

## Code generation (all CPU, at stage construction; tables → GPU once)

| Constellation | Source file | Method |
|---|---|---|
| GPS L1 C/A | `gpsCACode` | G1/G2 LFSR (IS-GPS-200 taps) |
| GPS L1C-P / L2C / L5 | `gpsL1CCode` / `gpsL2CCode` / `gpsL5Code` | Weil / LFSR / XA·XB |
| Galileo E1-B/C | `galileoE1Code` | ICD Annex C **memory code** hex tables + CS25 |
| BeiDou B1C pilot | `beidouB1CCode` | Legendre-10243 **Weil** codes + per-PRN 1800-chip secondary |

`ChannelizedReplicaBank` turns component codes into the **expanded combined stream**
(TDM zero-stuffing for L2C; BOC(m,m) → 2m alternating-sign slots/chip — this is what gives
the per-chip hoprate path AND the GPU kernel BOC support). All consumers (per-sample
channels(), hoprate, GPU Phi tables) read the same expanded table; cp API stays in
component chips.

## CPU/GPU split (GB10 prototype = A40 x-engine stand-in)

GPU: the despread bulk compute only (replica synthesis fused with correlate; ~300k
PRN-despreads/s mixed-precision). CPU: F-engine (prototype-only), searches (F2 will move
acquisition onto the device ring), assemblers, combiners, brokers, viewer. Records/status
never touch the GPU.

## Known open items
See `multi-constellation-buildout` memory + §6 of `gnss_gpu_migration.md`: F2 (GPU search),
F3 (peel), cross-estimator consistency audit (deep vs amp_snr scales differ suspiciously,
esp. E/C — possible residual decoherence), per-sat overlay-alignment stability (C22-type:
strong amp, dead deep), viewer polish, BDS-2 exclusion by name filter.
