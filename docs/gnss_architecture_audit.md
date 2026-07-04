# GNSS Framework Architecture Audit — 2026-07-04

Scope: the airspy/CHORD GNSS replica-correlator framework as of `kv/gps-replica-correlator`
(~4,850 lines of C++ stages, ~3,800 lines of Python control plane, 22 live configs), audited
before extending to peeling at scale and non-GPS constellations. Companion to
`gnss_chord_framework.md` (the end-goal) and `gnss_pipeline_reference.md` (the pipeline).

Verdict up front: **the organic growth converged on the right architecture.** The two big
structural bets — correlate-at-data (ship tiny records, not voltage) and a split
control/data plane (Python broker at ~Hz over REST, C++ stages at MSPS) — are exactly the
CHORD-shaped choices and both have been validated hard (bit-identical cross-node records,
coherent reassembly == full-band to 5e-7). No rewrite is warranted. There is one real
architectural refactor worth doing (R1 below), a short list of hygiene items, and a clear
GPU placement story.

---

## 1. What the organic growth got right (keep, and defend)

**K1 — Correlate-at-data.** Per-sub-band trackers emit ~44-byte records; only records cross
nodes; the combiner reassembles coherently. This matches CHORD's corner-turn reality and has
survived on-sky validation. Every new feature this framework grows should preserve the rule:
*voltage never leaves a node; decisions never happen at sample rate.*

**K2 — The absolute-anchor invariant.** Code phase is referenced to capture sample 0 and the
replica indexes chips by absolute sample (`phase = cp0 + n·chip_per_sample`, n absolute;
sample_seq counts through drops). This single convention is why distributed combining,
drop-robustness, record gap-healing (UTC-indexed wipes), and the CL time-assist all composed
without touching each other. It is the framework's load-bearing wall. Document it wherever a
new stage is written; never introduce a stage with window-relative code phase.

**K3 — SignalDescriptor + generic replica bank.** One table row per signal (carrier, chip
rate, code length, BPSK/BOC, TDM parity, pilot/data, secondary length, time-assisted) and one
replica bank that consumes it. This absorbed L1 C/A, L2C CM, L2C CL, L5 I/Q, and L1C-P
without structural change — the abstraction is earning its keep and is the correct extension
point for Galileo/BeiDou (see §4).

**K4 — Control plane in Python, data plane in C++.** The broker owns consensus seeds, the
almanac, both clock-bias estimators (carrier bias + code-side l−a), Doppler-rate
feed-forward, narrowed-search hints, coast/drop policy, and now the CL time-assist. All of it
runs at 4 Hz over REST and none of it belongs at sample rate. This is also the CHORD
operations shape (central orchestrator, per-node engines).

**K5 — Single-source-of-truth headers.** `gnssSignal.hpp` and `clockProfile.hpp` are parsed
by run_live.sh/Python rather than duplicated. Mildly fragile (regex over C++), entirely
worth it; keep the pattern, keep the tables machine-parsable.

**K6 — clock_profile two-axis model + measured coherence.** accuracy_ppm sizes the search,
coherence_s bounds integration, `auto` is the default, and the combiner's auto-coherence
ladder *measures* the usable depth live (`coherence_s` in get_status). Framework-grade: it
made the L2C re-investigation a config change.

**K7 — Async, lossy acquisition.** Search decoupled behind the valve so it can never
backpressure the stream. Also exactly the property that makes acquisition a GPU candidate
(§3): it tolerates latency and batching by design.

---

## 2. Debt and the recommended refactor (prioritized)

### R1 — Dumb correlators, slow loops (the one real refactor)

The tracker currently owns four interacting code-alignment policies: broker seed-following,
per-record max-power pull-in, hold_lock freeze, and the Doppler-rate feed-forward. One
session's evidence of the cost: two independent false-hold-on-noise bugs (arm statistic below
the noise floor; energy-unfair |A| arming), the GLRT ranking fix, and the unresolved
pull-in-vs-coherence tension (±1-chip re-picks find the peak but scramble deep phase;
pull-in 0 holds phase but sits off-peak) that currently blocks both the L1 nav-bit jitter
and the CL deep lock.

This is the classic receiver-architecture lesson: **correlators should be dumb and fast;
loops should close slowly, in one place.**

Proposal:
- The tracker emits **Early/Prompt/Late** prompt-spaced complex correlators per record
  (record grows from 11 to 15 floats; E/L spacing ~0.5 chip from the same replica bank —
  two extra despreads per record, no decisions).
- The tracker's only mode is: *despread exactly at the commanded state* {cp0, cp_rate,
  f, f_rate}, absolute-anchored (K2). Pull-in and hold are deleted from the hot path.
- The **broker** closes the code loop (DLL discriminator from shipped E/P/L, smoothed, at
  ~Hz) and the fine carrier if wanted, commanding state updates through the existing
  set_seeds path. The l−a/cp_rate machinery it already owns *is* most of this loop.

Payoffs: kills the entire false-arm bug class; phase-continuous despreads by construction
(deep integration stops fighting the tracker); one authoritative per-PRN state machine
(ACQUIRE→TRACK→COAST) in the broker instead of heuristics spread across two languages; and
the GPU story gets simpler because the batched despread kernel carries no decision logic
(§3). This refactor is the *correct* fix for both open threads (L1 nav-bit dropouts, CL
seed-fraction tuning) — preferable to further parameter tuning of pull-in/hold.

Cost estimate: a few days including re-validating L1 deep-wipe and the distributed
bit-identical tests. Do it before peeling-at-scale and before new constellations.

### R2 — Make the config generator authoritative

22 `live_*.yaml`, mostly near-duplicates; `gen_live_config.py` exists but several configs
are hand-edited descendants (live_l2c_cl was sed-derived). Every cross-cutting change this
session (CORS, clock_profile, Doppler trim, affinity) was a 20-file mechanical sweep.
Action: fold the accumulated deltas back into the generator (CL = `GNSS_SIGNAL=L2C_CL
SEARCH_SIGNAL=L2C_CM`), regenerate all live configs from it, and treat hand-editing a
generated config as a bug. Half a day; removes the biggest ongoing drift risk.

### R3 — Delete dead stages

`GnssChannelizedCorrelator` (417 lines) + `gnssChannelizedAcquire` (243) +
`GnssRecordCollector` are the retired variant-A path: referenced by zero configs. Git
history preserves them. Delete (~700 lines, 30 minutes, purely subtractive).

### R4 — Named record schema

The record is 11 magic float slots duplicated informally across tracker, combiner, Python
analyzers, and rawFileWrite's 4-byte-header framing (a mis-parse cost real debugging time
this session). One `gnssRecord.hpp` with named constants, parsed by the Python tools the
same way gnssSignal.hpp is. Do together with R1 (which changes the record anyway).

### R5 — Broker internal structure (opportunistic)

575 lines, all in one loop with per-PRN state scattered across five dicts. It has earned a
`PrnState` dataclass and three module seams (almanac/geometry, clock model, assists). Not
urgent; do when R1 touches it anyway.

### R6 — Table-driven secondary overlays

The combiner special-cases overlay names ("L5_NH20", "L5_NH10", "L1CO", "COHERENT") and the
L1CO cache. Galileo E1C (CS25 per-PRN secondary) and BeiDou B1C will multiply the cases.
Move overlay sequences behind the code-generator/descriptor interface so the combiner asks
the signal, not a string table. Small; prerequisite for new constellations, not for peeling.

### Non-refactors (explicitly rejected)

- Collapsing stages into fewer/bigger ones: the stage granularity mirrors CHORD deployment
  boundaries; keep it.
- Moving control logic into C++: the broker's cadence is ~Hz; Python is the right tool.
- A general rewrite: everything above is incremental; the validated invariants (K1–K7) are
  the asset.
- GLONASS FDMA support: per-satellite carriers break the one-carrier-per-signal descriptor
  assumption. It is the one genuinely structural extension on the horizon — defer until a
  concrete need; CDMA constellations (Galileo, BeiDou, QZSS) need only R6.

---

## 3. GPU placement

Principle: GPU belongs where **data rate × parallel axes** is large and the work is
latency-tolerant. Records are tiny — nothing downstream of record production ever needs a
GPU. That immediately sorts the pipeline:

| Stage | Axes at CHORD scale | GPU? | Where |
|---|---|---|---|
| Voltage peel | dish × channel × PRN at full rate | **Yes — required** | Sub-band node, in-stream |
| Tracker despread (+replica) | dish × channel × PRN × record | **Yes — the bulk compute** | Sub-band node |
| Acquisition/search | PRN × Doppler × cp × window | **Yes — best FLOPs/effort** | Either (see below) |
| Channelizer | — | Already CHORD's F-engine | Upstream |
| Combiner / deep integration | records only | No | Wherever records land (CPU) |
| Broker / loops (R1) | ~Hz | No | CPU, anywhere |

**Answering the sub-band question directly:** the two stages that *must* be sub-band-local
are the ones touching full-rate voltage — **peel** (it edits the stream in place) and
**despread** (it consumes the stream). That constraint is not a limitation, it is the
correlate-at-data design (K1) restated for GPUs: CHORD's node GPUs see exactly the sub-band
each stage needs. Neither ever needs cross-band data — a PRN's despread is per-channel and
combines *afterwards* in record space.

**Acquisition** is the flexible one. Per `gnss_chord_framework.md`, CHORD acquisition is
single-antenna (≈ free in data terms): one dish's sub-band (or gathered band) at modest
rate. Options, in preference order: (a) run it on a node GPU's spare capacity against its
own sub-band — fits "GPU only on sub-bands" with zero new hardware; (b) the current
gathered-raw-voltage → central search path with a GPU on the aggregator — only worth
standing up if per-sub-band sensitivity proves insufficient. An independent aggregator GPU
is therefore *optional, deferrable*, and nothing in the framework blocks adding it later.

**Shape of the kernels** (all conventional, no research risk):
- Search: batched cuFFT circular correlations + incoherent accumulation — the canonical GPU
  GNSS acquisition kernel. Latency-tolerant by design (K7).
- Despread: after R1 the tracker is a *batched complex dot product* against cached
  channelized replicas — GEMV/GEMM-shaped, decision-free. Replicas are periodic per
  (PRN, Doppler-bucket): precompute into GPU memory, regenerate on bucket change.
- Peel: despread (above) + scale + subtract, streaming; v2's smooth-gain is a small state
  update per record.

**Practical on-ramp on the GX10:** kotekan already carries the CHORD CUDA framework
(`lib/cuda`, `cudaCommand`, chordMVPSetup) and the ARM commit set
`CMAKE_CUDA_ARCHITECTURES=121`; the GB10's 121 GB *unified* memory removes the usual
host↔device staging question entirely. Recommended order:
1. **GPU search** first — it is today's actual CPU pain (blind grids, acquisition
   transients), it is the best-understood kernel, and it teaches the cudaCommand idioms on
   a stage whose async design tolerates any integration clumsiness.
2. **GPU batched despread** second — after R1, as CHORD-scale preparation (dish × PRN
   fan-out) and to unlock many-signal simultaneous tracking on the GX10.
3. **GPU peel** third — port when moving from single-sat demos to multi-PRN standing
   mitigation; same despread kernel plus a subtract.

---

## 4. Readiness for what's next

- **Peeling at scale:** GnssVoltagePeel v1/v2 exist and are validated single-dish. The CHORD
  form is Tier-1 GPU (above). Blocked mainly on R1 (peel wants the same commanded-state,
  decision-free despread) — not on new physics.
- **Galileo E1 (OS/E1C):** BOC(1,1) already supported in the replica bank; needs the E1
  memory codes (table constants, like L2CL) + R6 for the CS25 secondary. No structural work.
- **BeiDou B1C / QZSS:** same pattern as E1/L1C. No structural work.
- **GLONASS L1OF:** FDMA — the one structural stretch (per-PRN carrier). Defer.
- **L1 nav-bit jitter + CL deep lock (the two open threads):** both are symptoms of the
  pull-in/hold architecture; both are fixed properly by R1 rather than by more tuning.

## 5. Recommended sequence

1. R3 (delete dead stages) — 30 min, subtractive.
2. R2 (generator authoritative) — ½ day, kills config drift before it compounds.
3. **R1 (E/P/L records + broker-closed loops) + R4 (record schema)** — the real one; also
   the correct fix for both open tracking threads. Re-validate L1 deep-wipe + distributed
   bit-identity after.
4. GPU search on the GX10 (USE_CUDA=ON, arch 121, cudaCommand) — first CUDA port.
5. Then peeling at scale / Galileo, in either order, on a clean foundation.
