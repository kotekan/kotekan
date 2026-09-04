# CHORD-side development handoff

Orientation for an agent starting CHORD-hardware development branched from
`kv/gps-replica-correlator`. Goal on your end: stand up the pieces to handle **CHORD data
structures (frequency combs, etc.), rates, and on-sky systems** on real CHORD hardware
(instrumented dishes, RFSoC F-engine, GPU nodes). This is intended to run **largely orthogonally**
to the ongoing airspy-GPS work on the origin node; we merge as things evolve.

## What this repo/branch is

This is **kotekan** — which is already CHORD's real-time correlator/beamformer framework — plus a
GNSS receiver built on the same infrastructure on branch `kv/gps-replica-correlator`. The GNSS
work (a 3-band airspy GPS/GNSS receiver: search → track → coherent-combine → voltage peel) was a
vehicle for exercising CHORD-shaped patterns at small scale on a single node. Much of the
architecture is deliberately CHORD-mirrored, so you are branching from a codebase that already
speaks CHORD idioms — you are NOT starting from scratch.

**Native CHORD code already present** (not GNSS): `lib/utils/CHORDTelescope.{cpp,hpp}`,
`lib/metadata/chordMetadata.{cpp,hpp}`, `lib/cuda/chordMVPSetup.{cpp,hpp}`, the generated
`FRBBeamformer_chord`/`BasebandBeamformer_chord` kernels under `lib/cuda/generated/`, and
`lib/testing/testCHORDTelescope.*`. Start here for the F-engine/telescope data model.

## Reusable patterns (built + proven on the GNSS side — mirror, don't reinvent)

- **Distributed band-split pipeline.** A PFB splits the wideband stream into subbands processed on
  separate nodes/stages, recombined coherently and **bit-identically** across nodes. See the memory
  of `distributed-band design`, `distributed pipeline stages` (CHORD-mirror split/track/combine),
  and `distributed search stages` (ship-raw-voltage → Gather → Search). This is the template for a
  CHORD frequency-comb / subband layout.
- **GPU kernel contract.** `int8` GEMM ran ~7× fp32; the gather kernel ~4.15×; a **4+4-bit**
  quantization contract (complex rms ~2.1, ~1.5/component) is used end to end. `GnssQuantize44.cpp`,
  `lib/cuda/cudaGnss*.{cpp,cu}`, and `chordMVPSetup` are the references. ⚠️ a benchmark once ranked
  kernels BACKWARDS (deployed geometry ≠ bench geometry) — validate kernel choices against
  air/real geometry, not just a microbench.
- **The tri-band node.** `config/run_3band.sh` is "THE NODE" — one kotekan instance, three front
  ends, three control planes. It shows the launch/teardown discipline, the valve drop (silent-loss)
  instrumentation, and the health-gating you will want on CHORD nodes.
- **Shared-knowledge framework.** `docs/gnss_shared_knowledge_framework.md` — the typology for
  sharing state across chains/bands/restarts (three mechanisms; fuse continuous state, pin+verify
  integers, supply-chain symbol content shadow-scored). Directly relevant to sharing calibration /
  fringe state across CHORD feeds and subbands.
- **CHORD framework doc** (GNSS-side, but the path-A/B decomposition and voltage-peel are
  CHORD-motivated): `docs/gnss_chord_framework.md`.

## What to build on the CHORD side (your scope)

1. **Data structures**: the frequency-comb / channelized F-engine layout, the RFSoC packet/frame
   model, and the metadata (chordMetadata) needed to carry feed/pol/freq/time through the pipeline.
2. **Rates**: size the buffers/valves in TIME not frame-count (see the "count-where-a-time-was-meant"
   lesson below) — CHORD rates are far above the airspy node's, so any hardcoded record/frame count
   is a latent bug.
3. **On-sky systems**: pointing/telescope model (CHORDTelescope), calibration ingest, and the
   real-time health/observability the node needs.

## Build env + hardware rules (learned the hard way)

- **Build**: ninja, e.g. `ninja -C build_cuda kotekan/kotekan`. Cobalt build env: gcc-12, disable
  ASDF/GDAL/DPDK (see memory `cobalt-build-env`). The GNSS GPU box is an ARM/GB10 (GX10, CUDA 13)
  with a **FRAGILE-DRIVER RULE**: no nvidia/kernel/apt upgrades. Your CHORD GPU nodes are beefier —
  confirm their own build/driver constraints before touching them.
- **Standing discipline** (from `run3band-launch-teardown`): launch once by absolute path via
  `setsid`; stop via a pidfile TERM; never combine stop+start in one shell command; never a blind
  `pkill` (self-match has killed the soak). Gate launches on the hardware actually being present.
- **`valve_dropped_frames_total` is the silent-data-loss observable** — a dropped frame zero-fills
  the ring and every coherent window touching it shreds, looking exactly like decohered signal.
  Watch it whenever you add GPU load. And do NOT run heavy compute (e.g. numpy/Viterbi benches)
  that contends with a live real-time pipeline — it drops frames even when nice'd on "isolated"
  cores (it just cost the airspy soak ~2669 L5 frames; logged in the GNSS-side data-quality
  register `docs/data_quality_windows.md`, loader `python/scripts/gnss/dq_windows.py`).

## The one bug class to internalize: COUNT-where-a-TIME-was-meant

On a multi-rate system, a constant that is a COUNT of records/frames means a different DURATION on
each chain. This class bit the GNSS node repeatedly (`buffer_depth`, `coherence_s`, the coherence
ladder's `min_len`, a seed-hold in snapshots, `bit_export` window sizing). **It surfaces first in
the chain with the COARSEST quantum.** CHORD spans a huge rate range across its combs — audit every
size/duration constant for units. Prefer expressing depths/holds/windows in TIME and deriving the
count from the local rate.

## Orthogonality + merge strategy

- Branch from `kv/gps-replica-correlator`. Keep CHORD-native work in the CHORD stages
  (`lib/.../chord*`, `CHORDTelescope`, `chordMVPSetup`, generated CHORD kernels) and new
  CHORD-specific stages; avoid editing the GNSS-only stages (`lib/stages/gnss/*`,
  `python/scripts/gnss/*`, `config/run_3band.sh` and friends) so the two efforts don't collide.
- Shared surface likely to touch both: `lib/cuda/*` kernels, quantization contract, metadata,
  buffer/valve infrastructure, build config. Coordinate changes there; they're the natural merge
  points.
- The airspy node keeps running the current plan (GNSS shared-knowledge S3/S4: L2C CNAV decode is
  now live, next is the ephemeris-vs-BRDC cross-check and the L5-I cross-band feed). Expect periodic
  merges; keep commits scoped and well-described so they rebase cleanly.

## Fast start checklist

1. `git checkout kv/gps-replica-correlator` and read `docs/gnss_shared_knowledge_framework.md` +
   `docs/gnss_chord_framework.md` for the architecture idioms.
2. Read `lib/utils/CHORDTelescope.hpp`, `lib/metadata/chordMetadata.hpp`, `lib/cuda/chordMVPSetup.hpp`
   for the native CHORD data model, and `tests/.../testCHORDTelescope` for how it's exercised.
3. Skim `config/run_3band.sh` for the launch/teardown/health discipline to mirror on CHORD nodes.
4. Build with ninja; confirm your GPU nodes' driver/build constraints before any system change.
5. Stand up a minimal CHORD F-engine ingest → channelize → metadata path first; validate the
   frequency-comb data structure and rates end to end before layering on-sky systems.
