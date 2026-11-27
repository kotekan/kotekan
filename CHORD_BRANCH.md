# chord branch contents

This file exists only on the `chord` branch. It lists how `chord` differs from `develop`. See the "chord branch" section of AGENTS.md for the workflow.

Base: `develop` at `b4a01af38` (#1926).

## Open PRs carried in chord

| PR | Branch | Head in chord | In chord since | Head since | Subject |
|---|---|---|---|---|---|
| #1732 | `jbm/eigen-phase-mask` | `dbe6a1008` | 2026-10-03 | 2026-11-05 | EigenMaskedSubspaceSolver: subspace steps on the masked matrix, phase reference |
| #1743 | `jbm/dish_pointings` | `6bba47a3c` | 2026-10-17 | 2026-11-05 | Fringestop only array dishes; require a set pointing |
| #1843 | `jbm/rest-logs` | `076aed9d2` | 2026-11-04 | 2026-11-20 | Serve recent log lines at /logs |
| #1844 | `jbm/n2-truncate` | `b3ddcf97d` | 2026-11-05 | 2026-11-20 | hdf5N2Write: optional weight-based truncation of vis and weights |
| #1847 | `jbm/gpu-ring-graph` | `7b3fc8f2b` | 2026-11-04 | 2026-11-05 | Pipeline graph: GPU rings as host ring nodes, GPU tables, /pipeline_gpu_svg |
| #1848 | `jbm/n2-switch-per-freq` | `9d6821aeb` | 2026-11-05 | 2026-11-05 | hdf5N2Write: record fringestop and second-stage RFI switches per frequency |
| #1849 | `jbm/n2-gains` | `8868a27f3` | 2026-11-05 | 2026-11-20 | bufferGains: buffer posted calibration gains; N2Accumulate records them |
| #1851 | `jbm/async-start` | `944510918` | 2026-11-05 | 2026-11-06 | kotekan: start the pipeline on its own thread, report it in /status |
| #1903 | `jbm/incoherent-search` | `a18c1a010` | 2026-11-17 | 2026-11-23 | Incoherent-beam pulse search: IncoherentBeamPower and IncoherentBeamSearch |

The PR branches carry the site config they need, so config/chord in `chord` is `develop` plus these heads.

## Chord-only commits

| Commit | Files | Change |
|---|---|---|
| `12f44fcb5` | `CHORD_BRANCH.md`, `AGENTS.md` | This file and the chord workflow notes |
