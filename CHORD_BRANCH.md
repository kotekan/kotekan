# chord branch contents

This file exists only on the `chord` branch. It lists how `chord` differs from `develop`. See the "chord branch" section of AGENTS.md for the workflow.

Base: `develop` at `419f49ab1` (#1845).

## Open PRs carried in chord

| PR | Branch | Head in chord | In chord since | Head since | Subject |
|---|---|---|---|---|---|
| #1732 | `jbm/eigen-phase-mask` | `60419ef7e` | 2026-10-03 | 2026-10-09 | EigenMaskedSubspaceSolver: subspace steps on the masked matrix, phase reference |
| #1743 | `jbm/dish_pointings` | `9b6e29db6` | 2026-10-17 | 2026-11-04 | Fringestop only array dishes; require a set pointing |
| #1843 | `jbm/rest-logs` | `3bc9fd09a` | 2026-11-04 | 2026-11-04 | Serve recent log lines at /logs |
| #1844 | `jbm/n2-truncate` | `3e8a3c0d3` | 2026-11-05 | 2026-11-05 | hdf5N2Write: optional weight-based truncation of vis and weights |
| #1847 | `jbm/gpu-ring-graph` | `6b4d69c2a` | 2026-11-04 | 2026-11-04 | Pipeline graph: GPU rings as host ring nodes, GPU tables, /pipeline_gpu_svg |

## Chord-only commits

| Commit | Files | Change |
|---|---|---|
| `cd85d6afb` | `config/chord/common.j2` | Keep recent log lines for `GET /logs` (#1843) |
| `030e1715f` | `config/chord/recv.j2` | Truncate subset vis and weights before writing (#1844) |
| this file's commit | `CHORD_BRANCH.md`, `AGENTS.md` | This file and the chord workflow notes |
