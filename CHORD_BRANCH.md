# chord branch contents

This file exists only on the `chord` branch. It lists how `chord` differs from `develop`. See the "chord branch" section of AGENTS.md for the workflow.

Base: `develop` at `7e1dc5433` (#1850).

## Open PRs carried in chord

| PR | Branch | Head in chord | In chord since | Head since | Subject |
|---|---|---|---|---|---|
| #1732 | `jbm/eigen-phase-mask` | `dbe6a1008` | 2026-10-03 | 2026-11-05 | EigenMaskedSubspaceSolver: subspace steps on the masked matrix, phase reference |
| #1743 | `jbm/dish_pointings` | `6bba47a3c` | 2026-10-17 | 2026-11-05 | Fringestop only array dishes; require a set pointing |
| #1843 | `jbm/rest-logs` | `294bc2188` | 2026-11-04 | 2026-11-05 | Serve recent log lines at /logs |
| #1844 | `jbm/n2-truncate` | `def072ff5` | 2026-11-05 | 2026-11-05 | hdf5N2Write: optional weight-based truncation of vis and weights |
| #1847 | `jbm/gpu-ring-graph` | `7b3fc8f2b` | 2026-11-04 | 2026-11-05 | Pipeline graph: GPU rings as host ring nodes, GPU tables, /pipeline_gpu_svg |
| #1848 | `jbm/n2-switch-per-freq` | `9d6821aeb` | 2026-11-05 | 2026-11-05 | hdf5N2Write: record fringestop and second-stage RFI switches per frequency |
| #1849 | `jbm/n2-gains` | `f76d2aba0` | 2026-11-05 | 2026-11-05 | bufferGains: buffer posted calibration gains; N2Accumulate records them |
| #1851 | `jbm/async-start` | `6acfe2756` | 2026-11-05 | 2026-11-05 | kotekan: start the pipeline on its own thread, report it in /status |

## Chord-only commits

| Commit | Files | Change |
|---|---|---|
| `cd85d6afb` | `config/chord/common.j2` | Keep recent log lines for `GET /logs` (#1843) |
| `030e1715f` | `config/chord/recv.j2` | Truncate subset vis and weights before writing (#1844) |
| `12f44fcb5` | `CHORD_BRANCH.md`, `AGENTS.md` | This file and the chord workflow notes |
| `a82b96f02` | `config/chord/pathfinder.j2` | Record calibration gains in the X-engine N2 frames (#1849) |
| `ef35f45de` | `config/chord/common.j2` | Name the F-engine controller by its full name |
