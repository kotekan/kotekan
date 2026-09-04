# n2k_dual — two-input clone of the n2k correlator

Cloned 2026-08-06 from kotekan's vendored `external/n2k` (upstream commit in
`external/n2k/COMMIT`, https://github.com/kmsmith137/n2k). **`external/n2k` is not
modified** — n2k and the production N² path are under active development by others, so
this is a deliberately separate, mostly-duplicated module. If the two-input extension is
accepted upstream, this module retires.

## What it does

One kernel launch computes the visibility triangle over a **concatenated station axis**
fed from two separately-shaped inputs:

```
A: int8[T][F][nstations_a]   real antenna voltages (as today)
B: int8[T][F][nstations_b]   synthetic stations (GNSS replica waveforms)
```

Motivation: CHORD GNSS "path B" (`docs/gnss_gpu_search.md` §11) — the despread
correlations ride the optimised tensor-core kernel as A×B tiles, the antenna voltages are
loaded once, and the pure-A block of the output is a **verbatim prefix** of each (t,f)
slice (triangular tile packing orders by row), so handing the standard N² result to the
production pipeline is a strided copy.

## Divergences from n2k (complete list)

| Where | Change |
|---|---|
| `DualCorrelatorParams` | `(nstations_a, nstations_b, nfreq, block_class_mask)`; per-input `emat_*stride_{a,b}`; `ptable_nrows = 8` |
| ptable | 8 rows/thread instead of 6: row 6 = warp's input time stride (int32), row 7 = input select. Built per **enabled** block (`block_class_mask` can skip AA / mixed / BB block classes). blockId → (atile,btile) decode hoisted into `dual_enumerate_blocks()` (same order as n2k's `_init_block_data`). |
| `kernel_body` | takes `srcA`+`srcB`; warp-uniform base-pointer select; per-freq offset `f*(ts/NF)`; time stride `ts` is runtime per-warp instead of `constexpr emat_tstride`, threaded through `prefetch_chunk` / `do_initial_prefetch` / `correlate_t64` / `correlate_t128` |
| `kernel_t` | `(int*, const int8_t*, const int8_t*, const uint*, const int*, int, int)` |
| instantiations | `dual_kernel_{NS_total}_{NF}.cu`; the template depends only on the total |

Everything else — shared-memory layout, in-register transposes, MMA core, RFI masking,
output tiling/write path — is a verbatim copy.

## Contracts

- `nstations_a` and `nstations_b` each a multiple of 128 (a threadblock's load panel must
  fall wholly in one input). `nstations_b == 0` must reproduce `n2k::Correlator`
  **bit-exactly** (regression gate).
- 4-bit samples in **[-7, +7]; -8 silently corrupts** (`negate_4bit`). For the synthetic
  input this is a hard quantizer requirement.
- Mixed tiles have the B station on the row (i) side: `V = E_B · conj(E_A)`.
- Tiles of disabled block classes are never written; the caller must not read them.
- The per-(f,t) RFI mask gates A and B identically.
