# The GNSS branch as a j2 include

Jim Mertens' suggestion, 2026-08-18: *"keep the variable gnss data in a separate file, and
include it from the main j2. Maybe some loops could condense it too."* This directory is
step one of that restructure, with a gate proving it faithful before anything moves.

## Why it is worth doing, measured

Counted on the deployed `chord_gnss_cx19_multi.yaml`, not assumed:

| | |
|---|---|
| GNSS keys in a node config | **139** |
| distinct structural patterns | **31** (8× per-chain, 2× per-GPU) |
| fields across the repeated patterns | 233 |
| of those, **identical in every copy** | **161 (69%)** |

So roughly seven-eighths of ~100 KB per node is one template applied eight times. The 31%
that varies is mechanical: names built from the loop indices, CPU cores from a rotation,
and the per-chain channel / PRN / size data.

## The split

* **`gnss_chain.j2`** — structure. Every per-chain block (13 of them: the dual-correlator
  cudaProcess, both record assemblers, the combiner, the sink, telemetry pack + send, and
  six buffers), rendered by a nested loop over GPUs and chains. No node-specific values.
* **`gnss_vars_<node>.j2`** — data. One `{% set gnss = {...} %}` with the receiver-wide
  constants and a `gpus[].chains[]` list.

⚠️ **Import the vars, include the structure.** Jinja passes the parent context *down* into
an `{% include %}` but does not export that include's `{% set %}` names back *up*, so a
vars file that is merely included is invisible and the render dies with `'gnss' is
undefined`. `{% import %}` does export top-level assignments:

```jinja
{% import "gnss_vars_cx19.j2" as gv %}{% set gnss = gv.gnss %}
{% include "gnss_chain.j2" %}
```

This is the one mechanical constraint the split imposes, and it is why the top-level file
has to name the vars file rather than the chain template doing it.

## The gate

```
scripts/gnss/j2_chain_equiv.py config/generated/chord_gnss_cx19_multi.yaml
```

renders the template and compares every per-chain block, field by field, against what the
Python generator emits today. **All six nodes: EQUIVALENT, 104/104 blocks each.**

It earned its keep immediately — it failed twice on the first run and named both causes:

1. `spectrum_ring_depth` / `spectrum_window_samples` were missing from the template. I had
   templated `n2assemble` from a dump truncated at 420 characters and never saw them.
2. `n2assemble_tiles` has its **own** CPU core, distinct from `n2assemble`'s. The record
   assembler's core is per-GPU (31 / 57); the tiles assembler's is per-chain (59, 24, 31,
   62). I had assumed they shared one.

Neither would have been visible by reading the template.

## What is NOT done yet

* **The vars file is a bridge, not the source of truth.** It is currently *extracted from
  generator output* so the template could be gated before the generator changes. The end
  state is `gen_chord_gnss_config.py` emitting it directly from the node table.
* **Per-GPU singletons** (`gnss{N}_srch_*`, the primary `gnss{N}_telem_*`, `gnss{N}_n2*`
  without a chain tag) are not claimed by this include yet — 35 of the 139 keys. Same
  treatment, second pass.
* **The real prize is dropping the captured base.** Today the generator injects the GNSS
  branch into `config/base/live_config_20260730.json`, a frozen copy of production taken
  in July. Once the branch is a clean include, it can be included *into*
  `chord_pathfinder.j2` itself and the frozen copy goes away — which is the part that
  actually stops us drifting from upstream. That is a config-deployment change and wants
  its own review.
