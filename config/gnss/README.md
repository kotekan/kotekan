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
Python generator emits today. **All six nodes: EQUIVALENT, 137/137 blocks each** — the entire GNSS branch.

It earned its keep immediately — it failed twice on the first run and named both causes:

1. `spectrum_ring_depth` / `spectrum_window_samples` were missing from the template. I had
   templated `n2assemble` from a dump truncated at 420 characters and never saw them.
2. `n2assemble_tiles` has its **own** CPU core, distinct from `n2assemble`'s. The record
   assembler's core is per-GPU (31 / 57); the tiles assembler's is per-chain (59, 24, 31,
   62). I had assumed they shared one.

Neither would have been visible by reading the template.

## ⚠️ Two orphan buffers, found by doing this

`gnss0_cmb_buf` and `gnss1_cmb_buf` are defined in **every** deployed node config and
referenced by **nothing** — checked against every string in the config, and contrast the
identically-named-but-live `gnss{N}_n2cmb_buf`. They cost ~206 KB per GPU, so this is
tidiness rather than a leak, but they are dead config that a reader has to rule out.

The template deliberately does **not** render them: templating an orphan launders dead
config into the new structure and makes it permanent. The gate lists them as NOT RENDERED
on every run so the discrepancy stays visible until the generator stops emitting them —
which is a one-line change plus a regenerate, and wants a node restart to land.

## What is NOT done yet

* **The vars file is a bridge, not the source of truth.** It is currently *extracted from
  generator output* so the template could be gated before the generator changes. The end
  state is `gen_chord_gnss_config.py` emitting it directly from the node table.
* ~~Per-GPU singletons~~ **DONE** (second pass, same day). Two findings made it cheap:
  the **primary chain is not a special case** — `gnss0_n2combine` is structurally
  identical to `gnss0_e5a_n2combine` field for field, including its command list, so it is
  simply a chain whose tag is `""` and the same loop body renders it. Only the per-GPU
  **search leg** (`srch_tap` / `srch_buf` / `srch_send`) is genuinely different, because
  one voltage tap per GPU serves every signal on that GPU; it gets its own short loop
  outside the chain loop. Plus `gnss_pool`.
* **The real prize is dropping the captured base.** Today the generator injects the GNSS
  branch into `config/base/live_config_20260730.json`, a frozen copy of production taken
  in July. Once the branch is a clean include, it can be included *into*
  `chord_pathfinder.j2` itself and the frozen copy goes away — which is the part that
  actually stops us drifting from upstream. That is a config-deployment change and wants
  its own review.
