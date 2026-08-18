# The GNSS branch as a j2 include

Jim Mertens' suggestion, 2026-08-18: *"keep the variable gnss data in a separate file, and
include it from the main j2. Maybe some loops could condense it too."*

**Status: done and wired.** `config/chord_pathfinder.j2` calls the include. It renders
nothing unless a node is named, and what it does render is field-for-field what we deploy.

```
kotekan -c config/chord_pathfinder.j2 -j '{"gnss_node": "cx19"}'
```

⚠️ **Deployment has NOT moved yet.** `node_up.sh` still runs the generated
`config/generated/chord_gnss_<node>_multi.yaml`, and the generator still injects into the
captured base. This template is proven equivalent to that; switching deployment onto it is
a separate change — see *What is left*.

## Why the split is worth it, measured

Counted on the deployed `chord_gnss_cx19_multi.yaml`, not assumed:

| | |
|---|---|
| GNSS keys in a node config | **139** |
| distinct structural patterns | **31** (8x per-chain, 2x per-GPU) |
| fields across the repeated patterns | 233 |
| of those, **identical in every copy** | **161 (69%)** |

So roughly seven-eighths of ~100 KB per node is one template applied eight times. The 31%
that varies is mechanical: names built from the loop indices, CPU cores from a rotation,
and the per-chain channel / PRN / size data.

## The split

* **`gnss_chain.j2`** — structure, no node-specific values. A nested loop over GPUs and
  chains renders the 13 per-chain blocks (dual-correlator cudaProcess, both record
  assemblers, combiner, sink, telemetry pack + send, six buffers). A second short loop
  renders the per-GPU acquisition **search leg** (`srch_tap` / `srch_buf` / `srch_send`),
  which is per-GPU rather than per-chain because one voltage tap serves every signal on
  that GPU. Then `gnss_pool`.
* **`gnss_vars_<node>.j2`** — data. One `set gnss = {...}` with the receiver-wide constants
  and a `gpus[].chains[]` list. One file per node; all six are committed.

**The primary chain is not a special case.** `gnss0_n2combine` is structurally identical to
`gnss0_e5a_n2combine` — same field set, same command list, differing only in signal and
data. It is a chain whose tag is the empty string, and the same loop body renders it.

⚠️ **Import the vars, include the structure.** Jinja passes the parent context *down* into
an include but does not export that include's assignments back *up*, so a vars file that is
merely included is invisible and the render dies with `'gnss' is undefined`. Import does
export top-level assignments, which is why `chord_pathfinder.j2` carries two lines rather
than one:

```jinja
{% import "gnss/gnss_vars_cx19.j2" as gnss_vars %}{% set gnss = gnss_vars.gnss %}
{% include "gnss/gnss_chain.j2" %}
```

⚠️ And a trap when documenting it: **jinja parses `{%` `%}` inside YAML comments**, because a
YAML comment is not a jinja comment. Prose describing jinja syntax must avoid the brace
sequences or be wrapped in `raw` — the template failed to compile until it was.

## The gates

**1. The include vs the generator**, field by field:

```
scripts/gnss/j2_chain_equiv.py config/generated/chord_gnss_cx19_multi.yaml
```

**All six nodes: EQUIVALENT, 137/137 blocks each** — the entire GNSS branch.

**2. `chord_pathfinder.j2` itself**, checked when the wiring landed:

| check | result |
|---|---|
| stock render (no `gnss_node`) | 145 keys, **0 GNSS** — byte-unchanged |
| with `gnss_node=cx19` | 282 keys, 137 GNSS |
| non-GNSS blocks vs a stock render | **identical** |
| 137 rendered blocks vs the deployed config, all 6 nodes | **identical** |

That last row is the one that matters: the branch rendered from *today's upstream template*
is what we are actually running.

**The gate earned its keep on its first run**, failing twice and naming both causes:

1. `spectrum_ring_depth` / `spectrum_window_samples` were missing from the template — I had
   templated `n2assemble` from a dump truncated at 420 characters and never saw them.
2. `n2assemble_tiles` has its **own** CPU core. The record assembler's is per-GPU (31 / 57);
   the tiles assembler's is per-chain (59, 24, 31, 62). I had assumed they shared one.

Neither would have been visible by reading the template.

## ⚠️ Two orphan buffers, found by doing this

`gnss0_cmb_buf` and `gnss1_cmb_buf` are defined in **every** deployed node config and
referenced by **nothing** — checked against every string in the config; contrast the
identically-shaped but live `gnss{N}_n2cmb_buf`. About 206 KB per GPU, so tidiness rather
than a leak, but dead config a reader has to rule out.

The template deliberately does **not** render them: templating an orphan launders dead
config into the new structure and makes it permanent. The gate lists them as NOT RENDERED
on every run so the discrepancy stays visible until the generator stops emitting them —
a one-line change plus a regenerate, wanting a node restart to land.

## What is left

* **The vars files are produced by extraction, not computed.**
  `scripts/gnss/j2_chain_equiv.py --keep` writes them from a generated config, so they
  agree with it by construction — weaker than it looks. Closing this means computing them
  in `gen_chord_gnss_config.py` from the node table and deleting the block-building in
  `build_n2dual_branch`, after which the CPU-core rotation and the frame-size formulas live
  in exactly one place instead of two, and this template becomes the only definition of the
  stage graph.
* **Moving deployment onto this path.** Once the above lands, `node_up.sh` can render
  `chord_pathfinder.j2` directly and `config/base/live_config_20260730.json` — our frozen
  July copy of production — goes away, which is what actually stops us drifting from
  upstream. That changes how configs are produced and deployed, so it wants review and a
  restart window rather than a quiet switch.
