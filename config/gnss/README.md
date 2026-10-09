# The GNSS branch as a j2 include

Jim Mertens' suggestion, 2026-08-18: *"keep the variable gnss data in a separate file, and
include it from the main j2. Maybe some loops could condense it too."*

**Status: not wired.** The include lived in the old single-file `config/chord_pathfinder.j2`.
Production's template is now `config/chord/pathfinder.j2`, which loads its includes from
`config/chord/`, so the hook has to be added there again, with these files reachable from that
directory. `gen_fleet.py --check` still regenerates these vars beside the node configs and
compares them byte for byte.

**Deployment still runs the generator's output** — `node_up.sh` starts
`config/generated/chord_gnss_<node>_multi.yaml` — but since 2026-10-02 the generator's
base is the **stock render of this same template** (no `gnss_node`, rendered exactly as
kotekan renders it), not a config captured from a running node. The captured base went
stale silently and our nodes shipped the August pipeline's N² for a month
(`fixtures/stock_parity_20261002/`). Two gates now hold both halves:

* the GNSS half: this include renders the generator's GNSS blocks (gate 1 below), and
  `gen_fleet.py --check` byte-compares the vars;
* the stock half: `scripts/gnss/stock_parity.py` diffs every non-GNSS block against the
  stock render — or, with `--live cx47`, a running stock node — and fails on anything
  not declared there with a reason. `gen_fleet.py` runs it on every config it writes or
  checks.

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
  and a `gpus[].chains[]` list. One file per node, **owned by the fleet driver** and
  regenerated and checked alongside the node configs. Neither is tracked in git;
  `example_chord_gnss_cx51_multi.yaml` is one node's output:

  ```
  python3 scripts/gnss/gen_fleet.py config/gnss_fleet_chord.yaml           # write
  python3 scripts/gnss/gen_fleet.py config/gnss_fleet_chord.yaml --check   # gate
  ```

  `gnss_chain_vars()` computes them and `build_n2dual_branch()` *consumes* them, so the
  CPU-core rotation and the frame-size formulas exist in exactly one place. Emitting is
  side-effect-free: the YAML that run writes is unchanged, byte for byte (the generator
  excludes `--emit-j2-vars` from the recipe it stamps into the config, for the same reason
  it excludes `--out`).

  > ⚠️ **They were hand-emitted once, in August, and then rotted for two weeks.** Found
  > 2026-09-02: all six were 107 diff lines behind the generator — the pre-08-31
  > single-NUMA core pools, no `phi_fp16`, no `despread_max_chips`, no B3I/E6 channels in
  > `band_power_chans`, `tiles` sized 2981888 instead of 917504 — while this file said they
  > were "field-for-field what we deploy". Nothing regenerated them and nothing checked
  > them, so the claim could not fail. `gen_fleet --check` now covers them; that is what
  > makes the sentence above true rather than merely intended.

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

> ⚠️ **Broken as of 2026-10-02, and not by the stock rebuild:** it dies with
> `KeyError: 'cpu_affinity'` on `<chain>_n2sink` against the configs deployed on 10-01
> too — the sink lost its pinned core and `extract()` was never told. The generator has
> also grown past it (244 GNSS blocks per node against the include's 202). The vars are
> still byte-checked by `gen_fleet.py --check`; this field-level gate needs repair.

**2. `chord_pathfinder.j2` itself**, checked when the wiring landed:

| check | result |
|---|---|
| stock render (no `gnss_node`) | 145 keys, **0 GNSS** — byte-unchanged |
| with `gnss_node=cx19` | 282 keys, 137 GNSS |
| non-GNSS blocks vs a stock render | **identical** |
| 137 rendered blocks vs the deployed config, all 6 nodes | **identical** |

That last row is the one that matters: the branch rendered from *today's upstream template*
is what we are actually running.

**The gates have caught five real errors so far**, none of them visible by reading the code:

1. `spectrum_ring_depth` / `spectrum_window_samples` were missing from the template — the
   `n2assemble` block had been templated from a dump truncated at 420 characters.
2. `n2assemble_tiles` has its **own** CPU core. The record assembler's is per-GPU (31 / 57);
   the tiles assembler's is per-chain (59, 24, 31, 62).
3. The `rec` buffer's `frame_size` carries a chan-export term that `cmb`'s does not
   (`+ n_prn * n_chan * chan_floats()`, present whenever telemetry is on). The first
   `gnss_chain_vars()` wrote both as the same expression, under-sizing `rec`.
4. This checker used to write a vars file and then delete it — harmless until those files
   became committed artifacts, at which point one run destroyed six tracked files. It now
   writes a dot-prefixed check copy unless `--keep`. A check that mutates its inputs is not
   a check.
5. **A duplicated variable, spotted by KV reading the emitted file** rather than by any
   gate: `sample_rate_hz` and `sample_rate_mhz` both held 3200000000.0. Not two values
   disagreeing — one value with a name carrying the wrong unit, because it had first been
   read off a dump truncated mid-number and taken for 3200 MHz (the same truncation that
   caused finding 1). Every site in the generator derives it from one expression,
   `float(fengine.sampling_rate_MHz) * 1e6`, so a second name could only ever be the same
   number wearing a wrong unit. Collapsed to `sample_rate_hz`, now computed from the
   F-engine block rather than read back out of a stage the writer had just written.

**Every variable in the emitted file is consumed by the template** — audited by matching
each key against `gnss.<k>` / `c.<k>` / `g.<k>` in `gnss_chain.j2`; 29 receiver-wide,
per-chain and per-GPU entries, no dead data.

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

* **Deleting the block-building.** `build_n2dual_branch()` now consumes the vars rather
  than recomputing them, so the duplication is gone — but it still assembles the ~137
  blocks that the template also renders. Removing that code is the step that makes the
  template the single definition of the stage graph; it is mechanical, and gate 2 is what
  proves it safe.
* **Moving deployment onto this path.** The base half is done (2026-10-02: the generator
  injects into the stock render). Rendering `chord_pathfinder.j2` directly would still lose
  what the generator adds to the stock half — four data-neutral, declared tweaks
  (`stock_parity.py`'s `DECLARED`) and the two values stock receives over REST at runtime
  (the EOP table, bffs's bad inputs), which the generator fetches live. Each of those is the
  remaining work: upstream it, move it into a GNSS-only block, or have `node_up.sh` push it
  after start.
