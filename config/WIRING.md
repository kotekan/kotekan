# Band-config generation wiring (audit 2026-07-19 rec C)

**POLICY: hand-editing a generated `live_*.yaml` is a bug.** Edit `config/gnss_node.yaml`
(the single per-signal source of truth), regenerate, and let the change propagate to every
chain that shares the row. This is the fix for the 07-18/19 bug class: the silently dropped
nh-assist flag, the fence fix that landed on 1 of 4 trackers, and doppler_step /
acquire_windows / gains drifting apart between hand-maintained copies.

## The pipeline

```
gnss_node.yaml  --gen_band_config.py-->  live_l1_dual20.yaml     --gen_3band_config.py--> live_3band.yaml
 (edit HERE)                             live_l2c_gpu.yaml
                                         live_l5_gpu.yaml
```

## The one-line switch

`run_3band.sh` already regenerates the merged config on every launch
(`python3 config/gen_3band_config.py`). To make the band files generated artifacts too, add
ONE line directly above it:

```sh
python3 config/gen_band_config.py --out-dir config || { echo "band config generation FAILED"; exit 1; }
python3 config/gen_3band_config.py || { echo "config generation FAILED"; exit 1; }   # (existing line)
```

Single-band launches (`run_band.sh {l1|l2c|l5}` / `run_live.sh`) read the same
`config/live_*.yaml` paths, so the same regeneration covers them — either via the line above
or by running `gen_band_config.py --out-dir config` after any node-table edit.

## Validation

```
python3 config/gen_band_config.py --check
```

generates all three files (to `config/generated/` by default) and does a STRUCTURAL diff
against `config/live_*.yaml`: both sides `yaml.safe_load`-ed and deep-compared, every
differing path reported, strict on scalar types (`50` vs `50.0` counts). Exit 0 = parsed
content identical (comments/ordering are free to differ). Run it before any deploy that
touched `gnss_node.yaml` or the generator. As of 2026-07-19 the diff is EMPTY for all three
files — the generated files are drop-in replacements for the hand ones.

Verified downstream compatibility (2026-07-19):

- `gen_3band_config.py` consumes the generated files; the resulting `live_3band.yaml` is
  structurally identical to the hand-derived one *except* for one benign artifact of the
  merger itself: with the hand files, the "UNIFIED L1 GPU PROCESS (…cudaProcess…)" comment
  sits between `epl_buf` and `gputrack`, so the merger's comment-attachment heuristic
  mistakes `l1_epl_buf` for a stage and pushes ~21 junk scalar keys into that buffer block.
  The generated-derived merge omits the junk (buffers ignore those keys either way; buffer
  frame sizes are numerically substituted by a separate pass, unaffected).
- All `run_live.sh` / `run_band.sh` grep contracts hold on generated files: column-0 stage
  names, flow-style `clock_profile:` and `bds_combiner: { … nh_assist: true … }` one-liners,
  double-quoted `seed_endpoint` / `base_dir` / CORS urls, `serial: 0x…` form, 4-space block
  indentation.

## Known-accidental divergences

Deliberately preserved as-is in `gnss_node.yaml` (byte-level current behavior is the
target), each flagged `# DIVERGENT (audit): …` at its row: doppler_step 100 (L1) vs 250 (L5)
on identical record geometry; acquire_windows 20/40/100 and acquire_snr 10/8/10 (brokers
hardcode `--acquire-snr 6`, so the config gate never reaches them); L5 gains 8/8/8 vs
11/11/11; hint_ttl_s 8 vs 20 within the L1 file; L2C's 4 s integration window vs ~1 s
fleet-wide; the L1 GPS tracker inheriting `signal` implicitly while every other tracker sets
it; L1's per-chain CUDA streams vs L5's serial stream-1 layout; bds combiners carrying no
`prns` list while gal combiners do. Fixing any of these is now a deliberate one-line edit in
the node table, followed by `--check` (which will then intentionally report the delta until
the regenerated files are installed).
