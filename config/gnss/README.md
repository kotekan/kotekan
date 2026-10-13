# config/gnss

`example_chord_gnss_cx51_multi.yaml` is one node's generated config, kept so the generator's
output can be read without a fleet. The configs the nodes start on are written to
`config/generated/` and are not tracked.

The node configs come from one manifest:

```
python3 scripts/gnss/gen_fleet.py config/gnss_fleet_chord.yaml           # write
python3 scripts/gnss/gen_fleet.py config/gnss_fleet_chord.yaml --check   # gate: regenerate in memory, byte-compare
```

`gen_fleet.py` runs `config/gen_chord_gnss_config.py` once per node with the flags the manifest
records, so every deployed config is reproducible from a committed file. The generator's base is
the stock render of `config/chord/pathfinder.j2` (no `gnss_node`), rendered exactly as kotekan
renders it, and the GNSS blocks are injected into that. `scripts/gnss/stock_parity.py` diffs the
non-GNSS half of every config written or checked against the stock render and fails on anything
not declared there with a reason. The EOP table and bffs's bad inputs are live data fetched at
generation time; `--check` normalises them out of the comparison.

The gather, aggregator and cube-archiver instances have their own recipes: the `REGENERATE WITH`
header of the gather config, `scripts/gnss/site/stack_components.sh` (aggregator) and
`scripts/gnss/site/cubearch_up.sh` (cube archiver).
