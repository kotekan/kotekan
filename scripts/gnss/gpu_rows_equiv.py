#!/usr/bin/env python3
"""Extract a node config's per-half GNSS GPU block for the row-equivalence tests.

    scripts/gnss/gpu_rows_equiv.py extract config/generated/chord_gnss_<node>_multi.yaml OUT.json

The output is a kotekan config fragment: the node config's root scalars, its gnss<g>_n2dual block,
and the host buffers that block names with their metadata pools, verbatim.
tests/boost/test_gnss_tile_selection reads it through GNSS_ROWS_CONFIG=OUT.json.
"""
import argparse
import json
import os
import sys

import yaml


def extract(config_path, gpu=0):
    """The root scalars, the gnss<gpu>_n2dual block and its buffers and pools of the node config
    at @p config_path."""
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.CSafeLoader)
    key = f"gnss{gpu}_n2dual"
    if key not in cfg:
        raise KeyError(f"{config_path}: no {key}")
    out = {k: v for k, v in cfg.items() if not isinstance(v, dict)}
    out[key] = cfg[key]
    bufs = list(cfg[key]["in_buffers"].values()) + list(
        cfg[key]["out_buffers"].values()
    )
    for b in bufs:
        out[b] = cfg[b]
        out[cfg[b]["metadata_pool"]] = cfg[cfg[b]["metadata_pool"]]
    out["source"] = os.path.basename(config_path)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser(
        "extract", help="write a node config's gnss<g>_n2dual block as JSON"
    )
    p.add_argument("config", help="generated node config (config/generated/*.yaml)")
    p.add_argument("out", help="JSON to write")
    p.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "extract":
        r = extract(a.config, a.gpu)
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1, sort_keys=True)
        print(f"{a.out}: gnss{a.gpu}_n2dual from {r['source']}", file=sys.stderr)


if __name__ == "__main__":
    main()
