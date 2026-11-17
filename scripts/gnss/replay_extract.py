#!/usr/bin/env python3
"""Extract production chain blocks from a generated node config for test_gnss_chain_replay.

    scripts/gnss/replay_extract.py config/generated/chord_gnss_<node>_multi.yaml OUT.json
    scripts/gnss/replay_extract.py CONFIG OUT.json --chains gnss0_gps_l5 gnss0_gal_e5a

Each chain's four host-stage blocks (<pre>_n2assemble_tiles, <pre>_n2assemble, <pre>_n2combine,
<pre>_telem_pack) are copied verbatim, a row of a host as the stage it stands for
(gnss_fleet.assembler_rows, combiner_rows, telem_rows), except that the buffer keys name the
replay's buffers and cpu_affinity is empty. A --gpu-rows config is refused. The test reads the
result through GNSS_REPLAY_CONFIG=OUT.json. The default chains are gnss0's gps_l5 and gal_e5a,
which share channels and PRNs, and glo_l2oc, the chain with a record_cp_mod_chips.
"""
import argparse
import json
import os
import sys

import yaml

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config")
)
from gen_fleet import _split_header  # noqa: E402
import gnss_fleet  # noqa: E402

DEFAULT_CHAINS = ("gnss0_gps_l5", "gnss0_gal_e5a", "gnss0_glo_l2oc")


def extract(config_path, chains=DEFAULT_CHAINS):
    """The replay config (a dict) for @p chains of the node config at @p config_path."""
    with open(config_path) as f:
        _, body = _split_header(f.read())
    cfg = yaml.load(body, Loader=yaml.CSafeLoader)
    stages = dict(cfg)
    stages.update(gnss_fleet.assembler_rows(cfg))
    stages.update(gnss_fleet.combiner_rows(cfg))
    stages.update(gnss_fleet.telem_rows(cfg))
    out = []
    for pre in chains:

        def block(suffix):
            key = pre + suffix
            if key not in stages:
                raise KeyError(f"{config_path}: no {key}")
            b = dict(stages[key])
            b["cpu_affinity"] = []
            return b

        if "gnss_row" in stages.get(pre + "_n2assemble", {}):
            raise ValueError(
                f"{config_path}: a --gpu-rows config has no {pre}_n2assemble_tiles,"
                " which the replay needs; extract from the node's config without"
                " --gpu-rows"
            )
        adapter, assemble = block("_n2assemble_tiles"), block("_n2assemble")
        combine, telem = block("_n2combine"), block("_telem_pack")
        adapter.update(
            ctl_buf=pre + "_n2ctl_buf",
            tiles_buf=pre + "_n2tiles_buf",
            out_buf=pre + "_n2epl_buf",
        )
        assemble.update(
            in_buf=pre + "_n2epl_buf",
            out_buf=pre + "_n2rec_buf",
            cube_buf=pre + "_cube_buf",
        )
        combine.update(in_bufs=[pre + "_n2rec_buf"], out_buf=pre + "_n2cmb_buf")
        # A combine host row's metric label: the replay sets its own on the rows it hosts.
        combine.pop("chain", None)
        telem.update(in_buf=pre + "_n2rec_buf", out_buf=pre + "_telem_buf")
        out.append(
            {
                "pre": pre,
                "gpu": int(pre[len("gnss")]),
                "adapter": adapter,
                "assemble": assemble,
                "combine": combine,
                "telem": telem,
            }
        )
    return {
        "source": os.path.basename(config_path),
        "root": {"samples_per_data_set": cfg["samples_per_data_set"]},
        "chains": out,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", help="generated node config (config/generated/*.yaml)")
    ap.add_argument("out", help="replay config to write (JSON)")
    ap.add_argument("--chains", nargs="+", default=list(DEFAULT_CHAINS))
    a = ap.parse_args()
    r = extract(a.config, a.chains)
    with open(a.out, "w") as f:
        json.dump(r, f, indent=1, sort_keys=True)
    print(f"{a.out}: {len(r['chains'])} chains from {r['source']}", file=sys.stderr)


if __name__ == "__main__":
    main()
