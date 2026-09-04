#!/usr/bin/env python3
"""Derive the live element set from the BASE CONFIG'S OWN dpdk keys and check it against
what the fleet manifest declares -- the gate that would have caught the 2026-08-31 restart.

WHY THIS EXISTS. `array.live_element_ranges` in config/chord_gnss_node.yaml is a HUMAN
RESTATEMENT of something the production config already knows: which CRS boards are present
(`missing_source_ids`), and which output location each one lands at (`dpdk.crs_board_remap`).
On 2026-08-31 production changed both -- the remap went identity -> pol-grouping and the
live boards moved from output locations 0,1,2,3 to 0,1,8,9 -- and nothing in our tree
noticed, because the restatement still said [0, 31] and every gate we owned tested our own
code against our own assumptions.

The failure that would have followed is the quiet kind: the mixed-tile gather takes the
first ceil(n_live/16) tile COLUMNS, so it would have read elements 0-31 = 16 live feeds and
16 dead panels, at full frame size, with correct-looking records. Half the aperture, no
error anywhere. A gate that compares the restatement to the source is the only thing that
fails loudly instead.

    scripts/gnss/live_element_gate.py [base.json] [node.yaml]
"""
import json
import os
import sys

import yaml

# scripts/gnss/<this> -> the repo root is two levels up.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUTS_PER_BOARD = 8


def live_from_base(base):
    """The element ranges the base config's own keys imply, as [(lo, hi), ...]."""
    dpdk = base["dpdk"]
    n_boards = int(base["num_crs_boards"])
    # Output location i is fed by the board whose raw source_id is remap[i]; absent = identity.
    remap = dpdk.get("crs_board_remap") or list(range(n_boards))
    # ⚠️ missing_source_ids is stated in OUTPUT-LOCATION order, not raw board order -- that is
    # what its own comment in the production config says, and getting it backwards silently
    # swaps which half of the array you believe in.
    missing = set(base.get("missing_source_ids")
                  or base.get("process_packet_mask_0", {}).get("missing_source_ids", []))
    live_locs = sorted(set(range(len(remap))) - missing)
    ranges, run = [], None
    for loc in live_locs:
        lo, hi = loc * INPUTS_PER_BOARD, (loc + 1) * INPUTS_PER_BOARD - 1
        if run and lo == run[1] + 1:
            run = (run[0], hi)
        else:
            if run:
                ranges.append(run)
            run = (lo, hi)
    if run:
        ranges.append(run)
    return ranges


def declared(node_yaml):
    arr = yaml.safe_load(open(node_yaml))["array"]
    rr = arr.get("live_element_ranges")
    if rr is None:
        lo, hi = arr["live_elements"]
        rr = [[lo, hi]]
    return [(int(a), int(b)) for a, b in rr]


def main():
    manifest = os.path.join(ROOT, "config", "gnss_fleet_chord.yaml")
    cfg_dir = os.path.dirname(manifest)
    base_path = (sys.argv[1] if len(sys.argv) > 1
                 else os.path.join(cfg_dir, yaml.safe_load(open(manifest))["base"]))
    node_path = (sys.argv[2] if len(sys.argv) > 2
                 else os.path.join(cfg_dir, "chord_gnss_node.yaml"))

    base = json.load(open(base_path))
    want, got = live_from_base(base), declared(node_path)
    fmt = lambda rr: ", ".join(f"[{a}..{b}]" for a, b in rr)
    print(f"base    {os.path.basename(base_path)}")
    print(f"  crs_board_remap    {base['dpdk'].get('crs_board_remap', '<identity>')}")
    print(f"  missing_source_ids {sorted(set(base.get('missing_source_ids', [])))}")
    print(f"  => live elements   {fmt(want)}  ({sum(b - a + 1 for a, b in want)} elements)")
    print(f"declared in {os.path.basename(node_path)}: {fmt(got)}")

    if want != got:
        print("\n*** MISMATCH -- the fleet would gather the WRONG element columns.")
        print("    Fix array.live_element_ranges in the node yaml, regenerate, dry-run.")
        return 1
    for lo, hi in got:
        if lo % 16 or (hi + 1) % 16:
            print(f"\n*** [{lo}..{hi}] is not 16-aligned -- a tile column would be half dead.")
            return 1
    cols = [c for lo, hi in got for c in range(lo // 16, (hi + 1) // 16)]
    print(f"tile columns to gather: {cols}   OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
