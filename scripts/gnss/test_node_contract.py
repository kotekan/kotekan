#!/usr/bin/env python3
"""node_contract.py on a generated cx52 config: the counts that config must give, and one
mutation per facet that must show in that facet's diff.

    GNSS_CONTRACT_BASELINE=DIR python3 scripts/gnss/test_node_contract.py

DIR holds chord_gnss_cx52_multi.yaml as kv/chord-gnss 61211e686 generates it; the counts are
that config's. Without GNSS_CONTRACT_BASELINE the test exits 77. With GNSS_REPLAY_DUMP=DIR2
(test_gnss_chain_replay's config.json and endpoints.json) rest_surface of the replay's config
must equal the per-chain endpoints its stages registered.
"""
import collections
import copy
import json
import os
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_fleet  # noqa: E402
import node_contract as nc  # noqa: E402

NODE = "cx52"
NAME = "chord_gnss_%s_multi.yaml" % NODE

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def snap_of(cfg, head, d):
    """The snapshot of cfg written as the generator writes it, under d."""
    path = os.path.join(d, NAME)
    with open(path, "w") as f:
        f.write(
            head
            + yaml.dump(
                cfg, Dumper=yaml.CSafeDumper, default_flow_style=False, sort_keys=True
            )
        )
    return nc.snapshot(path, NODE, NODE)


def facet_lines(lines, facet):
    return [ln for ln in lines if ln[2:].startswith(facet + ":")]


def test_counts(base):
    rest = base["rest"]
    kind = collections.Counter(p.split("/")[1].rsplit("_", 1)[1] for p in rest)
    check(
        len(rest) == 495
        and kind == {"inject": 120, "n2assemble": 285, "n2combine": 90},
        "495 per-chain paths, 120 inject + 285 assembler + 90 combiner (%d, %s)"
        % (len(rest), dict(kind)),
    )
    per = collections.Counter(p.split("/")[1].rsplit("_", 1)[0] for p in rest)
    split = collections.Counter(per.values())
    check(split == {16: 15, 17: 15}, "16 paths on 15 chains, 17 on 15 (%s)" % split)
    pl = base["placement"]
    n = sum(1 for b in pl["numa"] if b[:4] == "gnss" and b[4:5].isdigit())
    check(
        n > 0 and pl["misplaced"] == [],
        "%d gnss<g>_ buffers on NUMA node g (misplaced %s)" % (n, pl["misplaced"]),
    )
    st = base["stall"]
    check(
        len(st) == 30 and set(st.values()) == {8.75},
        "30 chains with a stall budget of 8.75 (%s)"
        % dict(collections.Counter(st.values())),
    )
    empty = [k for k, v in base["census"].items() if not v]
    check(not empty, "every census list is non-empty (empty: %s)" % empty)


def test_mutations(cfg, head, base, d):
    def run(name, mutate):
        c = copy.deepcopy(cfg)
        mutate(c)
        s = snap_of(c, head, d)
        lines = nc.diff(base, s)
        print("  %s: %d difference(s)" % (name, len(lines)))
        return s, lines

    _, lines = run("unchanged", lambda c: None)
    check(lines == [], "an unchanged config, rewritten, diffs empty: %s" % lines[:3])

    def rename(c):
        c["gnss0_gal_e5a_assemble"] = c.pop("gnss0_gal_e5a_n2assemble")

    _, lines = run("rename", rename)
    for facet in ("rest", "identity"):
        check(
            facet_lines(lines, facet),
            "renaming gnss0_gal_e5a_n2assemble shows in %s" % facet,
        )

    lane = sorted(
        k
        for k, v in cfg.items()
        if k.endswith("_n2assemble")
        and v.get("lane_transfer_freq_ids")
        and len(v.get("channel_ids", [])) >= 2
    )[0]
    _, lines = run("lane transfer", lambda c: c[lane].pop("lane_transfer_freq_ids"))
    check(
        "- rest: GET /%s/get_lane_inst" % lane in lines,
        "dropping %s's lane_transfer_freq_ids drops its get_lane_inst" % lane,
    )

    def tiles(c):
        c["gnss0_gps_l5_n2tiles_buf"]["num_frames"] = 2

    _, lines = run("tiles", tiles)
    check(
        "~ stall: /gnss0_gps_l5_n2assemble_tiles: 8.75 -> 6.75" in lines,
        "n2tiles num_frames 2 shows in stall: %s" % facet_lines(lines, "stall"),
    )

    def merge(c):
        rec = c["gnss0_gps_l5_n2rec_buf"]
        c["gnss0_gps_l5_n2merge_buf"] = {
            k: rec[k] for k in ("frame_size", "kotekan_buffer", "metadata_pool")
        }
        c["gnss0_gps_l5_n2merge_buf"].update(num_frames=4, numa_node=0)
        c["gnss0_gps_l5_n2merge"] = {
            "kotekan_stage": "bufferMerge",
            "in_bufs": ["gnss0_gps_l5_n2rec_buf"],
            "out_buf": "gnss0_gps_l5_n2merge_buf",
            "timeout": -1,
            "cpu_affinity": c["gnss0_gps_l5_telem_pack"]["cpu_affinity"],
        }
        c["gnss0_gps_l5_telem_pack"]["in_buf"] = "gnss0_gps_l5_n2merge_buf"

    _, lines = run("merge", merge)
    check(
        "+ stall: /gnss0_gps_l5_n2merge = -1.0" in lines,
        "a bufferMerge with timeout -1 between n2rec and telem_pack shows in stall: %s"
        % facet_lines(lines, "stall"),
    )

    def move(c):
        moved = [k for k in c if k.startswith("gnss0_") and k.endswith("_n2assemble")]
        c["gnss0_records"] = {"chains": {k: c.pop(k) for k in moved}}

    s, lines = run("moved blocks", move)
    cen = s["census"]
    was = base["census"]
    for tool in (
        "elem_shared_ref.endpoints",
        "elem_shared_ref.config_zeros",
        "lane_relay.inst_stage",
    ):
        g0 = [e for e in cen[tool] if e.startswith(("gnss0_", "0: "))]
        had = [e for e in was[tool] if e.startswith(("gnss0_", "0: "))]
        check(
            had and not g0,
            "%s finds none of the moved gnss0 assemblers (had %d, now %d)"
            % (tool, len(had), len(g0)),
        )
    check(facet_lines(lines, "census"), "the moved blocks show in census")


def test_replay(dump):
    """rest_surface of the replay's config against what its stages registered."""
    cpath = os.path.join(dump, "config.json")
    epath = os.path.join(dump, "endpoints.json")
    if not (os.path.exists(cpath) and os.path.exists(epath)):
        check(False, "GNSS_REPLAY_DUMP=%s holds config.json and endpoints.json" % dump)
        return
    with open(cpath) as f:
        want = nc.rest_surface(json.load(f))
    with open(epath) as f:
        have = nc.endpoint_set(json.load(f))
    check(
        want == have and want,
        "replay: %d paths restated, %d registered; missing %s, extra %s"
        % (len(want), len(have), sorted(want - have)[:3], sorted(have - want)[:3]),
    )


def main():
    d = os.environ.get("GNSS_CONTRACT_BASELINE")
    if not d:
        print("skip: GNSS_CONTRACT_BASELINE is not set")
        return 77
    path = os.path.join(d, NAME)
    with open(path) as f:
        text = f.read()
    head = gen_fleet._split_header(text)[0]
    cfg = nc.load(path)
    base = nc.snapshot(path, NODE, NODE)
    dump = os.environ.get("GNSS_REPLAY_DUMP")
    if not dump:
        print("GNSS_REPLAY_DUMP is not set: the replay cross-check does not run")
    with tempfile.TemporaryDirectory(prefix="node_contract_") as tmp:
        tests = [
            lambda: test_counts(base),
            lambda: test_mutations(cfg, head, base, tmp),
        ] + ([lambda: test_replay(dump)] if dump else [])
        for t in tests:
            try:
                t()
            except Exception as e:
                check(False, "raised %s: %s" % (type(e).__name__, e))
    if _fails:
        print("\n%d FAILED" % len(_fails))
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
