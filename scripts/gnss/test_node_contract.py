#!/usr/bin/env python3
"""node_contract.py on a generated cx52 config: the counts that config must give, and one
mutation per facet that must show in that facet's diff.

    GNSS_CONTRACT_BASELINE=DIR python3 scripts/gnss/test_node_contract.py

DIR holds a generated chord_gnss_cx52_multi.yaml; the counts are those of the config the
generator writes for cx52 today. Without GNSS_CONTRACT_BASELINE only test_only_keys is checked
and the test exits 77. With GNSS_REPLAY_DUMP=DIR2
(test_gnss_chain_replay's config.json and endpoints.json) rest_surface of the replay's config
must equal the per-chain endpoints its stages registered. With GNSS_CONTRACT_FOLDED=DIR3 (the
same node generated with --fold-epl) the folded config must keep every chain's stall budget at
8.75 or more and differ from DIR's only in placement and stall.
"""
import collections
import copy
import json
import os
import re
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


def held_zeros(cfg):
    """{assembler: the elements its elem_sum_shared_ref holds at 0, as the census prints
    config_zeros}, read from the parsed config."""
    out = {}
    for u, s in nc._stages(cfg, "GnssGpuRecordAssemble"):
        r = s[0].get("elem_sum_shared_ref") or []
        z = [e for e in range(len(r) // 2) if r[2 * e] == 0.0 and r[2 * e + 1] == 0.0]
        out[u.rsplit("/", 1)[1]] = ",".join(map(str, z))
    return out


def tool_zeros(census):
    return dict(e.split(": ", 1) for e in census["elem_shared_ref.config_zeros"])


def add_merge(c, name, src, stage, key):
    """Put a bufferMerge with timeout -1 between buffer src and stage's key."""
    keep = ("frame_size", "kotekan_buffer", "metadata_pool", "num_frames", "numa_node")
    c[name + "_buf"] = {k: c[src][k] for k in keep}
    c[name] = {
        "kotekan_stage": "bufferMerge",
        "in_bufs": [src],
        "out_buf": name + "_buf",
        "timeout": -1,
        "cpu_affinity": c[stage]["cpu_affinity"],
    }
    c[stage][key] = name + "_buf"


def test_counts(base, cfg):
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
    cen = base["census"]
    empty = [k for k, v in cen.items() if not v]
    check(not empty, "every census list is non-empty (empty: %s)" % empty)
    none = [e for e in cen["gnss_fleet.stage_chain"] if e.endswith("-> None")]
    check(not none, "stage_chain names every owner and sender (not: %s)" % none[:3])
    bad = [
        e for e in cen["viscap_read.primary_tag"] if e.split(": ", 1)[1][:4] == "exit"
    ]
    check(not bad, "primary_tag finds both GPUs' primary (%s)" % bad)
    built = {
        c: {int(g) for g in gs.split(",")}
        for c, gs in (e.split(": ") for e in cen["gen_fleet.built_instances"])
    }
    owners, lost = 0, []
    for e in cen["gnss_fleet.stage_chain"]:
        m = re.match(r"owner /gnss(\d+)_\S+ -> (\S+)$", e)
        if m:
            owners += 1
            if int(m.group(1)) not in built.get(m.group(2), ()):
                lost.append(e)
    check(
        owners == 30 and not lost,
        "built_instances holds all %d assemblers' chains (not: %s)" % (owners, lost),
    )
    held = held_zeros(cfg)
    check(
        tool_zeros(cen) == held and any(held.values()),
        "config_zeros reads the zeros every assembler's elem_sum_shared_ref holds",
    )


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

    inject = [
        x for x in cfg["gnss0_n2dual"]["commands"] if x.get("name") == "cudaGnssInject"
    ][0]
    for where, block in (("an inject command", inject), ("the root", cfg)):
        block["psi_sync_build"] = True
        try:
            snap_of(cfg, head, d)
            refused = ""
        except ValueError as e:
            refused = str(e)
        del block["psi_sync_build"]
        check(
            "psi_sync_build" in refused,
            "snapshot refuses psi_sync_build set on %s: %r" % (where, refused),
        )

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

    def ctl(c):
        c["gnss0_gps_l5_n2ctl_buf"]["num_frames"] = 2

    _, lines = run("ctl", ctl)
    check(
        "~ stall: /gnss0_gps_l5_n2assemble_tiles: 8.75 -> 6.75" in lines,
        "n2ctl num_frames 2 shows in stall: %s" % facet_lines(lines, "stall"),
    )

    merges = (
        ("n2rec and telem_pack", "n2rec_buf", "telem_pack", "in_buf"),
        ("n2epl and the assembler", "n2epl_buf", "n2assemble", "in_buf"),
        ("n2ctl and the adapter", "n2ctl_buf", "n2assemble_tiles", "ctl_buf"),
    )
    p = "gnss0_gps_l5_"
    for where, src, stage, key in merges:
        _, lines = run(
            "merge " + src,
            lambda c: add_merge(c, p + "merge", p + src, p + stage, key),
        )
        check(
            facet_lines(lines, "stall") == ["+ stall: /gnss0_gps_l5_merge = -1.0"],
            "a bufferMerge with timeout -1 between %s shows in stall: %s"
            % (where, facet_lines(lines, "stall")),
        )

    _, lines = run("no assembler", lambda c: c.pop("gnss0_gps_l5_n2assemble"))
    check(
        "~ stall: /gnss0_gps_l5_n2assemble_tiles: 8.75 -> 'unresolved'" in lines,
        "an epl with no assembler shows in stall: %s" % facet_lines(lines, "stall"),
    )

    moved = [k for k in cfg if k.startswith("gnss0_") and k.endswith("_n2assemble")]

    def move(c):
        c["gnss0_records"] = {"chains": {k: c.pop(k) for k in moved}}

    s, lines = run("moved blocks", move)
    cen = s["census"]
    was = base["census"]
    for tool in ("elem_shared_ref.endpoints", "lane_relay.inst_stage"):
        g0 = [e for e in cen[tool] if e.startswith(("gnss0_", "0: "))]
        had = [e for e in was[tool] if e.startswith(("gnss0_", "0: "))]
        check(
            had and not g0,
            "%s finds none of the moved gnss0 assemblers (had %d, now %d)"
            % (tool, len(had), len(g0)),
        )
    held = held_zeros(cfg)
    read = tool_zeros(cen)
    check(
        set(moved) <= set(read)
        and any(held[k] for k in moved)
        and not any(read[k] for k in moved),
        "config_zeros, asked about each moved gnss0 assembler, reads none of its zeros",
    )
    lost = [e for e in cen["gnss_fleet.stage_chain"] if e.startswith("owner /gnss0_")]
    check(
        len(lost) == len(moved) and all(e.endswith("-> None") for e in lost),
        "lane_relay.OWNER parses none of the %d moved gnss0 lane owners" % len(lost),
    )
    check(facet_lines(lines, "census"), "the moved blocks show in census")


def test_fold(base, d):
    s = nc.snapshot(os.path.join(d, NAME), NODE, NODE)
    st = s["stall"]
    asm = {k: v for k, v in st.items() if k.endswith("_n2assemble")}
    check(
        len(asm) == 30 and len(st) == 30 and min(asm.values()) >= 8.75,
        "folded: 30 assemblers, each with a stall budget of 8.75 or more (%s)"
        % dict(collections.Counter(st.values())),
    )
    lines = nc.diff(base, s)
    facets = {ln[2:].split(":", 1)[0] for ln in lines}
    check(
        facets <= {"placement", "stall"},
        "folded: only placement and stall differ (%s)" % sorted(facets),
    )
    place = [ln for ln in lines if ln[2:].startswith("placement:")]
    stages = [ln.split(" ")[3] for ln in place if ln.startswith("- placement: cpu ")]
    bufs = [ln.split(" ")[3] for ln in place if ln.startswith("- placement: numa ")]
    check(
        len(stages) == 30
        and all(x.endswith("_n2assemble_tiles") for x in stages)
        and len(bufs) == 30
        and all(x.endswith("_n2epl_buf") for x in bufs)
        and len(place) == 60,
        "folded: placement loses the 30 adapters and the 30 epl buffers, nothing else "
        "(%d stages, %d buffers, %d lines)" % (len(stages), len(bufs), len(place)),
    )


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


def test_test_only():
    tree = {
        "gnss0_n2dual": {
            "commands": [{"name": "cudaGnssInject", "psi_sync_build": False}]
        },
        "psi_sync_build": True,
    }
    found = nc.test_only_keys(tree)
    check(
        found == ["/: psi_sync_build", "/gnss0_n2dual/commands/0: psi_sync_build"],
        "test_only_keys finds psi_sync_build at the root and on a command: %s" % found,
    )
    check(
        nc.test_only_keys({"gnss0_n2dual": {"commands": [{}]}}) == [],
        "test_only_keys finds nothing in a tree without a test-only key",
    )


def main():
    test_test_only()
    d = os.environ.get("GNSS_CONTRACT_BASELINE")
    if not d:
        if _fails:
            print("\n%d FAILED" % len(_fails))
            return 1
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
        folded = os.environ.get("GNSS_CONTRACT_FOLDED")
        tests = (
            [
                lambda: test_counts(base, cfg),
                lambda: test_mutations(cfg, head, base, tmp),
            ]
            + ([lambda: test_replay(dump)] if dump else [])
            + ([lambda: test_fold(base, folded)] if folded else [])
        )
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
