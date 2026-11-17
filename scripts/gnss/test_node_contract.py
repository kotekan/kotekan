#!/usr/bin/env python3
"""node_contract.py on a generated cx52 config: the counts that config must give, and one
mutation per facet that must show in that facet's diff.

    GNSS_CONTRACT_BASELINE=DIR python3 scripts/gnss/test_node_contract.py

DIR holds a generated chord_gnss_cx52_multi.yaml; the counts are those of the config the
generator writes for cx52 today. Without GNSS_CONTRACT_BASELINE only test_only_keys is checked
and the test exits 77. With GNSS_REPLAY_DUMP=DIR2
(test_gnss_chain_replay's config.json and endpoints.json) rest_surface of the replay's config
must equal the per-chain endpoints its stages registered. With GNSS_CONTRACT_FOLDED=DIR3 (the
same node generated with --fold-epl) every folded assembler must carry its adapter's inputs
and keys, keep its chain's stall budget at 8.75 or more and differ from DIR's only in placement
and stall. With GNSS_CONTRACT_ROWS=DIR4 (the same node generated with --fold-epl --gpu-rows)
every assembler and injector must read its row of its half's gnss_rows, and DIR4 must differ
from DIR3 only in the placement of the tiles and ctl buffers: 60 per-chain ones out, 4 per-half
ones in, on the half's NUMA node, holding the same bytes. With GNSS_CONTRACT_HOSTED=DIR5 (the
same node generated with --fold-epl --gpu-rows --inject-host) its contract must equal DIR4's, and
each half must have one injector whose rows, merged with its own keys, are DIR4's injectors.
With GNSS_CONTRACT_RECORDS=DIR6 (DIR4's flags and --record-host) its contract must be DIR4's plus
each half's record host, whose rows, read by gnss_fleet.assembler_rows, must be DIR4's
assemblers; GNSS_CONTRACT_RECORDS_HOSTED=DIR7 (DIR5's flags and --record-host) likewise
against DIR5. With GNSS_CONTRACT_COMBINE=DIR8 (DIR7's flags and --combine-host) its contract must be
DIR7's plus each half's combine host, whose rows, read by gnss_fleet.combiner_rows, must be DIR7's
combiners; GNSS_CONTRACT_COMBINE_ALONE=DIR9 (--combine-host alone) likewise against DIR. With
GNSS_CONTRACT_TELEM=DIR10 (DIR8's flags and --telem-pack-host) its contract must be DIR8's but for
each half's telemetry host and one GnssTelemSend in place of its chains' bufferSends, to the same
address, with the host's rows, read by gnss_fleet.telem_rows, DIR8's packs;
GNSS_CONTRACT_TELEM_ALONE=DIR11 (--telem-pack-host alone) likewise against DIR.
"""
import collections
import copy
import json
import operator
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
    cmb = [e for e in cen["gnss_fleet.stage_chain"] if e.startswith("combiner ")]
    check(
        len(cmb) == 30
        and all(
            re.match(r"combiner /(gnss\d_\w+)_n2combine -> \w+$", e)
            and "GET %s/get_status" % e.split(" ")[1] in rest
            for e in cmb
        ),
        "stage_chain names all 30 combiners, each one rest_surface serves (%d)"
        % len(cmb),
    )
    tel = [e for e in cen["gnss_fleet.stage_chain"] if e.startswith("telem ")]
    check(
        len(tel) == 30
        and all(
            re.match(r"telem /gnss(\d)_(\w+)_telem_pack -> \2, sends \2/cx52\.\1$", e)
            for e in tel
        ),
        "stage_chain names all 30 telemetry packs, each sending its chain from its half (%s)"
        % [e for e in tel if not re.search(r"-> (\w+), sends \1/", e)][:3],
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
        """gnss0's assemblers as rows of one host, as --record-host hosts them."""
        rows = {k[: -len("_n2assemble")]: c.pop(k) for k in moved}
        workers = [
            {"rows": [k], "cpu_affinity": r.pop("cpu_affinity")}
            for k, r in rows.items()
        ]
        for r in rows.values():
            del r["kotekan_stage"]
        c["gnss0_records"] = {
            "kotekan_stage": "GnssGpuRecordAssemble",
            "cpu_affinity": sorted({x for w in workers for x in w["cpu_affinity"]}),
            "chains": rows,
            "workers": workers,
        }

    s, lines = run("moved blocks", move)
    check(
        lines
        == [
            "+ placement: cpu /gnss0_records = %r"
            % s["placement"]["cpu"]["/gnss0_records"]
        ],
        "the moved blocks show only as their host's cores: %s" % lines[:4],
    )

    # The same move read by tools that see only a top-level stage per assembler, as before
    # gnss_fleet.assembler_rows read hosts: elem_shared_ref and lane_relay find none of gnss0's.
    rows_of = nc.gnss_fleet.assembler_rows

    def top_level(c):
        return [(n, b) for n, b in rows_of(c) if n in c]

    nc.gnss_fleet.assembler_rows = top_level
    nc.elem_shared_ref._REFS.clear()
    try:
        s, lines = run("moved blocks, top-level tools", move)
    finally:
        nc.gnss_fleet.assembler_rows = rows_of
        nc.elem_shared_ref._REFS.clear()
    cen = s["census"]
    was = base["census"]
    lever_census, lever_rest = facet_lines(lines, "census"), facet_lines(lines, "rest")
    for tool in ("elem_shared_ref.endpoints", "lane_relay.inst_stage"):
        g0 = [e for e in cen[tool] if e.startswith(("gnss0_", "0: "))]
        had = [e for e in was[tool] if e.startswith(("gnss0_", "0: "))]
        check(
            had and not g0,
            "top-level tools: %s finds none of the moved gnss0 assemblers (had %d, now %d)"
            % (tool, len(had), len(g0)),
        )
    check(
        lever_census and lever_rest,
        "top-level tools: the moved blocks show in census and rest",
    )

    combiners = [k for k in cfg if k.startswith("gnss0_") and k.endswith("_n2combine")]

    def move_combiners(c):
        """gnss0's combiners as rows of one host, as --combine-host hosts them."""
        rows = {k[: -len("_n2combine")]: c.pop(k) for k in combiners}
        workers = [
            {"rows": [k], "cpu_affinity": r.pop("cpu_affinity")}
            for k, r in rows.items()
        ]
        for r in rows.values():
            del r["kotekan_stage"]
        c["gnss0_combine"] = {
            "kotekan_stage": "GnssCoherentCombiner",
            "cpu_affinity": sorted({x for w in workers for x in w["cpu_affinity"]}),
            "chains": rows,
            "workers": workers,
        }

    s, lines = run("moved combiners", move_combiners)
    check(
        len(combiners) == 16
        and lines
        == [
            "+ placement: cpu /gnss0_combine = %r"
            % s["placement"]["cpu"]["/gnss0_combine"]
        ],
        "the moved combiners show only as their host's cores: %s" % lines[:4],
    )

    # The same move read through a combiner_rows that sees only top-level stages: the census
    # and the REST surface lose gnss0's combiners.
    cmb_of = nc.gnss_fleet.combiner_rows
    nc.gnss_fleet.combiner_rows = lambda c: [(n, b) for n, b in cmb_of(c) if n in c]
    try:
        s, lines = run("moved combiners, top-level tools", move_combiners)
    finally:
        nc.gnss_fleet.combiner_rows = cmb_of
    lost = ["- census: gnss_fleet.stage_chain combiner /%s -> " % k for k in combiners]
    cen = facet_lines(lines, "census")
    check(
        len(cen) == 16
        and all(any(ln.startswith(x) for ln in cen) for x in lost)
        and len(facet_lines(lines, "rest")) == 16 * 3,
        "top-level tools: the moved combiners leave the census (%d lines) and rest (%d)"
        % (len(cen), len(facet_lines(lines, "rest"))),
    )

    packs = [k for k in cfg if k.startswith("gnss0_") and k.endswith("_telem_pack")]

    def move_packs(c):
        """gnss0's telemetry packs as rows of one host, as --telem-pack-host hosts them."""
        rows = {k[: -len("_telem_pack")]: c.pop(k) for k in packs}
        workers = [
            {"rows": [k], "cpu_affinity": r.pop("cpu_affinity")}
            for k, r in rows.items()
        ]
        for r in rows.values():
            del r["kotekan_stage"]
        c["gnss0_telem"] = {
            "kotekan_stage": "GnssTelemPack",
            "cpu_affinity": sorted({x for w in workers for x in w["cpu_affinity"]}),
            "chains": rows,
            "workers": workers,
        }

    s, lines = run("moved telemetry packs", move_packs)
    check(
        len(packs) == 16
        and lines
        == [
            "+ placement: cpu /gnss0_telem = %r" % s["placement"]["cpu"]["/gnss0_telem"]
        ],
        "the moved telemetry packs show only as their host's cores: %s" % lines[:4],
    )

    # The same move read through a telem_rows that sees only top-level stages: the census and
    # the telemetry identity lose gnss0's packs.
    tel_of = nc.gnss_fleet.telem_rows
    nc.gnss_fleet.telem_rows = lambda c: [(n, b) for n, b in tel_of(c) if n in c]
    try:
        s, lines = run("moved telemetry packs, top-level tools", move_packs)
    finally:
        nc.gnss_fleet.telem_rows = tel_of
    lost = ["- census: gnss_fleet.stage_chain telem /%s -> " % k for k in packs]
    cen = facet_lines(lines, "census")
    check(
        len(cen) == 16
        and all(any(ln.startswith(x) for ln in cen) for x in lost)
        and len(facet_lines(lines, "identity")) == 16 * 5,
        "top-level tools: the moved packs leave the census (%d lines) and identity (%d)"
        % (len(cen), len(facet_lines(lines, "identity"))),
    )


def test_fold(base, cfg, d):
    path = os.path.join(d, NAME)
    s = nc.snapshot(path, NODE, NODE)
    st = s["stall"]
    asm = {k: v for k, v in st.items() if k.endswith("_n2assemble")}
    check(
        len(asm) == 30 and len(st) == 30 and min(asm.values()) >= 8.75,
        "folded: 30 assemblers, each with a stall budget of 8.75 or more (%s)"
        % dict(collections.Counter(st.values())),
    )
    # Each folded assembler sets its adapter's inputs and shape keys itself.
    folded = dict(nc._stages(nc.load(path), "GnssGpuRecordAssemble"))
    keys = (
        "tiles_buf",
        "ctl_buf",
        "num_elements",
        "num_synth",
        "lanes_per_slot",
        "hops_per_record",
    )
    (fft,) = {b[0]["fft_length"] for _, b in nc._blocks(cfg) if "fft_length" in b[0]}
    bad = []
    for u, ad in nc._stages(cfg, "GnssN2RecordAssemble"):
        a = folded.get(u[: -len("_tiles")], [{}])[0]
        want = {k: ad[0].get(k) for k in keys}
        want.update(fft_len=fft, in_buf=None)
        if {k: a.get(k) for k in want} != want:
            bad.append(u)
    check(
        len(folded) == 30 and not bad,
        "folded: every assembler carries its adapter's inputs and keys, and no in_buf (%s)"
        % bad,
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


def test_rows(fold_dir, rows_dir):
    fpath, rpath = os.path.join(fold_dir, NAME), os.path.join(rows_dir, NAME)
    fcfg, rcfg = nc.load(fpath), nc.load(rpath)
    lines = nc.diff(nc.snapshot(fpath, NODE, NODE), nc.snapshot(rpath, NODE, NODE))
    facets = {ln[2:].split(":", 1)[0] for ln in lines}
    check(facets <= {"placement"}, "rows: only placement differs (%s)" % sorted(facets))
    gone = [ln.split(" ")[3] for ln in lines if ln.startswith("- placement: numa ")]
    new = [ln for ln in lines if ln.startswith("+ placement: numa ")]
    want = [
        "+ placement: numa gnss%d_rows_%s_buf = %d" % (g, k, g)
        for g in (0, 1)
        for k in ("ctl", "tiles")
    ]
    check(
        len(gone) == 60
        and all(re.match(r"gnss\d_\w+_n2(tiles|ctl)_buf$", b) for b in gone)
        and sorted(new) == sorted(want)
        and len(lines) == 64,
        "rows: placement loses the 60 n2tiles and n2ctl buffers and gains the 4 per-half "
        "ones on their half's node, nothing else (%d out, %s in, %d lines)"
        % (len(gone), new, len(lines)),
    )

    def nbytes(cfg, names):
        return sum(cfg[b]["frame_size"] * cfg[b]["num_frames"] for b in names)

    added = [ln.split(" ")[3] for ln in new]
    check(
        nbytes(fcfg, gone) == nbytes(rcfg, added),
        "rows: the per-half buffers hold the bytes the per-chain ones did (%d, %d)"
        % (nbytes(fcfg, gone), nbytes(rcfg, added)),
    )
    # Each row's assembler carries its half's gnss_rows entry, and each injector names its row
    # and none of the keys the row replaces.
    bad = []
    for g in (0, 1):
        half = rcfg["gnss%d_n2dual" % g]
        rows = {r["name"]: r for r in half["gnss_rows"]}
        fg = sum(len(r["channels"]) for r in rows.values())
        for c in half["commands"]:
            if c["name"] != "cudaGnssInject":
                continue
            r = rows.get(c.get("gnss_row"))
            if r is None or {
                "num_synth",
                "synth_lane_base",
                "gnss_local_channels",
                "gnss_ctl_name",
            } & set(c):
                bad.append(c.get("gnss_row"))
                continue
            a = rcfg[r["name"] + "_n2assemble"]
            if (
                a.get("gnss_row") != r
                or a.get("fg_total") != fg
                or a.get("tiles_buf") != "gnss%d_rows_tiles_buf" % g
                or a.get("ctl_buf") != "gnss%d_rows_ctl_buf" % g
            ):
                bad.append(r["name"])
        if len(rows) != len(
            [c for c in half["commands"] if c["name"] == "cudaGnssInject"]
        ):
            bad.append("gnss%d rows" % g)
    check(not bad, "rows: every injector and assembler reads its half's row (%s)" % bad)


def test_hosted(rows_dir, hosted_dir):
    rpath, hpath = os.path.join(rows_dir, NAME), os.path.join(hosted_dir, NAME)
    rcfg, hcfg = nc.load(rpath), nc.load(hpath)
    lines = nc.diff(nc.snapshot(rpath, NODE, NODE), nc.snapshot(hpath, NODE, NODE))
    check(not lines, "hosted: the contract equals the rows form's (%s)" % lines[:5])
    bad = []
    for g in (0, 1):
        half = "gnss%d_n2dual" % g
        want = [c for c in rcfg[half]["commands"] if c["name"] == "cudaGnssInject"]
        got = [c for c in hcfg[half]["commands"] if c["name"] == "cudaGnssInject"]
        if len(got) != 1 or "chains" not in got[0]:
            bad.append("%s: %d injectors" % (half, len(got)))
            continue
        host = {k: v for k, v in got[0].items() if k != "chains"}
        rows = [dict(host, **r) for r in got[0]["chains"].values()]
        key = operator.itemgetter("gnss_row")
        if sorted(rows, key=key) != sorted(want, key=key) or any(
            k != r["gnss_row"] for k, r in got[0]["chains"].items()
        ):
            bad.append("%s: rows" % half)
        others = [c for c in hcfg[half]["commands"] if c["name"] != "cudaGnssInject"]
        if others != [
            c for c in rcfg[half]["commands"] if c["name"] != "cudaGnssInject"
        ]:
            bad.append("%s: other commands" % half)
        if {k: v for k, v in hcfg[half].items() if k != "commands"} != {
            k: v for k, v in rcfg[half].items() if k != "commands"
        }:
            bad.append("%s: block keys" % half)
    rest = {k: v for k, v in hcfg.items() if not re.match(r"gnss\d_n2dual$", k)}
    if rest != {k: v for k, v in rcfg.items() if not re.match(r"gnss\d_n2dual$", k)}:
        bad.append("blocks outside the halves")
    check(
        not bad,
        "hosted: one injector per half hosts the rows form's injectors (%s)" % bad,
    )


def test_records(rows_dir, rec_dir):
    """A config generated with --record-host against the same flags without it."""
    rpath, hpath = os.path.join(rows_dir, NAME), os.path.join(rec_dir, NAME)
    rcfg, hcfg = nc.load(rpath), nc.load(hpath)
    hs = nc.snapshot(hpath, NODE, NODE)
    lines = nc.diff(nc.snapshot(rpath, NODE, NODE), hs)
    want = [
        "+ placement: cpu /gnss%d_records = %r"
        % (g, hs["placement"]["cpu"]["/gnss%d_records" % g])
        for g in (0, 1)
    ]
    check(
        sorted(lines) == want,
        "records: the contract is that without the flag, plus each half's host (%s)"
        % lines[:5],
    )
    asm = {
        k: v
        for k, v in rcfg.items()
        if isinstance(v, dict) and v.get("kotekan_stage") == "GnssGpuRecordAssemble"
    }
    rows = dict(nc.gnss_fleet.assembler_rows(hcfg))
    check(
        len(asm) == 30 and rows == asm,
        "records: the hosts' rows, merged with their host's keys and their worker's cores, "
        "are the assemblers without the flag (%d, %d)" % (len(asm), len(rows)),
    )
    bad = []
    for g in (0, 1):
        host = hcfg.get("gnss%d_records" % g) or {}
        keys = [r["name"] for r in hcfg["gnss%d_n2dual" % g]["gnss_rows"]]
        workers = [
            {"rows": [k], "cpu_affinity": asm[k + "_n2assemble"]["cpu_affinity"]}
            for k in keys
        ]
        if host.get("workers") != workers or sorted(host.get("chains") or {}) != sorted(
            keys
        ):
            bad.append("gnss%d: one worker per row in gnss_rows order on its core" % g)
        if host.get("cpu_affinity") != sorted(
            {c for w in workers for c in w["cpu_affinity"]}
        ):
            bad.append("gnss%d: host cores" % g)
        for k, r in (host.get("chains") or {}).items():
            if set(r) & set(host) or not {"out_buf", "prns", "channel_ids"} <= set(r):
                bad.append("gnss%d: row %s keys" % (g, k))

    def others(c):
        return {
            k: v
            for k, v in c.items()
            if not (k.endswith("_n2assemble") or re.match(r"gnss\d_records$", k))
        }

    if others(rcfg) != others(hcfg):
        bad.append("blocks outside the assemblers")
    check(
        not bad,
        "records: each half's host runs every row on its own worker on its core, sets no row "
        "key, and nothing else changes (%s)" % bad,
    )
    name, block = rows.popitem()
    hcfg[name] = block
    try:
        nc.rest_surface(hcfg)
        refused = ""
    except ValueError as e:
        refused = str(e)
    check(
        name in refused,
        "records: a row named as a top-level block is refused: %r" % refused,
    )


def test_combine(a_dir, c_dir):
    """A config generated with --combine-host against the same flags without it."""
    apath, cpath = os.path.join(a_dir, NAME), os.path.join(c_dir, NAME)
    acfg, ccfg = nc.load(apath), nc.load(cpath)
    cs = nc.snapshot(cpath, NODE, NODE)
    lines = nc.diff(nc.snapshot(apath, NODE, NODE), cs)
    want = [
        "+ placement: cpu /gnss%d_combine = %r"
        % (g, cs["placement"]["cpu"]["/gnss%d_combine" % g])
        for g in (0, 1)
    ]
    check(
        sorted(lines) == want,
        "combine: the contract is that without the flag, plus each half's host (%s)"
        % lines[:5],
    )
    cmb = {
        k: v
        for k, v in acfg.items()
        if isinstance(v, dict) and v.get("kotekan_stage") == "GnssCoherentCombiner"
    }
    rows = dict(nc.gnss_fleet.combiner_rows(ccfg))
    chains = {k: r.pop("chain", None) for k, r in rows.items()}
    check(
        len(cmb) == 30 and rows == cmb,
        "combine: the hosts' rows, merged with their host's keys and their worker's cores, "
        "are the combiners without the flag (%d, %d)" % (len(cmb), len(rows)),
    )
    check(
        all(c == nc.gnss_fleet.stage_chain(k, "n2combine") for k, c in chains.items()),
        "combine: each row's chain is its broker chain (%s)"
        % [
            k
            for k, c in chains.items()
            if c != nc.gnss_fleet.stage_chain(k, "n2combine")
        ],
    )
    bad = []
    for g in (0, 1):
        host = ccfg.get("gnss%d_combine" % g) or {}
        keys = [k[: -len("_n2combine")] for k in cmb if k.startswith("gnss%d_" % g)]
        workers = host.get("workers") or []
        if sorted(r for w in workers for r in w["rows"]) != sorted(keys) or any(
            w["cpu_affinity"] != cmb[w["rows"][0] + "_n2combine"]["cpu_affinity"]
            or len(w["rows"]) != 1
            for w in workers
        ):
            bad.append("gnss%d: one worker per row on its core" % g)
        if host.get("cpu_affinity") != sorted(
            {c for w in workers for c in w["cpu_affinity"]}
        ):
            bad.append("gnss%d: host cores" % g)
        for k, r in (host.get("chains") or {}).items():
            if set(r) & set(host) or not {"in_bufs", "n_prn"} <= set(r):
                bad.append("gnss%d: row %s keys" % (g, k))

    def others(c):
        return {
            k: v
            for k, v in c.items()
            if not (k.endswith("_n2combine") or re.match(r"gnss\d_combine$", k))
        }

    if others(acfg) != others(ccfg):
        bad.append("blocks outside the combiners")
    check(
        not bad,
        "combine: each half's host runs every row on its own worker on its core, sets no row "
        "key, and nothing else changes (%s)" % bad,
    )


def test_telem(a_dir, t_dir):
    """A config generated with --telem-pack-host against the same flags without it."""
    apath, tpath = os.path.join(a_dir, NAME), os.path.join(t_dir, NAME)
    acfg, tcfg = nc.load(apath), nc.load(tpath)
    a_s, t_s = nc.snapshot(apath, NODE, NODE), nc.snapshot(tpath, NODE, NODE)
    lines = nc.diff(a_s, t_s)
    sends = {
        u: v for u, v in a_s["identity"]["send"].items() if u.endswith("_telem_send")
    }
    want = sorted(
        ["- identity: send %s = %r" % (u, v) for u, v in sends.items()]
        + [
            "+ identity: send /gnss%d_telem_send = %r"
            % (g, t_s["identity"]["send"].get("/gnss%d_telem_send" % g))
            for g in (0, 1)
        ]
        + [
            "+ placement: cpu %s = %r" % (u, t_s["placement"]["cpu"][u])
            for u in (
                "/gnss0_telem",
                "/gnss0_telem_send",
                "/gnss1_telem",
                "/gnss1_telem_send",
            )
        ]
        + ["- placement: cpu %s = %r" % (u, a_s["placement"]["cpu"][u]) for u in sends]
        + [
            "%s placement: numa %s = %r" % (sign, b, snap["placement"]["numa"][b])
            for sign, snap in (("-", a_s), ("+", t_s))
            for b in snap["placement"]["numa"]
            if re.match(r"gnss\d(_\w+)?_telem_buf$", b)
        ]
    )
    addr = set(sends.values())
    check(
        len(sends) == 30
        and len(addr) == 1
        and {t_s["identity"]["send"].get("/gnss%d_telem_send" % g) for g in (0, 1)}
        == addr
        and sorted(lines) == want,
        "telem: the contract is that without the flag but for each half's host, buffer and "
        "send, to the one address its chains sent to (%s)"
        % sorted(set(lines) ^ set(want))[:5],
    )
    packs = {
        k: v
        for k, v in acfg.items()
        if isinstance(v, dict) and v.get("kotekan_stage") == "GnssTelemPack"
    }
    rows = dict(nc.gnss_fleet.telem_rows(tcfg))
    bufs = {g: [] for g in (0, 1)}
    for k, v in packs.items():
        bufs[int(k[4])].append(acfg[v.pop("out_buf")])
    out = {k: r.pop("out_buf") for k, r in rows.items()}
    bad = []
    if not (len(packs) == 30 and rows == packs):
        bad.append("rows are not the packs (%d, %d)" % (len(packs), len(rows)))
    for g in (0, 1):
        tb = tcfg.get("gnss%d_telem_buf" % g) or {}
        if {o for k, o in out.items() if k.startswith("gnss%d_" % g)} != {
            "gnss%d_telem_buf" % g
        }:
            bad.append("gnss%d: rows write another buffer" % g)
        if tb.get("num_frames") != sum(b["num_frames"] for b in bufs[g]) or tb.get(
            "frame_size"
        ) != max(b["frame_size"] for b in bufs[g]):
            bad.append("gnss%d_telem_buf size" % g)
        send = tcfg.get("gnss%d_telem_send" % g) or {}
        if send.get("kotekan_stage") != "GnssTelemSend" or not send.get("drop_frames"):
            bad.append("gnss%d_telem_send" % g)

    def others(c):
        return {
            k: v
            for k, v in c.items()
            if not re.match(r"gnss\d(_\w+)?_telem_(pack|buf|send)$", k)
            and not re.match(r"gnss\d_telem$", k)
        }

    if others(acfg) != others(tcfg):
        bad.append("blocks outside the telemetry legs")
    check(
        not bad,
        "telem: each half's rows are its packs, writing one buffer as deep as theirs and as "
        "wide as the widest, sent by one GnssTelemSend, and nothing else changes (%s)"
        % bad,
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
        rows = os.environ.get("GNSS_CONTRACT_ROWS")
        hosted = os.environ.get("GNSS_CONTRACT_HOSTED")
        records = os.environ.get("GNSS_CONTRACT_RECORDS")
        records_hosted = os.environ.get("GNSS_CONTRACT_RECORDS_HOSTED")
        combine = os.environ.get("GNSS_CONTRACT_COMBINE")
        combine_alone = os.environ.get("GNSS_CONTRACT_COMBINE_ALONE")
        telem = os.environ.get("GNSS_CONTRACT_TELEM")
        telem_alone = os.environ.get("GNSS_CONTRACT_TELEM_ALONE")
        tests = (
            [
                lambda: test_counts(base, cfg),
                lambda: test_mutations(cfg, head, base, tmp),
            ]
            + ([lambda: test_replay(dump)] if dump else [])
            + ([lambda: test_fold(base, cfg, folded)] if folded else [])
            + ([lambda: test_rows(folded, rows)] if folded and rows else [])
            + ([lambda: test_hosted(rows, hosted)] if rows and hosted else [])
            + ([lambda: test_records(rows, records)] if rows and records else [])
            + (
                [lambda: test_records(hosted, records_hosted)]
                if hosted and records_hosted
                else []
            )
            + (
                [lambda: test_combine(records_hosted, combine)]
                if records_hosted and combine
                else []
            )
            + ([lambda: test_combine(d, combine_alone)] if combine_alone else [])
            + ([lambda: test_telem(combine, telem)] if combine and telem else [])
            + ([lambda: test_telem(d, telem_alone)] if telem_alone else [])
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
