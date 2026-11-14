#!/usr/bin/env python3
"""What a generated GNSS node config promises the code and the tools that read it, as a
snapshot two configs can be diffed on.

    node_contract.py snapshot CONFIG [--node N] [--host H] [-o OUT.json]
    node_contract.py diff A.json B.json [--allow FILE]
    node_contract.py live --host H CONFIG

CONFIG is a generated chord_gnss_<node>_multi.yaml; the census reads the configs beside it, as
the site tools read config/generated. diff prints one line per difference and exits 1 if any
is left after FILE's fnmatch patterns (one per line, '#' comments), which are matched against
the difference's key, `facet: key ...`. live does one read-only GET http://H:12048/endpoints
and compares its per-chain paths with CONFIG's rest facet.

The facets:

  rest       the per-chain REST paths kotekan registers: the inject's four
             (cudaGnssChordTrack.cpp), the assembler's (GnssGpuRecordAssemble's constructor)
             and the combiner's (GnssCoherentCombiner::main_thread), conditions restated
  identity   the names other stages and processes key on: each assembler's ProjBoard group,
             lane owner and cube sender, GnssN2Project board_group, cudaGnssBeamDump
             beam_lane_owner, the telemetry header fields and frame size, bufferSend addresses
  placement  numa_node of every buffer and cpu_affinity of every stage; a gnss<g>_ buffer off
             NUMA node g is listed as misplaced
  stall      per chain, the GPU frames of slack before cudaOutputData blocks the half:
             min(tiles, ctl) + epl + usable rec frames / records per frame. A bufferMerge with
             no timeout on a chain's path from cudaOutputData's tiles and ctl through its
             records is listed under its own name with its (negative) timeout
  census     what elem_shared_ref, lane_relay, gen_fleet, viscap_read and gnss_fleet.stage_chain
             find in the config, each by its own code

snapshot refuses a config that sets a test-only key (TEST_ONLY) anywhere: such a key must never
reach a node.
"""
import argparse
import fnmatch
import json
import os
import re
import sys
import urllib.request

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
K = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(K, "config"))
sys.path.insert(0, os.path.join(K, "python", "scripts", "gnss"))
sys.path.insert(0, os.path.join(HERE, "site"))
import elem_shared_ref  # noqa: E402
import gen_fleet  # noqa: E402
import gnss_fleet  # noqa: E402
import lane_relay  # noqa: E402
import viscap_read  # noqa: E402

PORT = 12048
FACETS = ("rest", "identity", "placement", "stall", "census")
INJECT = (
    ("POST", "seed_endpoint"),
    ("POST", "set_trim_endpoint"),
    ("GET", "get_prns_endpoint"),
    ("POST", "set_prns_endpoint"),
)
# psi_sync_build: cudaGnssChordTrack waits for every Psi build on the GPU thread.
TEST_ONLY = ("psi_sync_build",)
PER_CHAIN = re.compile(r"_(inject|n2assemble|n2combine)/[^/]+$")
IN_KEYS = ("in_buf", "in_bufs", "tiles_buf", "ctl_buf", "buf")
OUT_KEYS = ("out_buf", "out_bufs", "cube_buf")


def load(path):
    with open(path) as f:
        return yaml.load(gen_fleet._split_header(f.read())[1], Loader=yaml.CSafeLoader)


def _blocks(cfg):
    """(unique_name, scope) of every stage and buffer block, scope = [the block, its parents,
    ..., the root]: kotekan's factories walk the tree the same way, and a key is looked up
    from the block outwards."""
    out = []

    def walk(node, path, scope):
        for k, v in node.items():
            if not isinstance(v, dict):
                continue
            s = [v] + scope
            if "kotekan_stage" in v or "kotekan_buffer" in v:
                out.append((path + "/" + k, s))
            else:
                walk(v, path + "/" + k, s)

    walk(cfg, "", [cfg])
    return out


def _get(scope, key, default=None):
    for b in scope:
        if key in b:
            return b[key]
    return default


def _stages(cfg, kind=None):
    return [
        (u, s)
        for u, s in _blocks(cfg)
        if "kotekan_stage" in s[0] and kind in (None, s[0]["kotekan_stage"])
    ]


def _buffers(cfg):
    """{buffer name: scope}; a buffer is named by its key alone."""
    return {u.rsplit("/", 1)[1]: s for u, s in _blocks(cfg) if "kotekan_buffer" in s[0]}


def _names(v):
    """The buffer names in a stage's buffer key: a name, a list of names, or bufferMerge's
    list of {internal name: buffer}."""
    if isinstance(v, str):
        return [v]
    if not isinstance(v, list):
        return []
    out = []
    for x in v:
        out += list(x.values()) if isinstance(x, dict) else [x]
    return out


def _slash(p):
    return p if p.startswith("/") else "/" + p


def _inst_on(scope):
    """GnssGpuRecordAssemble's _inst_on: lane directions published, lane transfers, two
    distinct first channels, and lane_transfer_inst_slope not turned off."""
    fids = _get(scope, "channel_ids", [])
    if _get(scope, "elem_dir_publish", "off") != "all" or not fids:
        return False
    df = fids[1] - fids[0] if len(fids) >= 2 else 0
    return bool(
        _get(scope, "lane_transfer_freq_ids", [])
        and df != 0
        and _get(scope, "lane_transfer_inst_slope", True)
    )


def rest_surface(cfg):
    """{'GET /path', 'POST /path'} of every per-chain endpoint kotekan registers."""
    out = set()
    for u, s in _stages(cfg, "cudaProcess"):
        for c in s[0].get("commands") or []:
            if c.get("name") == "cudaGnssInject":
                out.update("%s %s" % (verb, _slash(c[key])) for verb, key in INJECT)
    for u, s in _stages(cfg, "GnssGpuRecordAssemble"):
        eps = [
            ("GET", "get_spectrum"),
            ("POST", "set_sat_geometry"),
            ("POST", "set_reference_element"),
        ]
        if _get(s, "elem_sum", False):
            eps += [
                ("POST", "set_elem_gain"),
                ("POST", "set_elem_sum_adapt"),
                ("POST", "set_elem_sum_shared"),
                ("POST", "set_elem_sum_shared_ref"),
                ("GET", "get_elem_cal"),
            ]
            if _get(s, "channel_ids", []):
                eps.append(("POST", "set_elem_proj"))
        if _inst_on(s):
            eps.append(("GET", "get_lane_inst"))
        out.update("%s %s/%s" % (verb, u, e) for verb, e in eps)
    for u, s in _stages(cfg, "GnssCoherentCombiner"):
        eps = ["get_status", "get_elements"]
        if _get(s, "record_export", 0) > 0:
            eps.append("get_records")
        out.update("GET %s/%s" % (u, e) for e in eps)
    return out


def endpoint_set(reply):
    """The per-chain paths of a GET /endpoints reply, in rest_surface's form."""
    return {
        "%s %s" % (verb, p)
        for verb in ("GET", "POST")
        for p in reply.get(verb, [])
        if PER_CHAIN.search(p)
    }


def _cube(u, scope, host):
    """The assembler's cube sender string, or None with the cube off."""
    if not (_get(scope, "beam_cube", False) and _get(scope, "channel_ids", [])):
        return None
    return _get(scope, "beam_cube_chain", host + "/" + u)


def identity(cfg, host):
    bufs = _buffers(cfg)
    asm = {}
    for u, s in _stages(cfg, "GnssGpuRecordAssemble"):
        asm[u] = {
            "group": u.split("_", 1)[0],
            "owner": u + "/lane/",
            "cube": _cube(u, s, host),
        }
    telem = {}
    for u, s in _stages(cfg, "GnssTelemPack"):
        t = {k: _get(s, k) for k in ("chain", "inst", "max_prn", "n_chan")}
        t["frame_size"] = _get(bufs[_get(s, "out_buf")], "frame_size")
        telem[u] = t
    beam = {}
    for u, s in _stages(cfg, "cudaProcess"):
        for c in s[0].get("commands") or []:
            if c.get("name") == "cudaGnssBeamDump" and c.get("beam_lane_owner"):
                beam["%s/%s" % (u, c.get("beam_name"))] = c["beam_lane_owner"]
    return {
        "assembler": asm,
        "board_group": {
            u: _get(s, "board_group", "") for u, s in _stages(cfg, "GnssN2Project")
        },
        "beam_lane_owner": beam,
        "telem": telem,
        "send": {
            u: "%s:%s" % (_get(s, "server_ip"), _get(s, "server_port"))
            for u, s in _stages(cfg, "bufferSend")
        },
    }


def placement(cfg):
    numa, misplaced = {}, []
    for name, s in sorted(_buffers(cfg).items()):
        numa[name] = _get(s, "numa_node", 0)
        m = re.match(r"gnss(\d)_", name)
        if m and numa[name] != int(m.group(1)):
            misplaced.append(name)
    return {
        "numa": numa,
        "cpu": {u: _get(s, "cpu_affinity", []) for u, s in _stages(cfg)},
        "misplaced": misplaced,
    }


def _ins(scope):
    return {n for k in IN_KEYS for n in _names(scope[0].get(k))}


def _outs(scope):
    return {n for k in OUT_KEYS for n in _names(scope[0].get(k))}


def _downstream(stages, buf, kind):
    """The scope of the first stage of this kind downstream of buf, or None."""
    seen, todo = {buf}, [buf]
    while todo:
        b = todo.pop(0)
        for u, s in stages:
            if b not in _ins(s):
                continue
            if s[0]["kotekan_stage"] == kind:
                return s
            todo += sorted(_outs(s) - seen)
            seen |= _outs(s)
    return None


def _upstream(stages, bufs):
    """bufs and every buffer the host stages that fill them read, back to the GPU."""
    out, todo = set(bufs), list(bufs)
    while todo:
        b = todo.pop()
        for u, s in stages:
            if b in _outs(s):
                todo += sorted(_ins(s) - out)
                out |= _ins(s)
    return out


def stall_budget(cfg):
    """{adapter unique_name: GPU frames of slack, or 'unresolved' with no assembler downstream
    of its epl}, plus {bufferMerge unique_name: timeout} for every merge with no timeout on a
    chain's path from cudaOutputData's tiles and ctl through its records. cudaOutputData fills
    tiles and ctl together and the adapter drains them together, so the shallower counts."""
    bufs = _buffers(cfg)
    stages = _stages(cfg)
    out, reach = {}, set()
    for u, s in _stages(cfg, "GnssN2RecordAssemble"):
        tiles, ctl, epl = _get(s, "tiles_buf"), _get(s, "ctl_buf"), _get(s, "out_buf")
        reach |= _upstream(stages, (tiles, ctl))
        a = _downstream(stages, epl, "GnssGpuRecordAssemble")
        if a is None:
            out[u] = "unresolved"
            continue
        rec = _get(a, "out_buf")
        rb = bufs[rec]
        per_frame = _get(s, "samples_per_data_set") // _get(s, "hops_per_record")
        hold = 1 if _get(rb, "peek_hold", False) else 0
        out[u] = (
            min(_get(bufs[tiles], "num_frames"), _get(bufs[ctl], "num_frames"))
            + _get(bufs[epl], "num_frames")
            + (_get(rb, "num_frames") - hold) / per_frame
        )
        reach.add(rec)
    grown = True
    while grown:
        grown = False
        for u, s in stages:
            if _ins(s) & reach:
                new = _outs(s) - reach
                reach |= new
                grown = grown or bool(new)
    for u, s in _stages(cfg, "bufferMerge"):
        timeout = _get(s, "timeout", -1.0)
        if timeout < 0 and set(_names(s[0].get("in_bufs"))) & reach:
            out[u] = float(timeout)
    return out


def census(cfg_path, node, host=None):
    """{tool: [what it finds]} for this node's config. elem_shared_ref reads its configs by
    name from one directory, so CONFIG must carry the generated name. config_zeros is asked
    about every assembler in the config, not only those endpoints() finds."""
    if os.path.basename(cfg_path) != "chord_gnss_%s_multi.yaml" % node:
        raise ValueError(
            "%s: the census needs chord_gnss_%s_multi.yaml" % (cfg_path, node)
        )
    cfg = load(cfg_path)
    with open(cfg_path) as f:
        text = f.read()
    saved = elem_shared_ref.GEN
    elem_shared_ref.GEN = os.path.join(
        os.path.dirname(os.path.abspath(cfg_path)), "chord_gnss_%s_multi.yaml"
    )
    try:
        eps = [st for n, st in elem_shared_ref.endpoints() if n == node]
        zeros = [
            "%s: %s"
            % (st, ",".join(map(str, sorted(elem_shared_ref.config_zeros(node, st)))))
            for st in sorted(
                u.rsplit("/", 1)[1] for u, s in _stages(cfg, "GnssGpuRecordAssemble")
            )
        ]
    finally:
        elem_shared_ref.GEN = saved
    tags = []
    for g in (0, 1):
        try:
            tags.append("%d: %r" % (g, viscap_read.primary_tag(cfg_path, g)))
        except SystemExit as e:
            tags.append("%d: exit %s" % (g, e))
    chains = []
    for u, s in _stages(cfg, "GnssGpuRecordAssemble"):
        m = lane_relay.OWNER.match(u + "/lane/0")
        own = gnss_fleet.stage_chain(m.group(1), "n2assemble") if m else None
        cube = _cube(u, s, host or node)
        chains.append("owner %s/lane/ -> %s" % (u, own))
        if cube:
            chains.append(
                "cube %s -> %s" % (cube, gnss_fleet.stage_chain(cube, "n2assemble"))
            )
    for owner in identity(cfg, node)["beam_lane_owner"].values():
        chains.append(
            "beam_lane_owner %s -> %s"
            % (owner, gnss_fleet.stage_chain(owner, "n2assemble"))
        )
    built = gen_fleet.built_instances(text)
    return {
        "elem_shared_ref.endpoints": eps,
        "elem_shared_ref.config_zeros": zeros,
        "lane_relay.inst_stage": [
            "%d: %s" % (g, st)
            for g in (0, 1)
            for st in [lane_relay.inst_stage(cfg, g)]
            if st
        ],
        "gen_fleet.built_instances": [
            "%s: %s" % (c, ",".join(map(str, sorted(g))))
            for c, g in sorted(built.items())
        ],
        "viscap_read.primary_tag": tags,
        "gnss_fleet.stage_chain": chains,
    }


def test_only_keys(node, path=""):
    """'path: key' for every TEST_ONLY key set in the tree, in blocks and in lists of blocks
    (a cudaProcess's commands)."""
    out = []
    if isinstance(node, dict):
        out += ["%s: %s" % (path or "/", k) for k in TEST_ONLY if k in node]
        for k, v in node.items():
            out += test_only_keys(v, "%s/%s" % (path, k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += test_only_keys(v, "%s/%d" % (path, i))
    return out


def snapshot(cfg_path, node, host):
    cfg = load(cfg_path)
    bad = test_only_keys(cfg)
    if bad:
        raise ValueError("%s sets test-only keys: %s" % (cfg_path, ", ".join(bad)))
    return {
        "node": node,
        "host": host,
        "rest": sorted(rest_surface(cfg)),
        "identity": identity(cfg, host),
        "placement": placement(cfg),
        "stall": stall_budget(cfg),
        "census": census(cfg_path, node, host),
    }


def _flat(x, key):
    """(key tuple, value) leaves; a list of strings is a set, one True leaf per member."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _flat(v, key + (str(k),))
    elif isinstance(x, list) and x and all(isinstance(i, str) for i in x):
        for i in x:
            yield key + (i,), True
    else:
        yield key, x


def diff(a, b, allow=()):
    """One line per facet difference between snapshots a and b: '- key' (only in a), '+ key'
    (only in b) or '~ key: a -> b', key = 'facet: key ...'. Keys matching an allow pattern
    are left out."""
    fa = dict(_flat({f: a[f] for f in FACETS}, ()))
    fb = dict(_flat({f: b[f] for f in FACETS}, ()))
    out = []
    for k in sorted(set(fa) | set(fb)):
        va, vb = fa.get(k), fb.get(k)
        if k in fa and k in fb and va == vb:
            continue
        key = "%s: %s" % (k[0], " ".join(k[1:]))
        if any(fnmatch.fnmatchcase(key, p) for p in allow):
            continue
        if k not in fb:
            out.append("- %s" % key + ("" if va is True else " = %r" % (va,)))
        elif k not in fa:
            out.append("+ %s" % key + ("" if vb is True else " = %r" % (vb,)))
        else:
            out.append("~ %s: %r -> %r" % (key, va, vb))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="write one config's snapshot as JSON")
    s.add_argument("config")
    s.add_argument("--node", help="default: from the config's name")
    s.add_argument("--host", help="the short hostname the cube sender carries (--node)")
    s.add_argument("-o", "--out", help="default: stdout")
    d = sub.add_parser("diff", help="diff two snapshots; exit 1 on a difference")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--allow", help="fnmatch patterns of expected differences")
    lv = sub.add_parser("live", help="compare a node's /endpoints with its config")
    lv.add_argument("--host", required=True)
    lv.add_argument("config")
    a = ap.parse_args()

    if a.cmd == "snapshot":
        m = re.match(r"chord_gnss_(\w+?)_multi\.yaml$", os.path.basename(a.config))
        node = a.node or (m and m.group(1))
        if not node:
            ap.error("--node is needed for %s" % a.config)
        snap = snapshot(a.config, node, a.host or node)
        text = json.dumps(snap, indent=1, sort_keys=True) + "\n"
        if a.out:
            with open(a.out, "w") as f:
                f.write(text)
        else:
            sys.stdout.write(text)
        return 0
    if a.cmd == "diff":
        allow = []
        if a.allow:
            with open(a.allow) as f:
                allow = [
                    ln.strip() for ln in f if ln.strip() and not ln.startswith("#")
                ]
        with open(a.a) as fa, open(a.b) as fb:
            lines = diff(json.load(fa), json.load(fb), allow)
        print("\n".join(lines) if lines else "identical")
        return 1 if lines else 0
    url = "http://%s:%d/endpoints" % (a.host, PORT)
    with urllib.request.urlopen(url, timeout=10) as r:
        have = endpoint_set(json.loads(r.read()))
    want = rest_surface(load(a.config))
    for p in sorted(want - have):
        print("- %s  (in the config, not registered)" % p)
    for p in sorted(have - want):
        print("+ %s  (registered, not in the config)" % p)
    print("%d per-chain paths registered, %d expected" % (len(have), len(want)))
    return 1 if want != have else 0


if __name__ == "__main__":
    sys.exit(main())
