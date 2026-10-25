#!/usr/bin/env python3
"""gnss_fleet.py and gen_fleet.py's fleet checks: slots and ports from the list's order, role
off, the stock hosts, each chain's instances by the covering rule, the chains yaml's `fleet`
expansion, and the --check gate against hand-written endpoint lists.

    python3 config/test_gnss_fleet.py

Hermetic: the node table and the signal descriptors are the repo's (no network).
"""
import os
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "gnss"))
import gnss_fleet as gf  # noqa: E402
import gen_fleet  # noqa: E402

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def site():
    with open(gf.NODE_FILE) as fh:
        return yaml.safe_load(fh)


def manifest(nodes, extra=None):
    """A manifest with the repo's common block and these nodes."""
    with open(gf.MANIFEST) as fh:
        man = yaml.safe_load(fh)
    man["nodes"] = nodes
    man.update(extra or {})
    return man


def refused(man, needle):
    try:
        gf.Fleet(man, site())
    except gf.FleetError as e:
        return needle in str(e)
    return False


def test_slots_ports_and_roles():
    fl = gf.Fleet(manifest({"cx19": {}, "cx42": {"role": "off"}, "cx43": {}}), site())
    check(fl.hosts == ["cx19", "cx43"], "role off leaves the fleet (%s)" % fl.hosts)
    check(
        [fl.search_port_base(h) for h in ("cx19", "cx42", "cx43")]
        == [11040, 11042, 11044],
        "ports follow the slot, and an off node keeps its slot",
    )
    check(
        fl.stock == ["cx27", "cx42", "cx44", "cx47", "cx51", "cx52"],
        "stock = the site's hosts not in the fleet, role off included (%s)" % fl.stock,
    )
    check(
        fl.gen_flags("cx42") == {} and "role" not in fl.gen_flags("cx19"),
        "role is not a generator flag",
    )
    live = gf.Fleet(None, None)
    check(
        live.hosts == ["cx19", "cx27", "cx42", "cx43", "cx44", "cx51"]
        and [live.search_port_base(h) for h in live.hosts]
        == [11040, 11042, 11044, 11046, 11048, 11050],
        "the repo's fleet keeps the ports it ran with before the list (%s)"
        % [live.search_port_base(h) for h in live.hosts],
    )


def test_refusals():
    check(
        refused(manifest({"cx19": {"search-port-base": 11040}}), "derived"),
        "a hand-written search-port-base is refused",
    )
    m = manifest({"cx19": {}})
    m["common"] = dict(m["common"], **{"search-port-base": 11040})
    check(refused(m, "common sets search-port-base"), "...in common too")
    check(
        refused(manifest({"cx99": {}}), "not in the site's node table"),
        "a host without combs in the node table is refused",
    )
    check(
        refused(manifest({"cx19": {"role": "spare"}}), "role"),
        "an unknown role is refused",
    )
    check(
        refused(manifest({"cx19": {"role": "off"}}), "every node is role off"),
        "a fleet with no GNSS node is refused",
    )


def test_instances():
    fl = gf.Fleet(None, None)
    l2c = fl.instances("gps_l2c")
    check(
        l2c == [("cx19", 1), ("cx27", 0), ("cx42", 1), ("cx43", 1), ("cx51", 1)],
        "L2C lives on the GPUs that hold its channels, none on cx44 (%s)" % l2c,
    )
    l5 = fl.instances("gps_l5")
    check(
        len(l5) == 12 and l5[:2] == [("cx19", 0), ("cx19", 1)],
        "L5 on every GPU, slot-major",
    )
    fl2 = gf.Fleet(manifest({"cx44": {}, "cx47": {}, "cx52": {}}), site())
    check(
        fl2.instances("gps_l2c") == [("cx47", 1)],
        "a joining host's chains follow its combs (cx47 holds 6283 on GPU 1: %s)"
        % fl2.instances("gps_l2c"),
    )
    try:
        fl.instances("gps_l1ca")
        check(False, "a chain no node builds is refused")
    except gf.FleetError as e:
        check("no node builds it" in str(e), "a chain no node builds is refused")


def chains_cfg():
    return {
        "common": {"rest-url": "fleet", "interval": 2},
        "chains": {
            "gps_l5": {
                "signal": "gps_l5",
                "trackers": "fleet",
                "combiner": "fleet",
                "rf-stats-endpoints": "fleet",
            },
            "gps_l2c": {
                "signal": "gps_l2c",
                "trackers": "fleet",
                "combiner": "fleet",
                "spectrum-endpoints": None,
                "dll-combiners": "http://x:1/y",
            },
        },
    }


def test_expand():
    cfg = chains_cfg()
    out = gf.expand_chains(cfg, gf.Fleet(None, None))
    c = out["chains"]
    check(cfg["common"]["rest-url"] == "fleet", "the input is not modified")
    check(out["common"]["rest-url"] == "http://cx19:12048", "rest-url: the first host")
    check(
        c["gps_l5"]["trackers"].split(",")[:2]
        == ["http://cx19:12048/gnss0_inject", "http://cx19:12048/gnss1_inject"]
        and len(c["gps_l5"]["trackers"].split(",")) == 12,
        "the primary's trackers carry no tag",
    )
    check(
        c["gps_l5"]["rf-stats-endpoints"].split(",")[0]
        == "http://cx19:12048/gnss0_srch_tap",
        "rf-stats-endpoints are the search taps",
    )
    check(
        c["gps_l2c"]["combiner"] == "gnss1_l2c_n2combine",
        "combiner: bare, the chain's GPU on rest-url's host (%s)"
        % c["gps_l2c"]["combiner"],
    )
    check(
        c["gps_l2c"]["spectrum-endpoints"] is None
        and c["gps_l2c"]["dll-combiners"] == "http://x:1/y",
        "null and hand values are left alone",
    )
    fl = gf.Fleet(manifest({"cx44": {}, "cx51": {}}), site())
    c = gf.expand_chains(chains_cfg(), fl)["chains"]
    check(
        c["gps_l2c"]["combiner"] == "http://cx51:12048/gnss1_l2c_n2combine",
        "combiner: a URL when rest-url's host lacks the chain (%s)"
        % c["gps_l2c"]["combiner"],
    )
    plain = {"chains": {"a": {"trackers": "http://h:1/t"}}}
    check(
        gf.expand_chains(plain, lambda: check(False, "fleet built for nothing"))
        == plain,
        "a yaml without `fleet` needs no fleet list",
    )


def test_load_chains_uses_the_fleet_beside_it():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "gnss_fleet_chord.yaml"), "w") as fh:
            yaml.safe_dump(manifest({"cx44": {}}), fh)
        with open(os.path.join(d, "chord_gnss_node.yaml"), "w") as fh:
            yaml.safe_dump(site(), fh)
        path = os.path.join(d, "chains.yaml")
        cfg = chains_cfg()
        with open(path, "w") as fh:
            yaml.safe_dump(cfg, fh)
        try:
            gf.load_chains(path)
            check(False, "a chain no fleet GPU holds is refused")
        except gf.FleetError as e:
            check(
                "gps_l2c: no GPU of the fleet (cx44)" in str(e),
                "a chain no fleet GPU holds is refused, naming it",
            )
        del cfg["chains"]["gps_l2c"]
        with open(path, "w") as fh:
            yaml.safe_dump(cfg, fh)
        out = gf.load_chains(path)
    check(
        out["chains"]["gps_l5"]["trackers"].startswith("http://cx44:12048/"),
        "load_chains reads the fleet list beside the chains file",
    )


def test_repo_chains_follow_the_fleet():
    fl = gf.Fleet(None, None)
    probs = gen_fleet.check_chains(fl)
    check(probs == [], "the repo's chains yaml follows the fleet (%s)" % probs)
    with tempfile.TemporaryDirectory() as d:
        cfg = chains_cfg()
        cfg["chains"]["gps_l5"]["trackers"] = "http://cx19:12048/gnss{0..1}_inject"
        cfg["chains"]["gps_l5"]["dll-combiners"] = "fleet"
        cfg["chains"]["gps_l2c"][
            "dll-combiners"
        ] = "http://cx47:12048/gnss0_l2c_n2combine"
        path = os.path.join(d, "chains.yaml")
        with open(path, "w") as fh:
            yaml.safe_dump(cfg, fh)
        probs = gen_fleet.check_chains(fl, path)
    check(
        any("trackers is written by hand" in p for p in probs),
        "--check refuses a hand-written endpoint list",
    )
    check(
        any("cx47, which is not in the fleet list" in p for p in probs),
        "--check names a host outside the fleet",
    )


def test_built_instances_matches_derived():
    text = "# header\n" + yaml.safe_dump(
        {
            "gnss0_n2combine": {},
            "gnss1_n2combine": {},
            "gnss1_l2c_n2combine": {},
            "gnss0_e5a_n2combine": {},
            "gnss0_e5a_n2assemble": {},
        }
    )
    got = gen_fleet.built_instances(text)
    check(
        got == {"": {0, 1}, "_l2c": {1}, "_e5a": {0}},
        "one n2combine per chain instance is read from a config (%s)" % got,
    )
    fl = gf.Fleet(None, None)
    d = gen_fleet.derived_instances(fl, "cx44")
    check(
        "_l2c" not in d and d[""] == {0, 1},
        "cx44 is derived without L2C, with L5 on both GPUs",
    )


def test_check_instances_names_a_mismatch():
    fl = gf.Fleet(manifest({"cx44": {}}, {"common": {}}), site())
    text = "# h\n" + yaml.safe_dump({"gnss0_n2combine": {}, "gnss1_n2combine": {}})
    check(
        gen_fleet.check_instances(fl, {"cx44": text}) == [], "a matching node is quiet"
    )
    text = "# h\n" + yaml.safe_dump({"gnss0_n2combine": {}})
    probs = gen_fleet.check_instances(fl, {"cx44": text})
    check(
        len(probs) == 1 and probs[0].startswith("cx44: the generator builds"),
        "a GPU the generator did not build is named (%s)" % probs,
    )


def test_live_comb_reads_the_frames():
    """live_comb takes each GPU's comb from its n2_buffer frame's freq_id; a node that does
    not answer, or a frame outside the science band, gives no verdict."""
    import io
    import json
    import urllib.request

    sb = site()["science_band"]
    frames = {"n2_buffer_0": 7668, "n2_buffer_1": 7678}
    asked = []

    def fake(url, timeout=None):
        asked.append(url)
        name = url.rsplit("name=", 1)[1]
        if name not in frames:
            raise OSError("refused")
        return io.BytesIO(json.dumps({"metadata": {"freq_id": frames[name]}}).encode())

    real = urllib.request.urlopen
    urllib.request.urlopen = fake
    try:
        check(
            gf.live_comb("cx19", 12048, sb) == [4, 14],
            "both GPUs' combs from the frames",
        )
        check(
            asked[0] == "http://cx19:12048/buffer_frame?name=n2_buffer_0",
            "asked on the node's REST port (%s)" % asked[0],
        )
        frames["n2_buffer_1"] = 0
        check(
            gf.live_comb("cx19", 12048, sb) is None, "a freq_id outside the band: none"
        )
        del frames["n2_buffer_1"]
        check(
            gf.live_comb("cx19", 12048, sb) is None, "a GPU that does not answer: none"
        )
    finally:
        urllib.request.urlopen = real


def test_write_combs_edits_one_line():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "n.yaml")
        with open(gf.NODE_FILE) as fh:
            orig = fh.read()
        with open(p, "w") as fh:
            fh.write(orig)
        gf.write_combs(p, {"cx19": [4, 12], "cx99": [0, 1]})
        with open(p) as fh:
            new = fh.read()
        got = yaml.safe_load(new)["nodes"]
        changed = [
            (a, b) for a, b in zip(orig.splitlines(), new.splitlines()) if a != b
        ]
    check(
        got["cx19"]["gpu_offsets_mod16"] == [4, 12]
        and got["cx99"]["gpu_offsets_mod16"] == [0, 1]
        and got["cx27"]["gpu_offsets_mod16"] == [0, 8]
        and "bringup_order" in got,
        "the host's comb is set, a new host added, the rest kept",
    )
    check(
        len(new.splitlines()) == len(orig.splitlines()) + 1
        and len(changed) >= 1
        and all("cx19" in a for a, _ in changed[:1]),
        "only cx19's line changes and one line is added (comments kept)",
    )


def test_check_combs():
    fl = gf.Fleet(None, None)
    real = gf.comb_report
    gf.comb_report = lambda fleet, hosts, timeout=4.0: [
        ("cx19", [4, 14], [4, 12]),
        ("cx27", [0, 8], [0, 8]),
        ("cx42", [5, 12], None),
    ]
    try:
        probs = gen_fleet.check_combs(fl, ["cx19", "cx27", "cx42"])
    finally:
        gf.comb_report = real
    check(
        len(probs) == 1
        and probs[0].startswith("cx19: the node table says combs [4, 14]")
        and "[4, 12]" in probs[0],
        "a moved comb fails, a node that does not answer only notes (%s)" % probs,
    )


def test_aggregator_cmd():
    fl = gf.Fleet(manifest({"cx19": {}, "cx42": {"role": "off"}, "cx43": {}}), site())
    man = {
        "aggregator": {
            "out": "agg.yaml",
            "base": "base/live_config_20260730.json",
            "flags": {"rest-port": 12050, "aggregator-extra-search": ["A", "B:1"]},
        }
    }
    cmd, out = gen_fleet.aggregator_cmd(man, fl, "/tmp/x")
    i = cmd.index("--aggregator-instance")
    check(
        cmd[i + 1 : i + 3] == ["cx19:11040", "cx43:11044"]
        and cmd[cmd.index("--node") + 1] == "cx19",
        "the aggregator's feeds are the fleet's hosts at their slots' ports (%s)"
        % cmd[i + 1 : i + 3],
    )
    check(
        cmd.count("--aggregator-extra-search") == 2 and out == "/tmp/x/agg.yaml",
        "its flags come from the manifest",
    )


def main():
    os.chdir(ROOT)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        try:
            t()
        except Exception as e:
            check(False, "%s raised %s: %s" % (t.__name__, type(e).__name__, e))
    if _fails:
        print("\n%d FAILED" % len(_fails))
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
