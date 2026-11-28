#!/usr/bin/env python3
"""lane_relay.py's owner parse against the names a node serves.

    python3 scripts/gnss/site/test_lane_relay.py

A lane's owner is the assembler's kotekan unique_name, which starts with '/', plus
"/lane/<prn>" (GnssGpuRecordAssemble).
"""
import concurrent.futures
import os
import sys
import tempfile
import types

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lane_relay  # noqa: E402

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def test_parse_owner():
    check(
        lane_relay.parse_owner("/gnss1_gps_l2c_n2assemble/lane/11") == ("gps_l2c", 11),
        "an owner names its chain",
    )
    check(
        lane_relay.parse_owner("/gnss0_gps_l5_n2assemble/lane/26") == ("gps_l5", 26)
        and lane_relay.parse_owner("/gnss0_sbas_l5_n2assemble/lane/131")
        == ("sbas_l5", 131),
        "GPS and SBAS L5 are two chains",
    )
    check(
        lane_relay.parse_owner("gnss1_gal_e6_n2assemble/lane/3") == ("gal_e6", 3),
        "no leading slash",
    )
    check(
        lane_relay.parse_owner("/gnss1_l2c_n2assemble/lane/11") == ("gps_l2c", 11)
        and lane_relay.parse_owner("/gnss0_n2assemble/lane/26") == ("gps_l5", 26)
        and lane_relay.parse_owner("/gnss0_l5_n2assemble/lane/131") == ("sbas_l5", 131),
        "a node on a config with the band tags: untagged = GPS L5, l5 = SBAS L5",
    )
    check(
        lane_relay.parse_owner("/gnss1_gps_l2c_n2combine/lane/11") == (None, None),
        "not an assembler",
    )


def test_transfer_table_refuses_an_unknown_chain():
    chains = {"gps_l5": None, "gps_l2c": None, "gal_e6": None}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fleet.yaml")
        with open(path, "w") as fh:
            yaml.safe_dump(
                {"common": {"fleet-lane-transfer": ["gps_l2c:1217.4-1237.8"]}}, fh
            )
        t = lane_relay.transfer_table(path, chains)
        check(
            list(t) == ["gps_l2c"] and t["gps_l2c"][0][0] == 6234,
            "a chain's targets are keyed by its broker key (%s)" % t,
        )
        with open(path, "w") as fh:
            yaml.safe_dump(
                {"common": {"fleet-lane-transfer": ["l2c:1217.4-1237.8"]}}, fh
            )
        try:
            lane_relay.transfer_table(path, chains)
            check(False, "a band tag is refused")
        except SystemExit as e:
            check(
                "not a chain the fleet builds" in str(e),
                "a band tag is refused: %s" % e,
            )


def test_repo_tables_name_chains():
    k = lane_relay.K
    chains = lane_relay.gnss_fleet.Fleet(
        os.path.join(k, lane_relay.FLEET), os.path.join(k, lane_relay.NODES)
    ).tracker_signals()
    for key in ("fleet-lane-transfer", "fleet-lane-transfer-bright"):
        t = lane_relay.transfer_table(os.path.join(k, lane_relay.FLEET), chains, key)
        check(
            set(t) <= set(chains),
            "the manifest's %s names chains (%s)" % (key, sorted(t)),
        )


def test_inst_stage():
    asm = {"kotekan_stage": "GnssGpuRecordAssemble"}
    cfg = {
        "gnss0_gps_l5_n2assemble": {"channel_ids": [1, 2, 3]},
        "gnss0_sbas_l5_n2assemble": {
            "channel_ids": [6103, 6119],
            "lane_transfer_freq_ids": [6110],
        },
        "gnss0_gps_l2c_n2assemble": {
            "channel_ids": [6285],
            "lane_transfer_freq_ids": [6250],
        },
        "gnss1_gal_e6_n2assemble": {
            "channel_ids": [6541, 6557],
            "lane_transfer_freq_ids": [6480],
        },
        "gnss1_gal_e5b_n2assemble": {
            "channel_ids": [6151, 6167, 6183],
            "lane_transfer_freq_ids": [6200],
        },
    }
    for v in cfg.values():
        v.update(asm)
    check(
        lane_relay.inst_stage(cfg, 0) == "gnss0_sbas_l5_n2assemble",
        "SBAS L5 when the GPS one transfers nothing",
    )
    check(
        lane_relay.inst_stage(cfg, 1) == "gnss1_gal_e5b_n2assemble",
        "else the most lane channels",
    )
    cfg["gnss0_gps_l5_n2assemble"]["lane_transfer_freq_ids"] = [6000]
    check(
        lane_relay.inst_stage(cfg, 0) == "gnss0_gps_l5_n2assemble",
        "the GPS L5 one first",
    )
    hosted = {
        "gnss%d_records"
        % g: dict(
            asm,
            chains={
                k[: -len("_n2assemble")]: {x: y for x, y in v.items() if x not in asm}
                for k, v in cfg.items()
                if k.startswith("gnss%d" % g)
            },
        )
        for g in (0, 1)
    }
    check(
        [lane_relay.inst_stage(hosted, g) for g in (0, 1)]
        == [lane_relay.inst_stage(cfg, g) for g in (0, 1)],
        "the same from the rows of record hosts",
    )
    check(
        lane_relay.inst_stage(
            {"gnss0_gps_l2c_n2assemble": cfg["gnss0_gps_l2c_n2assemble"]}, 0
        )
        is None,
        "one lane channel serves no estimate",
    )
    stage = {"source": "lanes"}
    own = lane_relay.Gpu("node-a", 0, stage, 3, "gnss0_gps_l5_n2assemble")
    by_chain = lane_relay.Gpu(
        "node-a",
        0,
        stage,
        3,
        "gnss0_gps_l5_n2assemble",
        lane_relay.gnss_fleet.config_host_bases(hosted, "http://node-a:12048"),
    )
    check(
        own.inst_url == "http://node-a:12048/gnss0_gps_l5_n2assemble/get_lane_inst"
        and by_chain.inst_url
        == "http://node-a:12048/gnss0_records/get_lane_inst?chain=gps_l5",
        "get_lane_inst at the row's own path, or by chain from its record host (%s)"
        % by_chain.inst_url,
    )
    unhosted = lane_relay.Gpu(
        "node-a",
        0,
        stage,
        3,
        "gnss0_gps_l5_n2assemble",
        lane_relay.gnss_fleet.config_host_bases(cfg, "http://node-a:12048"),
    )
    check(
        unhosted.inst_url
        == "http://node-a:12048/gnss0_gps_l5_n2assemble/get_lane_inst",
        "a config without record hosts: the row's own path (%s)" % unhosted.inst_url,
    )


def test_inst_errors_are_logged():
    """A failed get_lane_inst is logged, rate-limited, naming the url, as set_lanes is."""
    relay = lane_relay.Relay.__new__(lane_relay.Relay)
    relay.a = types.SimpleNamespace(inst_period=0.0, inst_phase="")
    relay.phase, relay.phase_why = None, ""
    relay.collect = lambda: ({}, {}, {})
    relay.bright = {}
    relay.inst_t = -1e9
    unit = types.SimpleNamespace(
        name="node-a/0",
        source=True,
        inst_url="http://node-a:12048/gnss0_records/get_lane_inst?chain=gps_l5",
    )
    relay.units = [unit]
    relay.get_inst = lambda u: (u, {"error": "HTTPError 404"})
    relay.pool = concurrent.futures.ThreadPoolExecutor(1)
    logged = []
    relay.rl = lambda key, msg, every: logged.append((key, msg))
    try:
        status = relay.cycle(post=False)
    finally:
        relay.pool.shutdown()
    check(
        logged
        == [(("inst", "node-a/0"), "get_lane_inst %s: HTTPError 404" % unit.inst_url)]
        and status["inst"]["instances"] == 0,
        "get_lane_inst's failure is logged with its url (%s)" % logged,
    )


class _Unit:
    def __init__(self, name, live):
        self.name, self.node, self.live = name, name.split("/")[0], live

    def wants(self, ranges):
        return lane_relay.Gpu.wants(self, ranges)


def test_bright_transfer():
    best = {
        ("bds_b3i", 14): ({"power": 400.0}, "cx51/1"),
        ("bds_b3i", 21): ({"power": 10.0}, "cx27/0"),
        ("bds_b3i", 22): ({"power": 8.0}, "cx27/0"),
        ("glo_l2of", 4): ({"power": 50.0}, "cx27/0"),
    }
    powers = {"bds_b3i": [400.0, 10.0, 8.0, 12.0], "glo_l2of": [50.0, 40.0]}
    b = lane_relay.bright_lanes(best, powers, 15.0)
    check(
        b == {("bds_b3i", 14)},
        "only the lane 15 dB above its band's median (%s)" % sorted(b),
    )
    check(
        lane_relay.bright_lanes(best, powers, 20.0) == set(),
        "a higher gate passes none",
    )
    check(("glo_l2of", 4) not in b, "fewer than 3 lanes in a band: no gate")

    relay = lane_relay.Relay.__new__(lane_relay.Relay)
    relay.table = {"glo_l2of": [[6339, 6440]]}
    relay.bright = {"bds_b3i": [[6234, 6450]], "glo_l2of": [[6234, 6338]]}
    src = _Unit("cx51/1", [6237, 6301, 6390, 6494])
    other = _Unit("cx27/0", [6240, 6400])
    lane = {
        "freq_id": 6494,
        "t_utc": 0.0,
        "los": [0, 0, 1],
        "los_rate": [0, 0, 0],
        "coh": 0.9,
        "power": 400.0,
        "owner": "/gnss1_bds_b3i_n2assemble/lane/14",
        "q": [[1, 0]],
    }
    best = {("bds_b3i", 14): (lane, src)}
    body = relay.body_for(src, best, None, {("bds_b3i", 14)})
    check(
        len(body["lanes"]) == 1 and body["lanes"][0]["fids"] == [[6234, 6450]],
        "a bright lane goes back to its own GPU, with only the bright targets",
    )
    check(
        relay.body_for(src, best, None, set())["lanes"] == [],
        "not bright: a band with no transfer table sends nothing",
    )
    lane2 = dict(lane, owner="/gnss0_glo_l2of_n2assemble/lane/4", freq_id=6390)
    best2 = {("glo_l2of", 4): (lane2, src)}
    body = relay.body_for(other, best2, None, {("glo_l2of", 4)})
    check(
        body["lanes"][0]["fids"] == [[6339, 6440], [6234, 6338]],
        "another GPU gets the fleet targets plus the bright ones",
    )
    body = relay.body_for(src, best2, None, {("glo_l2of", 4)})
    check(
        body["lanes"][0]["fids"] == [[6234, 6338]], "its own GPU only the bright ones"
    )


def test_served_lane():
    """A lane its GPU only serves (L1 C/A) goes back to that GPU too; of one satellite's
    L1 C/A and L2C lanes only L2C's goes to the shared L2 targets."""
    relay = lane_relay.Relay.__new__(lane_relay.Relay)
    relay.table = {"gps_l1ca": [[6181, 6390]], "gps_l2c": [[6181, 6390]]}
    relay.bright = {}
    relay.sys = {"gps_l1ca": "G", "gps_l2c": "G"}
    src, other = _Unit("cx43/0", [6237, 6301]), _Unit("cx43/1", [6285, 6301])
    l1 = {
        "freq_id": 8070,
        "t_utc": 0.0,
        "los": [0, 0, 1],
        "los_rate": [0, 0, 0],
        "coh": 0.9,
        "power": 40.0,
        "owner": "/gnss0_gps_l1ca_n2assemble/lane/19",
        "q": [[1, 0]],
    }
    best = {("gps_l1ca", 19): (l1, src)}
    body = relay.body_for(src, best, None, set(), keep=relay.nearest(best))
    check(
        len(body["lanes"]) == 1 and body["lanes"][0]["fids"] == [[6181, 6390]],
        "a lane measured on a channel its GPU does not project goes back to it",
    )
    l1b = dict(l1, owner="/gnss0_gps_l1ca_n2assemble/lane/5")
    l2 = dict(l1, owner="/gnss1_gps_l2c_n2assemble/lane/5", freq_id=6285)
    best = {
        ("gps_l1ca", 19): (l1, src),
        ("gps_l1ca", 5): (l1b, src),
        ("gps_l2c", 5): (l2, other),
    }
    keep = relay.nearest(best)
    check(
        keep == {("gps_l1ca", 19), ("gps_l2c", 5)},
        "L1 C/A's lane only for the satellite with no L2C lane (%s)" % sorted(keep),
    )
    body = relay.body_for(other, best, None, set(), keep=keep)
    check(
        sorted(L["id"].rsplit("/", 1)[1] for L in body["lanes"]) == ["19"],
        "L2C's own GPU gets L1 C/A's lane of the other satellite only",
    )


def test_inst_phase():
    """The phase table goes along turned to the newest start, an empty one while the steps
    since its start are not measured, and an unchanged one only every PHASE_REPOST_S."""
    import json
    import fengine_steps

    d = tempfile.mkdtemp()
    tab = {
        "made_frame0": 100,
        "adc_sample_ns": 0.3125,
        "n_inputs": 4,
        "fids": [6000, 6010],
        "phase_deg": [[0.0, 10.0, None, -20.0], [5.0, 15.0, 30.0, -15.0]],
        "t_tr_utc": "x",
    }
    paths = {k: os.path.join(d, k + ".json") for k in ("tab", "frame0", "steps")}
    json.dump(tab, open(paths["tab"], "w"))
    json.dump([{"frame0_int": 100}, {"frame0_int": 200}], open(paths["frame0"], "w"))
    steps = {"adc_sample_ns": 0.3125, "n_inputs": 4, "rows": []}
    fengine_steps.add_start(steps, 100, "a")
    fengine_steps.add_start(steps, 200, "b")
    fengine_steps.save(steps, paths["steps"])
    r = lane_relay.Relay.__new__(lane_relay.Relay)
    r.a = types.SimpleNamespace(
        inst_phase=paths["tab"], frame0=paths["frame0"], steps=paths["steps"]
    )
    r.phase, r.phase_why, r.phase_sent = None, "", {}
    u = types.SimpleNamespace(name="cx43/0")
    check(r.phase_due(u) is None, "nothing read, nothing sent")
    r.phase = r.read_phase()
    check(
        r.phase == {} and "not measured" in r.phase_why,
        "an unmeasured start clears the table (%s)" % r.phase_why,
    )
    steps["rows"][-1]["steps"] = [0, 4, 0, -1]
    steps["rows"][-1]["pol_step"] = 0
    fengine_steps.save(steps, paths["steps"])
    r.phase = r.read_phase()
    check(
        r.phase.get("made_frame0") == 200 and r.phase_why == "",
        "turned to the newest start",
    )
    turn = 360.0 * 6000 * 195312.5 * 4 * 0.3125e-9
    got = r.phase["phase_deg"][0][1]
    check(
        abs((got - 10.0 - turn + 180.0) % 360.0 - 180.0) < 0.01,
        "input 1 turned by its 4 samples",
    )
    check(r.phase["phase_deg"][0][2] is None, "an empty entry stays empty")
    body = r.body_for(u, {}, None, (), r.phase_due(u))
    check(body.get("inst_phase") == r.phase, "the table goes in the body")
    key = json.dumps(r.phase, sort_keys=True)
    r.phase_sent[u.name] = (key, lane_relay.time.monotonic())
    check(r.phase_due(u) is None, "an unchanged table is not sent again at once")
    r.phase_sent[u.name] = (
        key,
        lane_relay.time.monotonic() - lane_relay.PHASE_REPOST_S,
    )
    check(r.phase_due(u) == r.phase, "but is after PHASE_REPOST_S")


def main():
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
