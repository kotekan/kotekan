#!/usr/bin/env python3
"""lane_relay.py's owner parse against the names a node serves.

    python3 scripts/gnss/site/test_lane_relay.py

A lane's owner is the assembler's kotekan unique_name, which starts with '/', plus
"/lane/<prn>" (GnssGpuRecordAssemble).
"""
import os
import sys
import tempfile

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
    check(
        lane_relay.inst_stage(
            {"gnss0_gps_l2c_n2assemble": cfg["gnss0_gps_l2c_n2assemble"]}, 0
        )
        is None,
        "one lane channel serves no estimate",
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
    src, other = _Unit("cx51/1", [6237, 6301]), _Unit("cx27/0", [6240, 6400])
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
