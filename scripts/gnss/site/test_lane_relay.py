#!/usr/bin/env python3
"""lane_relay.py's owner parse against the names a node serves.

    python3 scripts/gnss/site/test_lane_relay.py

A lane's owner is the assembler's kotekan unique_name, which starts with '/', plus
"/lane/<prn>" (GnssGpuRecordAssemble).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lane_relay  # noqa: E402

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def test_parse_owner():
    check(lane_relay.parse_owner("/gnss1_l2c_n2assemble/lane/11") == ("l2c", 11), "tagged owner")
    check(lane_relay.parse_owner("/gnss0_n2assemble/lane/26") == ("l5", 26), "untagged = l5")
    check(lane_relay.parse_owner("gnss1_e6_n2assemble/lane/3") == ("e6", 3), "no leading slash")
    check(lane_relay.parse_owner("/gnss1_l2c_n2combine/lane/11") == (None, None), "not an assembler")


def test_inst_stage():
    cfg = {
        "gnss0_n2assemble": {"channel_ids": [1, 2, 3]},
        "gnss0_l5_n2assemble": {"channel_ids": [6103, 6119], "lane_transfer_freq_ids": [6110]},
        "gnss0_l2c_n2assemble": {"channel_ids": [6285], "lane_transfer_freq_ids": [6250]},
        "gnss1_e6_n2assemble": {"channel_ids": [6541, 6557], "lane_transfer_freq_ids": [6480]},
        "gnss1_e5b_n2assemble": {"channel_ids": [6151, 6167, 6183], "lane_transfer_freq_ids": [6200]},
    }
    check(lane_relay.inst_stage(cfg, 0) == "gnss0_l5_n2assemble", "l5 when the GPS one transfers nothing")
    check(lane_relay.inst_stage(cfg, 1) == "gnss1_e5b_n2assemble", "else the most lane channels")
    cfg["gnss0_n2assemble"]["lane_transfer_freq_ids"] = [6000]
    check(lane_relay.inst_stage(cfg, 0) == "gnss0_n2assemble", "the GPS L5 one first")
    check(lane_relay.inst_stage({"gnss0_l2c_n2assemble": cfg["gnss0_l2c_n2assemble"]}, 0) is None,
          "one lane channel serves no estimate")


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
