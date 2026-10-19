#!/usr/bin/env python3
"""rf_stats_gate.py's fixture rewrite, offline: no kotekan build needed.

    python3 scripts/gnss/test_rf_stats_gate_cfg.py

Each arm of the gate runs config/gates/rf_stats_gate.yaml with testDataGen's constu8 byte and
the tap's band_power_chans rewritten. The rewritten yaml must carry the arm's byte and channels,
and a fixture the rewrite cannot find its anchors in must be refused.
"""
import os
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rf_stats_gate  # noqa: E402

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def test_value_and_chans_are_rewritten():
    cfg = yaml.safe_load(rf_stats_gate.render_cfg(0x8F, [0, 1, 2]))
    check(
        cfg["gen"]["value"] == 0x8F, "constu8 value 0x8F (got %r)" % cfg["gen"]["value"]
    )
    check(
        cfg["tap"]["band_power_chans"] == [0, 1, 2],
        "band_power_chans [0, 1, 2] (got %r)" % cfg["tap"]["band_power_chans"],
    )
    cfg = yaml.safe_load(rf_stats_gate.render_cfg(0xFF, []))
    check(
        cfg["gen"]["value"] == 0xFF, "constu8 value 0xFF (got %r)" % cfg["gen"]["value"]
    )
    check(
        cfg["tap"]["band_power_chans"] == [],
        "band_power_chans [] for the control (got %r)" % cfg["tap"]["band_power_chans"],
    )


def test_missing_anchor_is_refused():
    with open(rf_stats_gate.CFG) as f:
        src = f.read()
    orig = rf_stats_gate.CFG
    with tempfile.TemporaryDirectory() as td:
        rf_stats_gate.CFG = os.path.join(td, "rf_stats_gate.yaml")
        with open(rf_stats_gate.CFG, "w") as f:
            f.write(src.replace("type: constu8", "type: const"))
        try:
            rf_stats_gate.render_cfg(0x8F, [0, 1, 2])
            refused = False
        except RuntimeError:
            refused = True
        finally:
            rf_stats_gate.CFG = orig
    check(refused, "a fixture without the constu8 value line is refused")


def main():
    print("rf_stats_gate.py fixture rewrite checks")
    test_value_and_chans_are_rewritten()
    test_missing_anchor_is_refused()
    if _fails:
        print("FAILED (%d)" % len(_fails))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
