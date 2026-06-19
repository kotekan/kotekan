import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python", "scripts"))
import gps_broker as gb  # noqa: E402


def W(name, signal="GPS_L1CA", lo=1, hi=32):
    return gb.Worker(name, signal, lo, hi)


def test_load_balances_across_equal_workers():
    workers = [W("a"), W("b")]
    targets = [("GPS_L1CA", p) for p in (1, 2, 3, 4)]
    wp, unplaced = gb.assign_targets(workers, targets)
    assert not unplaced
    assert len(wp["a"]) == 2 and len(wp["b"]) == 2     # evenly split
    assert sorted(wp["a"] + wp["b"]) == [1, 2, 3, 4]   # all placed once


def test_respects_prn_subset():
    workers = [W("a", lo=1, hi=16), W("b", lo=17, hi=32)]
    wp, _ = gb.assign_targets(workers, [("GPS_L1CA", 5), ("GPS_L1CA", 20)])
    assert wp["a"] == [5]
    assert wp["b"] == [20]


def test_signal_must_match():
    workers = [W("a", signal="GPS_L1CA")]
    wp, unplaced = gb.assign_targets(workers, [("GPS_L2C_CL", 9)])
    assert wp["a"] == []
    assert unplaced == [("GPS_L2C_CL", 9)]


def test_deterministic_order():
    # Names out of order in -> same result as sorted; first of a tie is 'a'.
    wp, _ = gb.assign_targets([W("b"), W("a")], [("GPS_L1CA", 7)])
    assert wp["a"] == [7] and wp["b"] == []


def test_parse_workers_defaults_and_fields():
    ws = gb.parse_workers("gps_a,gps_b")
    assert [w.name for w in ws] == ["gps_a", "gps_b"]
    assert all(w.signal == "GPS_L1CA" and (w.prn_lo, w.prn_hi) == (1, 32) for w in ws)

    ws = gb.parse_workers("l2:GPS_L2C_CL:5-10")
    assert ws[0] == gb.Worker("l2", "GPS_L2C_CL", 5, 10)
