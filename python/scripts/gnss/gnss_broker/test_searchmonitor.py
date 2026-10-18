#!/usr/bin/env python3
"""--detectors-monitor: the withheld search's clock against the joint clock (#191 step 1).

    /home/kvand/gnss/venv-ft/bin/python -m gnss_broker.test_searchmonitor

The question the experiment asks is a difference of two clocks in one chain's chips, so the
checks pin the sign (search minus joint), the wrap (a search clock just under the code period
against a joint just over 0 is a small negative, not a whole code), the metres, and that
with no joint clock the line still reports the search's own.

@author Keith Vanderlinde
"""
import types

from gnss_broker import deadreckon
from gnss_broker.receiver import Receiver

REF_RATE, REF_LEN = 10.23e6, 10230.0
_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def _run(offs, joint_clk, joint=True, now=100.0, state=None):
    lines = []
    deadreckon._log_rl = lambda key, msg, every_s=0.0: lines.append(msg)
    deadreckon._log = lambda msg: lines.append(msg)
    js = types.SimpleNamespace(clk=joint_clk)
    rx = types.SimpleNamespace(
        joint_unit=lambda: (REF_RATE, REF_LEN) if joint else None,
        joint_clk_delta=Receiver.joint_clk_delta,
    )
    ctx = types.SimpleNamespace(
        args=types.SimpleNamespace(chip_rate_hz=REF_RATE),
        code_len=REF_LEN,
        band_id="t",
        chain_id="gal_e5a",
        rx=rx,
        dr_state=state if state is not None else {},
        drp=types.SimpleNamespace(now_w=now),
        joint_state=lambda *a: js,
    )
    deadreckon.search_monitor(ctx, offs)
    return lines


def main():
    print("circular median")
    med, mad = deadreckon.circular_median([(1, 150.0), (2, 150.3), (3, 149.8)], REF_LEN)
    check(abs(med - 150.0) < 1e-9 and abs(mad - 0.2) < 1e-9, "median 150.0, MAD 0.2")
    # Five, not four: the solve's median is the upper one of an even count.
    med, mad = deadreckon.circular_median(
        [(1, 10229.9), (2, 0.1), (3, 10229.8), (4, 5000.0), (5, 10229.7)], REF_LEN
    )
    check(abs(((med - 10229.9 + REF_LEN / 2) % REF_LEN) - REF_LEN / 2) < 0.05 and mad < 0.3,
          "a cluster straddling the wrap medians inside it, the outlier does not drag it")

    print("\nsearch minus joint")
    lines = _run([(1, 150.0), (2, 150.3), (3, 149.8)], 150.6)
    l = [x for x in lines if x.startswith("SEARCH-MONITOR gal_e5a:")]
    check(len(l) == 1 and "search - joint -0.600 chips = -17.6 m" in l[0],
          "search 150.0 vs joint 150.6: -0.600 chips = -17.6 m (the sign is search - joint)")
    check("1:-0.60" in l[0] and "2:-0.30" in l[0] and "3:-0.80" in l[0],
          "per PRN against the joint clock")
    lines = _run([(1, 10229.9), (2, 10229.7), (3, 0.1)], 0.4)
    l = [x for x in lines if x.startswith("SEARCH-MONITOR gal_e5a:")]
    check(len(l) == 1 and "search - joint -0.500 chips" in l[0],
          "search 10229.9 vs joint 0.4 wraps to -0.500, not a whole code")
    lines = _run([(1, 150.0)], 150.6, joint=False)
    check(any("no joint clock to compare with" in x for x in lines),
          "without a joint clock the search's own clock is still reported")

    print("\nthe 5-minute summary")
    st, lines = {}, []
    for i in range(4):
        lines += _run([(1, 150.0 + 0.01 * i)], 150.6, now=100.0 + 2.0 * i, state=st)
    s = [x for x in lines if "5-min" in x]
    check(len(s) == 1 and "over 3 cycles" in s[0] and "median -0.5" in s[0],
          "the 5-min median line appears once the history holds 3 cycles, then not again "
          "within 60 s (%s)" % (s[0][:70] if s else "none"))

    print("\nFAILED (%d)" % len(_fails) if _fails else "\nOK")
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
