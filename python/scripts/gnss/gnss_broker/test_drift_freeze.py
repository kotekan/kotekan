"""The dead-reckon clock DRIFT must not be measured on frozen data, and must not outlive it.

    /home/kvand/gnss/venv-ft/bin/python -m gnss_broker.test_drift_freeze     (from python/scripts/gnss)

WHY THIS EXISTS (2026-09-03 19:37 UTC). The F-engine stopped streaming; every instance's
pow_hop froze; `*** TIME BASE FROZEN` printed on schedule -- LOG ONLY. The orbit model kept
moving with the wall clock against the frozen detections, so consecutive clock solves
differed by the SKY's motion and dr_clock_quality read that as clock drift: the 0.05-EMA
walked -0.005 -> -0.031 -> -0.26 -> -0.3543 chips/s with every step inside the 1.0 bound,
and then held -0.3543 for SEVENTEEN HOURS, through an F-engine restart and a fleet restart,
because a clock walking 10 chips between solves seeds every tracker off-peak and no solve
ever came back to re-measure it. Same-band adoption copied it to gal_e5a and bds_b2a within
10 s; the cross-band bootstrap does not carry drift, so the other five chains were spared and
the fault wore a band-split costume that read as "GPSDO lost". #75 (2026-08-18) was the same
class; the detector it produced was left log-only (the #98 shape).

Three properties, each a test:
  1. FROZEN DATA FREEZES THE STATE: while the newest telemetry hop has not advanced past
     --dr-clock-freeze-s, the raw solve is not applied and no drift is differenced; raw_prev
     is dropped so the first pair across the gap is never taken either.
  2. A DRIFT EXPIRES: a value unrefreshed for --dr-drift-max-age-s reverts to the (l-a) model.
  3. ADOPTION CARRIES "UNKNOWN": the adopter takes the donor's drift even when it is None,
     instead of keeping its own.

@author Keith Vanderlinde
"""

import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnss_broker import deadreckon as dr  # noqa: E402

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


class _Rx:
    def __init__(self):
        self.contrib = []

    def contribute_dr_clock(self, chain, band, chips, drift, t, code_length, chip_rate_hz=None):
        self.contrib.append((chips, drift, t))


def _ctx(now, hop_t, freeze_s=6.0, drift=None, drift_t=None, clk=150.0, raw=150.0, offs=6):
    c = NS()
    c.args = NS(dr_min_sats=4, dr_clock_drift=None, dr_max_drift_chips_s=1.0,
                dr_clock_alpha=0.2, chip_rate_hz=10.23e6, dr_clock_freeze_s=freeze_s,
                dr_drift_max_age_s=600.0)
    c.code_len = 10230.0
    c.chain_id, c.band_id = "gps_l5", "1176.45MHz"
    c.rx = _Rx()
    c.fe_axis = [(30781548544.0, hop_t)] if hop_t is not None else [None]
    c.drp = NS(offs=[(p, 0.0) for p in range(offs)], raw_clk=raw, now_w=now, drift=0.0,
               t_code=0.02, la=1e-9)
    c.dr_state = {"clk": clk, "clk_t": now - 2.0, "drift": drift}
    if drift_t is not None:
        c.dr_state["drift_t"] = drift_t
    return c


def test_frozen_axis_holds_the_solve():
    t = 1000.0
    # live axis: two solves 2 s apart, the second 1 chip away -> drift measured
    c = _ctx(now=t, hop_t=t - 0.5, raw=150.0)
    dr.dr_clock_quality(c)
    c.drp.now_w = t + 2.0
    c.fe_axis[0] = (c.fe_axis[0][0] + 390625.0, t + 1.5)
    c.drp.raw_clk = 149.0
    c.dr_state["clk_t"] = t
    dr.dr_clock_quality(c)
    check(c.dr_state.get("drift") is not None and abs(c.dr_state["drift"] + 0.5) < 1e-9,
          "live axis: drift measured from the pair (-0.5 chips/s)")
    check(c.dr_state.get("drift_t") == t + 2.0, "live axis: drift carries the measurement stamp")
    n_contrib = len(c.rx.contrib)
    # now the hop stops advancing; 7 s later a 'solve' arrives that has moved 10 chips
    # (the sky moved, the data did not)
    c.drp.now_w = t + 9.0
    c.drp.raw_clk = 139.0
    drift_before, clk_before = c.dr_state["drift"], c.dr_state["clk"]
    dr.dr_clock_quality(c)
    check(c.dr_state["drift"] == drift_before, "frozen axis: drift EMA NOT updated")
    check(c.dr_state["clk"] == clk_before, "frozen axis: clock NOT stepped toward the stale solve")
    check("raw_prev" not in c.dr_state, "frozen axis: raw_prev dropped")
    check(c.dr_state.get("clk_frozen") is True, "frozen axis: state marked frozen")
    check(len(c.rx.contrib) == n_contrib, "frozen axis: nothing published to siblings")
    # still frozen 60 s later: still held (no re-log storm either -- just held)
    c.drp.now_w = t + 69.0
    c.drp.raw_clk = 100.0
    dr.dr_clock_quality(c)
    check(c.dr_state["drift"] == drift_before and "raw_prev" not in c.dr_state,
          "frozen 60 s: still held, still no raw_prev")
    # axis advances again: this cycle's solve is the FIRST of a new pair -- no
    # difference across the gap, drift untouched, raw_prev re-armed
    c.drp.now_w = t + 71.0
    c.fe_axis[0] = (c.fe_axis[0][0] + 1.0, t + 70.5)
    c.drp.raw_clk = 151.0
    dr.dr_clock_quality(c)
    check(c.dr_state["drift"] == drift_before, "resume: first solve after the gap is not differenced")
    check(c.dr_state.get("raw_prev") == (151.0, t + 71.0), "resume: raw_prev re-armed")
    check("clk_frozen" not in c.dr_state, "resume: frozen mark cleared")
    # and the next pair measures again
    c.drp.now_w = t + 73.0
    c.fe_axis[0] = (c.fe_axis[0][0] + 1.0, t + 72.5)
    c.drp.raw_clk = 151.2
    dr.dr_clock_quality(c)
    check(abs(c.dr_state["drift"] - (drift_before + 0.05 * (0.1 - drift_before))) < 1e-9,
          "resume: second solve after the gap measures (EMA step toward +0.1)")


def test_freeze_disabled_is_the_old_behaviour():
    t = 1000.0
    c = _ctx(now=t, hop_t=t - 100.0, freeze_s=0.0, raw=150.0)
    dr.dr_clock_quality(c)
    check(c.dr_state.get("raw_prev") == (150.0, t), "freeze_s=0: solve applied on a stale axis (opt-out)")


def test_no_axis_at_all_does_not_freeze():
    t = 1000.0
    c = _ctx(now=t, hop_t=None, raw=150.0)
    dr.dr_clock_quality(c)
    check(c.dr_state.get("raw_prev") == (150.0, t), "no fe_axis yet: solve applied (nothing to be stale)")


def test_drift_expires():
    """The expiry lives at the propagation site (a big function); assert the RULE on its text
    and exercise the arithmetic the same way, so the two cannot drift silently."""
    import inspect
    src = inspect.getsource(dr)
    check("dr_drift_max_age_s" in src and "EXPIRED (--dr-drift-max-age-s" in src,
          "expiry rule present at the propagation site")
    i = src.index("EXPIRED (--dr-drift-max-age-s")
    j = src.index('ctx.drp.drift = ctx.dr_state.get("drift")', i)
    check('ctx.dr_state["drift"] = None' in src[i:j], "expiry sets drift to None (falls back to the (l-a) model)")
    k = src.rindex("_dmax = getattr(ctx.args", 0, i)
    cond = src[k:i]
    check("ctx.drp.now_w - _dt > _dmax" in cond, "expiry is age > max, on the measurement stamp")
    check("_dt is None" in src[i:j] and 'ctx.dr_state["drift_t"] = ctx.drp.now_w' in src[i:j],
          "an unstamped drift gets one lifetime from now, not forever")


def test_adopt_carries_unknown():
    import inspect
    src = inspect.getsource(dr.dr_clock_adopt)
    check('if dr.get("drift_chips_s") is not None else None' in src,
          "file-transport adopt: drift := donor's, None included")
    src2 = inspect.getsource(dr)
    check('ctx.dr_state["drift"] = float(_sd) if _sd is not None else None' in src2,
          "in-process adopt: drift := donor's, None included")
    check('ctx.dr_state["drift_t"] = ctx.drp.rx_sib.t' in src2,
          "in-process adopt: stamped with the donor's time")


if __name__ == "__main__":
    test_frozen_axis_holds_the_solve()
    test_freeze_disabled_is_the_old_behaviour()
    test_no_axis_at_all_does_not_freeze()
    test_drift_expires()
    test_adopt_carries_unknown()
    if _fails:
        print("FAILED: %d" % len(_fails))
        sys.exit(1)
    print("OK")
