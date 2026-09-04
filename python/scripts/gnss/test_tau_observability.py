#!/usr/bin/env python3
"""The degenerate tau row destroys the clock (2026-08-21 incident, fixed 2026-08-22).

REPRODUCES THE MEASURED FAILURE. On 2026-08-21 four broker runs of identical code and
config split two ways, decided by which chain constructed the joint state first:

    ref_band latched to the MEASUREMENT band  ->  sigma(clk) 0.11-0.30, converged
    ref_band latched to the OTHER band        ->  sigma(clk) 19.901 forever, biases to -1693

In the second case 1176.45 MHz -- where every code measurement lives -- was given a tau row
with `dual 0`. tau separates from clk ONLY through satellites seen in BOTH bands, so with
none the two are exactly degenerate: the clock inherits sigma_tau0 and never converges,
P00 stays at 396, and `birth_max`'s `P00 < 100` precondition is therefore never satisfied,
leaving every birth unvetted for the life of the process.

Both fixes are asserted here, and both are shown to FAIL against the old behaviour --
tau_min_dual=0 / birth_gate_after=huge restore it exactly, which is what makes this a gate
rather than a decoration.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnss_broker.state_filter import JointReceiverState

REF, MEAS = "1207.14MHz", "1176.45MHz"
CLK = 151.0


def _run(**kw):
    """Feed 40 cycles of single-band measurements, ref_band pinned to the OTHER band."""
    js = JointReceiverState(code_len=10230.0, ref_band=REF, **kw)
    t = 0.0
    for cyc in range(40):
        t += 2.0
        for prn in range(1, 7):
            js.update(("gal_e5a", prn), CLK + 0.1 * prn, 0.3, t, band=MEAS)
        js.gauge()
    return js


def test_no_row_when_unobservable():
    js = _run()
    assert MEAS not in js._band_idx, "a tau row was created with zero dual-band satellites"
    assert js.tau(MEAS) == 0.0, "tau must be pinned at 0, got %r" % js.tau(MEAS)
    print("  no tau row when dual=0                      OK")


def test_clock_converges():
    js = _run()
    sig = js.sigma()
    assert sig < 1.0, "sigma(clk) did not converge: %.3f (the incident sat at 19.901)" % sig
    assert abs(js.clk - CLK) < 2.0, "clk %.3f is nowhere near truth %.1f" % (js.clk, CLK)
    print("  sigma(clk) converged to %.4f                OK" % sig)


def test_old_behaviour_reproduces_the_failure():
    """THE NEGATIVE CONTROL. Without the fix the clock must NOT converge."""
    js = _run(tau_min_dual=0)
    assert MEAS in js._band_idx, "control did not even create the row"
    sig = js.sigma()
    assert sig > 5.0, ("control converged (%.3f) -- then this test proves nothing and the "
                       "fix is untested" % sig)
    print("  control (tau_min_dual=0) stuck at %.3f     OK" % sig)


def test_birth_gate_cannot_be_held_open_by_a_stuck_P00():
    """birth_max must apply on evidence, not only on P00 < 100."""
    js = _run(tau_min_dual=0)          # the degenerate state: P00 pinned high
    assert js.P[0, 0] > 100.0, "control P00 %.1f is not the stuck regime" % js.P[0, 0]
    n0 = len(js._idx)
    js.update(("gal_e5a", 99), CLK + 900.0, 0.3, 200.0, band=MEAS)   # absurd newcomer
    assert len(js._idx) == n0, "a 900-chip newcomer was born despite birth_max"
    print("  birth_max applies with P00=%.0f              OK" % js.P[0, 0])


def test_birth_gate_control():
    """And it MUST be admitted when the evidence bound is removed -- else the gate is
    firing for some other reason and the test is measuring nothing."""
    js = _run(tau_min_dual=0, birth_gate_after=10 ** 9)
    n0 = len(js._idx)
    js.update(("gal_e5a", 99), CLK + 900.0, 0.3, 200.0, band=MEAS)
    assert len(js._idx) == n0 + 1, ("control refused the birth too -- birth_gate_after is "
                                    "not what admitted it")
    print("  control admits it without the bound         OK")


if __name__ == "__main__":
    for fn in (test_no_row_when_unobservable, test_clock_converges,
               test_old_behaviour_reproduces_the_failure,
               test_birth_gate_cannot_be_held_open_by_a_stuck_P00,
               test_birth_gate_control):
        fn()
    print("test_tau_observability: ALL PASS")
