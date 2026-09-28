"""#142: the receiver clock's guards -- at the source that solves it, and on the paths that copy it.

    /home/kvand/gnss/venv-ft/bin/python -m gnss_broker.test_clockguard     (from python/scripts/gnss)

The legacy clock solve is a circular median of per-satellite code offsets from the search's
detections. Inside a boresight transit the search reports noise code phases just over its bar,
two of them agreeing inside the MAD bound pass as a median, and the clock steps hundreds of
chips; every consumer then copied the step (the cross-band bootstrap, unbounded) or escaped
onto it (#104's 300-s staleness escape). Every check below drives the REAL functions from a
FRESH state -- a prime, then the first solve -- because a mid-stream replay never exercises the
first cycles after a restart, and that is where a guard is most likely to deadlock or crash.

  SOURCE    dr_clock_solve + dr_clock_quality: the bootstrap stays unlimited; an established
            clock moves only on >= --dr-update-min-sats with a step inside
            --dr-clock-step-max-chips; a larger step needs consecutive agreeing solves of NEW
            detections (a re-pin); a confirmed clock freezes inside a transit, never re-rolls,
            and is held at zero rate and contributed whenever it is not updated.
  COPY      dr_clock_adopt_rx (cross-band bootstrap + same-band adoption) and dr_joint_clk:
            the cross-band bootstrap is bounded by the #104 bound as a TIME; a refusal holds at
            zero rate; the 300-s escape is suppressed while JOINT-CLK adopts, and returns 300 s
            after it stops.

@author Keith Vanderlinde
"""

import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnss_broker import deadreckon as dr  # noqa: E402
from gnss_broker.receiver import Receiver  # noqa: E402
from gnss_broker.sky import BORESIGHT_AZ_DEG, BORESIGHT_EL_DEG  # noqa: E402

_fails = []
L5_RATE, L5_LEN = 10.23e6, 10230.0
E6_RATE, E6_LEN = 5.115e6, 5115.0
L2C_RATE, L2C_LEN = 0.5115e6, 10230.0
TRUE_CLK = 150.2                       # reference chips (10.23 Mcps), the receiver clock
LOGS = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def _capture():
    dr._log = lambda msg: LOGS.append(msg)
    dr._log_rl = lambda key, msg, every_s=10.0: LOGS.append(msg)


def _logged(sub, since=0):
    return [m for m in LOGS[since:] if sub in m]


class _Rx:
    """The contribution side of the Receiver, recorded."""

    def __init__(self):
        self.contrib = []

    def contribute_dr_clock(self, chain, band, chips, drift, t, code_length, chip_rate_hz=None,
                            epoch=None, held=False):
        self.contrib.append((t, chips, drift))


def _wrap(d, L=L5_LEN):
    return ((d + L / 2.0) % L) - L / 2.0


# ---- the solving chain ------------------------------------------------------------------------
def _l5(freeze_deg=5.0, min_sats=4, step_max=5.0, repin=3):
    """gps_l5 as the broker starts it: --dr-clock-chips 0.0 PRIME, no drift, the #142 knobs."""
    c = NS()
    c.args = NS(dr_min_sats=2, dr_max_solve_mad_chips=100.0, dr_solve_refused_rebootstrap_s=300.0,
                dr_clock_drift=None, dr_max_drift_chips_s=1.0, dr_clock_alpha=0.2,
                chip_rate_hz=L5_RATE, dr_clock_freeze_s=6.0, dr_drift_max_age_s=600.0,
                dr_update_min_sats=min_sats, dr_clock_step_max_chips=step_max,
                dr_clock_repin_solves=repin, dr_clock_transit_freeze_deg=freeze_deg)
    c.code_len = L5_LEN
    c.chain_id, c.band_id = "gps_l5", "1176.45MHz"
    c.rx = _Rx()
    c.fe_axis = [None]
    c.drp = NS(offs=[], raw_clk=None, now_w=None, drift=0.0, t_code=0.02, la=0.0, pd={})
    c.dr_state = {"clk": 0.0, "clk_t": 0.0, "clk_primed": True, "drift": None}
    c.best = {}
    c.hop = 1000
    return c


def _sky(sep_deg=None):
    """A model sky: one satellite far from boresight, plus (optionally) one sep_deg from it."""
    pd = {("G", 1): {"el": 40.0, "az": 90.0}}
    if sep_deg is not None:
        pd[("C", 34)] = {"el": BORESIGHT_EL_DEG - sep_deg, "az": BORESIGHT_AZ_DEG}
    return pd


def _cycle(c, t, offsets, sky=None, fresh=True):
    """One dead-reckon pass of the solve: offsets are this cycle's per-satellite d_i (chips);
    fresh=False re-presents the previous cycle's detections (a stalled search table)."""
    c.t0 = c.drp.now_w = t
    c.fe_axis[0] = (int(t * 195312.5), t)
    c.drp.drift = c.dr_state.get("drift")
    if c.drp.drift is None:
        c.drp.drift = 0.0
    c.drp.pd = sky if sky is not None else _sky()
    c.drp.offs = [(p, d % L5_LEN) for p, d in enumerate(offsets, start=1)]
    if fresh:
        c.hop += 1000
    c.best = {p: (100.0, 0.0, 0.0, c.hop, 0, 0.0, 0.0) for p, _ in c.drp.offs}
    n0 = len(getattr(c.rx, "contrib", ()))
    dr.dr_clock_solve(c)
    dr.dr_clock_quality(c)
    return len(getattr(c.rx, "contrib", ())) > n0


def _good(n, clk=TRUE_CLK):
    """n real satellites: clk plus a per-satellite bias of a chip or two."""
    return [clk + b for b in (0.3, -0.4, 1.1, -0.9, 0.6, -1.3, 0.2, 0.9)[:n]]


def _clk_now(c):
    """dr_seed's clk_now: the clock the seeds actually ride."""
    return (c.dr_state["clk"] + c.drp.drift * (c.drp.now_w - c.dr_state["clk_t"])) % c.code_len


def _established(c, t0=100.0):
    """Cold start -> bootstrap -> one in-bound 5-sat update: a CONFIRMED clock at t0 + 2."""
    _cycle(c, t0, _good(5))
    _cycle(c, t0 + 2.0, _good(5))
    return t0 + 2.0


def test_cold_start_unchanged():
    print("cold start: the prime, the first solve, the first cycles")
    _capture()
    c = _l5()
    # cycle 1 of a restart: no detections yet (the ephemeris just loaded) -- nothing happens,
    # nothing is contributed, and nothing reads a field the first pass has not stamped
    ok = not _cycle(c, 100.0, [])
    check(ok and c.dr_state["clk"] == 0.0 and c.dr_state.get("clk_primed"),
          "no detections: the 0.00 prime stands, nothing contributed")
    # the first solve: a 2-sat median 150 chips from the prime, with a satellite AT boresight
    n = len(LOGS)
    got = _cycle(c, 102.0, [TRUE_CLK + 0.3, TRUE_CLK - 0.4], sky=_sky(0.2))
    check(got and abs(c.dr_state["clk"] - (TRUE_CLK + 0.3)) < 1e-9,
          "first solve of --dr-min-sats 2 SNAPS 150 chips from the prime (never step-limited, "
          "min-sats not raised, transit ignored)")
    check(_logged("receiver clock BOOTSTRAP", n) and "clk_primed" not in c.dr_state,
          "...logged as the BOOTSTRAP, prime spent")
    check(c.dr_state.get("clk_confirmed") is False and c.dr_state.get("clk_transit") is None,
          "...an UNCONFIRMED clock: a satellite at boresight does not freeze it")
    # cycle 3: an in-bound 5-sat update confirms it; the EMA is the unchanged arithmetic
    before = c.dr_state["clk"]
    _cycle(c, 104.0, _good(5), sky=_sky(0.2))
    raw = c.drp.raw_clk
    exp = (before + 0.2 * _wrap(raw - before)) % L5_LEN
    check(abs(c.dr_state["clk"] - exp) < 1e-12 and c.dr_state.get("clk_confirmed") is True,
          "first in-bound 5-sat update: EMA exactly as before, clock now CONFIRMED")
    # cycle 4: confirmed + satellite at boresight -> the freeze engages
    n = len(LOGS)
    held = c.dr_state["clk"]
    got = _cycle(c, 106.0, [TRUE_CLK + 500.0] * 7, sky=_sky(0.2))
    check(got and c.dr_state["clk"] == held and _logged("receiver clock FROZEN", n),
          "confirmed clock + boresight satellite: FROZEN, a 7-sat +500 solve is not consulted")


def test_unconfirmed_bootstrap_not_frozen_but_repinned():
    print("a bad bootstrap inside a transit is corrected, not frozen")
    _capture()
    c = _l5()
    # restart inside a transit: the first solve is two noise detections agreeing
    _cycle(c, 100.0, [TRUE_CLK + 3000.0, TRUE_CLK + 3040.0], sky=_sky(1.0))
    check(abs(_wrap(c.dr_state["clk"] - TRUE_CLK - 3040.0)) < 1e-9,
          "bootstrap on two agreeing noise detections (the pre-#142 draw, unchanged)")
    # the real sky, thin (2-3 sats): an UNCONFIRMED clock re-pins on --dr-min-sats solves
    n = len(LOGS)
    for k in range(3):
        _cycle(c, 102.0 + 2 * k, _good(3), sky=_sky(1.0))
    check(abs(_wrap(c.dr_state["clk"] - TRUE_CLK)) < 2.0 and _logged("RE-PIN", n)
          and "UNCONFIRMED" in _logged("RE-PIN", n)[0],
          "3 consecutive agreeing 3-sat solves RE-PIN the unconfirmed clock onto the sky "
          "(inside the transit)")
    check(c.dr_state.get("clk_confirmed") is True, "...and the re-pin confirms it")


def test_two_sat_corrupted_step_rejected():
    print("two corrupted detections that agree do not move an established clock")
    _capture()
    c = _l5()
    t = _established(c)
    held = c.dr_state["clk"]
    n = len(LOGS)
    got = _cycle(c, t + 2.0, [TRUE_CLK + 527.0, TRUE_CLK + 549.0])
    check(got and c.dr_state["clk"] == held, "2-sat +538 solve (MAD 22): clock UNCHANGED, still contributed")
    check(_logged("HELD at", n) and "repin" not in c.dr_state,
          "...held as a THIN solve (< --dr-update-min-sats 4); builds no re-pin candidate")
    check(c.dr_state["clk_t"] == t + 2.0 and abs(_clk_now(c) - held) < 1e-12,
          "...ZERO-RATE hold: clk_t moved to now, dr_seed's clk_now is the held value")
    # the same with four satellites agreeing (a 4-sat corrupted solve): refused as a STEP
    n = len(LOGS)
    got = _cycle(c, t + 4.0, [TRUE_CLK + 527.0, TRUE_CLK + 549.0, TRUE_CLK + 560.0, TRUE_CLK - 1.0])
    check(got and c.dr_state["clk"] == held and _logged("step", n)
          and c.dr_state["repin"]["n"] == 1,
          "4-sat +549 solve: step REFUSED, clock unchanged, re-pin candidate 1/3")
    got = _cycle(c, t + 6.0, [TRUE_CLK - 2200.0, TRUE_CLK - 2190.0, TRUE_CLK - 2170.0, TRUE_CLK])
    check(c.dr_state["clk"] == held and c.dr_state["repin"]["n"] == 1,
          "a disagreeing 4-sat noise solve restarts the count (1/3), the clock never moves")
    # a normal solve clears it and updates
    _cycle(c, t + 8.0, _good(6))
    check("repin" not in c.dr_state and abs(_wrap(c.dr_state["clk"] - held)) < 1.0,
          "an in-bound 6-sat solve clears the candidate and updates by the EMA")


def test_genuine_repin_after_confirmation():
    print("a genuine re-pin: consecutive agreeing solves of new detections")
    _capture()
    c = _l5()
    t = _established(c)
    held = c.dr_state["clk"]
    new = TRUE_CLK + 300.0
    for k in range(2):
        _cycle(c, t + 2.0 + 2 * k, _good(5, new))
        check(c.dr_state["clk"] == held and c.dr_state["repin"]["n"] == k + 1,
              "solve %d of 3 at +300: held (candidate %d)" % (k + 1, k + 1))
    n = len(LOGS)
    got = _cycle(c, t + 6.0, _good(5, new))
    check(got and abs(_wrap(c.dr_state["clk"] - new)) < 1.5 and _logged("RE-PIN", n),
          "solve 3: RE-PIN, the step taken whole (snap), contributed")
    check("raw_prev" not in c.dr_state and c.dr_state.get("clk_confirmed") is True,
          "...no drift pair across the step; the clock stays confirmed")
    # a stalled search: the SAME detections three cycles running are one measurement
    c2 = _l5()
    t = _established(c2)
    held = c2.dr_state["clk"]
    _cycle(c2, t + 2.0, _good(5, new))
    _cycle(c2, t + 4.0, _good(5, new), fresh=False)
    _cycle(c2, t + 6.0, _good(5, new), fresh=False)
    check(c2.dr_state["clk"] == held and c2.dr_state["repin"]["n"] == 1,
          "stalled table: the same detections 3 times count ONCE -- no re-pin")
    _cycle(c2, t + 8.0, _good(5, new))
    _cycle(c2, t + 10.0, _good(5, new))
    check(abs(_wrap(c2.dr_state["clk"] - new)) < 1.5,
          "...fresh detections then complete it (3 independent measurements)")


def test_transit_freeze():
    print("the transit freeze: hold, contribute, no re-roll, resume")
    _capture()
    c = _l5()
    t = _established(c)
    held = c.dr_state["clk"]
    drift_before = c.dr_state.get("drift")
    n = len(LOGS)
    # 40 minutes inside the pooled veto: scattered solves, empty cycles, confident noise
    contributed = 0
    for k in range(1200):
        tt = t + 2.0 + 2.0 * k
        offs = ([] if k % 5 == 0 else
                [TRUE_CLK + 1000.0 * ((k * 7 + j * 3) % 9) for j in range(4)] if k % 5 == 1 else
                [TRUE_CLK + 700.0] * 6)
        contributed += bool(_cycle(c, tt, offs, sky=_sky(2.0 + (k % 3))))
    check(c.dr_state["clk"] == held and contributed == 1200,
          "1200 cycles (40 min): clock unchanged, contributed on EVERY cycle")
    check(len(_logged("receiver clock FROZEN", n)) == 1 and not _logged("FORCING", n),
          "...one FROZEN line; the MAD re-bootstrap never fires (its clock is stopped)")
    check(c.dr_state.get("drift") == drift_before and c.dr_state.get("drift_t") == c.drp.now_w
          if drift_before is not None else True,
          "...the drift is left alone and its age stands still")
    check(abs(_clk_now(c) - held) < 1e-12, "...zero rate: clk_now is the held value")
    n = len(LOGS)
    tt = c.drp.now_w + 2.0
    _cycle(c, tt, _good(6, TRUE_CLK + 2.0), sky=_sky(8.0))
    check(_logged("transit CLEAR", n) and c.dr_state.get("clk_transit") is None
          and abs(_wrap(c.dr_state["clk"] - held) - 0.4) < 1e-6,
          "clear of the veto: CLEAR logged, the next in-bound solve (+2) moves it by the EMA "
          "(+0.4)")
    # off: the same transit with --dr-clock-transit-freeze-deg 0 is not frozen
    c0 = _l5(freeze_deg=0.0)
    t = _established(c0)
    _cycle(c0, t + 2.0, _good(6), sky=_sky(0.1))
    check(c0.dr_state.get("clk_transit") is None and not c0.dr_state.get("clk_frozen"),
          "--dr-clock-transit-freeze-deg 0: no freeze at 0.1 deg")


def test_confirmed_clock_never_rerolled():
    print("MAD refusals: a confirmed clock is held; an unconfirmed one still re-rolls")
    _capture()
    c = _l5()
    t = _established(c)
    held = c.dr_state["clk"]
    n = len(LOGS)
    got = 0
    for k in range(200):                              # 400 s of scatter, no transit
        got += bool(_cycle(c, t + 2.0 + 2 * k, [TRUE_CLK + 2000.0 * j for j in range(5)]))
    check(c.dr_state["clk"] == held and got == 200 and not _logged("FORCING", n),
          "400 s of MAD refusals: confirmed clock held and contributed every cycle, no re-roll")
    got = _cycle(c, c.drp.now_w + 2.0, [TRUE_CLK + 900.0])
    check(got and c.dr_state["clk"] == held and c.dr_state["clk_t"] == c.drp.now_w,
          "one offset (no median at all): held at zero rate and contributed")
    # the pre-#142 latch escape still exists where nothing confirmed the clock
    c2 = _l5()
    _cycle(c2, 100.0, _good(3))                       # bootstrap, unconfirmed
    n = len(LOGS)
    for k in range(160):
        _cycle(c2, 102.0 + 2 * k, [TRUE_CLK + 2000.0 * j for j in range(5)])
    check(_logged("FORCING A RE-BOOTSTRAP", n), "unconfirmed clock: the 300-s re-roll still fires")


def test_knobs_off_is_pre_142():
    print("every knob at 0: the pre-#142 arithmetic, cycle for cycle")
    _capture()
    c = _l5(freeze_deg=0.0, min_sats=0, step_max=0.0)
    seq = [_good(5), _good(2, TRUE_CLK + 538.0), _good(4, TRUE_CLK + 549.0), _good(3), _good(6)]
    exp_clk = None
    ok = True
    for k, offs in enumerate(seq):
        _cycle(c, 100.0 + 2 * k, offs, sky=_sky(0.5))
        raw = c.drp.raw_clk
        if exp_clk is None:
            exp_clk = raw
        else:
            exp_clk = (exp_clk + 0.2 * _wrap(raw - exp_clk)) % L5_LEN   # drift 0.0 until measured
        ok = ok and abs(_wrap(c.dr_state["clk"] - exp_clk)) < 1e-6
    check(ok, "bootstrap then EMA on every solve, thin and stepped ones included")


# ---- the copy paths ---------------------------------------------------------------------------
def _consumer(rx, chain, rate, code_len, band):
    c = NS()
    c.args = NS(dr_clock_adopt=True, dr_clock_adopt_max_chips=5.0, chip_rate_hz=rate)
    c.rx = rx
    c.chain_id, c.band_id, c.code_len = chain, band, code_len
    c.drp = NS(offs=[], rx_sib=None, now_w=None)
    c.dr_state = {"clk": 0.0, "clk_t": 0.0, "clk_primed": True, "drift": None}
    return c


def _donor(rx, t, clk_ref):
    """gps_l5 contributing clk_ref reference chips at t (+ the NH joint fit's 20-ms epoch)."""
    rx.contribute_dr_clock("gps_l5", "1176.45MHz", clk_ref % L5_LEN, 0.0, t, L5_LEN,
                           chip_rate_hz=L5_RATE)
    rx.contribute_clock_mod_epoch("gps_l5", (clk_ref / L5_RATE) % 0.02, 0.02, 6, t)


def _adopt(c, t):
    c.t0 = c.drp.now_w = t
    dr.dr_clock_adopt_rx(c)


def test_crossband_bound_is_a_time():
    print("cross-band bootstrap: the #104 bound as a TIME at 10.23 / 5.115 / 0.5115 Mcps")
    _capture()
    for name, rate, ln in (("gal_e5b", L5_RATE, L5_LEN), ("gal_e6", E6_RATE, E6_LEN),
                           ("gps_l2c", L2C_RATE, L2C_LEN)):
        rx = Receiver(log=lambda m: None)
        c = _consumer(rx, name, rate, ln, "other")
        r = rate / L5_RATE
        _adopt(c, 99.0)
        check(c.drp.rx_sib is None and c.dr_state["clk"] == 0.0 and c.dr_state.get("clk_primed"),
              "%s: no donor yet (restart cycle 1): nothing adopted, the prime stands" % name)
        _donor(rx, 100.0, TRUE_CLK)
        _adopt(c, 100.5)
        check(abs(c.dr_state["clk"] - TRUE_CLK * r) < 1e-6 and "clk_primed" not in c.dr_state
              and c.dr_state.get("clk_src_t") == 100.0,
              "%s: first bootstrap unbounded from the prime, converted (%.4f chips)"
              % (name, c.dr_state["clk"]))
        # +4 reference chips (391 ns): inside the bound on every chain
        _donor(rx, 102.0, TRUE_CLK + 4.0)
        _adopt(c, 102.5)
        check(abs(c.dr_state["clk"] - (TRUE_CLK + 4.0) * r) < 1e-6,
              "%s: +4 ref chips (%.3f of ours, bound %.3f) ADOPTED" % (name, 4.0 * r, 5.0 * r))
        # +6 more reference chips (587 ns): outside it on every chain, whatever its chip count
        n = len(LOGS)
        held = c.dr_state["clk"]
        _donor(rx, 104.0, TRUE_CLK + 10.0)
        _adopt(c, 104.5)
        check(c.dr_state["clk"] == held and c.dr_state["clk_t"] == 104.5
              and _logged("cross-band clock from 'gps_l5'", n),
              "%s: +6 ref chips (%.3f of ours) REFUSED, held at zero rate (clk_t = now)"
              % (name, 6.0 * r))
        # the escape: 300 s after the source stamp of the last adoption (the donor's t, 102.0),
        # with no JOINT-CLK adoption
        for k in range(1, 149):
            _donor(rx, 104.0 + 2 * k, TRUE_CLK + 10.0)
            _adopt(c, 104.5 + 2 * k)
        check(c.dr_state["clk"] == held and c.dr_state["clk_t"] == 400.5,
              "%s: still refusing (held) 298.5 s after the last adoption's source stamp" % name)
        _donor(rx, 402.0, TRUE_CLK + 10.0)
        _adopt(c, 402.5)
        check(abs(c.dr_state["clk"] - (TRUE_CLK + 10.0) * r) < 1e-6,
              "%s: adopts once the local clock is 300 s stale (the escape stands)" % name)


class _FakeJoint(object):
    def __init__(self, clk, sigma=0.05, n=14):
        self.clk, self._sigma = clk, sigma
        self._idx = {("G", i): i for i in range(n)}

    def sigma(self, key=None):
        return self._sigma


def _jclk(c, t, joint):
    """dr_seed's JOINT-CLK consumer on this chain's clk_now, as the seeds would ride it."""
    c.t0 = c.drp.now_w = t
    c.drp.clk_now = c.dr_state["clk"]
    c.args.joint_min_sats, c.args.joint_clk_max_chips, c.args.joint_clk_max_sigma = 4, 5.0, 0.5
    c.joint_state = lambda _rx, _b, _a: joint
    dr.dr_joint_clk(c)
    return c.drp.clk_now


def test_escape_suppressed_by_joint_adoption():
    print("#104's escape: suppressed while JOINT-CLK adopts, back 300 s after it stops")
    _capture()
    for name, rate, ln, band in (("gal_e5a", L5_RATE, L5_LEN, "1176.45MHz"),
                                 ("gal_e5b", L5_RATE, L5_LEN, "1207.14MHz")):
        rx = Receiver(log=lambda m: None)
        rx.joint_declare_unit("gps_l5", L5_RATE, L5_LEN)
        c = _consumer(rx, name, rate, ln, band)
        _donor(rx, 100.0, TRUE_CLK)
        _adopt(c, 100.5)
        joint = _FakeJoint(TRUE_CLK + 0.4)
        now = _jclk(c, 100.6, joint)
        check(abs(now - (TRUE_CLK + 0.4)) < 1e-9 and c.dr_state.get("jclk_t") == 100.6,
              "%s: JOINT-CLK ADOPTED and stamped jclk_t" % name)
        # the donor steps +500 (a stepped legacy clock) and stays there
        t = 102.0
        while t < 1000.0:
            _donor(rx, t, TRUE_CLK + 500.0)
            _adopt(c, t + 0.5)
            _jclk(c, t + 0.6, joint)
            t += 2.0
        check(abs(c.dr_state["clk"] - TRUE_CLK) < 1e-9,
              "%s: 900 s of a +500 donor, joint healthy and adopted every cycle: never escaped"
              % name)
        # JOINT refused (unhealthy: sigma over its bound) -> no stamp; the escape returns 300 s on
        bad = _FakeJoint(TRUE_CLK + 0.4, sigma=0.9)
        t_stop = c.dr_state["jclk_t"]
        while t < t_stop + 298.0:
            _donor(rx, t, TRUE_CLK + 500.0)
            _adopt(c, t + 0.5)
            _jclk(c, t + 0.6, bad)
            t += 2.0
        check(abs(c.dr_state["clk"] - TRUE_CLK) < 1e-9 and c.dr_state["jclk_t"] == t_stop,
              "%s: joint unhealthy: no stamp; still refusing <300 s after the last one" % name)
        while t < t_stop + 304.0:
            _donor(rx, t, TRUE_CLK + 500.0)
            _adopt(c, t + 0.5)
            t += 2.0
        check(abs(c.dr_state["clk"] - (TRUE_CLK + 500.0)) < 1e-9,
              "%s: 300 s after the last joint adoption the escape adopts the donor" % name)


def test_samesband_refusal_holds_zero_rate():
    print("#104 same-band refusal: hold at zero rate, not the refused donor's drift")
    _capture()
    rx = Receiver(log=lambda m: None)
    c = _consumer(rx, "gal_e5a", L5_RATE, L5_LEN, "1176.45MHz")
    rx.contribute_dr_clock("gps_l5", "1176.45MHz", TRUE_CLK, -0.0129, 100.0, L5_LEN,
                           chip_rate_hz=L5_RATE)
    _adopt(c, 100.5)
    check(c.dr_state["drift"] == -0.0129 and c.dr_state["clk_src_t"] == 100.0,
          "adopted with the donor's drift and source stamp")
    rx.contribute_dr_clock("gps_l5", "1176.45MHz", TRUE_CLK + 300.0, -0.0129, 160.0, L5_LEN,
                           chip_rate_hz=L5_RATE)
    _adopt(c, 160.5)
    now = (c.dr_state["clk"] + c.dr_state["drift"] * (160.5 - c.dr_state["clk_t"])) % L5_LEN
    check(abs(now - TRUE_CLK) < 1e-12,
          "refused +300: clk_now is the held clock (no -0.0129 chips/s walk over 60 s)")


def test_declared_repin_reaches_consumers():
    print("a DECLARED snap (bootstrap / re-pin) reaches the copy paths at once; an undeclared step does not")
    _capture()
    rx = Receiver(log=lambda m: None)
    src = _l5()
    src.rx = rx
    rx.contribute_clock_mod_epoch("gps_l5", (TRUE_CLK / L5_RATE) % 0.02, 0.02, 6, 99.0)
    same = _consumer(rx, "gal_e5a", L5_RATE, L5_LEN, "1176.45MHz")
    cross = _consumer(rx, "gps_l2c", L2C_RATE, L2C_LEN, "1227.60MHz")
    # the source bootstraps 15 chips off (a bad first pass) -- epoch 1; both consumers adopt it
    bad = TRUE_CLK - 15.0
    _cycle(src, 100.0, _good(7, bad))
    src_e = src.dr_state.get("clk_epoch")
    _adopt(same, 100.5)
    _adopt(cross, 100.7)
    check(src_e == 1 and abs(same.dr_state["clk"] - src.dr_state["clk"]) < 1e-9
          and same.dr_state.get("clk_src_epoch") == 1 and cross.dr_state.get("clk_src_epoch") == 1,
          "bootstrap = clock epoch 1, adopted by both (primed) consumers")
    # the real sky: the source re-pins on the 3rd agreeing solve (cycle 4 of the run)
    n = len(LOGS)
    for k in range(3):
        _cycle(src, 102.0 + 2 * k, _good(7))
        rx.contribute_clock_mod_epoch("gps_l5", (TRUE_CLK / L5_RATE) % 0.02, 0.02, 6, 102.0 + 2 * k)
        _adopt(same, 102.5 + 2 * k)
        _adopt(cross, 102.7 + 2 * k)
    check(src.dr_state.get("clk_epoch") == 2 and abs(_wrap(src.dr_state["clk"] - TRUE_CLK)) < 1.5,
          "source RE-PIN +15 = clock epoch 2")
    check(abs(_wrap(same.dr_state["clk"] - src.dr_state["clk"])) < 1e-9
          and _logged("sibling clock from 'gps_l5' stepped", n),
          "same-band consumer adopts the declared +15 step at once (not 300 s later)")
    check(abs(_wrap(cross.dr_state["clk"] - src.dr_state["clk"] * L2C_RATE / L5_RATE,
                    L2C_LEN)) < 1e-6 and _logged("cross-band clock from 'gps_l5' stepped", n),
          "cross-band L2C consumer adopts it at once too (converted)")
    # an UNDECLARED step in the published value (epoch unchanged) is still refused
    held_s, held_x = same.dr_state["clk"], cross.dr_state["clk"]
    rx.contribute_dr_clock("gps_l5", "1176.45MHz", (src.dr_state["clk"] + 400.0) % L5_LEN, 0.0,
                           110.0, L5_LEN, chip_rate_hz=L5_RATE, epoch=2)
    _adopt(same, 110.5)
    _adopt(cross, 110.7)
    check(same.dr_state["clk"] == held_s and cross.dr_state["clk"] == held_x,
          "+400 with the same epoch: refused on both paths")


def test_held_value_is_valid_now():
    print("a HELD donor value is taken at the consumer's own now (no (l-a) extrapolation)")
    _capture()
    rx = Receiver(log=lambda m: None)
    src = _l5()
    src.rx = rx
    t = _established(src)
    rx.contribute_clock_mod_epoch("gps_l5", (TRUE_CLK / L5_RATE) % 0.02, 0.02, 6, t)
    cons = {n: _consumer(rx, n, r, ln, b) for n, r, ln, b in
            (("gal_e5a", L5_RATE, L5_LEN, "1176.45MHz"), ("gal_e6", E6_RATE, E6_LEN, "x"),
             ("gps_l2c", L2C_RATE, L2C_LEN, "y"))}
    for c in cons.values():
        _adopt(c, t + 1.5)
    # the source now holds through a transit; the consumers run 1.5 s behind its cycle and a
    # poisoned code-rate EMA would give a cross-band chain f_chip * 0.4 ppm of extrapolation
    _cycle(src, t + 2.0, [TRUE_CLK + 700.0] * 6, sky=_sky(1.0))
    check(rx.dr_clock_any_band(exclude="x").extra.get("held") is True,
          "the source's held contribution is marked held")
    ok = True
    for n, c in cons.items():
        _adopt(c, t + 3.5)
        rate = c.args.chip_rate_hz
        drift = c.dr_state.get("drift")
        drift = rate * 0.4e-6 if drift is None else drift
        now = (c.dr_state["clk"] + drift * (c.t0 - c.dr_state["clk_t"])) % c.code_len
        exp = (src.dr_state["clk"] * rate / L5_RATE) % c.code_len
        ok = ok and c.dr_state["clk_t"] == c.t0 and abs(now - exp) < 1e-6
    check(ok, "e5a / e6 / l2c: clk_t = their own now, clk_now = the held clock exactly")


if __name__ == "__main__":
    test_cold_start_unchanged()
    test_unconfirmed_bootstrap_not_frozen_but_repinned()
    test_two_sat_corrupted_step_rejected()
    test_genuine_repin_after_confirmation()
    test_transit_freeze()
    test_confirmed_clock_never_rerolled()
    test_knobs_off_is_pre_142()
    test_crossband_bound_is_a_time()
    test_escape_suppressed_by_joint_adoption()
    test_samesband_refusal_holds_zero_rate()
    test_declared_repin_reaches_consumers()
    test_held_value_is_valid_now()
    if _fails:
        print("FAILED: %d" % len(_fails))
        for f in _fails:
            print("  - " + f)
        sys.exit(1)
    print("OK")
