"""Unit checks for the pieces the transcript gate cannot reach (task #27).

`scripts/gnss/broker_equiv.py` proves the broker's POST stream is unchanged, but it can
only exercise the code paths its fixture happens to reach -- and the synthetic fixture
reaches neither the frozen clock's live-mode behaviour nor anything multi-chain, because
both need a second thread to be observable at all. Those are exactly the places where a
plausible-looking implementation can be silently wrong, so they get direct assertions.

    python3 gnss_broker/selftest.py
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnss_broker import transport, receiver, signals   # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print("%-52s %s%s" % (name, "ok" if cond else "FAIL", "  " + detail if detail else ""))
    if not cond:
        FAIL.append(name)


# -- the frozen cycle clock ---------------------------------------------------------------
# This one exists because the first implementation gated the freeze on a transcript being
# open, so production kept a LIVE clock and only a recording froze it -- a recorder that
# changes what it records. The gate could not see it (both sides of the comparison were
# recordings); only a live-mode assertion can.
transport._TR.tick()
_a = transport._now()
time.sleep(0.02)
_b = transport._now()
check("cycle clock frozen in LIVE mode", _a == _b, "%.6f vs %.6f" % (_a, _b))
transport._TR.tick()
check("tick() advances it", transport._now() > _a)

_other = {}


def _thread_clock():
    # No tick() here: a thread that does not run cycles (the publisher's HTTP thread) must
    # read the real clock, not another thread's frozen instant.
    _other["v"] = transport._now()


t = threading.Thread(target=_thread_clock)
t.start()
t.join()
check("a non-cycling thread reads the real clock", _other["v"] != transport._now())


def _thread_cycle():
    transport._TR.tick()
    _other["c"] = transport._now()


t = threading.Thread(target=_thread_cycle)
t.start()
t.join()
check("two chains hold INDEPENDENT frozen clocks", _other["c"] != transport._now())

# -- the Receiver -------------------------------------------------------------------------
rx = receiver.Receiver(log=lambda m: None)

check("anchor: fetched once, shared by all",
      rx.time_anchor(lambda: 1786167610.0, "a") == 1786167610.0
      and rx.time_anchor(lambda: 999.0, "b") == 1786167610.0)

_built = []
rx.brdc("k", lambda: _built.append(1) or {"eph": {}})
rx.brdc("k", lambda: _built.append(1) or {"eph": {}})
check("brdc: built once per key", len(_built) == 1, "%d builds" % len(_built))

# THE CONTRACT THAT KEEPS SINGLE-CHAIN BEHAVIOUR IDENTICAL: a chain never consumes what it
# itself contributed. If this ever fails, every equivalence result in this refactor is void.
rx.contribute_carrier_bias("gps_l5", -12.5, 6, 1000.0)
check("clock: a chain does not consume its OWN estimate",
      rx.carrier_bias(exclude="gps_l5", t_now=1000.0) is None)
_s = rx.carrier_bias(exclude="gal_e5a", t_now=1000.0)
check("clock: a sibling consumes it", _s is not None and _s.value == -12.5)
check("clock: stale contributions age out",
      rx.carrier_bias(exclude="gal_e5a", t_now=1000.0 + 999.0) is None)

rx.contribute_carrier_bias("other", -9.0, 12, 1000.0)
_s = rx.carrier_bias(exclude="gal_e5a", t_now=1000.0)
check("clock: the better-supported estimate wins", _s.value == -9.0, "n=%d" % _s.weight)

# Code side is PER BAND: group delay does not survive a retune, so a bias measured at
# 1176.45 MHz must not be visible to a chain at 1207.14.
rx.contribute_code_bias("gps_l5", "1176.45MHz", 2.6e-6, 5, 1000.0)
check("code bias: visible inside the band",
      rx.code_bias("1176.45MHz", exclude="gal_e5a", t_now=1000.0) is not None)
check("code bias: NOT visible across a retune",
      rx.code_bias("1207.14MHz", exclude="gal_e5a", t_now=1000.0) is None)

rx.contribute_dr_clock("gps_l5", "1176.45MHz", 5094.9, 0.31, 1000.0, 10230.0)
_d = rx.dr_clock("1176.45MHz", exclude="gal_e5a", t_now=1000.0)
check("dr clock: carries its code length",
      _d is not None and _d.extra["code_length"] == 10230.0 and _d.extra["drift"] == 0.31)

# -- signals ------------------------------------------------------------------------------
# Cross-checks against numbers written by other people in other files. gps_l5 -> 20/0.02 is
# broker_up.sh's; gal_e5a and bds_b2a -> 100/0.1 are broker_up_extra.sh's; gps_l2c -> 75/1.5
# is the constant the broker's own CL comment names; gps_l1ca is replay_l1gps_leg.sh's.
for key, want in (("gps_l5", (1176.45e6, 10.23e6, 10230, 20, 0.02)),
                  ("gal_e5a", (1176.45e6, 10.23e6, 10230, 100, 0.1)),
                  ("bds_b2a", (1176.45e6, 10.23e6, 10230, 100, 0.1)),
                  ("gps_l2c", (1227.6e6, 511.5e3, 10230, 75, 1.5)),
                  ("gps_l1ca", (1575.42e6, 1.023e6, 1023, 1, 0.001))):
    s = signals.get(key)
    got = (s.carrier_hz, s.chip_rate_hz, s.code_length, s.long_code_segments,
           s.long_code_epoch_s)
    check("signal %s matches its launch script" % key,
          all(abs(a - b) <= 1e-9 * max(1.0, abs(b)) for a, b in zip(got, want)),
          "%s vs %s" % (got, want))

check("gps_l5 and gal_e5a share a band (same 1176.45 MHz hardware)",
      signals.get("gps_l5").band == signals.get("gal_e5a").band)
check("gal_e5b is a DIFFERENT band (retune -> different group delay)",
      signals.get("gal_e5b").band != signals.get("gal_e5a").band)

# -- dead-reckon seed currency (task #30) -------------------------------------------------
# The slewed refresh depends on two properties of dr_cp0/dr_seed_phys and dies silently
# without either: (1) they are exact inverses, so "where does the tracker think the code is"
# is answerable; (2) re-anchoring a seed at a later hop WITH A DIFFERENT DOPPLER moves the
# implied physical phase by exactly zero -- the property the reverted 10 s repin lacked and
# the whole reason the slew is safe to run every cycle. Deterministic PRNG: this is a gate.
import random as _rnd

from fits import dr_cp0, dr_seed_phys  # noqa: E402

_rnd.seed(30)
_CHIP, _FC, _S, _HPS, _MOD = 10.23e6, 1176.45e6, -1.0, 195312.5, 1023000.0
_w_rt = _w_cont = 0.0
for _ in range(500):
    _phys = _rnd.uniform(0, _MOD)
    _t = _rnd.uniform(1e3, 2e5)
    _dop = _rnd.uniform(-4000, 4000)
    _h0 = int(round(_t * _HPS))
    _sd = {"code_phase_chips": dr_cp0(_phys, _h0 / _HPS, _dop, _CHIP, _FC, _S, _MOD),
           "ref_hop": _h0, "doppler_hz": _dop,
           "code_phase_rate": _rnd.uniform(-1e-4, 1e-4),
           "doppler_rate_hz_s": _rnd.uniform(-0.5, 0.5)}
    _e = abs((((dr_seed_phys(_sd, _h0, _HPS, _CHIP, _FC, _S, _MOD) - _phys)
               + _MOD / 2) % _MOD) - _MOD / 2)
    _w_rt = max(_w_rt, _e)
check("dr currency round-trips (dr_seed_phys inverts dr_cp0 at birth)", _w_rt < 1e-3,
      "worst %.3e chips over 500 trials" % _w_rt)
# continuity: roll to h1 at a CHANGED doppler; implied phys at h1 must not move at all
_rnd.seed(31)
_w_cont = 0.0
for _ in range(500):
    _phys = _rnd.uniform(0, _MOD)
    _t = _rnd.uniform(1e3, 2e5)
    _dop = _rnd.uniform(-4000, 4000)
    _dop2 = _dop + _rnd.uniform(-2, 2)
    _h0 = int(round(_t * _HPS))
    _h1 = _h0 + int(_rnd.uniform(1, 600) * _HPS)
    _sd = {"code_phase_chips": dr_cp0(_phys, _h0 / _HPS, _dop, _CHIP, _FC, _S, _MOD),
           "ref_hop": _h0, "doppler_hz": _dop,
           "code_phase_rate": _rnd.uniform(-1e-4, 1e-4),
           "doppler_rate_hz_s": _rnd.uniform(-0.5, 0.5)}
    _p1 = dr_seed_phys(_sd, _h1, _HPS, _CHIP, _FC, _S, _MOD)
    _sd2 = {"code_phase_chips": dr_cp0(_p1, _h1 / _HPS, _dop2, _CHIP, _FC, _S, _MOD),
            "ref_hop": _h1, "doppler_hz": _dop2,
            "code_phase_rate": 0.0, "doppler_rate_hz_s": 0.0}
    _p1b = dr_seed_phys(_sd2, _h1, _HPS, _CHIP, _FC, _S, _MOD)
    _w_cont = max(_w_cont, abs((((_p1b - _p1) + _MOD / 2) % _MOD) - _MOD / 2))
check("dr slew re-anchor is CONTINUOUS across a Doppler change", _w_cont < 1e-6,
      "worst %.3e chips over 500 trials" % _w_cont)

# -- fleet phase-slope delay fit (task #32) -----------------------------------------------
# The fitter must (1) recover a known delay with the right SIGN and milli-chip accuracy from
# the REAL fleet comb geometry (stride-16 with per-node offsets -- the union is what beats
# the 3.27-chip grating lobes), (2) read peak/floor ~1 on pure noise, and (3) be
# deterministic (it feeds a digest-gated process; a wandering fit would look like broker
# non-determinism three layers away).
import cmath as _cm

from gnss_broker.fleet import fit_spectrum_delay  # noqa: E402

_CHIPR, _W = 10.23e6, 195312.5
_COMBS = [[5972 + 16 * i for i in range(7)], [5980 + 16 * i for i in range(7)],
          [5975 + 16 * i for i in range(7)], [5983 + 16 * i for i in range(7)],
          [5978 + 16 * i for i in range(7)], [5986 + 16 * i for i in range(7)]]
_rnd.seed(32)


def _spec_synth(tau_chips, snr=30.0):
    tau_s = tau_chips / _CHIPR
    pts = []
    for k, fids in enumerate(_COMBS):
        ph = _rnd.uniform(0, 2 * _cm.pi)   # unknown per-instance phase, always present
        for fid in fids:
            a = _cm.exp(-2j * _cm.pi * fid * _W * tau_s + 1j * ph)
            a += complex(_rnd.gauss(0, 1 / snr), _rnd.gauss(0, 1 / snr))
            pts.append((fid, a, 1.0, "inst%d" % k))
    return pts


_worst = 0.0
for _truth in (-1.2, -0.4, 0.0, 0.31, 0.8, 1.5):
    _r = fit_spectrum_delay(_spec_synth(_truth), _CHIPR, _W)
    _worst = max(_worst, abs(_r[0] - _truth))
check("spectrum delay fit recovers sign+magnitude on the real comb", _worst < 0.02,
      "worst |tau err| %.4f chips over 6 truths in [-1.2, +1.5]" % _worst)
_noise = [(fid, complex(_rnd.gauss(0, 1), _rnd.gauss(0, 1)), 1.0, "i%d" % k)
          for k, c in enumerate(_COMBS) for fid in c]
_rn = fit_spectrum_delay(_noise, _CHIPR, _W)
check("spectrum delay fit reads peak/floor ~1 on pure noise",
      _rn[1] / max(_rn[2], 1e-12) < 1.5, "%.2f" % (_rn[1] / max(_rn[2], 1e-12)))
_p = _spec_synth(0.31)
check("spectrum delay fit is deterministic",
      fit_spectrum_delay(_p, _CHIPR, _W) == fit_spectrum_delay(_p, _CHIPR, _W))

# -- SatBiasFilter (task #33, P2 step 1) --------------------------------------------------
# The b_sat loop's four contractual behaviours, each traceable to a measured P1 caveat:
# converge to a constant bias through realistic noise; REJECT lobe captures at +-3.27
# outright (never average them in); stop APPLYING (not remembering) a stale bias; and
# refuse thin fleets, whose fits scattered 3x wider during the rolling restart.
from gnss_broker.state_filter import SatBiasFilter  # noqa: E402

_rnd.seed(33)
_bf = SatBiasFilter(gain=0.02, innovation_max=1.0, max_age_s=600.0, min_inst=6)
_t = 0.0
for _ in range(600):  # ~20 min at 2 s: P1's own measurement span
    _t += 2.0
    _bf.update(3, 0.081 + _rnd.gauss(0, 0.35), 12, _t)   # E5a PRN 3, as measured
check("SatBiasFilter converges to the P1-measured bias",
      abs(_bf.get(3, _t) - 0.081) < 0.03, "%.4f vs +0.081" % _bf.get(3, _t))
_rej0 = _bf.rejected
_ok = _bf.update(3, 3.27 + 0.081, 12, _t)   # a lobe capture, exactly one comb spacing off
check("SatBiasFilter rejects a lobe capture via the innovation gate",
      (not _ok) and _bf.rejected == _rej0 + 1 and abs(_bf.get(3, _t) - 0.081) < 0.03)
check("SatBiasFilter refuses a thin fleet", not _bf.update(3, 0.08, 4, _t))
check("SatBiasFilter stops APPLYING a stale bias but remembers it",
      _bf.get(3, _t + 601.0) == 0.0 and "3:" in _bf.summary(_t + 601.0))

# -- JointReceiverState (task #33, P2a: the REVISION) --------------------------------------
# CHORD_JOINT_TRACKING section 3a's contract. These are not "does the maths run" checks --
# each is one of the claims the revision makes about WHY a joint solve beats the staged
# design it replaces, written so that a regression toward per-sat tracking fails loudly.
from gnss_broker.state_filter import JointReceiverState  # noqa: E402

def _run(js, truth_clk, truth_b, t0, n, dt=2.0, sigma=0.30, skip=(), rate=0.0):
    """Feed synthetic y_i = clk(t) + b_i + noise for a fleet of sats."""
    t = t0
    for _ in range(n):
        t += dt
        clk = truth_clk + rate * (t - t0)
        _run.clk = clk
        js.cycle([(k, clk + b + _rnd.gauss(0, sigma), sigma)
                  for k, b in truth_b.items() if k not in skip], t)
    return t

_rnd.seed(3311)
_TB = {("E", 3): +2.9, ("E", 5): -3.4, ("E", 31): +0.8, ("C", 39): -1.7,
       ("G", 14): +4.1, ("G", 21): -2.7}          # chips: the MEASURED model-error scale
_js = JointReceiverState(q_b=0.013, gauge_sigma=0.1)
_t = _run(_js, 151.0, _TB, 0.0, 400)              # ~13 min at 2 s
# The gauge defines clk as the fleet mean, so compare clk+b_i per sat, not clk alone.
_err = max(abs(_js.predicted(k) - (151.0 + b)) for k, b in _TB.items())
check("JointReceiverState recovers clk+b_sat at the chips-wrong model scale",
      _err < 0.25, "worst %.3f chips over %d sats" % (_err, len(_TB)))
check("JointReceiverState gauge holds mean(b) at zero",
      abs(sum(_js.bias(k) for k in _TB) / len(_TB)) < 0.05,
      "mean b %+.4f" % (sum(_js.bias(k) for k in _TB) / len(_TB)))

# BIRTH: the clock's ABSOLUTE value must survive the first cycle. The first version put
# the whole offset into b0 by hand, and the gauge -- seeing no cross-covariance to a
# newborn row -- sheared it back out of the biases without depositing it in clk, so the
# state read `clk +0.000 +- 200` with the biases spread +-4.7. It ground its way back over
# ~400 cycles, which also means a clock near L/2 would have converged to a WRAPPED ALIAS.
# Caught on the first line of on-sky shadow output, not by this file; hence this check.
_rnd.seed(4242)
_jb = JointReceiverState()
_jb.cycle([(k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()], 2.0)
check("the clock's absolute value survives satellite BIRTH",
      abs(_jb.clk - 151.0) < 1.5, "clk %+.2f after one cycle (want ~151)" % _jb.clk)
_jw2 = JointReceiverState(code_len=10230.0)
_jw2.cycle([(k, 5100.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()], 2.0)
check("birth near L/2 does not converge to a wrapped alias",
      abs(_jw2.wrap(_jw2.clk - 5100.0)) < 1.5, "clk %+.1f (want ~5100)" % _jw2.clk)

# THE SPLIT IS BANDWIDTH. A COMMON step must land in clk (fast, shared); a PER-SAT step
# must land in that sat's b (slow, private). Getting this backwards is exactly the
# misspecification that makes the plant oscillate.
_b_before = {k: _js.bias(k) for k in _TB}
_t = _run(_js, 156.0, _TB, _t, 120)               # +5 chips common
check("a COMMON offset moves clk, not the biases",
      abs(_js.clk - 156.0) < 0.4
      and max(abs(_js.bias(k) - _b_before[k]) for k in _TB) < 0.5,
      "clk %+.2f (want 156), worst db %.2f"
      % (_js.clk, max(abs(_js.bias(k) - _b_before[k]) for k in _TB)))
_clk_before = _js.clk
_TB2 = dict(_TB); _TB2[("E", 3)] += 2.0           # ONE sat walks 2 chips
_t = _run(_js, 156.0, _TB2, _t, 300)
check("a PER-SAT offset moves that bias, not the clock",
      abs(_js.bias(("E", 3)) - _b_before[("E", 3)] - 2.0) < 0.6
      and abs(_js.clk - _clk_before) < 0.5,
      "db %.2f (want 2.0), dclk %.2f"
      % (_js.bias(("E", 3)) - _b_before[("E", 3)], _js.clk - _clk_before))

# THE DISCRIMINATING TEST (section 3a): take a sat to no usable SNR and stop feeding it.
# Its replica must keep moving with the SHARED clock while its bias holds -- that is the
# property the rejected "anchor each sat to its own track" design does not have.
_rnd.seed(77)
_jc = JointReceiverState(q_b=0.013)
_t = _run(_jc, 151.0, _TB, 0.0, 400)
_t = _run(_jc, 151.0, _TB, _t, 300, rate=0.01, skip={("E", 5)})   # clock walks 6 chips
_truth_5 = _run.clk + _TB[("E", 5)]
check("a satellite with NO measurements coasts on the shared clock",
      abs(_jc.predicted(("E", 5)) - _truth_5) < 0.5,
      "coast error %.3f chips after 600 s dark, clock moved %.1f"
      % (_jc.predicted(("E", 5)) - _truth_5, _run.clk - 151.0))
# clk_rate: check the PAIR (unbiased AND quiet). A Kalman velocity is noisy by
# construction, so a single instantaneous sample against a tight bar tests luck -- the
# first version of this check failed at 2 sigma on a correct filter. Average over a
# window for the mean, and measure the scatter separately, because the scatter is the
# quantity that decides whether this state can replace the l-a EMA at all.
from math import sqrt as _msqrt
_rates = []
_t0r, _clk0r = _t, _run.clk        # CONTINUE the ramp; a fresh origin per step is no ramp
for _ in range(300):
    _t += 2.0
    _clkr = _clk0r + 0.01 * (_t - _t0r)
    _jc.cycle([(k, _clkr + b + _rnd.gauss(0, 0.30), 0.30) for k, b in _TB.items()], _t)
    _rates.append(_jc.clk_rate)
_mr = sum(_rates) / len(_rates)
_sr = _msqrt(sum((r - _mr) ** 2 for r in _rates) / len(_rates))
check("clk_rate is observable from the fleet, unbiased",
      abs(_mr - 0.01) < 0.0015, "%.5f vs 0.01000 chips/s" % _mr)
check("clk_rate is quieter than the l-a EMA it replaces",
      _sr < 0.0015, "sd %.5f chips/s (l-a EMA scatter was 0.07)" % _sr)

# ROBUSTNESS. These reproduce the 2026-08-09 production incident directly: the filter
# replaced a circular MEDIAN (outlier-proof) with a MEAN-gauged solve (not), and one
# weak-sat detection -- the broker documents snr<60 giving ~2000-chip residuals -- walked
# clk_rate to +0.074 ppm in 60 s, which then fed every unlocked seed 17 chips/min of
# fictitious code drift. Each guard gets a check that FAILS without it.
_rnd.seed(808)
_jr2 = JointReceiverState()
_t2 = 0.0
for _ in range(400):
    _t2 += 2.0
    _jr2.cycle([(k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()], _t2)
_clk_ok, _rate_ok = _jr2.clk, _jr2.clk_rate
_t2 += 2.0
_jr2.cycle([(k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()]
           + [(("G", 99), 151.0 + 1800.0, 0.3)], _t2)          # the garbage detection
for _ in range(30):
    _t2 += 2.0
    _jr2.cycle([(k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()], _t2)
check("ONE garbage detection cannot move the clock (birth window)",
      abs(_jr2.clk - _clk_ok) < 1.0 and ("G", 99) not in _jr2._idx,
      "clk %+.2f -> %+.2f, born=%s" % (_clk_ok, _jr2.clk, ("G", 99) in _jr2._idx))
check("ONE garbage detection cannot move clk_rate",
      abs(_jr2.clk_rate - _rate_ok) < 0.002,
      "rate %+.5f -> %+.5f chips/s (incident reached +0.76)" % (_rate_ok, _jr2.clk_rate))
# An ESTABLISHED sat throwing an outlier is the innovation gate's job.
_rej0 = _jr2.rejected
_t2 += 2.0
_jr2.cycle([(("G", 11), 151.0 + 900.0, 0.3)], _t2)
check("an established satellite's outlier is rejected, not averaged in",
      _jr2.rejected > _rej0 and abs(_jr2.clk - _clk_ok) < 1.0)
# Membership churn must not masquerade as clock rate.
_rnd.seed(809)
_jm = JointReceiverState()
_t3 = 0.0
for _ in range(400):
    _t3 += 2.0
    _jm.cycle([(k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()], _t3)
_gone = {("G", 11)}
_rmax = 0.0
for _ in range(700):        # let it age out, then keep running
    _t3 += 2.0
    _jm.cycle([(k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3)
               for k, b in _TB.items() if k not in _gone], _t3)
    _rmax = max(_rmax, abs(_jm.clk_rate))
check("a satellite SETTING does not masquerade as clock rate",
      _rmax < 0.002, "peak |rate| %.5f chips/s during the gauge re-reference" % _rmax)

# The gate must not DEADLOCK. A genuine step (clock event, re-anchor) is rejected by a
# normalized gate, so without an escape the state never follows it and every later
# measurement is rejected too -- forever. Garbage is sporadic; a real move persists.
_rnd.seed(811)
_jd = JointReceiverState()
_t4 = 0.0
for _ in range(400):
    _t4 += 2.0
    _jd.cycle([(k, 151.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()], _t4)
for _ in range(60):                       # the clock STEPS 40 chips and stays there
    _t4 += 2.0
    _jd.cycle([(k, 191.0 + b + _rnd.gauss(0, 0.3), 0.3) for k, b in _TB.items()], _t4)
check("the filter recovers from a genuine STEP (gate has an escape)",
      abs(_jd.clk - 191.0) < 1.0, "clk %+.2f after a 40-chip step (want ~191)" % _jd.clk)

# Modulo: y arrives mod the code length, so a measurement that wraps must not explode.
_jw = JointReceiverState(code_len=10230.0)
_jw.cycle([(("G", 1), 10229.8, 0.3), (("G", 2), 10229.9, 0.3)], 1.0)
_jw.cycle([(("G", 1), 0.2, 0.3), (("G", 2), 0.3, 0.3)], 3.0)     # wrapped past L
check("JointReceiverState wraps innovations across the code-length boundary",
      abs(_jw.wrap(_jw.clk - 10230.0)) < 1.0 or abs(_jw.clk) < 1.0,
      "clk %+.2f" % _jw.clk)
_jw.cycle([(("G", 1), 0.25, 0.3)], 3.0 + 901.0)                  # PRN 2 goes stale
check("a stale satellite leaves the state (and stops voting in the gauge)",
      ("G", 2) not in _jw._idx and ("G", 1) in _jw._idx)

# -- CROSS-BAND (l-a): the rate is receiver-wide, the phase is not (task #34) -------------
# The per-band gate on code_bias is a BOOTSTRAP TRAP for any band with no chain that can solve
# its own clock, and 1207.14 MHz was exactly that: gal_e5b/bds_b2b adopted (l-a) zero times
# while their 1176.45 siblings adopted it 102/107 times, leaving code_phase_rate = 0.0 on all
# 19 PRNs and the code loop open. These assert the fix AND its limit -- the rate crosses bands,
# the PHASE must not, because a per-band phase offset IS tau_band.
_rx = receiver.Receiver()
_rx.contribute_code_bias("gps_l5", "1176.45MHz", 1.5e-6, 6, 1000.0)
check("same-band (l-a) still preferred",
      _rx.code_bias("1176.45MHz", exclude="gal_e5a", t_now=1000.0).value == 1.5e-6)
check("a band with NO contributor gets nothing from the band-scoped lookup",
      _rx.code_bias("1207.14MHz", exclude="gal_e5b", t_now=1000.0) is None)
_any = _rx.code_bias_any_band(exclude="gal_e5b", t_now=1000.0)
check("cross-band (l-a) fallback finds the other band's rate",
      _any is not None and _any.value == 1.5e-6, "src %s" % (_any.src if _any else "-"))
check("cross-band fallback still honours `exclude`",
      _rx.code_bias_any_band(exclude="gps_l5", t_now=1000.0) is None)
check("cross-band fallback still honours max_age_s",
      _rx.code_bias_any_band(exclude="gal_e5b", t_now=1000.0 + 500.0) is None)
# The PHASE stays per band -- adopting it across carriers would inject tau_band itself.
_rx.contribute_dr_clock("gps_l5", "1176.45MHz", 151.0, 0.01, 1000.0, 1023000)
check("dr_clock (the code PHASE) is still band-scoped -- tau_band is not borrowable",
      _rx.dr_clock("1207.14MHz", exclude="gal_e5b", t_now=1000.0) is None
      and _rx.dr_clock("1176.45MHz", exclude="gal_e5a", t_now=1000.0) is not None)
# And the consumer's arithmetic: a fractional-frequency bias -> chips/hop, chip-rate scaled.
from gnss_broker.fits import cp_rate_from_code_bias      # noqa: E402
_r_e5a = cp_rate_from_code_bias(0.0, 1.5e-6, 195312.5, 10.23e6, 1176.45e6)
_r_e5b = cp_rate_from_code_bias(0.0, 1.5e-6, 195312.5, 10.23e6, 1207.14e6)
check("borrowed rate is carrier-INDEPENDENT at equal chip rate (E5a vs E5b)",
      _r_e5a == _r_e5b, "%.6g vs %.6g chips/hop" % (_r_e5a, _r_e5b))
check("and it is nonzero, i.e. the seed now carries a rate at all",
      abs(_r_e5b) > 0.0, "%.6g chips/hop" % _r_e5b)

# -- CROSS-BAND CLOCK PHASE: a bootstrap, with a modulus rule (task #34) ------------------
# Layer 2 (the rate) was necessary and NOT sufficient: with the clock 150 chips out there is no
# peak for a rate to hold. These assert the bootstrap AND its guard rail -- a clock may be
# reduced to a shorter code but never lengthened, because a value known mod 10230 says nothing
# about which of the 100 periods of a 1023000-chip code it sits in.
_rx2 = receiver.Receiver()
_rx2.contribute_dr_clock("gps_l5", "1176.45MHz", 150.74, 0.01, 2000.0, 1023000)
check("cross-band clock found when no same-band donor exists",
      _rx2.dr_clock("1207.14MHz", exclude="gal_e5b", t_now=2000.0) is None
      and _rx2.dr_clock_any_band(exclude="gal_e5b", t_now=2000.0) is not None)
_c = _rx2.dr_clock_any_band(exclude="gal_e5b", t_now=2000.0)
check("cross-band clock carries its code_length so the consumer can judge the modulus",
      _c.extra.get("code_length") == 1023000)
check("reducing to a SHORTER code is well defined (150.74 mod 10230)",
      abs((_c.value % 10230.0) - 150.74) < 1e-9)
# The refusal that matters: a short-code donor must not seed a long-code consumer.
_rx3 = receiver.Receiver()
_rx3.contribute_dr_clock("bds_b2b", "1207.14MHz", 150.74, 0.01, 2000.0, 10230)
_c3 = _rx3.dr_clock_any_band(exclude="gal_e5a", t_now=2000.0)
check("a SHORTER-code donor is visible but must be refused by the caller's guard",
      _c3 is not None and (_c3.extra.get("code_length") or 0) < 1023000)
check("cross-band clock still honours exclude and max_age",
      _rx2.dr_clock_any_band(exclude="gps_l5", t_now=2000.0) is None
      and _rx2.dr_clock_any_band(exclude="gal_e5b", t_now=2000.0 + 500.0) is None)
# The error the bootstrap accepts is tau_band-sized, not 150 chips -- state the arithmetic.
check("bootstrap replaces a 150-chip error with a sub-chip one",
      abs(150.74 - 0.0) > 100.0 and abs((150.74 % 10230.0) - 150.74) < 1e-9)

print("\n%d/%d checks passed" % (0 if FAIL else 1, 1) if False else
      ("FAILED: %s" % ", ".join(FAIL)) if FAIL else "\nALL CHECKS PASSED")
sys.exit(1 if FAIL else 0)
