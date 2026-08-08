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

print("\n%d/%d checks passed" % (0 if FAIL else 1, 1) if False else
      ("FAILED: %s" % ", ".join(FAIL)) if FAIL else "\nALL CHECKS PASSED")
sys.exit(1 if FAIL else 0)
