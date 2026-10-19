"""The nav-bits stage's 60 s health lines: each decoder reports on its own cadence.

    python3 -m gnss_broker.test_navbits

`stage_nav_bits` logs each decoder's per-PRN health every 60 s, cross-checks a decoded
ephemeris against BRDC and records both for the decode-health export (ctx.dh_obs). An L5
broker can run the LNAV decoder on its own pilot's nav_obs and a CNAV decoder on an aux chain
(--cnav-combiner) at once. This drives the stage with fake LNAV and CNAV decoders and checks
that one call logs and records both health lines.

@author Keith Vanderlinde
"""

import sys

from gnss_broker import navbits
from gnss_broker.loopstate import NavDecoders


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


class _Obj(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeDecoder(object):
    """A decoder holding one PRN that has not synced: health() only, no ephemeris."""

    def __init__(self, prn, health):
        self._p = {prn: None}
        self._h = health

    def health(self, prn):
        return self._h

    def ephemeris(self, prn):
        return None


def main():
    print("\nthe nav-bits health lines")

    nav = NavDecoders()
    nav.navbits = _FakeDecoder(3, {"synced": False, "run": 1, "need": 3, "hist": 5})
    nav.cnav = _FakeDecoder(5, {"eph": False, "synced": False, "emits": 2, "g2": None})
    nav.cnav_sig = "GPS_L5_CNAV"
    obs = []
    ctx = _Obj(
        args=_Obj(nav_bits=True, nav_decoder="lnav"),
        status={},
        nav=nav,
        brdc_alm=None,
        alm_sys="G",
        dh_obs=lambda sig, prn, h, e, eph_s: obs.append((sig, prn)),
    )
    lines = []
    saved = (navbits._log, navbits._now)
    navbits._log = lambda msg: lines.append(msg)
    navbits._now = lambda: 1000.0
    try:
        navbits.stage_nav_bits(ctx)
    finally:
        navbits._log, navbits._now = saved

    check(
        any(m.startswith("navbit PRN 3") for m in lines),
        "the LNAV health line is logged",
    )
    check(
        any(m.startswith("cnav PRN 5") for m in lines),
        "the CNAV health line is logged in the same cycle (lines %r)" % lines,
    )
    check(
        sorted(obs) == [("GPS_L1_LNAV", 3), ("GPS_L5_CNAV", 5)],
        "both decoders reach the decode-health export (got %r)" % obs,
    )

    print("-" * 70)
    if _fails:
        for f in _fails:
            print("FAIL: %s" % f)
        return 1
    print("GATE GOOD: the nav-bits health lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
