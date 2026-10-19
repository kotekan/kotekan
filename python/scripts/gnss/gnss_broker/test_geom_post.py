"""The satellite-geometry feed (--post-sat-geometry) posts only azimuths that exist.

    python3 -m gnss_broker.test_geom_post

`stage_almanac_predict` posts each satellite's [az, el] to the record assemblers'
/set_sat_geometry for the element steering. BRDC rows carry the azimuth (element 5); TLE rows
(gps_beamtrack.predict_dopplers, the --almanac-source tle path and the BRDC-unavailable
fallback) do not, and the prediction pads element 5 with 0.0. This drives the stage on both
paths with the predictions stubbed and `_post` captured, and checks that a BRDC sky is posted
with its azimuths and a TLE sky is not posted at all, rather than every satellite at due north.

@author Keith Vanderlinde
"""

import sys
import types

from gnss_broker import almanac


_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


class _Obj(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)


def run(raw, brdc):
    """One stage call whose prediction returns `raw` (BRDC path when brdc, else the TLE path);
    returns the list of (url, body) posts."""
    posts = []
    ctx = _Obj(
        args=_Obj(
            almanac=True,
            doppler_sign=1,
            lat=49.3,
            lon=-119.6,
            alt=545.0,
            carrier_hz=1176.45e6,
            mask_deg=10.0,
            post_sat_geometry=1,
            bias_det_fresh_s=30.0,
            bias_min_snr=10.0,
            bias_min_sats=2,
        ),
        capable_t=None,
        capable=None,
        brdc_alm=object() if brdc else None,
        almanac_sats=[],
        alm_sys="G",
        alm_min_prn=1,
        alm_now=lambda: None,
        dll_combiners=["http://cx19:12048/n2combine"],
        geom_post_t=[0.0],
        transit_pd=None,
        publisher=None,
        cb=_Obj(ema=None, value=0.0),
        best={},
        det_fresh={},
        t0=1000.0,
    )
    saved = (almanac.brdc_predict, almanac._post, almanac._log_rl, almanac.timebase)
    saved_bt = sys.modules.get("gps_beamtrack")
    almanac.brdc_predict = lambda *a: raw
    almanac._post = lambda url, body, timeout=5.0: posts.append((url, body))
    almanac._log_rl = lambda key, msg, every_s=10.0: None
    almanac.timebase = _Obj(observe=lambda c, log_rl: None)
    sys.modules["gps_beamtrack"] = types.SimpleNamespace(
        predict_dopplers=lambda *a, **kw: raw
    )
    try:
        almanac.stage_almanac_predict(ctx)
    finally:
        almanac.brdc_predict, almanac._post, almanac._log_rl, almanac.timebase = saved
        if saved_bt is None:
            del sys.modules["gps_beamtrack"]
        else:
            sys.modules["gps_beamtrack"] = saved_bt
    check(5 in ctx.pred, "the stubbed prediction ran (PRN 5 predicted)")
    return posts


def main():
    print("\nthe satellite-geometry feed")

    # 1. BRDC rows (9 elements): PRN 5 posted with its azimuth, rates and epoch.
    posts = run({5: (-1200.0, 0.4, 45.0, 2.1e7, 1e-5, 120.0, 0.01, -0.02, 1.7e9)}, True)
    body = posts[0][1] if posts else {}
    check(
        len(posts) == 1
        and posts[0][0] == "http://cx19:12048/n2assemble/set_sat_geometry",
        "BRDC sky: one post to the assembler",
    )
    check(
        body.get("5") == [120.0, 45.0, 0.01, -0.02, 1.7e9],
        "BRDC sky: PRN 5 posted [az, el, az_rate, el_rate, t] (got %r)" % body.get("5"),
    )

    # 2. TLE rows (4 elements, no azimuth): nothing posted.
    posts = run({5: (-1200.0, 0.4, 45.0, 2.1e7)}, False)
    check(
        not posts, "TLE sky: no geometry posted (got %r)" % [b for u, b in posts],
    )

    print("-" * 70)
    if _fails:
        for f in _fails:
            print("FAIL: %s" % f)
        return 1
    print("GATE GOOD: the satellite-geometry feed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
