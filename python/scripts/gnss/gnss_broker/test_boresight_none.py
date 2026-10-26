"""The boresight: read from the nodes' running config (telescope.dish_coelev_deg), installed
once per process, kept through a failed re-read, and off under --boresight none, where
nearest_boresight reports nothing so the D3 transit veto, the transit freezes and holds and the
assemblers' "_bore" post all stay off.

    python3 -m gnss_broker.test_boresight_none

@author Keith Vanderlinde
"""

from gnss_broker import sky
from gnss_broker.cli import build_parser

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def _getter(table):
    """A /config GET over {base: coelev or Exception}."""

    def get(url):
        v = table[url[: -len("/config")]]
        if isinstance(v, Exception):
            raise v
        return {"telescope": {"dish_coelev_deg": v}} if v is not None else {}

    return get


def main():
    saved = (
        sky.BORESIGHT_AZ_DEG,
        sky.BORESIGHT_EL_DEG,
        sky.BORESIGHT_KNOWN,
        sky.BORESIGHT_SOURCE,
        sky._bore_inited,
    )
    try:
        # geometry: the tilt from zenith along the meridian, negative = south
        az, el = sky.boresight_from_coelev(-27.3)
        check(
            (az, round(el, 2)) == (180.0, 62.7),
            "coelev -27.3 -> az 180 el 62.70 (Tau A)",
        )
        az, el = sky.boresight_from_coelev(-8.59)
        check((az, round(el, 2)) == (180.0, 81.41), "coelev -8.59 -> az 180 el 81.41")
        check(sky.boresight_from_coelev(5.0) == (0.0, 85.0), "a north tilt points az 0")

        # reading the fleet: answering nodes must agree; a down node is not a disagreement
        bases = ["http://a:1", "http://b:1", "http://c:1"]
        v, d = sky.read_pointing(bases, _getter({b: -27.3 for b in bases}))
        check(v == -27.3 and "3/3" in d, "three agreeing nodes: %s" % d)
        v, d = sky.read_pointing(
            bases,
            _getter({bases[0]: -27.3, bases[1]: OSError("down"), bases[2]: -27.3}),
        )
        check(v == -27.3 and "2/3" in d, "one node down still reads: %s" % d)
        v, d = sky.read_pointing(
            bases, _getter({bases[0]: -27.3, bases[1]: -8.6, bases[2]: -27.3})
        )
        check(v is None and "DISAGREE" in d, "a disagreement installs nothing: %s" % d)
        v, d = sky.read_pointing(bases, _getter({b: None for b in bases}))
        check(v is None, "no dish_coelev_deg anywhere reads nothing")

        # installing, and keeping it through a failed re-read
        sky.set_boresight(None, None, None)
        pd = {
            ("E", 16): {"el": 62.0, "az": 180.0},
            ("G", 13): {"el": 81.41, "az": 180.0},
            ("C", 19): {"el": -5.0, "az": 10.0},
        }
        check(sky.nearest_boresight(pd) is None, "unknown boresight: nothing is near")
        ok = sky.load_boresight(bases, _getter({b: -27.3 for b in bases}))
        near = sky.nearest_boresight(pd)
        check(
            ok
            and near is not None
            and near[1] == ("E", 16)
            and abs(near[0] - 0.7) < 1e-6,
            "installed at el 62.70: E16 at el 62.0 is nearest, 0.70 deg (%s)" % (near,),
        )
        check(
            abs(sky.boresight_sep_deg(62.7, 180.0)) < 1e-9,
            "boresight_sep_deg reads the installed boresight, not an import-time default",
        )
        ok = sky.load_boresight(bases, _getter({b: OSError("down") for b in bases}))
        check(
            ok and sky.BORESIGHT_KNOWN and sky.BORESIGHT_EL_DEG == 62.7,
            "every node down: the installed boresight is kept, guards stay on",
        )
        sky.load_boresight(bases, _getter({b: -8.59 for b in bases}))
        check(round(sky.BORESIGHT_EL_DEG, 2) == 81.41, "a re-pointing is adopted")

        # init_boresight: once per process
        sky._bore_inited = False
        sky.init_boresight(
            "none", bases, _getter({b: -27.3 for b in bases}), watch=False
        )
        check(sky.nearest_boresight(pd) is None, "none: no satellite is near boresight")
        sky.init_boresight(
            "telescope", bases, _getter({b: -27.3 for b in bases}), watch=False
        )
        check(not sky.BORESIGHT_KNOWN, "a second chain's init does not re-apply")
        sky._bore_inited = False
        sky.init_boresight((180.0, 81.41), bases, None, watch=False)
        near = sky.nearest_boresight(pd)
        check(
            near is not None and near[1] == ("G", 13),
            "a fixed AZ,EL is installed as given",
        )
    finally:
        (
            sky.BORESIGHT_AZ_DEG,
            sky.BORESIGHT_EL_DEG,
            sky.BORESIGHT_KNOWN,
            sky.BORESIGHT_SOURCE,
            sky._bore_inited,
        ) = saved

    ap = build_parser("test")
    base = ["--trackers", "http://cx19:12048/gnss0_inject"]  # the one required flag
    check(ap.parse_args(base).boresight == "telescope", "parser defaults to telescope")
    check(
        ap.parse_args(base + ["--boresight", "none"]).boresight == "none",
        "accepts none",
    )
    check(
        ap.parse_args(base + ["--boresight", "180,62.7"]).boresight == (180.0, 62.7),
        "accepts AZ,EL",
    )
    try:
        ap.parse_args(base + ["--boresight", "nominal"])
        check(False, "nominal (the old hardcoded pointing) is refused")
    except SystemExit:
        check(True, "nominal (the old hardcoded pointing) is refused")
    check(sky.BORESIGHT_KNOWN is False, "the module default is unknown until read")
    if _fails:
        raise SystemExit("FAILED: %d" % len(_fails))
    print("ok")


if __name__ == "__main__":
    main()
