"""--boresight none: with the dishes unpointed, nearest_boresight reports nothing, so the D3
transit veto, the dead-reckon transit freeze and the assemblers' "_bore" post all stay off;
nominal keeps the fixed boresight and the parser defaults to it.

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


def main():
    pd = {
        ("G", 32): {"el": sky.BORESIGHT_EL_DEG, "az": sky.BORESIGHT_AZ_DEG},
        ("E", 10): {"el": 40.0, "az": 90.0},
        ("C", 19): {"el": -5.0, "az": 10.0},
    }
    saved = sky.BORESIGHT_KNOWN
    try:
        sky.BORESIGHT_KNOWN = True
        near = sky.nearest_boresight(pd)
        check(
            near is not None and near[1] == ("G", 32) and near[0] < 1e-6,
            "nominal: the satellite on boresight is nearest, at 0 deg",
        )
        sky.BORESIGHT_KNOWN = False
        check(sky.nearest_boresight(pd) is None, "none: no satellite is near boresight")
        check(sky.nearest_boresight({}) is None, "none: an empty sky is still None")
    finally:
        sky.BORESIGHT_KNOWN = saved
    ap = build_parser("test")
    base = [
        "--trackers",
        "http://cx19:12048/gnss0_inject",
    ]  # the parser's one required flag
    check(ap.parse_args(base).boresight == "nominal", "parser defaults to nominal")
    check(
        ap.parse_args(base + ["--boresight", "none"]).boresight == "none",
        "parser accepts none",
    )
    check(sky.BORESIGHT_KNOWN is True, "the module default is the nominal boresight")
    if _fails:
        raise SystemExit("FAILED: %d" % len(_fails))
    print("ok")


if __name__ == "__main__":
    main()
