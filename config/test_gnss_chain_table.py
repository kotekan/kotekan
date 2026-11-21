#!/usr/bin/env python3
"""gnss_chain_table.py: the repo's rows validate, and validate() refuses a duplicate id, a
descriptor signal_by_name does not resolve, a malformed row and a misplaced primary;
dll_spacing_drift names a node or broker value off its row; the gnss-obs@ unit check reports
the stack target's missing units and fails only when strict.

    python3 config/test_gnss_chain_table.py

Hermetic: the manifest and the interface golden are the repo's (no network).
"""
import contextlib
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gnss_chain_table as ct  # noqa: E402

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def repo():
    return ct.load()


def test_repo_table():
    t = repo()
    check(ct.validate(t) == [], "the repo's rows validate (%s)" % ct.validate(t))
    check(
        t[0].primary and t[0].key == "gps_l5" and not any(r.primary for r in t[1:]),
        "gps_l5 is the first row and the only primary",
    )
    check(
        len({r.id for r in t}) == len(t) == len({r.key for r in t}),
        "ids and keys are unique",
    )


def test_duplicate_id():
    t = repo()
    t[2] = t[2]._replace(id=t[1].id)
    bad = ct.validate(t)
    check(
        any("id %d is on 2 rows" % t[1].id in b for b in bad),
        "a duplicate id is refused (%s)" % bad,
    )


def test_descriptor_signal_by_name_lacks():
    t = repo()
    t[3] = t[3]._replace(descriptor="GAL_E5B_Q_XX")
    bad = ct.validate(t)
    check(
        any(
            "descriptor GAL_E5B_Q_XX is not one signal_by_name resolves" in b
            for b in bad
        ),
        "a descriptor missing from signal_by_name's list (the golden's) is refused (%s)"
        % bad,
    )
    names = {r.descriptor for r in repo()} - {"GW_1202"}
    bad = ct.validate(repo(), descriptors=names)
    check(
        any("descriptor GW_1202 is not" in b for b in bad),
        "...against a given list too (%s)" % bad,
    )


def test_key_follows_the_descriptor():
    t = repo()
    t[1] = t[1]._replace(key="gal_e5")
    bad = ct.validate(t)
    check(
        any("key must be gal_e5a" in b for b in bad),
        "a key that is not the descriptor's chain key is refused (%s)" % bad,
    )


def test_primary_first_and_only():
    t = repo()
    t[0], t[1] = t[1], t[0]
    check(
        any("the first row, and only it, is the primary" in b for b in ct.validate(t)),
        "a primary that is not the first row is refused",
    )
    t = repo()
    t[5] = t[5]._replace(primary=True)
    check(
        any("primary: gps_l5, sbas_l5" in b for b in ct.validate(t)),
        "a second primary is refused",
    )


def test_malformed_rows():
    for row, needle in (
        ({"id": 1, "key": "gps_l5", "descriptor": "GPS_L5_Q_NH"}, "no prns"),
        (
            dict(
                id=1,
                key="gps_l5",
                descriptor="GPS_L5_Q_NH",
                prns="1",
                **{
                    "dll-spacing-chips": 0.5,
                    "telem-rows": 14,
                    "proj-sys": "G",
                    "shared-ref-band": "l5",
                    "dll-spacing": 0.5,
                }
            ),
            "unknown key dll-spacing",
        ),
    ):
        try:
            ct.rows({"chains": [row]})
            check(False, "%s: refused" % needle)
        except ct.TableError as e:
            check(needle in str(e), "%s: refused (%s)" % (needle, e))
    t = repo()
    t[4] = t[4]._replace(prns="6,7,7", telem_rows=0, proj_sys="X", dll_spacing_chips=2)
    bad = " | ".join(ct.validate(t))
    check(
        all(
            s in bad
            for s in ("without repeats", "telem-rows", "proj-sys", "dll-spacing-chips")
        ),
        "repeated PRNs, a zero row count, an unknown letter and a spacing past a chip are "
        "refused (%s)" % bad,
    )


def test_dll_spacing_drift():
    t = repo()
    want = {r.key: r.dll_spacing_chips for r in t}
    check(ct.dll_spacing_drift(t, want, "node") == [], "the rows' own values agree")
    got = ct.dll_spacing_drift(t, dict(want, glo_l2of=0.5), "node cx19")
    check(
        got == ["node cx19: glo_l2of dll spacing 0.5, the chains table says 0.33"],
        "a node tap off its row is named (%s)" % got,
    )
    got = ct.dll_spacing_drift(t, {"glo_l2oc": 0.5, "gal_e1c": 0.2}, "broker")
    check(
        got == ["broker: glo_l2oc dll spacing 0.5, the chains table says 0.4"],
        "a broker spacing off its row is named (%s)" % got,
    )


TARGET = """[Unit]
Wants=gnss-gather.service gnss-broker.service
Wants=gnss-obs@gps_l5.service gnss-obs@gal_e5a.service
# Wants=gnss-obs@bds_b2a.service
Wants=gnss-obs@sbas_l9.service
"""


def test_obs_units():
    t = repo()[:3]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "gnss-stack.target")
        with open(path, "w") as f:
            f.write(TARGET)
        missing, extra = ct.obs_unit_check(t, path)
        check(
            missing == ["gnss-obs@bds_b2a.service"]
            and extra == ["gnss-obs@sbas_l9.service"],
            "a chain without a writer and a writer without a chain are named, a "
            "commented line is not read (%s, %s)" % (missing, extra),
        )
        rc = {}
        for strict in (False, True):
            with contextlib.redirect_stdout(io.StringIO()):
                rc[strict] = ct.main(
                    ["obs-units", "--target", path] + (["--strict"] if strict else [])
                )
        check(
            rc == {False: 0, True: 1},
            "the check reports and passes, and fails only when strict (%s)" % rc,
        )
    # The repo's target against the repo's rows: reported, not failed. Adding the units is a
    # decision for whoever runs the stack.
    missing, extra = ct.obs_unit_check(repo())
    print(
        "  [NOTE] gnss-stack.target starts %d gnss-obs@ units for %d chains; missing %s; "
        "extra %s"
        % (
            len(ct.stack_obs_units()),
            len(repo()),
            [u[9:-8] for u in missing] or "none",
            [u[9:-8] for u in extra] or "none",
        )
    )


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        try:
            t()
        except Exception as e:
            check(False, "%s raised %s: %s" % (t.__name__, type(e).__name__, e))
    if _fails:
        print("\n%d FAILED" % len(_fails))
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
