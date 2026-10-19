#!/usr/bin/env python3
"""viscap_metadata's sat_azel.csv geometry and channels.csv rows, offline.

    python3 scripts/gnss/test_viscap_metadata.py

sat_azel.csv (here on synthetic GPS records) is the broker's sky as gnss_ephemeris.predict_all
computes it, with the health gate and the elevation mask off: every satellite with a record
inside max_age, unhealthy and below the horizon included, each with its record's health word.
The geometry must come from predict_all itself, so a correction there reaches the bundle.
channels.csv names every chain's channels and, for a capture, the capvistiles channel axis,
which is the union of the captured chains' channels and need not match any one chain's.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import viscap_metadata as vm  # noqa: E402

eph = vm.eph

LAT, LON, ALT = 49.3209, -119.6236, 545.0
T_UTC = 1792000000.0  # unix seconds
MAX_AGE = 6 * 3600

_fails = []


def check(ok, what):
    print("  [%s] %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        _fails.append(what)


def gps_record(m0, omega0, health, toe_offset_s=0.0):
    """A circular-ish GPS Keplerian record with the fields sat_pos_clk reads."""
    t = eph.gpst_of_utc(T_UTC)
    toe = round(t / 16.0) * 16.0 + toe_offset_s
    return dict(
        sys="G",
        sqrta=5153.7,
        dn=4.5e-9,
        toe_gpst=toe,
        toe_sow=toe % 604800.0,
        toc_gpst=toe,
        m0=m0,
        ecc=0.01,
        omega=0.5,
        cus=0.0,
        cuc=0.0,
        crs=0.0,
        crc=0.0,
        cis=0.0,
        cic=0.0,
        i0=0.96,
        idot=0.0,
        omega0=omega0,
        omegadot=-8.0e-9,
        af0=1e-5,
        af1=0.0,
        af2=0.0,
        health=health,
    )


def sky():
    """Eight satellites around the orbit, two of them unhealthy, plus one whose only record is
    older than max_age."""
    E = {}
    for k in range(8):
        h = 1 if k in (2, 5) else 0
        E[("G", k + 1)] = [gps_record(0.8 * k, 0.7 * k, h)]
    E[("G", 30)] = [gps_record(0.3, 0.2, 0, toe_offset_s=-(MAX_AGE + 3600))]
    return E


def test_geometry_is_predict_alls():
    E = sky()
    calls = []
    real = eph.predict_all

    def spy(*a, **kw):
        calls.append(kw)
        return real(*a, **kw)

    eph.predict_all = spy
    try:
        got = vm.predict_all_health_agnostic(E, LAT, LON, ALT, T_UTC, MAX_AGE)
    finally:
        eph.predict_all = real
    check(
        len(calls) == 1 and calls[0].get("health_gate") is False,
        "computed by eph.predict_all(health_gate=False) (calls: %r)" % calls,
    )
    want = real(
        E, LAT, LON, ALT, T_UTC, mask_deg=-90.0, max_age=MAX_AGE, health_gate=False
    )
    check(
        sorted(got) == sorted(want) and len(want) == 8,
        "the 8 satellites inside max_age, the stale one dropped (got %s)" % sorted(got),
    )
    bad = [
        (key, f)
        for key in want
        for f in ("az", "el", "range_m", "range_rate_mps", "toe_age_s")
        if key in got and got[key][f] != want[key][f]
    ]
    check(
        not bad, "az/el/range/range rate/toe age are predict_all's (differ: %s)" % bad
    )
    els = [v["el"] for v in got.values()]
    check(
        min(els) < 0 < max(els),
        "no elevation mask: satellites on both sides of the horizon (el %.1f .. %.1f)"
        % (min(els), max(els)),
    )
    hw = {key: got[key].get("health") for key in got}
    check(
        hw == {key: E[key][0]["health"] for key in got},
        "each satellite carries its record's health word (got %r)" % hw,
    )


def injector(signal, local, fids):
    return dict(
        name="cudaGnssInject",
        signal=signal,
        gnss_local_channels=local,
        channel_ids=fids,
    )


def test_channel_rows():
    """A merged GPU with chains on different channels and the capture gather over them, and
    one without a capture: every chain's channels in its own order, then the capture axis."""
    cfg = {
        "gnss0_n2dual": {
            "commands": [
                injector("GPS_L5_Q", [3, 5], [5972, 5988]),
                injector("GAL_E5B_Q_CS", [9, 7], [6148, 6132]),
                injector("BDS_B3I_NH", [11], [6452]),
                {
                    "name": "cudaCorrelatorDual",
                    "gnss_gathers": ["gps_l5", "gal_e5b", "bds_b3i", "cap"],
                    "gather_cap_channels": [3, 5, 7, 9],
                },
            ]
        },
        "gnss1_n2dual": {
            "commands": [
                injector("GPS_L5_Q", [4], [5982]),
                {"name": "cudaCorrelatorDual"},
            ]
        },
        "gnss0_n2dual_buf": {"kotekan_buffer": "standard"},
    }
    want = [
        (0, "gps_l5", 0, 5972),
        (0, "gps_l5", 1, 5988),
        (0, "gal_e5b", 0, 6148),
        (0, "gal_e5b", 1, 6132),
        (0, "bds_b3i", 0, 6452),
        (0, "cap", 0, 5972),
        (0, "cap", 1, 5988),
        (0, "cap", 2, 6132),
        (0, "cap", 3, 6148),
        (1, "gps_l5", 0, 5982),
    ]
    want = [["cx99", g, ch, i, f, "%.7f" % (f * 0.1953125)] for g, ch, i, f in want]
    got = vm.channel_rows("cx99", cfg)
    check(
        got == want, "channels.csv rows per chain and the capture axis (got %s)" % got
    )


def main():
    test_geometry_is_predict_alls()
    test_channel_rows()
    print(
        "test_viscap_metadata: %s" % ("FAIL (%d)" % len(_fails) if _fails else "PASS")
    )
    sys.exit(1 if _fails else 0)


if __name__ == "__main__":
    main()
