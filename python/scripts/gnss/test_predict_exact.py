#!/usr/bin/env python3
"""best_eph and predict_all against the scans they replaced, to the bit.

best_eph picks a record by bisection over an index a parsed record list carries, and
predict_all can be asked for one constellation (`sysc`) and can share its propagations between
calls of one pass (`memo`). None of that may move a number: the seeds, the Doppler hints and
the transit freezes all come from these rows. So the previous best_eph and predict_all are
kept here, as they were, and every comparison is `is` or `==`, never a tolerance.

  * best_eph on built lists (unsorted, repeated toes, a query midway between two toes, the
    edges of the validity window) and on every satellite of the BRDC fixtures, at and either
    side of every record boundary and every window edge.
  * predict_all over a sweep of epochs through the fixtures, for the options each broker caller
    passes: the almanac sky, the dead-reckon sky (signal, atmosphere), the transit sky (no
    health gate, a day of validity), `sysc` against the full call's rows of that constellation,
    and `memo` shared between the dead-reckon and transit calls of one epoch.

The fixtures are the two BRDC files in scripts/gnss/fixtures, and a full mixed-constellation
BRDC set when one is on this host: the directory in GNSS_BRDC_FIXTURE_DIR, else the sidecar of
the on-sky L5 transcript.

    python3 python/scripts/gnss/test_predict_exact.py
"""
import glob
import math
import os
import random
import sys
import unittest
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gnss_atmos  # noqa: E402
import gnss_ephemeris as ge  # noqa: E402

FIX = os.path.join(HERE, "..", "..", "..", "scripts", "gnss", "fixtures")
FULL_DIR = os.environ.get(
    "GNSS_BRDC_FIXTURE_DIR",
    "/home/kvand/gnss/fixtures/broker_onsky_l5_20260826.jsonl.gz.brdc",
)
DRAO = (49.32075144444, -119.62081125, 545.0)
AGES = (14400.0, ge.GLONASS_MAX_AGE_S, ge.SBAS_MAX_AGE_S, 86400.0)


def old_best_eph(records, t_gpst, max_age=14400.0):
    if records and records[0].get("sys") == "W":
        cand = [e for e in records if e["t0_gpst"] <= t_gpst <= e["t1_gpst"]]
        return min(cand, key=lambda e: abs(t_gpst - e["toe_gpst"])) if cand else None
    cand = [e for e in records if abs(t_gpst - e["toe_gpst"]) < max_age]
    return min(cand, key=lambda e: abs(t_gpst - e["toe_gpst"])) if cand else None


def old_predict_all(
    eph,
    lat,
    lon,
    alt,
    t_utc,
    mask_deg=0.0,
    max_age=14400.0,
    signal=None,
    dcb=None,
    health_gate=True,
    tropo=False,
    iono=None,
):
    t = ge.gpst_of_utc(t_utc)
    rx = ge._ecef_of_llh(lat, lon, alt)
    out = {}
    for key, recs in eph.items():
        e = old_best_eph(
            recs,
            t,
            min(max_age, ge.GLONASS_MAX_AGE_S)
            if key[0] == "R" and max_age <= 14400.0
            else min(max_age, ge.SBAS_MAX_AGE_S)
            if key[0] == "S" and max_age <= 14400.0
            else max_age,
        )
        if e is None:
            continue
        if (
            health_gate
            and e.get("health", 0) not in (0, 0.0)
            and key[0] not in "CS"
            and key not in ge.HEALTH_GATE_EXEMPT
        ):
            continue
        tau = 0.075
        for _ in range(2):
            pos, vel, clk = ge.sat_pos_clk(e, t - tau)
            th = ge.OMEGA_E[e["sys"]] * tau
            px = pos[0] * math.cos(th) + pos[1] * math.sin(th)
            py = -pos[0] * math.sin(th) + pos[1] * math.cos(th)
            pos_rx = (px, py, pos[2])
            az, el, rng = ge._azel(rx, pos_rx, lat, lon)
            tau = rng / ge.C_LIGHT
        if el < mask_deg:
            continue
        vel_rx = (
            vel[0] * math.cos(th) + vel[1] * math.sin(th),
            -vel[0] * math.sin(th) + vel[1] * math.cos(th),
            vel[2],
        )
        a = sum((p - r) * v for p, r, v in zip(pos_rx, rx, vel_rx)) / rng
        b = (px * rx[1] - py * rx[0]) / rng
        rr = a / (1.0 + (a - ge.OMEGA_E[e["sys"]] * b) / ge.C_LIGHT)
        _gd = ge.group_delay_s(e, signal, dcb)
        if e["sys"] == "W":
            r_m, r_p = (
                ge._range_m(e, rx, lat, lon, t - 1.0),
                ge._range_m(e, rx, lat, lon, t + 1.0),
            )
            range_acc = r_p - 2.0 * rng + r_m
        out[key] = dict(
            az=az,
            el=el,
            range_m=rng,
            range_rate_mps=rr,
            sat_clk_s=clk + _gd,
            tgd_s=_gd,
            toe_age_s=t - e["toe_gpst"],
            tropo_m=gnss_atmos.tropo_delay_m(el, lat, alt) if tropo else 0.0,
            iono_m=(
                gnss_atmos.klobuchar_delay_m(
                    iono[0], iono[1], lat, lon, el, az, t % 604800.0, iono[2]
                )
                if iono
                else 0.0
            ),
        )
        if e["sys"] == "W":
            out[key]["range_acc_mps2"] = range_acc
    return out


def _gw_eph(t0):
    """Two GuoWang table satellites (load_gw_table's record shape), one with two segments."""

    def seg(start, r0):
        n = 12
        return dict(
            sys="W",
            health=0,
            t0_gpst=start,
            t1_gpst=start + 30.0 * (n - 1),
            dt=30.0,
            toe_gpst=start + 150.0,
            clk_s=-1.25e-4,
            pos=[
                [r0 + 7000.0 * k, -3.6e6 + 2000.0 * k, 5.9e6 - 900.0 * k]
                for k in range(n)
            ],
            vel=[[7000.0 / 30.0, 2000.0 / 30.0, -30.0]] * n,
        )

    return {
        ("W", 3): [seg(t0 - 100.0, -2.1e6), seg(t0 + 200.0, -2.0e6)],
        ("W", 7): [seg(t0 - 200.0, -2.3e6)],
    }


def _sets():
    """[(name, eph)]: each fixture parsed as the broker parses it."""
    out = []
    for name in ("brdc_gal_e14_20261031.rnx", "brdc_sbas_20261018.rnx"):
        out.append((name, ge.parse_rinex_nav([os.path.join(FIX, name)])))
    full = sorted(glob.glob(os.path.join(FULL_DIR, "*.rnx*")))
    if full:
        out.append((FULL_DIR, ge.parse_rinex_nav(full)))
    return out


def _around(x):
    return (math.nextafter(x, -math.inf), x, math.nextafter(x, math.inf))


class TestBestEph(unittest.TestCase):
    def _same(self, recs, t, age):
        self.assertIs(
            ge.best_eph(recs, t, age), old_best_eph(recs, t, age), (t, age, len(recs))
        )

    def test_built_lists(self):
        rng = random.Random(7)
        for _ in range(300):
            n = rng.randrange(0, 12)
            # Whole-second toes, some repeated, in no order: exact ties are common.
            recs = ge.EphRecords(
                {"sys": "G", "toe_gpst": float(rng.randrange(0, 12) * 1800), "k": k}
                for k in range(n)
            )
            ts = [float(rng.randrange(-20000, 40000)) / 2.0 for _ in range(40)]
            for e in recs:
                for age in (900.0, 1800.0, 14400.0):
                    ts += _around(e["toe_gpst"])
                    ts += _around(e["toe_gpst"] - age) + _around(e["toe_gpst"] + age)
            for t in ts:
                for age in (900.0, 1800.0, 14400.0):
                    self._same(recs, t, age)
            self._same(recs, float("nan"), 14400.0)
            self._same(recs, float("inf"), 14400.0)
            # a plain list is scanned as before
            for t in ts[:40]:
                self.assertIs(ge.best_eph(list(recs), t), old_best_eph(recs, t))

    def test_a_list_that_grew_is_indexed_again(self):
        recs = ge.EphRecords(
            [{"sys": "G", "toe_gpst": 0.0}, {"sys": "G", "toe_gpst": 7200.0}]
        )
        self.assertIs(ge.best_eph(recs, 9000.0), recs[1])
        recs.append({"sys": "G", "toe_gpst": 9000.0})
        self.assertIs(ge.best_eph(recs, 9000.0), recs[2])

    def test_the_parse_returns_indexed_lists(self):
        # Both parse paths (one file, a merged list of files), or the broker runs the scan.
        path = os.path.join(FIX, "brdc_sbas_20261018.rnx")
        for eph in (ge.parse_rinex_nav(path), ge.parse_rinex_nav([path])):
            self.assertTrue(eph)
            for recs in eph.values():
                self.assertIsInstance(recs, ge.EphRecords)
                ge.best_eph(recs, recs[0]["toe_gpst"])
                self.assertEqual(recs.toe_index[0], len(recs))

    def test_guowang_segments(self):
        eph = _gw_eph(1.4e9)
        for recs in eph.values():
            for e in recs:
                for t in _around(e["t0_gpst"]) + _around(e["t1_gpst"]):
                    self._same(recs, t, 14400.0)
            self._same(recs, 1.4e9 + 250.0, 14400.0)

    def test_every_boundary_of_the_fixtures(self):
        n = 0
        rng = random.Random(11)
        for name, eph in _sets():
            for key, recs in eph.items():
                toes = sorted(e["toe_gpst"] for e in recs)
                if len(toes) > 12:
                    # a run of neighbours from a long list (an SBAS list has hundreds), ends kept
                    k = rng.randrange(1, len(toes) - 10)
                    toes = toes[:1] + toes[k : k + 10] + toes[-1:]
                ts = []
                for a, b in zip(toes, toes[1:]):
                    ts += _around((a + b) / 2.0) + ((a + b) / 2.0 + 1e-3,)
                for toe in toes:
                    ts += _around(toe)
                    for age in AGES:
                        ts += _around(toe - age) + _around(toe + age)
                for t in ts:
                    for age in AGES:
                        self._same(recs, t, age)
                        n += 1
        self.assertGreater(n, 3000)


class TestPredictAll(unittest.TestCase):
    IONO = (
        (1.1e-8, 2.2e-8, -1.2e-7, -1.2e-7),
        (9.6e4, 1.3e5, -1.3e5, -5.2e5),
        1176.45e6,
    )

    def _epochs(self, eph, n):
        toes = sorted(e["toe_gpst"] for recs in eph.values() for e in recs)
        lo, hi = toes[0] - 5 * 3600.0, toes[-1] + 5 * 3600.0
        # GPST -> the UTC datetime the broker hands predict_all; off the whole second, so the
        # sweep does not sit on the toes.
        return [
            datetime.fromtimestamp(
                ge.GPS_EPOCH.timestamp()
                - ge.GPS_UTC_LEAP
                + lo
                + (hi - lo) * (k + 0.37) / n,
                tz=timezone.utc,
            )
            for k in range(n)
        ]

    def _same(self, got, want, what):
        self.assertEqual(list(got), list(want), what)  # the same satellites, in order
        self.assertEqual(got, want, what)

    def test_rows_are_the_scans_rows(self):
        n_rows = 0
        for name, eph in _sets():
            syss = sorted(set(k[0] for k in eph))
            sig = {"G": "gps_l5", "E": "gal_e5a", "C": "bds_b3i", "R": "glo_l3oc"}
            for t in self._epochs(eph, 24 if len(eph) > 20 else 12):
                # the almanac sky (sky.brdc_predict)
                alm = old_predict_all(eph, *DRAO, t, mask_deg=-90.0)
                self._same(ge.predict_all(eph, *DRAO, t, mask_deg=-90.0), alm, name)
                self._same(
                    ge.predict_all(eph, *DRAO, t), old_predict_all(eph, *DRAO, t), name
                )
                # the dead-reckon sky and the transit sky of one pass, sharing a memo
                dr_kw = dict(
                    mask_deg=-90.0, signal=sig.get(syss[0]), tropo=True, iono=self.IONO,
                )
                tr_kw = dict(mask_deg=-90.0, max_age=86400.0, health_gate=False)
                dr = old_predict_all(eph, *DRAO, t, **dr_kw)
                tr = old_predict_all(eph, *DRAO, t, **tr_kw)
                memo = {}
                self._same(ge.predict_all(eph, *DRAO, t, memo=memo, **dr_kw), dr, name)
                self._same(ge.predict_all(eph, *DRAO, t, memo=memo, **tr_kw), tr, name)
                self._same(ge.predict_all(eph, *DRAO, t, memo=memo, **dr_kw), dr, name)
                self._same(ge.predict_all(eph, *DRAO, t, **dr_kw), dr, name)
                self._same(ge.predict_all(eph, *DRAO, t, **tr_kw), tr, name)
                # a mask above the horizon, with and without the memo
                up = old_predict_all(eph, *DRAO, t, mask_deg=10.0)
                self._same(ge.predict_all(eph, *DRAO, t, mask_deg=10.0), up, name)
                self._same(
                    ge.predict_all(eph, *DRAO, t, mask_deg=10.0, memo=memo), up, name
                )
                # one constellation: the full call's rows of it, in the full call's order
                for s in syss + ["Q"]:
                    for kw, full in (({"mask_deg": -90.0}, alm), (dr_kw, dr)):
                        self._same(
                            ge.predict_all(eph, *DRAO, t, sysc=s, **kw),
                            {k: v for k, v in full.items() if k[0] == s},
                            (name, s),
                        )
                n_rows += len(alm) + len(dr) + len(tr)
        self.assertGreater(n_rows, 100)

    def test_guowang_rows(self):
        t0 = 1.4e9
        eph = dict(ge.parse_rinex_nav([os.path.join(FIX, "brdc_gal_e14_20261031.rnx")]))
        eph.update(_gw_eph(t0))
        n_w = 0
        for dt in (-150.0, 0.0, 100.0, 199.0, 201.0, 400.0, 900.0):
            t = datetime.fromtimestamp(
                ge.GPS_EPOCH.timestamp() - ge.GPS_UTC_LEAP + t0 + dt, tz=timezone.utc
            )
            want = old_predict_all(eph, *DRAO, t, mask_deg=-90.0)
            memo = {}
            for _ in range(2):
                self._same(
                    ge.predict_all(eph, *DRAO, t, mask_deg=-90.0, memo=memo), want, dt
                )
            self._same(
                ge.predict_all(eph, *DRAO, t, mask_deg=-90.0, sysc="W"),
                {k: v for k, v in want.items() if k[0] == "W"},
                dt,
            )
            n_w += sum(1 for k in want if k[0] == "W")
        self.assertGreater(n_w, 6)

    def test_a_memo_serves_each_record_once(self):
        eph = ge.parse_rinex_nav([os.path.join(FIX, "brdc_gal_e14_20261031.rnx")])
        t = datetime.fromtimestamp(1793407620.0, tz=timezone.utc)
        calls = []
        real = ge.sat_pos_clk

        def spy(e, t_gpst):
            calls.append(id(e))
            return real(e, t_gpst)

        ge.sat_pos_clk = spy
        try:
            memo = {}
            a = ge.predict_all(eph, *DRAO, t, mask_deg=-90.0, memo=memo)
            n = len(calls)
            b = ge.predict_all(
                eph,
                *DRAO,
                t,
                mask_deg=-90.0,
                max_age=86400.0,
                health_gate=False,
                memo=memo
            )
        finally:
            ge.sat_pos_clk = real
        self.assertGreater(n, 0)
        self.assertEqual(sorted(a), sorted(b))
        self.assertEqual(len(calls), n)  # the second call propagated nothing
        # rows are the caller's own: editing one call's row leaves the other's alone
        a[("E", 15)]["az"] = None
        self.assertIsNotNone(b[("E", 15)]["az"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
