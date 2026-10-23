#!/usr/bin/env python3
"""SBAS (S) records in a RINEX 3 mixed-nav file: parsed, keyed by PRN (S31 -> 131), propagated from
the MT9 state vector as a two-body orbit, and seen from DRAO where the WAAS GEOs stand (#190
coverage).

Why each check exists: the key (a two-digit RINEX number is not a PRN); the geometry (a WAAS GEO
is geostationary at its assigned longitude, so a wrong unit or axis shows as a wrong radius or
longitude); the propagation (one record's state vector must land on the next record's position
to a few metres over the 4-minute cadence under either model, else the frame rotation, the
acceleration term or the epoch is wrong; and on a synthetic circular orbit, where the answer is
exact, the two-body integration must hold the satellite at its longitude to a metre over hours,
else the omega x r term or the rotation back to ECEF has the wrong sign); the model switch (the
polynomial stays callable for comparison and the two agree over a minute); and the age bound (a
GEO record is good for an hour or two, not the Keplerian default).
"""
import math
import os
import sys
import unittest
import warnings
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gnss_ephemeris as ge  # noqa: E402

FIX = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts", "gnss", "fixtures",
    "brdc_sbas_20261018.rnx",
)
GM = ge.MU["S"]
W = ge.OMEGA_E["S"]
R_GEO = (GM / W ** 2) ** (1.0 / 3.0)  # the geostationary radius of the module's constants
SIDEREAL_DAY_S = 2.0 * math.pi / W


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _geo_record(lon_deg, t0=1.4e9, vel=(0.0, 0.0, 0.0), acc=(0.0, 0.0, 0.0)):
    """A synthetic S record at the geostationary radius over lon_deg with the given ECEF
    velocity and acceleration (both zero: perfectly geostationary)."""
    lam = math.radians(lon_deg)
    return dict(
        sys="S",
        prn=131,
        af0=0.0,
        af1=0.0,
        frame_time=0.0,
        pos=(R_GEO * math.cos(lam), R_GEO * math.sin(lam), 0.0),
        vel=tuple(vel),
        acc=tuple(acc),
        health=0.0,
        ura=2.0,
        iodn=0.0,
        toc_gpst=t0,
        toe_gpst=t0,
    )


class TestTwoBody(unittest.TestCase):
    """The two-body propagation on synthetic orbits whose answer is exact."""

    def test_geo_radius_of_the_constants(self):
        self.assertAlmostEqual(R_GEO / 1e3, 42164.17, delta=0.01)

    def test_geostationary_stays_at_its_longitude(self):
        """A circular equatorial orbit at the geostationary radius keeps its ECEF position: to
        < 1 m after 6 h, after -6 h, and after one sidereal day (1437 RK4 steps), with an ECEF
        velocity under 1 mm/s. A wrong sign on omega x r or on the rotation back would move it
        by thousands of km."""
        e = _geo_record(-117.0)
        for dt in (21600.0, -21600.0, SIDEREAL_DAY_S):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pos, vel = ge.sbas_state(e, e["toe_gpst"] + dt, model="twobody")
            self.assertLess(_dist(pos, e["pos"]), 1.0, dt)
            self.assertLess(math.sqrt(sum(v * v for v in vel)), 1e-3, dt)

    def test_inclined_circular_orbit_matches_the_closed_form(self):
        """A circular orbit at the geostationary radius inclined 5 deg, handed to the record as
        ECEF (r, v) at its ascending node: after 6 h the integrated ECEF state matches the
        closed-form inertial circle rotated into ECEF to < 1 m and < 1 mm/s (the z motion that a
        geostationary case cannot exercise; 3.7e6 m of it at 6 h)."""
        inc = math.radians(5.0)
        t0 = 1.4e9
        v_i = (0.0, R_GEO * W * math.cos(inc), R_GEO * W * math.sin(inc))
        # vel_ecef = v_i - omega x r = v_i - (0, w R, 0) at the node on +x
        e = _geo_record(0.0, t0, vel=(v_i[0], v_i[1] - W * R_GEO, v_i[2]))
        dt = 21600.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # past SBAS_MAX_AGE_S on purpose
            pos, vel = ge.sbas_state(e, t0 + dt, model="twobody")
        ph = W * dt  # the mean motion at this radius is w
        c, s = math.cos(ph), math.sin(ph)
        r_i = (R_GEO * c, R_GEO * s * math.cos(inc), R_GEO * s * math.sin(inc))
        v_it = (-R_GEO * W * s, R_GEO * W * c * math.cos(inc), R_GEO * W * c * math.sin(inc))
        v_rel = (v_it[0] + W * r_i[1], v_it[1] - W * r_i[0], v_it[2])
        r_e = (c * r_i[0] + s * r_i[1], -s * r_i[0] + c * r_i[1], r_i[2])
        v_e = (c * v_rel[0] + s * v_rel[1], -s * v_rel[0] + c * v_rel[1], v_rel[2])
        self.assertLess(_dist(pos, r_e), 1.0)
        self.assertGreater(abs(pos[2]), 3.0e6)
        for k in range(3):
            self.assertAlmostEqual(vel[k], v_e[k], delta=1e-3)

    def test_constant_acceleration_model_still_callable(self):
        """model="constaccel" is the MT9 polynomial pos + vel dt + acc dt^2 / 2; an unknown
        model is an error, not a silent default."""
        e = _geo_record(-117.0, vel=(1.0, -2.0, 0.5), acc=(1e-5, 2e-5, -3e-5))
        dt = 1800.0
        pos, vel = ge.sbas_state(e, e["toe_gpst"] + dt, model="constaccel")
        for k in range(3):
            self.assertAlmostEqual(
                pos[k], e["pos"][k] + e["vel"][k] * dt + 0.5 * e["acc"][k] * dt * dt, places=6
            )
            self.assertAlmostEqual(vel[k], e["vel"][k] + e["acc"][k] * dt, places=9)
        with self.assertRaises(ValueError):
            ge.sbas_state(e, e["toe_gpst"] + dt, model="kepler")

    def test_models_agree_over_a_minute_on_the_synthetic_orbit(self):
        e = _geo_record(-117.0)
        for dt in (1.0, 30.0, 60.0, -60.0):
            t = e["toe_gpst"] + dt
            self.assertLess(
                _dist(ge.sbas_state(e, t, model="twobody")[0], ge.sbas_state(e, t, model="constaccel")[0]),
                1.0,
                dt,
            )


class TestSBAS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = ge.parse_rinex_nav([FIX])

    def test_keyed_by_prn_and_geostationary(self):
        """S31/S33/S35 -> PRNs 131/133/135 (WAAS: Eutelsat 117 West B, SES-15, Galaxy 30) at
        geostationary radius and their assigned longitudes, 117 W for PRN 131."""
        keys = sorted(k for k in self.recs if k[0] == "S")
        self.assertEqual([k[1] for k in keys], [131, 133, 135])
        e = self.recs[("S", 131)][0]
        pos, vel, clk = ge.sat_pos_clk(e, e["toe_gpst"])
        r = math.sqrt(sum(x * x for x in pos))
        lon = math.degrees(math.atan2(pos[1], pos[0]))
        self.assertAlmostEqual(r / 1e3, 42166.0, delta=30.0)
        self.assertAlmostEqual(lon, -117.0, delta=0.1)
        self.assertLess(abs(pos[2]), 20e3)  # on the equator to 20 km
        self.assertLess(math.sqrt(sum(v * v for v in vel)), 2.0)  # geostationary: < 2 m/s
        self.assertLess(abs(clk), 1e-6)

    def test_state_vector_propagates_onto_the_next_record(self):
        """Under either model, and under the default through sat_pos_clk."""
        recs = self.recs[("S", 131)]
        self.assertGreater(len(recs), 5)
        for model in ("twobody", "constaccel", None):
            worst = 0.0
            for a, b in zip(recs, recs[1:]):
                p = ge.sbas_state(a, b["toe_gpst"], model=model)[0] if model else ge.sat_pos_clk(a, b["toe_gpst"])[0]
                worst = max(worst, _dist(p, b["pos"]))
            self.assertLess(worst, 5.0, model)  # metres, over the ~4 min cadence

    def test_models_agree_over_a_minute_on_a_real_record(self):
        """For dt <= 60 s the two models agree to < 1 m on a WAAS record (its own velocity and
        MT9 acceleration); at an hour they part by tens of metres, which is the point."""
        e = self.recs[("S", 131)][0]
        for dt in (1.0, 30.0, 60.0, -60.0):
            t = e["toe_gpst"] + dt
            p_tb, v_tb = ge.sbas_state(e, t, model="twobody")
            p_ca, v_ca = ge.sbas_state(e, t, model="constaccel")
            self.assertLess(_dist(p_tb, p_ca), 1.0, dt)
            self.assertLess(_dist(v_tb, v_ca), 1e-2, dt)
        t = e["toe_gpst"] + 3600.0
        self.assertGreater(
            _dist(ge.sbas_state(e, t, model="twobody")[0], ge.sbas_state(e, t, model="constaccel")[0]), 1.0
        )

    def test_default_model_is_two_body_and_switchable(self):
        """sat_pos_clk (what predict_all uses) follows SBAS_MODEL: two-body unless switched."""
        e = self.recs[("S", 131)][0]
        t = e["toe_gpst"] + 3600.0
        self.assertEqual(ge.SBAS_MODEL, "twobody")
        self.assertEqual(ge.sat_pos_clk(e, t)[0], ge.sbas_state(e, t, model="twobody")[0])
        old = ge.SBAS_MODEL
        try:
            ge.SBAS_MODEL = "constaccel"
            self.assertEqual(ge.sat_pos_clk(e, t)[0], ge.sbas_state(e, t, model="constaccel")[0])
        finally:
            ge.SBAS_MODEL = old

    def test_epoch_returns_the_record(self):
        e = self.recs[("S", 131)][0]
        pos, vel = ge.sbas_state(e, e["toe_gpst"])
        for k in range(3):
            self.assertAlmostEqual(pos[k], e["pos"][k], places=6)
            self.assertAlmostEqual(vel[k], e["vel"][k], places=9)

    def test_velocity_is_the_integrated_state(self):
        """sat_pos_clk hands back the integrated velocity, consistent with the position's own
        time derivative to the mm/s the Doppler hints need."""
        e = self.recs[("S", 131)][0]
        t = e["toe_gpst"] + 1800.0
        _pos, vel, clk = ge.sat_pos_clk(e, t)
        pp = ge._pos_only(e, t + 0.5)
        pm = ge._pos_only(e, t - 0.5)
        for k in range(3):
            self.assertAlmostEqual(vel[k], pp[k] - pm[k], places=3)
        self.assertAlmostEqual(clk, e["af0"] + e["af1"] * 1800.0, places=15)

    def test_beyond_2h_warns(self):
        e = self.recs[("S", 131)][0]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ge.sbas_state(e, e["toe_gpst"] + ge.SBAS_MAX_AGE_S + 1.0)
        self.assertEqual(len(w), 1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ge.sbas_state(e, e["toe_gpst"] + ge.SBAS_MAX_AGE_S)
        self.assertEqual(len(w), 0)

    def test_seen_from_drao_due_south_at_33_deg(self):
        t = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
        pa = ge.predict_all(self.recs, 49.32, -119.62, 528.0, t, mask_deg=-90.0, health_gate=False)
        self.assertIn(("S", 131), pa)
        v = pa[("S", 131)]
        self.assertAlmostEqual(v["el"], 33.4, delta=0.5)
        self.assertAlmostEqual(v["az"], 176.5, delta=1.0)
        self.assertLess(abs(v["range_rate_mps"]), 1.0)

    def test_record_units_metres_or_km(self):
        """The rapid BRDC writes the GAGAN records (S27, S28) in metres, not RINEX's km: read as
        metres when the radius says so, dropped when it is no orbit at all."""
        e_km = self.recs[("S", 131)][0]
        af = (e_km["af0"], e_km["af1"], e_km["frame_time"])
        km = [c * 1e-3 for c in e_km["pos"]]
        v = [c * 1e-3 for c in e_km["vel"]]
        a = [c * 1e-3 for c in e_km["acc"]]
        orb_km = [km[0], v[0], a[0], 0.0, km[1], v[1], a[1], 2.0, km[2], v[2], a[2], 5.0]
        orb_m = [c * 1e3 if i % 4 != 3 else c for i, c in enumerate(orb_km)]
        e_m = ge._sbas_record(31, af, orb_m, e_km["toe_gpst"])
        self.assertIsNotNone(e_m)
        for i in range(3):
            self.assertAlmostEqual(e_m["pos"][i], e_km["pos"][i], places=3)
            self.assertAlmostEqual(e_m["vel"][i], e_km["vel"][i], places=6)
            self.assertAlmostEqual(e_m["acc"][i], e_km["acc"][i], places=9)
        # and the two readings propagate alike, under either model, an hour out
        t = e_km["toe_gpst"] + 3600.0
        for model in ("twobody", "constaccel"):
            self.assertLess(
                _dist(ge.sbas_state(e_m, t, model=model)[0], ge.sbas_state(e_km, t, model=model)[0]), 1e-3
            )
        self.assertIsNone(ge._sbas_record(31, af, [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0], e_km["toe_gpst"]))
        self.assertIsNone(ge._sbas_record(31, af, [7.0e4, 0, 0, 0, 0.0, 0, 0, 0, 0.0, 0, 0, 0], e_km["toe_gpst"]))

    def test_age_bound(self):
        """A record older than SBAS_MAX_AGE_S is not used for a clock-grade prediction (two-body
        errs ~210 m, 0.7 L5 chip, at that age against the next record; sbas_eph_check.py)."""
        recs = self.recs[("S", 131)]
        t_late = recs[-1]["toe_gpst"] + ge.SBAS_MAX_AGE_S + 60.0
        self.assertIsNone(ge.best_eph(recs, t_late, ge.SBAS_MAX_AGE_S))
        self.assertIsNotNone(ge.best_eph(recs, recs[-1]["toe_gpst"] + 600.0, ge.SBAS_MAX_AGE_S))


if __name__ == "__main__":
    unittest.main(verbosity=2)
