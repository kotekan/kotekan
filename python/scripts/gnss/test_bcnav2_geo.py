#!/usr/bin/env python3
"""beidou_bcnav2.sv_position_bcnav2 for a GEO (SatType 1): the BDS ICD's R_X(-5 deg) tilt.

A broadcast C01 record (BRDC00WRD, 2026-10-18 23:00 BDT, toe 82800) mapped onto the B-CNAV
field names (A_dot = dn_0_dot = 0, so the CNAV propagation reduces to the D1/D2 one) must
  * agree with an independent transcription of RTKLIB eph2pos's BDS GEO branch, and
  * stay within 2 deg of the equator, as a geostationary orbit does.
"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import beidou_bcnav2 as B  # noqa: E402

# C01 broadcast record (RINEX field order: toe, sqrtA, e, M0, omega, OMEGA0, OMEGADOT, i0,
# IDOT, Delta n, Cuc, Cus, Crc, Crs, Cic, Cis).
C01 = {
    "toe": 82800.0,
    "sqrtA": 6.493500297546e03,
    "e": 5.619690055028e-04,
    "M0": -1.088255016132e00,
    "omega": 3.803602673947e-01,
    "OMEGA0": 2.905860603181e00,
    "OMEGADOT": -5.460584598124e-09,
    "i0": 7.967923466281e-02,
    "IDOT": 1.242908915002e-10,
    "dn": 6.591703142323e-09,
    "Cuc": 6.756279617548e-06,
    "Cus": -1.492444425821e-06,
    "Crc": 3.943750000000e01,
    "Crs": 2.061562500000e02,
    "Cic": -8.055940270424e-08,
    "Cis": -1.396983861923e-08,
}


def _bcnav_eph(r):
    A = r["sqrtA"] ** 2
    return {
        "t_oe": r["toe"],
        "SatType": 1.0,
        "A_ref": B.A_REF_IGSO,
        "dA": A - B.A_REF_IGSO,
        "A_dot": 0.0,
        "dn_0": r["dn"],
        "dn_0_dot": 0.0,
        "M_0": r["M0"],
        "e": r["e"],
        "omega": r["omega"],
        "Omega_0": r["OMEGA0"],
        "i_0": r["i0"],
        "Omega_dot": r["OMEGADOT"],
        "i_0_dot": r["IDOT"],
        "C_IS": r["Cis"],
        "C_IC": r["Cic"],
        "C_RS": r["Crs"],
        "C_RC": r["Crc"],
        "C_US": r["Cus"],
        "C_UC": r["Cuc"],
    }


def _rtklib_geo(r, t):
    """RTKLIB eph2pos (BDS GEO branch), transcribed: SIN_5/COS_5 are sin/cos(-5 deg)."""
    mu, omge = 3.986004418e14, 7.292115e-5
    SIN_5, COS_5 = -0.0871557427476582, 0.9961946980917456
    A = r["sqrtA"] ** 2
    tk = t - r["toe"]
    M = r["M0"] + (math.sqrt(mu / (A * A * A)) + r["dn"]) * tk
    E, Ek = M, 0.0
    for _ in range(30):
        if abs(E - Ek) <= 1e-13:
            break
        Ek = E
        E -= (E - r["e"] * math.sin(E) - M) / (1.0 - r["e"] * math.cos(E))
    u = math.atan2(math.sqrt(1.0 - r["e"] ** 2) * math.sin(E), math.cos(E) - r["e"])
    u += r["omega"]
    rad = A * (1.0 - r["e"] * math.cos(E))
    i = r["i0"] + r["IDOT"] * tk
    sin2u, cos2u = math.sin(2.0 * u), math.cos(2.0 * u)
    u += r["Cus"] * sin2u + r["Cuc"] * cos2u
    rad += r["Crs"] * sin2u + r["Crc"] * cos2u
    i += r["Cis"] * sin2u + r["Cic"] * cos2u
    x, y, cosi = rad * math.cos(u), rad * math.sin(u), math.cos(i)
    O = r["OMEGA0"] + r["OMEGADOT"] * tk - omge * r["toe"]
    xg = x * math.cos(O) - y * cosi * math.sin(O)
    yg = x * math.sin(O) + y * cosi * math.cos(O)
    zg = y * math.sin(i)
    sino, coso = math.sin(omge * tk), math.cos(omge * tk)
    return (
        xg * coso + yg * sino * COS_5 + zg * sino * SIN_5,
        -xg * sino + yg * coso * COS_5 + zg * coso * SIN_5,
        -yg * SIN_5 + zg * COS_5,
    )


TIMES = [C01["toe"] + dt for dt in range(-6 * 3600, 6 * 3600 + 1, 1800)]


class TestGeoRotation(unittest.TestCase):
    def test_matches_rtklib(self):
        eph = _bcnav_eph(C01)
        for t in TIMES:
            got = B.sv_position_bcnav2(eph, t)
            want = _rtklib_geo(C01, t)
            self.assertLess(math.dist(got, want), 1e-3, "t=%.0f" % t)

    def test_stays_on_the_equator(self):
        eph = _bcnav_eph(C01)
        for t in TIMES:
            x, y, z = B.sv_position_bcnav2(eph, t)
            lat = math.degrees(math.atan2(z, math.hypot(x, y)))
            self.assertLess(abs(lat), 2.0, "t=%.0f" % t)


if __name__ == "__main__":
    unittest.main()
