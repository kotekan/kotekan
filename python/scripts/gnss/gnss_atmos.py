"""Atmospheric code delays for the range model: Saastamoinen troposphere and Klobuchar ionosphere.

Both return a one-way delay in metres to ADD to the geometric range (the signal arrives later).
The broker's cp_predicted and the observables writer take them from the predict_all row
(tropo_m, iono_m), so every consumer of the model sees one atmosphere.

Troposphere: Saastamoinen zenith delays (hydrostatic + wet) from a standard atmosphere at the
site height (15 C, 1013.25 hPa, 50 % humidity at sea level, 6.5 K/km), mapped with Chao's
function. Good to ~0.1 m above 10 deg elevation and ~0.3 m at 5 deg; a weather feed or a
seasonal model would be the next step, not a different structure.

Ionosphere: the Klobuchar model (IS-GPS-200 20.3.3.5.2.5) with the GPSA/GPSB coefficients from
the broadcast nav header, at the signal's own carrier (the L1 delay scales by (f_L1/f)^2: x1.79
at L5/E5a/B2a). The model removes about half the delay in the rms sense; a GIM is the next step.
The same GPS coefficients serve every constellation: the ionosphere is one, the broadcast
parameters only describe it.

The other broadcast models, for the position menu's comparisons: BeiDou's Klobuchar (D1) and its
global model BDGIM (B-CNAV2/B-CNAV3), with the coefficients as the decoders read them
(decoded_iono), and CODE's global maps (Ionex).
"""

import math

C_LIGHT = 299792458.0
F_L1 = 1575.42e6


def _standard_atmosphere(h_m, rh=0.5):
    """(P hPa, T K, e hPa) at height h from the ICAO standard atmosphere, with relative humidity
    rh held at the sea-level value."""
    t = 288.15 - 0.0065 * h_m
    p = 1013.25 * (1.0 - 2.2557e-5 * h_m) ** 5.2568
    e = rh * 6.108 * math.exp((17.15 * t - 4684.0) / (t - 38.45))
    return p, t, e


def tropo_zenith_m(lat_deg, h_m):
    """Saastamoinen zenith delay (m): hydrostatic + wet, standard atmosphere at the site."""
    p, t, e = _standard_atmosphere(h_m)
    phi = math.radians(lat_deg)
    zhd = 0.0022768 * p / (1.0 - 0.00266 * math.cos(2.0 * phi) - 0.00028 * h_m / 1000.0)
    zwd = 0.002277 * (1255.0 / t + 0.05) * e
    return zhd + zwd


def tropo_mapping(el_deg):
    """Chao's mapping function: 1 at zenith, ~5.6 at 10 deg, ~10 at 5 deg."""
    el = math.radians(max(el_deg, 0.0))
    return 1.0 / (math.sin(el) + 0.00143 / (math.tan(el) + 0.0445))


def tropo_delay_m(el_deg, lat_deg, h_m):
    """Slant tropospheric delay (m) at elevation el_deg for a site at lat_deg, h_m."""
    return tropo_zenith_m(lat_deg, h_m) * tropo_mapping(el_deg)


def klobuchar_delay_m(
    alpha, beta, lat_deg, lon_deg, el_deg, az_deg, gps_tow_s, f_hz=F_L1
):
    """Klobuchar slant ionospheric delay (m) at carrier f_hz.

    alpha, beta: the four GPSA / GPSB coefficients from the nav header (s, s/semicircle, ...).
    gps_tow_s: GPS time of week (s). Angles in degrees; the algorithm works in semicircles.
    """
    phi_u = lat_deg / 180.0
    lam_u = lon_deg / 180.0
    el = max(el_deg, 0.0) / 180.0
    az = math.radians(az_deg)
    # earth-centred angle to the pierce point and its sub-point
    psi = 0.0137 / (el + 0.11) - 0.022
    phi_i = phi_u + psi * math.cos(az)
    phi_i = max(-0.416, min(0.416, phi_i))
    lam_i = lam_u + psi * math.sin(az) / math.cos(phi_i * math.pi)
    phi_m = phi_i + 0.064 * math.cos((lam_i - 1.617) * math.pi)
    # local time at the pierce point
    t = (43200.0 * lam_i + gps_tow_s) % 86400.0
    # slant factor
    f = 1.0 + 16.0 * (0.53 - el) ** 3
    per = sum(b * phi_m**n for n, b in enumerate(beta))
    if per < 72000.0:
        per = 72000.0
    amp = sum(a * phi_m**n for n, a in enumerate(alpha))
    if amp < 0.0:
        amp = 0.0
    x = 2.0 * math.pi * (t - 50400.0) / per
    if abs(x) < 1.57:
        t_iono = f * (5.0e-9 + amp * (1.0 - x * x / 2.0 + x**4 / 24.0))
    else:
        t_iono = f * 5.0e-9
    return C_LIGHT * t_iono * (F_L1 / f_hz) ** 2


F_B1I = 1561.098e6


def klobuchar_bds_delay_m(
    alpha, beta, lat_deg, lon_deg, el_deg, az_deg, bdt_sow, f_hz=F_B1I
):
    """BeiDou's Klobuchar slant delay (m) at f_hz from the D1 alpha/beta (BDS-SIS-ICD-B1I 2.0
    5.2.4.7). Unlike GPS's: a pierce point at 375 km on a sphere, its geographic latitude (no
    geomagnetic shift), the period capped at 172800 s, and the B1I delay scaled by
    (f_B1I/f)^2. bdt_sow: BeiDou time of week (s), GPS time of week less 14 s."""
    R, h = 6378.0, 375.0
    el = math.radians(max(el_deg, 0.0))
    az = math.radians(az_deg)
    phu = math.radians(lat_deg)
    psi = math.pi / 2 - el - math.asin(R / (R + h) * math.cos(el))
    phm = math.asin(
        math.sin(phu) * math.cos(psi) + math.cos(phu) * math.sin(psi) * math.cos(az)
    )
    lam = math.radians(lon_deg) + math.asin(
        math.sin(psi) * math.sin(az) / math.cos(phm)
    )
    x = abs(phm) / math.pi
    a2 = max(0.0, sum(a * x**n for n, a in enumerate(alpha)))
    a4 = min(172800.0, max(72000.0, sum(b * x**n for n, b in enumerate(beta))))
    t = (bdt_sow + lam * 43200.0 / math.pi) % 86400.0
    iz = 5.0e-9
    if abs(t - 50400.0) < a4 / 4.0:
        iz += a2 * math.cos(2.0 * math.pi * (t - 50400.0) / a4)
    mf = 1.0 / math.sqrt(1.0 - (R / (R + h) * math.cos(el)) ** 2)
    return C_LIGHT * mf * iz * (F_B1I / f_hz) ** 2


# BDGIM (BDS-SIS-ICD-B1C 1.0 / B2a 1.0, 7.8.2): the broadcast terms' (n, m) (Table 7-11), the
# predicted terms' (n, m), and Table 7-12's coefficients (TECu): row k of BDGIM_A is a_k (a_0
# first), row k-1 of BDGIM_B is b_k, at the periods BDGIM_T_DAYS[k-1].
# fmt: off
BDGIM_NM = ((0, 0), (1, 0), (1, 1), (1, -1), (2, 0), (2, 1), (2, -1), (2, 2), (2, -2))
BDGIM_NM0 = (
    (3, 0), (3, 1), (3, -1), (3, 2), (3, -2), (3, 3), (3, -3), (4, 0), (4, 1), (4, -1),
    (4, 2), (4, -2), (5, 0), (5, 1), (5, -1), (5, 2), (5, -2),
)
BDGIM_T_DAYS = (1, 0.5, 0.33, 14.6, 27, 121.6, 182.51, 365.25, 4028.71, 2014.35, 1342.9, 1007.18)
BDGIM_A = (
    (-0.61, -1.31, -2, -0.03, 0.15, -0.48, -0.4, 2.28, -0.16, -0.21, -0.1, -0.13, 0.21, 0.68, 1.06, 0, -0.12),
    (-0.51, -0.43, 0.34, -0.01, 0.17, 0.02, -0.06, 0.3, 0.44, -0.28, -0.31, -0.17, 0.04, 0.39, -0.12, 0.12, 0),
    (-0.06, -0.05, 0.06, 0.17, 0.15, 0, 0.11, -0.05, -0.16, 0.02, 0.11, 0.04, 0.12, 0.07, 0.02, -0.14, -0.14),
    (0.01, -0.03, 0.01, -0.01, 0.05, -0.03, 0.05, -0.03, -0.01, 0, -0.08, -0.04, 0, -0.02, -0.03, 0, -0.03),
    (-0.01, 0, 0.01, 0, 0.01, 0, -0.01, -0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0.03, 0.01, 0.02, 0.01, 0, -0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (-0.19, -0.02, 0.12, -0.1, 0.06, 0, -0.02, -0.08, -0.02, -0.07, 0.01, 0.03, 0.15, 0.06, -0.05, -0.03, -0.1),
    (-0.18, 0.06, -0.55, -0.02, 0.09, -0.08, 0, 0.86, -0.18, -0.05, -0.07, 0.04, 0.14, -0.03, 0.37, -0.11, -0.12),
    (1.09, -0.14, -0.21, 0.52, 0.27, 0, 0.11, 0.17, 0.23, 0.35, -0.05, 0.02, -0.6, 0.02, 0.01, 0.27, 0.32),
    (-0.34, -0.09, -1.22, 0.05, 0.15, -0.29, -0.17, 1.58, -0.06, -0.15, 0, 0.13, 0.28, -0.08, 0.62, -0.01, -0.04),
    (-0.13, 0.07, -0.37, 0.05, 0.06, -0.11, -0.07, 0.46, 0, -0.04, 0.01, 0.07, 0.09, -0.05, 0.15, -0.01, 0.01),
    (-0.06, 0.13, -0.07, 0.03, 0.02, -0.05, -0.05, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (-0.03, 0.08, -0.01, 0.04, 0.01, -0.02, -0.02, -0.04, 0, 0, 0, 0, 0, 0, 0, 0, 0),
)
BDGIM_B = (
    (0.23, -0.2, -0.31, 0.16, -0.03, 0.02, 0.04, 0.18, 0.34, 0.45, 0.19, -0.25, -0.12, 0.18, 0.4, -0.09, 0.21),
    (0.02, -0.08, -0.06, -0.11, 0.15, -0.14, 0.01, 0.01, 0.04, -0.14, -0.05, 0.08, 0.08, -0.01, 0.01, 0.11, -0.12),
    (0, -0.02, -0.03, -0.05, -0.01, -0.07, -0.03, -0.01, 0.02, -0.01, 0.03, -0.1, 0.01, 0.05, -0.01, 0.04, 0),
    (0, -0.02, 0.01, 0, -0.01, 0.01, 0, -0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0.01, 0, 0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (-0.09, 0.07, 0.03, 0.06, 0.09, 0.01, 0.02, 0, -0.04, -0.02, -0.01, 0.01, -0.1, 0, -0.01, 0.02, 0.05),
    (0.15, -0.31, 0.13, 0.05, -0.09, -0.03, 0.06, -0.36, 0.08, 0.05, 0.06, -0.02, -0.05, 0.06, -0.2, 0.04, 0.07),
    (0.5, -0.08, -0.38, 0.36, 0.14, 0.04, 0, 0.25, 0.17, 0.27, -0.03, -0.03, -0.32, -0.1, 0.2, 0.1, 0.3),
    (0, -0.11, -0.22, 0.01, 0.02, -0.03, -0.01, 0.49, -0.03, -0.02, 0.01, 0.02, 0.04, -0.04, 0.16, -0.02, -0.01),
    (0.05, 0.03, 0.07, 0.02, -0.01, 0.03, 0.02, -0.04, -0.01, -0.01, 0.02, 0.03, 0.02, -0.04, -0.04, -0.01, 0),
    (0.03, -0.02, 0.04, -0.01, -0.03, 0.02, 0.01, 0.04, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0.04, -0.02, -0.04, 0, -0.01, 0, 0.01, 0.07, 0, 0, 0, 0, 0, 0, 0, 0, 0),
)
# fmt: on
# the north geomagnetic pole the model uses
BDGIM_POLE = (math.radians(80.27), math.radians(-72.58))


def _legendre_norm(n, m, x):
    """BDGIM's normalised associated Legendre function P~(n, m) at x = sin(lat) (7-12, 7-13)."""
    p_mm = 1.0
    for k in range(1, 2 * m, 2):
        p_mm *= k
    p_mm *= (1.0 - x * x) ** (m / 2.0)
    p2, p1 = 0.0, p_mm
    for k in range(m + 1, n + 1):
        if k == m + 1:
            p2, p1 = p1, x * (2 * m + 1) * p_mm
        else:
            p2, p1 = p1, ((2 * k - 1) * x * p1 - (k + m - 1) * p2) / (k - m)
    norm = (
        math.factorial(n - m)
        * (2 * n + 1)
        * (1 if m == 0 else 2)
        / math.factorial(n + m)
    )
    return math.sqrt(norm) * p1


def _bdgim_terms(nm, phi, lam):
    return [
        _legendre_norm(abs(n), abs(m), math.sin(phi))
        * (math.cos(m * lam) if m >= 0 else math.sin(-m * lam))
        for n, m in nm
    ]


def bdgim_delay_m(alpha, lat_deg, lon_deg, el_deg, az_deg, t_unix, f_hz):
    """BeiDou's global model (BDGIM) slant delay (m) at f_hz from the nine broadcast
    coefficients alpha (TECu; B-CNAV2/B-CNAV3 type 30), with the 17 predicted terms from Table
    7-12. A single layer at 400 km, in a solar-fixed geomagnetic frame."""
    R, H = 6378.0, 400.0
    el = math.radians(max(el_deg, 0.0))
    az = math.radians(az_deg)
    phu, lau = math.radians(lat_deg), math.radians(lon_deg)
    phM, laM = BDGIM_POLE
    psi = math.pi / 2 - el - math.asin(R / (R + H) * math.cos(el))
    phg = math.asin(
        math.sin(phu) * math.cos(psi) + math.cos(phu) * math.sin(psi) * math.cos(az)
    )
    lag = lau + math.atan2(
        math.sin(psi) * math.sin(az) * math.cos(phu),
        math.cos(psi) - math.sin(phu) * math.sin(phg),
    )
    phm = math.asin(
        math.sin(phM) * math.sin(phg)
        + math.cos(phM) * math.cos(phg) * math.cos(lag - laM)
    )
    lam = math.atan2(
        math.cos(phg) * math.sin(lag - laM) * math.cos(phM),
        math.sin(phM) * math.sin(phm) - math.sin(phg),
    )
    mjd = t_unix / 86400.0 + 40587.0
    day = math.floor(mjd)
    slon = math.pi * (1.0 - 2.0 * (mjd - day))
    lam -= math.atan2(math.sin(slon - laM), math.sin(phM) * math.cos(slon - laM))
    # the predicted terms at the odd hour nearest the epoch
    tp = day + (2 * math.floor((mjd - day) * 12.0) + 1) / 24.0
    beta = list(BDGIM_A[0])
    for k, T in enumerate(BDGIM_T_DAYS):
        c, s = math.cos(2 * math.pi / T * tp), math.sin(2 * math.pi / T * tp)
        for j in range(17):
            beta[j] += BDGIM_A[k + 1][j] * c + BDGIM_B[k][j] * s
    vtec = sum(b * a for b, a in zip(beta, _bdgim_terms(BDGIM_NM0, phm, lam)))
    vtec += sum(a * t for a, t in zip(alpha, _bdgim_terms(BDGIM_NM, phm, lam)))
    mf = 1.0 / math.sqrt(1.0 - (R / (R + H) * math.cos(el)) ** 2)
    return mf * 40.28e16 / f_hz**2 * vtec


# Where nav_decode's payload files (--payload-file nav_payload_<name>.json) hold each broadcast
# model's coefficients, best first: (file name, signal, message, field labels).
_KLOB = ("α0", "α1", "α2", "α3", "β0", "β1", "β2", "β3")
_BDGIM = tuple("α%d" % i for i in range(1, 10))
DECODED_IONO = {
    "gps": (
        ("misc", "GPS_L5_CNAV", "type 30", _KLOB),
        ("l2c", "GPS_L2C_CNAV", "type 30", _KLOB),
    ),
    "bds": (("d1", "BDS_B3I_D1", "subframe 1", _KLOB),),
    "bdgim": (
        ("bcnav2", "BDS_B2A_BCNAV2", "type 30", _BDGIM),
        ("bcnav3", "BDS_B2B_BCNAV3", "type 30", _BDGIM),
    ),
    "gal": (
        ("inav", "GAL_E5BI_INAV", "word 5", ("a_i0", "a_i1", "a_i2")),
        ("fnav", "GAL_E5AI_FNAV", "page 1", ("a_i0", "a_i1", "a_i2")),
    ),
}


def decoded_iono(payload_dir):
    """Each broadcast model's coefficients as the satellites send them now, from nav_decode's
    payload files: {model: {"c": [floats], "src": signal, "t": decode time or None}}, the
    newest decode of the first source that has the message. "gps" and "bds" are the Klobuchar
    alpha and beta (s, s/semicircle^n), "bdgim" alpha1-9 (TECu), "gal" a_i0-a_i2."""
    import json
    import os

    out = {}
    for model, sources in DECODED_IONO.items():
        for name, signal, msg, labels in sources:
            try:
                with open(os.path.join(payload_dir, "nav_payload_%s.json" % name)) as f:
                    prns = json.load(f)["signals"][signal]["prns"]
            except (OSError, ValueError, KeyError):
                continue
            best = None
            for e in prns.values():
                m = (e.get("msgs") or {}).get(msg)
                if not m:
                    continue
                vals = {r[0]: r[1] for r in m[3]}
                if any(not isinstance(vals.get(k), (int, float)) for k in labels):
                    continue
                if best is None or (m[1] or 0) > (best["t"] or 0):
                    best = {
                        "c": [float(vals[k]) for k in labels],
                        "src": signal,
                        "t": m[1],
                    }
            if best:
                out[model] = best
                break
    return out


def parse_iono_corr(lines):
    """The IONOSPHERIC CORR header lines of a RINEX 3 nav file -> {"GPSA": [4 floats],
    "GPSB": [...], "GAL": [...], "BDSA": [...], ...}; the first occurrence of each wins.
    Stops at END OF HEADER."""
    out = {}
    for line in lines:
        if "END OF HEADER" in line:
            break
        if "IONOSPHERIC CORR" not in line:
            continue
        toks = line[:60].split()
        if len(toks) < 2:
            continue
        try:
            vals = [float(v.replace("D", "E")) for v in toks[1:5]]
        except ValueError:
            continue
        out.setdefault(toks[0], vals)
    return out


def iono_params(paths, cache_dir=None, write=True):
    """Klobuchar coefficients for the model: {"alpha", "beta", "src", "age_days"} from the first
    of `paths` (nav files, .gz or plain, best first) whose header carries GPSA/GPSB, which also
    refreshes `cache_dir`/iono_corr.json; else the cached set with its age (the coefficients
    change over weeks, so a stale set beats none); else None. cache_dir defaults to the BRDC
    cache; "" disables the cache. write=False reads the cache without refreshing it: the broker
    is the nav cache's one writer, the observables writers read."""
    import gzip
    import json
    import os
    import threading
    import time

    if cache_dir is None:
        try:
            from gnss_brdc_supply import CACHE as cache_dir
        except Exception:
            cache_dir = ""
    cpath = os.path.join(cache_dir, "iono_corr.json") if cache_dir else None
    for path in paths:
        try:
            op = gzip.open if str(path).endswith(".gz") else open
            with op(path, "rt", errors="replace") as f:
                head = []
                for line in f:
                    head.append(line)
                    if "END OF HEADER" in line or len(head) > 400:
                        break
        except Exception:
            continue
        ic = parse_iono_corr(head)
        if (
            "GPSA" in ic
            and "GPSB" in ic
            and len(ic["GPSA"]) == 4
            and len(ic["GPSB"]) == 4
        ):
            out = {
                "alpha": ic["GPSA"],
                "beta": ic["GPSB"],
                "src": os.path.basename(str(path)),
                "age_days": 0.0,
            }
            if cpath and write:
                try:
                    # a temp name unique per process and thread: the dead-reckon reloads run
                    # on one thread per chain, and a shared temp file would publish a torn one
                    tmp = "%s.tmp.%d.%d" % (cpath, os.getpid(), threading.get_ident())
                    with open(tmp, "w") as f:
                        json.dump(dict(out, t=time.time()), f)
                    os.replace(tmp, cpath)
                except Exception:
                    pass
            return out
    if cpath and os.path.exists(cpath):
        try:
            with open(cpath) as f:
                c = json.load(f)
            return {
                "alpha": c["alpha"],
                "beta": c["beta"],
                "src": "cache:" + c.get("src", "?"),
                "age_days": (time.time() - c["t"]) / 86400.0,
            }
        except Exception:
            return None
    return None


# -- global ionosphere maps (IONEX) ---------------------------------------------------------------

# CODE's daily maps, best first: final, rapid (the next day), today's prediction and yesterday's.
IONEX_PRODUCTS = ("FIN", "RAP", "P0D", "P1D")
# how often a day served by a lesser product looks for a better one
IONEX_RECHECK_S = 3600.0
_IONEX_CHECKED = {}  # (year, doy) -> wall time of the last look for a better product


class Ionex(object):
    """One IONEX file's TEC maps, interpolated as the format's description recommends (Schaer):
    bilinear in space, linear in time between the two maps around the epoch, each rotated by the
    Earth's turn since its own epoch so the ionosphere stays sun-fixed between them."""

    def __init__(self, path):
        import datetime
        import gzip

        op = gzip.open if str(path).endswith(".gz") else open
        self.maps = []  # [(unix t, {lat: [TECU over lon]})]
        exp, cur, lat, vals, in_map, t = -1, None, None, [], False, None
        with op(path, "rt") as f:
            for line in f:
                lab = line[60:].strip()
                if lab == "EXPONENT":
                    exp = int(line[:6])
                elif lab == "HGT1 / HGT2 / DHGT":
                    self.h_km = float(line[2:8])
                elif lab == "BASE RADIUS":
                    self.r_km = float(line[:8])
                elif lab == "LAT1 / LAT2 / DLAT":
                    self.lat1, self.lat2, self.dlat = (
                        float(line[i : i + 6]) for i in (2, 8, 14)
                    )
                elif lab == "LON1 / LON2 / DLON":
                    self.lon1, self.lon2, self.dlon = (
                        float(line[i : i + 6]) for i in (2, 8, 14)
                    )
                elif lab == "START OF TEC MAP":
                    in_map, cur, lat = True, {}, None
                elif lab == "END OF TEC MAP":
                    if lat is not None:
                        cur[lat] = vals
                    self.maps.append((t, cur))
                    in_map = False
                elif lab == "START OF RMS MAP":
                    break
                elif in_map and lab == "EPOCH OF CURRENT MAP":
                    y, mo, d, h, mi, s = (int(x) for x in line[:36].split())
                    t = datetime.datetime(
                        y, mo, d, h, mi, s, tzinfo=datetime.timezone.utc
                    ).timestamp()
                elif in_map and lab == "LAT/LON1/LON2/DLON/H":
                    if lat is not None:
                        cur[lat] = vals
                    lat, vals = float(line[2:8]), []
                elif in_map and lat is not None:
                    vals += [
                        float(v) * 10.0 ** exp if int(v) != 9999 else float("nan")
                        for v in line.split()
                    ]
        self.nlon = int(round((self.lon2 - self.lon1) / self.dlon)) + 1
        self.nlat = int(round((self.lat2 - self.lat1) / self.dlat)) + 1
        self.t0, self.t1 = self.maps[0][0], self.maps[-1][0]

    def _grid(self, m, lat, lon):
        lon = (lon + 180.0) % 360.0 - 180.0
        i = (lat - self.lat1) / self.dlat
        j = (lon - self.lon1) / self.dlon
        i0 = min(max(int(math.floor(i)), 0), self.nlat - 2)
        j0 = min(max(int(math.floor(j)), 0), self.nlon - 2)
        p, q = i - i0, j - j0
        a = m[self.lat1 + i0 * self.dlat]
        b = m[self.lat1 + (i0 + 1) * self.dlat]
        return (
            (1 - p) * (1 - q) * a[j0]
            + (1 - p) * q * a[j0 + 1]
            + p * (1 - q) * b[j0]
            + p * q * b[j0 + 1]
        )

    def vtec(self, t, lat, lon):
        """Vertical TEC (TECU) at the pierce point (lat, lon) and unix time t, or None outside
        the file's span."""
        if not self.t0 <= t <= self.t1:
            return None
        k = 0
        while k < len(self.maps) - 2 and self.maps[k + 1][0] <= t:
            k += 1
        (ta, ma), (tb, mb) = self.maps[k], self.maps[k + 1]
        ea = self._grid(ma, lat, lon + (t - ta) * 360.0 / 86400.0)
        eb = self._grid(mb, lat, lon + (t - tb) * 360.0 / 86400.0)
        return ((tb - t) * ea + (t - ta) * eb) / (tb - ta)

    def slant_delay_m(self, t, lat_deg, lon_deg, el_deg, az_deg, f_hz):
        """Slant ionospheric group delay (m) at f_hz, from the map's single layer: the VTEC at the
        pierce point times the thin-shell obliquity. None outside the file's span."""
        r, h = self.r_km, self.h_km
        e, a = math.radians(max(el_deg, 0.0)), math.radians(az_deg)
        la, lo = math.radians(lat_deg), math.radians(lon_deg)
        z = math.asin(r / (r + h) * math.cos(e))  # zenith angle at the pierce point
        psi = math.pi / 2.0 - e - z
        lat_p = math.asin(
            math.sin(la) * math.cos(psi) + math.cos(la) * math.sin(psi) * math.cos(a)
        )
        lon_p = lo + math.asin(math.sin(psi) * math.sin(a) / math.cos(lat_p))
        v = self.vtec(t, math.degrees(lat_p), math.degrees(lon_p))
        if v is None:
            return None
        return 40.3e16 * v / math.cos(z) / (f_hz * f_hz)


def ionex_path(t, cache_dir=None, fetch=True):
    """The best CODE global ionosphere map for the UTC day of unix time t: final, rapid, today's
    prediction or yesterday's (IONEX_PRODUCTS), from `cache_dir`/ionex, else fetched from
    CDDIS with the Earthdata token (gnss_brdc_supply). A day held only as a lesser product is
    looked up again at most every IONEX_RECHECK_S. The local path, or None."""
    import os
    import time
    import urllib.request

    from gnss_brdc_supply import CACHE, _earthdata_token

    d = time.gmtime(t)
    year, doy = d.tm_year, d.tm_yday
    folder = os.path.join(cache_dir or CACHE, "ionex")
    names = [
        "COD0OPS%s_%04d%03d0000_01D_01H_GIM.INX.gz" % (p, year, doy)
        for p in IONEX_PRODUCTS
    ]
    have = [n for n in names if os.path.exists(os.path.join(folder, n))]
    best = have[0] if have else None
    if best == names[0] or not fetch:
        return os.path.join(folder, best) if best else None
    if time.time() - _IONEX_CHECKED.get((year, doy), 0.0) < IONEX_RECHECK_S and best:
        return os.path.join(folder, best)
    _IONEX_CHECKED[(year, doy)] = time.time()
    tok = _earthdata_token()
    if tok:
        os.makedirs(folder, exist_ok=True)
        for n in names[: names.index(best) if best else len(names)]:
            url = "https://cddis.nasa.gov/archive/gnss/products/ionex/%04d/%03d/%s" % (
                year,
                doy,
                n,
            )
            try:
                req = urllib.request.Request(
                    url, headers={"Authorization": "Bearer " + tok}
                )
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
            except Exception:
                continue
            if data[:2] != b"\x1f\x8b":
                continue  # not a gzip: an error page
            tmp = os.path.join(folder, n + ".part")
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, os.path.join(folder, n))
            return os.path.join(folder, n)
    return os.path.join(folder, best) if best else None
