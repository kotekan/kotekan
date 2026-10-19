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
    gps_tow_s: GPS time of week (s). Angles in degrees; the algorithm works in semicircles."""
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
    per = sum(b * phi_m ** n for n, b in enumerate(beta))
    if per < 72000.0:
        per = 72000.0
    amp = sum(a * phi_m ** n for n, a in enumerate(alpha))
    if amp < 0.0:
        amp = 0.0
    x = 2.0 * math.pi * (t - 50400.0) / per
    if abs(x) < 1.57:
        t_iono = f * (5.0e-9 + amp * (1.0 - x * x / 2.0 + x ** 4 / 24.0))
    else:
        t_iono = f * 5.0e-9
    return C_LIGHT * t_iono * (F_L1 / f_hz) ** 2


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


def iono_params(paths, cache_dir=None):
    """Klobuchar coefficients for the model: {"alpha", "beta", "src", "age_days"} from the first
    of `paths` (nav files, .gz or plain, best first) whose header carries GPSA/GPSB, which also
    refreshes `cache_dir`/iono_corr.json; else the cached set with its age (the coefficients
    change over weeks, so a stale set beats none); else None. cache_dir defaults to the BRDC
    cache; "" disables the cache."""
    import gzip
    import json
    import os
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
            if cpath:
                try:
                    tmp = cpath + ".tmp"
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
