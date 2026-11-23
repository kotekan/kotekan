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
