"""Galileo's broadcast ionosphere model NeQuick-G, written from its specification alone:
"European GNSS (Galileo) Open Service - Ionospheric Correction Algorithm for Galileo Single
Frequency Users", Issue 1.2, September 2016 (section numbers and equations below are its own).

The model's data (12 CCIR foF2 / M(3000)F2 coefficient files and the MODIP grid, Annex C) are
attached to the specification PDF. On first use they are fetched from the PDF at _SPEC_URL into
data_dir (default <BRDC cache>/nequick_g_cr) and checked against the PDF's MD5 checksums.

Where section 2 and Annex F differ, Annex F's smooth joins are used (foF1, and the join on the
E amplitude also when there is no F1 layer), as section 2.5.5.2's note asks; the ray is
integrated with Annex F's adaptive Kronrod G7-K15 rule (section 2.5.8.3). As the specification
says, geodetic coordinates are used as if they were spherical, on an Earth of radius 6371.2 km.
"""

import math
import os

import numpy as np

_SPEC_URL = (
    "https://www.gsc-europa.eu/sites/default/files/sites/all/files/"
    "Galileo_Ionospheric_Model.pdf"
)
_FILES = ["modipNeQG_wrapped.txt"] + ["ccir%d.txt" % (m + 10) for m in range(1, 13)]
_DATA = {}  # data_dir -> (MODIP grid, foF2 and M(3000)F2 coefficients by month)

_RE = 6371.2  # km, Table 2
_CHI0 = 86.23292796211615  # deg, Eq. 28
_HME = 120.0  # km, Eq. 78

# Kronrod G7-K15 nodes and weights (Annex F.2.6.1); the G7 nodes are the odd K15 nodes
# fmt: off
_XK = np.array([
    -0.991455371120812639206854697526329, -0.949107912342758524526189684047851,
    -0.864864423359769072789712788640926, -0.741531185599394439863864773280788,
    -0.586087235467691130294144838258730, -0.405845151377397166906606412076961,
    -0.207784955007898467600689403773245, 0.0,
    0.207784955007898467600689403773245, 0.405845151377397166906606412076961,
    0.586087235467691130294144838258730, 0.741531185599394439863864773280788,
    0.864864423359769072789712788640926, 0.949107912342758524526189684047851,
    0.991455371120812639206854697526329,
])
_WK = np.array([
    0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
    0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
    0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
    0.204432940075298892414161999234649, 0.209482141084727828012999174891714,
    0.204432940075298892414161999234649, 0.190350578064785409913256402421014,
    0.169004726639267902826583426598550, 0.140653259715525918745189590510238,
    0.104790010322250183839876322541518, 0.063092092629978553290700663189204,
    0.022935322010529224963732008058970,
])
_WG = np.array([
    0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
    0.381830050505118944950369775488975, 0.417959183673469387755102040816327,
    0.381830050505118944950369775488975, 0.279705391489276667901467771423780,
    0.129484966168869693270611432679082,
])
# fmt: on
# levels of splitting allowed (the specification leaves it to the implementation): Annex E's
# rays need 6; the rare rays that would split further change by < 0.1 mm
_KRONROD_LEVELS = 8


def _legendre_terms(q):
    """For each coefficient of a CCIR map, in the order of Eq. 61-67 (foF2, q = Eq. 63) or
    Eq. 69-75 (M(3000)F2, q = Eq. 71): the power of sin(MODIP), the longitude order (also the
    power of cos(lat)) and the column of cos (0..8) or sin (9..17) of that order."""
    terms = [(k, 0, 0) for k in range(q[0])]
    for n in range(1, len(q)):
        for k in range(q[n]):
            terms += [(k, n, n), (k, n, n + 9)]
    return np.array(terms).T


_F2_TERMS = _legendre_terms((12, 12, 9, 5, 2, 1, 1, 1, 1))
_FM3_TERMS = _legendre_terms((7, 8, 6, 3, 2, 1, 1))


def _fetch(data_dir):
    """Save the 13 files attached to the specification PDF (Annex C) into data_dir. Each is a
    Flate stream named by its /Filespec and carries its MD5 as /CheckSum."""
    import hashlib
    import re
    import urllib.request
    import zlib

    # the server refuses urllib's default User-Agent
    req = urllib.request.Request(_SPEC_URL, headers={"User-Agent": "gnss_nequick"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            pdf = r.read()
    except OSError as e:
        raise RuntimeError(
            "NeQuick-G data missing from %s and %s could not be fetched: %s"
            % (data_dir, _SPEC_URL, e)
        )
    os.makedirs(data_dir, exist_ok=True)
    for num, name in re.findall(rb"/EF<</F (\d+) 0 R>>/F\(([\w.]+)\)", pdf):
        m = re.search(rb"[\r\n]%s 0 obj\s*<<(.*?)>>\s*stream\r?\n" % num, pdf, re.S)
        n = int(re.search(rb"/Length (\d+)", m.group(1)).group(1))
        data = zlib.decompress(pdf[m.end() : m.end() + n])
        md5 = re.search(rb"/CheckSum<(\w+)>", m.group(1)).group(1).decode().lower()
        name = name.decode()
        if hashlib.md5(data).hexdigest() != md5:
            raise RuntimeError("%s: checksum mismatch in %s" % (name, _SPEC_URL))
        path = os.path.join(data_dir, name)
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)


def _load(data_dir):
    """(MODIP grid [39, 39], foF2 coefficients [month, activity, 76, 13], M(3000)F2
    coefficients [month, activity, 49, 9]), read once per process (2.5.3)."""
    if data_dir is None:
        from gnss_brdc_supply import CACHE

        data_dir = os.path.join(CACHE, "nequick_g_cr")
    d = _DATA.get(data_dir)
    if d is None:
        if not all(os.path.exists(os.path.join(data_dir, n)) for n in _FILES):
            _fetch(data_dir)
        vals = []
        for n in _FILES:
            with open(os.path.join(data_dir, n)) as f:
                vals.append(np.array(f.read().split(), dtype=float))
        ccir = np.array(vals[1:])
        d = (
            vals[0].reshape(39, 39),
            ccir[:, :1976].reshape(12, 2, 76, 13),
            ccir[:, 1976:].reshape(12, 2, 49, 9),
        )
        _DATA[data_dir] = d
    return d


def _clipexp(x):
    """exp clipped at +-80 (Annex F.2.12.2)."""
    return np.exp(np.maximum(np.minimum(x, 80.0), -80.0))


def _join(f1, f2, alpha, x):
    """f1 for x >> 0 and f2 for x << 0, joined smoothly over a width 1/alpha (F.2.12.1)."""
    ee = _clipexp(alpha * x)
    return (f1 * ee + f2) / (ee + 1.0)


def _epstein(x, y, z, w):
    """Epstein layer of peak x/4 at height y and thickness z, at height w (Eq. 3)."""
    e = _clipexp((w - y) / z)
    return x * e / (1.0 + e) ** 2


def _interp(z1, z2, z3, z4, x):
    """Third-order interpolation through (-1, z1), (0, z2), (1, z3), (2, z4) at x (2.5.7.1)."""
    d = 2.0 * x - 1.0
    g1, g2, g3, g4 = z3 + z2, z3 - z2, z4 + z1, (z4 - z1) / 3.0
    zx = (9 * g1 - g3 + (9 * g2 - g4) * d + (g3 - g1) * d**2 + (g4 - g2) * d**3) / 16
    return np.where(np.abs(2.0 * x) < 1e-10, z2, zx)


def _modip(grid, lat, lon):
    """MODIP (deg) at arrays lat, lon (deg) from the wrapped grid (2.5.4.3): row r is latitude
    5 r - 95, column c longitude 10 c - 190. Latitude +-90 gives +-90."""
    a = (np.clip(lat, -90.0, 90.0) + 90.0) / 5.0 + 1.0
    b = ((lon + 180.0) % 360.0) / 10.0
    i = np.clip(np.floor(a).astype(int) - 2, -1, 34)
    j = np.clip(np.floor(b).astype(int) - 2, -2, 33)
    rows = i[:, None, None] + np.arange(1, 5)[:, None]
    cols = j[:, None, None] + np.arange(2, 6)
    z = grid[rows, cols]
    zk = _interp(z[:, 0], z[:, 1], z[:, 2], z[:, 3], (a - i - 2)[:, None])
    return _interp(zk[:, 0], zk[:, 1], zk[:, 2], zk[:, 3], b - j - 2)


class _Ray(object):
    """The model along one straight ray from (lat1, lon1, h1) to (lat2, lon2, h2) (deg, km):
    the ray's geometry (2.5.8.2) and the constants of its time and Az, and the electron density
    at distance s from the ray perigee (or height s on a vertical ray)."""

    def __init__(self, ai, month, ut, lat1, lon1, h1, lat2, lon2, h2, data):
        self.grid, f2, fm3 = data
        self.month, self.ut = month, ut
        # Az at the receiver's MODIP, held along the ray (2.5.4.4, 3.1, 3.3), and R12 (Eq. 19)
        mu = float(_modip(self.grid, np.array([lat1]), np.array([lon1]))[0])
        az = sum(a * mu**i for i, a in enumerate(ai)) if any(ai) else 63.7
        self.az = min(max(az, 0.0), 400.0)
        self.r12 = math.sqrt(167273.0 + (self.az - 63.7) * 1123.6) - 408.99
        # the CCIR maps at R12 (Eq. 44, 46) and their Fourier series in UT (Eq. 49-51)
        w = self.r12 / 100.0
        af2 = f2[month - 1, 0] * (1.0 - w) + f2[month - 1, 1] * w
        am3 = fm3[month - 1, 0] * (1.0 - w) + fm3[month - 1, 1] * w
        t = math.radians(15.0 * ut - 180.0)
        harm = [1.0]
        for k in range(1, 7):
            harm += [math.sin(k * t), math.cos(k * t)]
        self.cf2 = af2 @ harm
        self.cm3 = am3 @ harm[:9]
        # solar declination (Eq. 20-25)
        tday = 30.5 * month - 15.0 + (18.0 - ut) / 24.0
        am = math.radians(0.9856 * tday - 3.289)
        al = am + math.radians(
            1.916 * math.sin(am) + 0.020 * math.sin(2 * am) + 282.634
        )
        self.sdec = 0.39782 * math.sin(al)
        self.cdec = math.sqrt(1.0 - self.sdec**2)

        # zenith angle and perigee of the ray (2.5.8.2.2-2.5.8.2.4, F.2.5)
        self.lat1, self.lon1 = lat1, lon1
        r1, r2 = _RE + h1, _RE + h2
        self.rp, self.zeta = 0.0, 0.0
        if abs(lat2 - lat1) >= 1e-5 or abs(lon2 - lon1) >= 1e-5:
            p1, p2 = math.radians(lat1), math.radians(lat2)
            dl = math.radians(lon2 - lon1)
            cdel = math.sin(p1) * math.sin(p2)
            cdel += math.cos(p1) * math.cos(p2) * math.cos(dl)
            sdel = math.sqrt(max(0.0, 1.0 - cdel**2))
            zeta = math.atan2(sdel, cdel - r1 / r2)
            self.rp, self.zeta = r1 * math.sin(zeta), math.degrees(zeta)
        if self.rp < 0.1:
            return
        if abs(abs(lat1) - 90.0) < 1e-10:
            php = math.copysign(zeta, lat1)
            lamp = math.radians(lon2) + (math.pi if zeta >= 0 else 0.0)
        else:
            ssig = math.sin(dl) * math.cos(p2) / sdel
            csig = (math.sin(p2) - cdel * math.sin(p1)) / (sdel * math.cos(p1))
            delp = math.pi / 2 - zeta
            sphp = math.sin(p1) * math.cos(delp) - math.cos(p1) * math.sin(delp) * csig
            cphp = math.sqrt(max(0.0, 1.0 - sphp**2))
            php = math.atan2(sphp, cphp)
            slam = -ssig * math.sin(delp) / cphp
            clam = (math.cos(delp) - math.sin(p1) * sphp) / (math.cos(p1) * cphp)
            lamp = math.atan2(slam, clam) + math.radians(lon1)
        self.sphp, self.cphp, self.lonp = math.sin(php), math.cos(php), lamp
        if abs(abs(math.degrees(php)) - 90.0) < 1e-10:
            self.ssigp, self.csigp = 0.0, (-1.0 if php > 0 else 1.0)
        else:
            dl = math.radians(lon2) - lamp
            cpsi = self.sphp * math.sin(p2) + self.cphp * math.cos(p2) * math.cos(dl)
            spsi = math.sqrt(max(0.0, 1.0 - cpsi**2))
            self.ssigp = math.cos(p2) * math.sin(dl) / spsi
            self.csigp = (math.sin(p2) - self.sphp * cpsi) / (self.cphp * spsi)

    def s_at(self, h):
        """The integration variable at height h (km): distance from the perigee, or h."""
        if self.rp < 0.1:
            return h
        return math.sqrt((_RE + h) ** 2 - self.rp**2)

    def ne_at(self, s):
        """Electron density (m^-3) at an array of distances s (km) along the ray (2.5.8.2.6)."""
        if self.rp < 0.1:
            return self.ne(s, np.full(len(s), self.lat1), np.full(len(s), self.lon1))
        h = np.sqrt(s**2 + self.rp**2) - _RE
        tand = s / self.rp
        cd = 1.0 / np.sqrt(1.0 + tand**2)
        sd = tand * cd
        sphs = self.sphp * cd + self.cphp * sd * self.csigp
        lat = np.degrees(np.arctan2(sphs, np.sqrt(np.maximum(0.0, 1.0 - sphs**2))))
        dlon = np.arctan2(sd * self.ssigp * self.cphp, cd - self.sphp * sphs)
        return self.ne(h, lat, np.degrees(dlon + self.lonp))

    def ne(self, h, lat, lon):
        """Electron density (m^-3) at arrays of height h (km) and lat, lon (deg): the profile
        parameters of 2.5.5 with Annex F.2.9.1's joins, then 2.5.6."""
        # foF2 and M(3000)F2 from the CCIR maps (Eq. 52-76)
        phi = np.radians(lat)
        sm = np.sin(np.radians(_modip(self.grid, lat, lon)))[:, None] ** np.arange(12)
        cp = np.cos(phi)[:, None] ** np.arange(9)
        nl = np.radians(lon)[:, None] * np.arange(9)
        tr = np.hstack((np.cos(nl), np.sin(nl)))
        p, n, c = _F2_TERMS
        fof2 = (sm[:, p] * cp[:, n] * tr[:, c]) @ self.cf2
        p, n, c = _FM3_TERMS
        m3 = (sm[:, p] * cp[:, n] * tr[:, c]) @ self.cm3

        # solar zenith angle (Eq. 4, 26-29) and foE (Eq. 30-35)
        lt = self.ut + lon / 15.0
        cchi = np.sin(phi) * self.sdec + np.cos(phi) * self.cdec * np.cos(
            np.pi / 12.0 * (12.0 - lt)
        )
        chi = np.degrees(np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - cchi**2)), cchi))
        chieff = _join(90.0 - 0.24 * _clipexp(20.0 - 0.2 * chi), chi, 12.0, chi - _CHI0)
        seas = (-1, -1, 0, 0, 1, 1, 1, 1, 0, 0, -1, -1)[self.month - 1]
        ee = _clipexp(0.3 * lat)
        seasp = seas * (ee - 1.0) / (ee + 1.0)
        foe = np.sqrt(
            (1.112 - 0.019 * seasp) ** 2
            * math.sqrt(self.az)
            * np.cos(np.radians(chieff)) ** 0.6
            + 0.49
        )
        # foF1: 1.4 foE by day, 0 at night, less 15 % near foF2 (Eq. 37 as in F.2.9.1)
        fof1 = _join(1.4 * foe, 0.0, 1000.0, foe - 2.0)
        fof1 = _join(0.0, fof1, 1000.0, foe - fof1)
        fof1 = _join(fof1, 0.85 * fof1, 60.0, 0.85 * fof2 - fof1)
        fof1 = np.where(fof1 < 1e-6, 0.0, fof1)
        nmf2, nmf1, nme = 0.124 * fof2**2, 0.124 * fof1**2, 0.124 * foe**2

        # peak heights and thicknesses (Eq. 79-89)
        r = fof2 / foe
        dm = 0.253 / (_join(r, 1.75, 20.0, r - 1.75) - 1.215) - 0.012
        mf = m3 * np.sqrt((0.0196 * m3**2 + 1.0) / (1.2967 * m3**2 - 1.0))
        hmf2 = 1490.0 * mf / (m3 + dm) - 176.0
        hmf1 = (hmf2 + _HME) / 2.0
        b2bot = (
            0.385
            * nmf2
            / (0.01 * np.exp(-3.467 + 0.857 * np.log(fof2**2) + 2.02 * np.log(m3)))
        )
        b1top = 0.3 * (hmf2 - hmf1)
        b1bot = 0.5 * (hmf1 - _HME)
        betop = np.maximum(b1bot, 7.0)

        # amplitudes (Eq. 90-98)
        a1 = 4.0 * nmf2
        day = fof1 >= 0.5
        a2a, a3a = 0.0, 4.0 * nme
        if day.any():
            for _ in range(5):
                a2a = 4.0 * (
                    nmf1
                    - _epstein(a1, hmf2, b2bot, hmf1)
                    - _epstein(a3a, _HME, betop, hmf1)
                )
                a2a = _join(a2a, 0.8 * nmf1, 1.0, a2a - 0.8 * nmf1)
                a3a = 4.0 * (
                    nme
                    - _epstein(a2a, hmf1, b1bot, _HME)
                    - _epstein(a1, hmf2, b2bot, _HME)
                )
        a2 = np.where(day, a2a, 0.0)
        a3a = np.where(day, a3a, 4.0 * (nme - _epstein(a1, hmf2, b2bot, _HME)))
        a3 = _join(a3a, 0.05, 60.0, a3a - 0.005)

        # topside thickness H0 (Eq. 99-106)
        if 4 <= self.month <= 9:
            ka = 6.705 - 0.014 * self.r12 - 0.008 * hmf2
        else:
            ka = -7.77 + 0.097 * (hmf2 / b2bot) ** 2 + 0.153 * nmf2
        kb = _join(ka, 2.0, 1.0, ka - 2.0)
        ha = _join(8.0, kb, 1.0, kb - 8.0) * b2bot
        x = (ha - 150.0) / 100.0
        h0 = ha / ((0.041163 * x - 0.183981) * x + 1.424472)

        # bottomside (2.5.6.1), evaluated at hmF2 for points above it
        hb = np.maximum(np.minimum(h, hmf2), 100.0)
        bf1 = np.where(h > hmf1, b1top, b1bot)
        be = np.where(h > _HME, betop, 5.0)
        f = np.exp(10.0 / (1.0 + np.abs(hb - hmf2)))
        alpha = np.array(
            [(hb - hmf2) / b2bot, (hb - hmf1) / bf1 * f, (hb - _HME) / be * f]
        )
        ea = np.exp(np.maximum(np.minimum(alpha, 25.0), -25.0))
        near = np.abs(alpha) <= 25.0
        s = np.where(near, np.array([a1, a2, a3]) * ea / (1.0 + ea) ** 2, 0.0)
        ds = np.where(near, (1.0 - ea) / (np.array([b2bot, bf1, be]) * (1.0 + ea)), 0.0)
        bc = 1.0 - 10.0 * (s * ds).sum(0) / s.sum(0)
        z = (h - 100.0) / 10.0
        low = np.where(h < 100.0, np.exp(1.0 - bc * z - np.exp(-z)), 1.0)
        nbot = s.sum(0) * low * 1e11

        # topside (2.5.6.2), with g = 0.125 and r = 100
        dh = np.maximum(h - hmf2, 0.0)
        ea = _clipexp(
            dh / (h0 * (1.0 + 100.0 * 0.125 * dh / (100.0 * h0 + 0.125 * dh)))
        )
        ntop = np.where(ea > 1e11, 4.0 * nmf2 / ea, 4.0 * nmf2 * ea / (1.0 + ea) ** 2)
        return np.where(h > hmf2, ntop * 1e11, nbot)

    def integrate(self, edges, tols):
        """Integral of the electron density (m^-3 km) over the segments between edges along
        the ray, each to its tolerance: Annex F.2.6.1's adaptive Kronrod G7-K15, with the
        intervals of one level of splitting evaluated together."""
        lo, hi, tol = np.array(edges[:-1]), np.array(edges[1:]), np.array(tols)
        total = 0.0
        for level in range(_KRONROD_LEVELS + 1):
            half, mid = (hi - lo) / 2.0, (hi + lo) / 2.0
            y = self.ne_at((half[:, None] * _XK + mid[:, None]).ravel()).reshape(-1, 15)
            intk = half * (y @ _WK)
            err = np.abs(intk - half * (y[:, 1::2] @ _WG))
            done = (
                (err <= tol * np.abs(intk)) | (err <= tol) | (level == _KRONROD_LEVELS)
            )
            total += intk[done].sum()
            lo, mid, hi, tol = lo[~done], mid[~done], hi[~done], tol[~done]
            lo, hi, tol = np.append(lo, mid), np.append(mid, hi), np.append(tol, tol)
            if not len(lo):
                break
        return total


def stec_tecu(
    ai,
    month,
    ut_hours,
    rx_lat_deg,
    rx_lon_deg,
    rx_h_m,
    sat_lat_deg,
    sat_lon_deg,
    sat_h_m,
    data_dir=None,
):
    """Slant TEC (TECU) from the receiver to the satellite (geodetic deg, m) in month (1-12)
    at ut_hours (UTC), for the broadcast coefficients ai = (ai0, ai1, ai2). All zero means
    the default Az of 63.7 (3.1). ValueError for a ray that passes through the Earth."""
    h1, h2 = rx_h_m / 1000.0, sat_h_m / 1000.0
    ray = _Ray(
        ai,
        month,
        ut_hours,
        rx_lat_deg,
        rx_lon_deg,
        h1,
        sat_lat_deg,
        sat_lon_deg,
        h2,
        _load(data_dir),
    )
    if abs(ray.zeta) > 90.0 and ray.rp < _RE:
        raise ValueError("NeQuick-G: the ray to the satellite passes through the Earth")
    # break points at 1000 and 2000 km, from the surface if the receiver is below it (F.2.3)
    hs = [max(h1, 0.0)] + [b for b in (1000.0, 2000.0) if h1 < b < h2] + [h2]
    tols = [0.001 if h <= 1000.0 else 0.01 for h in hs[1:]]
    return ray.integrate([ray.s_at(h) for h in hs], tols) * 1e-13


def slant_delay_m(
    ai,
    t_unix,
    rx_lat_deg,
    rx_lon_deg,
    rx_h_m,
    el_deg,
    az_deg,
    sat_h_m,
    f_hz,
    data_dir=None,
):
    """NeQuick-G slant group delay (m) at f_hz (Eq. 1) for a satellite seen at el_deg, az_deg
    at unix time t_unix. The satellite is placed on that line of sight at height sat_h_m, in
    the model's spherical frame, so the ray's zenith angle there is 90 - el. Elevations below
    0 count as 0."""
    import time

    month = time.gmtime(t_unix).tm_mon
    ut = (t_unix % 86400.0) / 3600.0
    el = math.radians(max(el_deg, 0.0))
    az = math.radians(az_deg)
    la = math.radians(rx_lat_deg)
    r1, r2 = _RE + rx_h_m / 1000.0, _RE + sat_h_m / 1000.0
    psi = math.pi / 2.0 - el - math.asin(r1 / r2 * math.cos(el))
    sl2 = math.sin(la) * math.cos(psi) + math.cos(la) * math.sin(psi) * math.cos(az)
    lat2 = math.asin(sl2)
    dlon = math.atan2(
        math.sin(az) * math.sin(psi) * math.cos(la), math.cos(psi) - math.sin(la) * sl2
    )
    stec = stec_tecu(
        ai,
        month,
        ut,
        rx_lat_deg,
        rx_lon_deg,
        rx_h_m,
        math.degrees(lat2),
        rx_lon_deg + math.degrees(dlon),
        sat_h_m,
        data_dir,
    )
    return 40.3e16 / (f_hz * f_hz) * stec


def orbit_height_m(sysc, prn):
    """Nominal orbit height (m) of a constellation's satellites: BeiDou's GEO and IGSO at
    35786 km, its MEO at 21528 km; "W" is a LEO constellation at 1100 km."""
    if sysc == "C":
        if 1 <= prn <= 10 or prn in (13, 16) or 59 <= prn <= 63:
            return 35786e3
        return 21528e3
    return {"G": 20200e3, "E": 23222e3, "R": 19100e3, "S": 35786e3, "W": 1100e3}.get(
        sysc, 20200e3
    )
