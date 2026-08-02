"""Single-point position (PVT self-survey) from the code observable.

Now that the code observable is metre-good (the REC_CP export-currency fix), the obs logs carry a
per-satellite `code_resid_m` = measured code range MINUS the model range predicted at the CONFIG
position, plus `az`/`el`. That residual is exactly a LINEARIZED single-point-position observation
about the config position:

    code_resid_i  ~=  -e_i . dx  +  c*dt_group  +  (iono_i + multipath_i + noise_i)

where e_i is the line-of-sight unit vector (rx -> sat, from az/el) and dx is the receiver-position
correction. The common receiver/dongle clock is a free parameter PER GROUP -- one group per
(constellation, band), since each band's dongle and each constellation carry a different clock/
inter-system bias. Least-squares over (dx, {dt_group}) recovers the position; the post-fit residual
scales the covariance -> a real 1-sigma error ellipse. Because the config position is the true site,
this is a SELF-SURVEY / validation: it should return the config position to within a few metres
(single-frequency, so iono is uncorrected and sets the floor), and the offset + error are the
"is the whole chain self-consistent" check.

Pure geometry; no network, no broker state. Solve per group and a combined best-fit.
"""

import math

import numpy as np

C_LIGHT = 299792458.0
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def _llh_to_ecef(lat, lon, alt):
    la, lo = math.radians(lat), math.radians(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(la) ** 2)
    return np.array([(n + alt) * math.cos(la) * math.cos(lo),
                     (n + alt) * math.cos(la) * math.sin(lo),
                     (n * (1.0 - WGS84_E2) + alt) * math.sin(la)])


def _ecef_to_llh(p):
    x, y, z = p
    lon = math.atan2(y, x)
    r = math.hypot(x, y)
    lat = math.atan2(z, r * (1.0 - WGS84_E2))
    for _ in range(6):
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
        alt = r / math.cos(lat) - n
        lat = math.atan2(z, r * (1.0 - WGS84_E2 * n / (n + alt)))
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    alt = r / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), alt


def _enu_axes(lat, lon):
    """Rows E, N, U (ECEF components) at the given geodetic lat/lon -- to turn an ENU
    position correction into ECEF and to project the LOS unit vector into ENU."""
    la, lo = math.radians(lat), math.radians(lon)
    sl, cl, so, co = math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)
    return np.array([[-so, co, 0.0],
                     [-sl * co, -sl * so, cl],
                     [cl * co, cl * so, sl]])


def _los_enu(az_deg, el_deg):
    """LOS unit vector (rx -> sat) in local ENU from az (from North, clockwise) and el."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)])


def _fit(rows, npar):
    """One weighted LS pass. rows: (los[3], resid, group_idx). Returns (x, post, Ninv) or None."""
    n = len(rows)
    if n < npar:
        return None
    H = np.zeros((n, npar))
    r = np.zeros(n)
    for i, (los, resid, g) in enumerate(rows):
        H[i, 0:3] = -los
        H[i, 3 + g] = 1.0
        r[i] = resid
    try:
        Ninv = np.linalg.inv(H.T @ H)
    except np.linalg.LinAlgError:
        return None
    x = Ninv @ (H.T @ r)
    return x, r - H @ x, Ninv


def _solve(rows, n_groups, group_of, reject_floor=60.0, reject_k=5.0):
    """Robust LS with RAIM-style outlier rejection. The code residuals carry gross outliers
    (bad-tracking sats / wrong sub-code-period ambiguity, tens of km off a clean ~20 m cluster),
    so drop the single worst satellite while its post-fit residual exceeds
    max(reject_floor, reject_k * median|post|) and redundancy remains. Returns
    (dx[3], clocks[n_groups], resid_rms, sigma_diag, Ninv, n_used, n_rej) or None."""
    npar = 3 + n_groups
    kept = list(rows)
    n_rej = 0
    while True:
        f = _fit(kept, npar)
        if f is None:
            return None
        x, post, Ninv = f
        ap = np.abs(post)
        w = int(np.argmax(ap))
        thr = max(reject_floor, reject_k * float(np.median(ap)))
        if ap[w] > thr and len(kept) > npar + 2:
            kept.pop(w)
            n_rej += 1
            continue
        break
    dof = max(1, len(kept) - npar)
    sigma0 = math.sqrt(float(post @ post) / dof)
    cov = sigma0 ** 2 * Ninv
    return (x[0:3], x[3:], sigma0, np.sqrt(np.clip(np.diag(cov), 0.0, None)), Ninv,
            len(kept), n_rej)


def _dops(Ninv3):
    """PDOP/HDOP/VDOP from the position block of (H^T H)^-1 (unit-weighted geometry)."""
    d = np.clip(np.diag(Ninv3), 0.0, None)
    return dict(pdop=float(math.sqrt(d[0] + d[1] + d[2])),
                hdop=float(math.sqrt(d[0] + d[1])),
                vdop=float(math.sqrt(d[2])))


def solve(measurements, lat0, lon0, alt0, min_el_deg=10.0):
    """PVT self-survey. `measurements`: iterable of dicts {group, az, el, resid_m} where `group`
    is a (constellation, band) label. Returns {groups: {group: result}, combined: result|None}.

    Each result: n_sats, position (lat/lon/alt + ECEF), offset dENU (m) from the config a-priori,
    clock (m), resid_rms_m, sigma_enu (1-sigma m), dops. `combined` solves ALL groups jointly with
    one clock per group -- the best-fit position + error."""
    apr = _llh_to_ecef(lat0, lon0, alt0)
    R = _enu_axes(lat0, lon0)   # rows E,N,U in ECEF
    # bucket usable measurements by group
    by_group = {}
    for m in measurements:
        az, el, res = m.get("az"), m.get("el"), m.get("resid_m")
        g = m.get("group")
        if az is None or el is None or res is None or g is None or el < min_el_deg:
            continue
        by_group.setdefault(g, []).append((_los_enu(az, el), float(res)))
    # GROSS pre-filter: the good satellites cluster within ~tens of metres of the group median
    # while bad ones (wrong sub-code-period ambiguity / mislock) sit km away. Cut those against
    # the robust median BEFORE the LS, so even a thin group (too few sats to reject in-fit) is
    # not wrecked by an outlier the iterative RAIM cannot afford to drop.
    for g in list(by_group):
        obs = by_group[g]
        if len(obs) >= 4:
            med = float(np.median([o[1] for o in obs]))
            by_group[g] = [o for o in obs if abs(o[1] - med) < 1000.0]

    def _ok(Ninv, sig_diag):
        # Reject a degenerate / ill-conditioned geometry (e.g. all sats at one elevation ->
        # the vertical is unconstrained) rather than publish a garbage position.
        pd = math.sqrt(max(0.0, sum(np.clip(np.diag(Ninv[0:3, 0:3]), 0.0, None))))
        return pd < 30.0 and float(np.max(sig_diag[0:3])) < 1000.0

    def _pack(dx_enu, clock_m, sigma0, sig_diag, Ninv, n_used, n_rej):
        pos = apr + R.T @ dx_enu           # ENU correction -> ECEF
        lat, lon, alt = _ecef_to_llh(pos)
        return dict(n_sats=n_used, n_rejected=n_rej,
                    lat=lat, lon=lon, alt=alt, ecef=pos.tolist(),
                    d_e=float(dx_enu[0]), d_n=float(dx_enu[1]), d_u=float(dx_enu[2]),
                    clock_m=float(clock_m), resid_rms_m=float(sigma0),
                    sigma_e=float(sig_diag[0]), sigma_n=float(sig_diag[1]),
                    sigma_u=float(sig_diag[2]),
                    **_dops(Ninv[0:3, 0:3]))

    out = {"groups": {}, "combined": None}
    for g, obs in sorted(by_group.items()):
        rows = [(los, res, 0) for los, res in obs]
        s = _solve(rows, 1, {0: g})
        if s:
            dx, clk, sig0, sd, Ninv, nu, nr = s
            if _ok(Ninv, sd):
                out["groups"][g] = _pack(dx, clk[0], sig0, sd, Ninv, nu, nr)

    # combined: one clock column per group
    groups = sorted(by_group)
    gi = {g: i for i, g in enumerate(groups)}
    rows = [(los, res, gi[g]) for g in groups for los, res in by_group[g]]
    s = _solve(rows, len(groups), gi)
    if s and _ok(Ninv=s[4], sig_diag=s[3]):
        dx, clks, sig0, sd, Ninv, nu, nr = s
        c = _pack(dx, 0.0, sig0, sd, Ninv, nu, nr)
        c["clock_m"] = None
        c["clocks_m"] = {g: float(clks[gi[g]]) for g in groups}
        c["n_groups"] = len(groups)
        out["combined"] = c
    return out
