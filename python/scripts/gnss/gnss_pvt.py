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
    is a (constellation, band) label. A group whose label ends in "-IF" is an IONO-FREE
    (dual-frequency) group; those are solved together into `combined_if`, and the single-frequency
    groups into `combined`, so the two never mix (their clock biases differ and an IF sat is
    correlated with its own single-freq rows). Returns {groups: {group: result}, combined,
    combined_if}.

    Each result: n_sats, position (lat/lon/alt + ECEF), offset dENU (m) from the config a-priori,
    clock (m), resid_rms_m, sigma_enu (1-sigma m), dops. `combined`/`combined_if` solve their
    groups jointly with one clock per group -- the best-fit position + error. `combined_if` is the
    few-metre iono-free answer; `combined` the single-frequency (iono-limited) one for comparison."""
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

    # combined: one clock column per group. Solve single-frequency groups and iono-free (-IF)
    # groups SEPARATELY -- an IF group's clock folds two dongle clocks together and its sats are
    # the same physical satellites as the single-freq rows, so mixing them double-counts and
    # cross-contaminates the two clock frames.
    def _combined(sel):
        if not sel:
            return None
        gi = {g: i for i, g in enumerate(sel)}
        rows = [(los, res, gi[g]) for g in sel for los, res in by_group[g]]
        s = _solve(rows, len(sel), gi)
        if not (s and _ok(Ninv=s[4], sig_diag=s[3])):
            return None
        dx, clks, sig0, sd, Ninv, nu, nr = s
        c = _pack(dx, 0.0, sig0, sd, Ninv, nu, nr)
        c["clock_m"] = None
        c["clocks_m"] = {g: float(clks[gi[g]]) for g in sel}
        c["n_groups"] = len(sel)
        return c

    all_groups = sorted(by_group)
    out["combined"] = _combined([g for g in all_groups if not g.endswith("-IF")])
    out["combined_if"] = _combined([g for g in all_groups if g.endswith("-IF")])
    return out


if __name__ == "__main__":
    # Synthetic self-test: a known site, sats at assorted az/el, and a per-sat ELEVATION-mapped
    # ionosphere injected on L1 and L5 with the physical 1/f^2 ratio. Elevation-dependent iono is
    # NOT absorbable by a constant per-group clock, so the single-frequency solve is biased (mostly
    # vertical); the iono-free combination must remove it and recover the a-priori position.
    import random
    random.seed(1)
    lat0, lon0, alt0 = 43.9687, -79.2521, 260.0
    fq = {"L1": 1575.42e6, "L5": 1176.45e6}
    dx_true = np.array([4.0, -3.0, 6.0])           # ENU offset (m) we must recover
    clk = {"L1": 12.0, "L5": -5.0}                 # independent per-band dongle clocks (m)
    m_l1, m_if = [], []
    for _ in range(14):
        az, el = random.uniform(0, 360), random.uniform(15, 85)
        los = _los_enu(az, el)
        iono_l1 = 3.0 / math.sin(math.radians(el))  # L1 slant iono delay (m), bigger at low el
        base = -float(los @ dx_true)
        r1 = base + clk["L1"] + iono_l1 + random.gauss(0, 0.3)
        r5 = base + clk["L5"] + iono_l1 * (fq["L1"] / fq["L5"]) ** 2 + random.gauss(0, 0.3)
        m_l1.append({"group": "G-L1", "az": az, "el": el, "resid_m": r1})
        rif = (fq["L1"] ** 2 * r1 - fq["L5"] ** 2 * r5) / (fq["L1"] ** 2 - fq["L5"] ** 2)
        m_if.append({"group": "G-IF", "az": az, "el": el, "resid_m": rif})
    r_l1 = solve(m_l1, lat0, lon0, alt0)["groups"]["G-L1"]
    r_if = solve(m_if, lat0, lon0, alt0)["groups"]["G-IF"]
    e_l1 = math.sqrt(sum((r_l1[k] - dx_true[i]) ** 2 for i, k in enumerate(("d_e", "d_n", "d_u"))))
    e_if = math.sqrt(sum((r_if[k] - dx_true[i]) ** 2 for i, k in enumerate(("d_e", "d_n", "d_u"))))
    print("single-freq L1 position error: %.2f m  (sigma_u %.2f, iono-biased)"
          % (e_l1, r_l1["sigma_u"]))
    print("iono-free    IF position error: %.2f m  (sigma_u %.2f)" % (e_if, r_if["sigma_u"]))
    assert e_if < e_l1, "iono-free did not beat single-frequency (%.2f vs %.2f)" % (e_if, e_l1)
    assert e_if < 2.0, "iono-free residual too large: %.2f m" % e_if
    print("PASS: dual-frequency iono-free removes the elevation-dependent iono bias")
